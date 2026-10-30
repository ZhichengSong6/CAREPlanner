#!/usr/bin/env python3
import copy
import ast
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/care_visibility_cdf/scripts'))
from stopping_trajectory_prefix import build_trajectory_prefix, trajectory_sha256, validate_trajectory_report, apply_absolute_observation_deadline
from phase_e_trajectory_observer_node import TrajectoryPrefixObserver
import test_phase_e_observation_prefix as prefix_tests
Fixture = prefix_tests.Fixture

JOINTS = ['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']


def duration(ns):
    return NS(to_nsec=lambda: ns)


def trajectory(knots=None):
    knots = knots or [(0., 0., 1.), (1., 0., -1.)]
    return NS(header=NS(stamp=duration(8_000_000_000), seq=1, frame_id='base_link'),
        joint_names=list(JOINTS), points=[NS(time_from_start=duration(round(t*1e9)),
            positions=[q]+[0.]*6, velocities=[v]+[0.]*6, accelerations=[], effort=[]) for t,q,v in knots])


def construct(t=None, **options):
    args = dict(trajectory=t or trajectory(), joint_names=JOINTS, start_window_ns=[10_000_000_000,10_200_000_000],
        horizon_s=1., target_abs_velocity=[.1]*7, applied_velocity=[0.]*7,
        stopping_budget=lambda applied, target: ([.02]*7,.3))
    args.update(options)
    return build_trajectory_prefix(**args)


class CurveTests(unittest.TestCase):
    def test_interior_overshoot_is_included_when_both_endpoints_are_zero(self):
        segment=construct()['segments'][0]
        self.assertLessEqual(segment['q_low'][0],0.)
        self.assertGreaterEqual(segment['q_high'][0],.25)
        self.assertLess(segment['q_high'][0],.250000000001)
    def test_partial_piece_uses_its_interior_extremum(self):
        s=construct(phase_s=.3,horizon_s=.3)['segments'][0]
        self.assertLessEqual(s['q_low'][0],.21);self.assertGreaterEqual(s['q_high'][0],.25)
        self.assertAlmostEqual(s['phase_begin_s'],.3);self.assertAlmostEqual(s['phase_end_s'],.6)
    def test_negative_curvature_and_non_midpoint_extrema(self):
        s=construct(trajectory([(0.,0.,-1.),(1.,.2,1.4)]))['segments'][0]
        self.assertLess(s['q_low'][0],-.2);self.assertGreaterEqual(s['q_high'][0],.2)
    def test_one_inconsistent_joint_makes_all_joints_linear_as_in_cpp(self):
        t=trajectory();t.points[1].positions[6]=.01
        s=construct(t)['segments'][0]
        self.assertLess(s['q_high'][0],1e-12)
    def test_missing_velocities_use_linear_interpolation(self):
        t=trajectory([(0.,0.,0.),(1.,.2,0.)]);t.points[0].velocities=[]
        s=construct(t)['segments'][0]
        self.assertLessEqual(s['q_low'][0],0.);self.assertGreaterEqual(s['q_high'][0],.2)
    def test_every_intermediate_piece_is_present(self):
        c=construct(trajectory([(0.,0.,0.),(.2,.2,0.),(.4,-.1,0.),(.6,0.,0.)]),horizon_s=.6)
        self.assertEqual([s['source_piece'] for s in c['segments']],[0,1,2])
        self.assertGreaterEqual(c['segments'][1]['q_high'][0],.2)
        self.assertLessEqual(c['segments'][1]['q_low'][0],-.1)
    def test_exact_knot_boundary_and_horizon_do_not_include_a_later_piece(self):
        c=construct(trajectory([(0.,0.,0.),(.5,.1,0.),(1.,2.,0.)]),horizon_s=.5)
        self.assertEqual(len(c['segments']),1)
        self.assertLess(c['segments'][0]['q_high'][0],.10000000001)
    def test_terminal_hold_extends_required_time_after_the_last_knot(self):
        c=construct(horizon_s=2.)
        self.assertEqual(c['segments'][-1]['interpolation'],'terminal_hold')
        self.assertAlmostEqual(c['segments'][-1]['required_until_s'],12.5)
    def test_single_point_and_already_finished_reference_are_held(self):
        for t, phase in [(trajectory([(0.,.1,0.)]),0.),(trajectory(),2.)]:
            c=construct(t,phase_s=phase)
            self.assertEqual(c['segments'][0]['interpolation'],'terminal_hold')
    def test_permuted_wire_joints_map_to_controller_order(self):
        t=trajectory();t.joint_names.reverse()
        for p in t.points:p.positions.reverse();p.velocities.reverse()
        c=construct(t)
        self.assertGreaterEqual(c['segments'][0]['q_high'][0],.25)
        self.assertLess(c['segments'][0]['q_high'][6],1e-12)
    def test_start_window_does_not_use_business_header_stamp(self):
        c=construct();s=c['segments'][0]
        self.assertEqual(c['trajectory_stamp_ns'],8_000_000_000)
        self.assertEqual(s['first_use_s'],10.);self.assertAlmostEqual(s['required_until_s'],11.5)
    def test_wider_old_applied_output_is_preserved_in_stop_budget(self):
        seen=[]
        construct(applied_velocity=[-.5]+[0.]*6,stopping_budget=lambda u,cap:(seen.append((u,cap)) or ([.1]*7,.5)))
        self.assertEqual(seen[0][0][0],.5);self.assertEqual(seen[0][1][0],.1)
    def test_capacity_never_silently_truncates_a_path(self):
        with self.assertRaisesRegex(ValueError,'segment_capacity'):
            construct(trajectory([(i*.01,0.,0.) for i in range(35)]),horizon_s=.34)
    def test_nonincreasing_time_missing_start_and_nonfinite_states_rejected(self):
        for mutate in [lambda t:setattr(t.points[1],'time_from_start',duration(0)),
                       lambda t:setattr(t.points[0],'time_from_start',duration(1)),
                       lambda t:t.points[0].positions.__setitem__(0,float('nan'))]:
            t=trajectory();mutate(t)
            with self.assertRaises(ValueError):construct(t)
    def test_content_hash_changes_for_same_stamp_position_velocity_and_timing_changes(self):
        t=trajectory();digest=trajectory_sha256(t)
        for mutate in [lambda p:p.positions.__setitem__(0,.01),lambda p:p.velocities.__setitem__(0,.2),
                       lambda p:setattr(p,'time_from_start',duration(1_100_000_000))]:
            other=copy.deepcopy(t);mutate(other.points[1])
            self.assertNotEqual(trajectory_sha256(other),digest)
    def test_ros_transport_sequence_does_not_change_trajectory_content_identity(self):
        t=trajectory();digest=trajectory_sha256(t);t.header.seq=999
        self.assertEqual(trajectory_sha256(t),digest)
    def test_absolute_observation_deadline_does_not_use_legacy_header_epoch_or_margin(self):
        result=apply_absolute_observation_deadline(dict(deadline_absolute_ros_s=1000.),
            dict(stopping_observation_deadline_abs_s=15.,stopping_observation_epoch_s=10.),'stopping_prefix_lookahead')
        self.assertEqual(result['deadline_absolute_ros_s'],15.)
        self.assertEqual(result['deadline_from_start_s'],5.)
        self.assertEqual(result['deadline_epoch_source'],'stopping_observation_start_window')
    def test_legacy_waypoint_deadline_is_unchanged(self):
        result=dict(deadline_absolute_ros_s=1000.)
        self.assertIs(apply_absolute_observation_deadline(result,{},'predicted'),result)


def state(now=10_000_000_000):
    return dict(t_ns=now,clock_fault=False,controller_session='s',applied=[0.]*7,
        max_braking_step_s=.02,
        velocity_limits=[2.]*7,acceleration_limits=[3.]*7,valid_until_ns=now+100_000_000,
        target_range_enforced=True,target_abs_velocity=[.1]*7)


class ObserverTests(unittest.TestCase):
    def fixture(self):
        self.f=Fixture();self.sent=[];self.rows=[]
        config=dict(horizon_s=1.,admission_wait_s=.5,input_max_age_s=.5,wall_budget_s=.5,
            lease_s=.1,update_gap_s=.001,max_braking_step_s=.02,target_abs_velocity=[.1]*7,
            observation_timing=dict(motion_s=.1,acquisition_s=.01,verification_s=.01,stop_reserve_s=.02))
        o=TrajectoryPrefixObserver(geometry=self.f,query=self.f.query,now_ns=lambda:self.f.now,
            joint_names=JOINTS,config=config,publish=self.sent.append,audit=self.rows.append)
        o.offer_state(state());o.offer(trajectory());return o
    def test_actual_candidate_produces_bound_report_with_whole_interval(self):
        o=self.fixture();o.tick();self.assertEqual(len(self.sent),1)
        validate_trajectory_report(self.sent[0],trajectory(),10.)
        self.assertGreaterEqual(self.f.boxes[0][0][0]+self.f.boxes[0][1][0],.25)
        self.assertFalse(self.sent[0]['runtime_admission'])
    def test_republishing_identical_input_does_not_slide_start_window(self):
        o=self.fixture();self.f.now+=600_000_000;t=trajectory();t.header.seq=99;o.offer(t);o.offer_state(state(self.f.now));o.tick()
        self.assertFalse(self.sent);self.assertIn('stale',self.rows[-1]['reason'])
    def test_replacement_during_query_discards_old_result_even_with_same_stamp(self):
        o=self.fixture();original=self.f.query
        def query(*args):
            t=trajectory();t.points[1].positions[0]=.01;o.offer(t);return original(*args)
        o.query=query;o.tick();self.assertFalse(self.sent)
        self.assertIn('replaced',self.rows[-1]['reason'])
    def test_range_expansion_during_query_is_rejected(self):
        o=self.fixture();original=self.f.query
        def query(*args):
            s=state();s['target_abs_velocity']=[.3]*7;o.offer_state(s);return original(*args)
        o.query=query;o.tick();self.assertFalse(self.sent)
        self.assertIn('context_changed',self.rows[-1]['reason'])
    def test_wider_unexpired_pending_grant_is_included(self):
        o=self.fixture();o.offer_lease(dict(controller_session='s',execution_stamp_ns=7,lease_sequence=2,
            valid_until_ns=self.f.now+10_000_000,target_range_enforced=True,target_abs_velocity=[.4]*7))
        o.tick();self.assertEqual(self.sent[0]['trajectory_construction']['proposed_target_abs_velocity'],[.4]*7)
    def test_unbounded_pending_grant_uses_full_controller_range(self):
        o=self.fixture();o.offer_lease(dict(controller_session='s',execution_stamp_ns=7,lease_sequence=2,
            valid_until_ns=self.f.now+10_000_000,target_range_enforced=False,target_abs_velocity=[0.]*7))
        o.tick();self.assertEqual(self.sent[0]['trajectory_construction']['proposed_target_abs_velocity'],[2.]*7)
    def test_stale_receipt_cannot_start_a_query(self):
        o=self.fixture();self.f.now+=51_000_000;o.tick();self.assertFalse(self.sent);self.assertFalse(self.f.calls)
    def test_longer_pending_deadline_cannot_be_erased_by_a_shorter_proposal(self):
        o=self.fixture();o.config['lease_s']=.02
        o.offer_lease(dict(controller_session='s',execution_stamp_ns=7,lease_sequence=2,
            valid_until_ns=self.f.now+200_000_000,target_range_enforced=True,target_abs_velocity=[.1]*7))
        o.tick();self.assertAlmostEqual(self.sent[0]['trajectory_construction']['conditional_command_model']['hold_s'],.2)
    def test_braking_step_comes_from_actual_controller_receipt(self):
        o=self.fixture();s=state();s['max_braking_step_s']=.0005;o.offer_state(s);o.tick()
        self.assertEqual(self.sent[0]['trajectory_construction']['conditional_command_model']['max_braking_step_s'],.0005)
    def test_applied_output_growth_outside_the_constructed_range_rejects_late_work(self):
        o=self.fixture();original=self.f.query
        def query(*args):
            s=state();s['applied']=[.5]*7;o.offer_state(s);return original(*args)
        o.query=query;o.tick();self.assertFalse(self.sent);self.assertIn('context_changed',self.rows[-1]['reason'])
    def test_late_work_is_not_published(self):
        o=self.fixture();original=self.f.query
        def query(*args):
            result=original(*args);self.f.now+=600_000_000;return result
        o.query=query;o.tick();self.assertFalse(self.sent)
    def test_single_worker_does_not_queue_work(self):
        o=self.fixture();o.worker.acquire();o.tick();o.worker.release()
        self.assertFalse(self.sent);self.assertFalse(self.f.calls)
    def test_same_stamp_content_mismatch_is_rejected_at_scheduler_boundary(self):
        o=self.fixture();o.tick();t=trajectory();t.points[1].positions[0]=.01
        with self.assertRaisesRegex(ValueError,'content_mismatch'):validate_trajectory_report(self.sent[0],t,10.)
    def test_lease_identity_conflict_poison_persists_until_both_deadlines(self):
        o=self.fixture();g=dict(controller_session='s',execution_stamp_ns=1,lease_sequence=1,
            valid_until_ns=self.f.now+100_000_000,target_range_enforced=True,target_abs_velocity=[.1]*7)
        o.offer_lease(g);g=copy.deepcopy(g);g['target_abs_velocity']=[.2]*7
        with self.assertRaisesRegex(ValueError,'conflicting'):o.offer_lease(g)
        o.tick();self.assertFalse(self.sent)


class SchedulerLifecycleTests(unittest.TestCase):
    def test_replacement_retires_owned_requests_without_waiting_for_another_report(self):
        helper=prefix_tests.ProductionAdapterTests();n=helper.node();helper.publish(n,Fixture().scan())
        self.assertTrue(n._obligations)
        n.trajectory.points[0].positions[0]=.2;n._process_stopping_prefix_demands()
        self.assertFalse(n._obligations);self.assertFalse(n._stopping_prefix_queue.pending)
    def test_stale_preferred_trajectory_retires_owned_requests(self):
        helper=prefix_tests.ProductionAdapterTests();n=helper.node();helper.publish(n,Fixture().scan())
        n.trajectory=None;n._process_stopping_prefix_demands();self.assertFalse(n._obligations)
    def test_report_without_content_binding_cannot_create_requests(self):
        helper=prefix_tests.ProductionAdapterTests();n=helper.node();n._stopping_prefix_pending=Fixture().scan()
        n._process_stopping_prefix_demands();self.assertFalse(n._obligations)
    def test_actual_online_dispatch_reaches_prefix_processing(self):
        root=Path(__file__).resolve().parents[1]/'src/care_visibility_cdf/scripts'
        base=ast.parse((root/'vbc_multi_deadline_obligation_impl.py').read_text())
        online=ast.parse((root/'vbc_deadline_waypoint_online_node.py').read_text())
        def method(tree, cls):
            c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
            return next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_maybe_generate')
        ns={};exec(compile(ast.Module(body=[method(base,'AccumulatedMultiDeadlineWaypointNode')],type_ignores=[]),'base_dispatch','exec'),ns)
        base_type=type('ActualBaseDispatch',(),{'_maybe_generate':ns['_maybe_generate']})
        ns={'AccumulatedMultiDeadlineWaypointNode':base_type}
        exec(compile(ast.Module(body=[method(online,'OnlineAccumulatedMultiDeadlineWaypointNode')],type_ignores=[]),'online_dispatch','exec'),ns)
        calls=[];node=NS(_c46_ready=True,_process_stopping_prefix_demands=lambda:calls.append('prefix'),
            _process_new_active_set=lambda:calls.append('active'),_publish_schedule=lambda:calls.append('publish'))
        ns['_maybe_generate'](node);self.assertEqual(calls,['prefix','active','publish'])
    def test_actual_acquisition_dispatch_reaches_prefix_before_visibility_completion(self):
        root=Path(__file__).resolve().parents[1]/'src/care_visibility_cdf/scripts'
        tree=ast.parse((root/'vbc_visibility_acquisition_impl.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VisibilityAcquisitionWaypointNode')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_maybe_generate')
        ns={};exec(compile(ast.Module(body=[method],type_ignores=[]),'acquisition_dispatch','exec'),ns)
        calls=[];node=NS(_c47_ready=True,_process_stopping_prefix_demands=lambda:calls.append('prefix'),
            _process_new_active_set=lambda:calls.append('active'),_update_actual_visibility_completion=lambda:calls.append('completion'),
            _publish_schedule=lambda:calls.append('publish'))
        ns['_maybe_generate'](node);self.assertEqual(calls,['prefix','active','completion','publish'])


if __name__ == '__main__':
    unittest.main()
