#!/usr/bin/env python3
import copy
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace as NS
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/care_visibility_cdf/scripts'))
from bounded_visibility_recovery import VisibilityRecoveryBudget, region_key, certified_distinct_q
from observation_event_trace import observation_region_token, observation_token
from test_phase_e_vbc_evidence import methods


def result():
    return dict(q_vis=[.1]*7,q_zero=[0.]*7,final_f_min=.03,shared_solution_mode='fixture',
        per_sensor_hybrid_used=True,per_sensor_selected_sensor_id=1,per_sensor_selected_sensor_frame='S1',
        per_sensor_selected_rank=2,per_sensor_hybrid=dict(accepted=True,selected_q_vis=[.1]*7))


def scalar_result():
    return dict(q_vis=[.1]*7, q_zero=[0.]*7, final_f_min=.03,
        shared_solution_mode='shared_projection_root_ascent')


def node():
    n=methods('src/care_visibility_cdf/scripts/vbc_blocker_aware_acquisition_impl.py',
        {'_process_final_recovery','_final_recovery_feedback_cb',
         '_final_recovery_alternative_seed'},dict(np=np,time=time,
        observation_region_token=observation_region_token,observation_token=observation_token,
        region_key=region_key,
        certified_distinct_q=certified_distinct_q,rospy=NS(logwarn=lambda *a:None)))
    for key in ['_schedule_publish_lock','_obligation_lock','_final_recovery_lock','_frontier_lock','_lock','_progressive_shared_lock']:
        setattr(n,key,threading.RLock())
    n._final_recovery_enabled=True;n._final_recovery_pending=None;n._final_recovery_active_request=None
    n._final_recovery_latest_seq=-1;n._final_recovery_budget=VisibilityRecoveryBudget()
    n._final_recovery_latest_feedback_seq=-1
    n._final_recovery_feedback_count=0
    n._final_recovery_hold_feedback_count=0
    n._final_recovery_nonhold_feedback_count=0
    n._final_recovery_preserved_pending_count=0
    n._final_recovery_process_count=0
    n._final_recovery_process_success_count=0
    n._final_recovery_live_obligation_fallback_count=0
    n._final_recovery_alternative_seed_attempt_count=0
    n._final_recovery_alternative_seed_success_count=0
    n._final_recovery_alternative_seed_rad=.05
    n._final_recovery_target_history={}
    n._final_recovery_target_history_capacity=64
    n._frontier_feedback_token='none';n._frontier_feedback_seq=-1
    n._frontier_feedback_failures=0;n._frontier_cache_key='cached'
    n._latest_measured_q=np.zeros(7)
    n._obligations=[dict(id=1,points=np.asarray([[.2,-.15,.55]]),q_vis=np.zeros(7),discovered_sweep_time_s=.1)]
    n._trace_published_target=dict(observation_token='care_obs_v1_test',
        region_token=observation_region_token(n._obligations[0]),q_vis=np.zeros(7))
    n._preferred_trajectory_locked=lambda:(object(),1.,'fixture')
    n.calls=[];n.published=[];n.traces=[]
    def generate(*args):n.calls.append(n._seed_override.copy());return result()
    n._generate_active_set_waypoint=generate;n._publish_schedule=lambda:n.published.append(True)
    n._trace_observation=lambda *a,**kw:n.traces.append(kw)
    return n


def hold(n,seq=1,token='care_obs_v1_test'):
    n._final_recovery_feedback_cb(NS(data='plan_seq=%d repair=1 probe=0 final_vbc_hold=1 observation_token=%s'%(seq,token)))


class RecoveryTests(unittest.TestCase):
    def test_dependency_protocol_does_not_reselect_parent_on_stall(self):
        n=node();n._final_recovery_enabled=False;n._safe_stall_recovery_enabled=True
        n._final_recovery_feedback_cb(NS(data='plan_seq=7 repair=1 probe=0 observation_dependency_policy=1 safe_stall_reselect=1 observation_token=care_obs_v1_test'))
        self.assertIsNone(n._final_recovery_pending)
        n._process_final_recovery()
        self.assertEqual(n.published,[])

    def test_safe_stall_reselect_independent_of_legacy_final_vbc_switch(self):
        n=node();n._final_recovery_enabled=False;n._safe_stall_recovery_enabled=True
        n._final_recovery_feedback_cb(NS(data='plan_seq=7 repair=1 probe=0 final_vbc_hold=0 safe_stall_reselect=1 observation_token=care_obs_v1_test'))
        self.assertEqual(n._final_recovery_pending,(7,'care_obs_v1_test'))
        n._process_final_recovery()
        self.assertEqual(n.published,[True])

    def test_safe_stall_does_not_enable_legacy_rejection_trigger(self):
        n=node();n._final_recovery_enabled=False;n._safe_stall_recovery_enabled=True
        hold(n,7)
        self.assertIsNone(n._final_recovery_pending)
        n._process_final_recovery()
        self.assertEqual(n.published,[])

    def test_safe_stall_probe_stale_and_failed_generation_do_not_retarget(self):
        n=node();n._final_recovery_enabled=False;n._safe_stall_recovery_enabled=True
        n._final_recovery_feedback_cb(NS(data='plan_seq=7 repair=1 probe=1 safe_stall_reselect=1 observation_token=care_obs_v1_test'))
        self.assertIsNone(n._final_recovery_pending)
        n._final_recovery_feedback_cb(NS(data='plan_seq=8 repair=1 probe=0 safe_stall_reselect=1 observation_token=care_obs_v1_unknown'))
        n._process_final_recovery();self.assertEqual(n.published,[])
        n._generate_active_set_waypoint=lambda *a: None
        n._final_recovery_feedback_cb(NS(data='plan_seq=9 repair=1 probe=0 safe_stall_reselect=1 observation_token=care_obs_v1_test'))
        n._process_final_recovery();self.assertEqual(n.published,[])
        self.assertTrue(np.all(n._obligations[0]['q_vis']==0))

    def test_same_point_budget_survives_fp_transport_and_target_churn(self):
        b=VisibilityRecoveryBudget();p=[[.2,-.15,.55]]
        self.assertTrue(b.reserve(p,[0.]*7));self.assertFalse(b.reserve(np.float32(p),[.001]*7))
        self.assertTrue(b.reserve([[.3,0.,.5]],[.001]*7));self.assertFalse(b.reserve(p,[0.]*7))
        self.assertTrue(b.reserve(p,[.011]*7))

    def test_capacity_nonfinite_and_empty(self):
        b=VisibilityRecoveryBudget();self.assertFalse(b.reserve([], [0.]*7))
        self.assertFalse(b.reserve([[1.,2.,3.]],[float('nan')]*7))
        for i in range(64):self.assertTrue(b.reserve([[float(i),0.,0.]],[0.]*7))
        self.assertFalse(b.reserve([[65.,0.,0.]],[0.]*7))

    def test_certified_only_and_not_optimizer_drift(self):
        r=result();self.assertTrue(certified_distinct_q(r,[0.]*7))
        for change in [dict(accepted=False),dict(selected_q_vis=[.2]*7)]:
            bad=copy.deepcopy(r);bad['per_sensor_hybrid'].update(change)
            self.assertFalse(certified_distinct_q(bad,[0.]*7))
        self.assertFalse(certified_distinct_q(r,[.099]*7))

    def test_scalar_proposer_is_allowed_when_per_sensor_is_disabled(self):
        self.assertTrue(certified_distinct_q(scalar_result(), [0.]*7))
        bad=scalar_result();bad['final_f_min']=float('nan')
        self.assertFalse(certified_distinct_q(bad, [0.]*7))

    def test_scalar_recovery_updates_live_obligation_without_sensor_fields(self):
        n=node()
        n._generate_active_set_waypoint=lambda *args: scalar_result()
        hold(n, 8)
        n._process_final_recovery()
        self.assertEqual(n.published, [True])
        self.assertEqual(n._obligations[0]['q_vis'].tolist(), [.1]*7)

    def test_scalar_recovery_tries_one_alternative_seed_after_same_target(self):
        n=node()
        calls=[]
        def generate(*args):
            calls.append(n._seed_override.copy())
            if len(calls) == 1:
                return dict(scalar_result(), q_vis=[0.]*7)
            return scalar_result()
        n._generate_active_set_waypoint=generate
        hold(n)
        n._process_final_recovery()
        self.assertEqual(len(calls), 2)
        self.assertGreater(np.max(np.abs(calls[1] - calls[0])), .01)
        self.assertEqual(n._final_recovery_alternative_seed_attempt_count, 1)
        self.assertEqual(n._final_recovery_alternative_seed_success_count, 1)
        self.assertEqual(n._obligations[0]['q_vis'].tolist(), [.1]*7)

    def test_actual_callback_refresh_once_from_measured(self):
        n=node();hold(n);n._process_final_recovery()
        self.assertEqual(n.published,[True]);self.assertEqual(n.calls[0].tolist(),[0.]*7)
        self.assertEqual(n._obligations[0]['q_vis'].tolist(),[.1]*7)
        self.assertEqual(n._obligations[0]['per_sensor_selected_sensor_id'],1)
        self.assertEqual(len(n._obligations),1)  # no pending point removed
        hold(n,2);n._process_final_recovery();self.assertEqual(len(n.calls),1)

    def test_disabled_stale_duplicate_and_nonhold_feedback_preserves_pending(self):
        n=node();hold(n,2);hold(n,1,'care_obs_v1_old');self.assertEqual(n._final_recovery_pending[0],2)
        n._final_recovery_enabled=False;n._process_final_recovery();self.assertFalse(n.calls)
        n._final_recovery_enabled=True
        n._final_recovery_feedback_cb(NS(data='plan_seq=3 repair=0 probe=0 final_vbc_hold=0'))
        self.assertEqual(n._final_recovery_pending[0],2)
        n._process_final_recovery();self.assertEqual(len(n.calls),1)
        hold(n,4,'care_obs_v1_old');n._process_final_recovery();self.assertEqual(len(n.calls),1)

    def test_same_plan_nonhold_feedback_does_not_cancel_hold(self):
        n=node();hold(n,2)
        n._final_recovery_feedback_cb(NS(
            data='plan_seq=2 repair=1 probe=0 final_vbc_hold=0 observation_token=care_obs_v1_test'))
        self.assertEqual(n._final_recovery_pending, (2, 'care_obs_v1_test'))
        n._process_final_recovery()
        self.assertEqual(len(n.calls), 1)

    def test_frontier_rejection_count_is_target_scoped(self):
        n = node()
        n._final_recovery_feedback_cb(NS(data=(
            'plan_seq=4 repair=1 probe=0 final_vbc_hold=0 '
            'final_vbc_failures=3 observation_token=care_obs_v1_test')))
        self.assertEqual(n._frontier_feedback_failures, 3)
        self.assertIsNone(n._frontier_cache_key)
        # A new q_vis token starts a fresh escalation ladder.
        n._final_recovery_feedback_cb(NS(data=(
            'plan_seq=5 repair=1 probe=0 final_vbc_hold=0 '
            'final_vbc_failures=1 observation_token=care_obs_v1_new')))
        self.assertEqual(n._frontier_feedback_token, 'care_obs_v1_new')
        self.assertEqual(n._frontier_feedback_failures, 1)

    def test_live_obligation_recovers_after_target_is_temporarily_preempted(self):
        n=node()
        # The published target may be a shared-solve copy whose q_vis differs
        # from the stored individual obligation. Exercise the publication
        # history join rather than the direct live-token fast path.
        held_ob=dict(n._obligations[0], q_vis=np.ones(7))
        held_token=observation_token(held_ob)
        n._final_recovery_target_history[held_token] = dict(
            observation_token=held_token,
            region_token=observation_region_token(n._obligations[0]),
            q_vis=np.ones(7))
        hold(n, 7, held_token)
        # A newly active blocker is now the published target. The held
        # obligation remains live, so its bounded refresh must still run.
        n._trace_published_target=dict(
            observation_token='care_obs_v1_blocker',
            region_token='care_region_v1_blocker', q_vis=np.ones(7))
        n._process_final_recovery()
        self.assertEqual(len(n.calls), 1)
        self.assertEqual(n._final_recovery_live_obligation_fallback_count, 1)
        self.assertEqual(n._obligations[0]['q_vis'].tolist(), [.1]*7)

    def test_no_seed_clear_outside_generation_lock(self):
        n=node()
        def publish():
            # Another serialized model user may own the shared override after
            # our generation lock is released; its seed must not be clobbered.
            self.assertIsNone(n._seed_override)
            n._seed_override=np.ones(7)
        n._publish_schedule=publish;hold(n);n._process_final_recovery()
        self.assertEqual(n._seed_override.tolist(),[1.]*7)

    def test_failed_missing_fields_or_stale_generation_never_mutates(self):
        for mode in ['uncertified','missing_field','cancel_during_generation','motion_during_generation']:
            n=node();original=n._obligations[0]['q_vis'].copy()
            def generate(*a):
                r=result()
                if mode=='uncertified':r['per_sensor_hybrid']['accepted']=False
                if mode=='missing_field':del r['per_sensor_selected_sensor_id']
                if mode=='cancel_during_generation':
                    n._final_recovery_feedback_cb(NS(
                        data='plan_seq=2 repair=1 probe=0 final_vbc_hold=1 observation_token=care_obs_v1_test'))
                if mode=='motion_during_generation':n._latest_measured_q=np.ones(7)*.02
                return r
            n._generate_active_set_waypoint=generate;hold(n);n._process_final_recovery()
            self.assertFalse(n.published,mode)
            self.assertEqual(n._obligations[0]['q_vis'].tolist(),original.tolist(),mode)
            self.assertIsNone(n._seed_override)


if __name__=='__main__':unittest.main()
