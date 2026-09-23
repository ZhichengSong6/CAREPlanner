#!/usr/bin/env python3
"""Offline protocol/publication regression. Gate/seen inputs are synthetic."""
import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/care_visibility_cdf/scripts'))
from candidate_replacement_runtime import (
    CandidateReplacement,
    SENSOR_MOUNT_GROUPS,
    high_witness_sensor_preference,
    replacement_mount_group_pool,
    sensor_mount_group_id,
)
from per_sensor_visibility_runtime import preferred_sensor_selection
from observation_identity import observation_token
from bounded_visibility_recovery import region_key
from test_phase_e_dependency_publish_chain import PublishChainTests


class RuntimeReplacementTests(unittest.TestCase):
    def test_high_ee_sensor_priority_precedes_other_mount_groups(self):
        ranked = list(enumerate([1, 4, 7, 6, 0, 3, 5, 2], 1))
        pool, meta = replacement_mount_group_pool(ranked, {7}, True)
        self.assertEqual([sid for _, sid in pool], [6])
        self.assertEqual(meta['phase'], 'high_witness_ee_priority')
        pool, meta = replacement_mount_group_pool(ranked, {7, 6}, True)
        self.assertTrue(all(sid not in (6, 7) for _, sid in pool))
        self.assertEqual(meta['phase'], 'untried_mount_group')

    def test_native_attempt_budget_matches_mount_group_count(self):
        header = (Path(__file__).resolve().parents[1]/'src/egocentric_arm_planner/'
                  'include/egocentric_arm_planner/safe_frontier_recovery.hpp').read_text()
        self.assertIn(f'static constexpr int max_attempts = {len(SENSOR_MOUNT_GROUPS)};',
                      header)

    def test_qp_failure_with_no_raw_candidate_can_replace(self):
        n,r,e,s,events,clock,h=self.fixture()
        r.offer_final_vbc(self.final_trigger(n,clock,reason='repair_qp_failure',
            raw_candidate_stamp_ns=0,audited_trajectory_stamp_ns=0))
        self.assertEqual(len(s),1)
        self.assertIn('reason=repair_qp_failure',s[0])
        self.grant(r);h.bounded(r.process)
        self.assertEqual(events[-1]['event'],'applied_requires_existing_gates')

    def test_qp_failure_rejects_fabricated_raw_or_wrong_token(self):
        for overrides in [dict(raw_candidate_stamp_ns=10),dict(observation_token='care_obs_v1_stale')]:
            n,r,e,s,events,clock,h=self.fixture()
            values=dict(reason='repair_qp_failure',raw_candidate_stamp_ns=0,audited_trajectory_stamp_ns=0)
            values.update(overrides)
            r.offer_final_vbc(self.final_trigger(n,clock,**values))
            self.assertEqual(s,[])

    def test_qp_failure_restores_owner_and_defers_blocker_until_hard_qp(self):
        n,r,e,s,events,clock,h=self.fixture()
        trigger=self.final_trigger(n,clock,reason='repair_qp_failure',
            raw_candidate_stamp_ns=0,audited_trajectory_stamp_ns=0)
        # Reproduce exp87: O1 emitted the authenticated QP trigger, then O3
        # became stack top before Python received the trigger.
        n._repair_stack=[2,1,3];n._publish_schedule()
        r.offer_final_vbc(trigger)
        self.assertEqual(n._repair_stack,[2,3,1])
        self.assertEqual(r.locked_qp_owner_id(),1)
        self.assertEqual(events[-2]['event'],'qp_replacement_owner_restored')
        self.assertIn('reason=repair_qp_failure',s[-1])

        # A newly reported O3 stays queued while reservation/generation runs.
        n._consider_active_layer([1,3],[3],.1)
        self.assertEqual(n._repair_stack,[2,3,1])
        self.assertEqual(n._pending_blocker_id,3)
        self.assertEqual(n._path_association_reason,
                         'qp_replacement_owner_locked')

        self.grant(r);h.bounded(r.process)
        replacement_token=n._trace_published_target['observation_token']
        self.assertEqual(r.locked_qp_owner_id(),1)
        n._consider_active_layer([1,3],[3],.1)
        self.assertEqual(n._repair_stack,[2,3,1])

        # Only the matching replacement token's hard-QP success releases the
        # owner. The queued blocker can then become active normally.
        r.observe_planner_summary(
            'C5_4_LOCAL_SCP event=scp_solved observation_token='+replacement_token)
        self.assertEqual(r.locked_qp_owner_id(),1)
        r.observe_planner_summary(
            'C5_4_LOCAL_SCP event=candidate_published observation_token='+
            replacement_token)
        self.assertIsNone(r.locked_qp_owner_id())
        n._consider_active_layer([1,3],[3],.1)
        self.assertEqual(n._repair_stack,[2,1,3])

    def test_replacement_pin_survives_safety_union_growth(self):
        n,r,e,s,events,clock,h=self.fixture()
        r.offer(e);self.grant(r);h.bounded(r.process)
        ob=next(o for o in n._obligations if o['id']==n._repair_stack[-1])
        ob['points']=np.vstack([ob['points'],[.15,-.1,.85]])
        def forbidden(*args):
            raise AssertionError('shared generation bypassed candidate ledger')
        n._compute_progressive_shared_target=forbidden
        np.testing.assert_equal(n._ordered_obligations()[0]['q_vis'],np.full(7,.4))
        self.assertEqual(len(ob['points']),2)

    def test_high_witness_initial_candidate_is_not_overwritten_by_shared(self):
        n,r,e,s,events,clock,h=self.fixture()
        n.per_sensor_high_witness_priority_enabled=True
        ob=next(o for o in n._obligations if o['id']==n._repair_stack[-1])
        ob['points']=np.array([[.1,-.15,.9]])
        def forbidden(*args):
            raise AssertionError('shared replaced selected wrist candidate')
        n._compute_progressive_shared_target=forbidden
        np.testing.assert_equal(n._ordered_obligations()[0]['q_vis'],ob['q_vis'])

    def fixture(self):
        helper=PublishChainTests()
        n,f,clock=helper.node()
        n._repair_stack=[2,1]  # exp75 path reorder, deliberately NO dependency edge
        n._dependency_pins[2]=dict(region=region_key(n._obligations[1]['points']),q_vis=np.full(7,2.))
        n._latest_measured_q=np.zeros(7)
        n._trace_measured=(np.zeros(7),clock.now)
        n._stack_cycle_block_count=0
        n._compute_progressive_shared_target=lambda *args: None
        n._preferred_trajectory_locked=lambda: (NS(),NS(),'fixture')
        n._publish_schedule()
        sent=[];events=[]
        r=CandidateReplacement(n,lambda:clock.now,sent.append,lambda event,**data: events.append(dict(event=event,**data)))
        n._candidate_replacement=r
        def generate(*args):
            assert not n._obligation_lock.locked()
            return dict(q_zero=np.zeros(7), per_sensor_hybrid=dict(accepted=True,
                selected_sensor_id=7,selected_sensor_frame='EE_sensor2_tof_link',selected_rank=7,
                selected_q_vis=[.4]*7,attempts=[dict(sensor_id=7)]))
        n._generate_active_set_waypoint=generate
        event=dict(mode_epoch=3,query_stamp_ns=1000000000,query_ros_s=1.,
            observation_token=n._trace_published_target['observation_token'],points=[[2.,0.,.5]])
        return n,r,event,sent,events,clock,helper

    def grant(self,r,**overrides):
        d=dict(request_id=r.serial,granted=1,mode_epoch=3,query_stamp_ns=1000000000,
               attempts=2,segments=3,deadline_ns=2000000000)
        if r.pending is not None and r.pending.get('operation') in ('promote_owner', 'restore_ancestor'):
            d['owner_promotion'] = 1
        d.update(overrides)
        r.receive(' '.join(f'{k}={v}' for k,v in d.items()))

    def final_trigger(self, n, clock, **overrides):
        active = int(n._repair_stack[-1])
        point = next(o for o in n._obligations if int(o['id']) == active)['points'][0]
        d = dict(version=1, reason='final_vbc_repeat', mode_epoch=3,
            query_stamp_ns=1000000000, query_ros_s=clock.now,
            observation_token=n._trace_published_target['observation_token'],
            obligation_id=active, raw_candidate_stamp_ns=400,
            audited_trajectory_stamp_ns=401,
            repeated_points=','.join(str(float(v)) for v in point))
        d.update(overrides)
        return ' '.join(f'{k}={v}' for k,v in d.items())

    def test_path_reordered_stack_replaces_and_survives_real_publication(self):
        n,r,e,s,events,clock,h=self.fixture();before=copy.deepcopy(n._obligations)
        r.offer(e);self.assertIn('action=reserve',s[-1]);self.grant(r)
        h.bounded(r.process)
        self.assertEqual(events[-1]['event'],'applied_requires_existing_gates')
        self.assertEqual(n._repair_stack,[2,1]);self.assertEqual(n._dependency_edges,{})
        np.testing.assert_equal(n._ordered_obligations()[0]['q_vis'],np.full(7,.4))
        np.testing.assert_equal(n._dependency_pins[2]['q_vis'],np.full(7,2.))
        for a,b in zip(n._obligations,before):
            self.assertEqual(a['id'],b['id']);np.testing.assert_equal(a['points'],b['points'])
        self.assertNotEqual(n._trace_published_target['observation_token'],e['observation_token'])
        self.assertEqual(len(n._obligations),3)
        n.seen_ids={1};clock.now=1.5
        h.bounded(n._update_actual_visibility_completion)
        self.assertEqual(n._repair_stack,[2])  # synthetic actual-seen path only

    def test_duplicate_request_and_grant_are_single_use(self):
        n,r,e,s,events,clock,h=self.fixture()
        r.offer(e);r.offer(e);self.assertEqual(len(s),1)
        self.grant(r);self.grant(r);r.process();count=len(n.messages)
        self.grant(r);r.process();self.assertEqual(count,len(n.messages))

    def test_final_vbc_repeat_replaces_without_ancestor_cycle(self):
        n,r,e,s,events,clock,h=self.fixture()
        n._repair_stack=[1];n._publish_schedule()
        r.offer_final_vbc(self.final_trigger(n,clock))
        self.assertIn('reason=final_vbc_repeat',s[-1])
        self.assertIn('obligation_id=1',s[-1])
        self.assertIn('trigger_raw_candidate_stamp_ns=400',s[-1])
        self.grant(r);h.bounded(r.process)
        self.assertEqual(events[-1]['event'],'applied_requires_existing_gates')
        self.assertEqual(n._repair_stack,[1])
        np.testing.assert_equal(n._obligations[0]['q_vis'],np.full(7,.4))

    def test_final_vbc_repeat_promotes_other_live_owner_then_resumes_parent(self):
        n,r,e,s,events,clock,h=self.fixture()
        n._repair_stack=[1];n._publish_schedule()
        other=next(o for o in n._obligations if int(o['id'])==2)
        repeated=','.join(str(float(v)) for v in other['points'][0])
        r.offer_final_vbc(self.final_trigger(n,clock,repeated_points=repeated))
        self.assertIn('action=reserve_owner',s[-1])
        self.assertIn('reason=final_vbc_owner_promotion',s[-1])
        self.assertEqual(events[-1]['owner_id'],2)
        self.grant(r);h.bounded(r.process)
        self.assertEqual(events[-1]['event'],
                         'owner_promotion_applied_requires_existing_gates')
        self.assertEqual(n._repair_stack,[1,2])
        self.assertEqual(n._dependency_edges,{2:1})
        self.assertEqual(next(iter(n._dependency_attempts.values())),2)
        np.testing.assert_equal(n._trace_published_target['q_vis'],np.full(7,2.))
        n._obligations=[ob for ob in n._obligations if int(ob['id'])!=2]
        n._prune_or_initialize_stack();n._publish_schedule()
        self.assertEqual(n._repair_stack,[1])
        self.assertEqual(n._dependency_resumes,1)

    def test_same_parent_owner_relation_promotes_once_then_stops(self):
        n,r,e,s,events,clock,h=self.fixture()
        n._repair_stack=[1];n._publish_schedule()
        other=next(o for o in n._obligations if int(o['id'])==2)
        repeated=','.join(str(float(v)) for v in other['points'][0])
        trigger=lambda: self.final_trigger(n,clock,repeated_points=repeated)
        r.offer_final_vbc(trigger());self.grant(r);h.bounded(r.process)
        self.assertEqual(n._repair_stack,[1,2])
        # Emulate a non-completion scheduler return while O2 remains live. The
        # once-only relation ledger must prevent another O1->O2 promotion.
        n._repair_stack=[1];n._dependency_edges.clear();n._publish_schedule()
        sent=len(s)
        r.offer_final_vbc(trigger())
        self.assertEqual(len(s),sent)
        self.assertIsNone(r.pending)
        self.assertEqual(events[-1]['event'],'point_owner_promotion_exhausted')

    def test_live_owner_already_in_stack_cannot_form_cycle(self):
        n,r,e,s,events,clock,h=self.fixture()
        n._repair_stack=[2,1];n._dependency_edges={1:2};n._publish_schedule()
        other=next(o for o in n._obligations if int(o['id'])==2)
        repeated=','.join(str(float(v)) for v in other['points'][0])
        r.offer_final_vbc(self.final_trigger(n,clock,repeated_points=repeated))
        self.assertIn('action=reserve_owner ',s[-1])
        self.assertEqual(r.pending['operation'],'restore_ancestor')
        self.grant(r);h.bounded(r.process)
        self.assertEqual(n._repair_stack,[1,2])
        self.assertEqual(n._dependency_edges,{1:2})
        self.assertEqual(events[-1]['event'],
                         'ancestor_owner_restored_requires_existing_gates')
        self.assertEqual(r.locked_qp_owner_id(),2)
        token=n._trace_published_target['observation_token']
        r.observe_planner_summary(
            'C5_4_LOCAL_SCP event=candidate_published observation_token='+token)
        self.assertIsNone(r.locked_qp_owner_id())
        n._obligations=[ob for ob in n._obligations if int(ob['id'])!=2]
        n._prune_or_initialize_stack();n._publish_schedule()
        self.assertEqual(n._repair_stack,[1])
        self.assertEqual(n._dependency_edges,{})

    def test_ancestor_owner_promotion_is_once_only_without_child_sensor_charge(self):
        n,r,e,s,events,clock,h=self.fixture()
        n._repair_stack=[2,1];n._publish_schedule()
        other=next(o for o in n._obligations if int(o['id'])==2)
        repeated=','.join(str(float(v)) for v in other['points'][0])
        trigger=self.final_trigger(n,clock,repeated_points=repeated)
        r.offer_final_vbc(trigger)
        self.assertEqual(r.pending['operation'],'restore_ancestor')
        self.grant(r);h.bounded(r.process)
        n._repair_stack=[2,1];n._publish_schedule()
        r.offer_final_vbc(self.final_trigger(n,clock,repeated_points=repeated))
        self.assertIsNone(r.pending)
        self.assertEqual(events[-1]['event'],'point_owner_promotion_exhausted')
        self.assertEqual(sum('action=reserve ' in msg for msg in s),0)

    def test_final_vbc_trigger_requires_any_live_point_token_and_obligation(self):
        for change in [dict(repeated_points='99.0,0.0,0.5'),
                       dict(observation_token='care_obs_v1_old'),
                       dict(obligation_id=3), dict(query_ros_s=-1.)]:
            n,r,e,s,events,clock,h=self.fixture()
            r.offer_final_vbc(self.final_trigger(n,clock,**change))
            self.assertEqual(s,[])

    def test_no_ancestor_no_request(self):
        n,r,e,s,*_=self.fixture();e['points']=[[99,0,.5]];r.offer(e);self.assertEqual(s,[])

    def test_stale_token_no_request(self):
        n,r,e,s,*_=self.fixture();e['observation_token']='care_obs_v1_old';r.offer(e);self.assertEqual(s,[])

    def test_denied_reservation_does_not_generate(self):
        n,r,e,s,events,*_=self.fixture();r.offer(e);self.grant(r,granted=0)
        n._generate_active_set_waypoint=lambda *a:self.fail('unreserved inference')
        r.process();self.assertEqual(events[-1]['event'],'reservation_denied')

    def test_changed_ancestor_geometry_rejects_grant(self):
        n,r,e,s,events,*_=self.fixture();r.offer(e);self.grant(r)
        n._obligations[1]['points']=np.array([[2.,0.,.6]])
        r.process();self.assertEqual(events[-1]['event'],'context_changed')

    def test_new_active_child_rejects_grant(self):
        n,r,e,s,events,*_=self.fixture();r.offer(e);self.grant(r);n._repair_stack.append(3)
        r.process();self.assertEqual(events[-1]['event'],'context_changed')

    def test_expired_generation_is_charged_but_not_published(self):
        n,r,e,s,events,clock,h=self.fixture();r.offer(e);self.grant(r,deadline_ns=999999999)
        old=observation_token(n._obligations[0]);r.process()
        self.assertEqual(events[-1]['event'],'generation_expired')
        self.assertEqual(old,observation_token(n._obligations[0]));self.assertIn(7,next(iter(r.families.values())))

    def test_motion_during_generation_rejects(self):
        n,r,e,s,events,*_=self.fixture();r.offer(e);self.grant(r);gen=n._generate_active_set_waypoint
        def moved(*a):
            result=gen(*a);n._trace_measured=(np.full(7,.02),1.);return result
        n._generate_active_set_waypoint=moved;r.process()
        self.assertEqual(events[-1]['event'],'measured_state_changed')

    def test_geometric_rejection_remembers_head(self):
        n,r,e,s,events,*_=self.fixture();r.offer(e);self.grant(r)
        n._generate_active_set_waypoint=lambda *a:dict(per_sensor_hybrid=dict(accepted=False,attempts=[dict(sensor_id=6)]))
        r.process();self.assertEqual(events[-1]['event'],'no_geometric_candidate')
        self.assertIn(6,next(iter(r.families.values())));self.assertIn('new_token=none',s[-1])

    def test_replacement_exports_tried_mount_groups(self):
        n,r,e,s,events,*_=self.fixture()
        n.per_sensor_replacement_min_conservative_g=-.02
        next(ob for ob in n._obligations if int(ob['id'])==1)[
            'per_sensor_selected_sensor_id']=1
        r.offer(e);self.grant(r)
        seen={}
        def generate(*args):
            seen.update(n._candidate_replacement_override)
            return dict(per_sensor_hybrid=dict(accepted=False,
                attempts=[dict(sensor_id=2)]))
        n._generate_active_set_waypoint=generate;r.process()
        self.assertEqual(seen['excluded'],[1])
        self.assertEqual(seen['tried_mount_groups'],[0])
        self.assertEqual(seen['minimum_conservative_g'],-.02)

    def test_high_witness_replacement_exports_s7_s6_priority(self):
        n,r,e,s,events,*_=self.fixture()
        n.per_sensor_high_witness_priority_enabled=True
        n.per_sensor_high_witness_z_min=.4
        seen={}
        def generate(*args):
            seen.update(n._candidate_replacement_override)
            return dict(per_sensor_hybrid=dict(accepted=False,
                attempts=[dict(sensor_id=7)]))
        n._generate_active_set_waypoint=generate
        r.offer(e);self.grant(r);r.process()
        self.assertEqual(seen['preferred_sensor_ids'],[7,6])
        self.assertEqual(seen['preference_reason'],
                         'repeated_exact_vbc_high_witness')

    def test_high_witness_threshold_requires_every_point_high(self):
        points=[[.15,0,.90],[-.15,-.05,.95]]
        self.assertEqual(high_witness_sensor_preference(points,True,.85),[7,6])
        self.assertEqual(high_witness_sensor_preference(points,False,.85),[])
        self.assertEqual(high_witness_sensor_preference(
            points+[[0,0,.70]],True,.85),[])

    def test_explicit_sensor_priority_precedes_scalar_rank(self):
        ranked=[(1,5),(2,1),(3,4),(4,3),(5,2),(6,0),(7,7),(8,6)]
        selection,trace=preferred_sensor_selection(ranked,4,[7,6])
        self.assertEqual([sid for _,sid in selection],[7,6,5,1])
        self.assertEqual(trace['requested_sensor_ids'],[7,6])

    def test_mount_groups_are_exhausted_before_opposite_sensors(self):
        ranked=[(rank,sid) for rank,sid in enumerate(range(8),1)]
        pool,trace=replacement_mount_group_pool(ranked,{0})
        self.assertEqual([sid for _,sid in pool],[2,3,4,5,6,7])
        self.assertEqual(trace['phase'],'untried_mount_group')
        pool,trace=replacement_mount_group_pool(ranked,{0,2,4,6})
        self.assertEqual([sid for _,sid in pool],[7])
        self.assertEqual(trace['phase'],'untried_mount_group')
        pool,trace=replacement_mount_group_pool(ranked,{0,2,4,6,7})
        self.assertEqual([sid for _,sid in pool],[1,3,5])
        self.assertEqual(trace['phase'],'paired_sensor_fallback')

    def test_ee_sensors_are_independent_mount_groups(self):
        self.assertEqual([sensor_mount_group_id(sid) for sid in range(8)],
                         [0,0,1,1,2,2,3,4])
        ranked=[(rank,sid) for rank,sid in enumerate(range(8),1)]
        pool,trace=replacement_mount_group_pool(ranked,{6})
        self.assertIn(7,[sid for _,sid in pool])
        self.assertEqual(trace['mount_groups'],[[0,1],[2,3],[4,5],[6],[7]])

    def test_group_priority_preserves_rank_order_within_eligible_groups(self):
        ranked=[(1,1),(2,7),(3,0),(4,2),(5,3),(6,4),(7,5),(8,6)]
        pool,trace=replacement_mount_group_pool(ranked,{0})
        self.assertEqual([sid for _,sid in pool],[7,2,3,4,5,6])
        self.assertNotIn(1,trace['eligible_sensor_ids'])

    def test_near_duplicate_is_not_new_candidate(self):
        n,r,e,s,events,*_=self.fixture();r.offer(e);self.grant(r);gen=n._generate_active_set_waypoint
        def duplicate(*a):
            d=gen(*a);d['per_sensor_hybrid']['selected_q_vis']=r.pending['previous_q'].tolist();return d
        n._generate_active_set_waypoint=duplicate;r.process()
        self.assertEqual(events[-1]['event'],'candidate_not_distinct')

    def test_malformed_grants_cannot_publish(self):
        n,r,e,s,events,*_=self.fixture();r.offer(e)
        r.receive('request_id=1 request_id=1 granted=1');r.process()
        self.assertEqual(len(s),1)
        self.grant(r,attempts=6);r.process();self.assertEqual(events[-1]['event'],'generation_error')


if __name__=='__main__':
    unittest.main()
