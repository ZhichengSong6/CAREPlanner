#!/usr/bin/env python3
"""Offline protocol/publication regression. Gate/seen inputs are synthetic."""
import copy
import json
import sys
import threading
import unittest
from unittest.mock import patch
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
    def expired_path_fixture(self, reason='geometric_restore_path_counterexample', deadline_ns=1100000000):
        fixture = self.fixture()
        n, r, e, sent, events, clock, helper = fixture
        old = self.final_trigger(n, clock, reason=reason)
        r.offer_final_vbc(old)
        self.grant(r, deadline_ns=deadline_ns)
        trigger = self.final_trigger(n, clock, reason='repair_qp_failure',
            query_stamp_ns=1150000000, raw_candidate_stamp_ns=0,
            audited_trajectory_stamp_ns=0)
        return fixture, old, trigger

    def test_expired_path_handoff_is_independent_of_callback_order(self):
        for before_cleanup in (True, False):
            with self.subTest(before_cleanup=before_cleanup):
                (n,r,e,s,events,clock,h),old,trigger=self.expired_path_fixture()
                if before_cleanup:
                    r.offer_final_vbc(trigger)
                    self.assertIsNotNone(r.waiting_qp_trigger)
                    self.assertEqual(len(s),1)
                h.bounded(r.process)
                if not before_cleanup:r.offer_final_vbc(trigger)
                self.assertEqual(len(s),3)
                self.assertIn('action=finish request_id=1 ',s[1])
                self.assertIn('action=reserve request_id=2 ',s[2])
                self.assertIn('query_stamp_ns=1150000000',s[2])
                self.assertIsNone(r.waiting_qp_trigger)
                self.assertEqual(r.pending['reason'],'repair_qp_failure')

    def test_all_existing_replace_reasons_can_defer_same_context_planner_trigger(self):
        reasons=('repair_qp_failure','observation_no_progress',
            'geometric_restore_path_counterexample','geometric_restore_proof_resource_exhausted',
            'geometric_restore_handoff_residual_insufficient','final_vbc_repeat','final_gcdf_repeat')
        for reason in reasons:
            with self.subTest(reason=reason):
                if reason in ('repair_qp_failure','observation_no_progress'):
                    n,r,e,s,events,clock,h=self.fixture()
                    old=self.final_trigger(n,clock,reason=reason,
                        raw_candidate_stamp_ns=0,audited_trajectory_stamp_ns=0)
                    r.offer_final_vbc(old);self.grant(r,deadline_ns=1100000000)
                    trigger=self.final_trigger(n,clock,reason='repair_qp_failure',
                        query_stamp_ns=1150000000,raw_candidate_stamp_ns=0,audited_trajectory_stamp_ns=0)
                else:
                    (n,r,e,s,events,clock,h),old,trigger=self.expired_path_fixture(reason)
                r.offer_final_vbc(trigger);self.assertIsNotNone(r.waiting_qp_trigger)
                h.bounded(r.process)
                self.assertEqual(len(s),3)

    def test_duplicate_active_trigger_does_not_become_next_reservation(self):
        (n,r,e,s,events,clock,h),old,trigger=self.expired_path_fixture()
        for _ in range(3):r.offer_final_vbc(old)
        self.assertIsNone(r.waiting_qp_trigger)
        h.bounded(r.process)
        r.offer_final_vbc(old)
        self.assertEqual(len(s),2)
        self.assertIsNone(r.pending)

    def test_busy_trigger_checks_exact_owner_epoch_token_region(self):
        for change in ('owner','epoch','token','region','operation'):
            with self.subTest(change=change):
                (n,r,e,s,events,clock,h),old,trigger=self.expired_path_fixture()
                if change=='owner':trigger=trigger.replace('obligation_id=1','obligation_id=2')
                if change=='epoch':trigger=trigger.replace('mode_epoch=3','mode_epoch=4')
                if change=='token':trigger=trigger.replace('observation_token='+r.pending['event']['observation_token'],
                    'observation_token=care_obs_v1_foreign')
                if change=='region':n._obligations[0]['points']=np.array([[8.,8.,8.]])
                if change=='operation':r.pending['operation']='restore_ancestor'
                r.offer_final_vbc(trigger)
                self.assertIsNone(r.waiting_qp_trigger)
                self.assertEqual(len(s),1)

    def test_waiting_full_identity_and_original_deadline_are_immutable(self):
        (n,r,e,s,events,clock,h),old,trigger=self.expired_path_fixture()
        r.offer_final_vbc(trigger);waiting=copy.deepcopy(r.waiting_qp_trigger)
        for _ in range(3):r.offer_final_vbc(trigger)
        changed=trigger.replace('reason=repair_qp_failure','reason=observation_no_progress')
        r.offer_final_vbc(changed)
        self.assertEqual(r.waiting_qp_trigger,waiting)
        h.bounded(r.process)
        self.assertEqual(len(s),3)
        r.offer_final_vbc(trigger);self.assertEqual(len(s),3)

    def test_deferred_trigger_ros_wall_expiry_and_clock_rollback(self):
        for change in ('ros','wall','negative'):
            with self.subTest(change=change):
                (n,r,e,s,events,clock,h),old,trigger=self.expired_path_fixture()
                r.offer_final_vbc(trigger)
                if change=='ros':clock.now+=1.01
                elif change=='negative':clock.now-=.01
                else:r.waiting_qp_trigger['expires_wall']=0.
                h.bounded(r.process)
                self.assertIsNone(r.waiting_qp_trigger)
                self.assertEqual(len(s),2)

    def test_deferred_trigger_revalidates_region_and_published_token_after_cleanup(self):
        for change in ('owner','region','token'):
            with self.subTest(change=change):
                (n,r,e,s,events,clock,h),old,trigger=self.expired_path_fixture()
                r.offer_final_vbc(trigger)
                if change=='owner':n._repair_stack=[2]
                elif change=='region':n._obligations[0]['points']=np.array([[8.,8.,8.]])
                else:n._trace_published_target['observation_token']='care_obs_v1_new'
                h.bounded(r.process)
                self.assertIsNone(r.pending);self.assertEqual(len(s),2)

    def test_deferred_trigger_does_not_survive_successful_token_change(self):
        (n,r,e,s,events,clock,h),old,trigger=self.expired_path_fixture(deadline_ns=2000000000)
        r.offer_final_vbc(trigger)
        h.bounded(r.process)
        self.assertIsNone(r.pending);self.assertIsNone(r.waiting_qp_trigger)
        self.assertEqual(len(s),2)
        self.assertIn('generation_result=applied_requires_existing_gates',s[-1])

    def test_late_old_grant_and_denied_new_budget_cannot_reopen_request(self):
        (n,r,e,s,events,clock,h),old,trigger=self.expired_path_fixture()
        old_grant=dict(r.grant)
        r.offer_final_vbc(trigger);h.bounded(r.process)
        r.receive(' '.join(str(k)+'='+str(v) for k,v in old_grant.items()))
        self.assertIsNone(r.grant)
        self.grant(r,granted=0,attempts=5,segments=6,query_stamp_ns=1150000000)
        h.bounded(r.process);r.offer_final_vbc(trigger);h.bounded(r.process)
        self.assertEqual(len(s),3);self.assertIsNone(r.pending)

    def test_deferred_trigger_generation_still_uses_existing_grant_and_gates(self):
        (n,r,e,s,events,clock,h),old,trigger=self.expired_path_fixture()
        r.offer_final_vbc(trigger);h.bounded(r.process)
        self.grant(r,attempts=5,query_stamp_ns=1150000000)
        h.bounded(r.process)
        self.assertEqual(len(s),4)
        self.assertIn('generation_result=applied_requires_existing_gates',s[-1])

    def test_waiting_trigger_has_priority_during_concurrent_ordinary_offer(self):
        (n,r,e,s,events,clock,h),old,trigger=self.expired_path_fixture()
        r.offer_final_vbc(trigger)
        retired=threading.Event();offering=threading.Event();measured=threading.Event()
        original=r._finish_generation
        def finish(*args,**kwargs):
            original(*args,**kwargs);retired.set()
            self.assertTrue(offering.wait(2.))
            self.assertTrue(measured.wait(2.))
        r._finish_generation=finish
        def offer():
            if retired.wait(2.):offering.set();r.offer(e)
        def measurement():
            if retired.wait(2.):r.seed_gate.observe_measured(np.zeros(7),clock.now);measured.set()
        threads=[threading.Thread(target=offer),threading.Thread(target=measurement)]
        for thread in threads:thread.start()
        h.bounded(r.process)
        for thread in threads:thread.join(2.);self.assertFalse(thread.is_alive())
        self.assertEqual(len(s),3);self.assertEqual(r.pending['reason'],'repair_qp_failure')

    def test_proof_resource_replaces_full_owner_region_with_real_rejected_pair(self):
        n,r,e,s,events,clock,h=self.fixture()
        points=n._obligations[0]['points'].copy()
        r.offer_final_vbc(self.final_trigger(n,clock,
            reason='geometric_restore_proof_resource_exhausted'))
        self.assertIn('reason=geometric_restore_proof_resource_exhausted',s[-1])
        self.assertGreater(r.pending['event']['raw_candidate_stamp_ns'],0)
        self.assertGreater(r.pending['event']['audited_trajectory_stamp_ns'],0)
        np.testing.assert_equal(r.pending['points'],points)
        self.grant(r,attempts=2);h.bounded(r.process)
        self.assertEqual(events[-1]['event'],'applied_requires_existing_gates')
        np.testing.assert_equal(n._obligations[0]['points'],points)
        self.assertEqual(r.qp_owner_lock['stage'],'hard_qp')

    def test_proof_resource_context_and_original_budget_gates(self):
        for change in ['stale','raw','audit','token','region','denied']:
            with self.subTest(change=change):
                n,r,e,s,events,clock,h=self.fixture()
                kw=dict(reason='geometric_restore_proof_resource_exhausted')
                if change=='stale':kw['query_ros_s']=-1.
                if change=='raw':kw['raw_candidate_stamp_ns']=0
                if change=='audit':kw['audited_trajectory_stamp_ns']=0
                if change=='token':kw['observation_token']='care_obs_v1_stale'
                r.offer_final_vbc(self.final_trigger(n,clock,**kw))
                if change in ['stale','raw','audit','token']:
                    self.assertEqual(s,[]);continue
                self.grant(r,granted=0 if change=='denied' else 1,attempts=5)
                if change=='region':n._obligations[0]['points']=np.array([[9.,9.,9.]])
                n._generate_active_set_waypoint=lambda *a:self.fail('invalid path replacement generated')
                h.bounded(r.process)
                self.assertIsNone(r.pending)
                self.assertIsNone(r.qp_owner_lock)

    def test_handoff_residual_replaces_full_owner_region_with_real_rejected_pair(self):
        n,r,e,s,events,clock,h=self.fixture()
        points=n._obligations[0]['points'].copy()
        r.offer_final_vbc(self.final_trigger(n,clock,
            reason='geometric_restore_handoff_residual_insufficient'))
        self.assertIn('reason=geometric_restore_handoff_residual_insufficient',s[-1])
        self.assertGreater(r.pending['event']['raw_candidate_stamp_ns'],0)
        self.assertGreater(r.pending['event']['audited_trajectory_stamp_ns'],0)
        np.testing.assert_equal(r.pending['points'],points)
        self.grant(r,attempts=2);h.bounded(r.process)
        self.assertEqual(events[-1]['event'],'applied_requires_existing_gates')
        np.testing.assert_equal(n._obligations[0]['points'],points)
        self.assertEqual(r.qp_owner_lock['stage'],'hard_qp')

    def test_handoff_residual_context_and_original_budget_gates(self):
        for change in ['stale','raw','audit','token','region','denied']:
            with self.subTest(change=change):
                n,r,e,s,events,clock,h=self.fixture()
                kw=dict(reason='geometric_restore_handoff_residual_insufficient')
                if change=='stale':kw['query_ros_s']=-1.
                if change=='raw':kw['raw_candidate_stamp_ns']=0
                if change=='audit':kw['audited_trajectory_stamp_ns']=0
                if change=='token':kw['observation_token']='care_obs_v1_stale'
                r.offer_final_vbc(self.final_trigger(n,clock,**kw))
                if change in ['stale','raw','audit','token']:
                    self.assertEqual(s,[]);continue
                self.grant(r,granted=0 if change=='denied' else 1,attempts=5)
                if change=='region':n._obligations[0]['points']=np.array([[9.,9.,9.]])
                n._generate_active_set_waypoint=lambda *a:self.fail('invalid path replacement generated')
                h.bounded(r.process)
                self.assertIsNone(r.pending)
                self.assertIsNone(r.qp_owner_lock)

    def test_path_counterexample_replaces_full_owner_region_with_real_rejected_pair(self):
        n,r,e,s,events,clock,h=self.fixture()
        points=n._obligations[0]['points'].copy()
        r.offer_final_vbc(self.final_trigger(n,clock,
            reason='geometric_restore_path_counterexample'))
        self.assertIn('reason=geometric_restore_path_counterexample',s[-1])
        self.assertGreater(r.pending['event']['raw_candidate_stamp_ns'],0)
        self.assertGreater(r.pending['event']['audited_trajectory_stamp_ns'],0)
        np.testing.assert_equal(r.pending['points'],points)
        self.grant(r,attempts=2);h.bounded(r.process)
        self.assertEqual(events[-1]['event'],'applied_requires_existing_gates')
        np.testing.assert_equal(n._obligations[0]['points'],points)
        self.assertEqual(r.qp_owner_lock['stage'],'hard_qp')

    def test_path_counterexample_context_and_original_budget_gates(self):
        for change in ['stale','raw','audit','token','region','denied']:
            with self.subTest(change=change):
                n,r,e,s,events,clock,h=self.fixture()
                kw=dict(reason='geometric_restore_path_counterexample')
                if change=='stale':kw['query_ros_s']=-1.
                if change=='raw':kw['raw_candidate_stamp_ns']=0
                if change=='audit':kw['audited_trajectory_stamp_ns']=0
                if change=='token':kw['observation_token']='care_obs_v1_stale'
                r.offer_final_vbc(self.final_trigger(n,clock,**kw))
                if change in ['stale','raw','audit','token']:
                    self.assertEqual(s,[]);continue
                self.grant(r,granted=0 if change=='denied' else 1,attempts=5)
                if change=='region':n._obligations[0]['points']=np.array([[9.,9.,9.]])
                n._generate_active_set_waypoint=lambda *a:self.fail('invalid path replacement generated')
                h.bounded(r.process)
                self.assertIsNone(r.pending)
                self.assertIsNone(r.qp_owner_lock)

    def test_final_gcdf_repeat_preserves_source_and_uses_existing_head_gates(self):
        n,r,e,s,events,clock,h=self.fixture()
        r.offer_final_vbc(self.final_trigger(n,clock,reason='final_gcdf_repeat'))
        self.assertIn('reason=final_gcdf_repeat',s[-1])
        self.assertGreater(r.pending['event']['raw_candidate_stamp_ns'],0)
        self.grant(r,attempts=5);h.bounded(r.process)
        self.assertEqual(events[-1]['event'],'applied_requires_existing_gates')

    def test_final_gcdf_repeat_stale_or_budget_denied_never_generates(self):
        for change in ['stale','missing_raw','missing_audit','denied']:
            with self.subTest(change=change):
                n,r,e,s,events,clock,h=self.fixture()
                kw=dict(reason='final_gcdf_repeat')
                if change=='stale':kw['query_ros_s']=-1.
                if change=='missing_raw':kw['raw_candidate_stamp_ns']=0
                if change=='missing_audit':kw['audited_trajectory_stamp_ns']=0
                r.offer_final_vbc(self.final_trigger(n,clock,**kw))
                if change!='denied':self.assertEqual(s,[]);continue
                n._generate_active_set_waypoint=lambda *a:self.fail('denied budget generated')
                self.grant(r,granted=0,attempts=5);h.bounded(r.process)
                self.assertIsNone(r.pending)

    def test_owner_restore_keeps_each_pinned_pose_mask_and_parent_identity(self):
        n,r,e,s,events,clock,h=self.fixture()
        parent=n._obligations[0]
        parent['q_vis_joint_mask']=np.array([1,1,0,0,0,0,0.])
        published=dict(parent,q_vis=np.arange(7)*.1,q_vis_joint_mask=np.ones(7))
        n._traced_waypoint(published,published['q_vis'])
        token=n._trace_published_target['observation_token']
        owner_mask=np.array([1,1,1,1,0,0,0.])
        n._dependency_pins[2]['q_vis_joint_mask']=owner_mask.copy()
        r.offer_final_vbc(self.final_trigger(n,clock,repeated_points='2,0,.5'))
        self.assertEqual(r.pending['operation'],'restore_ancestor')
        self.grant(r);h.bounded(r.process)
        np.testing.assert_equal(n._trace_published_target['q_vis_joint_mask'],owner_mask)
        n.seen_ids.add(2);clock.now+=.1;h.bounded(n._update_actual_visibility_completion)
        self.assertEqual(n._trace_published_target['observation_token'],token)
        np.testing.assert_equal(n._trace_published_target['q_vis_joint_mask'],np.ones(7))
        self.assertEqual(len(r.owner_promotions),1)

    def test_replacement_pin_keeps_new_mask_when_stored_target_changes(self):
        n,r,e,s,events,clock,h=self.fixture()
        r.offer(e);self.grant(r);h.bounded(r.process)
        active=next(o for o in n._obligations if o['id']==n._repair_stack[-1])
        active['q_vis_joint_mask']=np.array([1,1,0,0,0,0,0.])
        n._publish_schedule()
        np.testing.assert_equal(n._trace_published_target['q_vis_joint_mask'],np.ones(7))

    def test_observation_no_progress_uses_bounded_replacement_and_full_region(self):
        n,r,e,s,events,clock,h=self.fixture()
        points=n._obligations[0]['points'].copy()
        trigger=self.final_trigger(n,clock,reason='observation_no_progress',
            raw_candidate_stamp_ns=0,audited_trajectory_stamp_ns=0)
        r.offer_final_vbc(trigger)
        self.assertIn('reason=observation_no_progress',s[-1])
        np.testing.assert_equal(r.pending['points'],points)
        self.grant(r,attempts=5)
        h.bounded(r.process)
        self.assertEqual(events[-1]['event'],'applied_requires_existing_gates')
        np.testing.assert_equal(n._obligations[0]['points'],points)
        self.assertEqual(r.qp_owner_lock['stage'],'hard_qp')

    def test_observation_no_progress_rejects_stale_identity_and_denied_budget(self):
        for change in ['stale','raw','token','region_after_reserve','denied']:
            with self.subTest(change=change):
                n,r,e,s,events,clock,h=self.fixture()
                kw=dict(reason='observation_no_progress',raw_candidate_stamp_ns=0,
                        audited_trajectory_stamp_ns=0)
                if change=='stale':kw['query_ros_s']=-1.
                if change=='raw':kw['raw_candidate_stamp_ns']=100
                if change=='token':kw['observation_token']='care_obs_v1_stale'
                r.offer_final_vbc(self.final_trigger(n,clock,**kw))
                if change in ['stale','raw','token']:
                    self.assertEqual(s,[]);continue
                self.grant(r,granted=0 if change=='denied' else 1,attempts=5)
                if change=='region_after_reserve':n._obligations[0]['points']=np.array([[9.,9.,9.]])
                n._generate_active_set_waypoint=lambda *a:self.fail('invalid observation recovery generated')
                h.bounded(r.process)
                self.assertIsNone(r.pending)
                self.assertIsNone(r.qp_owner_lock)

    def test_observation_no_progress_survives_busy_denial_without_deadline_renewal(self):
        n,r,e,s,events,clock,h=self.fixture();r.offer(e)
        trigger=self.final_trigger(n,clock,reason='observation_no_progress',
            raw_candidate_stamp_ns=0,audited_trajectory_stamp_ns=0)
        r.offer_final_vbc(trigger);deadline=r.waiting_qp_trigger['expires_wall']
        r.offer_final_vbc(trigger);self.assertEqual(deadline,r.waiting_qp_trigger['expires_wall'])
        self.grant(r,granted=0);h.bounded(r.process)
        self.assertEqual(len(s),2);self.assertIn('reason=observation_no_progress',s[-1])

    def deferred_qp_fixture(self):
        fixture = self.fixture()
        n, r, e, sent, events, clock, helper = fixture
        r.offer(e)
        trigger = self.final_trigger(n, clock, reason='repair_qp_failure',
            raw_candidate_stamp_ns=0, audited_trajectory_stamp_ns=0)
        r.offer_final_vbc(trigger)
        return fixture, trigger

    def test_busy_qp_trigger_handoff_after_denied_dependency(self):
        (n, r, e, sent, events, clock, helper), trigger = self.deferred_qp_fixture()
        self.assertEqual(len(sent), 1)
        self.grant(r, granted=0, attempts=0, segments=0)
        helper.bounded(r.process)
        self.assertEqual(len(sent), 2)
        self.assertIn('reason=repair_qp_failure', sent[-1])
        self.assertEqual(r.pending['event']['query_stamp_ns'], e['query_stamp_ns'])
        self.assertIsNone(r.waiting_qp_trigger)
        self.grant(r, attempts=1)
        helper.bounded(r.process)
        self.assertEqual(events[-1]['event'], 'applied_requires_existing_gates')

    def test_busy_qp_trigger_expiry_context_duplicates_and_budget(self):
        for change in ('ros_expiry', 'wall_expiry', 'region_changed', 'token_changed',
                       'duplicate', 'different_trigger', 'budget_denied'):
            with self.subTest(change=change):
                (n, r, e, sent, events, clock, helper), trigger = self.deferred_qp_fixture()
                self.assertIsNotNone(r.waiting_qp_trigger)
                deadline = r.waiting_qp_trigger['expires_wall']
                if change == 'ros_expiry':
                    clock.now += 1.01
                elif change == 'wall_expiry':
                    r.waiting_qp_trigger['expires_wall'] = 0.
                elif change == 'region_changed':
                    n._obligations[0]['points'] = np.array([[.1, .2, .9]])
                elif change == 'token_changed':
                    n._trace_published_target['observation_token'] = 'care_obs_v1_changed'
                elif change == 'duplicate':
                    for _ in range(3):
                        r.offer_final_vbc(trigger)
                    self.assertEqual(r.waiting_qp_trigger['expires_wall'], deadline)
                elif change == 'different_trigger':
                    r.offer_final_vbc(trigger.replace('query_stamp_ns=1000000000',
                                                    'query_stamp_ns=1000000001'))
                    self.assertEqual(r.waiting_qp_trigger['event']['query_stamp_ns'], 1000000000)
                # A new ordinary observation must not preempt the queued trigger.
                self.grant(r, granted=0, attempts=0, segments=0)
                helper.bounded(r.process)
                self.assertIsNone(r.waiting_qp_trigger)
                if change in ('ros_expiry', 'wall_expiry', 'region_changed', 'token_changed'):
                    self.assertEqual(len(sent), 1)
                    self.assertIsNone(r.pending)
                else:
                    self.assertEqual(len(sent), 2)
                    self.grant(r, granted=0, attempts=5, segments=6)
                    helper.bounded(r.process)
                    helper.bounded(r.process)
                    self.assertEqual(len(sent), 2)
                    self.assertIsNone(r.pending)
                    self.assertIsNone(r.qp_owner_lock)

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

    def test_qp_failure_accepts_published_pinned_target_with_live_region(self):
        n,r,e,s,events,clock,h=self.fixture()
        active=next(o for o in n._obligations if o['id']==n._repair_stack[-1])
        n._dependency_pins[active['id']]=dict(
            region=region_key(active['points']),q_vis=np.full(7,.3))
        n._publish_schedule()
        self.assertNotEqual(observation_token(active),
                            n._trace_published_target['observation_token'])
        trigger=self.final_trigger(n,clock,reason='repair_qp_failure',
            raw_candidate_stamp_ns=0,audited_trajectory_stamp_ns=0)
        r.offer_final_vbc(trigger)
        self.assertEqual(len(s),1)
        self.assertIn('reason=repair_qp_failure',s[0])

        n2,r2,e2,s2,events2,clock2,h2=self.fixture()
        active2=next(o for o in n2._obligations if o['id']==n2._repair_stack[-1])
        stale_trigger=self.final_trigger(n2,clock2,reason='repair_qp_failure',
            raw_candidate_stamp_ns=0,audited_trajectory_stamp_ns=0)
        active2['points']=np.vstack([active2['points'],[.15,-.1,.85]])
        r2.offer_final_vbc(stale_trigger)
        self.assertEqual(s2,[])
        self.assertEqual(events2[-1]['reason'],'live_observation_token_changed')

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
        r.test_clock=clock
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
        if str(d['granted']) == '1':
            r.seed_gate.observe_commit_ack(' '.join(f'{k}={d[k]}' for k in
                ('request_id','mode_epoch','query_stamp_ns','deadline_ns')) +
                ' confirmed=1 last_execution_stamp_ns=0', r.test_clock.now)
        if (r.pending is not None and r.pending['operation']=='replace' and
                str(d['granted'])=='1'):
            # Default fixture supplies a stationary, fresh post-grant window.
            # Dedicated scheduling tests below explicitly drive moving samples.
            for _ in range(6):
                r.test_clock.now+=.025
                q=r.node._trace_measured[0]
                r.node._trace_measured=(q,r.test_clock.now)
                r.seed_gate.observe_measured(q,r.test_clock.now)
                r.seed_gate.observe_tracker('active=0 execution_stamp_ns=0',r.test_clock.now)

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

    def test_same_parent_owner_relation_promotes_once_then_replaces_parent(self):
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
        self.assertEqual(len(s),sent+1)
        self.assertEqual(r.pending['operation'],'replace')
        self.assertEqual(r.pending['active_id'],1)
        self.assertIn('action=reserve ',s[-1])
        self.assertIn('reason=final_vbc_repeat ',s[-1])
        self.assertEqual(events[-1]['event'],'owner_exhausted_replacement_requested')
        # Repeated delivery while pending cannot mint a second reservation.
        r.offer_final_vbc(trigger());self.assertEqual(len(s),sent+1)
        parent=next(ob for ob in n._obligations if ob['id']==1)
        before=parent['points'].copy()
        self.grant(r);h.bounded(r.process)
        np.testing.assert_equal(parent['points'],before)
        self.assertEqual(n._repair_stack,[1])
        self.assertEqual(n._dependency_edges,{})
        self.assertEqual(events[-1]['event'],'applied_requires_existing_gates')

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

    def test_ancestor_owner_promotion_once_then_child_sensor_reservation(self):
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
        self.assertEqual(r.pending['operation'],'replace')
        self.assertEqual(events[-1]['event'],'owner_exhausted_replacement_requested')
        self.assertEqual(sum('action=reserve ' in msg for msg in s),1)
        self.assertEqual(n._repair_stack,[2,1])

    def test_dependency_cycle_restores_existing_ancestor_once_in_drive_mode(self):
        n,r,e,s,events,clock,h=self.fixture()
        n.rolling_guidance_mode='drive'
        token=n._trace_published_target['observation_token']
        r.offer(e)
        self.assertEqual(r.pending['operation'],'restore_ancestor')
        self.assertIn('reason=observation_dependency_owner',s[-1])
        self.assertEqual(n._repair_stack,[2,1])
        self.grant(r);h.bounded(r.process)
        self.assertEqual(n._repair_stack,[1,2])
        self.assertEqual(len(r.owner_promotions),1)
        self.assertEqual(len(n._obligations),3)
        n._repair_stack=[2,1];n._publish_schedule()
        e['observation_token']=n._trace_published_target['observation_token']
        e['query_ros_s']=clock.now
        r.offer(e)
        self.assertEqual(r.pending['operation'],'replace')
        self.assertEqual(sum('action=reserve_owner' in msg for msg in s),1)

    def exhausted_owner_fixture(self):
        data=self.fixture();n,r,e,s,events,clock,h=data
        other=next(o for o in n._obligations if o['id']==2)
        repeated=','.join(str(float(v)) for v in other['points'][0])
        n._repair_stack=[2,1];n._publish_schedule()
        r.offer_final_vbc(self.final_trigger(n,clock,repeated_points=repeated))
        self.grant(r);h.bounded(r.process)
        n._repair_stack=[2,1];n._publish_schedule()
        return data,repeated

    def test_exhausted_owner_tracks_external_geometry_during_generation(self):
        (n,r,e,s,events,clock,h),point=self.exhausted_owner_fixture()
        r.offer_final_vbc(self.final_trigger(n,clock,repeated_points=point))
        self.grant(r)
        next(o for o in n._obligations if o['id']==2)['points']=np.array([[9.,9.,9.]])
        n._generate_active_set_waypoint=lambda *a:self.fail('stale external owner')
        r.process()
        self.assertEqual(events[-1]['event'],'context_changed')
        self.assertIn('new_token=none',s[-1])

    def test_exhausted_owner_denial_cannot_generate(self):
        (n,r,e,s,events,clock,h),point=self.exhausted_owner_fixture()
        r.offer_final_vbc(self.final_trigger(n,clock,repeated_points=point))
        self.grant(r,granted=0,attempts=5)
        n._generate_active_set_waypoint=lambda *a:self.fail('budget denied')
        r.process();self.assertIsNone(r.pending)
        self.assertEqual(events[-1]['event'],'reservation_denied')

    def test_exhausted_owner_failed_head_is_inherited(self):
        (n,r,e,s,events,clock,h),point=self.exhausted_owner_fixture()
        r.offer_final_vbc(self.final_trigger(n,clock,repeated_points=point));self.grant(r)
        n._generate_active_set_waypoint=lambda *a:dict(per_sensor_hybrid=dict(
            accepted=False,attempts=[dict(sensor_id=0)]))
        r.process();self.assertEqual(events[-1]['event'],'no_geometric_candidate')
        r.offer_final_vbc(self.final_trigger(n,clock,repeated_points=point,query_stamp_ns=1000000001))
        self.grant(r,query_stamp_ns=1000000001,attempts=3)
        def next_head(*a):
            self.assertIn(0,n._candidate_replacement_override['excluded'])
            return dict(per_sensor_hybrid=dict(accepted=False,attempts=[dict(sensor_id=4)]))
        n._generate_active_set_waypoint=next_head;r.process()
        self.assertTrue({0,4}.issubset(next(iter(r.families.values()))))

    def test_exhausted_owner_empty_sensor_pool_explicitly_declines(self):
        (n,r,e,s,events,clock,h),point=self.exhausted_owner_fixture()
        ob=next(o for o in n._obligations if o['id']==1)
        r.families[(3,1,ob.get('generation_event_id','legacy'))]=set(range(8))
        r.offer_final_vbc(self.final_trigger(n,clock,repeated_points=point))
        self.assertIsNone(r.pending)
        self.assertIn('action=decline ',s[-1])
        self.assertIn('trigger_raw_candidate_stamp_ns=400 ',s[-1])
        self.assertEqual(events[-1]['event'],'owner_exhausted_replacement_unavailable')

    def test_final_vbc_trigger_requires_any_live_point_token_and_obligation(self):
        for change in [dict(repeated_points='99.0,0.0,0.5'),
                       dict(observation_token='care_obs_v1_old'),
                       dict(obligation_id=3), dict(query_ros_s=-1.)]:
            n,r,e,s,events,clock,h=self.fixture()
            r.offer_final_vbc(self.final_trigger(n,clock,**change))
            self.assertEqual(s,[])

    def test_trigger_waits_for_blocker_without_reserving_or_charging(self):
        n,r,e,s,events,clock,h=self.fixture()
        text=self.final_trigger(n,clock,repeated_points='99.0,0.0,0.5')
        r.offer_final_vbc(text);r.offer_final_vbc(text)
        self.assertIsNotNone(r.waiting_blocker)
        self.assertEqual((r.serial,s,r.families),(0,[],{}))
        n._obligations.append(dict(id=3,points=np.array([[99.,0.,.5]]),
            q_vis=np.zeros(7),generation_event_id='new',per_sensor_selected_sensor_id=0))
        clock.now+=.1;r._process_waiting_blocker()
        self.assertIsNone(r.waiting_blocker)
        self.assertEqual(len(s),1)
        self.assertIn('action=reserve_owner',s[0])
        r._process_waiting_blocker();self.assertEqual(len(s),1)

    def test_waiting_blocker_expiry_does_not_refresh_trigger_or_reserve(self):
        n,r,e,s,events,clock,h=self.fixture()
        r.offer_final_vbc(self.final_trigger(n,clock,repeated_points='99.0,0.0,0.5'))
        clock.now+=.501;r._process_waiting_blocker()
        self.assertIsNone(r.waiting_blocker);self.assertEqual(s,[])
        self.assertEqual(events[-1]['event'],'waiting_blocker_expired')

    def test_waiting_blocker_target_change_discards_continuation(self):
        n,r,e,s,events,clock,h=self.fixture()
        r.offer_final_vbc(self.final_trigger(n,clock,repeated_points='99.0,0.0,0.5'))
        n._trace_published_target=dict(n._trace_published_target,
            observation_token='care_obs_v1_changed')
        r._process_waiting_blocker()
        self.assertIsNone(r.waiting_blocker);self.assertEqual(s,[])
        self.assertEqual(events[-1]['event'],'waiting_blocker_context_changed')

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
        self.assertEqual(old,observation_token(n._obligations[0]))
        self.assertNotIn(7,next(iter(r.families.values())))  # expired before inference

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
        self.assertIn('generation_result=no_geometric_candidate',s[-1])

    def test_qp_generation_retry_preserves_failed_sensor_exclusion(self):
        n,r,e,s,events,clock,h=self.fixture()
        trigger=self.final_trigger(n,clock,reason='repair_qp_failure',
            raw_candidate_stamp_ns=0,audited_trajectory_stamp_ns=0)
        r.offer_final_vbc(trigger);self.grant(r)
        n._generate_active_set_waypoint=lambda *a:dict(per_sensor_hybrid=dict(
            accepted=False,attempts=[dict(sensor_id=1)]))
        r.process()
        self.assertIn('generation_result=no_geometric_candidate',s[-1])
        old_token=n._trace_published_target['observation_token']
        # The planner supplies a fresh authenticated query for the same owner.
        r.offer_final_vbc(self.final_trigger(n,clock,reason='repair_qp_failure',
            query_stamp_ns=1000000001,raw_candidate_stamp_ns=0,audited_trajectory_stamp_ns=0))
        self.grant(r,query_stamp_ns=1000000001,attempts=3)
        def next_head(*a):
            self.assertIn(1,n._candidate_replacement_override['excluded'])
            return dict(q_zero=np.zeros(7),per_sensor_hybrid=dict(accepted=True,
                selected_sensor_id=7,selected_sensor_frame='EE_sensor2_tof_link',selected_rank=7,
                selected_q_vis=[.4]*7,attempts=[dict(sensor_id=7)]))
        n._generate_active_set_waypoint=next_head;h.bounded(r.process)
        self.assertEqual(events[-1]['event'],'applied_requires_existing_gates')
        self.assertNotEqual(n._trace_published_target['observation_token'],old_token)
        self.assertEqual(next(iter(r.families.values())),{1,7})

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
        self.assertIn('generation_result=candidate_not_distinct',s[-1])

    def test_malformed_grants_cannot_publish(self):
        n,r,e,s,events,*_=self.fixture();r.offer(e)
        r.receive('request_id=1 request_id=1 granted=1');r.process()
        self.assertEqual(len(s),1)
        self.grant(r,attempts=6);r.process();self.assertEqual(events[-1]['event'],'generation_error')


if __name__=='__main__':
    unittest.main()
