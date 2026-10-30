#!/usr/bin/env python3
"""Exercise real execution-audit callbacks with controlled ROS transport/time."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import rospy
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/egocentric_arm_planner/scripts'))
import c4_4_verified_regime_manager_node as regime
import optimized_trajectory_continuity_node as continuity


class Pub:
    def __init__(self, *args, **kw): self.messages = []
    def publish(self, msg): self.messages.append(copy.deepcopy(msg))


class ExecutionHoldTests(unittest.TestCase):
    def setUp(self):
        self.now = 19.333
        for p in [patch.object(rospy, 'get_param', side_effect=lambda key, default=None: default),
                  patch.object(rospy, 'Publisher', Pub), patch.object(rospy, 'Subscriber'),
                  patch.object(rospy, 'Timer'), patch.object(rospy, 'logwarn'),
                  patch.object(rospy, 'loginfo'),
                  patch.object(rospy.Time, 'now', side_effect=lambda: rospy.Time.from_sec(self.now))]:
            p.start(); self.addCleanup(p.stop)
        self.m = regime.C44VerifiedRegimeManager()
        self.m.execution_ready = True
        self.m.state = self.m.REPAIR
        self.token = 19066000000
        self.audit = 19329000000
        self.commit(self.token)
        self.tracker(self.token)
        self.m.hold_pub.messages.clear()

    def commit(self, token):
        t = JointTrajectory(); t.header.stamp = rospy.Time(0, token)
        self.m._committed_trajectory_cb(t)

    def tracker(self, token, active=True):
        self.m._tracker_summary_cb(String(data=(
            f'active={int(active)} complete=0 execution_stamp_ns={token} source=active_trajectory')))

    def identity(self, audit=None, token=None):
        self.m._execution_audit_identity_cb(String(data=(
            f'audit_trajectory_stamp_ns={self.audit if audit is None else audit} '
            f'execution_stamp_ns={self.token if token is None else token}')))

    def verdict(self, unsafe=True, audit=None):
        self.m._execution_summary_cb(String(data=(
            f'vbc success=1 trajectory_source=predicted has_violation={int(unsafe)} '
            f'trajectory_stamp_ns={self.audit if audit is None else audit}')))

    def held(self):
        return any(regime._tokens(msg.data).get('execution_stamp_ns') == str(self.token)
                   for msg in self.m.execution_vbc_stop_pub.messages)

    def test_first_unsafe_holds_already_repair_without_candidate_event(self):
        self.identity(); self.verdict()
        self.assertTrue(self.held())
        self.assertEqual(self.m.vbc_unsafe_execution_stamp_ns, self.token)
        self.assertEqual(self.m.candidate_event_count, 0)
        self.m._timer_cb(None)
        self.assertFalse(self.m.hold_pub.messages[-1].data)

    def test_verdict_before_identity_has_identical_effect(self):
        self.verdict(); self.assertFalse(self.held())
        self.identity(); self.assertTrue(self.held())

    def test_late_tracker_receipt_does_not_lose_unsafe(self):
        self.tracker(self.token-1)
        self.identity(); self.verdict(); self.assertFalse(self.held())
        self.tracker(self.token); self.assertTrue(self.held())

    def test_stale_token_cannot_hold_new_execution_or_refresh_audit(self):
        self.commit(self.token+100000000); self.tracker(self.token+100000000)
        self.identity(); self.verdict()
        self.assertFalse(self.held())
        self.assertIsNone(self.m.execution_summary_time)

    def test_same_trajectory_periodic_measured_unsafe_after_safe_is_processed(self):
        self.identity(); self.verdict(False)
        self.verdict(True)
        self.assertTrue(self.held())

    def test_safe_verdict_cannot_release_latched_unsafe_token(self):
        self.identity(); self.verdict(); self.verdict(False, self.audit+1)
        self.assertEqual(self.m.vbc_unsafe_execution_stamp_ns, self.token)
        self.tracker(self.token+1)
        self.assertEqual(self.m.vbc_unsafe_execution_stamp_ns, 0)

    def test_unbound_summary_cannot_refresh_freshness(self):
        self.verdict(False)
        self.assertIsNone(self.m.execution_summary_time)
        self.now += self.m.input_timeout+.01
        self.identity()
        self.assertIsNone(self.m.execution_summary_time)
        self.assertFalse(self.held())

    def test_invalid_or_ambiguous_identity_cannot_bind(self):
        for text in ['audit_trajectory_stamp_ns=1 execution_stamp_ns=-1',
                     'audit_trajectory_stamp_ns=1 execution_stamp_ns=2 execution_stamp_ns=3',
                     'audit_trajectory_stamp_ns=1 execution_stamp_ns=18446744073709551616']:
            self.m._execution_audit_identity_cb(String(data=text))
        self.assertNotIn(1, self.m.execution_audit_identities)
        self.identity(); self.identity(token=self.token+1); self.verdict()
        self.assertFalse(self.held())

    def test_older_safe_verdict_cannot_replace_newer_unsafe(self):
        self.identity(); self.verdict()
        self.identity(self.audit-1); self.verdict(False,self.audit-1)
        self.assertTrue(self.m.execution_last_unsafe)

    def test_direct_commit_audit_preserves_compatibility(self):
        self.verdict(audit=self.token)
        self.assertTrue(self.held())

    def test_missing_identity_keeps_existing_watchdog_hold(self):
        self.m.execution_ready_time = None
        self.m._timer_cb(None)
        self.assertTrue(self.m.hold_pub.messages[-1].data)

    def candidate_rejection(self, seq=80):
        # Recorded r2_011 rejection: this is a future candidate, with no
        # committed execution identity. The current execution is independent.
        self.m._candidate_outcome_cb(String(data=(
            f'seq={seq} result=unsafe committed=0 safety_gate=vbc '
            'verification_view=full_horizon raw_stamp_ns=29487000000 '
            'audit_stamp_ns=29495000000 execution_stamp_ns=0')))

    def test_candidate_rejections_cannot_relabel_current_execution_unsafe(self):
        self.m.state = self.m.NORMAL
        self.m.execution_ready_time = None
        self.identity(); self.verdict(False)
        self.candidate_rejection()
        self.assertEqual(self.m.state, self.m.NORMAL)
        self.assertTrue(self.m.replan_request_pub.messages)
        self.candidate_rejection(81)
        self.assertEqual(self.m.state, self.m.REPAIR)
        self.assertEqual(self.m.candidate_unsafe_count, 2)
        self.assertEqual(self.m.execution_safety_event_count, 0)
        self.assertEqual(self.m.vbc_unsafe_execution_stamp_ns, 0)
        self.assertFalse(self.m.execution_vbc_stop_pub.messages)
        self.m._timer_cb(None)
        self.assertFalse(self.m.hold_pub.messages[-1].data)

    def test_candidate_rejection_preserves_bound_execution_unsafe_stop(self):
        self.identity(); self.verdict()
        self.candidate_rejection()
        self.assertTrue(self.held())
        self.assertTrue(self.m.execution_vbc_hold_tokenized)
        self.assertEqual(self.m.vbc_unsafe_execution_stamp_ns, self.token)

    def test_real_execution_unsafe_still_stops_after_candidate_rejection(self):
        self.identity(); self.verdict(False)
        self.candidate_rejection()
        self.assertFalse(self.held())
        self.verdict()
        self.assertTrue(self.held())
        self.assertTrue(self.m.execution_vbc_hold_tokenized)

    def test_candidate_rejection_does_not_refresh_execution_watchdog(self):
        self.m.execution_ready_time = None
        self.identity(); self.verdict(False)
        received = self.m.execution_summary_time
        self.now += self.m.input_timeout + .01
        self.tracker(self.token)
        self.candidate_rejection()
        self.assertEqual(self.m.execution_summary_time, received)
        self.m._timer_cb(None)
        self.assertTrue(self.m.hold_pub.messages[-1].data)

    def test_bounded_buffers(self):
        for stamp in range(1,301): self.identity(stamp)
        self.assertEqual(len(self.m.execution_audit_identities),256)
        for stamp in range(1,100): self.verdict(audit=stamp+1000)
        self.assertEqual(len(self.m.pending_execution_audits),64)

    def test_suffix_identity_keeps_original_stamp_and_frame(self):
        n = continuity.OptimizedTrajectoryContinuityNode()
        t = JointTrajectory(); t.header.stamp = rospy.Time(0,self.token)
        t.header.frame_id = 'base_link'; t.joint_names = ['joint1']
        t.points = [JointTrajectoryPoint(positions=[0.], time_from_start=rospy.Duration(0.)),
                    JointTrajectoryPoint(positions=[.1], time_from_start=rospy.Duration(.5))]
        suffix = n._suffix_from_phase(t,.2)
        n._publish_execution_audit(suffix,self.token)
        f = regime._tokens(n.execution_audit_identity_pub.messages[-1].data)
        self.assertEqual(int(f['execution_stamp_ns']),self.token)
        self.assertEqual(int(f['audit_trajectory_stamp_ns']),suffix.header.stamp.to_nsec())
        self.assertEqual(suffix.header.frame_id,'base_link')
        self.assertEqual(t.header.stamp.to_nsec(),self.token)


if __name__ == '__main__': unittest.main()
