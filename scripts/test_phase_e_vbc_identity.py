#!/usr/bin/env python3
"""Actual receiver callbacks and rospy serialization; no geometry certificates."""
import copy
import io
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rospy
from rospy.msg import serialize_message
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory
import test_phase_e_candidate_admission as admission
import test_phase_e_execution_vbc_hold as holds


class WirePub(admission.Pub):
    def __init__(self):
        super().__init__()
        self.sequence = 0

    def publish(self, msg):
        self.sequence += 1
        b = io.BytesIO()
        serialize_message(b, self.sequence, msg)
        self.messages.append(JointTrajectory().deserialize(b.getvalue()[4:]))


class IdentityTests(unittest.TestCase):
    def test_path_floor_evidence_preserves_exact_audit_without_commit(self):
        self.dispatch()
        stamp=self.wire.header.stamp.to_nsec()
        evidence=json.dumps(dict(schema=1,kind='unknown_ordinary_path_floor',
            trajectory_stamp_ns=str(stamp),floor_m=.02,point=[.25,0,.95]),separators=(',',':'))
        msg=String(data='trajectory_stamp_ns={} state=invalid reason=geometric_restore_path_counterexample path_counterexample_evidence={}'.format(stamp,evidence))
        self.n._verification_protocol_cb(msg)
        self.assertIn('path_counterexample_evidence='+evidence,self.n.verification_event_pub.messages[-1].data)
        self.assertFalse(self.n.committed_pub.messages)
        self.assertEqual(self.n._verification_unsafe_count,0)

    def test_foreign_path_floor_evidence_does_not_reach_current_audit(self):
        self.dispatch()
        self.n._verification_protocol_cb(String(data='trajectory_stamp_ns={} state=invalid reason=geometric_restore_path_counterexample path_counterexample_evidence=foreign'.format(self.wire.header.stamp.to_nsec()-1)))
        self.assertIsNotNone(self.n._outstanding)
        self.summary()
        self.assertEqual(self.n._commit_count,1)

    def setUp(self):
        self.b = admission.AdmissionTests(methodName='runTest')
        self.b.setUp()
        self.addCleanup(self.b.doCleanups)
        self.n = self.b.n
        self.n.initial_shell_recovery_enabled = True
        self.n.verification_pub = WirePub()

    def next_raw(self):
        self.b.advance(.001)
        self.b.raw.header.stamp = rospy.Time.from_sec(self.b.now)
        self.b.start()

    def dispatch(self):
        self.next_raw()
        self.b.gcdf()
        self.context = json.loads(self.n.shell_context_pub.messages[-1].data)
        self.wire = self.n.verification_pub.messages[-1]

    def summary(self, context=None, stamp=None):
        c = context or self.context
        self.n._global_summary_cb(String(data=(
            'trajectory_source=predicted trajectory_stamp_ns={} has_violation=0 '
            'shell_recovery_count=0 shell_recovery_evidence=none vbc_evidence=none '
            'context_request_session={} context_request_sequence={}').format(
                stamp or self.wire.header.stamp.to_nsec(), c['request_session'], c['request_sequence'])))

    def protocol(self, state, reason='fixed_fault', stamp=None):
        self.n._verification_protocol_cb(String(data='trajectory_stamp_ns={} state={} reason={}'.format(
            stamp or self.wire.header.stamp.to_nsec(), state, reason)))

    def test_cancel_before_vbc_then_wire_counter_gap_still_commits(self):
        self.next_raw()
        key = self.n._gcdf_outstanding_diag['transfer_key']
        with self.n._lock:
            self.n._admission_cancel_locked(key)
        self.n._flush_admission(); self.b.pump()
        self.dispatch()
        self.assertEqual(self.context['schema'], 3)
        self.assertEqual(self.context['request_sequence'], 2)
        self.assertEqual(self.wire.header.seq, 1)
        self.assertEqual(self.context['q'], [list(p.positions) for p in self.wire.points])
        self.summary()
        self.assertEqual(self.n._commit_count, 1)

    def test_upstream_rejection_then_wire_counter_gap_still_commits(self):
        self.next_raw(); self.b.gcdf(False); self.b.pump()
        self.dispatch()
        self.assertEqual(self.context['request_sequence'], 2)
        self.assertEqual(self.wire.header.seq, 1)
        self.summary()
        self.assertEqual(self.n._commit_count, 1)

    def test_waiting_context_preserves_original_deadline_then_certifies(self):
        self.dispatch(); sent = self.n._outstanding_sent
        self.b.advance(.1); self.protocol('waiting', 'missing_context')
        self.assertEqual(self.n._outstanding_sent, sent)
        self.assertFalse(self.n.committed_pub.messages)
        self.protocol('validated', 'exact_content_bound'); self.summary()
        self.assertEqual(self.n._commit_count, 1)
        self.assertEqual(self.n._verification_timeout_count, 0)

    def test_missing_context_still_times_out_at_original_limit(self):
        self.dispatch(); self.protocol('waiting', 'missing_context')
        self.b.advance(.251); self.n._timer_cb(None)
        self.assertEqual(self.n._verification_timeout_count, 1)
        self.assertIn('vbc_protocol_reason=missing_context', self.n.verification_event_pub.messages[-1].data)
        self.assertFalse(self.n.committed_pub.messages)

    def test_invalid_protocol_is_terminal_once_without_geometry_failure(self):
        self.dispatch(); self.protocol('invalid', 'context_q_mismatch')
        self.protocol('invalid', 'context_q_mismatch'); self.summary()
        self.assertEqual(self.n._verification_protocol_invalid_count, 1)
        self.assertEqual(self.n._verification_unsafe_count, 0)
        self.assertEqual(self.n._verification_timeout_count, 0)
        self.assertEqual(self.n._repair_prefix_unsafe_count, 0)
        self.assertFalse(self.n.gcdf_rejection_feedback_pub.messages)
        self.assertFalse(self.n.committed_pub.messages)
        self.assertIn('result=protocol_invalid', self.n.verification_event_pub.messages[-1].data)

    def test_foreign_audit_stamp_cannot_cancel_outstanding(self):
        self.dispatch(); self.protocol('invalid', stamp=self.wire.header.stamp.to_nsec()-1)
        self.assertIsNotNone(self.n._outstanding)
        self.assertEqual(self.n._verification_protocol_invalid_count, 0)
        self.summary(); self.assertEqual(self.n._commit_count, 1)

    def test_foreign_session_with_copied_stamp_cannot_commit(self):
        self.dispatch(); c = copy.deepcopy(self.context); c['request_session'] = 'old_process'
        self.summary(c)
        self.assertEqual(self.n._verification_protocol_invalid_count, 1)
        self.assertFalse(self.n.committed_pub.messages)

    def test_foreign_business_sequence_with_copied_stamp_cannot_commit(self):
        self.dispatch(); c = copy.deepcopy(self.context); c['request_sequence'] += 1
        self.summary(c)
        self.assertEqual(self.n._verification_protocol_invalid_count, 1)
        self.assertFalse(self.n.committed_pub.messages)

    def test_duplicate_safe_reply_commits_once(self):
        self.dispatch(); self.summary(); self.summary()
        self.assertEqual(self.n._commit_count, 1)

    def test_old_protocol_error_after_next_dispatch_cannot_cancel_it(self):
        self.dispatch(); old = self.wire.header.stamp.to_nsec()
        self.summary(); self.b.pump(); self.dispatch()
        self.protocol('invalid', stamp=old)
        self.assertIsNotNone(self.n._outstanding)
        self.summary(); self.assertEqual(self.n._commit_count, 2)


class RegimeProtocolTests(unittest.TestCase):
    def test_protocol_error_preserves_mode_direction_and_unsafe_memory(self):
        b = holds.ExecutionHoldTests(methodName='runTest'); b.setUp(); self.addCleanup(b.doCleanups)
        n = b.m
        n.state = n.REPAIR
        n.candidate_unsafe_streak = 2
        old = (n.state, n.candidate_unsafe_streak, n.candidate_unsafe_count,
               n.probe_failure_count, len(n.replan_request_pub.messages))
        n._candidate_outcome_cb(String(data='seq=1 result=protocol_invalid committed=0 safety_gate=vbc_protocol'))
        self.assertEqual((n.state, n.candidate_unsafe_streak, n.candidate_unsafe_count,
                          n.probe_failure_count, len(n.replan_request_pub.messages)), old)
        self.assertEqual(n.candidate_protocol_invalid_count, 1)
        self.assertEqual(n.candidate_timeout_count, 0)


if __name__ == '__main__':
    unittest.main()
