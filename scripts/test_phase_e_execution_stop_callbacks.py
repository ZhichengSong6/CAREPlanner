#!/usr/bin/env python3
"""Real receiver callbacks: physical probe construction and stopping audits."""
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
import rospy
from std_msgs.msg import String,Header
from sensor_msgs.msg import JointState
import test_phase_e_recovery_motion as motion
import test_phase_e_execution_vbc_hold as holds
from execution_motion_limits import violation


class ReceiverStopTests(unittest.TestCase):
    def setUp(self):
        self.fixture=motion.MotionTests(methodName='runTest')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.base=self.fixture.base
        self.node=self.fixture.n
        self.node.enforce_execution_motion_limits=True
        self.node.execution_velocity_cap=2.
        self.node.recovery_velocity_limits=dict.fromkeys(self.base.b.raw.joint_names,2.)

    def test_probe_is_rebuilt_as_one_acceleration_consistent_curve_before_audit(self):
        n=self.node;n._probe_active=True;n._repair_active=False
        self.base.b.start()
        self.assertEqual(len(n.final_gcdf_query_pub.messages),1)
        path=n.final_gcdf_query_pub.messages[-1]
        self.assertIsNone(violation(path,[2.]*7,n.repair_acceleration_limits))
        self.assertEqual(path.points[0].positions,list(self.base.b.data['measured_q']))
        self.assertEqual(path.points[-1].velocities,[0.]*7)
        self.assertFalse(n.verification_pub.messages or n.committed_pub.messages)

    def test_stopping_wait_preserves_first_offer_deadline(self):
        n=self.node
        n._execution_tracker_cb(String(data='execution_stamp_ns=24463000000 phase_s=.654 source=stopping_hold execution_aborted=1 execution_state=STOPPING'))
        self.base.b.start()
        self.assertIsNotNone(n._pending_raw)
        self.assertFalse(n.final_gcdf_query_pub.messages or n.committed_pub.messages)
        entry=next(iter(n._admission_entries.values()));deadline=entry['deadline']
        self.base.b.advance(.01);self.base.measure();self.fixture.applied()
        n._execution_tracker_cb(String(data='execution_stamp_ns=24463000000 phase_s=.654 source=stopped_safety_hold execution_aborted=1 execution_state=STOPPED_HOLD'))
        self.assertEqual(entry['deadline'],deadline)
        self.assertEqual(len(n.final_gcdf_query_pub.messages),1)


class StoppingAuditTests(unittest.TestCase):
    def setUp(self):
        self.fixture=holds.ExecutionHoldTests(methodName='runTest')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.n=self.fixture.m

    def test_recorded_356_unsafe_verdicts_remain_bound_after_tracking_abort(self):
        f=self.fixture;n=self.n
        data=json.loads((ROOT/'outputs/phase_e_soft_mount_targeted_20261019/online_ledger_r2_010.json').read_text())
        f.token=24463000000;f.now=25.125;f.commit(f.token)
        for row in data['unsafe_execution_records']:
            f.now=row['ros_s']
            state='STOPPING' if f.now<25.56 else 'STOPPED_HOLD'
            n._tracker_summary_cb(String(data=(
                'active=0 complete=0 execution_stamp_ns=24463000000 source=stopping_hold '
                'execution_aborted=1 execution_state='+state)))
            f.audit=int(row['trajectory_stamp_ns']);f.identity();f.verdict()
            self.assertTrue(n.execution_last_unsafe)
        self.assertEqual(n.execution_unsafe_streak,356)
        self.assertEqual(n.tracker_abort_count,1)
        self.assertEqual(n.candidate_event_count,0)
        self.assertFalse(n.execution_vbc_stop_pub.messages) # already stopped; do not command the old trajectory

    def test_late_old_audit_cannot_label_a_new_execution(self):
        f=self.fixture
        old=f.token
        self.n._tracker_summary_cb(String(data=(
            'active=0 complete=0 execution_stamp_ns={} source=stopping_hold execution_aborted=1 execution_state=STOPPING').format(old)))
        f.commit(old+1000);f.tracker(old+1000)
        f.identity(token=old);f.verdict()
        self.assertFalse(self.n.execution_last_unsafe)
        self.assertIsNone(self.n.execution_summary_time)


if __name__=='__main__':
    unittest.main()
