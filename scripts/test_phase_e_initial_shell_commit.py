#!/usr/bin/env python3
"""Fault controls through the production GCDF/VBC/commit callbacks."""
import copy
import json
import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import String,Bool
import test_phase_e_candidate_admission as admission


class ShellCommitTests(unittest.TestCase):
    def setUp(self):
        self.b=admission.AdmissionTests(methodName='runTest');self.b.setUp();self.addCleanup(self.b.doCleanups)
        self.n=self.b.n;self.n.initial_shell_recovery_enabled=True;self.n.initial_shell_execution_margin_m=.01
        self.n._repair_active=True;self.n.repair_projection_consistent_dispatch_enabled=True
        self.measure()

    def measure(self,velocity=None,shift=0.):
        self.n._joint_state_cb(JointState(header=__import__('std_msgs.msg',fromlist=['Header']).Header(stamp=rospy.Time.from_sec(self.b.now)),
            name=self.b.raw.joint_names,position=[q+shift for q in self.b.data['measured_q']],velocity=velocity or [0.]*7))

    def dispatch(self):
        self.b.start();self.assertFalse(self.n.verification_pub.messages or self.n.shell_context_pub.messages)
        self.b.gcdf();self.assertFalse(self.n.committed_pub.messages)
        self.candidate=self.n.verification_pub.messages[-1]
        self.context=json.loads(self.n.shell_context_pub.messages[-1].data)
        self.assertEqual(self.context['kind'],'repair_prefix_brake_hold')
        self.e=dict(schema=1,trajectory_stamp_ns=str(self.candidate.header.stamp.to_nsec()),
            measured_stamp_ns=str(rospy.Time.from_sec(self.b.now).to_nsec()),map_snapshot_stamp_ns=str(rospy.Time.from_sec(self.b.now).to_nsec()),
            execution_floor_m=.01,actual_min_distance_m=.0198,onset_clearance_budget_m=.0098,
            points=[dict(point=[-.05,-.25,.85],floor_m=.019799,start_m=.0198,end_m=.0203)],
            measured_q=[self.n._latest_measured_q_by_name[name] for name in self.candidate.joint_names],joint_motion_radii=[1.]*7)

    def verdict(self,raw=None,count=1):
        raw=raw or json.dumps(self.e,separators=(',',':'))
        self.n._global_summary_cb(String(data='trajectory_source=predicted trajectory_stamp_ns={} has_violation=0 shell_recovery_count={} shell_recovery_evidence={} vbc_evidence=none context_request_session={} context_request_sequence={}'.format(
            self.candidate.header.stamp.to_nsec(),count,raw,self.context['request_session'],self.context['request_sequence'])))

    def test_exact_scoped_certificate_commits_original_payload(self):
        self.dispatch();self.verdict();self.assertEqual(len(self.n.committed_pub.messages),1)
        for a,b in zip(self.candidate.points,self.n.committed_pub.messages[0].points):
            self.assertEqual(a.positions,b.positions);self.assertEqual(a.velocities,b.velocities)
        self.assertEqual(len(self.candidate.points),len(self.n.committed_pub.messages[0].points))

    def test_checkpoint_rejection_cannot_request_recovery_vbc(self):
        self.b.start();self.b.gcdf(False)
        self.assertFalse(self.n.verification_pub.messages or self.n.shell_context_pub.messages or self.n.committed_pub.messages)

    def test_foreign_fk_snapshot_rejected(self):
        self.dispatch();self.e['trajectory_stamp_ns']=str(int(self.e['trajectory_stamp_ns'])+1)
        self.verdict();self.assertFalse(self.n.committed_pub.messages)

    def test_stale_measured_or_map_snapshot_rejected(self):
        self.dispatch();self.b.advance(.051);self.measure();self.verdict();self.assertFalse(self.n.committed_pub.messages)
        self.assertIn('measured_snapshot_stale', self.n.verification_event_pub.messages[-1].data)

    def test_snapshot_change_and_braking_use_clearance_budget(self):
        self.dispatch();self.measure(shift=.002);self.verdict();self.assertFalse(self.n.committed_pub.messages)
        self.assertIn('actual_clearance_consumed', self.n.verification_event_pub.messages[-1].data)

    def test_unsafe_incoming_speed_rejected_at_commit(self):
        self.dispatch();self.measure(velocity=[2.]*7);self.verdict();self.assertFalse(self.n.committed_pub.messages)

    def test_missing_speed_rejected(self):
        self.dispatch();self.n._latest_measured_velocity_by_name={};self.verdict();self.assertFalse(self.n.committed_pub.messages)

    def test_duplicate_json_key_rejected(self):
        self.dispatch();raw=json.dumps(self.e,separators=(',',':'))[:-1]+',"schema":1}'
        self.verdict(raw);self.assertFalse(self.n.committed_pub.messages)

    def test_mode_change_cancels_old_recovery(self):
        self.dispatch();self.n._repair_active_cb(Bool(data=False));self.verdict();self.assertFalse(self.n.committed_pub.messages)

    def test_normal_certificate_cannot_claim_recovery(self):
        self.dispatch();self.n._outstanding_repair=False;self.verdict();self.assertFalse(self.n.committed_pub.messages)

    def test_ordinary_vbc_still_commits(self):
        self.dispatch();self.verdict(raw='none',count=0);self.assertEqual(len(self.n.committed_pub.messages),1)


if __name__=='__main__':unittest.main()
