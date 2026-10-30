"""Recorded O5 evidence must still meet the original admission bounds."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/egocentric_arm_planner/scripts'))
import rospy
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
from optimized_trajectory_continuity_node import OptimizedTrajectoryContinuityNode


class O5CertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures=json.loads((ROOT/'outputs/phase_e_case002_o5_handoff_20261015/o5_fixture.json').read_text())['fixtures']

    def gate(self,f,age_ns=40000000):
        c=f['context'];e=copy.deepcopy(f['certificate'])
        now=int(e['measured_stamp_ns'])+age_ns
        n=OptimizedTrajectoryContinuityNode.__new__(OptimizedTrajectoryContinuityNode)
        n._outstanding=JointTrajectory(header=Header(stamp=rospy.Time(0,int(c['trajectory_stamp_ns'])),frame_id=c['frame']),joint_names=c['joint_names'])
        for q,v,t in zip(c['q'],c['v'],c['times_s']):
            n._outstanding.points.append(JointTrajectoryPoint(positions=q,velocities=v,time_from_start=rospy.Duration(t)))
        n._outstanding_diag={'geometric_restore':c['geometric_restore']}
        n._latest_measured_stamp_ns=n._latest_actuator_stamp_ns=now
        n._latest_measured_q_by_name=dict(zip(c['joint_names'],e['measured_q']))
        n._latest_measured_velocity_by_name=dict(zip(c['joint_names'],e['measured_velocity']))
        n._latest_actuator_velocity=dict(n._latest_measured_velocity_by_name)
        n.repair_acceleration_limits=c['acceleration'];n.repair_brake_dt_s=.05;n.repair_brake_max_steps=20
        n.initial_shell_execution_margin_m=.01;n.tracking_feedback_gain=2.5
        return n,e,now

    def check(self,n,e,now):
        detail={}
        result=n._shell_commit_valid_locked(json.dumps(e),int(e['trajectory_stamp_ns']),rospy.Time(0,now),detail)
        return result,detail

    def test_exact_recorded_native_evidence_can_pass_when_fresh(self):
        for f in self.fixtures:
            with self.subTest(audit=f['source_audit']):
                n,e,now=self.gate(f)
                ok,d=self.check(n,e,now)
                self.assertTrue(ok,d)
                self.assertLess(d['motion_bound_m'],e['certified_residual_m'])

    def test_original_50ms_boundary_is_unchanged(self):
        for f in self.fixtures:
            with self.subTest(audit=f['source_audit']):
                n,e,now=self.gate(f,50000000);self.assertTrue(self.check(n,e,now)[0])
                n,e,now=self.gate(f,50000001);ok,d=self.check(n,e,now)
                self.assertFalse(ok);self.assertEqual(d['failure_kind'],'certificate_state_stale')

    def test_live_and_actuator_state_must_be_fresh(self):
        for field,reason in [('_latest_measured_stamp_ns','live_state_stale'),('_latest_actuator_stamp_ns','missing_actuator_state')]:
            n,e,now=self.gate(self.fixtures[0],60000000 if field=='_latest_measured_stamp_ns' else 40000000)
            setattr(n,field,now-50000001)
            ok,d=self.check(n,e,now);self.assertFalse(ok);self.assertEqual(d['failure_kind'],reason)

    def test_changed_measured_or_applied_motion_cannot_use_old_proof(self):
        for field in ['_latest_measured_q_by_name','_latest_actuator_velocity']:
            n,e,now=self.gate(self.fixtures[0]);getattr(n,field)['joint1']+=.001
            ok,d=self.check(n,e,now);self.assertFalse(ok)
            self.assertEqual(d['failure_kind'],'geometric_certificate_state_changed')

    def test_old_map_and_foreign_restoration_points_still_fail(self):
        n,e,now=self.gate(self.fixtures[0]);e['map_snapshot_stamp_ns']=str(now-50000001)
        ok,d=self.check(n,e,now);self.assertFalse(ok);self.assertEqual(d['failure_kind'],'map_snapshot_stale')
        n,e,now=self.gate(self.fixtures[0]);e['restoration_points']=[[0.,0.,0.]]
        ok,d=self.check(n,e,now);self.assertFalse(ok);self.assertEqual(d['reason'],'geometric_restore_certificate_mismatch')

    def test_changed_audited_motion_does_not_match_certificate(self):
        n,e,now=self.gate(self.fixtures[0]);n._outstanding.points[0].positions=list(n._outstanding.points[0].positions)
        n._outstanding.points[0].positions[0]+=.000001
        ok,d=self.check(n,e,now);self.assertFalse(ok);self.assertEqual(d['reason'],'geometric_certificate_motion_mismatch')


if __name__=='__main__':unittest.main()
