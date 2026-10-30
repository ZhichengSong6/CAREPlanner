#!/usr/bin/env python3
"""Production repair dispatch with delayed admission and fresh measured rebase."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import test_phase_e_candidate_admission as admission
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
import rospy


class RepairDispatchTests(unittest.TestCase):
    def receiver(self):
        base=admission.AdmissionTests(methodName='runTest');base.setUp();self.addCleanup(base.doCleanups)
        fixture=json.loads((ROOT/'outputs/phase_e_executable_view_fix_20261010/math/qp_fixture.json').read_text())
        raw=JointTrajectory();raw.header.stamp=rospy.Time.from_sec(base.now);raw.joint_names=base.raw.joint_names
        for k,q in enumerate(fixture['q']):
            p=JointTrajectoryPoint(positions=q,velocities=fixture['u'][k] if k<20 else [0.]*7,
                                   time_from_start=rospy.Duration(k*.05));raw.points.append(p)
        base.n._joint_state_cb(JointState(name=raw.joint_names,position=fixture['q'][0]))
        return base,raw

    def dispatch(self,base,raw,enabled):
        n=base.n;n.repair_projection_consistent_dispatch_enabled=enabled
        n._pending_raw=copy.deepcopy(raw);n._pending_raw_received=rospy.Time.from_sec(base.now-.08)
        n._pending_repair=True;n._repair_active=True
        return n._dispatch_pending_locked(rospy.Time.from_sec(base.now))

    def test_delayed_new_proposal_keeps_planned_prefix(self):
        base,raw=self.receiver();n=base.n
        expected=n._repair_prefix_with_braking_tail(raw,.25)
        candidate,stage=self.dispatch(base,raw,True)
        self.assertEqual(stage,'gcdf');self.assertFalse(n.committed_pub.messages)
        self.assertEqual(n._gcdf_outstanding_diag['dispatch_suffix_phase_s'],0.)
        self.assertAlmostEqual(n._gcdf_outstanding_diag['raw_candidate_age_s'],.08)
        for a,b in zip(expected.points,candidate.points):
            self.assertEqual(a.positions,b.positions);self.assertEqual(a.velocities,b.velocities)
        self.assertEqual(len(expected.points),len(candidate.points))

    def test_measured_rebase_still_requires_certification(self):
        base,raw=self.receiver();n=base.n
        measured=[v+.001 for v in raw.points[0].positions]
        n._joint_state_cb(JointState(name=raw.joint_names,position=measured))
        candidate,stage=self.dispatch(base,raw,True)
        self.assertEqual(stage,'gcdf');self.assertEqual(candidate.points[0].positions,measured)
        self.assertTrue(n._gcdf_outstanding_diag['handoff_rebase_applied'])
        self.assertFalse(n.committed_pub.messages)
        from care_collision_cdf.msg import CollisionCDFConstraintBatch
        from std_msgs.msg import String
        batch=CollisionCDFConstraintBatch();batch.header.stamp=candidate.header.stamp
        n._final_gcdf_batch_cb(batch);self.assertFalse(n.committed_pub.messages)
        n._global_summary_cb(String(data='trajectory_source=predicted trajectory_stamp_ns={} has_violation=1 vbc_evidence=fixed'.format(candidate.header.stamp.to_nsec())))
        self.assertFalse(n.committed_pub.messages)

    def test_legacy_dispatch_switch_retains_age_suffix(self):
        base,raw=self.receiver();self.dispatch(base,raw,False)
        self.assertAlmostEqual(base.n._gcdf_outstanding_diag['dispatch_suffix_phase_s'],.08)


if __name__=='__main__':unittest.main()
