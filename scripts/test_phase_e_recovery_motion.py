#!/usr/bin/env python3
"""Recorded direction regression and production motion/commit fault controls."""
import copy
import json
import math
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/egocentric_arm_planner/scripts'))
from recovery_motion import ramp_step,interpolate_motion
from initial_shell_recovery import commit_check
import test_phase_e_initial_shell_commit as shell
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
import rospy


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.base=shell.ShellCommitTests(methodName='runTest');self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        self.n=self.base.n;self.n.repair_motion_consistency_enabled=True
        self.applied()

    def applied(self):
        self.n._actuator_state_cb(JointState(header=Header(stamp=rospy.Time.from_sec(self.base.b.now)),
            name=self.base.b.raw.joint_names,velocity=[0.]*7))

    def test_recorded_outward_command_keeps_direction_from_rest(self):
        fixture=json.loads((ROOT/'outputs/phase_e_case021_rolling_analysis_20261013/native_controls.json').read_text())
        d=fixture['direction_control'] if 'direction_control' in fixture else fixture['direction']
        # Production native guard gradient, exact recorded command.
        gradient=fixture['native']['states'][0]['gradient'] if 'native' in fixture else fixture['states'][0]['gradient']
        a=[3.,3.,4.,4.,6.,6.,6.];u=d['raw_velocity']
        output=ramp_step([0.]*7,u,a,.01)
        self.assertGreater(sum(x*y for x,y in zip(gradient,output)),0.)
        self.assertTrue(all(abs(v)<=limit*.01+1e-12 for v,limit in zip(output,a)))

    def test_constructed_position_and_velocity_share_one_ramp(self):
        path=self.n._repair_prefix_with_braking_tail(self.base.b.raw,.25)
        self.assertIsNotNone(path)
        for a,b in zip(path.points,path.points[1:]):
            dt=(b.time_from_start-a.time_from_start).to_sec()
            for j,limit in enumerate(self.n.repair_acceleration_limits):
                self.assertAlmostEqual(b.positions[j]-a.positions[j],.5*dt*(a.velocities[j]+b.velocities[j]),places=10)
                self.assertLessEqual(abs(b.velocities[j]-a.velocities[j]),limit*dt+1e-9)
        self.assertEqual(path.points[-1].velocities,[0.]*7)

    def test_missing_actuator_or_stale_measurement_cannot_build_motion(self):
        self.n._latest_actuator_stamp_ns=0
        self.assertIsNone(self.n._repair_prefix_with_braking_tail(self.base.b.raw,.25))
        self.applied();self.base.b.advance(.051)
        self.assertIsNone(self.n._repair_prefix_with_braking_tail(self.base.b.raw,.25))

    def test_nonzero_incoming_velocity_is_preserved(self):
        self.base.measure(velocity=[.1]*7)
        self.n._latest_actuator_velocity=dict(zip(self.base.b.raw.joint_names,[.1]*7))
        path=self.n._repair_prefix_with_braking_tail(self.base.b.raw,.25)
        self.assertIsNotNone(path);self.assertEqual(path.points[0].velocities,[.1]*7)

    def test_invalid_applied_velocity_is_not_assumed_zero(self):
        self.n._latest_actuator_velocity={}
        self.assertIsNone(self.n._repair_prefix_with_braking_tail(self.base.b.raw,.25))

    def test_normal_full_horizon_certifies_feasible_stop_before_commit(self):
        self.n._repair_active=False
        self.base.b.start()
        path=self.n.final_gcdf_query_pub.messages[-1]
        self.assertEqual(self.n._gcdf_outstanding_view,'full_horizon')
        self.assertEqual(self.n._gcdf_outstanding_diag['normal_motion_consistent'],1)
        self.assertGreater(path.points[-1].time_from_start.to_sec(),self.n._duration(self.base.b.raw))
        for a,b in zip(path.points,path.points[1:]):
            dt=(b.time_from_start-a.time_from_start).to_sec()
            for j,limit in enumerate(self.n.repair_acceleration_limits):
                self.assertAlmostEqual(b.positions[j]-a.positions[j],.5*dt*(a.velocities[j]+b.velocities[j]),places=10)
                self.assertLessEqual(abs(b.velocities[j]-a.velocities[j]),limit*dt+1e-9)
        self.assertEqual(path.points[-1].velocities,[0.]*7)
        self.assertFalse(self.n.verification_pub.messages or self.n.committed_pub.messages)
        self.base.b.gcdf(False)
        self.assertFalse(self.n.verification_pub.messages or self.n.committed_pub.messages)

    def test_normal_missing_actuator_state_cannot_reach_audit_or_commit(self):
        self.n._repair_active=False
        self.n._latest_actuator_stamp_ns=0
        self.base.b.start()
        self.assertFalse(self.n.final_gcdf_query_pub.messages or self.n.committed_pub.messages)
        self.assertEqual(self.n._last_prefix_build_reason,'actuator_state_stale')

    def test_normal_ordinary_certification_commits_the_exact_rebuilt_curve(self):
        self.n._repair_active=False
        self.base.b.start()
        audited=self.n.final_gcdf_query_pub.messages[-1]
        self.base.b.gcdf()
        self.assertFalse(self.n.committed_pub.messages)
        self.assertEqual(self.n.verification_pub.messages[-1].points,audited.points)
        context=json.loads(self.n.shell_context_pub.messages[-1].data)
        from std_msgs.msg import String
        self.n._global_summary_cb(String(data=(
            'trajectory_source=predicted trajectory_stamp_ns={} has_violation=0 '
            'shell_recovery_count=0 shell_recovery_evidence=none '
            'context_request_session={} context_request_sequence={}').format(
                self.n.verification_pub.messages[-1].header.stamp.to_nsec(),
                context['request_session'],context['request_sequence'])))
        self.assertEqual(len(self.n.committed_pub.messages),1)
        self.assertEqual(self.n.committed_pub.messages[-1].points,audited.points)

    def test_execution_suffix_keeps_the_same_curve(self):
        import vbc_execution_reference_gate_node as gate
        msg=JointTrajectory(joint_names=self.base.b.raw.joint_names)
        msg.points=[JointTrajectoryPoint(positions=[0.]*7,velocities=[.2]*7,time_from_start=rospy.Duration(0.)),
            JointTrajectoryPoint(positions=[.3]*7,velocities=[.4]*7,time_from_start=rospy.Duration(1.))]
        for cls in [type(self.n),gate.VBCExecutionReferenceGate]:
            piece=cls._interpolate_point(msg.points[0],msg.points[1],.5)
            self.assertAlmostEqual(piece.positions[0],.125)
            self.assertAlmostEqual(piece.velocities[0],.3)
            self.assertAlmostEqual(piece.accelerations[0],.2)
            remaining=interpolate_motion(piece.positions,msg.points[1].positions,
                piece.velocities,msg.points[1].velocities,.5,.5)
            self.assertIsNotNone(remaining)
            self.assertAlmostEqual(remaining[0][0],.20625)


class CertificateTests(shell.ShellCommitTests):
    def test_stale_measured_or_map_snapshot_rejected(self):
        self.dispatch();self.base_clock_before=self.n._outstanding_sent
        self.b.advance(.051);self.measure();self.verdict()
        self.assertFalse(self.n.committed_pub.messages)
        self.assertIsNotNone(self.n._outstanding)
        self.assertEqual(self.n._outstanding_sent,self.base_clock_before)
        self.verdict();self.assertFalse(self.n.committed_pub.messages)
        self.assertIn('measured_snapshot_stale',self.n.verification_event_pub.messages[-1].data)
        self.assertIn('safety_gate=state_evidence',self.n.verification_event_pub.messages[-1].data)

    # Existing fault controls exercise the new certificate as well as v1.
    def dispatch(self):
        self.n.repair_motion_consistency_enabled=True
        self.n._latest_actuator_stamp_ns=self.n._latest_measured_stamp_ns
        self.n._latest_actuator_velocity=dict(self.n._latest_measured_velocity_by_name)
        super().dispatch()
        velocities=[self.n._latest_measured_velocity_by_name[name] for name in self.candidate.joint_names]
        self.e.update(schema=2,certified_residual_m=.004,
            motion_horizon_s=self.candidate.points[-1].time_from_start.to_sec(),
            measured_velocity=velocities,initial_q=list(self.candidate.points[0].positions),
            initial_velocity=list(self.candidate.points[0].velocities))
        self.n._latest_actuator_stamp_ns=self.n._latest_measured_stamp_ns
        self.n._latest_actuator_velocity=dict(zip(self.candidate.joint_names,velocities))

    def test_snapshot_change_and_braking_use_clearance_budget(self):
        self.dispatch();context=self.n.shell_context_pub.messages[-1].data
        self.measure(shift=.002);self.verdict()
        self.assertIsNotNone(self.n._outstanding)
        self.assertTrue(self.n._outstanding_diag['shell_refresh_requested'])
        self.assertEqual(context,self.n.shell_context_pub.messages[-1].data)
        self.verdict();self.assertFalse(self.n.committed_pub.messages)
        self.assertIn('geometric_certificate_state_changed',self.n.verification_event_pub.messages[-1].data)

    def test_large_nominal_brake_does_not_consume_proved_geometry_again(self):
        self.measure(velocity=[.1]*7);self.dispatch()
        self.n._latest_actuator_velocity=dict(zip(self.candidate.joint_names,[.1]*7))
        self.e.update(measured_velocity=[.1]*7,initial_velocity=[.1]*7)
        self.verdict();self.assertEqual(len(self.n.committed_pub.messages),1)

    def test_velocity_certificate_cannot_describe_another_motion(self):
        self.dispatch();self.e['initial_velocity'][0]+=.001;self.verdict()
        self.assertFalse(self.n.committed_pub.messages)
        self.assertIn('geometric_certificate_motion_mismatch',self.n.verification_event_pub.messages[-1].data)

    def test_missing_actuator_certificate_rejected(self):
        self.dispatch();self.n._latest_actuator_stamp_ns=0;self.verdict()
        self.assertFalse(self.n.committed_pub.messages)

    def test_nonfinite_reserve_rejected(self):
        self.dispatch();self.e['certified_residual_m']=math.nan;self.verdict()
        self.assertFalse(self.n.committed_pub.messages)


if __name__=='__main__':unittest.main()
