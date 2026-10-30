#!/usr/bin/env python3
"""NORMAL goal motion and safety handoff through production callbacks."""
import copy
import math
import unittest
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import rospy
import test_phase_e_recovery_motion as motion
from recovery_motion import has_nodal_motion


class NormalGoalStopTests(motion.MotionTests):
    def test_repair_nodal_prefix_and_actual_braking_share_one_position(self):
        raw=self.ramp()
        path=self.n._repair_prefix_with_braking_tail(raw,.5,True)
        self.assertEqual(self.n._last_repair_nodal_motion,1)
        self.assertEqual(self.n._last_normal_nodal_motion,0)
        at_prefix=min(path.points,key=lambda p:abs(p.time_from_start.to_sec()-.5))
        self.assertLessEqual(abs(at_prefix.time_from_start.to_sec()-.5),2e-9)
        for a,b in zip(at_prefix.positions,raw.points[1].positions):self.assertAlmostEqual(a,b,places=9)
        q=list(at_prefix.positions);v=list(at_prefix.velocities)
        while max(abs(x) for x in v)>1e-6:
            following=[max(0.,x-a*.05) for x,a in zip(v,self.n.repair_acceleration_limits)]
            q=[x+.025*(a+b) for x,a,b in zip(q,v,following)];v=following
        for a,b in zip(q,path.points[-1].positions):self.assertAlmostEqual(a,b,places=9)
        self.assertEqual(path.points[-1].velocities,[0.]*7)

    def ramp(self, incoming=0.):
        # A solved constant-acceleration proposal: go from incoming velocity
        # to .2 then back to zero, with one shared integrated representation.
        raw=copy.deepcopy(self.base.b.raw)
        q=list(self.n._latest_measured_q_by_name[x] for x in raw.joint_names)
        raw.points=[JointTrajectoryPoint(positions=q,velocities=[incoming]*7,time_from_start=rospy.Duration(0.))]
        old=[incoming]*7
        for k,v in enumerate([.2, .0],1):
            new=[v]*7;q=[x+.25*(a+b) for x,a,b in zip(q,old,new)]
            raw.points.append(JointTrajectoryPoint(positions=list(q),velocities=new,time_from_start=rospy.Duration(.5*k)))
            old=new
        return raw

    def test_preserves_solved_ramp_goal_and_continuous_hold(self):
        raw=self.ramp()
        path=self.n._repair_prefix_with_braking_tail(raw,1.,True,normal_velocity_reference=True)
        self.assertEqual(self.n._last_normal_nodal_motion,1)
        self.assertEqual(path.points[-1].velocities,[0.]*7)
        for x,y in zip(raw.points[-1].positions,path.points[-1].positions):self.assertAlmostEqual(x,y,places=9)
        self.assertGreaterEqual(path.points[-1].time_from_start.to_sec()-path.points[-2].time_from_start.to_sec(),.1-1e-9)
        self.assertAlmostEqual(self.n._last_repair_brake_duration_s,0.)

    def test_new_normal_proposal_still_needs_both_safety_verdicts(self):
        self.n._repair_active=False;self.base.b.raw=self.ramp()
        self.base.b.start()
        self.assertEqual(self.n._gcdf_outstanding_diag['normal_nodal_motion'],1)
        audited=self.n.final_gcdf_query_pub.messages[-1]
        self.assertFalse(self.n.committed_pub.messages)
        self.base.b.gcdf()
        self.assertFalse(self.n.committed_pub.messages)
        self.base.b.vbc(False)
        self.assertFalse(self.n.committed_pub.messages)
        self.assertEqual(audited.points[-1].velocities,[0.]*7)

    def test_false_terminal_zero_keeps_original_full_braking(self):
        raw=self.ramp()
        raw.points[-1].velocities=[2.]*7
        self.assertFalse(has_nodal_motion(raw.points,self.n.repair_acceleration_limits))
        old=self.n._repair_prefix_with_braking_tail(raw,1.,True)
        rebuilt=self.n._repair_prefix_with_braking_tail(raw,1.,True,normal_velocity_reference=True)
        self.assertEqual(self.n._last_normal_nodal_motion,0)
        self.assertEqual(rebuilt.points,old.points)

    def test_fresh_replanning_uses_new_incoming_speed_without_reset(self):
        raw=self.ramp()
        self.base.measure(velocity=[.1]*7)
        self.n._latest_actuator_velocity=dict(zip(raw.joint_names,[.1]*7))
        path=self.n._repair_prefix_with_braking_tail(raw,1.,True,normal_velocity_reference=True)
        self.assertEqual(path.points[0].velocities,[.1]*7)
        for a,b in zip(path.points,path.points[1:]):
            dt=(b.time_from_start-a.time_from_start).to_sec()
            for j,limit in enumerate(self.n.repair_acceleration_limits):
                self.assertAlmostEqual(b.positions[j]-a.positions[j],.5*dt*(a.velocities[j]+b.velocities[j]),places=9)
                self.assertLessEqual(abs(b.velocities[j]-a.velocities[j]),limit*dt+1e-9)

    def test_stopped_goal_stays_stopped(self):
        raw=self.ramp();q=list(raw.points[0].positions)
        for p in raw.points:p.positions=list(q);p.velocities=[0.]*7
        path=self.n._repair_prefix_with_braking_tail(raw,1.,True,normal_velocity_reference=True)
        self.assertTrue(all(p.positions==q and p.velocities==[0.]*7 for p in path.points))

    def test_invalid_nodal_payload_never_skips_reconstruction(self):
        for kind in ['position','dimension','nan','acceleration']:
            raw=self.ramp()
            if kind=='position':raw.points[1].positions[0]+=.1
            if kind=='dimension':raw.points[1].velocities=[]
            if kind=='nan':raw.points[1].velocities[0]=math.nan
            if kind=='acceleration':raw.points[1].velocities[0]=10.
            self.assertFalse(has_nodal_motion(raw.points,self.n.repair_acceleration_limits),kind)


if __name__=='__main__':unittest.main()
