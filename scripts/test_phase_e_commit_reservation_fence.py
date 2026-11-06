#!/usr/bin/env python3
"""Production commit callbacks with deterministic ROS transport and time."""
import copy
import importlib.util
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/egocentric_arm_planner/scripts'))
import optimized_trajectory_continuity_node as production
sys.path.insert(0,str(ROOT/'src/care_visibility_cdf/scripts'))
from candidate_replacement_runtime import StableGenerationSeed

class Pub:
    def __init__(self,*a,**kw):self.messages=[]
    def publish(self,m):self.messages.append(copy.deepcopy(m))

class FenceTests(unittest.TestCase):
    def setUp(self):
        self.now=18.8;self.wall=100.
        patches=[patch.object(rospy,'get_param',side_effect=lambda k,d=None:d),
            patch.object(rospy,'Publisher',Pub),patch.object(rospy,'Subscriber'),
            patch.object(rospy,'Timer'),patch.object(rospy,'logwarn'),
            patch.object(rospy.Time,'now',side_effect=lambda:rospy.Time(0,round(self.now*1e9))),
            patch.object(production.time,'monotonic',side_effect=lambda:self.wall)]
        for p in patches:p.start();self.addCleanup(p.stop)
        self.n=production.OptimizedTrajectoryContinuityNode()
        self.names=['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']
        self.measure()
    def measure(self):
        state=JointState(name=self.names,position=[0.]*7,velocity=[0.]*7)
        state.header.stamp=rospy.Time.from_sec(self.now)
        self.n._joint_state_cb(state)
        self.n._actuator_state_cb(state)
    def raw(self,stamp):
        m=JointTrajectory();m.header.stamp=rospy.Time.from_sec(stamp);m.joint_names=self.names
        m.points=[JointTrajectoryPoint(positions=[0.]*7,velocities=[0.]*7,time_from_start=rospy.Duration(0)),
                  JointTrajectoryPoint(positions=[0.]*7,velocities=[0.]*7,time_from_start=rospy.Duration(1))]
        return m
    def grant(self,rid=6,epoch=2,deadline=20.810):
        s=f'request_id={rid} granted=1 mode_epoch={epoch} query_stamp_ns=18776000000 deadline_ns={round(deadline*1e9)}'
        self.n._replacement_grant_cb(String(data=s));return s
    def safe(self,stamp):
        self.n._global_summary_cb(String(data=f'trajectory_source=predicted trajectory_stamp_ns={stamp} has_violation=0'))
    def finish(self,rid=6):self.n._replacement_finish_cb(String(data=f'action=finish request_id={rid} new_token=none'))
    def test_original_publish_reserve_delayed_safe_order_cannot_commit(self):
        self.now=18.807;self.n._raw_cb(self.raw(self.now))
        audit=self.n.verification_pub.messages[-1].header.stamp.to_nsec()
        self.now=18.810;self.grant()
        self.assertEqual(len(self.n.replacement_ack_pub.messages),1)
        self.now=18.940;self.safe(audit)
        self.assertFalse(self.n.committed_pub.messages)
        self.assertEqual(self.n._commit_count,0)
        self.finish();self.safe(audit)
        self.n._raw_cb(self.raw(18.807))
        self.assertFalse(self.n.committed_pub.messages)
        # A fresh candidate still passes the original VBC and start gates.
        self.now=19.1;self.measure()
        self.n._raw_cb(self.raw(self.now));fresh=self.n.verification_pub.messages[-1].header.stamp.to_nsec()
        self.safe(fresh);self.assertEqual(len(self.n.committed_pub.messages),1)
    def test_pre_ack_commit_is_reported_and_tracker_receipt_required(self):
        self.n._raw_cb(self.raw(self.now));audit=self.n.verification_pub.messages[-1].header.stamp.to_nsec()
        self.now=18.805;self.safe(audit);self.assertEqual(len(self.n.committed_pub.messages),1)
        self.now=18.810;self.grant()
        ack=self.n.replacement_ack_pub.messages[-1].data
        g=StableGenerationSeed();g.observe_commit_ack(ack,self.now)
        fields=dict(w.split('=',1) for w in ack.split())
        self.assertEqual(g.confirmed(fields)[1],'waiting_for_tracker_commit_receipt')
        g.observe_tracker('TRACKER active=1 execution_stamp_ns=18805000000 source=active_trajectory',self.now)
        self.assertTrue(g.confirmed(fields)[0])
        self.assertFalse(g.ready(self.now,self.now)[0])
    def test_gcdf_inflight_and_pending_are_cancelled(self):
        self.n.final_gcdf_enabled=True;self.n._raw_cb(self.raw(self.now))
        self.n._raw_cb(self.raw(self.now+.001))
        self.assertIsNotNone(self.n._gcdf_outstanding);self.assertIsNotNone(self.n._pending_raw)
        self.grant();self.assertIsNone(self.n._gcdf_outstanding);self.assertIsNone(self.n._pending_raw)
        self.n._raw_cb(self.raw(18.9));self.assertIsNone(self.n._pending_raw)
    def test_v2_uses_planner_wall_lease_even_if_ros_runs_ahead(self):
        text=('request_id=6 granted=1 mode_epoch=2 query_stamp_ns=18776000000 '
              'deadline_ns=20810000000 lease_timing=wall_v2 '
              'wait_budget_s=3 generation_budget_s=6 deadline_wall_ns=1010000000000')
        with patch.object(production.time,'time',return_value=1000.):
            self.n._replacement_grant_cb(String(data=text))
            deadline=self.n._replacement_fence['wall_deadline']
            self.now=100.;self.wall=104.
            self.assertTrue(self.n._replacement_active_locked(rospy.Time.now()))
            self.n._replacement_grant_cb(String(data=text))
            self.assertEqual(self.n._replacement_fence['wall_deadline'],deadline)
            self.wall=110.001
            self.assertFalse(self.n._replacement_active_locked(rospy.Time.now()))

    def test_duplicates_do_not_extend_deadline_and_wrong_finish_does_not_release(self):
        text=self.grant();wall=self.n._replacement_fence['wall_deadline']
        self.wall+=1;self.n._replacement_grant_cb(String(data=text))
        self.assertEqual(self.n._replacement_fence['wall_deadline'],wall)
        self.finish(5);self.assertIsNotNone(self.n._replacement_fence)
        self.finish();self.assertIsNone(self.n._replacement_fence)
        self.n._replacement_grant_cb(String(data=text));self.assertIsNone(self.n._replacement_fence)
    def test_finish_before_grant_cannot_rearm(self):
        self.finish();self.grant();self.assertIsNone(self.n._replacement_fence)
        self.assertFalse(self.n.replacement_ack_pub.messages)
    def test_new_epoch_supersedes_old_and_late_finish_cannot_release_it(self):
        self.grant();self.grant(rid=7,epoch=3)
        self.finish(6)
        self.assertEqual(self.n._replacement_fence['request_id'],7)
        self.assertEqual(self.n._replacement_fence['mode_epoch'],3)
        self.grant(rid=6,epoch=2)
        self.assertEqual(self.n._replacement_fence['request_id'],7)
    def test_expiry_and_paused_ros_release_without_replaying_old_work(self):
        for wall in [False,True]:
            with self.subTest(wall=wall):
                self.n._replacement_closed_id=0;self.now=18.81;self.wall=100.;self.grant()
                if wall:self.wall=103.001
                else:self.now=20.811
                self.assertFalse(self.n._replacement_active_locked(rospy.Time.now()))
                self.assertIsNone(self.n._outstanding)
    def test_ack_identity_and_duplicate_receipt_do_not_shift_stability_window(self):
        self.grant();ack=self.n.replacement_ack_pub.messages[-1].data
        fields=dict(w.split('=',1) for w in ack.split());g=StableGenerationSeed()
        self.assertFalse(g.confirmed(fields)[0]);g.observe_commit_ack(ack,18.81)
        self.assertTrue(g.confirmed(fields)[0]);g.observe_commit_ack(ack,19.)
        self.assertEqual(g.commit_ack['received'],18.81)
        for key in ['request_id','mode_epoch','query_stamp_ns','deadline_ns']:
            wrong=dict(fields);wrong[key]=str(int(wrong[key])+1)
            self.assertFalse(g.confirmed(wrong)[0])
    def test_ack_cannot_overtake_actual_committed_publication(self):
        entered=threading.Event();release=threading.Event();errors=[]
        original=self.n.committed_pub.publish
        def publish(m):entered.set();release.wait(2);original(m)
        self.n.committed_pub.publish=publish
        self.n._raw_cb(self.raw(self.now));audit=self.n.verification_pub.messages[-1].header.stamp.to_nsec()
        self.now=18.805
        def run(f):
            try:f()
            except BaseException as e:errors.append(e)
        first=threading.Thread(target=lambda:run(lambda:self.safe(audit)));first.start()
        self.assertTrue(entered.wait(1))
        second=threading.Thread(target=lambda:run(self.grant));second.start()
        self.assertFalse(self.n.replacement_ack_pub.messages)
        release.set();first.join(2);second.join(2)
        self.assertFalse(first.is_alive() or second.is_alive());self.assertFalse(errors)
        self.assertEqual(len(self.n.committed_pub.messages),1)
        self.assertIn('last_execution_stamp_ns=18805000000',self.n.replacement_ack_pub.messages[-1].data)

if __name__=='__main__':unittest.main()
