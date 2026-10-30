"""Task-return ranking guards and original finite branch behaviour."""
from pathlib import Path
import copy
import os
import sys
import threading
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/care_visibility_cdf/scripts'))
from observation_task_rejoin import task_rejoin_context,candidate_cost,improves_candidate
import per_sensor_visibility_runtime as mod

class RejoinTests(unittest.TestCase):
    def test_finite_context_and_limits(self):
        for q,v in [(np.full(7,np.nan),np.ones(7)),(np.zeros(7),np.zeros(7))]:
            with self.assertRaises(ValueError):task_rejoin_context(q,np.ones(7),v,np.ones(7))

    def test_cost_includes_approach_return_and_pose(self):
        c=task_rejoin_context(np.zeros(7),np.ones(7),np.ones(7),np.ones(7))
        a=np.eye(4);b=np.eye(4);b[0,3]=.1
        r=candidate_cost(np.full(7,.25),c,a,b)
        self.assertAlmostEqual(r['approach_s'],1.)
        self.assertAlmostEqual(r['rejoin_s'],2.*np.sqrt(.75))
        self.assertAlmostEqual(r['task_pose_error'],5.)

    def test_total_cost_can_outweigh_one_leg_or_endpoint_pose(self):
        old=dict(approach_s=1.,rejoin_s=2.,total_s=3.,task_pose_error=5.)
        good=dict(approach_s=.5,rejoin_s=1.,total_s=1.5,task_pose_error=4.)
        self.assertTrue(improves_candidate(good,old,.1))
        for key,value in [('approach_s',1.1),('rejoin_s',2.1),('task_pose_error',5.1)]:
            changed=dict(good);changed[key]=value;self.assertTrue(improves_candidate(changed,old,0.))
        for key in good:
            for value in [float('nan'),float('inf'),-1.]:
                changed=dict(good);changed[key]=value
                self.assertFalse(improves_candidate(changed,old,0.))
        self.assertFalse(improves_candidate(good,old,2.))

    def test_first_candidate_survives_alternative_failure(self):
        r=object.__new__(mod.PerSensorVisibilityRuntime)
        r.pose_transform_reuse_enabled=False
        r.device=torch.device('cpu');r.max_branch_attempts=4;r.min_conservative_g=0.
        r.task_ee_chain=[];r.sensor_masks_np=np.ones((8,7))
        r.rank_sensors=lambda *a:(torch.zeros(8),torch.arange(8))
        calls=[]
        def branch(points,seed,sid):
            calls.append(sid)
            if sid:raise RuntimeError('controlled alternative failure')
            return dict(sensor_id=sid,sensor_frame='controlled',q_candidate=np.zeros(7).tolist())
        r._optimize_branch=branch;r._candidate_geometry=lambda *a,**kw:dict(accepted=True)
        # Configuration is trace-only; these fields do not enter branch policy.
        for name,value in dict(projection_iters=10,projection_damping=.5,projection_epsilon_f=.03,
            projection_max_step_norm=.25,root_refine_iters=12,root_tolerance_f=.002,
            branch_ascent_steps=1,branch_step_size=.05,branch_max_step_norm=.25,
            branch_fallback_ascent_steps=8,require_primitive_los=True,checkpoint_path='controlled',
            checkpoint={'step':0,'args':{}},ray_args=type('Ray',(),{'self_filter_padding':0.})()).items():setattr(r,name,value)
        with patch.object(mod,'fk_transform',return_value=np.eye(4)):
            result=r.generate([[.2,0,.5]],np.zeros(7),task_rejoin=task_rejoin_context(
                np.zeros(7),np.ones(7),np.ones(7),np.ones(7)))
        self.assertEqual(result['selected_sensor_id'],0)
        self.assertEqual(calls,[0,1])
        self.assertIn('alternative_error',result['task_rejoin'])

    def test_node_uses_original_task_and_actual_planner_limits(self):
        root=Path(__file__).resolve().parents[1]
        for path in [root/'devel/lib/python3/dist-packages',Path('/opt/ros/noetic/lib/python3/dist-packages'),Path('/usr/lib/python3/dist-packages')]:
            sys.path.append(str(path))
        os.environ['ROS_PACKAGE_PATH']=str(root/'src')+':/opt/ros/noetic/share'
        from vbc_deadline_waypoint_rolling_impl import RollingVbcDeadlineWaypointNode
        from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
        import rospy
        node=object.__new__(RollingVbcDeadlineWaypointNode);node._lock=threading.RLock()
        node.task_rejoin_enabled=True;node.task_rejoin_limits_namespace='/planner/mpc'
        node.trajectory_topic='/care_planner/task_trajectory'
        names=['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']
        node._trajectory=JointTrajectory(joint_names=list(reversed(names)))
        node._trajectory.points=[JointTrajectoryPoint(positions=list(reversed(list(range(7)))))]
        node._predicted_trajectory=JointTrajectory(joint_names=names,
            points=[JointTrajectoryPoint(positions=[99.]*7)])
        limits={'/planner/mpc/joint_velocity_limits':dict(zip(names,[2,2,2,2,2.5,2.5,2.5])),
                '/planner/mpc/joint_acceleration_limits':dict(zip(names,[3,3,4,4,6,6,6]))}
        with patch.object(rospy,'get_param',side_effect=lambda key:limits[key]):
            context=node._task_rejoin_context(np.zeros(7))
        self.assertEqual(context['q_task'],list(range(7)))
        self.assertEqual(context['acceleration'],[3,3,4,4,6,6,6])
        self.assertEqual(context['source']['topic'],'/care_planner/task_trajectory')
        with patch.object(rospy,'get_param',side_effect=KeyError('missing limits')):
            self.assertIsNone(node._task_rejoin_context(np.zeros(7)))

if __name__=='__main__':unittest.main()
