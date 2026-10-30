"""Production ownership/pose matching without ROS or robot startup."""
from pathlib import Path
import copy
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/care_visibility_cdf/scripts'),
    str(ROOT/'devel/lib/python3/dist-packages'),'/opt/ros/noetic/lib/python3/dist-packages',
    '/usr/lib/python3/dist-packages']
from vbc_blocker_aware_acquisition_impl import BlockerAwareVisibilityAcquisitionWaypointNode as Node
from observation_identity import observation_region_token,observation_token
from bounded_visibility_recovery import region_key

class OwnerTests(unittest.TestCase):
    def node(self):
        n=object.__new__(Node)
        n._obligations=[];n._dependency_pins={};n._trace_published_target=None
        n.region_match_distance_m=.12;n._obligation_match_qvis_min_f=0.
        for name in ['check','accept','reject','error']:setattr(n,'_qvis_match_'+name+'_count',0)
        n._cell_key=lambda p:tuple(np.rint(np.asarray(p)/.05).astype(int))
        n._dependency_attempts={('existing',):2}
        n._per_sensor_runtime=NS(branch_score_numpy=lambda *a:-.18647,
            _candidate_geometry=lambda *a:dict(accepted=False))
        return n

    def ob(self,n,oid,points):
        pts=np.asarray(points,dtype=float)
        return dict(id=oid,points=pts,keys=tuple(sorted({n._cell_key(p) for p in pts})),
            centroid=pts.mean(axis=0),q_vis=np.zeros(7),q_vis_joint_mask=np.ones(7),
            per_sensor_hybrid_used=True,per_sensor_selected_sensor_id=0)

    def region(self,n,points):
        ob=self.ob(n,0,points)
        return {k:ob[k] for k in ['points','keys','centroid']}

    def test_owned_points_reuse_owner_despite_bad_old_pose_and_preserve_budget(self):
        n=self.node();ob=self.ob(n,7,[[-.15,.05,.4]]);n._obligations=[ob]
        before=copy.deepcopy(ob);budget=copy.deepcopy(n._dependency_attempts)
        with patch('rospy.loginfo'),patch.object(n,'_match_candidate_compatible',
                side_effect=AssertionError('ownership must not depend on pose')):
            for _ in range(4):self.assertIs(n._match_existing(self.region(n,ob['points'])),ob)
        self.assertEqual(len(n._obligations),1)
        self.assertEqual(n._dependency_attempts,budget)
        self.assertEqual(observation_token(ob),observation_token(before))

    def test_exact_solver_point_transport_and_subset_are_same_owner(self):
        n=self.node();ob=self.ob(n,6,[[-.2,0,.5],[-.15,.05,.4]]);n._obligations=[ob]
        with patch('rospy.loginfo'):
            self.assertIs(n._match_existing(self.region(n,np.float32([[-.15,.05,.4]]))),ob)

    def test_nearby_unowned_point_is_not_absorbed_by_invalid_pose(self):
        n=self.node();n._obligations=[self.ob(n,7,[[-.15,.05,.4]])]
        with patch('rospy.logwarn_throttle'):
            self.assertIsNone(n._match_existing(self.region(n,[[-.15,.05,.45]])))

    def test_current_published_pose_and_sensor_match_as_one_snapshot(self):
        n=self.node();ob=self.ob(n,7,[[-.15,.05,.4]]);n._obligations=[ob]
        q=np.full(7,.1)
        n._trace_published_target=dict(region_token=observation_region_token(ob),q_vis=q,
            q_vis_joint_mask=np.array([1,1,1,1,0,0,0.]),
            per_sensor_hybrid_used=True,per_sensor_selected_sensor_id=5)
        seen=[]
        n._per_sensor_runtime=NS(branch_score_numpy=lambda p,q,s:-.1,
            _candidate_geometry=lambda p,q,s:seen.append((q.copy(),s)) or dict(accepted=s==5))
        self.assertTrue(n._match_candidate_compatible(ob,self.region(n,[[-.15,.05,.45]])))
        np.testing.assert_equal(seen[0][0],q);self.assertEqual(seen[0][1],5)
        np.testing.assert_equal(ob['q_vis'],np.zeros(7))

    def test_foreign_publication_cannot_validate_new_geometry(self):
        n=self.node();ob=self.ob(n,7,[[-.15,.05,.4]])
        n._trace_published_target=dict(region_token='different',q_vis=np.ones(7),
            per_sensor_hybrid_used=True,per_sensor_selected_sensor_id=5)
        with patch('rospy.logwarn_throttle'):
            self.assertFalse(n._match_candidate_compatible(ob,self.region(n,[[-.15,.05,.45]])))

    def test_refined_child_cannot_absorb_foreign_coarse_region(self):
        n=self.node();ob=self.ob(n,7,[[-.15,.05,.4]]);ob['refinement_depth']=1
        self.assertFalse(n._match_candidate_compatible(ob,self.region(n,[[-.15,.05,.45]])))

if __name__=='__main__':unittest.main()
