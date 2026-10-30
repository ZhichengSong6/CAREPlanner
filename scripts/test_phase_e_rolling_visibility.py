#!/usr/bin/env python3
"""Observation identity and feedback isolation at the production guidance boundary."""
import sys
from pathlib import Path
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/care_visibility_cdf/scripts'))
from observation_identity import observation_target_state, restore_observation_target_state, observation_token
from rolling_visibility_guidance import RollingVisibilityGuidance

class Field:
    def __init__(self): self.inputs=[]
    def branch_score_numpy(self,p,q,s): return float(q[1])
    def branch_value_and_grad_numpy(self,p,q,s):
        self.inputs.append((np.array(p),np.array(q),s))
        return float(q[1]), np.array([0.,1.,0.,0.,0.,0.,0.])

class Tests(unittest.TestCase):
    def snapshot(self):
        return dict(observation_token='care_obs_v1_test',region_token='care_region_v1_test',
            per_sensor_selected_sensor_id=2, q_vis=np.array([1.,0,0,0,0,0,0]),
            q_vis_joint_mask=np.array([1.,1,1,0,0,0,0]))
    def eval(self,field=None,q=None,feedback=None,stamp=1.,now=1.,drive=True,snapshot=None):
        return self.guide.evaluate(field or self.field,snapshot or self.snapshot(),
            [[0,0,1],[0,1,1]],np.zeros(7) if q is None else q,stamp,now,feedback,.05,drive)
    def setUp(self): self.guide=RollingVisibilityGuidance(); self.field=Field()
    def test_shadow_preserves_forward_and_drive_adds_observation_motion(self):
        shadow=self.eval(drive=False); drive=self.eval()
        self.assertEqual(shadow['target'][1],0)
        self.assertGreater(drive['target'][1],0)
        self.assertGreater(drive['target'][0],0)
        self.assertLessEqual(np.max(np.abs(drive['target'])),.05)
    def test_only_exact_fresh_feedback_removes_seen_points(self):
        fb=dict(token='care_obs_v1_test',region='care_region_v1_test',query_time=1.,
            points=np.array([[0,0,1],[0,1,1]]),point_seen=[True,False])
        self.eval(feedback=fb); self.assertEqual(len(self.field.inputs[-1][0]),1)
        fb['token']='care_obs_v1_old'
        self.eval(feedback=fb); self.assertEqual(len(self.field.inputs[-1][0]),2)
        fb['token']='care_obs_v1_test'; fb['query_time']=0.
        self.eval(feedback=fb); self.assertEqual(len(self.field.inputs[-1][0]),2)
    def test_changed_measurement_rechecks_field_and_preserves_target_identity(self):
        a=self.eval(); q=np.zeros(7); q[1]=.002; b=self.eval(q=q)
        self.assertEqual(len(self.field.inputs),2)
        self.assertEqual(a['observation_token'],b['observation_token'])
        self.assertAlmostEqual(b['model_gain'],.002)
    def test_stale_measurement_and_unknown_sensor_fall_back(self):
        self.assertIsNone(self.eval(stamp=.7))
        s=self.snapshot(); s['per_sensor_selected_sensor_id']=-1
        self.assertIsNone(self.eval(snapshot=s))
    def test_sensor_changes_identity_and_pin_restores_pose_mask_head(self):
        a=dict(id=1,points=[[0,0,1]],q_vis=np.zeros(7),q_vis_joint_mask=np.ones(7),
            per_sensor_selected_sensor_id=2)
        saved=observation_target_state(a); b=dict(a,per_sensor_selected_sensor_id=6)
        self.assertNotEqual(observation_token(a),observation_token(b))
        restore_observation_target_state(b,saved)
        self.assertEqual(observation_token(a),observation_token(b))
        a['q_vis'][0]=1.; self.assertEqual(saved['q_vis'][0],0.)
    def test_finite_step_rejects_misleading_derivative(self):
        field=Field()
        field.branch_score_numpy=lambda p,q,s: -abs(float(q[1]))
        result=self.eval(field=field)
        self.assertEqual(result['backoff'],2)
        self.assertEqual(result['target'][1],0.)
    def test_fresh_stamp_reuses_value_but_advances_transport_sequence(self):
        a=self.eval();b=self.eval(stamp=1.1,now=1.1)
        self.assertEqual(len(self.field.inputs),1)
        self.assertGreater(b['sequence'],a['sequence'])
    def test_geometry_uses_measured_pose_and_exact_unseen_set(self):
        self.field._fov_objective=lambda p,q,s:float(q[1]-.25)
        a=self.eval();q=np.zeros(7);q[1]=.003;b=self.eval(q=q)
        self.assertTrue(a['geometry_valid']);self.assertEqual(len(a['geometry_signature']),64)
        self.assertEqual(a['geometry_signature'],b['geometry_signature'])
        self.assertAlmostEqual(b['conservative_fov_margin']-a['conservative_fov_margin'],.003)
        fb=dict(token='care_obs_v1_test',region='care_region_v1_test',query_time=1.,
            points=np.array([[0,0,1],[0,1,1]]),point_seen=[True,False])
        c=self.eval(feedback=fb)
        self.assertNotEqual(a['geometry_signature'],c['geometry_signature'])
    def test_missing_geometry_does_not_invent_progress(self):
        a=self.eval();self.assertFalse(a['geometry_valid'])
        self.assertEqual(a['geometry_signature'],'none')

if __name__=='__main__': unittest.main()
