"""Actual finite-branch generator with controlled map cost responses."""
from pathlib import Path
import json,sys,unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/care_visibility_cdf/scripts'))
import per_sensor_visibility_runtime as mod
from observation_task_rejoin import task_rejoin_context


class RuntimeTests(unittest.TestCase):
    def runtime(self):
        r=object.__new__(mod.PerSensorVisibilityRuntime)
        r.device=torch.device('cpu');r.max_branch_attempts=4;r.min_conservative_g=0.
        r.task_ee_chain=[];r.sensor_masks_np=np.ones((8,7))
        r.rank_sensors=lambda *args:(torch.zeros(8),torch.arange(8))
        r._optimize_branch=lambda points,seed,sid:dict(sensor_id=sid,sensor_frame='controlled',
            q_candidate=np.full(7,0. if sid==0 else .25).tolist())
        r._candidate_geometry=lambda *args,**kwargs:dict(accepted=True)
        for name,value in dict(projection_iters=10,projection_damping=.5,projection_epsilon_f=.03,
            projection_max_step_norm=.25,root_refine_iters=12,root_tolerance_f=.002,
            branch_ascent_steps=1,branch_step_size=.05,branch_max_step_norm=.25,
            branch_fallback_ascent_steps=8,require_primitive_los=True,checkpoint_path='controlled',
            checkpoint={'step':0,'args':{}},ray_args=type('Ray',(),{'self_filter_padding':0.})()).items():setattr(r,name,value)
        return r

    def generate(self,callback):
        r=self.runtime();c=task_rejoin_context(np.zeros(7),np.ones(7),np.ones(7),np.ones(7))
        c['corridor_evaluator']=callback
        with patch.object(mod,'fk_transform',return_value=np.eye(4)):
            result=r.generate([[.2,0.,.5]],np.zeros(7),task_rejoin=c)
        json.dumps(result) # No callback or nonserializable context in traces.
        return result

    def valid(self,risk):
        return dict(status='valid',approach=dict(risk_fraction=risk),rejoin=dict(risk_fraction=risk),map_identity=['map',1])

    def test_new_unknown_cost_changes_a_real_generator_choice_within_original_slots(self):
        r=self.generate(lambda q,ctx:self.valid(1. if q[0]==0. else 0.))
        self.assertEqual(r['selected_sensor_id'],1)
        self.assertEqual(len(r['attempts']),2)
        self.assertEqual(r['task_rejoin']['selected_by'],'bounded_task_rejoin_improvement')

    def test_missing_alternative_snapshot_keeps_the_first_accepted_fallback(self):
        r=self.generate(lambda q,ctx:self.valid(1.) if q[0]==0. else dict(status='unavailable'))
        self.assertEqual(r['selected_sensor_id'],0)
        self.assertEqual(len(r['attempts']),4)

    def test_preview_failure_keeps_the_original_motion_proxy(self):
        r=self.generate(lambda q,ctx:dict(status='unavailable'))
        self.assertEqual(r['selected_sensor_id'],0)
        self.assertEqual(len(r['attempts']),4)


if __name__=='__main__':unittest.main(verbosity=2)
