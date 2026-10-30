"""Real sensor FK controls for geometric ascent; no network or ROS execution."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
from urdf_parser_py.urdf import URDF

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/care_visibility_cdf/scripts'))
from per_sensor_visibility_runtime import PerSensorVisibilityRuntime,DEFAULT_SENSOR_FRAMES,DEFAULT_JOINT_NAMES
from validate_visibility_oracle import find_chain_joints

class FovAscentTests(unittest.TestCase):
    def setUp(self):
        self.r=PerSensorVisibilityRuntime.__new__(PerSensorVisibilityRuntime)
        robot=URDF.from_xml_file(str(ROOT/'src/arm_description/urdf/Arm.urdf'))
        self.r.sensor_chains=[find_chain_joints(robot,'base_link',f) for f in DEFAULT_SENSOR_FRAMES]
        self.r.cons_hfov=np.deg2rad(50.);self.r.cons_vfov=np.deg2rad(66.)
        self.r.cons_z_min=.2;self.r.cons_z_max=.7;self.r.cons_delta=.06
        self.r.branch_step_size=.05;self.r.branch_max_step_norm=.25
        self.r.q_min=torch.full((7,),-2.);self.r.q_max=torch.full((7,),2.)
        self.r.branch_score=lambda *args:-1.
        self.r.branch_value_and_grad=lambda *args:(-1.,torch.zeros((1,7)))
    def test_geometry_step_improves_actual_fov_and_preserves_inactive_joints(self):
        q=torch.tensor([[-.344714,-.241394,-1.15114,.719927,-.065711,.219955,-.214443]])
        points=torch.tensor([[-.15,0.,.3]])
        before=self.r._fov_objective(points.numpy(),q[0].numpy(),2)
        score,g,objective=self.r._ascent_objective(points,q,2,True,-.02)
        self.assertEqual(objective,'conservative_fov')
        candidate,diag=self.r._ascent_step(q,g)
        after=self.r._fov_objective(points.numpy(),candidate[0].numpy(),2)
        self.assertGreater(after,before+1e-6)
        self.assertLessEqual(float(torch.linalg.vector_norm(candidate-q)),.050001)
        names={j.name for j in self.r.sensor_chains[2] if j.type!='fixed'}
        for j,name in enumerate(DEFAULT_JOINT_NAMES):
            if name not in names:self.assertEqual(float(q[0,j]),float(candidate[0,j]))
    def test_ordinary_ascent_keeps_learned_objective(self):
        score,g,objective=self.r._ascent_objective(torch.tensor([[.15,0.,.3]]),torch.zeros((1,7)),2,False,-.02)
        self.assertEqual(objective,'learned_visibility')

if __name__=='__main__':unittest.main()
