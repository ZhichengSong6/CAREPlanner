#!/usr/bin/env python3
"""Test diagnostic certificates/reconstruction, not runtime behavior."""
import copy
import unittest
import numpy as np
from diagnose_phase_e_repair_conflicts import lp_check, reconstruct, grid_point, JOINTS
from summarize_phase_e_fix34_regression import bound_conflicts


def fixture():
    cfg=dict(local_planner=dict(num_intervals=1,horizon_duration=.05,
        enforce_acceleration_constraints=False,cdf=dict(safety_margin=0.,linearization_tolerance_inf=.0001)),
        mpc=dict(joint_position_margin=.01,joint_position_limits={j:dict(lower=-3.,upper=3.) for j in JOINTS}))
    a=[[j,j,1.] for j in range(7)]+[[j,7+j,-.05] for j in range(7)]
    data=dict(P=dict(shape=[14,14],coo=[]),A=dict(shape=[7,14],coo=a),
        G=dict(shape=[1,14],coo=[[0,0,1.]]),b=[0.]*7,c=[0.]*14,
        x_l=[-.5]*7+[-1.]*7,x_u=[.5]*7+[1.]*7,h_l=[.8],h_u=[1e30],
        q_linearization_flat=[0.]*7,gradient_flat=[1.]+[0.]*6,
        original_timestep=[1],source_type=[0],selected_pairs=[[0,0]],distance=[-.8])
    return data,cfg


class DiagnosisTests(unittest.TestCase):
    def test_row_box_certificate_and_independent_lp(self):
        d,_=fixture(); original=copy.deepcopy(d)
        self.assertAlmostEqual(bound_conflicts(d)[0]['deficit'],.3)
        self.assertEqual(lp_check(d)['status'],2)
        self.assertTrue(lp_check(d,[])['feasible'])
        self.assertEqual(d,original,'diagnostic mutated the original problem')

    def test_feasible_original_lp_is_not_called_infeasible(self):
        d,_=fixture();d['h_l']=[.02]
        self.assertEqual(bound_conflicts(d),[])
        self.assertTrue(lp_check(d)['feasible'])

    def test_negative_gradient_uses_lower_variable_bound(self):
        d,_=fixture();d['G']['coo'][0][2]=-2.
        d['h_l']=[1.2]
        self.assertAlmostEqual(bound_conflicts(d)[0]['deficit'],.2)

    def test_reconstruct_unclipped_center_and_dynamics(self):
        d,cfg=fixture();q0,center,radius,error=reconstruct(d,cfg)
        np.testing.assert_array_equal(q0,np.zeros(7))
        np.testing.assert_array_equal(center,np.zeros((1,7)))
        self.assertEqual(radius,.5); self.assertEqual(error,0)

    def test_refuses_joint_clipped_or_nonuniform_box(self):
        for lo in (-2.99,-.6):
            d,cfg=fixture(); d['x_l'][0]=lo
            with self.assertRaises(ValueError): reconstruct(d,cfg)

    def test_refuses_wrong_q_mapping_and_wrong_timestep(self):
        d,cfg=fixture(); d['q_linearization_flat'][0]=.001
        with self.assertRaises(ValueError): reconstruct(d,cfg)
        d,cfg=fixture(); d['A']['coo'][7][2]=-.1
        with self.assertRaises(AssertionError): reconstruct(d,cfg)
        d,cfg=fixture(); d['G']['coo'][0][1]=1
        with self.assertRaises(AssertionError): reconstruct(d,cfg)

    def test_grid_float32_identity_not_nearest_arbitrary_point(self):
        cfg=dict(x_min=-.95,y_min=-.95,z_min=0.,resolution=.05)
        p=np.array([.1,.1,.3],dtype=np.float32)
        np.testing.assert_array_equal(grid_point(p,cfg).astype(np.float32),p)
        with self.assertRaises(ValueError): grid_point([.101,.1,.3],cfg)


if __name__=='__main__': unittest.main(verbosity=2)
