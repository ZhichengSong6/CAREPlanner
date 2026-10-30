#!/usr/bin/env python3
"""Independent Pinocchio checks of interval geometry and complete grid masks."""
import itertools
import math
from pathlib import Path
import unittest
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from audit_case001_motion_prior import collision_sdf, frame_placements
from compare_phase_e_stop_enclosures import target_range_budget
from phase_e_local_stop_geometry import LocalPrimitiveEnclosure, multiply, trig
from phase_e_stopping_contract import bounded_update_travel

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'
META = dict(frame='base_link', resolution=.05, map_min=[-.95, -.95, 0.],
            map_max=[.95, .95, 1.15], grid_size=[39, 39, 24])


def primitive_extent(shape, placement):
    rotation = np.asarray(placement.rotation)
    if shape['kind'] == 'box':
        return np.abs(rotation) @ shape['half_size']
    if shape['kind'] == 'cylinder':
        return shape['radius']*np.sqrt(rotation[:, 0]**2+rotation[:, 1]**2)+shape['half_length']*np.abs(rotation[:, 2])
    return np.full(3, shape['radius'])


class LocalGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = LocalPrimitiveEnclosure(URDF, subdivisions=8)

    def test_trigonometric_extrema_and_full_turn(self):
        for low, high in [(-.01, .01), (1.5, 1.6), (-4., 4.), (30., 31.), (-9.5, -9.4)]:
            for cosine in [False, True]:
                a, b = trig(low, high, cosine)
                values = (np.cos if cosine else np.sin)(np.linspace(low, high, 10001))
                self.assertTrue(np.all(values >= a))
                self.assertTrue(np.all(values <= b))
        self.assertLess(trig(-4., 4.)[0], -1.)
        self.assertGreater(trig(-4., 4.)[1], 1.)

    def test_interval_product_handles_sign_changes(self):
        low, high = multiply((np.array([-2., 0., -4.]), np.array([3., 2., -1.])),
                             (np.array([-5., -3., -2.]), np.array([2., 4., 6.])))
        self.assertTrue(np.all(low <= [-15., -6., -24.]))
        self.assertTrue(np.all(high >= [10., 8., 8.]))

    def test_interval_fk_contains_independent_exact_solid_extrema(self):
        rng = np.random.default_rng(20261019)
        checked = 0
        for q, travel in [(np.zeros(7), np.zeros(7)),
                          (np.array([1.57, -1.57, 0., 1.57, -1.57, 0., 0.]), np.full(7, .15)),
                          (rng.uniform(-1., 1., 7), np.array([.8, .5, .3, .7, .4, .2, .6])),
                          (np.zeros(7), np.array([3.2, 0., 0., 0., 0., 0., 0.]))]:
            boxes = self.geometry.boxes(q, travel)
            samples = [q+np.asarray(sign)*travel for sign in itertools.product([-1., 1.], repeat=7)]
            samples.extend(q+rng.uniform(-1., 1., 7)*travel for _ in range(256))
            for sample in samples:
                full = np.zeros(self.geometry.model.nq); full[self.geometry.qindices] = sample
                poses = frame_placements(self.geometry.model, self.geometry.data, self.geometry.shapes, full)
                for i in self.geometry.active:
                    size = primitive_extent(self.geometry.shapes[i], poses[i])+.020
                    low, high = poses[i].translation-size, poses[i].translation+size
                    self.assertTrue(any(np.all(low >= a) and np.all(high <= b) for a,b in boxes[i]),
                                    (self.geometry.shapes[i]['name'], sample))
                    checked += 1
        self.assertEqual(checked, 4*384*len(self.geometry.active))

    def test_raster_contains_complete_sampled_pose_grid_sets(self):
        rng = np.random.default_rng(7)
        q = np.array([.4, -.4, .3, -.2, .1, 0., 0.])
        travel = np.array([.12, .07, .09, .04, .08, .10, .05])
        result = self.geometry.raster(q, travel, META)
        selected = np.zeros(len(self.geometry.points), dtype=bool)
        selected[result['grid_indices']] = True
        for sample in [q, q-travel, q+travel]+[q+rng.uniform(-1., 1., 7)*travel for _ in range(12)]:
            full = np.zeros(self.geometry.model.nq); full[self.geometry.qindices] = sample
            poses = frame_placements(self.geometry.model, self.geometry.data, self.geometry.shapes, full)
            for i in self.geometry.active:
                required = collision_sdf(self.geometry.points, self.geometry.shapes[i], poses[i]) <= .020
                self.assertFalse(np.any(required & ~selected))

    def test_zero_travel_preserves_stationary_grid_and_map_boundary(self):
        result = self.geometry.raster(np.zeros(7), np.zeros(7), META)
        from phase_e_stop_shadow import PrimitiveEnclosure
        baseline = PrimitiveEnclosure(URDF).raster(np.zeros(7), np.zeros(7), META)
        self.assertEqual(result['grid_indices'], baseline['grid_indices'])
        self.assertEqual(result['inside'], baseline['inside'])
        bad = dict(META, map_max=[.95, .95, .2], grid_size=[39, 39, 5])
        self.assertFalse(self.geometry.raster(np.zeros(7), np.zeros(7), bad)['inside'])

    def test_reference_margin_points_are_required_at_every_refinement(self):
        q=np.array([.4,-.4,.3,-.2,.1,0.,0.]);travel=np.array([.12,.07,.09,.04,.08,.1,.05])
        for n in [1,2,4,8]:
            result=self.geometry.raster(q,travel,META,subdivisions=n)
            reference=result['reference_pose_grid_indices']
            self.assertTrue(set(reference).issubset(result['grid_indices']))
            full=np.zeros(self.geometry.model.nq);full[self.geometry.qindices]=q
            poses=frame_placements(self.geometry.model,self.geometry.data,self.geometry.shapes,full)
            distance=np.min([collision_sdf(self.geometry.points[reference],self.geometry.shapes[i],poses[i])
                             for i in self.geometry.active],axis=0)
            self.assertTrue(np.all(distance<=.020-1e-10))

    def test_invalid_inputs_and_resources_fail_closed(self):
        for travel in [[-.1]*7, [math.nan]*7, [0.]*6]:
            with self.assertRaises(ValueError):
                self.geometry.boxes(np.zeros(7), travel)
        with self.assertRaises(ValueError):
            LocalPrimitiveEnclosure(URDF, subdivisions=0)
        with self.assertRaises(ValueError):
            self.geometry.raster(np.zeros(7), np.zeros(7), dict(META, grid_size=[1000]*3))

    def test_refinement_workspace_preserves_points_and_separates_bindings(self):
        q=np.array([.4,-.4,.3,-.2,.1,0.,0.]);travel=np.array([.12,.07,.09,.04,.08,.1,.05])
        workspace={}
        for n in [1,2,4,8]:
            a=self.geometry.raster(q,travel,META,subdivisions=n)
            b=self.geometry.raster(q,travel,META,subdivisions=n,workspace=workspace)
            for key in ['grid_indices','inside','reference_pose_grid_indices','point_set_sha256','minimum','maximum']:
                self.assertEqual(a[key],b[key],key)
        context=next(iter(workspace.values()))
        self.assertGreater(context['interval_pose_hits'],0)
        self.assertGreater(context['sdf_reused_points'],0)
        q=q+.01
        a=self.geometry.raster(q,travel,META,subdivisions=2)
        b=self.geometry.raster(q,travel,META,subdivisions=2,workspace=workspace)
        self.assertEqual(a['grid_indices'],b['grid_indices'])
        self.assertEqual(len(workspace),2)

    def test_unsupported_joint_axis_cannot_silently_use_z_rotation(self):
        root=ET.parse(URDF).getroot()
        root.find("joint[@name='joint2']/axis").set('xyz','0 1 0')
        with tempfile.TemporaryDirectory(prefix='care_axis_',dir='/tmp') as directory:
            path=Path(directory)/'unsupported.urdf';ET.ElementTree(root).write(path)
            with self.assertRaisesRegex(ValueError,'unsupported_joint_axis_joint2'):
                LocalPrimitiveEnclosure(path)

    def test_restricted_budget_preserves_existing_nonzero_output(self):
        travel, horizon = target_range_budget([1., 0., 0., 0., 0., 0., 0.],
            [3.]*7, [2.]*7, [0.]*7, .1, .001, .02)
        self.assertGreaterEqual(travel[0], 1./6)
        np.testing.assert_array_equal(travel[1:], 0.)
        self.assertGreater(horizon, .3)

    def test_full_target_range_matches_existing_receipt_anchored_budget(self):
        rng = np.random.default_rng(100)
        for _ in range(100):
            applied = rng.uniform(-2., 2., 7)
            args = (applied, [3., 3., 4., 4., 6., 6., 6.], [2.]*7)
            hold, gap = rng.uniform(0., .15), rng.uniform(.001, .05)
            travel, horizon = target_range_budget(*args, [2.]*7, hold, gap, .02)
            reference, until = bounded_update_travel(*args, hold, gap, .02)
            np.testing.assert_array_equal(travel, reference)
            self.assertEqual(horizon, until)


if __name__ == '__main__':
    unittest.main()
