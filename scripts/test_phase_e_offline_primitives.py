#!/usr/bin/env python3
"""Offline tool migration tests. No cases are changed or regenerated."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pinocchio as pin
import coal

from phase_e_offline_body_geometry import (BodyPrimitive, load_body_primitives, world_primitives,
    pair_clearance, primitive_inside_workspace, aabb_extent, selected_body_path)
from test_phase_e_goal_terminal_feasibility import load_body_spheres, terminal_clearance, BoxObstacle
from generate_phase_e_random_initial_q_pool import load_all_body_spheres, transformed, nonadjacent_self_clearance, body_inside_workspace
from build_phase_e_random_q0_cross_pairs import load_startup_prior_spheres

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'
YAML = ROOT/'src/care_confidence_map/config/body_samples.yaml'


class OfflineTests(unittest.TestCase):
    def test_real_models_and_no_yaml(self):
        with patch('yaml.safe_load', side_effect=AssertionError('primitive read YAML')):
            risk = load_body_spheres(URDF, .015, 'primitive')
            all_body = load_all_body_spheres(URDF, .02, 'primitive')
            startup = load_startup_prior_spheres(URDF, .03, False, 'primitive')
        self.assertEqual(len(risk), 24); self.assertEqual(len(all_body), 26); self.assertEqual(len(startup), 26)
        self.assertTrue(all(isinstance(p, BodyPrimitive) for p in risk))
        self.assertTrue(all(p.inflation == .03 for p in startup))
        self.assertEqual(len(load_body_spheres(YAML, .015)), 87)
        self.assertEqual(len(load_all_body_spheres(YAML, 0.)), 90)

    def test_selected_path_ignores_missing_yaml(self):
        args = argparse.Namespace(geometry_backend='primitive', primitive_urdf=None)
        self.assertEqual(selected_body_path(args, ROOT, Path('/missing.yaml')), URDF)

    def test_coal_primitive_pairs_and_inflation(self):
        for kind, dims, shape in [('box', (.4, .4, .4), coal.Box(.4, .4, .4)),
                                  ('sphere', (.2,), coal.Sphere(.2)), ('cylinder', (.2, .4), coal.Cylinder(.2, .4))]:
            p = BodyPrimitive('link2', 'link2', np.zeros(3), np.eye(3), shape, .015, 0, kind, dims)
            value = pair_clearance(p, (np.eye(3), np.array([3., 0., 0.])), coal.Box(2., 2., 2.), (np.eye(3), np.zeros(3)))
            self.assertAlmostEqual(value, 1.785, places=7)
            self.assertLess(pair_clearance(p, (np.eye(3), np.zeros(3)), coal.Box(2., 2., 2.), (np.eye(3), np.zeros(3))), 0)
            np.testing.assert_allclose(aabb_extent(p, np.eye(3)), [.215]*3, atol=1e-12)

    def test_actual_fk_all_tool_consumers(self):
        model = pin.buildModelFromUrdf(str(ROOT/'src/arm_description/urdf/Arm.urdf'))
        body = load_body_spheres(URDF, .015, 'primitive')
        obstacle = BoxObstacle('fixture', np.array([2., 0., 0.]), np.eye(3), np.array([.2]*3))
        for scale in [0., .03, -.03]:
            q = scale*np.arange(1, 8)
            value, info = terminal_clearance(model, q, body, [obstacle])
            self.assertGreater(value, 0); self.assertEqual(info['geometry_backend'], 'primitive')
            actual, poses = transformed(model, q, body)
            self.assertEqual(len(poses), len(body))
            distance, _ = nonadjacent_self_clearance(actual, poses)
            self.assertTrue(np.isfinite(distance))
            self.assertTrue(body_inside_workspace(actual, poses, [-10, 10, -10, 10, -10, 10], 0.))
            self.assertFalse(body_inside_workspace(actual, poses, [10, 11, 10, 11, 10, 11], 0.))

    def test_fail_closed_bad_geometry(self):
        with self.assertRaises(ValueError): load_body_primitives(URDF, -.01)
        with self.assertRaises(ValueError): load_body_primitives(URDF, float('nan'))
        with self.assertRaises(ValueError): load_body_spheres(YAML, 0., 'typo')
        model = pin.buildModelFromUrdf(str(ROOT/'src/arm_description/urdf/Arm.urdf'))
        with self.assertRaises(ValueError): world_primitives(model, np.full(7, np.nan), load_body_primitives(URDF, 0.))
        with self.assertRaises(ValueError): terminal_clearance(model, np.zeros(7), load_body_primitives(URDF, 0.), [])
        with self.assertRaises(ValueError): primitive_inside_workspace([], [], [-1, 1]*3, 0.)
        from phase_e_offline_body_geometry import primitive_self_clearance
        with self.assertRaises(ValueError): primitive_self_clearance([], [], {})

    def test_four_cli_entries(self):
        import sys
        for name in ['test_phase_e_goal_terminal_feasibility.py', 'generate_phase_e_obstacle_goals.py',
                     'build_phase_e_random_q0_cross_pairs.py', 'generate_phase_e_random_initial_q_pool.py']:
            p = subprocess.run([sys.executable, str(ROOT/'scripts'/name), '--help'], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn('--geometry-backend', p.stdout); self.assertIn('--primitive-urdf', p.stdout)


if __name__ == '__main__':
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-file', type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OfflineTests))
    if args.output_file:
        with args.output_file.open('x') as handle:
            json.dump(dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
                           failures=len(result.failures), errors=len(result.errors)), handle, indent=2)
            handle.write('\n')
    sys.exit(0 if result.wasSuccessful() else 1)
