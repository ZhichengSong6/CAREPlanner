#!/usr/bin/env python3
"""Bounded install entry/provenance checks; no ROS graph or model execution."""
import ast
import json
import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT/'scripts/with_phase_e_primitive_release.sh'
PREFIX = ROOT/'install/primitive_release'


class EntryTests(unittest.TestCase):
    def run_entry(self, *args, env=None):
        return subprocess.run(['bash', str(ENTRY), *args], cwd=ROOT,
                              capture_output=True, text=True, env=env, timeout=10)

    def test_no_implicit_launch(self):
        result = self.run_entry()
        self.assertEqual(result.returncode, 2)
        self.assertIn('Usage:', result.stderr)

    def test_child_exit_code_preserved(self):
        self.assertEqual(self.run_entry('/usr/bin/false').returncode, 1)

    def test_contaminated_overlay_removed_in_child_only(self):
        keys = ['CMAKE_PREFIX_PATH', 'ROS_PACKAGE_PATH', 'PYTHONPATH',
                'LD_LIBRARY_PATH', 'PKG_CONFIG_PATH', 'ROSLISP_PACKAGE_DIRECTORIES']
        sentinel = '/tmp/care_invalid_overlay/devel'
        env = dict(os.environ, **{key: sentinel for key in keys})
        result = self.run_entry('/usr/bin/python3', '-c',
            'import json, os, rospkg; print(json.dumps(dict('
            'package=rospkg.RosPack().get_path("care_confidence_map"), env=dict(os.environ))))', env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data['package'], str(PREFIX/'share/care_confidence_map'))
        for key in keys:
            self.assertNotIn(sentinel, data['env'].get(key, ''))
            self.assertEqual(env[key], sentinel)

    def test_installed_lifecycle_code_matches_source_and_is_wired(self):
        specs = [('egocentric_arm_planner', 'joint_velocity_rate_limiter_node.py', 'lib', '_timer_cb'),
                 ('care_visibility_cdf', 'vbc_deadline_waypoint_node.py', 'share', '_run_timer_callback'),
                 ('care_visibility_cdf', 'vbc_visibility_acquisition_impl.py', 'share', None)]
        for package, filename, folder, callback in specs:
            with self.subTest(file=filename):
                source = ROOT/'src'/package/'scripts'/filename
                installed = PREFIX/folder/package/('scripts' if folder == 'share' else '')/filename
                tree = ast.parse(source.read_text())
                self.assertEqual(ast.dump(tree), ast.dump(ast.parse(installed.read_text())))
                if callback:
                    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
                    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
                    calls = [n for n in ast.walk(init) if isinstance(n, ast.Call)]
                    self.assertTrue(any(isinstance(n.func, ast.Attribute) and n.func.attr == 'on_shutdown'
                                        and isinstance(n.args[0], ast.Attribute) and n.args[0].attr == '_on_shutdown'
                                        for n in calls))
                    self.assertTrue(any(isinstance(n.func, ast.Attribute) and n.func.attr == 'Timer'
                                        and isinstance(n.args[1], ast.Attribute) and n.args[1].attr == callback
                                        for n in calls))


if __name__ == '__main__':
    unittest.main()
