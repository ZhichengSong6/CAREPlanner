"""Offline source/staged asset checks; no ROS master or robot."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'src/care_confidence_map'
ALIASES = ['config/body_samples.yaml', 'scripts/generate_body_samples.py',
           'scripts/body_sample_marker_publisher.py', 'src/body_sample_model.cpp']
STAGE = os.environ.get('CARE_LEGACY_STAGE_PREFIX')


class SourceAssets(unittest.TestCase):
    def test_aliases_are_single_source_of_truth(self):
        for path in ALIASES:
            old, canonical = PACKAGE/path, PACKAGE/'legacy'/path
            self.assertTrue(old.is_symlink(), path)
            self.assertFalse(Path(os.readlink(old)).is_absolute())
            self.assertEqual(old.resolve(strict=True), canonical.resolve(strict=True))
            self.assertEqual(old.read_bytes(), canonical.read_bytes())

    def test_generator_old_cli_still_works(self):
        result = subprocess.run([sys.executable, str(PACKAGE/ALIASES[1]), '--help'],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--urdf', result.stdout)

    def test_install_uses_canonical_scripts_and_source(self):
        cmake = (PACKAGE/'CMakeLists.txt').read_text()
        self.assertIn('  legacy/src/body_sample_model.cpp', cmake)
        self.assertIn('  legacy/scripts/generate_body_samples.py', cmake)
        self.assertIn('  legacy/scripts/body_sample_marker_publisher.py', cmake)
        self.assertNotIn('\n  scripts/generate_body_samples.py', cmake)
        self.assertIn('install(DIRECTORY legacy/config', cmake)


@unittest.skipUnless(STAGE, 'set CARE_LEGACY_STAGE_PREFIX for staged install checks')
class StagedAssets(unittest.TestCase):
    def test_installed_yaml_alias(self):
        share = Path(STAGE)/'share/care_confidence_map'
        alias = share/'config/body_samples.yaml'
        self.assertTrue(alias.is_symlink())
        self.assertEqual(alias.resolve(strict=True), (share/'legacy/config/body_samples.yaml').resolve(strict=True))
        self.assertEqual(alias.read_bytes(), (PACKAGE/ALIASES[0]).read_bytes())

    def test_installed_scripts_are_real_executables(self):
        for name in ['generate_body_samples.py', 'body_sample_marker_publisher.py']:
            path = Path(STAGE)/'lib/care_confidence_map'/name
            self.assertTrue(path.is_file())
            self.assertFalse(path.is_symlink())
            self.assertTrue(os.access(path, os.X_OK))
            compile(path.read_text(), str(path), 'exec')

    def test_installed_generator_byte_identical(self):
        source = PACKAGE/ALIASES[0]
        body = yaml.safe_load(source.read_text())['body_sampling']
        params = ['cylinder_spacing', 'cylinder_safety_margin', 'cylinder_radius_scale',
                  'cylinder_center_span_scale', 'box_cell_size', 'box_safety_margin', 'box_radius_scale']
        with tempfile.TemporaryDirectory(prefix='care_legacy_generator_') as tmp:
            output = Path(tmp)/'samples.yaml'
            cmd = [sys.executable, str(Path(STAGE)/'lib/care_confidence_map/generate_body_samples.py'),
                   '--urdf', str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
                   '--output', str(output)]
            for key in params: cmd.extend(['--'+key.replace('_', '-'), str(body[key])])
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10)
            self.assertEqual(output.read_bytes(), source.read_bytes())


if __name__ == '__main__': unittest.main()
