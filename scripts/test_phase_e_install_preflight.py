#!/usr/bin/env python3
"""Read-only preflight guards, without starting a ROS graph or loading models."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import check_phase_e_install as preflight


class InstallPreflightTests(unittest.TestCase):
    def test_elf_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'fixture'
            path.write_bytes(b'\x7fELFtest')
            self.assertTrue(preflight.is_elf(path))
            path.write_bytes(b'#!/usr/bin/python3')
            self.assertFalse(preflight.is_elf(path))

    def test_rejects_source_package_fallback(self):
        with patch('rospkg.RosPack.get_path', return_value='/source/package'):
            result = preflight.inspect(Path('/install'))
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['failure_stage'], 'package resolution')

    def inspect_fixture(self, prefix):
        with patch('rospkg.RosPack.get_path', side_effect=lambda p: str(prefix/'share'/p)):
            return preflight.inspect(prefix)

    def test_rejects_bundled_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            prefix = Path(folder); (prefix/'final.pt').write_bytes(b'fixture')
            result = self.inspect_fixture(prefix)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['failure_stage'], 'artifact provenance')

    def test_rejects_source_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            prefix = Path(folder); (prefix/'external').symlink_to('/etc/hosts')
            result = self.inspect_fixture(prefix)
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('external artifact symlink', result['error'])

    def test_rejects_missing_binaries(self):
        with tempfile.TemporaryDirectory() as folder:
            prefix = Path(folder); (prefix/'manifest.txt').write_text('fixture')
            result = self.inspect_fixture(prefix)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['failure_stage'], 'dynamic linker')


if __name__ == '__main__':
    unittest.main()
