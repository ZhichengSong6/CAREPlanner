import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
import phase_e_owned_processes as ownership

SCRIPT = Path(__file__).resolve().parent/'run_phase_c4_4_verified_regime_smoke.sh'


class SmokeGuards(unittest.TestCase):
    def test_backend_ready_names(self):
        prefix = SCRIPT.read_text().split('INITIAL_GATE_MAX_TRIES=')[0]
        for backend, expected in [('samples', 'trajectory_risk_node'), ('primitive', 'gcdf_primitive_anchors')]:
            env = dict(os.environ, GEOMETRY_BACKEND=backend, GCDF_GEOMETRY_BACKEND=backend)
            proc = subprocess.run(['bash'], input=prefix+'\nprintf "%s" "$GCDF_EXPORT_NODE"\n',
                                  text=True, capture_output=True, env=env, check=True)
            self.assertEqual(proc.stdout, expected)
        self.assertEqual(SCRIPT.read_text().count('/${GCDF_EXPORT_NODE}'), 3)

    def test_unknown_backend_rejected(self):
        prefix = SCRIPT.read_text().split('INITIAL_GATE_MAX_TRIES=')[0]
        proc = subprocess.run(['bash'], input=prefix, text=True, capture_output=True,
                              env=dict(os.environ, GEOMETRY_BACKEND='typo'))
        self.assertEqual(proc.returncode, 2)

    def test_no_global_kill_or_destructive_reset(self):
        text = SCRIPT.read_text()
        self.assertNotIn('pkill', text)
        self.assertNotIn('rm -rf', text)
        self.assertIn('OUT/LOG already exists', text)

    def test_descendants_only(self):
        with patch.object(ownership.subprocess, 'check_output', return_value='1 0\n2 1\n3 2\n4 0\n'), \
             patch.object(ownership, 'identity', side_effect=lambda pid: str(pid)):
            owned = {}; ownership.remember_descendants({1}, owned)
            self.assertEqual(owned, {2: '2', 3: '3'})

    def test_reused_pid_not_signalled(self):
        with patch.object(ownership, 'identity', return_value='new'), patch.object(ownership.os, 'kill') as kill:
            self.assertEqual(ownership.reap_owned({123: 'old'})['remaining'], [])
            kill.assert_not_called()

    def test_current_process_identity(self):
        self.assertIsNotNone(ownership.identity(os.getpid()))


if __name__ == '__main__': unittest.main()
