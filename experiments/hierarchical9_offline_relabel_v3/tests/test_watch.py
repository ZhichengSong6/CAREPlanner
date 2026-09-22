"""Read-only log-viewer regression tests; mock Slurm, never real submissions.

Nested directory intentionally keeps these tests outside common.code_identity's
existing top-level *.py inputs, so ongoing label runs remain compatible.
"""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

WATCH = Path(__file__).resolve().parents[1] / 'watch.sh'


class WatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.log = self.root / 'real_job.out'
        self.state = self.root / 'state.env'
        self.state.write_text('JOB_ID=16521\nPHASE=base\nWORKERS=4\nOUT=' +
                              shlex.quote(str(self.root / 'out')) + '\nLOG=' +
                              shlex.quote(str(self.log)) + '\n')
        self.env = dict(os.environ, PATH=str(self.bin)+':'+os.environ['PATH'],
                        LOG_PATH=str(self.log), MOCK_ROOT=str(self.root),
                        WATCH_INTERVAL_SECONDS='0.001')
        self.script('squeue', 'echo "RUNNING|3090node1"')
        self.script('sacct', 'exit 0')
        self.script('scontrol', 'echo "JobId=16521 StdOut=$LOG_PATH"')
        self.script('tail', 'echo "$*" >> "$MOCK_ROOT/tail_calls"\ncat -- "$LOG_PATH"')
        for name in ('sbatch', 'scancel', 'srun', 'kill', 'touch'):
            self.script(name, 'echo forbidden >> "$MOCK_ROOT/forbidden"; exit 97')

    def script(self, name, body):
        path = self.bin / name
        path.write_text('#!/usr/bin/env bash\n'+body+'\n')
        path.chmod(0o755)

    def run_watch(self, mode='main'):
        before = self.state.read_bytes()
        result = subprocess.run(['bash', str(WATCH), str(self.state), mode],
                                env=self.env, capture_output=True, text=True, timeout=5)
        self.assertEqual(before, self.state.read_bytes())
        self.assertFalse((self.root / 'forbidden').exists())
        return result

    def assert_no_tail(self):
        self.assertFalse((self.root / 'tail_calls').exists())
        self.assertFalse(self.log.exists())

    def test_pending_waits_before_tail_and_then_reads_real_output(self):
        self.script('squeue', '''
if [[ ! -f "$MOCK_ROOT/seen" ]]; then
  echo 1 > "$MOCK_ROOT/seen"; echo 'PENDING|Resources'
else
  echo 'REAL_LABEL_PROGRESS' > "$LOG_PATH"; echo 'RUNNING|3090node1'
fi''')
        result = self.run_watch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('state=PENDING reason=Resources', result.stdout)
        self.assertIn('[waiting]', result.stdout)
        self.assertIn('REAL_LABEL_PROGRESS', result.stdout)
        self.assertNotIn('cannot open', result.stderr)
        self.assertEqual(len((self.root/'tail_calls').read_text().splitlines()), 1)

    def test_existing_log_follows_without_wait(self):
        self.log.write_text('CURRENT_OUTPUT\n')
        result = self.run_watch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('CURRENT_OUTPUT', result.stdout)
        self.assertIn('-F', (self.root/'tail_calls').read_text())

    def test_ended_missing_log_diagnoses_and_stops(self):
        self.script('squeue', 'exit 0')
        self.script('sacct', "echo '16521|FAILED|1:0'")
        result = self.run_watch()
        self.assertEqual(result.returncode, 2)
        self.assertIn('state=FAILED', result.stdout)
        self.assertIn('StdOut=', result.stderr)
        self.assert_no_tail()

    def test_completed_log_prints_without_following(self):
        self.log.write_text('[done] real output\n')
        self.script('squeue', 'exit 0')
        self.script('sacct', "echo '16521|COMPLETED|0:0'")
        result = self.run_watch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('-F', (self.root/'tail_calls').read_text())
        self.assertIn('[finished]', result.stdout)

    def test_cancelled_by_user_is_terminal(self):
        self.script('squeue', 'exit 0')
        self.script('sacct', "echo '16521|CANCELLED by 12345|0:15'")
        result = self.run_watch()
        self.assertEqual(result.returncode, 2)
        self.assertIn('state=CANCELLED ', result.stdout)
        self.assert_no_tail()

    def test_unknown_is_not_claimed_pending_and_wait_is_bounded(self):
        self.script('squeue', 'exit 0')
        result = self.run_watch()
        self.assertEqual(result.returncode, 2)
        self.assertIn('state=UNKNOWN', result.stdout)
        self.assertNotIn('state=PENDING', result.stdout)
        self.assert_no_tail()

    def test_running_missing_log_does_not_wait_forever(self):
        result = self.run_watch()
        self.assertEqual(result.returncode, 2)
        self.assertIn('Job is active', result.stdout)
        self.assertIn('StdOut=', result.stderr)
        self.assert_no_tail()

    def test_scheduler_errors_are_visible_not_swallowed(self):
        self.script('squeue', 'echo controller-unavailable >&2; exit 1')
        self.script('sacct', 'echo accounting-unavailable >&2; exit 1')
        result = self.run_watch()
        self.assertEqual(result.returncode, 2)
        self.assertIn('controller-unavailable', result.stdout)
        self.assertIn('accounting-unavailable', result.stdout)
        self.assert_no_tail()

    def test_non_file_log_is_rejected(self):
        self.log.mkdir()
        result = self.run_watch()
        self.assertEqual(result.returncode, 2)
        self.assertIn('not a readable regular file', result.stderr)
        self.assertFalse((self.root/'tail_calls').exists())

    def test_other_modes_remain_read_only(self):
        for mode in ('workers', 'status', 'plan', 'summary'):
            result = self.run_watch(mode)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_no_tail()

    def test_invalid_state_fails_without_read_commands(self):
        self.state.write_text('JOB_ID=bad\n')
        result = self.run_watch()
        self.assertEqual(result.returncode, 2)
        self.assert_no_tail()

    def test_terminal_failed_log_still_prints_error_output(self):
        self.log.write_text('Traceback: actual worker error\n')
        self.script('squeue', 'exit 0')
        self.script('sacct', "echo '16521|FAILED|1:0'")
        result = self.run_watch()
        self.assertEqual(result.returncode, 2)
        self.assertIn('Traceback: actual worker error', result.stdout)
        self.assertNotIn('-F', (self.root/'tail_calls').read_text())


if __name__ == '__main__':
    unittest.main()
