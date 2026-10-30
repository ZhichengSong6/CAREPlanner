from __future__ import annotations
import json
import io
import contextlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import launcher as m


class Finished:
    def __init__(self, rc=0):
        self.returncode, self.pid = rc, 999999999
    def poll(self): return self.returncode
    def wait(self, timeout=None): return self.returncode


class Tests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_nodes_single_or_two_not_two_jobs(self):
        self.assertEqual(m.parse_nodes('3090node2,3090node3'), ['3090node2', '3090node3'])
        self.assertEqual(m.parse_nodes('3090node3'), ['3090node3'])
        for bad in ('', '3090node2,3090node2', '3090node1', '3090node2,3090node3,3090node4', '3090node2;exit'):
            with self.assertRaises(ValueError): m.parse_nodes(bad)

    def test_two_nodes_sbatch_requests_four_gpus_each(self):
        cmd = m.sbatch_command('/out', '/repo', '3090node2,3090node3', '3-00:00:00')
        for arg in ('--nodes=2', '--ntasks=2', '--ntasks-per-node=1', '--gres=gpu:3090:4', '--exclusive'):
            self.assertIn(arg, cmd)
        self.assertEqual(cmd.count('sbatch'), 1)
        self.assertNotIn('--gres=gpu:3090:8', cmd)
        self.assertFalse(any('--array' in a for a in cmd))

    def test_single_node_command(self):
        cmd = m.sbatch_command('/out', '/repo', '3090node2', '1:00:00')
        self.assertIn('--nodes=1', cmd)
        self.assertIn('--ntasks=1', cmd)
        self.assertIn('--gres=gpu:3090:4', cmd)

    def run_submit(self, root, dry_rc=0, real_rc=0, real_output='12345\n'):
        calls = []
        def run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            dry = '--test-only' in cmd
            return SimpleNamespace(returncode=dry_rc if dry else real_rc,
                stdout='test only\n' if dry else real_output, stderr='denied' if dry_rc or real_rc else '')
        with mock.patch.object(m, 'ensure_account_idle'), mock.patch.object(m, 'verify_launcher'), \
             mock.patch.object(m, 'verify_data', return_value=Path(root)/'formal/job.json'), \
             mock.patch.object(m.subprocess, 'run', side_effect=run):
            try:
                m.submit(root, root, root, '3090node2,3090node3', 'submit')
                error = None
            except ValueError as exc:
                error = exc
        return calls, error

    def test_submit_creates_exactly_one_job_record(self):
        with tempfile.TemporaryDirectory() as td:
            calls, error = self.run_submit(td)
            self.assertIsNone(error)
            self.assertEqual(len(calls), 2)  # one test-only + one real submission
            self.assertEqual(sum('--test-only' not in c[0] for c in calls), 1)
            record = m.read(Path(td)/'latest_submission.json')
            self.assertEqual(len(record['jobs']), 1)
            self.assertEqual(record['jobs'][0]['gpus'], 8)
            self.assertEqual(record['jobs'][0]['nodes'], 2)
            self.assertTrue(record['single_allocation'])
            self.assertEqual(calls[1][1]['env']['G25_NODES'], '3090node2,3090node3')

    def test_dry_failure_submits_no_job_and_no_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            calls, error = self.run_submit(td, dry_rc=1)
            self.assertIsNotNone(error)
            self.assertEqual(len(calls), 1)
            self.assertIn('--test-only', calls[0][0])
            self.assertFalse((Path(td)/'latest_submission.json').exists())

    def test_real_failure_no_second_submit(self):
        with tempfile.TemporaryDirectory() as td:
            calls, error = self.run_submit(td, real_rc=1)
            self.assertIsNotNone(error)
            self.assertEqual(sum('--test-only' not in c[0] for c in calls), 1)
            self.assertFalse((Path(td)/'latest_submission.json').exists())
            self.assertEqual(len(list((Path(td)/'logs').glob('submission_attempt_*.json'))), 1)

    def test_ambiguous_reply_saved_not_retried(self):
        with tempfile.TemporaryDirectory() as td:
            calls, error = self.run_submit(td, real_output='unrecognized reply')
            self.assertIsNotNone(error)
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(list((Path(td)/'logs').glob('submission_attempt_*.json'))), 1)

    def test_existing_job_blocks(self):
        with mock.patch.object(m.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='100 RUNNING job\n', stderr='')):
            with self.assertRaisesRegex(ValueError, 'Existing queued/running'):
                m.ensure_account_idle()

    def test_unreadable_queue_blocks(self):
        with mock.patch.object(m.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stdout='', stderr='unavailable')):
            with self.assertRaisesRegex(ValueError, 'Cannot read'):
                m.ensure_account_idle()

    def test_one_srun_two_coordinators(self):
        cmd = m.srun_command('/out', '/frozen', '3090node2,3090node3')
        for a in ('--nodes=2', '--ntasks=2', '--ntasks-per-node=1', '--kill-on-bad-exit=1'):
            self.assertIn(a, cmd)
        self.assertEqual(cmd[0], 'srun')
        self.assertNotIn('sbatch', cmd)

    def test_global_worker_ids_and_logs_do_not_collide(self):
        logs, slots = [], []
        for rank in range(2):
            for local in range(4):
                cmd, env, log = m.worker_spec('/out', '/source', '/job.json', '9', rank, local, str(local))
                slots.append(int(cmd[-1])); logs.append(str(log))
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'], str(local))
        self.assertEqual(slots, list(range(8)))
        self.assertEqual(len(set(logs)), 8)

    def test_merge_once_after_srun_finishes(self):
        events = []
        def popen(cmd, **kwargs):
            events.append('srun')
            return Finished()
        def run(cmd, **kwargs):
            events.append(cmd[2])
            return SimpleNamespace(returncode=0)
        with mock.patch.object(m, 'verify_launcher'), mock.patch.object(m, 'verify_data', return_value=Path('/job.json')), \
             mock.patch.dict(os.environ, {'SLURM_JOB_ID':'123', 'SLURM_JOB_NUM_NODES':'2'}), \
             mock.patch.object(m.subprocess, 'Popen', side_effect=popen), \
             mock.patch.object(m.subprocess, 'run', side_effect=run):
            m.batch('/out', '/source', '3090node2,3090node3')
        self.assertEqual(events, ['srun', 'merge', 'verify'])

    def test_failed_node_step_prevents_merge(self):
        with mock.patch.object(m, 'verify_launcher'), mock.patch.object(m, 'verify_data', return_value=Path('/job.json')), \
             mock.patch.dict(os.environ, {'SLURM_JOB_ID':'123', 'SLURM_JOB_NUM_NODES':'2'}), \
             mock.patch.object(m.subprocess, 'Popen', return_value=Finished(1)), mock.patch.object(m.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'Node step failed'):
                m.batch('/out', '/source', '3090node2,3090node3')
            run.assert_not_called()

    def test_node_rank_one_launches_slots_four_to_seven(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td)/'logs').mkdir()
            calls=[]
            def popen(cmd, **kwargs):
                calls.append((cmd, kwargs));return Finished()
            with mock.patch.object(m, 'verify_launcher'), mock.patch.object(m, 'verify_data', return_value=Path('/job.json')), \
                 mock.patch.object(m, 'shared_lock_check'), mock.patch.object(m.subprocess, 'run'), \
                 mock.patch.object(m.subprocess, 'Popen', side_effect=popen), \
                 mock.patch.dict(os.environ, {'SLURM_JOB_ID':'123','SLURM_PROCID':'1','SLURM_LOCALID':'0','CUDA_VISIBLE_DEVICES':'0,1,2,3'}):
                m.node(td, '/frozen', '3090node2,3090node3')
            self.assertEqual([int(c[0][-1]) for c in calls], [4,5,6,7])
            self.assertEqual(len(list((Path(td)/'logs').glob('worker_123_*.log'))), 4)

    def test_real_interprocess_file_lock_handshake(self):
        with tempfile.TemporaryDirectory() as td:
            code='import launcher; launcher.shared_lock_check('+repr(td)+', "test", int(__import__("sys").argv[1]), 2)'
            env=dict(os.environ, PYTHONPATH=str(m.HERE), PYTHONDONTWRITEBYTECODE='1')
            procs=[subprocess.Popen([sys.executable,'-c',code,str(r)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for r in range(2)]
            try:
                for proc in procs:
                    out,err=proc.communicate(timeout=15)
                    self.assertEqual(proc.returncode, 0, out+err)
            finally:
                for proc in procs:
                    if proc.poll() is None: proc.kill();proc.wait()

    def test_freeze_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            snap=m.freeze(td)
            self.assertEqual(m.source_hashes(snap), m.source_hashes(m.HERE))
            (snap/'launcher.py').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'snapshot changed'): m.freeze(td)

    def test_shell_spool_uses_absolute_snapshot_path(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);spool=root/'spool.sh';spool.write_text((m.HERE/'run.sh').read_text())
            fake=root/'python';fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n');fake.chmod(0o755)
            env=dict(os.environ,VIS_PYTHON=str(fake),G25_SINGLE_CODE='/frozen/launcher',
                     G25_CODE='/frozen/collector',G25_ROOT='/shared/out',G25_NODES='3090node2,3090node3',SLURM_JOB_ID='12')
            r=subprocess.run(['bash',str(spool),'_batch'],env=env,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertIn('/frozen/launcher/launcher.py',r.stdout)
            self.assertNotIn(str(root)+'/launcher.py',r.stdout)

    def test_syntax_and_no_training_entry(self):
        subprocess.run(['bash','-n',str(m.HERE/'run.sh')],check=True)
        for name in ('launcher.py','test_single_job.py'):
            compile((m.HERE/name).read_text(),name,'exec')
        self.assertNotIn('torch.distributed', (m.HERE/'launcher.py').read_text())


if __name__ == '__main__':
    unittest.main(verbosity=2)
