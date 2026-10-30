"""Single-job shell regressions using an actual temporary Git repo and fake Slurm.

No GPU, server access or dataset modifications occur in these tests.
"""
from pathlib import Path
import ast
import json
import os
import re
import time
from unittest import mock
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


class SingleJobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / 'repo'
        self.src = self.repo / 'experiments/fixture'
        self.src.mkdir(parents=True)
        self.script = self.src / 'run.sh'
        shutil.copyfile(HERE / 'run.sh', self.script)
        (self.src / 'test_fixture.py').write_text('import unittest\nclass Fixture(unittest.TestCase):\n def test_ok(self): self.assertTrue(True)\n')
        self.out = self.root / 'out'
        (self.out / 'formal/shards/00000').mkdir(parents=True)
        (self.out / 'formal/job.json').write_text('{"frozen": true}\n')
        (self.out / 'formal/shards/00000/marker').write_text('published shard\n')
        self.code = self.out / 'sources/old_frozen'
        self.code.mkdir(parents=True)
        (self.code / 'run.py').write_text(
            'import json, os, sys\n'
            'with open(os.environ["FAKE_CALLS"], "a") as f:\n'
            '    f.write(json.dumps(sys.argv[1:])+"\\n")\n'
            'if sys.argv[1] in ("submit", "resume"):\n'
            '    assert sys.argv[sys.argv.index("--nodes")+1] in ("3090node2", "3090node3")\n'
            '    print("[submitted] fake=1")\n')
        (self.out / 'source.env').write_text('export CODE=' + shlex.quote(str(self.code)) + '\n')
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        sq = self.bin / 'squeue'
        sq.write_text('#!/bin/sh\nprintf "%s" "${FAKE_QUEUE:-}"\nexit "${FAKE_QUEUE_RC:-0}"\n')
        sq.chmod(0o755)
        self.calls = self.root / 'calls.jsonl'
        self.env = os.environ.copy()
        for k in ('NODES', 'NODE', 'FAKE_QUEUE', 'FAKE_QUEUE_RC'):
            self.env.pop(k, None)
        self.env.update(VIS_PYTHON=sys.executable, GRID25_ROOT=str(self.out),
                        FAKE_CALLS=str(self.calls),
                        PATH=str(self.bin) + os.pathsep + self.env.get('PATH', ''))
        subprocess.run(['git', 'init', '-q', str(self.repo)], check=True)
        subprocess.run(['git', '-C', str(self.repo), 'symbolic-ref', 'HEAD',
                        'refs/heads/mainline-b/h9-scratch50k-r012-v1'], check=True)
        subprocess.run(['git', '-C', str(self.repo), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(self.repo), '-c', 'user.name=Test',
                        '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture'], check=True)

    def invoke(self, mode='submit', **env):
        e = dict(self.env); e.update(env)
        return subprocess.run(['bash', str(self.script), mode], env=e,
                              text=True, capture_output=True, timeout=30)

    def records(self):
        return [json.loads(s) for s in self.calls.read_text().splitlines()] if self.calls.exists() else []

    def test_default_one_node_one_submission_and_frozen_source_preserved(self):
        before = {p.relative_to(self.out): p.read_bytes() for p in self.out.rglob('*') if p.is_file()}
        r = self.invoke()
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        calls = self.records()
        self.assertEqual([c[0] for c in calls], ['prepare', 'submit'])
        self.assertEqual(calls[-1][calls[-1].index('--nodes') + 1], '3090node2')
        for p, data in before.items():
            self.assertEqual((self.out / p).read_bytes(), data)
        self.assertEqual(r.stdout.count('[submitted]'), 1)

    def test_other_single_node_resume(self):
        r = self.invoke('resume', NODES='3090node3')
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertEqual([c[0] for c in self.records()], ['prepare', 'resume'])
        self.assertEqual(self.records()[-1][-1], '3090node3')

    def test_multi_node_and_hostlist_rejected_before_calls(self):
        for value in ('3090node2,3090node3', '3090node[2-3]', '3090node2,3090node2'):
            with self.subTest(value=value):
                r = self.invoke(NODES=value)
                self.assertEqual(r.returncode, 2, r.stderr + r.stdout)
                self.assertIn('Single job only', r.stdout)
                self.assertEqual(self.records(), [])

    def test_running_or_pending_account_job_blocks_submit(self):
        for state in ('RUNNING', 'PENDING', 'COMPLETING'):
            with self.subTest(state=state):
                r = self.invoke(FAKE_QUEUE=f'123 {state} existing_job\n')
                self.assertEqual(r.returncode, 2)
                self.assertIn('no new job submitted', r.stdout)
                self.assertEqual(self.records(), [])

    def test_failed_queue_query_fails_closed(self):
        r = self.invoke(FAKE_QUEUE_RC='1')
        self.assertEqual(r.returncode, 2)
        self.assertIn('Cannot confirm', r.stdout)
        self.assertEqual(self.records(), [])

    def test_prepare_never_submits(self):
        r = self.invoke('prepare', FAKE_QUEUE='123 RUNNING job\n')
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertEqual([c[0] for c in self.records()], ['prepare'])

    def test_missing_frozen_pointer_keeps_existing_data(self):
        (self.out / 'source.env').unlink()
        r = self.invoke('resume')
        self.assertEqual(r.returncode, 2)
        self.assertIn('source.env is missing', r.stdout)
        self.assertEqual(self.records(), [])
        self.assertTrue((self.out / 'formal/shards/00000/marker').is_file())

    def test_shell_syntax(self):
        subprocess.run(['bash', '-n', str(HERE / 'run.sh')], check=True)


    def scheduler_functions(self):
        wanted = {'parse_nodes', 'sbatch_args', 'ensure_account_idle', 'submit'}
        tree = ast.parse((HERE / 'run.py').read_text())
        def require(ok, message):
            if not ok: raise ValueError(message)
        def write(path, value, replace=False):
            path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        scope = dict(Path=Path, os=os, re=re, time=time, subprocess=subprocess,
                     HERE=HERE, require=require, read=lambda p: json.loads(Path(p).read_text()),
                     write=write, verify_job=lambda job: None, ensure_idle=lambda root: None)
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(HERE/'run.py'), 'exec'), scope)
        return scope

    def test_python_controller_rejects_two_nodes_before_scheduler(self):
        scope = self.scheduler_functions()
        with mock.patch.object(subprocess, 'run') as calls:
            with self.assertRaisesRegex(ValueError, 'Single job only'):
                scope['submit'](self.out, self.repo, '3090node2,3090node3', 'submit')
            calls.assert_not_called()

    def test_python_controller_calls_sbatch_exactly_once(self):
        scope = self.scheduler_functions()
        def scheduler(args, **kwargs):
            if args[0] == 'squeue':
                return subprocess.CompletedProcess(args, 0, '', '')
            self.assertEqual(args[0], 'sbatch')
            self.assertIn('--nodes=1', args)
            self.assertIn('--gres=gpu:3090:4', args)
            self.assertIn('--nodelist=3090node2', args)
            self.assertEqual(kwargs['env']['G25_SLOT_BASE'], '0')
            return subprocess.CompletedProcess(args, 0, '12345\n', '')
        with mock.patch.object(subprocess, 'run', side_effect=scheduler) as calls:
            scope['submit'](self.out, self.repo, '3090node2', 'submit')
            self.assertEqual(sum(c.args[0][0]=='sbatch' for c in calls.call_args_list), 1)
        record = json.loads((self.out/'latest_submission.json').read_text())
        self.assertEqual(len(record['jobs']), 1)
        self.assertEqual(record['jobs'][0]['id'], 12345)

    def test_python_controller_account_job_blocks_sbatch(self):
        scope = self.scheduler_functions()
        with mock.patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess(
                ['squeue'], 0, '7 PENDING another_job\n', '')) as calls:
            with self.assertRaisesRegex(ValueError, 'Account already'):
                scope['submit'](self.out, self.repo, '3090node2', 'resume')
            self.assertEqual(calls.call_count, 1)
            self.assertEqual(calls.call_args.args[0][0], 'squeue')


    def test_fresh_submit_keeps_single_node_default(self):
        shutil.rmtree(self.out)
        (self.src/'run.py').write_text(
            'import json, os, sys\n'
            'with open(os.environ["FAKE_CALLS"], "a") as f:\n'
            '    f.write(json.dumps(sys.argv[1:])+"\\n")\n'
            'if sys.argv[1]=="submit":\n'
            '    assert sys.argv[sys.argv.index("--nodes")+1]=="3090node2"\n'
            '    print("[submitted] fake=1")\n')
        (self.src/'protocol.json').write_text('{}\n')
        (self.src/'README_zh.md').write_text('fixture\n')
        subprocess.run(['git','-C',str(self.repo),'add','.'],check=True)
        subprocess.run(['git','-C',str(self.repo),'-c','user.name=Test',
                        '-c','user.email=test@example.invalid','commit','-qm','fresh fixture'],check=True)
        r=self.invoke()
        self.assertEqual(r.returncode,0,r.stderr+r.stdout)
        self.assertEqual([c[0] for c in self.records()],['prepare','submit'])
        self.assertEqual(r.stdout.count('[submitted]'),1)
        self.assertTrue((self.out/'source.env').is_file())

    def test_unprepared_resume_refuses_submission(self):
        shutil.rmtree(self.out)
        r=self.invoke('resume')
        self.assertEqual(r.returncode,2,r.stderr+r.stdout)
        self.assertIn('no prepared run',r.stdout)
        self.assertEqual(self.records(),[])


if __name__ == '__main__':
    unittest.main(verbosity=2)
