"""CPU-only recovery tests: no GPUs, no real scheduler and no user data writes."""
from __future__ import annotations
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import resource_check as rc

HERE = Path(__file__).resolve().parent
BRANCH = 'mainline-b/h9-scratch50k-r012-v1'


def inventory(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}


def fixture(root):
    code = root / 'sources/frozen_original'
    code.mkdir(parents=True)
    out = root / 'formal'
    (out / 'dependencies').mkdir(parents=True)
    # Small stand-in with the same verification API, not the real trainer.
    (code / 'k2_common.py').write_text('''from pathlib import Path
import json,hashlib
ARMS=('A_continue','B_keep')
def verify(job):
    actual=hashlib.sha256((Path(__file__).parent/'train.py').read_bytes()).hexdigest()
    if actual!=job['test_train_sha']:raise ValueError('Frozen new source changed')
def stage(folder):
    root=Path(folder);m=json.loads((root/'complete.json').read_text())
    if m['status']!='COMPLETE':raise ValueError('Incomplete stage')
    for name,digest in m['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Corrupt stage')
    return m
''')
    (code / 'train.py').write_text('# frozen original training fixture\n')
    (code / 'run.py').write_text('# original coordinator\n')
    (code / 'run.sh').write_text('#!/bin/bash\nexit 0\n')
    job = {'out': str(out), 'deps': str(out/'dependencies'),
           'test_train_sha': hashlib.sha256((code/'train.py').read_bytes()).hexdigest()}
    (out/'job.json').write_text(json.dumps(job))
    for name in ['reference_inputs'] + [f'reference_mined/rank{i}' for i in range(4)] + ['replay']:
        folder = out/name
        folder.mkdir(parents=True)
        (folder/'data.txt').write_text('fixed data\n')
        (folder/'complete.json').write_text(json.dumps({'status': 'COMPLETE', 'files': {
            'data.txt': hashlib.sha256((folder/'data.txt').read_bytes()).hexdigest()}}))
    (out/'A_continue').mkdir()
    (root/'source.env').write_text(f'export CODE={code}\n')
    (root/'latest.env').write_text(f'export JOB=18150\nexport LOG={root}/old.log\n')
    return code, out


class RecoveryTests(unittest.TestCase):
    def test_admission_accepts_four_clean_cards(self):
        rows = [{'free_bytes': 22*rc.GIB, 'total_bytes': 24*rc.GIB} for _ in range(4)]
        self.assertTrue(rc.assess_memory(rows)['ok'])

    def test_uploaded_oom_pattern_is_rejected_before_training(self):
        rows = [{'free_bytes': int(mib*1024**2), 'total_bytes': int(23.56*rc.GIB)}
                for mib in (156.44, 156.12, 156.44, 70.56)]
        result = rc.assess_memory(rows)
        self.assertFalse(result['ok'])
        self.assertEqual(result['insufficient_logical_devices'], [0, 1, 2, 3])

    def test_device_count_threshold_and_bad_numbers(self):
        row = {'free_bytes':20*rc.GIB, 'total_bytes':24*rc.GIB}
        self.assertTrue(rc.assess_memory([row]*4)['ok'])
        with self.assertRaises(ValueError):
            rc.assess_memory([row]*3)
        rows = [row.copy() for _ in range(4)]
        rows[1]['free_bytes'] -= 1
        self.assertEqual(rc.assess_memory(rows)['insufficient_logical_devices'], [1])
        rows[1]['free_bytes'] = -1
        with self.assertRaises(ValueError):
            rc.assess_memory(rows)
        with self.assertRaises(ValueError):
            rc.assess_memory([row]*4, float('nan'))

    def test_validate_preserves_original_source_and_replay(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);code,out=fixture(root);before=inventory(root)
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                rc.validate_existing(root,code)
            self.assertEqual(before,inventory(root))
            self.assertIn('COMPLETE_HASH_VERIFIED_REUSE',captured.getvalue())
            self.assertIn('NO_SAVED_UPDATE_START_FIXED_TEACHER',captured.getvalue())

    def test_corrupt_original_source_or_replay_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);code,out=fixture(root)
            p=code/'train.py';old=p.read_text();p.write_text('changed')
            with self.assertRaisesRegex(ValueError,'Frozen new source changed'):
                rc.validate_existing(root,code)
            p.write_text(old)
            (out/'replay/data.txt').write_text('tampered')
            with self.assertRaisesRegex(ValueError,'Corrupt stage'):
                rc.validate_existing(root,code)

    def test_completed_training_or_bad_source_path_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);code,out=fixture(root)
            (out/'training_complete.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Training already complete'):
                rc.validate_existing(root,code)
            with self.assertRaisesRegex(ValueError,'original content-addressed'):
                rc.validate_existing(root,out)

    def test_gpu_check_cannot_run_on_login_node(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,'inside the allocated Slurm job'):
                rc.gpu_check()

    def test_shell_syntax_spool_and_no_training_changes(self):
        subprocess.run(['bash','-n',str(HERE/'run.sh')],check=True)
        text=(HERE/'run.sh').read_text()
        self.assertLess(text.index('== _worker'),text.index('HERE='))
        self.assertIn('--exclusive --gres=gpu:3090:4',text)
        self.assertIn('exec bash "$K2_CODE/run.sh" _worker',text)
        self.assertIn('export K2_ACTION=resume',text)
        self.assertNotIn('nvidia-smi --gpu-reset',text)
        self.assertNotIn('kill -9',text)

    def test_sandbox_scheduler_submit_resume_and_no_duplicate(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);repo=folder/'repo';repo.mkdir();root=folder/'outputs';root.mkdir()
            code,out=fixture(root)
            d=repo/'experiments/recovery';d.mkdir(parents=True)
            for name in ('run.sh','resource_check.py'):
                shutil.copyfile(HERE/name,d/name)
            # Avoid recursive invocation of this integration test from its runner.
            (d/'test_minimal.py').write_text('import unittest\nclass T(unittest.TestCase):\n def test_ok(self):self.assertTrue(True)\n')
            for cmd in (['git','init','-q','-b',BRANCH],['git','config','user.name','Test'],
                        ['git','config','user.email','test@example.invalid'],['git','add','.'],
                        ['git','commit','-qm','test fixture']):
                subprocess.run(cmd,cwd=repo,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            binary=folder/'bin';binary.mkdir();calls=folder/'sbatch_calls.jsonl'
            scripts={
                'squeue':'#!/bin/bash\nif [[ "${TEST_ACTIVE:-0}" == 1 ]]; then echo RUNNING; fi\n',
                'sacct':'#!/bin/bash\necho "${TEST_PREVIOUS:-18150}|${TEST_STATE:-FAILED}|"\n',
                'sbatch':'#!/usr/bin/env python3\nimport json,os,sys\nwith open(os.environ["TEST_CALLS"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\nprint("18200")\n'}
            for name,text in scripts.items():
                p=binary/name;p.write_text(text);p.chmod(0o755)
            env=dict(os.environ,PATH=str(binary)+os.pathsep+os.environ['PATH'],
                     K2_ROOT=str(root),VIS_PYTHON=os.sys.executable,TEST_CALLS=str(calls),
                     PYTHONDONTWRITEBYTECODE='1')
            env.pop('NODE',None)
            original=inventory(code);original_data=inventory(out)
            p=subprocess.run(['bash',str(d/'run.sh'),'resume'],env=env,capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            self.assertIn('action=RESUME_EXISTING',p.stdout)
            args=json.loads(calls.read_text().splitlines()[0])
            self.assertIn('--exclusive',args)
            self.assertIn('--gres=gpu:3090:4',args)
            self.assertEqual(args[-1],'_worker')
            self.assertIn('export JOB=18200',(root/'latest.env').read_text())
            self.assertEqual(inventory(code),original)
            self.assertEqual(inventory(out),original_data)
            # A second launch with active job must refuse, not create another task.
            env.update(TEST_ACTIVE='1',TEST_PREVIOUS='18200')
            p=subprocess.run(['bash',str(d/'run.sh'),'resume'],env=env,capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)
            self.assertIn('still RUNNING',p.stdout)
            self.assertEqual(len(calls.read_text().splitlines()),1)


if __name__=='__main__':
    unittest.main()
