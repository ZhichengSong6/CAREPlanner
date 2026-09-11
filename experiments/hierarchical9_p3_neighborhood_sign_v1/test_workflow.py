#!/usr/bin/env python3
"""Submission/archive/report routing fixtures; never submit real Slurm jobs."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import torch

import p3_protocol as p3
from pack_reports import pack
import evaluate_p3 as evaluator

HERE=Path(__file__).resolve().parent


class Workflows(unittest.TestCase):
    def test_single_submission_quota_rejection_duplicate(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);repo=base/'repo';root=base/'refs';bindir=base/'bin'
            target=repo/'experiments'/HERE.name
            target.mkdir(parents=True);root.mkdir();bindir.mkdir()
            for name in ('submit.sh','worker.sbatch'):
                shutil.copy2(HERE/name,target/name)
            (target/'SHA256SUMS').write_text('\n'.join(hashlib.sha256((target/n).read_bytes()).hexdigest()+'  experiments/'+HERE.name+'/'+n
                                         for n in ('submit.sh','worker.sbatch'))+'\n')
            for cmd in (['git','init','-q'],['git','add','.'],['git','-c','user.name=test','-c','user.email=test@example.invalid','commit','-qm','fixture']):
                subprocess.run(cmd,cwd=repo,check=True,capture_output=True)
            (root/'cache').mkdir()
            (root/'cache/manifest.json').write_text(json.dumps(dict(status='COMPLETE',args=dict(train_points=512,train_anchors=4,val_points=64,val_anchors=2))))
            for n in ('train.npz','val.npz'):(root/'cache'/n).write_text('fixture')
            for arm in ('P0','P1','P2'):
                (root/arm).mkdir();(root/arm/'final.pt').write_text('fixture; not a torch checkpoint')
                (root/arm/'run.json').write_text(json.dumps(dict(status='COMPLETE',args=dict(steps=2000))))
            fake=bindir/'sbatch';fake.write_text('#!/usr/bin/env python3\nimport os,sys,json\n'
                'with open(os.environ["CALLS"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n'
                'if os.environ.get("REJECT")=="1":print("AssocMaxSubmitJobLimit",file=sys.stderr);sys.exit(1)\n'
                'print("12345")\n');fake.chmod(0o755)
            env={**os.environ,'PATH':str(bindir)+os.pathsep+os.environ['PATH'],'P3_REFERENCE_ROOT':str(root),'CALLS':str(base/'calls')}
            command=['bash',str(target/'submit.sh'),'smoke']
            r=subprocess.run(command,cwd=repo,env={**env,'REJECT':'1'},capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0);self.assertFalse((root/'.p3_smoke_submission').exists())
            r=subprocess.run(command,cwd=repo,env=env,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            calls=[json.loads(x) for x in (base/'calls').read_text().splitlines()]
            self.assertEqual(len(calls),2)
            args=calls[-1]
            self.assertIn('--gres=gpu:3090:4',args)
            self.assertTrue(any(x=='--output='+str(root)+'/logs/p3_smoke_%j.out' for x in args))
            self.assertFalse(any(x.startswith(('--array','--dependency','--mem')) for x in args))
            self.assertIn('P3_SMOKE_JOB=12345',(root/'p3_jobs.env').read_text())
            r=subprocess.run(command,cwd=repo,env=env,capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0)
            self.assertEqual(len((base/'calls').read_text().splitlines()),2)
            # Missing smoke completion prevents pilot before another sbatch.
            r=subprocess.run(['bash',str(target/'submit.sh'),'pilot'],cwd=repo,env=env,capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0)
            self.assertEqual(len((base/'calls').read_text().splitlines()),2)

    def test_zip_only_reports_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'evaluation_p3').mkdir();(root/'P3').mkdir()
            for n in ('summary.md','report.json'):(root/'evaluation_p3'/n).write_text('fixture')
            (root/'evaluation_p3/manifest.json').write_text(json.dumps(dict(status='COMPLETE',mode='pilot',pilot_updates=2000)))
            for n in ('validation.jsonl','run.json','metrics.jsonl'):(root/'P3'/n).write_text('fixture')
            (root/'P3/final.pt').write_text('DO NOT PACKAGE')
            out=pack(root)
            with ZipFile(out) as z:
                self.assertIsNone(z.testzip());self.assertEqual(len(z.namelist()),7)
                self.assertFalse(any(n.endswith('.pt') for n in z.namelist()))
            with self.assertRaises(FileExistsError):pack(root)
            (root/'P3/validation.jsonl').unlink()
            with self.assertRaises(FileNotFoundError):pack(root)

    def test_evaluator_end_to_end_explicit_synthetic_fixtures(self):
        # This checks CLI routing/output, NOT the real robot/FOV/checkpoint libraries.
        def fraction(k,n):return dict(passed=int(k),count=int(n),rate=k/n if n else None)
        def distribution(x):return dict(mean=float(np.mean(x)),count=len(x))
        def selected(model,x,q,s,normal,device):
            value=np.asarray(q)[:,0]+model
            return dict(value=value,norm=np.ones(len(q)),cosine=np.ones(len(q)),normal_slope=np.ones(len(q)),linearized_zero_shift_rad=-value)
        def bstats(a,ids):return dict(abs_value=distribution(np.abs(a['value'][ids])),normal_cosine=distribution(a['cosine'][ids]))
        def confusion(v,g):return dict(sign_accuracy=fraction(np.sum((v>=0)==(g>=0)),len(g)))
        n=16;s=np.repeat(np.arange(8),2);kind=np.tile(np.arange(2),8)
        normal=np.zeros((n,7),np.float32);normal[:,0]=1
        d=dict(s=s,kind=kind,x_index=np.arange(n),x=np.zeros((n,3),np.float32),q=np.zeros((n,7),np.float32),normal=normal)
        class Cache:
            identity='synthetic-cache';lo=np.full(7,-1,np.float32);hi=np.ones(7,np.float32)
            manifest={'urdf_sha256':'fake'};arrays={'val':d};groups={'val':[np.array([i]) for i in range(16)]}
            def verify_dataset(self,*a):pass
            def tensors(self,split,ids,device):
                return (torch.tensor(np.concatenate((d['x'][ids],d['q'][ids]),1)),torch.tensor(s[ids]),
                        torch.tensor(normal[ids]),torch.tensor(2*s[ids]+kind[ids]))
        class Dataset:
            x_cpu=torch.tensor(d['x'])
            def q_limits(self,device):return torch.tensor(Cache.lo),torch.tensor(Cache.hi)
            def sensor_masks(self,device):return torch.ones(8,7)
        class Oracle:
            def __init__(self,*a):pass
            def value(self,x,q,s):return float(q[0])
        def solve(*a):
            return dict(fov_pass=True,predicted_root_within_002=True,solver_ms=1.,
                        failure_stage='FOV_PASS_NOT_EXECUTION_CERTIFIED',root_source='fixture')
        stat=SimpleNamespace(fraction=fraction,finite_dist=distribution,boundary_stats=bstats,confusion=confusion)
        def planning(*args):
            return {n+'/union':dict(count=4,proj_oracle_boundary_030=1.,asc1_g_ge_0p03=1.,asc10_g_ge_0p03=1.) for n in args[0]}
        ev=SimpleNamespace(SensorView=lambda v:v,selected=selected,field_sentinel=lambda *a:{},planning_sentinel=planning)
        api=SimpleNamespace(VisibilityQ0Dataset=lambda *a:Dataset(),DEFAULT_JOINT_NAMES=[],DEFAULT_SENSOR_FRAMES=[],PinocchioFOVOracle=Oracle)
        modules=dict(evaluate_pair=ev,evaluate_p2=stat,train_signed_visibility_cdf_pairwise_replace=api,
                     oracle=SimpleNamespace(SensorOracle=Oracle),core=SimpleNamespace(within=lambda q,l,h:bool(((q>=l)&(q<=h)).all()),json_safe=lambda x:x),
                     runtime_probe=SimpleNamespace(make_probe=lambda *a:None,run_probe=solve),audit=SimpleNamespace(preflight=lambda *a:{}))
        # Avoid mixing a fake scalar model with the real sign-loss evaluator.
        def fake_sign(*a,**kw):
            st=torch.zeros(32,12,dtype=torch.float64);return torch.tensor(0.),st
        class Pair:
            def __init__(self,*a):pass
            def verify(self,*a):return {'status':'SYNTHETIC'}
            def margins(self,x,s):return x[:,3],torch.zeros(len(x),dtype=torch.long)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            controls={n:{'args':{'artifact_root':td,'seed':0}} for n in ('P0','P1','P2')}
            cp=dict(mode='pilot',pilot_updates=2000,pair_sample_streams='MATCH')
            with patch.object(p3,'load_references',return_value=(Cache(),controls,{})), \
                 patch.object(p3.p2,'load_saved',return_value=(cp,'fake')), \
                 patch.object(p3,'assert_p3'),patch.object(p3.p2,'model_from',return_value=0.), \
                 patch.object(p3.old,'load_v1',return_value=(0.,{})), \
                 patch.object(p3.old,'module',side_effect=lambda name,path:modules[name]), \
                 patch.object(p3.old,'sha256',return_value='fake'), \
                 patch.object(evaluator.side,'PairwiseFOV',Pair), \
                 patch.object(evaluator.side,'sign_loss',side_effect=fake_sign), \
                 patch.object(sys,'argv',['evaluate_p3.py','--reference-root',td,'--mode','pilot','--device','cpu']):
                evaluator.main()
            out=root/'evaluation_p3'
            self.assertEqual(json.loads((out/'manifest.json').read_text())['evaluated_models'],['V1','P0','P1','P2','P3'])
            report=json.loads((out/'report.json').read_text())
            self.assertEqual(report['status'],'COMPLETE')
            self.assertIn('P2_vs_P3',next(iter(report['solves'].values()))['paired'])
            self.assertTrue((out/'summary.md').is_file());self.assertTrue((out/'boundary_samples.npz').is_file())


if __name__=='__main__':unittest.main(verbosity=2)
