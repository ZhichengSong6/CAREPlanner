#!/usr/bin/env python3
"""P2 numerical/provenance/submission tests. Synthetic data is never model-quality evidence."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import copy
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

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import protocol as p2
from evaluate_p2 import boundary_stats, confusion, solve_stats
old=p2.old
cal=p2.calibration


def tiny(): return nn.Sequential(nn.Linear(10,16),nn.Tanh(),nn.Linear(16,9))


def batch(device='cpu',n=64):
    torch.manual_seed(123)
    inp=torch.randn(n,10,device=device)*.2
    groups=torch.arange(n,device=device)%16
    groups=groups[torch.randperm(n,device=device)]
    s=groups//2
    normals=F.normalize(torch.randn(n,7,device=device),dim=1)
    counts=torch.bincount(groups,minlength=16).float()
    return (inp,s,normals,groups),counts


def reference_controls():
    cp=dict(format=old.PILOT_FORMAT,completed=True,parent_updates=50000,
        parent_sha256=old.V1_SHA,pilot_updates=2000,total_updates=52000,cache_manifest_sha256='cache',
        out_dim=9,frozen_parameters=0,initialization='V1_weights_only_fresh_Adam',
        optimizer_policy='fresh_Adam_constant_lr_no_scheduler_no_clipping',source_sha256={'engine':'hash'},
        boundary_weights=p2.CONTROL_WEIGHTS,sample_stream_sha256_by_rank=[{'global':'g','boundary':'b'}]*4,
        architecture={'fixture':True},output_layout={'fixture':True},
        args={**p2.FIXED,'artifact_root':'/artifacts','cache':'/reference/cache'})
    c0,c1=copy.deepcopy(cp),copy.deepcopy(cp)
    for arm,c in (('P0',c0),('P1',c1)):
        c['arm']=arm;c['args'].update(arm=arm,output='/reference/'+arm)
    return c0,c1


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_only_boundary_eikonal_changes(self):
        self.assertEqual(asdict(p2.weights()),dict(zero=5.,normal=.1,eikonal=0.))
        self.assertEqual(asdict(cal.BoundaryWeights()),p2.CONTROL_WEIGHTS)
        obj=old.module('objective',old.SCRATCH)
        self.assertEqual(obj.LossWeights().eikonal,.01)

    def test_loss_and_parameter_gradient_exact_ablation(self):
        torch.manual_seed(1);m=tiny();b,c=batch()
        l1,st=cal.boundary_loss(m,*b,c,cal.BoundaryWeights())
        l2,st2=cal.boundary_loss(m,*b,c,p2.weights())
        eik_only,_=cal.boundary_loss(m,*b,c,cal.BoundaryWeights(zero=0,normal=0,eikonal=.1))
        torch.testing.assert_close(l1-l2,eik_only,rtol=2e-5,atol=1e-7)
        g1=torch.autograd.grad(l1,tuple(m.parameters()),retain_graph=True)
        g2=torch.autograd.grad(l2,tuple(m.parameters()),retain_graph=True)
        ge=torch.autograd.grad(eik_only,tuple(m.parameters()))
        for a,b,g in zip(g1,g2,ge):torch.testing.assert_close(a-b,g,rtol=2e-4,atol=2e-7)
        self.assertTrue(torch.equal(st,st2))

    def test_microbatch_equivalence(self):
        torch.manual_seed(3);m=tiny();other=copy.deepcopy(m);b,c=batch()
        cal.boundary_loss(m,*b,c,p2.weights())[0].backward()
        for i in range(0,64,7):cal.boundary_loss(other,*(v[i:i+7] for v in b),c,p2.weights())[0].backward()
        for a,b in zip(m.parameters(),other.parameters()):torch.testing.assert_close(a.grad,b.grad,atol=3e-7,rtol=1e-4)

    def test_selected_sensor_not_union(self):
        m=nn.Linear(10,9);b,c=batch();b=(b[0],torch.full_like(b[1],7),b[2],torch.full_like(b[3],14))
        cal.boundary_loss(m,*b,c,p2.weights())[0].backward()
        self.assertEqual(float(m.weight.grad[:8].abs().max()),0.)
        self.assertGreater(float(m.weight.grad[8].norm()),0.)

    def test_default_model_raw_q_backward(self):
        m=old.module('model',old.SCRATCH).HierarchicalVisibilityCDF();b,c=batch(n=16)
        cal.boundary_loss(m,*b,c,p2.weights())[0].backward()
        self.assertEqual(sum(p.numel() for p in m.parameters()),1133705)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_ramp_and_exact_config(self):
        c0,_=reference_controls()
        a=p2.make_args(c0,Path('/new/P2'),'pilot')
        self.assertEqual(a.steps,2000);self.assertEqual(a.boundary_count,4096)
        for key in c0['args']:
            if key not in ('arm','output'):self.assertEqual(getattr(a,key),c0['args'][key])
        self.assertEqual([p2.alpha(k) for k in (1,100,500,2000)],[.002,.2,1.,1.])
        self.assertEqual(p2.make_args(c0,Path('/new/smoke'),'smoke').boundary_count,4096)

    def test_reference_gates(self):
        c0,c1=reference_controls();p2.validate_controls(c0,c1,'cache',{'engine':'hash'})
        for mutate in ('source','streams','weights','config','complete'):
            b=copy.deepcopy(c1)
            if mutate=='source':b['source_sha256']={}
            elif mutate=='streams':b['sample_stream_sha256_by_rank'][0]={'global':'bad','boundary':'b'}
            elif mutate=='weights':b['boundary_weights']={**p2.CONTROL_WEIGHTS,'zero':10}
            elif mutate=='config':b['args']['lr']=.001
            else:b['completed']=False
            with self.assertRaises(ValueError):p2.validate_controls(c0,b,'cache',{'engine':'hash'})

    def test_p2_identity_and_stream_gates(self):
        c0,_=reference_controls();references={'P0':'a','P1':'b'}
        cp=copy.deepcopy(c0)
        cp.update(format=p2.FORMAT,arm='P2',mode='pilot',boundary_weights=p2.WEIGHTS,
            reference_sha256=references,p2_source_sha256=p2.fingerprints(),pair_sample_streams='MATCH',
            args=vars(p2.make_args(c0,Path('/new/P2'),'pilot')))
        p2.assert_p2(cp,c0,'cache',references,require_pilot=True)
        for key,value in (('boundary_weights',p2.CONTROL_WEIGHTS),('p2_source_sha256',{}),('pair_sample_streams','MISMATCH')):
            bad=copy.deepcopy(cp);bad[key]=value
            with self.assertRaises(ValueError):p2.assert_p2(bad,c0,'cache',references,require_pilot=True)
        cp.update(mode='smoke',pilot_updates=2,total_updates=50002,pair_sample_streams='NOT_COMPARABLE_2_VS_2000',
                  args=vars(p2.make_args(c0,Path('/new/smoke'),'smoke')))
        p2.assert_p2(cp,c0,'cache',references,require_pilot=False)
        with self.assertRaises(ValueError):p2.assert_p2(cp,c0,'cache',references,require_pilot=True)

    def test_scale_invariant_shift_diagnostics(self):
        a=dict(value=np.array([1.,-1.,2.]),norm=np.array([2.,2.,2.]),cosine=np.array([1.,-1.,0.]),
               normal_slope=np.array([2.,-2.,0.]),linearized_zero_shift_rad=np.array([-.5,-.5,np.nan]))
        r=boundary_stats(a,np.arange(3))
        self.assertEqual(r['linearization_defined']['passed'],2)
        self.assertEqual(r['nonpositive_normal_slope']['passed'],2)
        self.assertEqual(r['abs_linearized_zero_shift_rad']['p50'],.5)
        self.assertEqual(r['well_conditioned_linearization']['passed'],2)
        self.assertIsNone(boundary_stats(a,np.array([],dtype=int))['abs_value']['mean'])

    def test_side_confusion_uses_real_g_not_offset(self):
        r=confusion([1,-1,-1,1],[1,-1,1,-1])
        self.assertEqual([r[k] for k in ('true_positive','true_negative','false_positive','false_negative')],[1,1,1,1])
        self.assertIsNone(confusion([-1],[-1])['positive_recall']['rate'])

    def test_failure_stage_cross_table(self):
        a=dict(fov_pass=True,predicted_root_within_002=False,solver_ms=1.,failure_stage='PASS',root_source='initial_branch_positive')
        b=dict(a,fov_pass=False,failure_stage='LEARNED_CANDIDATE_BUT_FOV_FAIL')
        r=solve_stats([{'models':{'V1':a,'P0':a,'P1':b,'P2':b}}],['V1','P0','P1','P2'])
        self.assertEqual(r['paired']['P0_vs_P2']['P0_only'],1)
        self.assertEqual(r['models']['P2']['root_source_by_failure']['initial_branch_positive / LEARNED_CANDIDATE_BUT_FOV_FAIL'],1)

    def test_one_job_submission_and_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);repo=root/'repo';exp=repo/'experiments'/p2.HERE.name
            exp.mkdir(parents=True)
            for name in ('submit.sh','worker.sbatch'):
                shutil.copy(p2.HERE/name,exp/name)
            sums=''.join(old.sha256(exp/n)+'  '+str((exp/n).relative_to(repo))+'\n' for n in ('submit.sh','worker.sbatch'))
            (exp/'SHA256SUMS').write_text(sums)
            for cmd in (['git','init','-q'],['git','add','.'],['git','-c','user.name=fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture']):
                subprocess.run(cmd,cwd=repo,check=True,capture_output=True)
            ref=root/'ref';(ref/'cache').mkdir(parents=True)
            (ref/'cache/manifest.json').write_text(json.dumps(dict(status='COMPLETE',args=dict(train_points=512,train_anchors=4,val_points=64,val_anchors=2))))
            for name in ('train.npz','val.npz'):(ref/'cache'/name).write_text('synthetic only')
            for arm in ('P0','P1'):
                (ref/arm).mkdir();(ref/arm/'final.pt').write_text('synthetic only')
                (ref/arm/'run.json').write_text(json.dumps(dict(status='COMPLETE',args=dict(steps=2000))))
            bindir=root/'bin';bindir.mkdir()
            mock=bindir/'sbatch'
            mock.write_text('#!/usr/bin/env python3\nimport os,sys,json\nwith open(os.environ["CALLS"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\nif os.environ.get("REJECT")=="1":sys.exit(1)\nprint("12345")\n')
            mock.chmod(0o755)
            env={**os.environ,'PATH':str(bindir)+os.pathsep+os.environ['PATH'],'P2_REFERENCE_ROOT':str(ref),'CALLS':str(root/'calls'),'REJECT':'1'}
            cmd=['bash',str(exp/'submit.sh'),'smoke']
            bad=subprocess.run(cmd,env=env,capture_output=True,text=True)
            self.assertNotEqual(bad.returncode,0);self.assertFalse((ref/'.p2_smoke_submission').exists())
            env['REJECT']='0';good=subprocess.run(cmd,env=env,capture_output=True,text=True)
            self.assertEqual(good.returncode,0,good.stdout+good.stderr)
            calls=[json.loads(line) for line in (root/'calls').read_text().splitlines()]
            self.assertEqual(len(calls),2)
            self.assertTrue(any(a=='--gres=gpu:3090:4' for a in calls[-1]))
            self.assertFalse(any(a.startswith(('--array','--dependency','--mem')) for a in calls[-1]))
            outputs=[a.split('=',1)[1] for a in calls[-1] if a.startswith('--output=')]
            self.assertTrue(Path(outputs[0]).is_absolute())
            duplicate=subprocess.run(cmd,env=env,capture_output=True)
            self.assertNotEqual(duplicate.returncode,0)
            self.assertEqual(len((root/'calls').read_text().splitlines()),2)


def ddp_test(backend):
    torch.set_num_threads(2)
    if backend=='nccl':
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']));device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    else:device=torch.device('cpu')
    dist.init_process_group(backend)
    try:
        world,rank=dist.get_world_size(),dist.get_rank()
        if world!=4:raise ValueError('Expected four ranks')
        engine=old.module('train_pilot',p2.LEGACY)
        baseline=old.module('train',old.SCRATCH);obj=old.module('objective',old.SCRATCH)
        b,bc=batch(device)
        torch.manual_seed(71)
        inp=torch.randn(32,10,device=device)*.2
        target=torch.randn(32,8,device=device)*.1
        tg=F.normalize(torch.randn(32,8,7,device=device),dim=-1)
        mask=torch.ones(32,8,dtype=torch.bool,device=device);mask[::3,2:4]=False
        uniform=(inp,target,tg,mask);counts=obj.counts_from_mask(mask)
        torch.manual_seed(23);m=tiny().to(device);ref=copy.deepcopy(m)
        ddp=DDP(m,device_ids=[device.index] if backend=='nccl' else None,broadcast_buffers=False,find_unused_parameters=False)
        optimizer=torch.optim.Adam(ddp.parameters(),lr=1e-4)
        optref=torch.optim.Adam(ref.parameters(),lr=1e-4)
        scaler=torch.cuda.amp.GradScaler(enabled=False)
        args=SimpleNamespace(max_amp_retries=0,boundary_microbatch=5,amp='off')
        maximum=0.;param_max=0.
        for alpha in (.002,1.):
            u=[tuple(v[rank*8+i:rank*8+min(i+3,8)] for v in uniform) for i in (0,3,6)]
            localb=tuple(v[rank*16:(rank+1)*16] for v in b)
            engine.perform_update(ddp,optimizer,scaler,u,counts,localb,bc,obj.LossWeights(),p2.weights(),alpha,args,baseline,obj)
            optref.zero_grad(set_to_none=True)
            gl,_=obj.loss_for_microbatch(ref,*uniform,counts,obj.LossWeights(),training=True)
            bl,_=cal.boundary_loss(ref,*b,bc,p2.weights())
            (gl+alpha*bl).backward()
            for a,z in zip(m.parameters(),ref.parameters()):
                maximum=max(maximum,float((a.grad-z.grad).abs().max()))
                torch.testing.assert_close(a.grad,z.grad,atol=4e-6,rtol=3e-4)
            optref.step()
            for a,z in zip(m.parameters(),ref.parameters()):
                param_max=max(param_max,float((a-z).detach().abs().max()))
                torch.testing.assert_close(a,z,atol=2e-6,rtol=1e-4)
        if rank==0:print(json.dumps(dict(status='PASS',test='P2 old-engine DDP gradients and Adam match full batch',backend=backend,max_abs_gradient_error=maximum,max_abs_parameter_error=param_max)),flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ddp',action='store_true');p.add_argument('--backend',choices=('gloo','nccl'),default='gloo')
    args=p.parse_args()
    if args.ddp:ddp_test(args.backend)
    else:unittest.main(argv=[sys.argv[0]],verbosity=2)
