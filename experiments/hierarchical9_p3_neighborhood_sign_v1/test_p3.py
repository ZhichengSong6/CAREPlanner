#!/usr/bin/env python3
"""Synthetic math/partition/geometry/routing tests. Real-model smoke remains mandatory."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import p3_protocol as p3
import neighborhood as side
import update
old, p2 = p3.old, p3.p2


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(10,16),nn.Tanh(),nn.Linear(16,9))
    def forward(self,x):
        return self.net(x)


class Planes:
    # Synthetic visibility interval: positive normal probes may leave by opposite plane.
    def margins(self,x,s):
        p=torch.stack((x[:,3],.015-x[:,3]),1)
        return p.min(1)


def boundary(n=48, device='cpu'):
    g=torch.arange(n,device=device)%16
    x=torch.zeros(n,10,device=device)
    normal=torch.zeros(n,7,device=device);normal[:,0]=1
    return x,g//2,normal,g


def synthetic(n=48,device='cpu'):
    b=boundary(n,device)
    return b,side.build_queries(b,np.full(n,.02,np.float32),Planes(),
             torch.full((7,),-1.,device=device),torch.full((7,),1.,device=device))


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_stream_independent_partition_and_refresh(self):
        before=torch.get_rng_state().clone();state=np.random.get_state()
        a=side.radii_for_update(0,1,4096)
        b=np.concatenate([side.radii_for_update(0,1,4096,r,4) for r in range(4)])
        np.testing.assert_array_equal(a,b)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertTrue(np.array_equal(state[1],np.random.get_state()[1]))
        self.assertFalse(np.array_equal(a,side.radii_for_update(0,2,4096)))
        self.assertTrue(((a>=.005)&(a<=.05)).all())

    def test_actual_geometry_not_offset_sign(self):
        _,q=synthetic()
        self.assertTrue((q['labels']==-1).all()) # +.02 exited opposite plane
        self.assertTrue(q['valid'].all())
        self.assertEqual(int((q['labels']!=q['expected_side']).sum()),48)

    def test_no_clamp_relabel_or_resampling(self):
        b=boundary(16);b[0][:,3]=.99
        q=side.build_queries(b,np.full(16,.02),Planes(),torch.full((7,),-1.),torch.full((7,),1.))
        self.assertEqual(int(q['in_limits'].sum()),16)
        self.assertTrue((q['proposed'][1::2,3]>1).all())
        torch.testing.assert_close(q['inputs'][1::2,3],b[0][:,3]) # filler, NOT clamped training point
        self.assertFalse(q['valid'][1::2].any())

    def test_ambiguous_geometry_excluded(self):
        class Zero:
            def margins(self,x,s):return torch.zeros(len(x)),torch.zeros(len(x),dtype=torch.long)
        b=boundary(16);q=side.build_queries(b,np.full(16,.02),Zero(),torch.full((7,),-1.),torch.full((7,),1.))
        self.assertTrue(q['ambiguous'].all());self.assertFalse(q['valid'].any())
        m=Toy();loss,stats=side.sign_loss(m,q,side.valid_counts(q));loss.backward()
        self.assertEqual(float(loss.detach()),0.)
        self.assertTrue(all(p.grad is not None and not p.grad.any() for p in m.parameters()))
        self.assertEqual(side.summary(stats)['missing_strata'],list(range(32)))

    def test_loss_mapping_direction_and_saturation(self):
        class Head(nn.Module):
            def __init__(self):super().__init__();self.v=nn.Parameter(torch.zeros(9))
            def forward(self,x):return self.v[None].expand(len(x),-1)
        _,q=synthetic(16);m=Head();loss,st=side.sign_loss(m,q,side.valid_counts(q));loss.backward()
        self.assertEqual(float(m.v.grad[0]),0.)
        self.assertTrue((m.v.grad[1:]>0).all()) # negative labels push f downward
        m.v.data[1:]=-.1;m.zero_grad()
        loss,st=side.sign_loss(m,q,side.valid_counts(q));loss.backward()
        self.assertEqual(float(loss.detach()),0.)
        self.assertTrue((m.v.grad==0).all()) # no runaway magnitude for already satisfied inequality
        self.assertEqual(side.summary(st)['sign_accuracy'],1.)

    def test_matched_point_sensor_and_positive_labels(self):
        class Matched:
            def margins(self,x,s):return x[:,0]-.3,torch.zeros(len(x),dtype=torch.long)
        b=boundary(16);b[0][:8,0]=1.
        q=side.build_queries(b,np.full(16,.01),Matched(),torch.full((7,),-1.),torch.full((7,),1.))
        self.assertTrue((q['labels'][:16]==1).all());self.assertTrue((q['labels'][16:]==-1).all())
        self.assertTrue((q['sensors']==b[1].repeat_interleave(2)).all())

    def test_microbatch_global_denominators(self):
        _,q=synthetic(48);q['valid'][::3]=False
        counts=side.valid_counts(q);m=Toy();ref=deepcopy(m)
        a,_=side.sign_loss(m,q,counts);a.backward()
        stats=torch.zeros(32,len(side.COLUMNS),dtype=torch.float64)
        for begin in range(0,len(q['inputs']),11):
            loss,st=side.sign_loss(ref,side.slice_batch(q,begin,begin+11),counts)
            loss.backward();stats+=st
        for p,r in zip(m.parameters(),ref.parameters()):torch.testing.assert_close(p.grad,r.grad,atol=2e-7,rtol=2e-5)
        self.assertAlmostEqual(float(a.detach()),side.summary(stats)['loss'],places=6)

    def test_full_default_model_raw_q_and_parameter_backward(self):
        model=old.module('model',old.SCRATCH).HierarchicalVisibilityCDF()
        self.assertEqual(sum(p.numel() for p in model.parameters()),1133705)
        b,q=synthetic(16)
        loss,_=p2.calibration.boundary_loss(model,*b,torch.bincount(b[3],minlength=16).float(),p2.weights())
        sl,_=side.sign_loss(model,q,side.valid_counts(q))
        (loss+sl).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        inp=torch.randn(3,10,requires_grad=True);v=model(inp)[:,8]
        grad=torch.autograd.grad(v.sum(),inp)[0][:,3:]
        self.assertEqual(tuple(grad.shape),(3,7));self.assertTrue(torch.isfinite(grad).all())

    def test_pairwise_fk_margins_mixed_x(self):
        class Base:
            hfov=50.;vfov=66.;zmin=.2;zmax=.7;delta=.01;specs=list(range(8))
            def fk(self,s,q):
                t=torch.eye(4).repeat(len(q),1,1)
                t[:,0,3]=q[:,0]+s*.01
                return t
        b=Base();p=side.PairwiseFOV(b)
        inp=torch.zeros(3,10);inp[:,:3]=torch.tensor([[0.,0.,.4],[.6,0.,.4],[0.,0.,.1]])
        s=torch.tensor([0,1,7]);g,plane=p.margins(inp,s)
        import math
        ax=math.tan(math.radians(50)/2)
        ay=math.tan(math.radians(66)/2)
        expected=[]
        for i in range(3):
            px=float(inp[i,0]-s[i]*.01);pz=float(inp[i,2])
            vals=[(px+pz*ax)/math.sqrt(1+ax*ax),(-px+pz*ax)/math.sqrt(1+ax*ax),
                  pz*ay/math.sqrt(1+ay*ay),pz*ay/math.sqrt(1+ay*ay),pz-.2,.7-pz]
            expected.append(min(vals)-.01)
        torch.testing.assert_close(g,torch.tensor(expected),atol=1e-7,rtol=1e-6)
        self.assertGreater(float(g[0]),0);self.assertLess(float(g[1]),0);self.assertLess(float(g[2]),0)

    def test_p3_identity_and_config_gate(self):
        refs={'P0':'a','P1':'b','P2':'c'};streams=[{'global':'a'*64,'boundary':'b'*64} for _ in range(4)]
        control=dict(args={**p2.FIXED,'artifact_root':'/a','cache':'/c','arm':'P0','output':'/P0'},
                     source_sha256={'old':'same'},architecture={'layers':'unchanged'},output_layout={'sensor_slice':[1,9]},
                     sample_stream_sha256_by_rank=streams)
        mode='pilot';args=vars(p3.make_args(control,Path('/P3'),mode))
        cp=dict(format=p3.FORMAT,arm='P3',completed=True,mode=mode,pilot_updates=2000,step=2000,
            total_updates=52000,parent_updates=50000,parent_sha256=old.V1_SHA,out_dim=9,frozen_parameters=0,
            initialization='V1_weights_only_fresh_Adam',optimizer_policy='fresh_Adam_constant_lr_no_scheduler_no_clipping',
            boundary_weights=p2.WEIGHTS,neighborhood_config=p3.NEIGHBOR,cache_manifest_sha256='cache',reference_sha256=refs,
            source_sha256=control['source_sha256'],p2_source_sha256=p2.fingerprints(),p3_source_sha256=p3.fingerprints(),
            architecture=control['architecture'],output_layout=control['output_layout'],args=args,
            sample_stream_sha256_by_rank=streams,neighborhood_stream_sha256_by_rank=[{'proposal':'a'*64,'labeled':'b'*64}]*4,
            pair_sample_streams='MATCH')
        p3.assert_p3(cp,control,'cache',refs,True)
        for key,value in [('arm','P2'),('pair_sample_streams','PREFIX_ONLY'),('neighborhood_config',{}),('p2_source_sha256',{})]:
            bad=deepcopy(cp);bad[key]=value
            with self.assertRaises(ValueError):p3.assert_p3(bad,control,'cache',refs,True)
        bad=deepcopy(cp);bad['args']['lr']=1e-3
        with self.assertRaises(ValueError):p3.assert_p3(bad,control,'cache',refs,True)

    def test_hash_covers_exclusions_and_labels(self):
        _,q=synthetic(16)
        a,b=hashlib.sha256(),hashlib.sha256();side.hash_queries(a,b,np.arange(16),q)
        r={k:v.clone() for k,v in q.items()};r['valid'][0]=False
        c,d=hashlib.sha256(),hashlib.sha256();side.hash_queries(c,d,np.arange(16),r)
        self.assertEqual(a.hexdigest(),c.hexdigest());self.assertNotEqual(b.hexdigest(),d.hexdigest())


def ddp_check(backend):
    if backend=='nccl':
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']));device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    else:device=torch.device('cpu')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    dist.init_process_group(backend)
    try:
        rank,world=dist.get_rank(),dist.get_world_size()
        if world!=4:raise ValueError('Test requires four processes')
        obj=old.module('objective',old.SCRATCH);baseline=old.module('train',old.SCRATCH)
        torch.manual_seed(514)
        inputs=torch.randn(36,10,device=device);target=torch.randn(36,8,device=device)
        gradients=torch.randn(36,8,7,device=device);valid=torch.rand(36,8,device=device)>.2
        counts=obj.counts_from_mask(valid)
        uniform=(inputs,target,gradients,valid)
        b,nb=synthetic(48,device)
        b=(torch.randn(48,10,device=device)*.01,b[1],b[2],b[3])
        # Recreate neighborhoods from nonzero anchors; one rank has ZERO valid side rows.
        nb=side.build_queries(b,np.full(48,.02,np.float32),Planes(),torch.full((7,),-1.,device=device),torch.full((7,),1.,device=device))
        nb['valid'][:24]=False
        bc=torch.bincount(b[3],minlength=16).float();nc=side.valid_counts(nb)
        args=argparse.Namespace(amp='off',max_amp_retries=0,boundary_microbatch=5)
        maxgrad,maxparam=0.,0.
        for w in (0.,5.):
            config=replace(p3.NeighborhoodConfig(),loss_weight=w,microbatch=7)
            model=Toy().to(device);ref=deepcopy(model)
            ddp=DDP(model,device_ids=[device.index] if backend=='nccl' else None,broadcast_buffers=False)
            optim=torch.optim.Adam(ddp.parameters(),lr=1e-4);ropt=torch.optim.Adam(ref.parameters(),lr=1e-4)
            scaler=torch.cuda.amp.GradScaler(enabled=False)
            for alpha in (.002,1.):
                ropt.zero_grad(set_to_none=True)
                ul,_=obj.loss_for_microbatch(ref,*uniform,counts,obj.LossWeights(),world_size=1,training=True)
                bl,_=p2.calibration.boundary_loss(ref,*b,bc,p2.weights(),training=True)
                sl,_=side.sign_loss(ref,nb,nc,config=config)
                (ul+alpha*(bl+sl)).backward();ropt.step()
                start=rank*9;local=[tuple(v[start+i:min(start+i+5,start+9)] for v in uniform) for i in (0,5)]
                bs=rank*12;lb=tuple(v[bs:bs+12] for v in b)
                ln=side.slice_batch(nb,rank*24,(rank+1)*24)
                update.perform_update(ddp,optim,scaler,local,counts,lb,bc,ln,nc,obj.LossWeights(),p2.weights(),
                                      alpha,args,baseline,obj,config)
                for p,r in zip(ddp.module.parameters(),ref.parameters()):
                    maxgrad=max(maxgrad,float((p.grad-r.grad).abs().max()))
                    maxparam=max(maxparam,float((p-r).abs().max().detach()))
                    torch.testing.assert_close(p.grad,r.grad,atol=3e-6,rtol=3e-5)
                    torch.testing.assert_close(p,r,atol=1e-7,rtol=1e-5)
            del ddp
        t=torch.tensor([maxgrad,maxparam],device=device);dist.all_reduce(t,op=dist.ReduceOp.MAX)
        if rank==0:print(json.dumps(dict(status='PASS',test='P3 combined losses/Adam vs whole batch; side weight=0 and 5; empty-valid rank',
                                        backend=backend,max_abs_gradient_error=float(t[0]),max_abs_parameter_error=float(t[1]))),flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--ddp',action='store_true');ap.add_argument('--backend',choices=('gloo','nccl'),default='gloo')
    args,extra=ap.parse_known_args()
    if args.ddp:ddp_check(args.backend)
    else:unittest.main(argv=[sys.argv[0],*extra],verbosity=2)
