#!/usr/bin/env python3
"""CPU numerical tests; optional actual four-rank Gloo/NCCL update-equivalence test."""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from common import (BoundaryCache, CACHE_FORMAT, V1_SHA, SCRATCH, AUDIT, SCRIPTS, URDF,
                    array_hash, module, setup_paths, sha256, validate_array_split, write_json, seed_for_update)
from calibration import BoundaryWeights, boundary_loss, ramp_weight, summary

REQUIRE_REPO = False


class TinyNine(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(30, 24), nn.Tanh())
        self.heads = nn.ModuleList([nn.Linear(24,1) for _ in range(9)])
    def forward(self,x):
        h=self.shared(torch.cat([x,x.sin(),x.cos()],1))
        return torch.cat([head(h) for head in self.heads],1)


def batch(device='cpu', n=64):
    gen=torch.Generator().manual_seed(417)
    inp=torch.randn(n,10,generator=gen)*.3
    groups=torch.arange(n)%16
    sensors=groups//2
    normals=torch.randn(n,7,generator=gen); normals=normals/normals.norm(dim=1,keepdim=True)
    return tuple(v.to(device) for v in (inp,sensors,normals,groups))


def fake_cache(path):
    data={'x':np.zeros((32,3),np.float32),'q':np.zeros((32,7),np.float32),
          'normal':np.tile(np.array([1,0,0,0,0,0,0],np.float32),(32,1)),
          's':np.arange(32,dtype=np.int64)//4,'kind':np.tile(np.array([0,0,1,1],np.int64),8),
          'x_index':np.arange(32,dtype=np.int64)%16,'g':np.zeros(32,np.float32),
          'plane':np.zeros(32,np.int64),'bank_gap':np.zeros(32,np.float32)}
    np.savez_compressed(path/'train.npz',**data)
    val={k:v.copy() for k,v in data.items()}; val['x_index']+=100
    np.savez_compressed(path/'val.npz',**val)
    meta={'format':CACHE_FORMAT,'status':'COMPLETE','parent_sha256':V1_SHA,
          'sensor_masks':np.ones((8,7)).tolist(),'q_min':[-2.]*7,'q_max':[2.]*7,
          'split':{'train_indices':list(range(16)),'val_indices':list(range(100,1100))},
          'files':{n:sha256(path/n) for n in ('train.npz','val.npz')}}
    write_json(path/'manifest.json',meta)
    return data,meta


class PilotTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(0)

    def test_exact_head_mapping_not_union_zero(self):
        class Plane(nn.Module):
            def __init__(self):
                super().__init__(); self.bias=nn.Parameter(torch.tensor(0.))
            def forward(self,x):
                return torch.cat([1000+x[:,3:4],(x[:,3:4]+self.bias).expand(-1,8)],1)
        model=Plane(); inp=torch.zeros(32,10)
        sensors=torch.arange(32)//4; groups=2*sensors+torch.arange(32)%2
        normal=torch.zeros(32,7);normal[:,0]=1
        loss,stats=boundary_loss(model,inp,sensors,normal,groups,torch.full((16,),2.),BoundaryWeights())
        self.assertAlmostEqual(float(loss.detach()),0.,places=7)
        loss.backward(); self.assertEqual(float(model.bias.grad),0.)
        self.assertEqual(int(stats[:,0].sum()),32)

    def test_microbatch_matches_whole(self):
        a=TinyNine(); b=copy.deepcopy(a); data=batch(); denom=torch.full((16,),4.)
        loss,stats=boundary_loss(a,*data,denom,BoundaryWeights());loss.backward()
        accumulated=torch.zeros_like(stats)
        for begin in range(0,64,11):
            l,st=boundary_loss(b,*(t[begin:begin+11] for t in data),denom,BoundaryWeights());l.backward();accumulated+=st
        for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x.grad,y.grad,rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(stats,accumulated,rtol=2e-6,atol=2e-6)

    def test_p0_zero_multiplier_no_parameter_signal(self):
        model=TinyNine();loss,_=boundary_loss(model,*batch(),torch.full((16,),4.),BoundaryWeights())
        (loss*0).backward()
        self.assertTrue(all(p.grad is not None and not p.grad.any() for p in model.parameters()))

    def test_p1_all_sensor_and_shared_gradients(self):
        model=TinyNine();loss,_=boundary_loss(model,*batch(),torch.full((16,),4.),BoundaryWeights());loss.backward()
        self.assertGreater(float(model.shared[0].weight.grad.norm()),0)
        for head in model.heads[1:]:self.assertGreater(float(head.weight.grad.norm()),0)
        self.assertEqual(float(model.heads[0].weight.grad.norm()),0) # no direct union boundary supervision

    def test_inactive_components_are_not_hidden(self):
        class Plane(nn.Module):
            def forward(self,x):return (x[:,3:4]+3*x[:,9:10]).expand(-1,9)
        inp=torch.zeros(32,10);s=torch.arange(32)//4;g=2*s+torch.arange(32)%2
        n=torch.zeros(32,7);n[:,0]=1
        _,st=boundary_loss(Plane(),inp,s,n,g,torch.full((16,),2.),BoundaryWeights(),training=False)
        self.assertAlmostEqual(summary(st,BoundaryWeights())['heads']['s0']['gradient_norm'],10**.5,places=5)

    def test_real_h9_parameter_count_and_q_gradient(self):
        m=module('model',SCRATCH).HierarchicalVisibilityCDF()
        self.assertEqual(sum(p.numel() for p in m.parameters()),1133705)
        data=batch(n=32);loss,_=boundary_loss(m,*data,torch.full((16,),2.),BoundaryWeights());loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_ramp(self):
        self.assertEqual(ramp_weight('P0',2000,500),0)
        self.assertEqual(ramp_weight('P1',0,500),0)
        self.assertAlmostEqual(ramp_weight('P1',1,500),.002)
        self.assertEqual(ramp_weight('P1',500,500),1)
        self.assertEqual(ramp_weight('P1',2000,500),1)

    def test_wrong_head_shape_rejected(self):
        with self.assertRaises(ValueError):
            boundary_loss(nn.Linear(10,8),*batch(),torch.full((16,),4.),BoundaryWeights())

    def test_missing_stratum_rejected(self):
        denom=torch.full((16,),4.);denom[15]=0
        with self.assertRaises(ValueError):boundary_loss(TinyNine(),*batch(),denom,BoundaryWeights())

    def test_split_and_stratified_sampler(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);fake_cache(path);cache=BoundaryCache(path)
            ids=cache.draw_indices(0,17,128)
            g=2*cache.arrays['train']['s'][ids]+cache.arrays['train']['kind'][ids]
            self.assertTrue(np.array_equal(np.bincount(g,minlength=16),np.full(16,8)))
            shards=np.concatenate([cache.draw_indices(0,17,128,r,4) for r in range(4)])
            self.assertTrue(np.array_equal(ids,shards))
            np.random.seed(122);np.random.randn(100)
            self.assertTrue(np.array_equal(ids,cache.draw_indices(0,17,128)))
            self.assertTrue(set(cache.arrays['train']['x_index']).isdisjoint(cache.arrays['val']['x_index']))

    def test_cache_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);fake_cache(path)
            with (path/'train.npz').open('ab') as f:f.write(b'changed')
            with self.assertRaises(ValueError):BoundaryCache(path)

    def test_val_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            data,_=fake_cache(Path(folder));data['x_index'][0]=100
            with self.assertRaises(ValueError):validate_array_split(data,np.arange(16),np.ones((8,7)),np.full(7,-2),np.full(7,2))

    def test_nonunit_normal_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            data,_=fake_cache(Path(folder));data['normal'][0,0]=20
            with self.assertRaises(ValueError):validate_array_split(data,np.arange(16),np.ones((8,7)),np.full(7,-2),np.full(7,2))

    def test_sample_stream_identity_gate(self):
        from evaluate_pair import assert_matched
        a={'sample_stream_sha256_by_rank':['a'],'parent_sha256':V1_SHA,'cache_manifest_sha256':'cache',
           'pilot_updates':2,'source_sha256':{},'boundary_weights':{},'args':{'arm':'P0','output':'p0','steps':2}}
        b=copy.deepcopy(a);b['args']['arm']='P1';b['args']['output']='p1'
        assert_matched(a,b)
        b['sample_stream_sha256_by_rank']=['b']
        with self.assertRaises(ValueError):assert_matched(a,b)

    def test_explicit_uniform_rng_stream(self):
        self.assertEqual(seed_for_update(0,1),seed_for_update(0,1))
        self.assertNotEqual(seed_for_update(0,1),seed_for_update(0,2))
        self.assertNotEqual(array_hash(np.zeros(3)),array_hash(np.ones(3)))

    def test_selected_input_gradient_and_linearized_shift(self):
        from evaluate_pair import selected
        class ShiftedPlane(nn.Module):
            def forward(self, inp):
                v = 20 * (inp[:,3:4] + .03)
                return torch.cat([v*0+500, v.expand(-1,8)],1)
        x=np.zeros((16,3),np.float32);q=np.zeros((16,7),np.float32)
        s=np.arange(16)%8;n=np.zeros((16,7),np.float32);n[:,0]=1
        out=selected(ShiftedPlane(),x,q,s,n,torch.device('cpu'))
        np.testing.assert_allclose(out['value'],.6,atol=1e-6)
        np.testing.assert_allclose(out['norm'],20,atol=1e-6)
        np.testing.assert_allclose(out['cosine'],1,atol=1e-6)
        np.testing.assert_allclose(out['linearized_zero_shift_rad'],-.03,atol=1e-6)

    def test_prepare_routing_with_explicit_synthetic_geometry(self):
        # Tests generator plumbing/split routing only. NOT real FK qualification.
        import io, types
        from unittest.mock import patch
        from prepare import build_split
        class Data:
            train_indices_np=np.arange(4)
            val_indices_np=np.arange(4,1004)
            x_cpu=torch.zeros(1004,3)
            valid_cpu=torch.ones(1004,2,8,dtype=torch.bool)
            qlib_cpu=torch.zeros(1004,2,7,8)
            qlib_cpu[:,1,1,:]=.5
            def q_limits(self,device):return torch.full((7,),-2.,device=device),torch.full((7,),2.,device=device)
            def sensor_masks(self,device):return torch.ones(8,7,device=device)
        class Oracle:
            def value_grad(self,x,q,s):
                n=torch.zeros(7);n[0]=1
                return float(q[0]-x[0]),n
        stub=types.ModuleType('core')
        def refine(fn,q,*args):
            q=q.clone();g,n=fn(q);q-=g*n
            return {'q':q,'accepted':True,'reason':'SYNTHETIC_PLANE'}
        def qualify(oracle,x,q,s,*args):
            g,n=oracle.value_grad(x,q,s)
            return {'regular':True,'reasons':[],'g_m':g,'active_plane':0,'normal':n}
        def tangent(q,n,mask,rng,radius):
            out=q.clone();out[2]+=radius;return out
        stub.refine_boundary=refine;stub.qualify_boundary=qualify;stub.tangent_seed=tangent
        stub.bank_label=lambda q,bank,mask,sign:{'distance_rad':float(((q-bank)*mask).norm(dim=1).min())}
        cfg=argparse.Namespace(tangent_radius_rad=.1,offbank_min_rad=.001)
        with patch.dict('sys.modules',{'core':stub}):
            tr,_=build_split(Data(),Oracle(),'train',2,2,0,torch.device('cpu'),cfg,io.StringIO())
            va,_=build_split(Data(),Oracle(),'val',2,2,0,torch.device('cpu'),cfg,io.StringIO())
        self.assertTrue(set(tr['x_index']).isdisjoint(va['x_index']))
        self.assertEqual(set(tr['kind']),{0,1})
        self.assertTrue((tr['g']==0).all())

    def test_pilot_checkpoint_loader_with_synthetic_checkpoint(self):
        from evaluate_pair import load_pilot
        model=module('model',SCRATCH).HierarchicalVisibilityCDF()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'final.pt'
            from common import PILOT_FORMAT
            cp={'format':PILOT_FORMAT,'parent_sha256':V1_SHA,'arm':'P1','completed':True,
                'cache_manifest_sha256':'SYNTHETIC_CACHE','pilot_updates':2,'args':{'steps':2},
                'model_state':model.state_dict()}
            torch.save(cp,path)
            write_json(path.parent/'run.json',{'status':'COMPLETE','final_sha256':sha256(path)})
            cache=argparse.Namespace(identity='SYNTHETIC_CACHE')
            got,_=load_pilot(path,torch.device('cpu'),cache,'P1')
            self.assertTrue(all(not p.requires_grad for p in got.parameters()))
            with self.assertRaises(ValueError):load_pilot(path,torch.device('cpu'),cache,'P0')

    def test_repository_dependencies(self):
        needed=[AUDIT/'audit.py',AUDIT/'core.py',AUDIT/'oracle.py',SCRIPTS/'train_signed_visibility_cdf_pairwise_replace.py',URDF]
        if not all(p.is_file() for p in needed):
            if REQUIRE_REPO:self.fail('Real repository dependencies are missing')
            self.skipTest('Isolated local harness; real repository geometry NOT_RUN')
        setup_paths()
        oracle=module('oracle',AUDIT)
        self.assertTrue(hasattr(oracle,'SensorOracle'))
        # Full geometry/numerical parity is required separately by prepare.py preflight.
        self.assertTrue(callable(module('audit',AUDIT).preflight))


def ddp_test(backend):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from train_pilot import perform_update
    rank_local=int(os.environ['LOCAL_RANK'])
    if backend=='nccl':torch.cuda.set_device(rank_local);device=torch.device('cuda',rank_local)
    else:device=torch.device('cpu')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    dist.init_process_group(backend)
    try:
        rank,world=dist.get_rank(),dist.get_world_size()
        if world!=4:raise ValueError('Test expects four ranks')
        baseline=module('train',SCRATCH);obj=module('objective',SCRATCH)
        args=argparse.Namespace(amp='off',boundary_microbatch=7,max_amp_retries=0)
        data=batch(device);n=len(data[0]);local=n//world
        denom=torch.full((16,),n//16,device=device,dtype=torch.float32)
        torch.manual_seed(815)
        target=torch.randn(n,8,device=device)
        tg=torch.randn(n,8,7,device=device);tg=tg/tg.norm(dim=-1,keepdim=True)
        valid=torch.ones(n,8,dtype=torch.bool,device=device);valid[:local,7]=False
        counts=obj.counts_from_mask(valid)
        worst=0.
        for alpha in (0.,.25):
            torch.manual_seed(1)
            source=TinyNine().to(device);reference=copy.deepcopy(source)
            model=DDP(source,device_ids=[rank_local] if backend=='nccl' else None,broadcast_buffers=False)
            optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
            refopt=torch.optim.Adam(reference.parameters(),lr=1e-4)
            scaler=torch.amp.GradScaler('cuda' if backend=='nccl' else 'cpu',enabled=False)
            sl=slice(rank*local,(rank+1)*local)
            uniform=[]
            for start in range(rank*local,(rank+1)*local,5):
                end=min(start+5,(rank+1)*local)
                uniform.append((data[0][start:end],target[start:end],tg[start:end],valid[start:end]))
            result=perform_update(model,optimizer,scaler,uniform,counts,tuple(x[sl] for x in data),denom,
                                  obj.LossWeights(),BoundaryWeights(),alpha,args,baseline,obj)
            refopt.zero_grad()
            gl,_=obj.loss_for_microbatch(reference,data[0],target,tg,valid,counts,obj.LossWeights())
            bl,_=boundary_loss(reference,*data,denom,BoundaryWeights())
            (gl+alpha*bl).backward()
            for a,b in zip(model.module.parameters(),reference.parameters()):
                err=float((a.grad-b.grad).abs().max());worst=max(worst,err)
                torch.testing.assert_close(a.grad,b.grad,atol=2e-5,rtol=2e-4)
            refopt.step()
            for a,b in zip(model.module.parameters(),reference.parameters()):
                torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
            assert math_finite(result)
        error=torch.tensor(worst,device=device);dist.all_reduce(error,op=dist.ReduceOp.MAX)
        if rank==0:print(json.dumps({'status':'PASS','test':'P0 and P1 combined-global-boundary DDP gradients and Adam update',
                                     'backend':backend,'max_abs_gradient_error':float(error)}),flush=True)
    finally:dist.destroy_process_group()


def math_finite(result):
    import math
    return math.isfinite(result['global']['loss']) and math.isfinite(result['boundary']['loss'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--require-repo',action='store_true')
    parser.add_argument('--ddp',action='store_true');parser.add_argument('--backend',choices=('gloo','nccl'),default='gloo')
    args,remaining=parser.parse_known_args()
    if args.ddp:ddp_test(args.backend)
    else:
        REQUIRE_REPO=args.require_repo
        unittest.main(argv=[__file__]+remaining,verbosity=2)
