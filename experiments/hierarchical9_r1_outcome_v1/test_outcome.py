"""CPU regression: derivative path, value parity, DDP averaging and isolation.

The scalar reference below is an independent implementation of the documented
legacy steps. Preflight additionally extracts the ACTUAL runtime methods and
compares them on real model/pair inputs before any GPU job is submitted.
"""
from __future__ import annotations
import copy
from dataclasses import replace
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import tempfile
import unittest
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from rollout import SETTINGS, solve, project, ascend, outcome_loss
from engine import coefficient, functional_outcome, reduce_aux, add_flat, Step
from support import load, original_split

def scalar_reference(score, q, mask, lo, hi, settings=SETTINGS):
    q=torch.maximum(torch.minimum(q,hi),lo).clone()
    def f(q):return float(score(q[None])[0].detach())
    def grad(q):
        z=q.detach().clone().requires_grad_(True)
        y=score(z[None])[0]
        return torch.autograd.grad(y,z)[0].detach()*mask
    def clamp(q):return torch.maximum(torch.minimum(q,hi),lo).detach()
    initial=f(q);best_q=q.clone();best_score=initial;fq=initial
    qz=q.clone() if fq>=0 or abs(fq)<=settings.projection_epsilon_f else None
    for _ in range(settings.projection_iters):
        if qz is not None:break
        g=grad(q);n2=float(g.square().sum())
        if n2<1e-12:break
        raw=f(q)*g/max(n2,1e-8)
        raw*=min(1.,settings.projection_max_step_norm/max(float(raw.norm()),1e-8))
        nxt=clamp(q-settings.projection_damping*raw);fn=f(nxt)
        if fn>best_score:best_q,best_score=nxt.clone(),fn
        if fq*fn<=0 and fq!=fn:
            left,right,fl=q.clone(),nxt.clone(),fq
            qz=q.clone() if abs(fq)<=abs(fn) else nxt.clone()
            fz=f(qz)
            for _ in range(settings.root_refine_iters):
                mid=(left+right)*.5;fm=f(mid)
                if abs(fm)<abs(fz):qz,fz=mid.clone(),fm
                if abs(fm)<=settings.root_tolerance_f:qz=mid.clone();break
                if fl*fm<=0:right=mid.clone()
                else:left,fl=mid.clone(),fm
            break
        if abs(fn)<=settings.projection_epsilon_f:qz=nxt.clone();break
        q,fq=nxt,fn
    root=qz is not None
    qvis=qz if root else best_q.clone()
    best=qvis.clone();bestf=best_score
    for _ in range(settings.branch_ascent_steps if root else settings.branch_fallback_ascent_steps):
        g=grad(qvis);n=float(g.norm())
        if n<1e-8:break
        raw=settings.branch_step_size*g/n
        raw*=min(1.,settings.branch_max_step_norm/max(float(raw.norm()),1e-8))
        qvis=clamp(qvis+raw)
        if not root and f(qvis)>bestf:best,bestf=qvis.clone(),f(qvis)
    return qvis if root else best

class LinearModel(nn.Module):
    def __init__(self,dtype=torch.float64):
        super().__init__();self.w=nn.Parameter(torch.tensor(1.,dtype=dtype));self.b=nn.Parameter(torch.tensor(.07,dtype=dtype))
    def forward(self, inp):
        y=(self.w*inp[:,3]+self.b)[:,None]
        return y.expand(-1,9)

def fixtures(n=8,dtype=torch.float64):
    x=torch.zeros(n,3,dtype=dtype);qm=torch.zeros(n,7,dtype=dtype);qp=qm.clone()
    qm[:,0]=-.2;qp[:,0]=.6;s=torch.arange(n)%8
    masks=torch.ones(8,7,dtype=dtype);lo=torch.full((7,),-2.,dtype=dtype);hi=-lo
    return (x,qm,qp,s),masks,lo,hi

def geom(x,q,s):return q[:,0]-.4

def distributed_worker(rank, port, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',world_size=2,rank=rank)
    try:
        p=torch.tensor(2.,requires_grad=True)
        loss=(p-(1 if rank==0 else 3))**2
        stats=torch.zeros(9);stats[7]=2 if rank==0 else 6
        g=reduce_aux(loss,stats,{'p':p},.3,8.,2)
        Path(output,f'{rank}.txt').write_text(str(float(g)))
    finally:dist.destroy_process_group()

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_schedule(self):
        c={'warmup':10000,'ramp':5000,'weight':.01}
        self.assertEqual(coefficient(1,c),0);self.assertEqual(coefficient(10000,c),0)
        self.assertAlmostEqual(coefficient(12500,c),.005);self.assertEqual(coefficient(50000,c),.01)

    def test_reference_value_parity_linear_and_nonlinear(self):
        torch.manual_seed(14)
        lo=torch.full((7,),-2.,dtype=torch.double);hi=-lo
        q=torch.randn(24,7,dtype=torch.double)*.4
        funcs=[lambda q:q[:,0]*1.3+.03,
               lambda q:torch.sin(q[:,0]*11)-.4,
               lambda q:q[:,0]**2+.1*q[:,1]-.3,
               lambda q:q[:,0]*0-1.]
        mask=torch.ones_like(q);mask[:,5:]=0
        for fun in funcs:
            a,_=solve(fun,q,mask,lo,hi,training=True)
            b=torch.stack([scalar_reference(fun,x,mm,lo,hi) for x,mm in zip(q,mask)])
            self.assertTrue(torch.allclose(a,b,atol=1e-10,rtol=1e-10),float((a-b).detach().abs().max()))

    def test_joint_mask_and_limits(self):
        q=torch.tensor([[2.,.8,1.,1.,1.,1.,1.]],dtype=torch.double)
        lo=torch.full((7,),-.5,dtype=torch.double);hi=-lo
        mask=torch.tensor([[1.,0,0,0,0,0,0]],dtype=torch.double)
        out,_=solve(lambda z:z.sum(1)-4,q,mask,lo,hi)
        self.assertTrue(((out>=lo)&(out<=hi)).all())
        self.assertTrue(torch.equal(out[:,1:],torch.full((1,6),.5,dtype=torch.double)))

    def test_endpoint_gradient_matches_parameter_finite_difference(self):
        model=LinearModel();batch,masks,lo,hi=fixtures()
        loss,stats,pars=functional_outcome(model,batch,geom,masks,lo,hi)
        grads=torch.autograd.grad(loss,tuple(pars.values()))
        for name,g in zip(pars,grads):
            p=getattr(model,name);v=p.detach().clone();eps=1e-5
            with torch.no_grad():p.copy_(v+eps)
            high=float(functional_outcome(model,batch,geom,masks,lo,hi)[0].detach())
            with torch.no_grad():p.copy_(v-eps)
            low=float(functional_outcome(model,batch,geom,masks,lo,hi)[0].detach())
            with torch.no_grad():p.copy_(v)
            self.assertAlmostEqual(float(g),(high-low)/(2*eps),places=5)
        self.assertGreater(abs(float(grads[1])),1.)
        self.assertEqual(float(stats[7]),16)

    def test_original_parameters_rng_and_grad_not_changed(self):
        model=LinearModel();batch,masks,lo,hi=fixtures()
        before={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
        loss,stats,pars=functional_outcome(model,batch,geom,masks,lo,hi)
        torch.autograd.grad(loss,tuple(pars.values()))
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_invalid_witness_is_masked_not_false_target(self):
        model=LinearModel();batch,masks,lo,hi=fixtures(8)
        x,qm,qp,s=batch;qm[0,0]=.7
        loss,stats,pars=functional_outcome(model,(x,qm,qp,s),geom,masks,lo,hi)
        self.assertEqual(float(stats[7]),14);self.assertAlmostEqual(float(stats[8]),.125)
        qm[:,0]=.7
        loss,stats,pars=functional_outcome(model,(x,qm,qp,s),geom,masks,lo,hi)
        self.assertEqual(float(stats[7]),0);self.assertEqual(float(loss.detach()),0)
        g=torch.autograd.grad(loss,tuple(pars.values()),allow_unused=True)
        self.assertTrue(all(v is None or torch.count_nonzero(v)==0 for v in g))

    def test_reduce_add_and_single_optimizer_update(self):
        model=LinearModel();batch,masks,lo,hi=fixtures()
        loss,stats,pars=functional_outcome(model,batch,geom,masks,lo,hi)
        unscaled=torch.autograd.grad(loss,tuple(pars.values()),retain_graph=True)
        flat=reduce_aux(loss,stats,pars,.01,8.,1)
        self.assertTrue(torch.allclose(flat,torch.stack(unscaled)*.08))
        for p in model.parameters():p.grad=torch.ones_like(p)*8
        add_flat(list(model.parameters()),flat)
        opt=torch.optim.SGD(model.parameters(),lr=.1)
        before=[p.clone() for p in model.parameters()]
        for p in model.parameters():p.grad/=8
        opt.step()
        for p,old,g in zip(model.parameters(),before,unscaled):
            self.assertTrue(torch.allclose(p,old-.1*(1.+.01*g)))

    def test_r1_architecture_real_backward(self):
        repo=Path(os.environ.get('O_TEST_REPO',str(Path(__file__).resolve().parents[2])))
        path=repo/'experiments/hierarchical9_scratch50k_r012_v1/model.py'
        arch=load('outcome_cpu_real_r1',path)
        torch.manual_seed(0);m=arch.build_model('R1')
        self.assertEqual(m.parameter_count(),2184329)
        batch,mask,lo,hi=fixtures(2,torch.float32)
        loss,stats,pars=functional_outcome(m,batch,geom,mask,lo,hi)
        gs=torch.autograd.grad(loss,tuple(pars.values()),allow_unused=True)
        self.assertTrue(all(g is None or torch.isfinite(g).all() for g in gs))
        self.assertTrue(any(g is not None and torch.count_nonzero(g)>0 for g in gs))

    def test_original_spatial_split_deterministic(self):
        x=np.zeros((100,3));v=np.ones((100,2,8),bool)
        a=original_split(x,v,0,10);b=original_split(x,v,0,10)
        self.assertFalse(np.intersect1d(a['train'],a['val']).size)
        self.assertTrue(np.array_equal(a['train'],b['train']))

    def test_two_rank_aux_gradient_mean(self):
        with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        with tempfile.TemporaryDirectory() as td:
            mp.spawn(distributed_worker,args=(port,td),nprocs=2,join=True)
            for rank in range(2):self.assertAlmostEqual(float(Path(td,f'{rank}.txt').read_text()),-2.4,places=5)

    def test_zero_weight_delegates_original_step(self):
        from unittest import mock
        events=[]
        def original(*a):events.append(True);return {'loss':3.}
        step=Step(original,None,None,None,None,None,None,{'weight':.01,'warmup':10000,'ramp':5000})
        with mock.patch.object(dist,'get_world_size',return_value=1),mock.patch.object(dist,'get_rank',return_value=0):
            r=step(None,None,torch.ones(9),None,None,object(),object())
        self.assertEqual(r,{'loss':3.});self.assertEqual(len(events),1)

    def test_real_pair_schema_hashes_split_filter_and_stateless_rng(self):
        import support
        from unittest import mock
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cache=root/'cache';pairs=root/'pairs'
            cache.mkdir();pairs.mkdir()
            xx=np.zeros((4,3),np.float32);xx[:,0]=np.arange(4)
            np.save(cache/'x.npy',xx)
            cm={'status':'COMPLETE','training_ready':True,'files':{'x.npy':{'sha256':support.sha(cache/'x.npy')}}}
            support.write(cache/'manifest.json',cm);cacheid=support.sha(cache/'manifest.json')
            fields={};totals={}
            for sp in ('train','val'):
                (pairs/sp).mkdir();rows=[(x,s,r) for x in range(4) for s in range(8) for r in range(3)]
                a={'x_index':np.array([v[0] for v in rows],np.int32),
                   'sensor':np.array([v[1] for v in rows],np.uint8),
                   'radius_id':np.array([v[2] for v in rows],np.uint8),
                   'q_minus':np.zeros((len(rows),7),np.float32),
                   'q_plus':np.ones((len(rows),7),np.float32)*.01}
                for k,v in a.items():
                    file=pairs/sp/(k+'.npy');np.save(file,v);fields[sp+'/'+file.name]=support.sha(file)
                totals[sp]=len(rows)
            pm={'format':'care_h9_v4_verified_pairs_v1','status':'COMPLETE','cache_identity':cacheid,
                'fields':fields,'total_by_split':totals}
            support.write(pairs/'manifest.json',pm)
            with mock.patch.object(support,'CACHE_SHA',cacheid), mock.patch.object(support,'PAIR_SHA',support.sha(pairs/'manifest.json')):
                ps=support.Pairs({'cache':str(cache),'pairs':str(pairs)},
                                 {'train':np.array([0]),'val':np.array([3])})
                rng=torch.get_rng_state().clone();npr=np.random.get_state()
                b=ps.batch('train',19,0,torch.device('cpu'));bb=ps.batch('train',19,0,torch.device('cpu'))
                self.assertTrue(torch.equal(b[0],bb[0]));self.assertTrue((b[0][:,0]==0).all())
                self.assertTrue((ps.batch('val',0,0,torch.device('cpu'))[0][:,0]==3).all())
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                self.assertTrue(np.array_equal(npr[1],np.random.get_state()[1]))
                self.assertEqual(ps.counts['train'],[1]*24)
                (pairs/'train/q_minus.npy').write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError,'Corrupt'):
                    support.Pairs({'cache':str(cache),'pairs':str(pairs)},{'train':np.array([0]),'val':np.array([3])})

    def test_actual_ddp_step_adds_outcome_without_changing_base_loss(self):
        import contextlib
        from types import SimpleNamespace
        from torch.nn.parallel import DistributedDataParallel as DDP
        with tempfile.TemporaryDirectory() as td:
            dist.init_process_group('gloo',init_method='file://'+td+'/pg',rank=0,world_size=1)
            try:
                m=LinearModel(torch.float32);ddp=DDP(m)
                data=torch.ones((4,10));target=torch.zeros((4,9))
                before=copy.deepcopy(m);bl=before(data).square().mean()
                bg=torch.autograd.grad(bl,tuple(before.parameters()))
                batch,mask,lo,hi=fixtures(2,torch.float32)
                al,ast,pars=functional_outcome(before,batch,geom,mask,lo,hi)
                ag=torch.autograd.grad(al,tuple(pars.values()))
                def objective(net,inp,truth,counts,weights,world_size,training):
                    loss=(net(inp)-truth).square().mean()
                    stats=torch.zeros(9,10,dtype=torch.double);stats[0,0]=len(inp);stats[0,1]=float(loss.detach())*len(inp)
                    return loss,stats
                def synced(ok,msg,device):
                    if not bool(ok):raise ValueError(msg)
                base=SimpleNamespace(STAT_NAMES=list(range(10)),loss_for_microbatch=objective,
                    amp_context=lambda amp:contextlib.nullcontext(),synchronized_check=synced,
                    summarize=lambda stats,w:{'loss':float(stats[0,1]/stats[0,0])},log=lambda *args:None)
                sampler=SimpleNamespace(batch=lambda *args:batch)
                cfg={'weight':.01,'warmup':0,'ramp':1,'pairs_per_sensor_per_rank':4}
                step=Step(lambda *args:None,base,sampler,geom,mask,lo,hi,cfg)
                args=SimpleNamespace(max_amp_retries=0,amp='off',log_every=1,steps=1)
                opt=torch.optim.SGD(ddp.parameters(),lr=.01)
                scaler=torch.amp.GradScaler('cpu',enabled=False)
                counts=torch.zeros(9);counts[0]=4
                result=step(ddp,[(data,target)],counts,None,args,opt,scaler)
                self.assertAlmostEqual(result['loss'],float(bl.detach()),places=6)
                for p,old,g,h in zip(m.parameters(),before.parameters(),bg,ag):
                    self.assertTrue(torch.allclose(p,old-.01*(g+.01*h),atol=1e-6))
            finally:dist.destroy_process_group()

    def test_checkpoint_resume_and_export_shape(self):
        m=LinearModel();opt=torch.optim.Adam(m.parameters(),lr=.001)
        loss=m(torch.ones(2,10,dtype=torch.double)).square().mean();loss.backward();opt.step()
        c={'model_state':m.state_dict(),'optimizer_state':opt.state_dict(),'step':123}
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'latest.pt';torch.save(c,p);cp=torch.load(p,weights_only=False)
            n=LinearModel();n.load_state_dict(cp['model_state'],strict=True)
            op=torch.optim.Adam(n.parameters(),lr=.001);op.load_state_dict(cp['optimizer_state'])
            self.assertTrue(torch.equal(m.w,n.w));self.assertEqual(cp['step'],123)

if __name__=='__main__':unittest.main()
