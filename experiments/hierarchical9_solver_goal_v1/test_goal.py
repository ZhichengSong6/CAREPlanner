"""Executed CPU tests: mixed derivatives, finite differences, DDP, sampler and resume schedule.

No test here certifies full CUDA/NCCL training or learned solver improvement.
"""
from __future__ import annotations
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from goal import paired_planes, rollout, geometry_objective, weight_at, WitnessPool, inject_once, load
from evaluate import summarize

HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/'config.json').read_text())
torch.set_num_threads(2)


class PlaneOracle:
    hfov=50.;vfov=66.;zmin=.2;zmax=.7;delta=.01
    specs=tuple(range(8))
    @staticmethod
    def fk(spec,q):
        # A differentiable translation sensor: x_sensor position = -q_0.
        row0=torch.stack((q[:,0]*0+1,q[:,0]*0,q[:,0]*0,-q[:,0]),-1)
        row1=torch.stack((q[:,0]*0,q[:,0]*0+1,q[:,0]*0,q[:,0]*0),-1)
        row2=torch.stack((q[:,0]*0,q[:,0]*0,q[:,0]*0+1,q[:,0]*0),-1)
        row3=torch.stack((q[:,0]*0,q[:,0]*0,q[:,0]*0,q[:,0]*0+1),-1)
        return torch.stack((row0,row1,row2,row3),1)


class TinyModel(nn.Module):
    def __init__(self,dtype=torch.double):
        super().__init__()
        self.w=nn.Parameter(torch.eye(9,10,dtype=dtype)*.02)
        self.b=nn.Parameter(torch.zeros(9,dtype=dtype)+.08)
        with torch.no_grad():self.w[:,3]=1.
    def forward(self,x):return x@self.w.T+self.b
    def forward_sensor(self,x,s):return self(x)[:,s+1]


def batch(n=2,dtype=torch.double):
    tx=math.tan(math.radians(50)/2);nx=math.sqrt(1+tx*tx)
    x=torch.tensor([-.4*tx+.01*nx,0,.4],dtype=dtype).repeat(n,1)
    qm=torch.zeros(n,7,dtype=dtype);qm[:,0]=-.02
    qp=qm.clone();qp[:,0]=.02
    return [(x.clone(),qm.clone(),qp.clone()) for _ in range(8)]


def loss_for(model,bs=None):
    m=torch.zeros(8,7,dtype=torch.double);m[:,0]=1
    return geometry_objective(model,batch() if bs is None else bs,PlaneOracle(),m,
                              torch.full((7,),-2.,dtype=torch.double),torch.full((7,),2.,dtype=torch.double),CFG)


def make_fake_runner():
    base=SimpleNamespace()
    def primitive(model,x,y,denom):
        l=(model(x)-y).square().sum()/denom
        return l,l.detach()
    base.loss_for_microbatch=primitive
    def run(model,batches,counts,weights,args,opt,scaler):
        opt.zero_grad(set_to_none=True);stats=0.
        for i,(x,y) in enumerate(batches):
            ctx=model.no_sync() if i<len(batches)-1 and hasattr(model,'no_sync') else contextlib.nullcontext()
            with ctx:
                l,st=base.loss_for_microbatch(model,x,y,counts)
                l.backward();stats+=float(st)
        opt.step()
        return {'loss':stats}
    return base,run


def ddp_child(rank,path,out):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    try:
        model=TinyModel(torch.double);ddp=torch.nn.parallel.DistributedDataParallel(model)
        x=torch.arange(80,dtype=torch.double).reshape(8,10)/1000
        y=torch.ones(8,9,dtype=torch.double)*.15
        localx,localy=x[rank*4:(rank+1)*4],y[rank*4:(rank+1)*4]
        base,run=make_fake_runner();opt=torch.optim.SGD(ddp.parameters(),lr=.001)
        bs=batch(2);bs=[(a+rank*.0001,b,c) for a,b,c in bs]
        def aux(m):
            l,met=loss_for(m,bs);return .01*l,met
        inject_once(base,run,ddp,[(localx[:2],localy[:2]),(localx[2:],localy[2:])],
                    4*9,None,None,opt,None,aux)
        torch.save(model.state_dict(),Path(out)/f'rank{rank}.pt')
    finally:dist.destroy_process_group()


class Tests(unittest.TestCase):
    def test_projection_math_and_no_inactive_joint_motion(self):
        m=TinyModel();x,q,_=batch()[0]
        mask=torch.tensor([1,0,0,0,0,0,0],dtype=torch.double)
        lo=torch.full((7,),-2.,dtype=torch.double);hi=-lo
        qk=rollout(m,x,q,0,mask,lo,hi,3)
        # Independently calculate scalar Newton updates for this linear field.
        slope=float(m.w[1,3].detach())
        bias=(x@m.w[1,:3].detach()+m.b[1].detach())[:,None]
        expected=q[:,0:1]
        for _ in range(3):expected=expected-.5*(expected+bias/slope).clamp(-.25,.25)
        expected=expected+.05
        self.assertTrue(torch.allclose(qk[:,0:1],expected,atol=1e-10))
        self.assertTrue(torch.equal(qk[:,1:],q[:,1:]))
        self.assertIsNotNone(qk.grad_fn)

    def test_margin_matches_hand_computed_normalized_planes(self):
        x,q,_=batch()[0]
        p=paired_planes(PlaneOracle(),x,q,0)
        nx=math.sqrt(1+math.tan(math.radians(25))**2)
        self.assertTrue(torch.allclose(p[:,0],q[:,0]/nx,atol=1e-10))
        self.assertTrue(torch.allclose(p[:,4],torch.full_like(p[:,4],.19)))
        self.assertTrue(torch.allclose(p[:,5],torch.full_like(p[:,5],.29)))
        q=q.requires_grad_();g=paired_planes(PlaneOracle(),x,q,0)[:,0].sum()
        grad=torch.autograd.grad(g,q)[0]
        self.assertAlmostEqual(float(grad[0,0]),1/nx,places=10)

    def test_parameter_gradient_matches_finite_difference(self):
        m=TinyModel();l,met=loss_for(m)
        grads=torch.autograd.grad(l,(m.w,m.b))
        self.assertGreater(abs(float(grads[1][1])),1e-6)
        eps=1e-6
        for parameter,grad,index in ((m.b,grads[1],(1,)),(m.w,grads[0],(1,3))):
            v=float(parameter[index].detach())
            with torch.no_grad():parameter[index]=v+eps
            plus=float(loss_for(m)[0].detach())
            with torch.no_grad():parameter[index]=v-eps
            minus=float(loss_for(m)[0].detach())
            with torch.no_grad():parameter[index]=v
            fd=(plus-minus)/(2*eps)
            self.assertAlmostEqual(float(grad[index]),fd,delta=max(2e-5,abs(fd)*2e-5))

    def test_a_gradient_step_lowers_actual_fov_objective(self):
        m=TinyModel();opt=torch.optim.SGD(m.parameters(),lr=1e-5)
        l,_=loss_for(m);old=float(l.detach());l.backward();opt.step()
        new=float(loss_for(m)[0].detach())
        self.assertLess(new,old)

    def test_bad_positive_witness_is_rejected(self):
        bs=batch();bs[3]=(bs[3][0],bs[3][1],bs[3][1])
        with self.assertRaisesRegex(ValueError,'Invalid ORIGINAL'):
            loss_for(TinyModel(),bs)

    def test_warmup_and_resume_step_have_same_weight(self):
        self.assertEqual(weight_at(10000,CFG),0.)
        self.assertAlmostEqual(weight_at(12500,CFG),.005)
        self.assertEqual(weight_at(15000,CFG),.01)
        self.assertEqual(weight_at(50000,CFG),.01)
        for resumed in (9999,10000,14999,24000):
            self.assertEqual(weight_at(resumed+1,CFG),weight_at(list(range(resumed+1,resumed+2))[0],CFG))

    def test_zero_aux_matches_baseline_adam_update(self):
        torch.manual_seed(7);x=torch.randn(4,10,dtype=torch.double);y=torch.randn(4,9,dtype=torch.double)
        a=TinyModel();b=copy.deepcopy(a);base,run=make_fake_runner()
        class View(nn.Module):
            def __init__(self,m):super().__init__();self.module=m
            def forward(self,x):return self.module(x)
        oa=torch.optim.Adam(a.parameters(),lr=.001);ob=torch.optim.Adam(b.parameters(),lr=.001)
        bs=[(x[:2],y[:2]),(x[2:],y[2:])]
        ra=run(View(a),bs,36,None,None,oa,None)
        calls=[]
        def zero(m):calls.append(1);return m.b.sum()*0.,torch.zeros(8,8)
        rb,_=inject_once(base,run,View(b),bs,36,None,None,ob,None,zero)
        self.assertEqual(len(calls),1);self.assertEqual(ra,rb)
        for u,v in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(u,v))

    def test_retries_reuse_aux_once_per_replay_and_restore_callback(self):
        base,run=make_fake_runner();saved=base.loss_for_microbatch;model=TinyModel()
        x=torch.zeros(4,10,dtype=torch.double);y=torch.zeros(4,9,dtype=torch.double);bs=[(x[:2],y[:2]),(x[2:],y[2:])]
        called=[]
        def runner(*args):
            # Emulate original runner's first overflow replay without optimizer updates.
            for _ in range(2):
                for a,b in bs:base.loss_for_microbatch(model,a,b,36)
            return {'loss':0}
        def extra(_):called.append(1);return model.b.sum()*0.,torch.zeros(8,8)
        inject_once(base,runner,SimpleNamespace(module=model),bs,36,None,None,None,None,extra)
        self.assertEqual(len(called),2);self.assertIs(base.loss_for_microbatch,saved)

    def test_two_rank_ddp_matches_single_global_gradient_update(self):
        with tempfile.TemporaryDirectory() as d:
            mp.spawn(ddp_child,args=(str(Path(d)/'store'),d),nprocs=2,join=True)
            a=torch.load(Path(d)/'rank0.pt',weights_only=True);b=torch.load(Path(d)/'rank1.pt',weights_only=True)
            for k in a:self.assertTrue(torch.equal(a[k],b[k]))
            model=TinyModel();opt=torch.optim.SGD(model.parameters(),lr=.001)
            x=torch.arange(80,dtype=torch.double).reshape(8,10)/1000;y=torch.ones(8,9,dtype=torch.double)*.15
            l=(model(x)-y).square().mean()
            for rank in (0,1):
                bs=[(u+rank*.0001,v,w) for u,v,w in batch(2)]
                l=l+.005*loss_for(model,bs)[0]
            l.backward();opt.step()
            for k,v in model.state_dict().items():self.assertTrue(torch.allclose(v,a[k],atol=1e-10,rtol=1e-9))

    def test_real_R1_input_derivative_graph_all_sensor_params_finite(self):
        repo=Path(os.environ.get('SG_TEST_REPO',str(HERE.parents[1])))
        path=repo/'experiments/hierarchical9_scratch50k_r012_v1/model.py'
        b=path.read_bytes();h=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        self.assertEqual(h,'0b9dab1986114d4b95b33bb0fa01fd8e0c68e53b')
        arch=load('sg_test_actual_arch',path);torch.manual_seed(0);model=arch.build_model('R1')
        self.assertEqual(model.parameter_count(),2184329)
        bs=batch(1,torch.float32)
        masks=torch.ones(8,7)
        loss,met=geometry_objective(model,bs,PlaneOracle(),masks,torch.full((7,),-2.),torch.full((7,),2.),CFG)
        loss.backward();self.assertTrue(torch.isfinite(met).all())
        for name,p in model.named_parameters():
            if name.startswith('union_'):continue
            self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)

    def test_witness_sampler_is_stateless_and_excludes_original_val(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cache=SimpleNamespace(identity='test',x=np.zeros((4,3),np.float32))
            counts={}
            for sp,xx in (('train',[0,1,2]),('val',[1,2,3])):
                (root/sp).mkdir();rows=[(x,s,r) for x in xx for s in range(8) for r in range(3)]
                for k,v in {'x_index':[x for x,s,r in rows],'sensor':[s for x,s,r in rows],
                            'radius_id':[r for x,s,r in rows],'q_minus':np.zeros((len(rows),7)),
                            'q_plus':np.ones((len(rows),7))*.02}.items():np.save(root/sp/(k+'.npy'),v)
                counts[sp]=len(rows)
            (root/'manifest.json').write_text(json.dumps({'format':'care_h9_v4_verified_pairs_v1','status':'COMPLETE',
                'cache_identity':'test','total_by_split':counts}))
            pool=WitnessPool(cache,root,np.array([0,1]),np.array([2,3]),CFG)
            for g in pool.groups['train']:self.assertTrue(np.isin(pool.a['train']['x_index'][g],[0,1]).all())
            before=torch.get_rng_state().clone();rng=np.random.get_state()
            one=pool.draw('train',123,0,torch.device('cpu'));two=pool.draw('train',123,0,torch.device('cpu'))
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            self.assertTrue(np.array_equal(rng[1],np.random.get_state()[1]))
            for x,y in zip(one,two):
                for a,b in zip(x,y):self.assertTrue(torch.equal(a,b))

    def test_report_requires_actual_frozen_baseline(self):
        rows=[]
        for i in range(1963):
            local=i<683;success=i<618 if local else i<683+673
            rows.append({'case_id':i,'group':'S/local_r01' if local else 'S/uniform_outside','sensor':i%8,
              'models':{n:{'fov_pass':success,'failure_stage':'PASS' if success else 'FAIL'} for n in ('R1','SOLVER_GOAL')}})
        r=summarize(rows);self.assertEqual(r['paired'],{'gained':0,'lost':0})
        rows[0]['models']['R1']['fov_pass']=False
        with self.assertRaisesRegex(ValueError,'NOT reproduced'):summarize(rows)

if __name__=='__main__':unittest.main()
