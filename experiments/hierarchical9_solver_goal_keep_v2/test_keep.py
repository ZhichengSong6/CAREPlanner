"""CPU tests, including real pinned R1 and real two-rank Gloo backward parity."""
from __future__ import annotations
import copy,hashlib,json,os,subprocess,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import k2_common as C
import keep,replay,train,evaluate
CFG=C.read(C.HERE/'protocol.json')
torch.set_num_threads(2)

class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(10,12),nn.Tanh(),nn.Linear(12,9))
    def forward(self,x):return self.net(x)
    def forward_sensor(self,x,s):return self(x)[:,s+1]

def make_batch(teacher,n=4,dtype=torch.float32):
    x=torch.randn(n,3,dtype=dtype)*.2;q=torch.randn(n,7,dtype=dtype)*.1
    batch=[];lo=torch.full((7,),-2.,dtype=dtype);hi=-lo;masks=torch.ones(8,7,dtype=dtype)
    for s in range(8):
        y,p,a=keep.predict(teacher,x,q,s,masks[s],lo,hi,False)
        batch.append(dict(x=x.clone(),q=q.clone(),score=y.detach(),project=p.detach(),ascent=a.detach()))
    return batch,masks,lo,hi

def changed_tiny():
    torch.manual_seed(711);teacher=Tiny().eval();m=copy.deepcopy(teacher)
    with torch.no_grad():m.net[-1].bias[1:].add_(.08)
    return teacher,m

def ddp_worker(rank,root):
    dist.init_process_group('gloo',init_method='file://'+str(Path(root)/'rendezvous'),rank=rank,world_size=2)
    try:
        teacher,model=changed_tiny();batch,masks,lo,hi=make_batch(teacher,8)
        torch.manual_seed(912);inp=torch.randn(16,10)*.2
        ddp=DDP(model);loc=inp[rank*8:(rank+1)*8]
        b=[{k:v[rank*4:(rank+1)*4] for k,v in z.items()} for z in batch]
        import contextlib
        for j in range(2):
            ctx=ddp.no_sync() if j==0 else contextlib.nullcontext()
            with ctx:
                base=ddp(loc[j*4:(j+1)*4]).square().sum()/(8*9)
                if j==0:base=base+CFG['keep_weight']*keep.objective(ddp.module,b,masks,lo,hi,CFG)[0]
                base.backward()
        vector=torch.cat([p.grad.reshape(-1) for p in ddp.parameters()])
        torch.save(vector,Path(root)/f'gradient{rank}.pt')
    finally:dist.destroy_process_group()

class Tests(unittest.TestCase):
    def test_protocol_budget_and_lr(self):
        self.assertEqual(CFG['updates'],20000);self.assertEqual(CFG['arms'],list(C.ARMS))
        self.assertEqual(CFG['keep_weight'],.01)
        self.assertEqual(train.learning_rate(1,CFG),CFG['lr_start'])
        self.assertAlmostEqual(train.learning_rate(20000,CFG),CFG['lr_end'])
        self.assertTrue(all(train.learning_rate(i,CFG)>=train.learning_rate(i+1,CFG) for i in range(1,20000)))

    def test_actions_mask_caps_and_limits(self):
        q=torch.zeros(3,7);q[2]=1.99;y=torch.tensor([100.,0.,-100.]);g=torch.ones(3,7);g[1]=0
        mask=torch.tensor([1.,1.,0,0,0,0,0]);lo=torch.full((7,),-2.);hi=-lo
        p,a=keep.actions(q,y,g,mask,lo,hi)
        self.assertTrue(torch.isfinite(p).all());self.assertTrue(torch.isfinite(a).all())
        self.assertTrue((p[:,2:]==0).all());self.assertTrue((a[:,2:]==0).all())
        self.assertTrue((p.norm(dim=1)<=.125+1e-6).all());self.assertTrue((a.norm(dim=1)<=.05+1e-6).all())
        self.assertTrue((q+p<=hi).all());self.assertTrue((q+a<=hi).all())

    def test_exact_runtime_scalar_step_formula(self):
        torch.manual_seed(77)
        q=torch.randn(30,7)*.2;y=torch.randn(30);g=torch.randn(30,7)
        mask=torch.tensor([1.,1.,1.,0,0,0,0]);lo=torch.full((7,),-.3);hi=-lo;q=q.clamp(-.3,.3)
        p,a=keep.actions(q,y,g,mask,lo,hi)
        for i in range(30):
            gg=g[i]*mask;raw=float(y[i])*gg/gg.square().sum().clamp_min(1e-8)
            expected=(q[i]-.5*raw*min(1.,.25/max(float(raw.norm()),1e-8))).clamp(-.3,.3)-q[i]
            self.assertTrue(torch.allclose(p[i],expected,atol=2e-7))
            asc=(q[i]+.05*gg/gg.norm()).clamp(-.3,.3)-q[i]
            self.assertTrue(torch.allclose(a[i],asc,atol=2e-7))

    def test_zero_penalty_at_teacher_and_small_deadzone(self):
        teacher,_=changed_tiny();batch,masks,lo,hi=make_batch(teacher)
        loss,_=keep.objective(teacher,batch,masks,lo,hi,CFG)
        self.assertEqual(float(loss.detach()),0.)
        loss.backward();self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in teacher.parameters()))
        z=torch.zeros(4);v=torch.zeros(4,7)
        loss,_=keep.soft_errors(z+.001,v+.0001,v-.0001,z,v,v,CFG)
        self.assertEqual(float(loss.sum()),0.)

    def test_nonzero_parameter_gradient_and_finite_difference(self):
        teacher,m=changed_tiny();batch,masks,lo,hi=make_batch(teacher)
        loss,_=keep.objective(m,batch,masks,lo,hi,CFG);loss.backward()
        analytic=float(m.net[-1].bias.grad[1]);self.assertGreater(abs(analytic),1e-5)
        orig=float(m.net[-1].bias[1].detach());values=[];eps=1e-3
        for sign in (-1,1):
            with torch.no_grad():m.net[-1].bias[1]=orig+sign*eps
            values.append(float(keep.objective(m,batch,masks,lo,hi,CFG)[0].detach()))
        with torch.no_grad():m.net[-1].bias[1]=orig
        fd=(values[1]-values[0])/(2*eps)
        self.assertAlmostEqual(analytic,fd,delta=max(1e-3,abs(fd)*.025))

    def test_real_original_model_backward(self):
        root=Path(os.environ.get('K2_TEST_REPO',C.HERE.parents[1]));p=root/'experiments/hierarchical9_scratch50k_r012_v1/model.py'
        raw=p.read_bytes();self.assertEqual(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),'0b9dab1986114d4b95b33bb0fa01fd8e0c68e53b')
        arch=C.load('k2_test_original_arch',p)
        torch.manual_seed(0);teacher=arch.build_model('R1');m=copy.deepcopy(teacher)
        self.assertEqual(m.parameter_count(),2184329)
        batch,masks,lo,hi=make_batch(teacher,2)
        with torch.no_grad():m.sensor_heads[6][-1].bias.add_(.02)
        loss,_=keep.objective(m,batch,masks,lo,hi,CFG);loss.backward()
        grad=[p.grad for p in m.sensor_tails[6].parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grad))
        self.assertGreater(sum(float(g.abs().sum()) for g in grad),0)
        x=torch.randn(4,10)
        self.assertTrue(torch.allclose(m(x)[:,1:9],m.forward_sensors(x),atol=1e-6))
        self.assertTrue(torch.allclose(m(x)[:,7],m.forward_sensor(x,6),atol=1e-6))

    def test_actual_two_rank_gloo_matches_global_backward(self):
        with tempfile.TemporaryDirectory() as td:
            mp.spawn(ddp_worker,args=(td,),nprocs=2,join=True)
            teacher,m=changed_tiny();batch,masks,lo,hi=make_batch(teacher,8)
            torch.manual_seed(912);inp=torch.randn(16,10)*.2
            loss=m(inp).square().mean()+CFG['keep_weight']*keep.objective(m,batch,masks,lo,hi,CFG)[0];loss.backward()
            expected=torch.cat([p.grad.reshape(-1) for p in m.parameters()])
            for rank in (0,1):
                observed=torch.load(Path(td)/f'gradient{rank}.pt',weights_only=True)
                self.assertTrue(torch.allclose(observed,expected,rtol=3e-5,atol=3e-6),float((observed-expected).abs().max()))

    def _bank(self,root):
        for split in ('train','val'):
            ids=np.arange(32);sensor=np.repeat(np.arange(8),4);cohort=np.tile([0,0,1,1],8)
            lens=np.tile([1,5],16);ptr=np.r_[0,np.cumsum(lens)];n=ptr[-1]
            x=np.stack([ids+(0 if split=='train' else 1000),ids*0,ids*0],1).astype(np.float32)
            np.savez(root/(split+'.npz'),ptr=ptr,sensor=sensor,cohort=cohort,case_id=ids,x_index=ids,x=x,
                 q=np.zeros((n,7),np.float32),score=np.zeros(n,np.float32),project=np.zeros((n,7),np.float32),ascent=np.zeros((n,7),np.float32))
        C.finish(root,teacher_sha256=C.TEACHER_SHA)

    def test_bank_trajectory_sampling_and_rng_isolation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self._bank(root);bank=keep.ReplayBank(root,C.TEACHER_SHA)
            state=torch.get_rng_state().clone();npstate=np.random.get_state()
            a=bank.draw('train',12,0,CFG,torch.device('cpu'));b=bank.draw('train',12,0,CFG,torch.device('cpu'))
            self.assertTrue(torch.equal(state,torch.get_rng_state()));self.assertTrue(np.array_equal(npstate[1],np.random.get_state()[1]))
            for x,y in zip(a,b):
                for k in x:self.assertTrue(torch.equal(x[k],y[k]))
            val=bank.draw('val',12,0,CFG,torch.device('cpu'))
            self.assertTrue(all((v['x'][:,0]>=1000).all() for v in val))
            ids=[]
            for step in range(1,101):ids+=bank.draw('train',step,0,CFG,torch.device('cpu'))[0]['x'][:,0].tolist()
            fraction=sum(int(i)%2==0 for i in ids)/len(ids)
            self.assertTrue(.4<fraction<.6,'Long paths should not receive five times the probability')

    def test_corrupt_bank_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self._bank(root);(root/'train.npz').write_bytes(b'bad')
            with self.assertRaises(ValueError):keep.ReplayBank(root,C.TEACHER_SHA)

    def test_fixed_attempt_plan_original_splits_only(self):
        class Oracle:
            def value(self,x,q,s):return float(q.reshape(-1)[0])
            def planes(self,x,q,s):return q[:,0:1].repeat(1,6)
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);pairs=out/'pairs';pairs.mkdir()
            np.savez(out/'splits.npz',x=np.zeros((4,3),np.float32),train=np.array([0,1]),val=np.array([2,3]))
            for split,xids in (('train',[0,1]),('val',[2,3])):
                p=pairs/split;p.mkdir();sensor=np.repeat(np.arange(8),2)
                for k,a in dict(x_index=np.tile(xids,8),sensor=sensor,q_plus=np.ones((16,7),np.float32)*.2,q_minus=np.ones((16,7),np.float32)*-.2).items():np.save(p/(k+'.npy'),a)
            cfg=copy.deepcopy(CFG);cfg['reference_attempts_per_sensor_cohort']={'train':2,'val':2}
            job=dict(out=str(out),pairs=str(pairs),cfg=cfg,lo=[-1.]*7,hi=[1.]*7)
            fake=(None,None,None,Oracle(),None,None,torch.ones(7)*-1,torch.ones(7),torch.ones(8,7))
            with patch.object(replay,'verify'),patch.object(replay,'runtime',return_value=fake):
                for rank in range(4):replay.create_starts(job,rank)
            replay.freeze_inputs(job);rows=list(C.lines(out/'reference_inputs/starts.jsonl'))
            self.assertEqual(len(rows),64)
            for r in rows:self.assertIn(r['x_index'],[0,1] if r['split']=='train' else [2,3])
            self.assertFalse(C.stage(out/'reference_inputs')['model_loaded'])

    def test_trace_contains_start_end_and_is_bounded(self):
        result={'q_candidate':[1.]*7,'q_zero':[.5]*7,'projection_history':[{'q':[i/10]*7} for i in range(20)],'ascent_history':[]}
        q=replay.trace_points(result,[0]*7,8)
        self.assertEqual(q.shape,(8,7));self.assertTrue(np.array_equal(q[0],[0]*7));self.assertTrue(np.array_equal(q[1],[1]*7))

    def test_checkpoint_resume_lineage_guard(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'latest.pt';job={'cfg':CFG};cp=dict(job=job,variant='B_keep',experiment_format=C.FORMAT,parent_sha256=C.TEACHER_SHA,replay_sha256='r',stage_step=100)
            torch.save(cp,p);self.assertEqual(train.checked_checkpoint(p,job,'B_keep','r')['stage_step'],100)
            with self.assertRaises(ValueError):train.checked_checkpoint(p,job,'A_continue','r')
            with self.assertRaises(ValueError):train.checked_checkpoint(p,job,'B_keep','other')

    def test_adam_resume_matches_uninterrupted_updates(self):
        torch.manual_seed(15);a=Tiny();b=copy.deepcopy(a);x=torch.randn(12,10)
        oa=torch.optim.Adam(a.parameters(),lr=.0001);ob=torch.optim.Adam(b.parameters(),lr=.0001)
        for _ in range(3):
            for m,o in ((a,oa),(b,ob)):o.zero_grad();m(x).square().mean().backward();o.step()
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'state.pt';torch.save({'m':b.state_dict(),'o':ob.state_dict()},p)
            b=Tiny();ob=torch.optim.Adam(b.parameters());cp=torch.load(p,weights_only=False);b.load_state_dict(cp['m']);ob.load_state_dict(cp['o'])
            for _ in range(2):
                for m,o in ((a,oa),(b,ob)):o.zero_grad();m(x).square().mean().backward();o.step()
            for x,y in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(x,y))

    def test_x_cluster_summary_and_zero_cases(self):
        rows=[]
        for pid in range(4):
            for _ in range(3):rows.append(dict(start={'point_id':pid},models={'L':{'fov_pass':pid<3},'R':{'fov_pass':pid<2}}))
        r=evaluate.comparison(rows,'L','R',4,2);self.assertEqual(r['gained'],3);self.assertEqual(r['lost'],0);self.assertEqual(r['delta_pp'],25.)
        self.assertIsNone(evaluate.comparison([],'L','R',4,2)['delta_pp'])

    def test_exclusive_stage_write_and_hash_guard(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);C.write(p/'a.json',{'x':1});C.finish(p)
            with self.assertRaises(FileExistsError):C.write(p/'a.json',{'x':2})
            (p/'a.json').write_text('{}')
            with self.assertRaises(ValueError):C.stage(p)

    def test_module_aliases_restore(self):
        import sys
        before=sys.modules.get('keep_test_alias')
        with C.aliases(keep_test_alias=SimpleNamespace(x=1)):self.assertEqual(sys.modules['keep_test_alias'].x,1)
        self.assertEqual(sys.modules.get('keep_test_alias'),before)

    def test_shell_syntax_and_spool_worker_path(self):
        p=C.HERE/'run.sh';subprocess.run(['bash','-n',str(p)],check=True)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);code=root/'snapshot';code.mkdir();job=root/'job.json';job.write_text('{}')
            py=root/'python';py.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n');py.chmod(0o755)
            env=os.environ.copy();env.update(K2_CODE=str(code),K2_JOB=str(job),K2_ROOT=str(root),K2_ACTION='train',VIS_PYTHON=str(py))
            spool=root/'slurm_spool';spool.mkdir();script=spool/'slurm_script';script.write_bytes(p.read_bytes())
            r=subprocess.run(['bash',str(script),'_worker'],env=env,cwd=spool,check=True,capture_output=True,text=True)
            self.assertIn(str(code/'run.py'),r.stdout);self.assertNotIn(str(spool/'run.py'),r.stdout)

if __name__=='__main__':unittest.main(verbosity=2)
