from __future__ import annotations
import contextlib
import copy
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run as ctl
import fast_core as fast


def upstream():
    repo=Path(os.environ.get('G25_TEST_REPO',str(HERE.parents[1])))
    directory=repo/'src/care_visibility_cdf/scripts'
    expected={'data_generator.py':'0195b7016a2668b4d247e75f25ef2a7845e719b0',
              'extract_visibility_zero_level_sets.py':'451e4a31279b9dd9ee89ec59e51f0915e08f07d3'}
    for name,digest in expected.items():
        b=(directory/name).read_bytes()
        if hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()!=digest:
            raise AssertionError('Upstream changed: '+name)
    sys.path.insert(0,str(directory))
    return importlib.import_module('data_generator'),importlib.import_module('extract_visibility_zero_level_sets')


def fixture():
    gen,ex=upstream();torch.set_num_threads(1)
    specs=[]
    for s,n in enumerate((2,2,3,3,4,4,7,7)):
        chain=[]
        for j in range(n):
            origin=torch.eye(4);origin[:3,3]=torch.tensor([.025*(j%2),.008*(s%2),.035])
            axis=torch.zeros(3);axis[(j+1)%3]=1
            chain.append(dict(name=f'j{j}',type='revolute',q_index=j,origin=origin,axis=axis))
        specs.append(chain)
    a=SimpleNamespace(zero_level_optimizer='adam',lr=.05,lbfgs_lr=1.,max_iter=6,
          epsilon=.01,horizontal_fov_deg=50.,vertical_fov_deg=66.,z_min=.2,z_max=.7,delta=.01,
          q_init_per_p=64,optimize_top_k_per_sensor=16,q0_per_sensor=16,seed=0,
          joint_names=[f'j{i}' for i in range(7)],sensor_frames=[f's{i}' for i in range(8)])
    lo=torch.full((7,),-3.14);hi=-lo
    limits={name:(-3.14,3.14) for name in a.joint_names}
    qi=gen.sample_q_batch(np.random.default_rng(999),limits,a.joint_names,16)
    return gen,ex,specs,a,lo,hi,limits,qi


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.f=fixture()

    def test_target_value_gradient_all_eight_heads(self):
        gen,ex,specs,a,lo,hi,limits,qi=self.f;p=torch.tensor([.1,.08,.45])
        for s in range(8):
            q=torch.tensor(qi,requires_grad=True)
            old=ex.visibility_objective_batch(p,q,specs,a.horizontal_fov_deg,a.vertical_fov_deg,a.z_min,a.z_max,a.delta,s)[0]
            og=torch.autograd.grad(old.sum(),q)[0]
            new=fast.target_value(ex,p,q,specs,a,s);ng=torch.autograd.grad(new.sum(),q)[0]
            torch.testing.assert_close(new,old,atol=0,rtol=0)
            torch.testing.assert_close(ng,og,atol=0,rtol=0)

    def test_refinement_same_candidates_and_diagnostics_all_heads(self):
        gen,ex,specs,a,lo,hi,limits,qi=self.f;p=np.array([.1,.08,.45],np.float32)
        for s in range(8):
            old=ex.refine_q_to_zero_level(p,qi,specs,lo,hi,a,'cpu',s)
            new=fast.refine(ex,p,qi,specs,lo,hi,a,'cpu',s)
            for i,(x,y) in enumerate(zip(old,new)):
                self.assertEqual(fast.compare_arrays(x,y,f'output{i}',0),0)

    def test_full_eighty_updates_exact_on_cpu(self):
        gen,ex,specs,a,lo,hi,limits,qi=self.f;a=copy.copy(a);a.max_iter=80
        p=np.array([.1,.08,.45],np.float32)
        old=ex.refine_q_to_zero_level(p,qi,specs,lo,hi,a,'cpu',6)
        new=fast.refine(ex,p,qi,specs,lo,hi,a,'cpu',6)
        for i,(x,y) in enumerate(zip(old,new)):fast.compare_arrays(x,y,str(i),0)
        self.assertGreater(len(new[0]),0)

    def test_fov_calls_reduce_without_less_iterations(self):
        gen,ex,specs,a,lo,hi,limits,qi=self.f;a=copy.copy(a);a.max_iter=3
        p=np.array([.1,.08,.45],np.float32)
        with mock.patch.object(ex,'visibility_g_batch',wraps=ex.visibility_g_batch) as call:
            fast.refine(ex,p,qi,specs,lo,hi,a,'cpu',0)
            self.assertEqual(call.call_count,a.max_iter+2)
            self.assertEqual([len(c.args[2]) for c in call.call_args_list],[1]*4+[8])

    def test_inactive_joints_unchanged(self):
        gen,ex,specs,a,lo,hi,limits,qi=self.f;a=copy.copy(a);a.epsilon=10.
        new=fast.refine(ex,np.array([.1,.08,.45],np.float32),qi,specs,lo,hi,a,'cpu',0)
        np.testing.assert_array_equal(new[0][:,2:],new[5][:,2:])

    def test_patch_restored_after_exception(self):
        gen,ex,*_=self.f;old=gen.refine_q_to_zero_level
        with self.assertRaises(ValueError):
            with fast.installed(gen,ex):
                self.assertIsNot(gen.refine_q_to_zero_level,old);raise ValueError('intentional')
        self.assertIs(gen.refine_q_to_zero_level,old)

    def test_unsupported_optimizer_refused(self):
        gen,ex,specs,a,lo,hi,limits,qi=self.f;a=copy.copy(a);a.zero_level_optimizer='lbfgs'
        with self.assertRaises(ValueError):fast.refine(ex,np.zeros(3,np.float32),qi,specs,lo,hi,a,'cpu',0)

    def test_discrete_mask_change_is_fatal(self):
        with self.assertRaises(ValueError):fast.compare_arrays(np.array([True]),np.array([False]),'mask',100)

    def test_shape_dtype_change_is_fatal(self):
        with self.assertRaises(ValueError):fast.compare_arrays(np.zeros(1,np.float32),np.zeros(1,np.float64),'q',1)

    def test_nan_and_invalid_slot_rejected(self):
        with self.assertRaises(ValueError):fast.compare_arrays(np.array([np.inf]),np.array([-np.inf]),'q',1)
        with self.assertRaises(ValueError):fast.compare_arrays(np.array([np.nan]),np.array([np.nan]),'q',1)

    def test_full_point_store_layout_and_old_reader(self):
        gen,ex,specs,a,lo,hi,limits,qi=self.f;a=copy.copy(a);a.max_iter=12
        points=np.array([[.1,.08,.45],[.11,.09,.46]],np.float32)
        banks=[]
        for mode in ('old','fast'):
            data=gen.allocate_yiming_style_arrays(2,a.q0_per_sensor,7,8)
            data['x'][:]=points;data['k'][:]=[50,51]
            with fast.installed(gen,ex) if mode=='fast' else contextlib.nullcontext():
                for i,p in enumerate(points):
                    seeds=gen.sample_q_batch(np.random.default_rng(np.random.SeedSequence([a.seed,50+i])),limits,a.joint_names,a.q_init_per_p)
                    with mock.patch.object(gen,'compute_q0_occlusion',side_effect=AssertionError('No LOS')):
                        results=gen.generate_for_one_point(p,seeds,specs,lo,hi,None,a,'cpu')
                    sa=copy.copy(a);sa.seed=a.seed+1000003*50
                    for s,r in enumerate(results):
                        if r is not None:gen.store_sensor_block(data,i,50+i,p,s,r['q0'],r['g0'],r['sensor_margins0'],r['active_planes0'],r['occluded'],sa)
            banks.append(data)
        self.assertEqual(banks[0].keys(),banks[1].keys())
        for name in banks[0]:fast.compare_arrays(banks[0][name],banks[1][name],name,0)
        self.assertGreater(banks[1]['valid_fov'].sum(),0)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'bank.npz';data=banks[1]
            np.savez_compressed(path,x=data['x'],q=data['q'],k=data['k'],valid_fov=data['valid_fov'],
                                sensor_chain_masks=gen.SENSOR_CHAIN_MASKS,q_min=lo.numpy(),q_max=hi.numpy())
            reader=importlib.import_module('train_signed_visibility_cdf_pairwise_replace')
            with contextlib.redirect_stdout(io.StringIO()):ds=reader.VisibilityQ0Dataset(str(path),1,0)
            self.assertEqual((ds.P,ds.K,ds.J,ds.S),(2,16,7,8))

    def test_real_parity_gate_small_cpu(self):
        gen,ex,specs,a,lo,hi,limits,qi=self.f
        gate=fast.parity_gate(gen,ex,specs,lo,hi,limits,a,'cpu',np.array([[.1,.08,.45]],np.float32),[1])
        self.assertEqual(gate['status'],'PASS');self.assertEqual(len(gate['rows']),8)


class ControlTests(unittest.TestCase):
    def test_plan_all_shards_exactly_once(self):
        p=ctl.plan(2707);flat=sum(p,[])
        self.assertEqual(len(p),43);self.assertEqual(len(p[-1]),19)
        self.assertEqual(sorted(flat),list(range(2707)));self.assertEqual(p,ctl.plan(2707))

    def test_one_job_3090_and_4090(self):
        for name,typ in [('3090node3','3090'),('4090node2','4090')]:
            cmd=ctl.sbatch_command(Path('/out'),Path('/repo'),Path('/source'),name,'collect')
            self.assertEqual(cmd.count('sbatch'),1);self.assertIn('--nodes=1',cmd)
            self.assertIn('--gres=gpu:'+typ+':4',cmd);self.assertIn('--time=1-00:00:00',cmd)
            self.assertFalse(any('dependency' in x or 'array' in x for x in cmd))

    def test_multiple_nodes_refused(self):
        for x in ('3090node2,3090node3','4090node[1-2]','AUTO',''):
            with self.assertRaises(RuntimeError):ctl.node_spec(x)

    def test_busy_or_unreadable_queue_blocks(self):
        for rc,out in [(0,'18768 RUNNING vis_grid'),(1,'')]:
            with mock.patch.object(ctl.subprocess,'run',return_value=SimpleNamespace(returncode=rc,stdout=out)):
                with self.assertRaises(RuntimeError):ctl.empty_queue()

    def test_immutable_write_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'complete.json';ctl.write(p,{'ok':1})
            with self.assertRaises(FileExistsError):ctl.write(p,{'ok':2})
            self.assertEqual(ctl.read(p),{'ok':1})

    def test_legacy_pointer_quoted_spaces(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);code=root/'some code';code.mkdir();(code/'g25_common.py').touch()
            (root/'source.env').write_text("export CODE='"+str(code)+"'\n")
            self.assertEqual(ctl.legacy_code(root),code)

    def test_legacy_pointer_missing_fails(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(RuntimeError):ctl.legacy_code(td)

    def test_next_segment_skips_existing_shards(self):
        m={'segments':[[0,1],[2,3],[4]]};c=object();job={}
        with mock.patch.object(ctl,'stage_exists',side_effect=lambda c,j,s:s in (0,1,3)):
            self.assertEqual(ctl.choose_segment(m,c,job),1)
        with mock.patch.object(ctl,'stage_exists',return_value=True):
            self.assertIsNone(ctl.choose_segment(m,c,job))

    def test_terminal_state_required(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);ctl.write(root/'fast_segments_v1/latest_submission.json',{'job_id':2})
            with mock.patch.object(ctl.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='RUNNING')):
                with self.assertRaises(RuntimeError):ctl.previous_done(root)
            with mock.patch.object(ctl.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='COMPLETED')):
                ctl.previous_done(root)

    def test_sbatch_failure_never_retries(self):
        m={'source':'/source','segments':[[0]]};c=object();job={}
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(ctl,'empty_queue'),mock.patch.object(ctl,'previous_done'), \
             mock.patch.object(ctl,'prepare',return_value=(Path(td)/'manifest.json',m)), \
             mock.patch.object(ctl,'base',return_value=(None,c,None,None,job)), \
             mock.patch.object(ctl,'choose_segment',return_value=0), \
             mock.patch.object(ctl.subprocess,'run',return_value=SimpleNamespace(returncode=1,stdout='',stderr='rejected')) as call:
            with self.assertRaises(RuntimeError):ctl.submit(td,'/repo','4090node2','next')
            self.assertEqual(call.call_count,1);self.assertIn('--test-only',call.call_args.args[0])

    def test_one_real_sbatch_after_test_only(self):
        m={'source':'/source','segments':[[0]]};c=object();job={}
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(ctl,'empty_queue'),mock.patch.object(ctl,'previous_done'), \
             mock.patch.object(ctl,'prepare',return_value=(Path(td)/'manifest.json',m)), \
             mock.patch.object(ctl,'base',return_value=(None,c,None,None,job)), \
             mock.patch.object(ctl,'choose_segment',return_value=0), \
             mock.patch.object(ctl.subprocess,'run',side_effect=[SimpleNamespace(returncode=0,stdout='allowed',stderr=''),SimpleNamespace(returncode=0,stdout='456\n',stderr='')]) as call:
            ctl.submit(td,'/repo','4090node2','next')
            self.assertEqual(call.call_count,2)
            self.assertIn('--test-only',call.call_args_list[0].args[0])
            self.assertNotIn('--test-only',call.call_args_list[1].args[0])
            self.assertEqual(ctl.read(Path(td)/'fast_segments_v1/latest_submission.json')['job_id'],456)

    def test_corrupt_new_provenance_refused_old_shards_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'gate.json';ctl.write(p,{'status':'PASS'})
            source={'a':'b'};producer={'engine':fast.ENGINE,'source_sha256':source,'gate_path':str(p),'gate_sha256':ctl.sha(p)}
            c=SimpleNamespace(stage=lambda j,s,h:dict(data_sha256='hash',producer=producer) if s else dict(data_sha256='old'))
            result=ctl.engine_integrity(c,{},[0,1],{'source_sha256':source})
            self.assertEqual(result['0']['engine'],'legacy_original')
            p.write_text('{}')
            with self.assertRaises(RuntimeError):ctl.engine_integrity(c,{},[1],{'source_sha256':source})

    def test_real_sbatch_rejection_and_ambiguous_reply_not_retried(self):
        m={'source':'/source','segments':[[0]]};c=object();job={}
        for rc,reply in ((1,''),(0,'not-a-job-id')):
            with tempfile.TemporaryDirectory() as td, \
                 mock.patch.object(ctl,'empty_queue'),mock.patch.object(ctl,'previous_done'), \
                 mock.patch.object(ctl,'prepare',return_value=(Path(td)/'manifest.json',m)), \
                 mock.patch.object(ctl,'base',return_value=(None,c,None,None,job)), \
                 mock.patch.object(ctl,'choose_segment',return_value=0), \
                 mock.patch.object(ctl.subprocess,'run',side_effect=[SimpleNamespace(returncode=0,stdout='ok',stderr=''),SimpleNamespace(returncode=rc,stdout=reply,stderr='test')]) as call:
                with self.assertRaises(RuntimeError):ctl.submit(td,'/repo','4090node2','next')
                self.assertEqual(call.call_count,2)
                self.assertEqual(len(list((Path(td)/'fast_segments_v1/submissions').glob('attempt_*.json'))),1)
                self.assertFalse((Path(td)/'fast_segments_v1/latest_submission.json').exists())

    def test_freeze_leaves_original_records_and_reuses_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'formal').mkdir();(root/'formal/job.json').write_text('{"original":true}')
            (root/'source.env').write_text('original source pointer')
            old_hash=ctl.sha(root/'formal/job.json');pointer=ctl.sha(root/'source.env')
            c=SimpleNamespace(nshards=lambda cfg:2707);job={'cfg':{}}
            with mock.patch.object(ctl,'base',return_value=(Path('/legacy'),c,None,None,job)):
                path,m=ctl.prepare(root)
                path2,m2=ctl.prepare(root)
                self.assertEqual(m,m2);self.assertEqual(path,path2)
                self.assertEqual(ctl.sha(root/'formal/job.json'),old_hash)
                self.assertEqual(ctl.sha(root/'source.env'),pointer)
                self.assertEqual(len(m['segments']),43)
                (Path(m['source'])/'fast_core.py').write_text('corrupt')
                with self.assertRaises(RuntimeError):ctl.prepare(root)

    def test_partial_segment_finishes_without_final_merge(self):
        for complete in (False,True):
            with tempfile.TemporaryDirectory() as td:
                root=Path(td);m={'root':str(root),'segments':[[0,1]],'source_sha256':{}}
                merger=mock.Mock();proc=mock.Mock();proc.poll.return_value=0;proc.returncode=0;proc.wait.return_value=0
                with mock.patch.dict(os.environ,{'SLURM_JOB_ID':'99','SLURM_JOB_NUM_NODES':'1','CUDA_VISIBLE_DEVICES':'0,1,2,3','G25F_GPU':'4090'}), \
                     mock.patch.object(ctl,'verify_manifest',return_value=(m,object(),None,merger,{})), \
                     mock.patch.object(ctl,'stage_exists',side_effect=lambda c,j,s:complete or s==0), \
                     mock.patch.object(ctl,'engine_integrity'),mock.patch.object(ctl.subprocess,'run'), \
                     mock.patch.object(ctl.subprocess,'Popen',return_value=proc) as launch:
                    ctl.batch(root/'manifest.json',0,'collect')
                    self.assertEqual(launch.call_count,4)
                    result=ctl.read(root/'fast_segments_v1/runs/99/result.json')
                    self.assertEqual(result['status'],'SEGMENT_COMPLETE' if complete else 'SEGMENT_PAUSED')
                    merger.merge.assert_not_called()

    def test_incomplete_dataset_cannot_merge(self):
        with tempfile.TemporaryDirectory() as td:
            m={'root':td};merger=mock.Mock()
            with mock.patch.dict(os.environ,{'SLURM_JOB_ID':'99','SLURM_JOB_NUM_NODES':'1'}), \
                 mock.patch.object(ctl,'verify_manifest',return_value=(m,object(),None,merger,{})), \
                 mock.patch.object(ctl,'choose_segment',return_value=3):
                with self.assertRaises(RuntimeError):ctl.batch(Path(td)/'manifest.json',-1,'merge')
                merger.merge.assert_not_called()

    def test_shell_syntax_and_spool_path(self):
        subprocess.run(['bash','-n',str(HERE/'run.sh')],check=True)
        with tempfile.TemporaryDirectory() as td:
            spool=Path(td)/'slurm_script';spool.write_bytes((HERE/'run.sh').read_bytes())
            env=os.environ.copy();env.update(VIS_PYTHON='/bin/echo',G25F_SOURCE='/frozen/source',
                G25F_MANIFEST='/data/manifest.json',G25F_SEGMENT='2',G25F_MODE='collect',SLURM_JOB_ID='7')
            r=subprocess.run(['bash',str(spool),'_batch'],env=env,capture_output=True,text=True,check=True)
            self.assertIn('/frozen/source/run.py batch',r.stdout)
            self.assertNotIn(td+'/run.py',r.stdout)


if __name__=='__main__':unittest.main()
