"""CPU tests. Synthetic geometry and scheduler fixtures are not HPC validation."""
from __future__ import annotations
import copy
import importlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile
import numpy as np
import torch
import g25_common as c
import collect
import merge
import run

REPO=Path(os.environ.get('G25_TEST_REPO',c.HERE.parents[1]))
CFG=c.read(c.HERE/'protocol.json')
# The production preflight NEVER installs a shim. CPU tests can import the actual
# upstream Torch math without URDF parsing by replacing only an unused class.
PARSER_SHIM=False
if importlib.util.find_spec('urdf_parser_py') is None:
    pkg=types.ModuleType('urdf_parser_py');mod=types.ModuleType('urdf_parser_py.urdf')
    mod.URDF=type('UnusedURDFParser',(),{});pkg.urdf=mod
    sys.modules['urdf_parser_py']=pkg;sys.modules['urdf_parser_py.urdf']=mod;PARSER_SHIM=True
sys.path.insert(0,str(REPO/c.SOURCE_DIR))
GEN=importlib.import_module('data_generator')
torch.set_num_threads(1)


def ref(k=4):
    r=dict(q_init_per_p=64,optimize_top_k_per_sensor=8,q0_per_sensor=k,
      zero_level_optimizer='lbfgs',max_iter=10,lr=.03,lbfgs_lr=1.,epsilon=.01,seed=0,
      joint_names=CFG['joint_names'],sensor_frames=CFG['sensor_frames'],q_min=CFG['q_min'],q_max=CFG['q_max'],
      sensor_chain_masks=GEN.SENSOR_CHAIN_MASKS.astype(int).tolist(),has_occlusion_labels=False,old_grid_shape=[30,30,24])
    r.update(CFG['fov']);return r


def fake_job(root,n=3,size=2):
    cfg=copy.deepcopy(CFG);cfg['grid_points']=n;cfg['points_per_shard']=size
    job=dict(out=str(root),deps=str(REPO),cfg=cfg,reference=ref(),code_hashes={},deps_sha256={})
    root.mkdir(exist_ok=True);(root/'shards').mkdir(exist_ok=True)
    np.save(root/'x.npy',c.grid(CFG)[:n],allow_pickle=False);c.write(root/'job.json',job)
    return job


def arrays(job,sid):
    ids=c.point_ids(job['cfg'],sid);x=np.load(Path(job['out'])/'x.npy')[ids]
    d=GEN.allocate_yiming_style_arrays(len(ids),job['reference']['q0_per_sensor'],7,8)
    d['x'][:]=x;d['k'][:]=ids
    # Leave one whole point unresolved, preserving its index.
    for i,pid in enumerate(ids):
        if pid==1:continue
        for s in range(8):
            q=np.zeros((1,7),np.float32);g=np.array([0.],np.float32)
            margins=np.full((1,8),.01,np.float32);planes=np.zeros((1,8),np.int16)
            GEN.store_sensor_block(d,i,int(pid),x[i],s,q,g,margins,planes,np.array([False]),c.args_for(job))
    d['valid_point_fov']=d['valid_fov'].any(axis=(1,2))
    return {k:d[k] for k in c.ROW_KEYS}


def save_shards(job):
    result=[]
    for sid in range(c.nshards(job['cfg'])):
        d=arrays(job,sid);p=Path(job['out'])/'shards'/f'{sid:05d}';p.mkdir()
        np.savez_compressed(p/'data.npz',**d)
        stats=collect.validate_arrays(d,c.point_ids(job['cfg'],sid),d['x'],job)
        c.write(p/'complete.json',dict(status='COMPLETE',shard_id=sid,point_ids=d['k'].tolist(),
           job_sha256=c.sha(Path(job['out'])/'job.json'),data_sha256=c.sha(p/'data.npz'),**stats))
        result.append(p/'data.npz')
    return result


def try_lock(path,q):
    with c.lock(path) as acquired:q.put(acquired)


class Tests(unittest.TestCase):
    def test_complete_integer_grid_bounds_spacing_and_order(self):
        x=c.grid(CFG);self.assertEqual(x.shape,(173225,3));self.assertEqual(len(np.unique(x,axis=0)),173225)
        np.testing.assert_array_equal(x[[0,-1]],np.array([[-.8,-.8,0],[.8,.8,1]],np.float32))
        for j,n in enumerate((65,65,41)):
            a=np.unique(x[:,j]);self.assertEqual(len(a),n);np.testing.assert_allclose(np.diff(a),.025,atol=1e-7)
        np.testing.assert_array_equal(x[:41,2],np.arange(0,1025,25,dtype=np.float64).astype(np.float32)/1000)

    def test_shards_complete_no_duplicates_with_partial_last(self):
        ids=np.concatenate([c.point_ids(CFG,i) for i in range(c.nshards(CFG))])
        np.testing.assert_array_equal(ids,np.arange(173225));self.assertEqual(c.nshards(CFG),2707)
        self.assertEqual(len(c.point_ids(CFG,2706)),41)
        with self.assertRaises(ValueError):c.point_ids(CFG,2707)

    def test_four_eight_worker_seed_and_store_downsample_identity(self):
        def sample(pid):
            limits=dict(zip(CFG['joint_names'],zip(CFG['q_min'],CFG['q_max'])))
            return GEN.sample_q_batch(np.random.default_rng(np.random.SeedSequence([0,pid])),limits,CFG['joint_names'],32)
        for pid in (0,1,64,173224):
            np.testing.assert_array_equal(sample(pid),sample(pid))
            q=sample(pid);global_seed=1000003*(pid+1)+1009*5
            local_i=pid%64;adjusted=1000003*(pid-local_i)+1000003*(local_i+1)+1009*5
            np.testing.assert_array_equal(GEN.greedy_farthest_downsample(q,4,global_seed),GEN.greedy_farthest_downsample(q,4,adjusted))
        self.assertFalse(np.array_equal(sample(0),sample(1)))

    def test_reference_metadata_inherits_not_parser_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'ref.npz';r=ref(500);r['optimize_top_k_per_sensor']=1000;r['q_init_per_p']=20000
            np.savez(p,**r,grid_shape=[30,30,24]);out=c.reference_metadata(p,CFG)
            self.assertEqual(out['q0_per_sensor'],500);self.assertEqual(out['optimize_top_k_per_sensor'],1000)
            self.assertEqual(out['q_init_per_p'],20000)

    def test_reference_missing_parameter_is_fatal(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'ref.npz';r=ref();del r['epsilon'];np.savez(p,**r,grid_shape=[30,30,24])
            with self.assertRaisesRegex(ValueError,'missing'):c.reference_metadata(p,CFG)

    def test_reference_fov_joint_mask_mismatch_fails(self):
        for k,v in [('delta',.02),('q_min',[-3.14]*7),('has_occlusion_labels',True)]:
            with tempfile.TemporaryDirectory() as td:
                p=Path(td)/'r.npz';r=ref();r[k]=v;np.savez(p,**r,grid_shape=[30,30,24])
                with self.assertRaises(ValueError):c.reference_metadata(p,CFG)

    def test_atomic_write_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.json';c.write(p,{'x':1})
            with self.assertRaises(FileExistsError):c.write(p,{'x':2})
            self.assertEqual(c.read(p),{'x':1})

    def test_process_lock_excludes_second_process(self):
        with tempfile.TemporaryDirectory() as td:
            ctx=mp.get_context('spawn');q=ctx.Queue();path=str(Path(td)/'lock')
            with c.lock(path) as got:
                self.assertTrue(got);p=ctx.Process(target=try_lock,args=(path,q));p.start()
                self.assertFalse(q.get(timeout=20));p.join(20);self.assertEqual(p.exitcode,0)
            p=ctx.Process(target=try_lock,args=(path,q));p.start();self.assertTrue(q.get(timeout=20));p.join(20)

    def test_compact_schema_preserves_unresolved_points(self):
        with tempfile.TemporaryDirectory() as td:
            job=fake_job(Path(td));d=arrays(job,0)
            s=collect.validate_arrays(d,d['k'],d['x'],job)
            self.assertEqual(s['points'],2);self.assertEqual(s['points_with_q0'],1)
            self.assertEqual(s['unresolved_sensor_pairs'],[1]*8)

    def test_invalid_boundary_or_joint_limit_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            j=fake_job(Path(td));d=arrays(j,0);d['g'][0,0,0]=.2
            with self.assertRaises(ValueError):collect.validate_arrays(d,d['k'],d['x'],j)
            d=arrays(j,0);d['q'][0,0,0,0]=10
            with self.assertRaises(ValueError):collect.validate_arrays(d,d['k'],d['x'],j)

    def test_streaming_npz_all_fields_roundtrip_and_headers(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);j=fake_job(root);paths=save_shards(j);out=root/'merged.npz'
            merge.stream_npz(out,paths,3,c.metadata(j));merge.archive_check(out,j)
            with np.load(out,allow_pickle=False) as z:
                for key in c.ROW_KEYS:
                    expect=np.concatenate([arrays(j,s)[key] for s in range(2)])
                    np.testing.assert_array_equal(z[key],expect)
                self.assertEqual(z['q'].shape,(3,4,7,8));self.assertFalse(bool(z['has_occlusion_labels']))
            with self.assertRaises(FileExistsError):merge.stream_npz(out,paths,3,c.metadata(j))

    def test_existing_R1_dataset_reader_accepts_streamed_bank(self):
        old=importlib.import_module('train_signed_visibility_cdf_pairwise_replace')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);j=fake_job(root);paths=save_shards(j);dest=root/'new.npz'
            merge.stream_npz(dest,paths,3,c.metadata(j))
            ds=old.VisibilityQ0Dataset(str(dest),val_count=1,seed=0)
            self.assertEqual((ds.P,ds.K,ds.J,ds.S),(3,4,7,8))
            self.assertEqual(set(ds.train_indices_np)|set(ds.val_indices_np),{0,2})
            np.testing.assert_array_equal(ds.sensor_masks_cpu.numpy(),GEN.SENSOR_CHAIN_MASKS)

    def test_final_merge_is_atomic_and_verifiable(self):
        with tempfile.TemporaryDirectory() as td:
            j=fake_job(Path(td));save_shards(j)
            with mock.patch.object(merge,'verify_job'):
                self.assertTrue(merge.merge(j));merge.verify_dataset(j)
            final=c.read(Path(td)/'dataset/complete.json')
            self.assertEqual(final['points'],3);self.assertFalse(final['training_ready'])
            self.assertEqual(final['unresolved_x_sensor_by_sensor'],[1]*8)

    def test_corrupt_completed_shard_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            j=fake_job(Path(td));p=save_shards(j)[0];c.stage(j,0)
            with p.open('ab') as f:f.write(b'bad')
            with self.assertRaisesRegex(ValueError,'Corrupt'):c.stage(j,0)

    def test_merge_missing_shard_never_publish_complete(self):
        with tempfile.TemporaryDirectory() as td:
            j=fake_job(Path(td));save_shards(j);import shutil;shutil.rmtree(Path(td)/'shards/00001')
            with mock.patch.object(merge,'verify_job'):
                self.assertFalse(merge.merge(j))
            self.assertFalse((Path(td)/'dataset').exists())

    def test_scheduler_eight_gpus_are_two_single_node_jobs(self):
        self.assertEqual(run.parse_nodes('3090node2,3090node3'),['3090node2','3090node3'])
        for n in run.parse_nodes('3090node2,3090node3'):
            a=run.sbatch_args('/output','/repo',n,Path('/frozen'),'3-00:00:00')
            self.assertIn('--nodes=1',a);self.assertIn('--gres=gpu:3090:4',a);self.assertIn('--exclusive',a)
        for n in ('3090node1','3090node2,3090node2','3090node[2-3]'):
            with self.assertRaises(ValueError):run.parse_nodes(n)

    def test_previous_active_and_unknown_state_stop_submission(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);c.write(root/'latest_submission.json',{'jobs':[{'id':123}]})
            with mock.patch.object(run.subprocess,'run',return_value=types.SimpleNamespace(stdout='RUNNING\n',returncode=0)):
                with self.assertRaisesRegex(ValueError,'active'):run.ensure_idle(root)
            with mock.patch.object(run.subprocess,'run',return_value=types.SimpleNamespace(stdout='',returncode=0)):
                with self.assertRaisesRegex(ValueError,'terminal'):run.ensure_idle(root)

    def test_actual_original_collector_synthetic_chain_cpu(self):
        # Actual existing Torch sampler/refiner, synthetic kinematic chains;
        # no production URDF parsing and no GPU claims.
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);j=fake_job(root,n=1,size=1);np.save(root/'x.npy',np.array([[.2,0,.35]],np.float32))
            dev=torch.device('cpu');lo=torch.tensor(CFG['q_min']);hi=torch.tensor(CFG['q_max'])
            spec={'name':'joint1','type':'revolute','q_index':0,'origin':torch.eye(4),'axis':torch.tensor([0.,1.,0.])}
            specs=[[spec] for _ in range(8)];limits=dict(zip(CFG['joint_names'],zip(CFG['q_min'],CFG['q_max'])))
            collect.collect_shard(j,0,GEN,specs,lo,hi,limits,dev)
            m=c.stage(j,0);self.assertGreater(m['valid_q0'],0);self.assertLessEqual(m['fov_recheck_max_abs_m'],2e-6)

    def test_shell_worker_uses_frozen_path_not_slurm_spool(self):
        script=c.HERE/'run.sh';subprocess.run(['bash','-n',str(script)],check=True)
        s=script.read_text();self.assertLess(s.index('if [[ "$MODE" == _node'),s.index('HERE='))
        self.assertIn('$G25_CODE/run.py',s)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'frozen').mkdir();(root/'spool').mkdir()
            (root/'frozen/run.py').write_text('import sys\nassert sys.argv[1]=="node"\nprint("SPOOL_OK")\n')
            import shutil;shutil.copyfile(script,root/'spool/slurm_script')
            env=os.environ.copy();env.update(VIS_PYTHON=sys.executable,G25_CODE=str(root/'frozen'),G25_JOB='/job.json',G25_ROOT=str(root),G25_SLOT_BASE='4',SLURM_JOB_ID='999')
            out=subprocess.run(['bash',str(root/'spool/slurm_script'),'_node'],env=env,capture_output=True,text=True)
            self.assertEqual(out.returncode,0,out.stderr);self.assertIn('SPOOL_OK',out.stdout)

    def test_original_collector_source_fingerprints(self):
        for name,expected in c.PINNED_BLOBS.items():self.assertEqual(c.blob(REPO/c.SOURCE_DIR/name),expected)

if __name__=='__main__':
    print('URDF parser shim for unused test import:',PARSER_SHIM)
    unittest.main(verbosity=2)
