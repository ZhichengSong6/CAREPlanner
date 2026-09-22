"""Synthetic plumbing/geometry tests; do not imply a real robot or Slurm run."""
from __future__ import annotations
from dataclasses import replace
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from common import (core,old_io,source_bank,sha256_file,read_record,write_record,record_path,
                    open_run,geometry_identity,lock)
from prepare import Sampling,prepare,choose_points,audit_plan
from batch_pipeline import worker,merge,union_targets
from batch_dataset import PairedLabels
from repo_oracle import DEFAULT_JOINTS,DEFAULT_SENSORS,FOV


class FakeOracle:
    """Single affine boundary in 7D, two active coordinates: NOT robot geometry."""
    def __init__(self,*args,**kwargs):
        self.identity=dict(urdf_sha256='synthetic_plane',source_sha256={'test':'synthetic_plane'})
    def geometry(self,x,s):
        def g(q):
            j=np.zeros((1,7));j[0,1]=1
            return np.array([q[1]]),j
        return g
    def reference_margins(self,x,qs):
        return np.repeat(np.asarray(qs,float).reshape(-1,7)[:,1,None],8,axis=1)
    def verify(self,*args):return dict(status='PASS',synthetic=True)


def fixture(root):
    p,k=80,3;q=np.zeros((p,k,7,8),np.float32)
    q[:,0,0,:]=-.3;q[:,1,0,:]=.2;q[:,2,0,:]=.6
    a=dict(x=np.column_stack((np.arange(p)*.01,np.zeros((p,2)))).astype(np.float32),
        q=q,k=np.arange(p),valid_fov=np.ones((p,k,8),bool),
        sensor_chain_masks=np.tile([1,1,0,0,0,0,0],(8,1)).astype(np.float32),
        q_min=np.full(7,-2,np.float32),q_max=np.full(7,2,np.float32),
        joint_names=np.array(DEFAULT_JOINTS),sensor_frames=np.array(DEFAULT_SENSORS))
    a.update({k:np.array(v,np.float32) for k,v in FOV.items()})
    source=root/'original.npz';old_io.write_npz(source,a);old_io.import_bank(source,root/'bank')
    bank=old_io.BankCache(root/'bank');smoke=root/'source_smoke'
    old_io.prepare_queries(bank,smoke,train_x=1,val_x=1,uniform_per_x=1,near_per_x=0)
    old_io.write_json(smoke/'run_spec.json',dict(solver=core.legacy.asdict(core.SolverConfig(starts=2)),geometry_identity=FakeOracle().identity))
    old_io.write_json(smoke/'preflight.json',dict(status='PASS',synthetic=True))
    return smoke


SMALL=Sampling(train_x=2,val_x=1,uniform_per_x=1,normal_pairs_per_x=0,coverage_per_sensor=1,audit_per_stratum=1)


class SamplingTests(unittest.TestCase):
    def test_invalid_counts_no_implicit_full(self):
        with self.assertRaises(ValueError):replace(SMALL,train_x=-1).validate()
    def test_noninteger_sampling_rejected(self):
        with self.assertRaises(ValueError):replace(SMALL,train_x=2.5).validate()
    def test_source_spec_mutation_is_detected(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);src=fixture(r);out=r/'out';prepare(src,out,r,r/'robot',SMALL,oracle_factory=FakeOracle)
            with (src/'run_spec.json').open('a') as f:f.write('\n')
            with self.assertRaises(ValueError):open_run(out)
    def test_impossible_sensor_coverage(self):
        support=np.ones((10,8),bool);support[:,7]=False
        with self.assertRaises(ValueError):choose_points(np.arange(10),support,4,1,np.random.default_rng(0))
    def test_coverage_uses_existing_pool(self):
        support=np.ones((12,8),bool);pool=np.arange(2,12)
        got=choose_points(pool,support,5,2,np.random.default_rng(2))
        self.assertEqual(len(np.unique(got)),5);self.assertTrue(set(got)<=set(pool))
    def test_frozen_audit_is_stratified_and_outcome_independent(self):
        q=dict(query_id=np.arange(8),split=np.repeat([0,1],4),support=np.ones((8,8),bool),
               reference_g_m=np.repeat(np.tile([-.2,-.1,.1,.2],2)[:,None],8,axis=1))
        a=audit_plan(q,1,123,1e-6);q['solver_success']=np.zeros((8,8),bool)
        b=audit_plan(q,1,123,1e-6)
        self.assertEqual(a,b);self.assertEqual(len(a['tasks']),32)
        self.assertEqual(len(set(map(tuple,a['tasks']))),32)
    def test_no_replacement_for_empty_sign_stratum(self):
        q=dict(query_id=np.arange(2),split=np.array([0,1]),support=np.ones((2,8),bool),reference_g_m=-np.ones((2,8)))
        a=audit_plan(q,1,3,1e-6);self.assertEqual(len(a['tasks']),16)
        self.assertEqual(sum(x['selected'] for x in a['strata'] if x['sign']==1),0)
    def test_prepare_determinism_split_and_exclusion(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);src=fixture(r);before=sha256_file(src/'queries.npz')
            for name in ('a','b'):
                prepare(src,r/name,r,r/'robot',replace(SMALL,normal_pairs_per_x=1),oracle_factory=FakeOracle)
            qa=old_io.read_json(r/'a/audit_plan.json');qb=old_io.read_json(r/'b/audit_plan.json');self.assertEqual(qa,qb)
            with np.load(r/'a/queries.npz') as a,np.load(r/'b/queries.npz') as b,np.load(src/'queries.npz') as old:
                for k in a.files:np.testing.assert_array_equal(a[k],b[k])
                self.assertFalse(set(old['x_index']) & set(a['x_index']))
                self.assertEqual(len(a['query_id']),9)
                # Normal pair changes only the active chain; declared NOT exact witness.
                self.assertTrue(np.isfinite(a['q_query']).all())
            self.assertEqual(sha256_file(src/'queries.npz'),before)
            with self.assertRaises(FileExistsError):prepare(src,r/'a',r,r/'robot',SMALL,oracle_factory=FakeOracle)
    def test_original_source_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);src=fixture(r)
            for out in (src,src/'nested',r/'bank/nested',r):
                with self.assertRaises(ValueError):prepare(src,out,r,r/'robot',SMALL,oracle_factory=FakeOracle)
    def test_resume_rejects_new_sampling(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);src=fixture(r);out=r/'out'
            prepare(src,out,r,r/'robot',SMALL,oracle_factory=FakeOracle)
            prepare(src,out,r,r/'robot',SMALL,True,oracle_factory=FakeOracle)
            with self.assertRaises(ValueError):prepare(src,out,r,r/'robot',replace(SMALL,seed=42),True,oracle_factory=FakeOracle)
    def test_corrupt_frozen_query_fails(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);src=fixture(r);out=r/'out';prepare(src,out,r,r/'robot',SMALL,oracle_factory=FakeOracle)
            with (out/'queries.npz').open('ab') as f:f.write(b'bad')
            with self.assertRaises(ValueError):open_run(out)


class RecordTests(unittest.TestCase):
    def test_atomic_record_and_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'r';write_record(p,dict(task=[1,2],value=np.nan),'spec')
            r,_=read_record(p,'spec',[1,2]);self.assertIsNone(r['value'])
            with self.assertRaises(FileExistsError):write_record(p,dict(task=[1,2]),'spec')
            with self.assertRaises(ValueError):read_record(p,'other',[1,2])
            with (p/'result.json.gz').open('ab') as f:f.write(b'bad')
            with self.assertRaises(ValueError):read_record(p,'spec',[1,2])
    def test_lock_refuses_second_writer(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'lock'
            with lock(p):
                with self.assertRaises(RuntimeError):
                    with lock(p):pass
    def test_union_missing_sensor_cannot_disappear(self):
        a=dict(query_id=np.array([0]),support=np.ones((1,8),bool),paired_value_mask=np.ones((1,8),bool),
               paired_grad_mask=np.ones((1,8),bool),old_value=np.arange(8)[None],new_value=np.arange(8)[None],
               old_grad=np.ones((1,8,7)),new_grad=np.ones((1,8,7)))
        a['paired_value_mask'][0,7]=False;union_targets(a)
        self.assertFalse(a['paired_union_value_mask'][0]);self.assertTrue(np.isnan(a['union_new_value'][0]))
    def test_union_tie_masks_gradient_not_value(self):
        a=dict(query_id=np.array([0]),support=np.ones((1,8),bool),paired_value_mask=np.ones((1,8),bool),
               paired_grad_mask=np.ones((1,8),bool),old_value=np.ones((1,8)),new_value=np.ones((1,8)),
               old_grad=np.ones((1,8,7)),new_grad=np.ones((1,8,7)))
        union_targets(a);self.assertTrue(a['paired_union_value_mask'][0]);self.assertFalse(a['paired_union_grad_mask'][0])


class IntegrationTests(unittest.TestCase):
    def test_full_base_audit_pair_reader_resume_and_no_implicit_gradients(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);src=fixture(r);out=r/'out';source_hash=sha256_file(src/'queries.npz')
            spec=prepare(src,out,r,r/'robot',SMALL,oracle_factory=FakeOracle)
            worker(out,r,r/'robot','cpu',0,2,'base',oracle_factory=FakeOracle)
            with self.assertRaises(FileNotFoundError):merge(out)
            worker(out,r,r/'robot','cpu',1,2,'base',oracle_factory=FakeOracle)
            summary=merge(out)
            self.assertFalse(summary['audit_complete'])
            self.assertEqual(sum(s.get('verified_gradients',0) for s in summary['by_sensor_split'].values()),0)
            self.assertGreater(sum(s.get('candidate_gradients',0) for s in summary['by_sensor_split'].values()),0)
            with self.assertRaises(ValueError):PairedLabels(out/'base_cache')
            original={p:sha256_file(p) for p in (out/'base').glob('*/result.json.gz')}
            worker(out,r,r/'robot','cpu',0,1,'base',oracle_factory=FakeOracle)
            self.assertEqual(original,{p:sha256_file(p) for p in original})
            with self.assertRaises(FileNotFoundError):merge(out,True)
            worker(out,r,r/'robot','cpu',0,1,'audit',oracle_factory=FakeOracle)
            merged=merge(out,True)
            self.assertTrue(merged['audit_complete']);self.assertFalse(merged['training_ready'])
            self.assertGreater(sum(s.get('verified_gradients',0) for s in merged['by_sensor_split'].values()),0)
            self.assertEqual(sha256_file(src/'queries.npz'),source_hash)
            self.assertEqual(original,{p:sha256_file(p) for p in original})
            a=PairedLabels(out/'paired_cache','old','train');b=PairedLabels(out/'paired_cache','new','train')
            self.assertEqual(len(a),2)
            for i in range(len(a)):
                for k in ('query_id','inputs','sensor_value_mask','sensor_grad_mask'):
                    np.testing.assert_array_equal(a[i][k],b[i][k])
                self.assertTrue(np.isfinite(b[i]['sensor_grad']).all())
            merge(out,True)
            first=next((out/'paired_cache').glob('shard_*.npz'))
            with first.open('ab') as f:f.write(b'bad')
            with self.assertRaises(ValueError):PairedLabels(out/'paired_cache')
    def test_no_audit_selected_never_promotes_candidate(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);src=fixture(r);out=r/'out';cfg=replace(SMALL,audit_per_stratum=0)
            prepare(src,out,r,r/'robot',cfg,oracle_factory=FakeOracle)
            worker(out,r,r/'robot','cpu',stage='base',oracle_factory=FakeOracle)
            merge(out);summary=merge(out,True)
            self.assertEqual(summary['audit_selected'],0)
            data=PairedLabels(out/'paired_cache','new','train')
            for sample in data:self.assertFalse(sample['sensor_grad_mask'].any())
    def test_base_geometry_change_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);src=fixture(r);out=r/'out';prepare(src,out,r,r/'robot',SMALL,oracle_factory=FakeOracle)
            class Wrong(FakeOracle):
                def __init__(self,*a,**kw):super().__init__();self.identity['urdf_sha256']='changed'
            with self.assertRaises(ValueError):worker(out,r,r/'robot','cpu',oracle_factory=Wrong)


class ShellTests(unittest.TestCase):
    def test_submission_once_watch_readonly_and_old_out_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);bins=r/'bin';bins.mkdir();source=r/'source';source.mkdir();(source/'manifest.json').write_text('{}')
            for name,body in {'python':'exit 0','sbatch':'echo "$*" >> "$CALLS"; echo 123456',
                              'squeue':'echo 123456','sacct':'exit 0'}.items():
                p=bins/name;p.write_text('#!/usr/bin/env bash\n'+body+'\n');p.chmod(0o755)
            env=dict(os.environ,PATH=str(bins)+':'+os.environ['PATH'],VIS_PYTHON=str(bins/'python'),
                     SOURCE_OUT=str(source),BATCH_BASE=str(r/'results'),BATCH_OUT=str(r/'new'),
                     OUT=str(r/'must_not_touch'),CALLS=str(r/'calls'),DEVICE='cuda',WORKERS='2')
            here=Path(__file__).parent
            c=subprocess.run(['bash',str(here/'submit.sh'),'base'],env=env,text=True,capture_output=True)
            self.assertEqual(c.returncode,0,c.stderr)
            self.assertFalse((r/'must_not_touch').exists())
            self.assertIn('--gres=gpu:2',(r/'calls').read_text())
            self.assertNotEqual(subprocess.run(['bash',str(here/'submit.sh'),'base'],env=env,capture_output=True).returncode,0)
            self.assertEqual(len((r/'calls').read_text().splitlines()),1)
            state=r/'results/base_job_123456.env'
            for mode in ('workers','status','summary','plan'):
                x=subprocess.run(['bash',str(here/'watch.sh'),str(state),mode],env=env,capture_output=True)
                self.assertEqual(x.returncode,0,x.stderr)
            self.assertEqual(len((r/'calls').read_text().splitlines()),1)
            (bins/'python').write_text('#!/usr/bin/env bash\nexit 5\n')
            env['BATCH_OUT']=str(r/'other')
            self.assertNotEqual(subprocess.run(['bash',str(here/'submit.sh'),'base'],env=env,capture_output=True).returncode,0)
            self.assertEqual(len((r/'calls').read_text().splitlines()),1)
    def test_audit_rejects_missing_base(self):
        with tempfile.TemporaryDirectory() as td:
            env=dict(os.environ,BATCH_OUT=td,SOURCE_OUT=td,VIS_PYTHON=sys.executable)
            (Path(td)/'manifest.json').write_text('{}')
            p=Path(__file__).with_name('submit.sh')
            c=subprocess.run(['bash',str(p),'audit'],env=env,capture_output=True)
            self.assertNotEqual(c.returncode,0)

if __name__=='__main__':unittest.main()
