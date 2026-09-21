"""Synthetic tests only; these do not claim validation on the user's robot/data."""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from label_core import SolverConfig,old_bank_label,label_continuous
from cache_io import (import_bank,BankCache,prepare_queries,read_json,write_json,write_npz,open_job)
from repo_oracle import DEFAULT_JOINTS,DEFAULT_SENSORS,FOV,sha256_file
from pipeline import label_rows,add_union_labels,prepare_preflight,worker,merge
from dataset import OfflineLabelDataset

CFG=SolverConfig(starts=3,maxiter=100)


def plane_geometry(q):
    q=np.asarray(q);h=np.array([q[1]]);j=np.zeros((1,len(q)));j[0,1]=1
    return h,j


def rectangle_geometry(q):
    q=np.asarray(q);h=np.array([q[0],1-q[0],q[1],1-q[1],1.,1.])
    j=np.zeros((6,len(q)));j[0,0]=1;j[1,0]=-1;j[2,1]=1;j[3,1]=-1
    return h,j


def circle_geometry(q):
    q=np.asarray(q);h=np.array([q[0]**2+q[1]**2-1]);j=np.zeros((1,len(q)));j[0,:2]=2*q[:2]
    return h,j


class CoreTests(unittest.TestCase):
    def test_old_floor_and_empty(self):
        x=old_bank_label([0,0],[[0,0]],[1,1],1)
        self.assertAlmostEqual(x['value'],1e-4,places=10)
        np.testing.assert_array_equal(x['grad'],[0,0]);self.assertFalse(x['grad_regular'])
        self.assertFalse(old_bank_label([0,0],np.empty((0,2)),[1,1],1)['valid'])

    def test_sparse_bank_example(self):
        q=np.array([0.,-.01]);bank=np.array([[.05,0.]])
        old=old_bank_label(q,bank,[1,1],-1)
        self.assertAlmostEqual(old['value'],-np.sqrt(.0026),places=7)
        r=label_continuous(q,bank,[1,1],[-1,-1],[1,1],plane_geometry,CFG)
        self.assertTrue(r['value_valid']);self.assertTrue(r['grad_valid'])
        self.assertAlmostEqual(r['value'],-.01,places=8)
        np.testing.assert_allclose(r['q_star'],[0,0],atol=1e-7)
        np.testing.assert_allclose(r['grad'],[0,1],atol=1e-6)

    def test_inside_sign(self):
        r=label_continuous([0,.07],[[.5,0]],[1,1],[-1,-1],[1,1],plane_geometry,CFG)
        self.assertTrue(r['grad_valid']);self.assertAlmostEqual(r['value'],.07,places=7)
        np.testing.assert_allclose(r['grad'],[0,1],atol=1e-6)

    def test_zero_boundary_normal_not_zero(self):
        r=label_continuous([0,0],[[.5,0]],[1,1],[-1,-1],[1,1],plane_geometry,CFG)
        self.assertTrue(r['grad_valid']);self.assertEqual(r['value'],0)
        np.testing.assert_allclose(r['grad'],[0,1],atol=1e-6)

    def test_curved_boundary_projection(self):
        r=label_continuous([2.,.3],[[1,0],[0,1],[-1,0]],[1,1],[-3,-3],[3,3],circle_geometry,CFG)
        self.assertTrue(r['value_valid']);self.assertTrue(r['grad_valid'])
        self.assertAlmostEqual(r['distance_rad'],np.hypot(2,.3)-1,places=6)
        np.testing.assert_allclose(r['q_star'],np.array([2,.3])/np.hypot(2,.3),atol=1e-5)

    def test_ambiguous_nearest_gradient_masked(self):
        r=label_continuous([0.,0.],[[1,0],[-1,0],[0,1]],[1,1],[-2,-2],[2,2],circle_geometry,replace(CFG,starts=4))
        self.assertTrue(r['value_valid']);self.assertTrue(r['ambiguity']);self.assertFalse(r['grad_valid'])
        self.assertFalse(r['global_nearest_certified'])

    def test_all_other_fov_constraints_enforced(self):
        r=label_continuous([-.2,-.3],[[0,0],[1,0],[0,1]],[1,1],[-2,-2],[2,2],rectangle_geometry,CFG)
        self.assertTrue(r['value_valid'])
        np.testing.assert_allclose(r['q_star'],[0,0],atol=1e-6)
        self.assertAlmostEqual(r['distance_rad'],np.hypot(.2,.3),places=6)
        # Corner candidates may have a valid distance but are conservatively not normal-supervised.
        self.assertFalse(r['grad_valid'])

    def test_mask_keeps_inactive_joint(self):
        q=[.73,-.2,.91]
        r=label_continuous(q,[[0,0,-1],[1,0,1]],[0,1,0],[-2]*3,[2]*3,plane_geometry,CFG)
        self.assertTrue(r['grad_valid'])
        np.testing.assert_allclose(r['q_star'],[.73,0,.91],atol=1e-6)
        np.testing.assert_allclose(r['grad'],[0,1,0],atol=1e-6)

    def test_no_boundary_failure_not_zero(self):
        def absent(q): return np.array([1.]),np.zeros((1,len(q)))
        r=label_continuous([0,0],[[.5,0]],[1,1],[-1,-1],[1,1],absent,CFG)
        self.assertFalse(r['value_valid']);self.assertTrue(np.isnan(r['value']))
        self.assertEqual(r['status'],'NO_FEASIBLE_BOUNDARY_FOUND')

    def test_no_query_clamping(self):
        with self.assertRaises(ValueError):
            label_continuous([0,2],[[0,0]],[1,1],[-1,-1],[1,1],plane_geometry,CFG)

    def test_old_distance_continuous_sign_jump(self):
        left=old_bank_label([0,-1e-6],[[.05,0]],[1,1],-1)['value']
        right=old_bank_label([0,1e-6],[[.05,0]],[1,1],1)['value']
        self.assertGreater(right-left,.099)

    def test_near_boundary_sign_guard(self):
        r=label_continuous([0,1e-7],[[.5,0]],[1,1],[-1,-1],[1,1],plane_geometry,CFG)
        self.assertFalse(r['value_valid'])
        self.assertFalse(r['grad_valid'])

    def test_fk64_dtype_and_analytic_derivative(self):
        import torch
        from repo_oracle import RepoOracle
        oracle=RepoOracle.__new__(RepoOracle)
        origin=torch.eye(4,dtype=torch.float64);origin[0,3]=1.
        oracle.double_specs=[[dict(q_index=0,type='revolute',origin=origin,axis=torch.tensor([0.,0.,1.],dtype=torch.float64)),
                              dict(q_index=1,type='prismatic',origin=torch.eye(4,dtype=torch.float64),axis=torch.tensor([1.,0.,0.],dtype=torch.float64))]]
        q=torch.tensor([.3,.8],dtype=torch.float64,requires_grad=True)
        t=oracle._fk64(q,0)
        self.assertEqual(t.dtype,torch.float64)
        np.testing.assert_allclose(t[:3,3].detach().numpy(),[1+.8*np.cos(.3),.8*np.sin(.3),0],atol=1e-12)
        grad=torch.autograd.grad(t[1,3],q)[0]
        np.testing.assert_allclose(grad.detach().numpy(),[.8*np.cos(.3),np.sin(.3)],atol=1e-12)

    def test_nearest_at_joint_limit_masks_gradient(self):
        def diagonal(q):
            return np.array([q[0]+q[1]-1.5]),np.array([[1.,1.]])
        r=label_continuous([0.,-.8],[[.5,1],[1,.5]],[1,1],[-1,-1],[1,1],diagonal,CFG)
        self.assertTrue(r['value_valid'])
        np.testing.assert_allclose(r['q_star'],[1,.5],atol=1e-6)
        self.assertFalse(r['grad_valid'])


class SyntheticOracle:
    """Test double only, explicitly not a CAREPlanner URDF oracle."""
    def __init__(self,*args,**kwargs):
        self.identity=dict(test_fixture='synthetic_rectangle_not_robot')
    def geometry(self,x,s): return rectangle_geometry
    def reference_margins(self,x,qs):
        qs=np.asarray(qs).reshape(-1,7)
        return np.array([[rectangle_geometry(q)[0].min()]*8 for q in qs])
    def verify(self,*args): return dict(status='PASS',test_fixture='synthetic_not_robot')


def make_fixture(path):
    p,k=12,4
    x=np.zeros((p,3),np.float32);x[:,0]=np.arange(p)*.01
    q=np.zeros((p,k,7,8),np.float32)
    anchors=np.array([[0,.3],[1,.3],[.3,0],[.3,1]],np.float32)
    for s in range(8): q[:,:,:2,s]=anchors[None]
    arrays=dict(x=x,q=q,k=np.arange(p,dtype=np.int64),valid_fov=np.ones((p,k,8),bool),
                sensor_chain_masks=np.tile([1,1,0,0,0,0,0],(8,1)).astype(np.float32),
                q_min=np.full(7,-2,np.float32),q_max=np.full(7,2,np.float32),
                joint_names=np.array(DEFAULT_JOINTS),sensor_frames=np.array(DEFAULT_SENSORS))
    arrays.update({k:np.array(v,np.float32) for k,v in FOV.items()})
    write_npz(path,arrays)


class PipelineTests(unittest.TestCase):
    def test_prepare_reproducibility_original_split_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'source.npz';make_fixture(source)
            original=sha256_file(source)
            import_bank(source,root/'bank');bank=BankCache(root/'bank')
            for name in ('a','b'):
                prepare_queries(bank,root/name,train_x=2,val_x=1,uniform_per_x=1,near_per_x=1,shard_size=2)
            with np.load(root/'a/queries.npz') as a,np.load(root/'b/queries.npz') as b:
                for k in a.files: np.testing.assert_array_equal(a[k],b[k])
                self.assertFalse(set(a['x_index'][a['split']==0])&set(a['x_index'][a['split']==1]))
            self.assertEqual(sha256_file(source),original)
            with self.assertRaises(FileExistsError): prepare_queries(bank,root/'a')
            train,val=bank.original_split()
            expected=np.arange(12);np.random.default_rng(0).shuffle(expected)
            np.testing.assert_array_equal(val,expected[:1]);np.testing.assert_array_equal(train,expected[1:])

    def test_missing_sensor_never_creates_union_winner(self):
        a=dict(query_id=np.array([0]),support=np.ones((1,8),bool),
               paired_value_mask=np.ones((1,8),bool),paired_grad_mask=np.ones((1,8),bool),
               old_value=np.arange(8,dtype=float)[None],new_value=np.arange(8,dtype=float)[None],
               old_grad=np.zeros((1,8,7)),new_grad=np.zeros((1,8,7)))
        a['paired_value_mask'][0,7]=False
        add_union_labels(a)
        self.assertFalse(a['paired_union_value_mask'][0]);self.assertTrue(np.isnan(a['union_new_value'][0]))

    def test_resume_merge_cache_reader_and_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'source.npz';make_fixture(source)
            import_bank(source,root/'bank');bank=BankCache(root/'bank')
            out=root/'job'
            prepare_queries(bank,out,train_x=1,val_x=1,uniform_per_x=1,near_per_x=0,shard_size=1)
            with patch('pipeline.RepoOracle',SyntheticOracle):
                prepare_preflight(out,root,root/'fixture.urdf','cpu',replace(CFG,starts=2))
                worker(out,root,root/'fixture.urdf','cpu',rank=0,world=2)
                with self.assertRaises(RuntimeError): merge(out)
                worker(out,root,root/'fixture.urdf','cpu',rank=1,world=2)
                first=sha256_file(out/'shards/shard_000000/labels.npz')
                worker(out,root,root/'fixture.urdf','cpu',rank=0,world=2)
                self.assertEqual(first,sha256_file(out/'shards/shard_000000/labels.npz'))
            summary=merge(out);self.assertEqual(summary['query_count'],2)
            old=OfflineLabelDataset(out,'old','train');new=OfflineLabelDataset(out,'new','train')
            self.assertEqual(len(old),1);self.assertEqual(old[0]['inputs'].shape,(10,))
            np.testing.assert_array_equal(old[0]['inputs'],new[0]['inputs'])
            np.testing.assert_array_equal(old[0]['sensor_value_mask'],new[0]['sensor_value_mask'])
            self.assertTrue(np.isfinite(new[0]['sensor_value']).all())
            from audit_budget import audit
            with patch('audit_budget.RepoOracle',SyntheticOracle):
                report=audit(out,root,root/'fixture.urdf','cpu',samples=2,starts=3,maxiter=150)
                self.assertEqual(report['samples'],2)
                with self.assertRaises(FileExistsError):
                    audit(out,root,root/'fixture.urdf','cpu',samples=2,starts=3,maxiter=150)
            self.assertEqual(first,sha256_file(out/'shards/shard_000000/labels.npz'))
            with (out/'shards/shard_000000/labels.npz').open('ab') as f: f.write(b'corrupt')
            with self.assertRaises(ValueError): merge(out)

    def test_required_metadata_not_guessed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);write_npz(root/'bad.npz',dict(x=np.zeros((1,3))))
            with self.assertRaises(ValueError): import_bank(root/'bad.npz',root/'bank')


if __name__=='__main__': unittest.main(verbosity=2)
