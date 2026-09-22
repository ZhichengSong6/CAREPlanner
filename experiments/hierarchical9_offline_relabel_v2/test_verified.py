"""Synthetic regressions + minimal recorded numerical-equivalence fixture.

NOT evidence of real-robot perturbed-distance verification.
"""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from verified_core import *
from review_smoke import prepare, parse_cases, read_source, merge, worker

CFG=SolverConfig(starts=3)
VC=VerifyConfig(probe_starts=2)


def plane(q):
    j=np.zeros((1,len(q)));j[0,1]=1
    return np.array([q[1]]),j


def rectangle(q):
    h=np.array([q[0],1-q[0],q[1],1-q[1]])
    j=np.zeros((4,len(q)));j[0,0]=1;j[1,0]=-1;j[2,1]=1;j[3,1]=-1
    return h,j


def circle(q):
    j=np.zeros((1,len(q)));j[0,:2]=2*np.array(q[:2])
    return np.array([q[0]**2+q[1]**2-1]),j


def make_c(aid,p,d,good=True):
    return dict(attempt_id=aid,q_star=p,distance_rad=d,qualified=good,
                stationarity_relative=0.,equality_residual_m=0.)


class SelectionTests(unittest.TestCase):
    def test_real_recorded_equivalence(self):
        c=json.loads((Path(__file__).parent/'fixtures/candidate_equivalence.json').read_text())['candidates']
        best,groups,amb,unc=select_equivalent(c,CFG,VC)
        self.assertTrue(best['qualified']);self.assertEqual(len(groups),1)
        self.assertFalse(amb or unc)
        self.assertAlmostEqual(best['distance_rad'],3.5848550592671455,places=12)

    def test_near_equal_distance_distinct_endpoints_not_merged(self):
        c=[make_c(0,[1,0],1),make_c(1,[-1,0],1+1e-8)]
        _,groups,amb,_=select_equivalent(c,CFG,VC)
        self.assertEqual(len(groups),2);self.assertTrue(amb)

    def test_materially_closer_uncertain_not_replaced(self):
        c=[make_c(0,[.5,0],.5,False),make_c(1,[1,0],1)]
        best,_,_,_=select_equivalent(c,CFG,VC)
        self.assertFalse(best['qualified'])

    def test_unconverged_competitor_not_asserted_true_ambiguity(self):
        c=[make_c(0,[1,0],1),make_c(1,[-1,0],1+1e-5,False)]
        _,_,amb,unc=select_equivalent(c,CFG,VC)
        self.assertFalse(amb);self.assertTrue(unc)

    def test_clustering_does_not_chain(self):
        c=[make_c(i,[i*1.5e-7,1],1) for i in range(3)]
        _,groups,_,_=select_equivalent(c,CFG,VC)
        self.assertEqual(len(groups),2)


class GradientTests(unittest.TestCase):
    def label(self,q,bank,geo=plane,mask=None,lo=None,hi=None):
        n=len(q)
        return label_verified(q,bank,mask or [1]*n,lo or [-2]*n,hi or [2]*n,geo,CFG,VC)

    def test_sparse_bank(self):
        r=self.label([0,-.01],[[.05,0]])
        self.assertTrue(r['grad_valid']);self.assertAlmostEqual(r['value'],-.01,places=8)
        np.testing.assert_allclose(r['grad'],[0,1],atol=1e-6)
        self.assertFalse(r['central_attempts_reused'])

    def test_inside(self):
        r=self.label([0,.3],[[.5,0]])
        self.assertTrue(r['grad_valid']);self.assertAlmostEqual(r['value'],.3,places=8)

    def test_corner_gradient_is_checked_not_blindly_unmasked(self):
        r=self.label([-.2,-.3],[[0,0],[1,0],[0,1]],rectangle)
        self.assertTrue(r['grad_valid']);self.assertEqual(r['verification']['completed_probes'],16)
        self.assertGreater(len(r['selected']['active_faces']),1)
        np.testing.assert_allclose(r['grad'],np.array([.2,.3])/np.hypot(.2,.3),atol=1e-6)

    def test_limit_projection_has_nonzero_distance_gradient(self):
        def diag(q): return np.array([q[0]+q[1]-1.5]),np.array([[1.,1.]])
        r=self.label([0,-.8],[[1,.5],[.5,1]],diag,lo=[-1,-1],hi=[1,1])
        self.assertTrue(r['grad_valid']);self.assertEqual(r['selected']['upper_bound_joints'],[0])
        np.testing.assert_allclose(r['grad'],np.array([1,1.3])/np.hypot(1,1.3),atol=1e-6)

    def test_inactive_joint_is_unchanged(self):
        r=self.label([.73,-.2,.91],[[0,0,-1],[1,0,1]],mask=[0,1,0])
        self.assertTrue(r['grad_valid'])
        np.testing.assert_allclose(r['q_star'],[.73,0,.91],atol=1e-6)
        np.testing.assert_allclose(r['grad'],[0,1,0],atol=1e-6)

    def test_smooth_zero_boundary(self):
        r=self.label([0,0],[[.5,0]])
        self.assertTrue(r['grad_valid']);self.assertEqual(r['value'],0)

    def test_zero_corner_stays_masked(self):
        r=self.label([0,0],[[0,0],[0,1]],rectangle)
        self.assertFalse(r['grad_valid']);self.assertIn('ZERO_DISTANCE_NONREGULAR_BOUNDARY',r['gradient_reasons'])

    def test_query_at_limit_cannot_use_two_sided_fd(self):
        r=self.label([2,-.2],[[0,0]])
        self.assertFalse(r['grad_valid']);self.assertIn('QUERY_TOO_CLOSE_TO_BOUNDARY_OR_LIMIT_FOR_TWO_SIDED_FD',r['gradient_reasons'])

    def test_curved_boundary(self):
        r=self.label([1.5,.3],[[1,0],[-1,0],[0,1]],circle)
        self.assertTrue(r['grad_valid']);self.assertAlmostEqual(r['value'],np.hypot(1.5,.3)-1,places=6)

    def test_true_multiple_projections_masked(self):
        r=self.label([0,0],[[1,0],[-1,0],[0,1]],circle)
        self.assertTrue(r['ambiguity']);self.assertFalse(r['grad_valid'])
        self.assertFalse(r['global_nearest_certified'])

    def test_no_boundary_not_zero(self):
        def absent(q):return np.array([1.]),np.zeros((1,len(q)))
        r=self.label([0,0],[[.5,0]],absent)
        self.assertFalse(r['value_valid']);self.assertTrue(np.isnan(r['value']))

    def test_near_boundary_sign_guard(self):
        r=self.label([0,1e-7],[[.5,0]])
        self.assertFalse(r['value_valid']);self.assertFalse(r['grad_valid'])

    def test_no_query_clamping(self):
        with self.assertRaises(ValueError):self.label([0,4],[[0,0]])

    def test_saved_feasible_flag_does_not_override_geometry(self):
        a=[dict(plane=0,q_star=[0,1],success=True,feasible=True)]
        r=screen_candidates([0,-.1],[1,1],[-2,-2],[2,2],plane,a)
        self.assertFalse(r['value_valid'])

    def test_central_attempts_are_reused(self):
        a=solve_attempts([0,-.1],[[0,0]],[1,1],[-2,-2],[2,2],plane,CFG)
        r=label_verified([0,-.1],[[0,0]],[1,1],[-2,-2],[2,2],plane,CFG,VC,attempts=a)
        self.assertTrue(r['central_attempts_reused']);self.assertTrue(r['grad_valid'])
        self.assertTrue(all('elapsed_ms' in row and 'geometry_calls' in row for row in a))

    def test_opposite_gradient_rejected_by_distance_fd(self):
        a=solve_attempts([0,-.1],[[0,0]],[1,1],[-2,-2],[2,2],plane,CFG)
        base=screen_candidates([0,-.1],[1,1],[-2,-2],[2,2],plane,a)
        base['gradient_candidate']=-base['gradient_candidate']
        def probe(q):
            b=solve_attempts(q,[[0,0]],[1,1],[-2,-2],[2,2],plane,CFG)
            return screen_candidates(q,[1,1],[-2,-2],[2,2],plane,b)
        r=verify_gradient(base,[0,-.1],[1,1],[-2,-2],[2,2],probe,CFG,VC)
        self.assertFalse(r['grad_valid']);self.assertIn('DISTANCE_FD_MISMATCH',r['gradient_reasons'])

    def test_one_sided_slopes_reject_symmetric_cusp(self):
        base=dict(value=-1.,distance_rad=1.,value_valid=True,grad_valid=False,
                  gradient_candidate=np.array([0.,1.]),gradient_reasons=['DISTANCE_DERIVATIVE_NOT_VERIFIED'])
        def probe(q):
            return dict(value=-1+abs(q[0])+q[1]-1,value_valid=True,ambiguity=False,
                        uncertain_competitor=False,q_star=np.array([0.,2.]),selected=None)
        r=verify_gradient(base,[0,1],[1,1],[-3,-3],[3,3],probe,CFG,VC)
        self.assertFalse(r['grad_valid']);self.assertIn('DISTANCE_FD_MISMATCH',r['gradient_reasons'])

    def test_new_closer_center_candidate_invalidates_stale_value(self):
        base=dict(value=-2.,distance_rad=2.,value_valid=True,grad_valid=False,
                  gradient_candidate=np.array([0.,1.]),gradient_reasons=['DISTANCE_DERIVATIVE_NOT_VERIFIED'])
        def probe(q):
            return dict(value=-1+q[1],value_valid=True,ambiguity=False,uncertain_competitor=False,
                        q_star=np.array([0.,1.]),selected=None)
        r=verify_gradient(base,[0,0],[1,1],[-3,-3],[3,3],probe,CFG,VC)
        self.assertFalse(r['value_valid']);self.assertFalse(r['grad_valid'])
        self.assertIn('CLOSER_BASE_CANDIDATE_DISCOVERED',r['gradient_reasons'])


class PipelineTests(unittest.TestCase):
    def test_case_parser(self):
        self.assertEqual(parse_cases('4:0,1:7'),[(4,0),(1,7)])
        for s in ('1:8','1:2,1:2','-1:0'):
            with self.assertRaises(ValueError):parse_cases(s)

    def test_full_synthetic_source_and_immutable_resume(self):
        import test_labels as v1tests
        import pipeline as v1pipeline
        from cache_io import import_bank, prepare_queries, BankCache
        from repo_oracle import sha256_file
        class Oracle(v1tests.SyntheticOracle):
            def __init__(self,*a,**k):
                self.identity=dict(urdf_sha256='synthetic',source_sha256={'fake':'synthetic'})
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); src=root/'v1';out=root/'v2';raw=root/'raw.npz'
            v1tests.make_fixture(raw);import_bank(raw,root/'bank');bank=BankCache(root/'bank')
            prepare_queries(bank,src,train_x=1,val_x=1,uniform_per_x=1,near_per_x=0,shard_size=1)
            with patch('pipeline.RepoOracle',Oracle):
                v1pipeline.prepare_preflight(src,root,root/'fake.urdf','cpu',CFG)
                v1pipeline.worker(src,root,root/'fake.urdf','cpu')
                v1pipeline.merge(src)
            before={str(p.relative_to(src)):sha256_file(p) for p in src.rglob('*') if p.is_file()}
            prepare(src,out,'1:0')
            with self.assertRaises(ValueError):prepare(src,out,'1:0')
            with self.assertRaises(ValueError):prepare(src,src/'bad','1:0')
            with patch('review_smoke.RepoOracle',Oracle):
                worker(src,out,root,root/'fake.urdf','cpu',0,1)
                first=sha256_file(out/'case_000001_S0/result.json')
                worker(src,out,root,root/'fake.urdf','cpu',0,1)
                self.assertEqual(first,sha256_file(out/'case_000001_S0/result.json'))
            r=merge(out);self.assertFalse(r['training_ready']);self.assertTrue(r['source_unchanged'])
            after={str(p.relative_to(src)):sha256_file(p) for p in src.rglob('*') if p.is_file()}
            self.assertEqual(before,after)
            with (src/'shards/shard_000000/labels.npz').open('ab') as f:f.write(b'corruption')
            with self.assertRaises(ValueError):read_source(src)


class ShellTests(unittest.TestCase):
    def test_submit_once_and_watch_never_submits(self):
        import os, subprocess
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);bins=root/'bin';bins.mkdir();source=root/'source';source.mkdir()
            (source/'dataset_index.json').write_text('{}')
            calls=root/'calls';shim=bins/'sbatch'
            shim.write_text('#!/bin/bash\necho called >> "$CALLS"\nprintf "99901\\n"\n')
            shim.chmod(0o755)
            # Isolated import stub for shell wiring only, NOT a geometry test.
            pkg=root/'urdf_parser_py';pkg.mkdir();(pkg/'__init__.py').write_text('')
            (pkg/'urdf.py').write_text('class URDF: pass\n')
            env={**os.environ,'PATH':str(bins)+':'+os.environ['PATH'],
                 'PYTHONPATH':str(root),'VIS_PYTHON':sys.executable,
                 'SOURCE_OUT':str(source),'VERIFY_BASE':str(root/'jobs'),
                 'OUT':str(source),'VERIFY_OUT':str(root/'output'),'CALLS':str(calls),'DEVICE':'cpu','WORKERS':'2'}
            here=Path(__file__).parent
            r=subprocess.run(['bash',str(here/'submit.sh')],env=env,text=True,capture_output=True)
            self.assertEqual(r.returncode,0,r.stderr+r.stdout)
            state=root/'jobs/latest_verify.env';self.assertTrue(state.is_file())
            for mode in ('summary','workers'):
                r=subprocess.run(['bash',str(here/'watch.sh'),str(state),mode],env=env,capture_output=True,text=True)
                self.assertEqual(r.returncode,0,r.stderr)
            self.assertEqual(calls.read_text().splitlines(),['called'])
            # A failed dependency check must not allocate another job.
            (root/'scipy.py').write_text('raise ImportError("missing SciPy fixture")\n')
            r=subprocess.run(['bash',str(here/'submit.sh')],env=env,capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0)
            self.assertEqual(calls.read_text().splitlines(),['called'])


if __name__=='__main__':unittest.main(verbosity=2)
