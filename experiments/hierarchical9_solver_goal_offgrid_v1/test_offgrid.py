"""CPU tests; no Slurm, optimizer, real HPC files or GPU assumptions."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest import mock
import numpy as np
import common as c
import geometry as g
import report
import run


def cfg(n=4):
    p=deepcopy(c.config());p['points']=n;p['bootstrap_repetitions']=128
    p['witness_random_q']=4;p['uniform_attempts']=64;p['nearest_seed_x']=1
    p['seeds_per_nearest_x']=2
    return p

class Plane:
    def __call__(self,q):
        # Visibility is -0.4 < q[0] < +0.4, with inactive remaining joints.
        h=np.array([q[0]+.4,.4-q[0],1.,1.,1.,1.]);j=np.zeros((6,7));j[0,0]=1;j[1,0]=-1
        return h,j
class FakeExact:
    def geometry(self,x,s):return Plane()
    def canonical(self,x,qs,s):return np.array([Plane()(q)[0].min() for q in qs])


def populated_fixture(n=4):
    p=cfg(n);lo=-np.ones(7);hi=np.ones(7);mask=np.array([1,0,0,0,0,0,0])
    points=np.array([[i*.005,.13,.27] for i in range(n)],np.float32)
    coverage=[];starts=[]
    for pid,x in enumerate(points):
        for sid in range(8):
            info,rows=g.generate_pair(x,pid,sid,np.zeros((1,7)),FakeExact(),lo,hi,mask,p)
            coverage.append(info);starts+=rows
    starts.sort(key=lambda x:(x['point_id'],x['sensor'],x['cohort'],x.get('sample_slot',x.get('radius_rad'))))
    starts=[dict(r,case_id=i) for i,r in enumerate(starts)]
    return p,points,coverage,starts


def solved(starts):
    rows=[]
    for r in starts:
        models={}
        for arm in c.ARMS:
            ok=r['case_id']%4!=0 or arm=='SOLVER_GOAL'
            q=np.zeros(7) if ok else np.ones(7)*.9
            models[arm]={'fov_pass':ok,'candidate_in_limits':True,'q_candidate':q.tolist(),
               'candidate_g_m':.01 if ok else -.01,'initial_g_m':r['initial_g_m'],
               'failure_stage':'FOV_PASS_NOT_EXECUTION_CERTIFIED' if ok else 'ROOT_NOT_FOUND_AND_FOV_FAIL',
               'solver_ms':1.,'value_calls':10,'gradient_calls':4}
        rows.append({'case_id':r['case_id'],'start':r,'models':models})
    return rows


class Tests(unittest.TestCase):
    def test_01_frozen_protocol_and_maximum_scale(self):
        p=c.config();self.assertEqual(p['points'],512)
        self.assertEqual(p['points']*8*(p['uniform_per_pair']+len(p['local_radii_rad'])),28672)
        self.assertEqual(c.NEW_SHA,'5507b1d16c5d34f4a3e8c01526c7bf3f278a51c07df9f4d878aaee476cee48f5')

    def test_02_offgrid_sampling_is_reproducible_after_float32(self):
        x=np.array(np.meshgrid(np.linspace(-1,1,5),np.linspace(-1,1,5),np.linspace(0,1,4))).reshape(3,-1).T
        p=cfg(512);a,_=c.select_points(x,[[-1,1],[-1,1],[0,1]],p)
        b,_=c.select_points(x,[[-1,1],[-1,1],[0,1]],p)
        self.assertTrue(np.array_equal(a,b));self.assertEqual(a.dtype,np.float32)
        from scipy.spatial import cKDTree
        self.assertGreaterEqual(cKDTree(x).query(a)[0].min(),p['min_distance_to_old_x_m'])
        self.assertGreaterEqual(cKDTree(a).query(a,k=2)[0][:,1].min(),p['min_between_new_x_m'])

    def test_03_witness_search_uses_feasibility_not_solver_status(self):
        p=cfg();p['witness_optimizer_starts']=2
        seed=np.zeros((2,7));seed[:,0]=.8
        w,margin,tries=g.witness(Plane(),seed,-np.ones(7),np.ones(7),np.array([1,0,0,0,0,0,0]),p)
        self.assertGreater(margin,0);self.assertTrue(tries);self.assertTrue(np.array_equal(w[1:],seed[0,1:]))

    def test_04_bounded_failed_witness_is_unresolved_not_infeasible(self):
        class Impossible:
            def __call__(self,q):return -np.ones(6),np.zeros((6,7))
        class Never:
            def geometry(self,x,s):return Impossible()
            def canonical(self,x,qs,s):return -np.ones(len(qs))
        info,rows=g.generate_pair(np.zeros(3),0,0,np.zeros((2,7)),Never(),-np.ones(7),np.ones(7),np.ones(7),cfg())
        self.assertEqual(info['status'],'UNRESOLVED_WITNESS');self.assertEqual(rows,[])

    def test_05_local_pairs_are_new_geometry_checked_exact_offsets(self):
        p,points,cover,starts=populated_fixture()
        self.assertEqual(len(starts),4*8*7)
        for row in starts:
            self.assertLess(row['initial_g_m'],-p['side_margin_m']);self.assertGreater(row['witness_g_m'],p['side_margin_m'])
            if row['cohort']=='local':
                n=np.asarray(row['normal']);star=np.asarray(row['q_boundary']);rad=row['radius_rad']
                np.testing.assert_allclose(row['q_init'],star-rad*n,rtol=0,atol=1e-6)
                np.testing.assert_allclose(row['witness_q'],star+rad*n,rtol=0,atol=1e-6)
                self.assertLess(abs(row['boundary_residual_m']),1e-6)

    def test_06_local_joint_limit_excludes_without_replacement(self):
        p=cfg();p['local_radii_rad']=[.01,5.]
        _,_,_,start=populated_fixture(1)
        info,rows=g.generate_pair(np.zeros(3),0,0,np.zeros((1,7)),FakeExact(),-np.ones(7),np.ones(7),np.array([1,0,0,0,0,0,0]),p)
        self.assertEqual(info['local_count'],1)
        self.assertTrue(any(e.get('reason')=='JOINT_LIMIT' for e in info['local_exclusions'] if isinstance(e,dict)))
        self.assertEqual(len(rows),5)

    def test_07_atomic_no_overwrite_and_stage_hash_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);c.write(p/'a.json',{'a':1})
            with self.assertRaises(FileExistsError):c.write(p/'a.json',{'a':2})
            c.write(p/'complete.json',{'status':'COMPLETE','files':{'a.json':c.sha(p/'a.json')}})
            c.stage_complete(p,'complete.json');(p/'a.json').write_text('changed')
            with self.assertRaisesRegex(ValueError,'corrupted'):c.stage_complete(p,'complete.json')

    def test_08_full_input_freeze_records_all_unresolved_and_never_uses_models(self):
        p,points,cov,starts=populated_fixture(4)
        # Retain one predeclared x with no qualified cases; never replace that x.
        for x in cov:
            if x['point_id']==3:x.update(status='UNRESOLVED_WITNESS',local_count=0,uniform_count=0)
        starts=[r for r in starts if r['point_id']!=3]
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);c.write(out/'points.json',{'points':points.tolist()})
            job={'out':str(out),'cfg':p,'points_sha256':c.sha(out/'points.json'),'lo':[-1]*7,'hi':[1]*7,'masks':[[1,0,0,0,0,0,0]]*8}
            for rank in range(4):
                folder=out/'geometry'/f'rank{rank}';folder.mkdir(parents=True)
                c.write_lines(folder/'starts.jsonl',[{k:v for k,v in r.items() if k!='case_id'} for r in starts if r['point_id']%4==rank])
                c.write(folder/'coverage.json',{'rows':[r for r in cov if r['point_id']%4==rank]})
                c.write(folder/'complete.json',{'status':'COMPLETE','files':{f.name:c.sha(f) for f in folder.iterdir()}})
            with mock.patch.object(run,'verify_sources'):
                run.freeze(job);run.freeze(job)
            self.assertEqual(len(c.read(out/'coverage.json')['rows']),32)
            self.assertEqual(len(c.read(out/'points.json')['points']),4)
            self.assertFalse(c.read(out/'inputs_frozen.json')['models_loaded_during_generation'])
            self.assertEqual(len(list(c.jsonl(out/'starts.jsonl'))),168)

    def test_09_report_does_not_require_historical_1291_success_count(self):
        p,_,cov,starts=populated_fixture(4);rows=solved(starts)
        report.verify_rows(rows,starts);out=report.summarize(rows,cov,p)
        self.assertEqual(out['all']['n'],224);self.assertEqual(out['all']['paired']['net'],56)
        self.assertEqual(out['all']['delta_pp'],25.)
        self.assertNotEqual(out['all']['R1'],1291)
        self.assertIn('UNRESOLVED',report.markdown(out))

    def test_10_case_corruption_and_nonfinite_results_rejected(self):
        _,_,_,starts=populated_fixture(1);rows=solved(starts)
        rows[0]['start']=dict(rows[0]['start'],x=[9,9,9])
        with self.assertRaisesRegex(ValueError,'start changed'):report.verify_rows(rows,starts)
        rows=solved(starts);rows[0]['models']['R1']['candidate_g_m']=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite FOV'):report.verify_rows(rows,starts)

    def test_11_cluster_bootstrap_uses_x_not_cases(self):
        # All within-x repetitions must remain together. Four distinct effects.
        m=np.array([[100,100,0],[100,0,100],[100,50,50],[0,0,0]])
        rng=np.random.default_rng(12);draw=rng.multinomial(4,[.25]*4,size=2000)
        a=report.bootstrap_table(m,draw);b=report.bootstrap_table(m*7,draw)
        np.testing.assert_allclose(a['ci95_delta_pp'],b['ci95_delta_pp'])
        self.assertEqual(a['x_clusters_with_cases'],3)
        self.assertLess(a['bootstrap_valid_draws'],2000)

    def test_12_full_512_x_report_scale_and_sensor_counts(self):
        p=cfg(512);p['bootstrap_repetitions']=128
        cov=[];rows=[]
        for pid in range(512):
            for s in range(8):
                cov.append({'point_id':pid,'sensor':s,'status':'WITNESS_CONFIRMED','local_count':3,'uniform_count':4,'local_exclusions':[]})
                for j in range(7):
                    st={'point_id':pid,'sensor':s,'cohort':'local' if j<3 else 'uniform','case_id':len(rows)}
                    models={a:{'fov_pass':pid%3!=0 or a=='SOLVER_GOAL','candidate_in_limits':True,'candidate_g_m':.01 if pid%3!=0 or a=='SOLVER_GOAL' else -.01,'solver_ms':2.,'value_calls':5,'gradient_calls':3,'failure_stage':'RECORDED'} for a in c.ARMS}
                    rows.append({'start':st,'models':models,'case_id':st['case_id']})
        out=report.summarize(rows,cov,p)
        self.assertEqual(out['all']['n'],28672);self.assertEqual(out['local']['n'],12288)
        self.assertEqual(out['uniform']['n'],16384)
        self.assertEqual(out['S7']['n'],3584);self.assertEqual(out['coverage']['x_with_cases'],512)
        c.dumps(out)

    def test_13_seedbank_uses_old_data_as_guesses_and_is_readonly(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);x=np.array([[0,0,0],[1,1,1]],np.float32);np.save(root/'x.npy',x)
            for sp,xi in (('train',0),('val',1)):
                (root/sp).mkdir();np.save(root/sp/'boundary_x_index.npy',np.full(16,xi,dtype=np.int32))
                np.save(root/sp/'boundary_sensor.npy',np.repeat(np.arange(8,dtype=np.uint8),2))
                np.save(root/sp/'boundary_q.npy',np.ones((16,7),np.float32)*xi)
            before={str(f):c.sha(f) for f in root.rglob('*.npy')}
            seeds=g.SeedBank(root,cfg());a,prov=seeds.seeds([.1,.2,.1],7)
            self.assertEqual(a.shape,(2,7));self.assertEqual(prov[0][1],0)
            self.assertEqual(before,{str(f):c.sha(f) for f in root.rglob('*.npy')})

    def test_14_real_r1_model_frozen_still_has_input_gradients(self):
        import torch
        torch.set_num_threads(2)
        repo=Path(os.environ.get('OFFGRID_TEST_REPO',Path(__file__).resolve().parents[2]))
        path=repo/'experiments/hierarchical9_scratch50k_r012_v1/model.py'
        self.assertEqual(c.git_blob(path),c.PINS['experiments/hierarchical9_scratch50k_r012_v1/model.py'])
        mod=c.load('offgrid_test_real_model',path)
        with torch.random.fork_rng():
            torch.manual_seed(123);m=mod.build_model('R1').eval().requires_grad_(False)
            before=run.parameters_sha(m);q=torch.rand(16,7,requires_grad=True);x=torch.rand(16,3)
            for s in (0,3,6,7):
                y=m.forward_sensor(torch.cat([x,q],dim=1),s)
                grad=torch.autograd.grad(y.sum(),q,retain_graph=True)[0]
                self.assertEqual(grad.shape,(16,7));self.assertTrue(torch.isfinite(grad).all())
            self.assertEqual(before,run.parameters_sha(m))
            self.assertTrue(all(p.grad is None for p in m.parameters()))

    def test_15_syntax_and_shell_parser(self):
        for p in c.HERE.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
        subprocess.run(['bash','-n',str(c.HERE/'run.sh')],check=True)

    def test_16_git_blob_and_sha256_are_different(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x';p.write_bytes(b'hello\n')
            self.assertEqual(c.git_blob(p),'ce013625030ba8dba906f756967f9e9ca394464a')
            self.assertEqual(len(c.sha(p)),64)

    def test_17_empty_subgroup_reports_NA_not_zero(self):
        z=report.bootstrap_table(np.zeros((4,3)),np.ones((5,4)))
        self.assertEqual(z['n'],0);self.assertIsNone(z['delta_pp']);self.assertIsNone(z['r1_rate'])

    def test_18_freeze_failure_does_not_load_models(self):
        self.assertNotIn('torch.load',Path(g.__file__).read_text())
        import inspect
        self.assertNotIn('torch.load(',inspect.getsource(c.preflight))
        self.assertIn("stage_complete(out,'inputs_frozen.json')",inspect.getsource(run.evaluate))
        self.assertIn('pa.run_probe(',inspect.getsource(run.evaluate))
        self.assertNotIn('optimizer',inspect.getsource(run.evaluate))

if __name__=='__main__':unittest.main(verbosity=2)
