"""CPU tests using the exact original runtime methods, extracted without ROS imports."""
from __future__ import annotations
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Dict, List, Sequence, Tuple
import unittest
import numpy as np
import torch
from budget_probe import instrument, execute, geometric_result, first_hit
from ie_common import sha, write, finish, stage, validate_starts, load
from report import comparison, stats, summarize

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('IE_TEST_REPO', str(HERE.parents[1])))
RUNTIME = REPO / 'src/care_visibility_cdf/scripts/per_sensor_visibility_runtime.py'


def legacy_algorithm():
    # Compile the actual methods, not a hand-reimplemented test solver. No ROS,
    # URDF, Pinocchio or checkpoint needed for these algorithm contract tests.
    source = RUNTIME.read_bytes()
    blob = hashlib.sha1(b'blob '+str(len(source)).encode()+b'\0'+source).hexdigest()
    if blob != 'c1de7ef95b4cdab8046882bfaf9ffdf62a0c1141':
        raise ValueError('Test runtime is not the frozen original source')
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PerSensorVisibilityRuntime')
    names = {'_clamp','_projection_step','_ascent_step','_refine_branch_root','_optimize_branch'}
    body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(body) == 5
    node = ast.ClassDef(name='OriginalAlgorithm', bases=[], keywords=[], body=body, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    ns = dict(torch=torch, math=math, np=np, Dict=Dict, List=List, Tuple=Tuple, Sequence=Sequence,
              DEFAULT_SENSOR_FRAMES=[f'S{i}' for i in range(8)])
    exec(compile(module, str(RUNTIME), 'exec'), ns)
    return ns['OriginalAlgorithm']


def probe_for(field, mask=None, lo=-3., hi=3.):
    base = legacy_algorithm()
    class TestProbe(base):
        def __init__(self):
            self.q_min = torch.full((7,), lo); self.q_max = torch.full((7,), hi)
            self.sensor_masks = torch.ones((8,7)) if mask is None else mask
            self.projection_iters=10; self.projection_damping=.5; self.projection_epsilon_f=.03
            self.projection_max_step_norm=.25; self.root_refine_iters=12; self.root_tolerance_f=.002
            self.branch_ascent_steps=1; self.branch_step_size=.05; self.branch_max_step_norm=.25
            self.branch_fallback_ascent_steps=8; self.value_calls=0; self.gradient_calls=0
        def branch_score(self, points, q, s):
            self.value_calls += 1
            with torch.no_grad():return float(field(points,q,s).reshape(-1).min())
        def branch_value_and_grad(self, points, q, s):
            self.gradient_calls += 1
            qv=q.detach().clone().requires_grad_(True)
            v=field(points,qv,s).reshape(-1).min()
            g=torch.autograd.grad(v,qv)[0]*self.sensor_masks[s]
            return float(v.detach()),g.detach()
    return TestProbe()


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_protocol_and_fixed_sources(self):
        cfg=json.loads((HERE/'protocol.json').read_text())
        self.assertEqual(cfg['budgets'],[1,2,3,5,10]);self.assertEqual(cfg['case_count'],22518)
        self.assertEqual(cfg['models'],['R1','B_keep']);legacy_algorithm()

    def test_one_budget_performs_at_most_one_gradient_update(self):
        p=instrument(probe_for(lambda x,q,s:q[:,0]))
        q=torch.tensor([-2.,0,0,0,0,0,0]);r,t=execute(p,torch.zeros(3),q,0,1)
        self.assertEqual(r['gradient_calls'],1);self.assertEqual(r['projection_calls'],1)
        self.assertEqual(r['termination'],'gradient_budget_exhausted')
        self.assertAlmostEqual(r['q_candidate'][0],-1.875)
        self.assertEqual(len(t),1)

    def test_full_and_sufficient_budget_match_uninstrumented(self):
        for distance in [-2.8,-.4,-.03,-.01,.01,.6]:
            q=torch.tensor([distance,0.,0,0,0,0,0]);x=torch.zeros(3)
            raw=probe_for(lambda x,q,s:q[:,0]);out=raw._optimize_branch(x.reshape(1,3),q.reshape(1,7),0)
            p=instrument(probe_for(lambda x,q,s:q[:,0]));full,_=execute(p,x,q,0,None)
            cap,_=execute(p,x,q,0,18)
            self.assertEqual(full['q_candidate'],out['q_candidate'])
            self.assertEqual(full['q_candidate'],cap['q_candidate'])
            self.assertEqual(full['value_calls'],raw.value_calls)
            self.assertEqual(full['gradient_calls'],raw.gradient_calls)
            self.assertEqual(full['root_source'],out['root_source'])

    def test_no_oracle_is_available_to_optimizer(self):
        p=instrument(probe_for(lambda x,q,s:q[:,0]+.1))
        r,t=execute(p,torch.zeros(3),torch.tensor([-.2,0,0,0,0,0,0]),0,None)
        a=geometric_result(r['q_candidate'],r['q_candidate'][0],[-3]*7,[3]*7)
        b=geometric_result(r['q_candidate'],r['q_candidate'][0]+10,[-3]*7,[3]*7)
        self.assertFalse(a['fov_pass']);self.assertTrue(b['fov_pass'])
        again,_=execute(p,torch.zeros(3),torch.tensor([-.2,0,0,0,0,0,0]),0,None)
        self.assertEqual(r['q_candidate'],again['q_candidate'])

    def test_near_zero_start_uses_ascent_not_extra_projection(self):
        p=instrument(probe_for(lambda x,q,s:q[:,0]))
        r,_=execute(p,torch.zeros(3),torch.tensor([-.01,0,0,0,0,0,0]),0,1)
        self.assertEqual(r['gradient_calls'],1);self.assertEqual(r['ascent_calls'],1)
        self.assertEqual(r['projection_calls'],0);self.assertEqual(r['termination'],'legacy_completed')
        self.assertGreater(r['q_candidate'][0],0)

    def test_fallback_best_iterate_and_max_eighteen_calls(self):
        p=instrument(probe_for(lambda x,q,s:q[:,0]-10.))
        r,_=execute(p,torch.zeros(3),torch.zeros(7),0,None)
        self.assertEqual(r['gradient_calls'],18);self.assertEqual(r['ascent_calls'],8)
        self.assertEqual(r['solution_mode'],'branch_best_effort_ascent')
        limited,_=execute(p,torch.zeros(3),torch.zeros(7),0,3)
        self.assertEqual(limited['gradient_calls'],3);self.assertEqual(limited['ascent_calls'],0)

    def test_bisection_counts_values_without_claiming_extra_gradient_steps(self):
        # Negative near a flat point; a capped Newton-like update crosses zero.
        field=lambda x,q,s:torch.sin(20*q[:,0])-.5
        q=torch.tensor([-.075,0,0,0,0,0,0]);p=instrument(probe_for(field))
        r,t=execute(p,torch.zeros(3),q,0,1)
        self.assertEqual(r['gradient_calls'],1)
        self.assertGreater(r['root_value_calls'],0)
        self.assertEqual([a['kind'] for a in t],['projection','refined_root'])
        self.assertEqual(r['termination'],'gradient_budget_exhausted')

    def test_masks_limits_and_zero_gradient_are_preserved(self):
        mask=torch.ones((8,7));mask[:,1:]=0
        field=lambda x,q,s:q[:,0]-2*q[:,1]-10
        q=torch.tensor([.95,.1,.2,.3,.4,.5,.6]);p=instrument(probe_for(field,mask,lo=-1.,hi=1.))
        r,_=execute(p,torch.zeros(3),q,0,3)
        self.assertLessEqual(r['q_candidate'][0],1.)
        self.assertTrue(np.allclose(r['q_candidate'][1:],q.numpy()[1:]))
        flat=instrument(probe_for(lambda x,q,s:q[:,0]*0.-1.))
        rr,_=execute(flat,torch.zeros(3),torch.zeros(7),0,None)
        self.assertEqual(rr['q_candidate'],[0.]*7)
        self.assertEqual(rr['noop_calls'],rr['gradient_calls'])

    def test_full_candidate_is_not_best_analytic_trace_point(self):
        trace=[dict(kind='projection',gradient_calls=1,q=[.2]*7,degenerate=False),
               dict(kind='ascent',gradient_calls=2,q=[.3]*7,degenerate=False)]
        hit,anno=first_hit(trace,[.01,-.01],[-3]*7,[3]*7)
        final=geometric_result([.3]*7,-.01,[-3]*7,[3]*7)
        self.assertEqual(hit,1);self.assertFalse(final['fov_pass'])
        self.assertTrue(anno[0]['fov_pass']);self.assertFalse(anno[1]['fov_pass'])

    def test_real_r1_architecture_with_frozen_parameters(self):
        arch=load('ie_test_arch',REPO/'experiments/hierarchical9_scratch50k_r012_v1/model.py')
        torch.manual_seed(7);m=arch.build_model('R1').eval().requires_grad_(False)
        saved={k:v.detach().clone() for k,v in m.state_dict().items()}
        def field(x,q,s):return m.forward_sensor(torch.cat((x.expand(len(q),-1),q),1),s)-.2
        p=instrument(probe_for(field));r,_=execute(p,torch.zeros(3),torch.zeros(7),6,3)
        self.assertLessEqual(r['gradient_calls'],3)
        self.assertTrue(np.isfinite(r['q_candidate']).all())
        self.assertTrue(all(torch.equal(v,saved[k]) for k,v in m.state_dict().items()))
        self.assertTrue(all(v.grad is None for v in m.parameters()))

    def test_primary_keeps_failed_samples_in_denominator(self):
        rows=[]
        for i,(a,b) in enumerate([(True,False),(False,True),(False,True),(False,False)]):
            row=dict(point_id=i,models={})
            for name,passed in [('R1',a),('B_keep',b)]:
                row['models'][name]={'1':dict(fov_pass=passed,solver_ms=1.,gradient_calls=1,value_calls=2,root_value_calls=0,
                    violation_m=0. if passed else .02,noop_calls=0,termination='gradient_budget_exhausted',
                    candidate_in_limits=True,candidate_g_m=.01 if passed else -.02)}
            rows.append(row)
        cfg=dict(point_count=4,bootstrap_repeats=400,bootstrap_seed=4,margin_thresholds_m=[0.,.01])
        r=summarize(rows,'1',cfg)
        self.assertEqual(r['models']['R1']['success_rate'],.25)
        self.assertEqual(r['models']['B_keep']['success_rate'],.5)
        self.assertEqual(r['models']['B_keep']['failure'],2)
        self.assertAlmostEqual(r['models']['B_keep']['violation_m']['mean'],.01)
        self.assertEqual(r['paired']['delta_pp'],25.)
        self.assertEqual(comparison(rows,'1',cfg),comparison(rows,'1',cfg))

    def test_invalid_geometry_is_rejected(self):
        with self.assertRaises(ValueError):geometric_result([0]*7,float('nan'),[-1]*7,[1]*7)
        self.assertFalse(geometric_result([2]*7,.1,[-1]*7,[1]*7)['fov_pass'])
        with self.assertRaises(ValueError):stats([1.,float('nan')])

    def test_no_first_hit_is_none_not_zero_steps(self):
        t=[dict(kind='projection',gradient_calls=1,q=[0.]*7,degenerate=False)]
        hit,_=first_hit(t,[-.01],[-1]*7,[1]*7)
        self.assertIsNone(hit);self.assertEqual(stats([])['n'],0)

    def test_atomic_no_overwrite_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);write(root/'x.json',dict(a=1));finish(root,count=1)
            self.assertEqual(stage(root)['count'],1)
            with self.assertRaises(FileExistsError):write(root/'x.json',dict(a=2))
            (root/'x.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Corrupt'):stage(root)

    def test_source_start_checks_reject_duplicates_and_invalid_cohort(self):
        r=dict(case_id=0,point_id=0,sensor=0,cohort='uniform',x=[0]*3,q_init=[0]*7,witness_q=[.1]*7,initial_g_m=-.01,witness_g_m=.01)
        cfg=dict(case_count=1,point_count=1)
        validate_starts([r],cfg,np.full(7,-1),np.ones(7))
        with self.assertRaises(ValueError):validate_starts([dict(r,cohort='wrong')],cfg,np.full(7,-1),np.ones(7))
        with self.assertRaises(ValueError):validate_starts([dict(r,case_id=1)],cfg,np.full(7,-1),np.ones(7))

    def test_full_regression_guard_rejects_changed_candidate(self):
        from evaluate import check_full
        a=dict(fov_pass=True,candidate_g_m=.1,q_candidate=[0.]*7,gradient_calls=2,value_calls=4,
               root_source='branch_projection_tolerance',solution_mode='branch_projection_root_ascent')
        check_full(a,a)
        with self.assertRaises(ValueError):check_full(dict(a,fov_pass=False),a)
        with self.assertRaises(ValueError):check_full(dict(a,value_calls=5),a)
        with self.assertRaises(ValueError):check_full(dict(a,q_candidate=[.1]*7),a)

    def test_report_end_to_end_with_all_denominators_and_resume(self):
        from report import merge
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);cfg=json.loads((HERE/'protocol.json').read_text())
            cfg.update(point_count=2,case_count=8,bootstrap_repeats=100)
            starts=[];refs=[];rows=[]
            for i in range(8):
                start=dict(case_id=i,point_id=i%2,sensor=i,cohort='local' if i<4 else 'uniform')
                starts.append(start);models={}
                for name in ('R1','B_keep'):
                    passed=i%3==0 or (name=='B_keep' and i%4==1)
                    sample=dict(budget=None,termination='legacy_completed',q_candidate=[0.]*7,
                        candidate_g_m=.01 if passed else -.01,candidate_in_limits=True,fov_pass=passed,
                        violation_m=0. if passed else .01,solver_ms=1.,gradient_calls=1,value_calls=2,
                        root_value_calls=0,projection_calls=1,ascent_calls=0,noop_calls=0,
                        root_source='branch_projection_tolerance',solution_mode='branch_projection_root_ascent',
                        first_hit_gradient_query_diagnostic=1 if passed else None,trace_diagnostic=[])
                    models[name]={str(k):dict(sample,budget=k) for k in cfg['budgets']}
                    models[name]['full']=sample
                rows.append(dict(start,models=models))
                refs.append(dict(case_id=i,models={n:models[n]['full'] for n in models}))
            for name,records in [('starts.jsonl',starts),('reference.jsonl',refs),('coverage.jsonl',[dict(status='WITNESS_CONFIRMED')]*16)]:
                (out/name).write_text(''.join(json.dumps(r)+'\n' for r in records))
            for rank in range(4):
                dest=out/'evaluation'/f'rank{rank}';dest.mkdir(parents=True)
                (dest/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows[rank::4]))
                finish(dest)
            job=dict(out=str(out),cfg=cfg,lo=[-1]*7,hi=[1]*7,code_hashes={},models={})
            import contextlib,io
            with contextlib.redirect_stdout(io.StringIO()):merge(job)
            manifest=stage(out/'report');data=json.loads((out/'report/report.json').read_text())
            self.assertEqual(data['groups']['all']['1']['n'],8)
            self.assertEqual(data['groups']['uniform']['full']['n'],4)
            self.assertIn('Uniform',(out/'report/summary.md').read_text())
            before=sha(out/'report/report.json')
            with contextlib.redirect_stdout(io.StringIO()):merge(job)
            self.assertEqual(before,sha(out/'report/report.json'))

    def test_shell_syntax_and_spool_absolute_source(self):
        script=HERE/'run.sh';subprocess.run(['bash','-n',str(script)],check=True)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);spool=root/'slurm_script';shutil_source=script.read_bytes();spool.write_bytes(shutil_source)
            code=root/'code';code.mkdir();(code/'evaluate.py').write_text("import sys\nassert sys.argv[1:]==['worker','--job','KNOWN_JOB']\nprint('SPOOL_OK')\n")
            env=os.environ.copy();env.update(IE_CODE=str(code),IE_JOB='KNOWN_JOB',IE_ROOT=str(root),VIS_PYTHON=__import__('sys').executable)
            p=subprocess.run(['bash',str(spool),'_worker'],cwd=root,env=env,text=True,capture_output=True)
            self.assertEqual(p.returncode,0,p.stderr);self.assertIn('SPOOL_OK',p.stdout)


if __name__=='__main__':
    unittest.main(verbosity=2)
