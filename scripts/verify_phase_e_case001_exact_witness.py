#!/usr/bin/env python3
"""Verify stored offline counterfactuals and independently resample their geometry."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from diagnose_phase_e_case001_exact_witness import (
    ROOT, RUN_ID, Geometry, continuity_methods, trajectory, numeric, values, sha,
    save, unpack_matrix, residuals)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--result-dir',type=Path,required=True)
    args=ap.parse_args();out=args.result_dir
    def read(name): return json.loads((out/name).read_text())
    alignment=read('alignment.json');base=read('baseline_problem.json')
    for path,digest in alignment['source_sha256'].items():
        p=Path(path) if Path(path).is_absolute() else ROOT/path
        assert sha(p)==digest,('changed input',path)
    checks=[]
    for path in sorted(out.glob('*problem.json')):
        p=json.loads(path.read_text())
        for key in ['P','c','A','b','x_l','x_u','max_iter','eps_abs','eps_rel']:
            assert p[key]==base[key],(path,key)
        original_rows=len(base['h_l'])
        np.testing.assert_array_equal(unpack_matrix(p['G'])[:original_rows],unpack_matrix(base['G']))
        assert p['h_l'][:original_rows]==base['h_l'] and p['h_u'][:original_rows]==base['h_u']
        solution=json.loads(path.with_name(path.name.replace('problem','solution')).read_text())
        if solution['status']=='solved':
            r=residuals(p,np.array(solution['x']))
            assert max(r.values())<1e-5,(path,r)
        else: r=None
        checks.append(dict(problem=path.name,status=solution['status'],residuals=r))
    assert alignment['reassembly_q_error_inf']<1e-8
    matched=[m for m in alignment['measured_state_candidates'] if abs(m['error_m'])<1e-8]
    assert len(matched)==1
    assert abs(matched[0]['sweep']['sample_s']-.3375)<1e-8
    assert abs(matched[0]['sweep']['entry_s']-.325)<1e-8
    run=ROOT/'outputs/c5_5_vbc_gcdf_regime'/RUN_ID/'run'
    measured=next(r for r in numeric(run/'joint_states.csv')
                  if int(r['field.header.stamp'])==matched[0]['joint_stamp_ns'])
    q0=np.array(base['b'][:7]);delta=values(measured,'field.position')-q0
    point=np.array(alignment['rejected_point']);geom=Geometry();classes=continuity_methods()
    grids={str(step):np.array([point+[i*step,j*step,k*step] for i in [-1,0,1]
                              for j in [-1,0,1] for k in [-1,0,1]]) for step in [.05,.025]}
    reports={}
    for name,file in [('A','baseline_solution.json'),('C_guard1mm','c_guard1mm_solution.json')]:
        x=np.array(read(file)['x'])
        msg=classes[0]._repair_prefix_with_braking_tail(trajectory(
            np.vstack((q0,x[:140].reshape(20,7)))+delta,x[140:].reshape(20,7),classes))
        duration=classes[0]._duration(msg)
        times=np.linspace(0,duration,int(np.ceil(duration/.001))+1)
        report={}
        for key,points in grids.items():
            ds=np.array([geom.minimum(classes[0]._point_at_time(msg,t).positions,points)[0] for t in times])
            report[key]=dict(points=points.tolist(),per_point_min_m=ds.min(axis=0).tolist(),
                             samples=len(times),center_min_m=float(ds[:,13].min()))
        reports[name]=report
    comparisons={}
    for key in grids:
        a=np.array(reports['A'][key]['per_point_min_m'])
        c=np.array(reports['C_guard1mm'][key]['per_point_min_m'])
        comparisons[key]=dict(a_intersections=int(np.sum(a<0)),c_intersections=int(np.sum(c<0)),
            newly_intersected_points=grids[key][(c<0)&(a>=0)].tolist(),
            cleared_points=grids[key][(c>=0)&(a<0)].tolist())
    assert reports['C_guard1mm']['0.025']['center_min_m']>.0009
    requery=read('nonlinear_checks.json')
    assert requery['learned_requery_all_pass']
    result=dict(status='OFFLINE_CHECKS_PASS_NOT_TASK_OR_FULL_GATE_PASS',
        source_hashes_verified=len(alignment['source_sha256']),counterfactual_checks=checks,
        grid_comparison=comparisons,dense_geometry=reports,
        added_source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),
            'scripts/solve_phase_e_offline_qp.cpp':sha(ROOT/'scripts/solve_phase_e_offline_qp.cpp')},
        limitations=['No frozen map confidence/visibility: sampled neighbor intersection is NOT a VBC verdict.',
                     '1 ms diagnostic samples are NOT a continuous physical safety certificate.',
                     'Reassembled QP validated against convergence and exact logged witness; original matrix/payload were not captured.'])
    save(out/'verification.json',result)
    print(json.dumps(dict(status=result['status'],qp_checks=len(checks),grids=comparisons),indent=2))


if __name__=='__main__': main()
