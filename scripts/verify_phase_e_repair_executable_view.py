#!/usr/bin/env python3
"""Frozen CASE028 constructor and derivative checks. Never publishes commands."""
import argparse
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from diagnose_phase_e_case001_exact_witness import Geometry, continuity_methods
from analyze_phase_e_observation_events import string_rows


def read_trajectory(path,stamp):
    with path.open() as f:
        row=next(r for r in csv.DictReader(f) if r.get('field.header.stamp')==str(stamp))
    q=[];u=[]
    for k in range(100):
        if not row.get('field.points%d.positions0'%k):break
        q.append([float(row['field.points%d.positions%d'%(k,j)]) for j in range(7)])
        u.append([float(row['field.points%d.velocities%d'%(k,j)]) for j in range(7)])
    return np.array(q),np.array(u[:-1])


def message(q,u,classes):
    _,rospy,JT,Point=classes
    m=JT();m.joint_names=['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']
    for k,v in enumerate(q):
        p=Point();p.positions=list(v);p.velocities=list(u[k]) if k<len(u) else [0.]*7
        p.time_from_start=rospy.Duration(k*.05);m.points.append(p)
    return m


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    run=ROOT/'outputs/c5_5_vbc_gcdf_regime/phase_e_goal_028_r1_cdf_conflict_reseed_5_20261010/run'
    q,u=read_trajectory(run/'raw_candidate_trajectory.csv',22286000000)
    classes=continuity_methods();constructor=classes[0];geom=Geometry();point=[.1,-.2,.8]
    prefixes=[.25,.15,.10,.05]
    base=dict(q=q.tolist(),u=u.tolist(),acceleration=[3.,3.,4.,4.,6.,6.,6.],prefixes=prefixes)
    tests=[base,dict(base,prefixes=[.173]),dict(base,u=(-u).tolist()),
           dict(base,u=np.zeros_like(u).tolist()),dict(base,acceleration=[0.]*7),
           dict(base,max_brake_steps=21)]
    (out/'inputs.json').write_text(json.dumps(tests,indent=2)+'\n')
    subprocess.run(['c++','-std=c++14','-O2','-I/usr/include/eigen3','-I'+str(ROOT/'src/egocentric_arm_planner/include'),
        str(ROOT/'scripts/test_phase_e_repair_executable_view.cpp'),'-o',str(out/'samples')],check=True,
        stdout=(out/'compile.log').open('w'),stderr=subprocess.STDOUT)
    subprocess.run([str(out/'samples'),str(out/'inputs.json'),str(out/'results.json')],check=True)
    results=json.loads((out/'results.json').read_text());assert all(r['ok'] for r in results[:4])
    assert all(not r['ok'] for r in results[4:])
    max_error=0.;max_derivative_error=0.;checks=0
    for fixture,result in zip(tests[:4],results[:4]):
        qs,us=np.array(fixture['q']),np.array(fixture['u']);raw=message(qs,us,classes)
        for sample in result['samples']:
            view=constructor._repair_prefix_with_braking_tail(raw,sample['prefix'])
            at=np.array(constructor._point_at_time(view,sample['time']).positions)
            max_error=max(max_error,float(np.max(np.abs(at-sample['q']))))
        # Frozen signed-speed fixture, away from clipping transitions. Use the
        # production Python construction as the independent derivative oracle.
        if np.max(np.abs(us))==0.:continue
        for sample in result['samples']:
            if not sample['u_weights']:continue
            for k,weight in sample['u_weights']:
                for j in range(7):
                    du=np.zeros_like(us);du[k,j]=1e-6
                    plus=constructor._repair_prefix_with_braking_tail(message(qs,us+du,classes),sample['prefix'])
                    minus=constructor._repair_prefix_with_braking_tail(message(qs,us-du,classes),sample['prefix'])
                    derivative=(np.array(constructor._point_at_time(plus,sample['time']).positions)-
                                np.array(constructor._point_at_time(minus,sample['time']).positions))/2e-6
                    expected=np.zeros(7);expected[j]=weight[j]
                    max_derivative_error=max(max_derivative_error,float(np.max(np.abs(derivative-expected))))
                    checks+=1
    assert max_error<2e-9,(max_error,results)
    assert max_derivative_error<2e-6,max_derivative_error
    view=constructor._repair_prefix_with_braking_tail(message(q,u,classes),.25)
    raw_min=min(float(geom.minimum(v,[point])[0][0]) for v in q)
    executable_min=min(float(geom.minimum(p.positions,[point])[0][0]) for p in view.points)
    assert raw_min>=.02 and executable_min<.02
    outcomes=string_rows(run/'verification_outcome.csv',{'start_ros_s':0.,'end_ros_s':100.})
    rejection=next(r for r in outcomes if r.get('raw_candidate_stamp_ns')=='22286000000')
    evidence=json.loads(rejection['vbc_evidence'])
    hit=next(v for v in evidence['violations'] if np.max(np.abs(np.array(v['point'])-point))<1e-5)
    fixture=dict(base,point=point,evidence=json.dumps(dict(evidence,violations=[hit]),separators=(',',':')),
                 audited_stamp_ns=evidence['trajectory_stamp_ns'])
    (out/'qp_fixture.json').write_text(json.dumps(fixture,indent=2)+'\n')
    report=dict(status='PASS',sample_parity_max_error=max_error,velocity_derivative_checks=checks,
        velocity_derivative_max_error=max_derivative_error,invalid_contracts_rejected=2,
        raw_min_knot_m=raw_min,constructed_min_knot_m=executable_min,
        qualification='Frozen actual raw path and production construction. No reconstruction of its original full CDF QP or map certificate.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
