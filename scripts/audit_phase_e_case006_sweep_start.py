#!/usr/bin/env python3
"""Verify first S5 committed-path geometry with the frozen production library."""
import csv
import json
import os
from pathlib import Path
import subprocess
from diagnose_phase_e_case006_o6_recovery import ROOT, BATCH, strings, digest


def main():
    out=ROOT/'outputs/phase_e_case006_observation_paths_20261006_v2/production_sweep'
    out.mkdir(parents=True,exist_ok=False)
    run=ROOT/'outputs/c5_5_vbc_gcdf_regime'/('phase_e_goal_006_'+BATCH)/'run'
    source=ROOT/'scripts/probe_phase_e_case006_sweep.cpp'
    library=ROOT/'devel/lib/libtrajectory_risk_evaluator.so'
    urdf=ROOT/'src/arm_description/urdf/Arm.urdf'
    primitive=ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'
    metadata=ROOT/'outputs/phase_e_r1_mainline_30'/BATCH/'metadata.json'
    frozen=json.loads(metadata.read_text())['source_hashes']
    for path in [library,urdf,primitive]:
        assert digest(path)==frozen[str(path.relative_to(ROOT))]
    verification=next(r for r in strings(run/'verification_outcome.csv')
                      if r.get('execution_stamp_ns')=='22133000000')
    assert verification['result']=='safe'
    for field in ['precommit_suffix_phase_s','precommit_suffix_start_shift_inf','total_start_shift_inf']:
        assert float(verification[field])==0
    with (run/'committed_trajectory.csv').open() as stream:
        row,=[r for r in csv.DictReader(stream) if r.get('field.header.stamp')=='22133000000']
    trajectory=[]
    for k in range(21):
        if not row.get(f'field.points{k}.positions0'): break
        trajectory.append(dict(time_s=int(row[f'field.points{k}.time_from_start'])/1e9,
            q=[float(row[f'field.points{k}.positions{j}']) for j in range(7)]))
    points=json.loads((out.parent/'snapshot.json').read_text())['points']
    request=dict(urdf=str(urdf),primitive_urdf=str(primitive),points=points,trajectory=trajectory)
    inputs=[source,library,urdf,primitive,metadata,run/'committed_trajectory.csv',run/'verification_outcome.csv',
            out.parent/'snapshot.json']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in inputs}
    env=dict(os.environ,LD_LIBRARY_PATH=str(ROOT/'devel/lib')+':/opt/ros/noetic/lib:/opt/ros/noetic/lib/x86_64-linux-gnu')
    binary=out/'probe'
    build=['g++','-std=c++14','-O0','-DPINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION','-DPINOCCHIO_WITH_URDFDOM',
        '-DPINOCCHIO_WITH_HPP_FCL','-DBOOST_MPL_LIMIT_LIST_SIZE=30','-DBOOST_MPL_LIMIT_VECTOR_SIZE=30',
        '-I'+str(ROOT/'src/care_confidence_map/include'),'-I'+str(ROOT/'devel/include'),
        '-I/opt/ros/noetic/include','-I/usr/include/eigen3',str(source),'-o',str(binary),
        '-L'+str(ROOT/'devel/lib'),'-L/opt/ros/noetic/lib','-ltrajectory_risk_evaluator',
        '-lpinocchio_default','-lroscpp','-lrosconsole','-lrostime']
    subprocess.run(build,env=env,check=True,capture_output=True,text=True)
    process=subprocess.run([str(binary)],input=json.dumps(request),text=True,capture_output=True,env=env,check=True)
    rows=json.loads(process.stdout)
    pi=next(i for i,p in enumerate(points) if max(abs(x-y) for x,y in zip(p,[.05,-.2,.75]))<1e-8)
    start=next(r for r in rows if r['mode']==0 and r['frame']==0 and r['point_index']==pi)
    first=next(r for r in rows if r['mode']==1 and r['frame']==0 and r['point_index']==pi)
    bound=next(r for r in rows if r['mode']==2 and r['frame']==0 and r['point_index']==pi)
    assert start['distance_m']<.04<first['distance_m']
    assert start['distance_m']>.02 and bound['enclosed_distance_m']<.04
    summary=dict(raw_ns=verification['raw_candidate_stamp_ns'],audited_ns=verification['audited_trajectory_stamp_ns'],
        execution_ns='22133000000',recorded_verification=verification['result'],
        committed_geometry_basis='exact execution identity; zero precommit phase/shift reported, audited message not independently recorded',
        point_index=pi,point=points[pi],start=start,first_midpoint=first,first_with_enclosure=bound,
        min_over_recorded_points_by_mode={str(m):min(r['enclosed_distance_m'] for r in rows if r['mode']==m) for m in range(3)},
        frozen_library_and_urdf_match=True,input_sha256=hashes,build_argv=build,
        qualification='geometry replay only; no live confidence queries, runtime change or execution')
    assert hashes=={str(p.relative_to(ROOT)):digest(p) for p in inputs}
    for name,data in [('input',request),('rows',rows),('summary',summary)]:
        (out/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['build_argv','input_sha256']},indent=2))


if __name__=='__main__':main()
