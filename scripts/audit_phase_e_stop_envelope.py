#!/usr/bin/env python3
"""Recorded CASE006 stopping-enclosure audit; no live admission or map replay.

Run with viscdf Python (Pinocchio). A frozen startup-prior comparison is only a
coverage diagnostic. It is NOT the evolving online confidence map.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
from scipy.spatial import cKDTree
import yaml

from audit_case001_motion_prior import (ROOT,URDF,CONFIG,JOINTS,load_shapes,frame_placements,
                                      collision_sdf,distance_matrix,surface_samples)
from phase_e_stop_envelope import stopping_budget,primitive_radii,coverage_check

RUN=ROOT/'outputs/c5_5_vbc_gcdf_regime/phase_e_goal_006_r1_case006_actuator_watchdog_20261001/run'
OUT=ROOT/'outputs/phase_e_stop_envelope_20261001/audit'


def rows(path,stamp):
    with path.open() as stream:
        return [r for r in csv.DictReader(stream) if r.get(stamp,'').isdigit()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=RUN)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args(); run=args.run_dir; out=args.output_dir
    controller=ROOT/'src/arm_description/config/controllers_watchdog.yaml'
    params=yaml.safe_load(controller.read_text())['care_arm']['arm_group_velocity_controller']
    conf=yaml.safe_load(CONFIG.read_text()); grid=conf['confidence_map']
    model=pin.buildModelFromUrdf(str(URDF)); data=model.createData()
    shapes=load_shapes(ET.parse(URDF).getroot())
    active=np.array([i for i,s in enumerate(shapes) if s['link'] not in ('base_link','link1')])
    radii=primitive_radii(URDF,shapes)
    qindices=[model.joints[model.getJointId(j)].idx_q for j in JOINTS]
    def poses(q):
        full=np.zeros(model.nq); full[qindices]=q
        return frame_placements(model,data,shapes,full)
    window=json.loads((run/'goal_stop_status.json').read_text())['benchmark_window']
    states=[r for r in rows(run/'actuator_watchdog_state.csv','field.header.stamp')
            if window['start_ros_ns']<=int(r['field.header.stamp'])<=window['end_ros_ns']]
    requests={(r['field.header.stamp'],r['field.header.seq']):r for r in rows(run/'actuator_stamped_request.csv','field.header.stamp')}
    measured=rows(run/'joint_states.csv','field.header.stamp')
    mt=np.array([int(r['field.header.stamp']) for r in measured],dtype=np.int64)
    mq=np.array([[float(r['field.position'+str([r['field.name'+str(k)] for k in range(7)].index(j))]) for j in JOINTS] for r in measured])
    assert np.all(np.diff(mt)>=0)
    profiles=[]; missing=[]
    for r in states:
        if r['field.mode']!='1' or r['field.clock_fault']!='0': continue
        request=requests.get((r['field.command_stamp'],r['field.command_sequence']))
        if request is None:
            missing.append(dict(t_ns=int(r['field.header.stamp']),reason='missing_exact_command_identity')); continue
        assert [request['field.name'+str(i)] for i in range(7)]==JOINTS
        t=int(r['field.header.stamp']); qi=int(np.searchsorted(mt,t,side='right')-1)
        if qi<0:
            missing.append(dict(t_ns=t,reason='no_prior_measured_configuration')); continue
        age=(t-int(r['field.command_stamp']))/1e9; state_age=(t-int(mt[qi]))/1e9
        b=stopping_budget([float(r['field.applied_velocity'+str(i)]) for i in range(7)],
            [float(request['field.velocity'+str(i)]) for i in range(7)],age,state_age,.001,
            params['acceleration_limits'],params['velocity_limits'],params['command_timeout'],params['max_braking_step'])
        displacement=radii@b.command_travel_rad
        profiles.append(dict(t_ns=t,command_stamp_ns=int(r['field.command_stamp']),command_sequence=int(r['field.command_sequence']),
            measured_stamp_ns=int(mt[qi]),state_age_s=state_age,q=mq[qi].tolist(),
            applied=[float(r['field.applied_velocity'+str(i)]) for i in range(7)],
            target=[float(request['field.velocity'+str(i)]) for i in range(7)],
            joint_command_travel_rad=b.command_travel_rad.tolist(),command_stop_s=b.command_stop_s,
            primitive_command_motion_bound_m=displacement.tolist(),
            max_active_primitive_command_bound_m=float(displacement[active].max())))
    assert profiles, 'no complete command/state evidence'

    # Exact initial grid-center prior, unchanged and never moved with the arm.
    resolution=grid['resolution']
    minimum=np.array([grid[a+'_min'] for a in 'xyz']); maximum=np.array([grid[a+'_max'] for a in 'xyz'])
    axes=[lo+resolution*np.arange(math.floor((hi-lo)/resolution)+1) for lo,hi in zip(minimum,maximum)]
    points=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    prior_radii=np.array([conf['current_body_prior']['link_inflation_radius'][s['link']] for s in shapes])
    initial=poses(np.zeros(7))
    prior=(distance_matrix(points,shapes,initial)<=prior_radii).any(axis=1)
    assert int(prior.sum())==1227
    tree=cKDTree(points)

    def raster(q,displacement):
        placements=poses(q); mask=np.zeros(len(points),dtype=bool); inside=True
        for i in active:
            shape,pose=shapes[i],placements[i]
            inflation=.020+float(displacement[i])
            nearby=np.asarray(tree.query_ball_point(pose.translation,shape['circumradius']+inflation+1e-9),dtype=int)
            if nearby.size: mask[nearby] |= collision_sdf(points[nearby],shape,pose)<=inflation+1e-12
            rotation=np.asarray(pose.rotation)
            if shape['kind']=='box': extent=np.abs(rotation)@shape['half_size']
            elif shape['kind']=='cylinder': extent=shape['radius']*np.sqrt(rotation[:,0]**2+rotation[:,1]**2)+shape['half_length']*np.abs(rotation[:,2])
            else: extent=np.full(3,shape['radius'])
            extent+=inflation
            inside=inside and bool(np.all(pose.translation-extent>=minimum) and np.all(pose.translation+extent<=maximum))
        return mask,inside

    footprint=[]; last_slot=None
    for p in profiles:
        slot=p['t_ns']//100_000_000
        if slot==last_slot: continue
        last_slot=slot
        current,_=raster(p['q'],np.zeros(len(shapes)))
        mask,inside=raster(p['q'],p['primitive_command_motion_bound_m'])
        outside=np.flatnonzero(mask & ~prior)
        result=coverage_check(has_physical_contract=False,state_age_s=p['state_age_s'],max_state_age_s=.02,
                              stop_horizon_s=p['command_stop_s'],evidence_valid_for_s=0.,complete_snapshot=False,
                              inside_map=inside,free_mask=prior[mask])
        footprint.append(dict(t_ns=p['t_ns'],current_shell_centers=int(current.sum()),
            current_shell_outside_startup_prior=int(np.sum(current & ~prior)),
            stop_enclosure_centers=int(mask.sum()),stop_enclosure_outside_startup_prior=int(len(outside)),
            added_centers_vs_current=int(np.sum(mask & ~current)),enclosure_inside_map=inside,
            first_outside_startup_prior=points[outside[0]].tolist() if len(outside) else None,
            missing_physical_and_online_evidence_check=result))

    # Independent FK surface checks supplement the global-radius proof. Exercise
    # all 128 joint-box corners at the worst-budget pose, plus random simultaneous
    # joint perturbations at representative real poses.
    worst=max(profiles,key=lambda p:p['max_active_primitive_command_bound_m'])
    cases=[(worst,np.array(c)) for c in itertools.product((-1.,1.),repeat=7)]
    rng=np.random.RandomState(61001)
    for i in np.linspace(0,len(profiles)-1,12,dtype=int):
        cases.extend((profiles[i],rng.uniform(-1,1,7)) for _ in range(12))
    checked_points=0; max_excess=-math.inf
    samples=[surface_samples(s) for s in shapes]
    for p,coefficients in cases:
        q=np.array(p['q']); dq=coefficients*np.array(p['joint_command_travel_rad'])
        before=poses(q); after=poses(q+dq)
        for i in active:
            a=np.asarray(before[i].translation)+samples[i]@np.asarray(before[i].rotation).T
            z=np.asarray(after[i].translation)+samples[i]@np.asarray(after[i].rotation).T
            excess=float(np.linalg.norm(z-a,axis=1).max())-p['primitive_command_motion_bound_m'][i]
            max_excess=max(max_excess,excess); checked_points+=len(a)
            assert excess<=1e-9, (i,excess)

    # Command zero on uncommanded joints does not imply physical zero travel.
    diagnostic=ROOT/'outputs/phase_e_actuator_watchdog_20260930/audit/audit.json'
    observed=json.loads(diagnostic.read_text())['trials']
    cross_axis=[dict(commanded_joint2_speed=r['initial_joint2_command_rad_s'],
        max_other_joint_observed_displacement_rad=max(abs(v) for i,v in enumerate(r['interpolated_measured_displacement_from_generation_rad']) if i!=1)) for r in observed]
    bounds=np.array([p['max_active_primitive_command_bound_m'] for p in profiles])
    result=dict(status='OFFLINE_COMMAND_ENCLOSURE_VERIFIED_PHYSICAL_ADMISSION_UNQUALIFIED',runtime_admission_enabled=False,
        assumptions=['No newer accepted command after the evaluated controller update',
                     'H=1 ms is a hypothetical update-gap bound, not a measured wall-time guarantee',
                     'Monotone ROS time and controller period equals elapsed update time',
                     'Command is immediately applied and held; physical tracking error/settling are unspecified'],
        scope='URDF primitive and 20 mm grid-center enclosure; not voxel-cube coverage or environment collision proof',
        matched_state_count=len(profiles),unmatched_state_count=len(missing),excluded_states=missing,primitive_count=len(shapes),
        active_primitive_count=len(active),ignored_links=['base_link','link1'],
        max_body_command_bound_m=dict(median=float(np.median(bounds)),p95=float(np.percentile(bounds,95)),maximum=float(bounds.max())),
        states_where_extra_command_bound_exceeds_20mm=int(np.sum(bounds>.020)),
        max_command_stop_s=max(p['command_stop_s'] for p in profiles),
        geometry_checks=dict(joint_box_poses=len(cases),surface_point_pairs=checked_points,max_excess_m=max_excess),
        prior_comparison=dict(sample_count=len(footprint),startup_prior_centers=int(prior.sum()),
            current_shell_fully_in_prior=sum(p['current_shell_outside_startup_prior']==0 for p in footprint),
            stop_enclosure_fully_in_prior=sum(p['stop_enclosure_outside_startup_prior']==0 for p in footprint),
            enclosure_may_leave_map=sum(not p['enclosure_inside_map'] for p in footprint),
            caution='Points outside startup prior may be covered by live observations; no online VBC violation is inferred'),
        observed_cross_axis_motion=cross_axis,worst_budget_snapshot=worst,
        evidence_interface_gaps=['QueryConfidence response has no snapshot version/time or future validity horizon',
                                 'Recorded active-set bundle is not a complete map snapshot',
                                 'Physical trajectory-error and settling-time bounds are absent'])
    inputs=[URDF,CONFIG,controller,run/'actuator_watchdog_state.csv',run/'actuator_stamped_request.csv',run/'joint_states.csv',
            run/'goal_stop_status.json',diagnostic,ROOT/'scripts/phase_e_stop_envelope.py',Path(__file__).resolve(),
            ROOT/'src/care_confidence_map/src/vbc_primitive_model.cpp',ROOT/'src/care_confidence_map/srv/QueryConfidence.srv']
    result['input_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    out.mkdir(parents=True,exist_ok=True)
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'profiles.json').write_text(json.dumps(profiles)+'\n')
    (out/'footprints.json').write_text(json.dumps(footprint)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','worst_budget_snapshot')},indent=2))


if __name__=='__main__': main()
