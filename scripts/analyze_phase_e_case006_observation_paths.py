#!/usr/bin/env python3
"""Frozen CASE006 O6/O9 all-head and timed-path diagnostic; never publishes.

Uses production R1 branch math, FOV/primitive LOS and prefix/braking math.
Paths are diagnostic quintic joint interpolations, NOT Sparse-SCP outputs.
UNKNOWN witnesses are a recorded subset, NOT a replay of the complete map.
"""
import argparse
import csv
import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from diagnose_phase_e_case006_o6_recovery import ROOT, BATCH, strings, digest
from audit_phase_e_replacement_score_gap import joint_limits
import per_sensor_visibility_runtime as ps
import check_visibility_self_occlusion as los
import diagnose_phase_e_case001_exact_witness as geom_module
from verify_phase_e_r1_segment_offline import continuity, Trajectory, Point, Duration
from diagnose_phase_e_r1_o1_o6_offline import clean


def save(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False)+'\n')


def key(point):
    return np.asarray(point, dtype=np.float32).tobytes()


class CachedFK:
    """Memoize unchanged production FK by joint objects and q, in this process."""
    def __init__(self):
        self.original = ps.fk_transform
        self.q = None
        self.cache = {}

    def __call__(self, chain, q_map):
        q = tuple(sorted(q_map.items()))
        if q != self.q:
            self.q, self.cache = q, {}
        k = tuple(id(j) for j in chain)
        if k not in self.cache:
            self.cache[k] = self.original(chain, q_map)
        return self.cache[k]


def visibility(runtime, points, q):
    # Same predicates as _candidate_geometry. Avoid raycasts when the FOV
    # predicate is already false or another head has established the union.
    visible=np.zeros(len(points),dtype=bool)
    qmap=dict(zip(geom_module.JOINTS,q))
    for sid in range(8):
        pose=ps.fk_transform(runtime.sensor_chains[sid],qmap)
        for i,p in enumerate(points):
            if visible[i]: continue
            margin,_,_=ps.sensor_margin(p,pose,runtime.cons_hfov,runtime.cons_vfov,
                                        runtime.cons_z_min,runtime.cons_z_max)
            if margin-runtime.cons_delta<0: continue
            blocked,_=los.raycast_self_occlusion(runtime.self_filter_robot,
                runtime.collision_chains,runtime.primitives,pose,p,qmap,runtime.ray_args)
            visible[i]=not blocked
    return visible,None


def timed_line(q0, q1):
    # Diagnostic rest-to-rest quintic. Limits taken from formal run URDF/config;
    # this timing does not assert planner/controller handoff equivalence.
    vmax = np.array([2., 2., 2., 2., 2.5, 2.5, 2.5])
    amax = np.array([3., 3., 4., 4., 6., 6., 6.])
    delta = q1-q0
    duration = max(1., float(np.max(1.875*np.abs(delta)/vmax)),
                   float(np.max(np.sqrt((10/math.sqrt(3))*np.abs(delta)/amax))))
    duration = math.ceil(duration/.05)*.05
    msg = Trajectory(); msg.joint_names = geom_module.JOINTS
    for t in np.linspace(0, duration, round(duration/.05)+1):
        s=t/duration; a=10*s**3-15*s**4+6*s**5
        v=(30*s**2-60*s**3+30*s**4)/duration
        p=Point(); p.positions=(q0+a*delta).tolist(); p.velocities=(v*delta).tolist()
        p.time_from_start=Duration(t); msg.points.append(p)
    return msg


def path_audit(runtime, geometry, points, groups, msg, target_q, continuity_math):
    """Dense point-center checks; negative witness tests, no safety admission."""
    duration=msg.points[-1].time_from_start.to_sec()
    # Include every knot plus <=10 ms samples and production-style sweep samples.
    sweep=geometry.sweep(msg)
    samples={round(t,12):np.array(continuity_math._point_at_time(msg,t).positions)
             for t in np.linspace(0,duration,max(65,math.ceil(duration/.01)+1))}
    samples.update({round(p.time_from_start.to_sec(),12):np.array(p.positions) for p in msg.points})
    samples.update({round(s['sample_s'],12):np.array(s['q']) for s in sweep})
    rows=[]
    for t,q in sorted(samples.items()):
        ds=geometry.distances(q,points); vis,_=visibility(runtime,points,q)
        rows.append(dict(time_s=t,q=q,distances_m=ds.min(axis=1),
                         closest_primitive=ds.argmin(axis=1),visible=vis))
    first_seen=np.full(len(points),np.inf)
    first_shell={str(m):[None]*len(points) for m in [.04,.02,0.]}
    missing={str(m):[] for m in [.04,.02]}
    for row in rows:
        t=row['time_s']; first_seen=np.where(row['visible'],np.minimum(first_seen,t),first_seen)
        for m in [.04,.02,0.]:
            for i in np.flatnonzero(row['distances_m']<=m):
                if first_shell[str(m)][i] is None:
                    first_shell[str(m)][i]=t
                if m and first_seen[i]>t-.3+1e-12:
                    missing[str(m)].append(dict(time_s=t,point_index=int(i),point=points[i],
                        distance_m=row['distances_m'][i],first_visible_s=first_seen[i],
                        primitive=geometry.primitives[int(row['closest_primitive'][i])]['name']))
    first_all={}
    for group,indices in groups.items():
        first_all[group]=next((row['time_s'] for row in rows if all(row['visible'][indices])),None)
    # Pointwise first-seen times model monotone evidence optimistically; no expiry,
    # environment LOS, confidence latency or full-map claim is introduced.
    q0=np.array(msg.points[0].positions); qe=np.array(msg.points[-1].positions)
    return dict(duration_s=duration,samples=len(rows),first_all_visible_s=first_all,
        first_point_visible_s=first_seen, first_shell_entry_s=first_shell,
        first_temporal_rejection={m:(v[0] if v else None) for m,v in missing.items()},
        violating_point_indices={m:sorted({v['point_index'] for v in hits}) for m,hits in missing.items()},
        min_point_distance_m=min(float(min(r['distances_m'])) for r in rows),
        endpoint_displacement_inf_rad=float(max(abs(qe-q0))),
        target_gap_start_inf_rad=float(max(abs(target_q-q0))),
        target_gap_end_inf_rad=float(max(abs(target_q-qe))),
        start_q=q0,end_q=qe, rows=rows,
        runtime_admission=False,complete_map_checked=False,
        temporal_model='initial recorded UNKNOWN; optimistic instantaneous persistent visibility; require 0.3s lead')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args(); out=args.output_dir; out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    run=ROOT/'outputs/c5_5_vbc_gcdf_regime'/('phase_e_goal_006_'+BATCH)/'run'
    trace=run/'projector_traces'; o6p,=trace.glob('c46_obligation_006_*.json'); o9p,=trace.glob('c46_obligation_009_*.json')
    o6,o9=[json.loads(p.read_text()) for p in [o6p,o9p]]
    applied=[json.loads(line) for line in (trace/'candidate_replacement.jsonl').read_text().splitlines()]
    s5=next(x for x in applied if x.get('request_id')==7 and 'hybrid' in x)
    checkpoint=Path(s5['hybrid']['config']['checkpoint'])
    assert digest(checkpoint)==s5['hybrid']['config']['checkpoint_sha256']
    protected=[checkpoint,Path(ps.__file__),Path(los.__file__),geom_module.CONTINUITY,
               ROOT/'src/arm_description/urdf/Arm.urdf',geom_module.URDF_PATH,
               o6p,o9p,trace/'candidate_replacement.jsonl']
    protected += [run/(n+'.csv') for n in ['local_gcdf_query','joint_states','observation_dependency',
                   'local_witness_selector_diagnostics','local_planner_summary']]
    hashes={str(p):digest(p) for p in protected}
    cfg=s5['hybrid']['config']; qmin,qmax=joint_limits()
    runtime=ps.PerSensorVisibilityRuntime(str(checkpoint),str(protected[4]),str(protected[5]),
        torch.device('cpu'),qmin,qmax,**{k:v for k,v in cfg.items() if k in
            inspect.signature(ps.PerSensorVisibilityRuntime).parameters and k not in ['checkpoint_path']})
    deps=strings(run/'observation_dependency.csv')
    dep=next(d for d in deps if d['query_stamp_ns']==21417000000)
    with (run/'local_gcdf_query.csv').open() as f:
        qr,=[r for r in csv.DictReader(f) if r.get('field.header.stamp')==str(dep['query_stamp_ns'])]
    q0=np.array([float(qr[f'field.points0.positions{i}']) for i in range(7)])
    with (run/'joint_states.csv').open() as f:
        matches=[r for r in csv.DictReader(f) if r['%time'].isdigit() and
                 int(r['field.header.stamp'])<=dep['query_stamp_ns'] and
                 np.max(abs(np.array([float(r[f'field.position{i}']) for i in range(7)])-q0))<1e-12]
    assert matches
    witness=[r for r in strings(run/'local_witness_selector_diagnostics.csv')
             if int(r['query_stamp_ns'])==dep['query_stamp_ns'] and int(r['plan_seq'])==dep['plan_seq']]
    assert sorted(int(r['witness_index']) for r in witness)==list(range(len(witness)))
    assert all(r['status']=='evaluated_unknown' for r in witness)
    unique={key(json.loads(r['point'])):json.loads(r['point']) for r in witness}
    points=np.array(list(unique.values())); indices={k:i for i,k in enumerate(unique)}
    groups={'O6':[indices[key(p)] for p in o6['active_set_points_xyz']],
            'O9':[indices[key(p)] for p in o9['active_set_points_xyz']]}
    groups['O6_O9']=groups['O6']+groups['O9']
    assert all(key(p) in unique for p in dep['points'])
    snapshot=dict(query_ns=dep['query_stamp_ns'],plan_seq=dep['plan_seq'],q0=q0,
        measured_stamp_ns=int(matches[-1]['field.header.stamp']),
        measured_velocity=[float(matches[-1][f'field.velocity{i}']) for i in range(7)],
        points=points,groups=groups,witness_rows=witness,dependency=dep)
    save(out/'snapshot.json',snapshot)
    # Reproduce the saved S5 branch from its exact seed before counterfactuals.
    branch=runtime._optimize_branch(torch.tensor(s5['points'],dtype=torch.float32),
                                   torch.tensor([s5['measured_seed']],dtype=torch.float32),5)
    err=float(np.max(abs(np.array(branch['q_candidate'])-s5['hybrid']['selected_q_vis'])))
    assert err<2e-5,err
    geometry=geom_module.Geometry()
    original_geometry=runtime._candidate_geometry(points,q0,5)
    fk=CachedFK(); ps.fk_transform=fk; los.fk_transform=fk; geom_module.fk_transform=fk
    assert runtime._candidate_geometry(points,q0,5)==original_geometry
    for check_q in [q0,np.array(s5['hybrid']['selected_q_vis'])]:
        expected=np.array([[p['conservative_g']>=0 and not p['primitive_self_occluded']
            for p in runtime._candidate_geometry(points,check_q,s)['per_point']] for s in range(8)]).any(axis=0)
        assert np.array_equal(visibility(runtime,points,check_q)[0],expected)
    cm=continuity()
    records=[]
    for group,ii in groups.items():
        target=points[ii]
        for sid in range(8):
            b=runtime._optimize_branch(torch.tensor(target,dtype=torch.float32),
                                       torch.tensor(q0[None],dtype=torch.float32),sid)
            q=np.array(b['q_candidate']);g=runtime._candidate_geometry(target,q,sid)
            vis,heads=visibility(runtime,points,q)
            row=dict(group=group,sensor=sid,branch=b,selected_head_geometry=g,
                all_targets_visible_any_head={n:bool(all(vis[idx])) for n,idx in groups.items()},
                endpoint_visible_unknown_points=[int(i) for i in np.flatnonzero(vis)],
                target_gap_inf_rad=float(max(abs(q-q0))))
            records.append(row)
            print('branch',group,sid,'accepted',g['accepted'],'g',round(g['min_conservative_g'],5),flush=True)
    save(out/'candidates.json',dict(records=records,s5_reproduction_error_inf_rad=err))
    # Examine all endpoint-accepted branches plus S5/S6/S7 even if endpoint fails.
    selected=[r for r in records if r['selected_head_geometry']['accepted'] or
              (r['group']=='O6' and r['sensor'] in [5,6,7])]
    selected.append(dict(group='recorded_S5',sensor=5,branch=dict(q_candidate=s5['hybrid']['selected_q_vis'])))
    results=[]
    for r in selected:
        q=np.array(r['branch']['q_candidate']);msg=timed_line(q0,q)
        prefix=cm._repair_prefix_with_braking_tail(msg);assert prefix is not None
        assert max(abs(v) for v in prefix.points[-1].velocities)==0
        full=path_audit(runtime,geometry,points,groups,msg,q,cm)
        short=path_audit(runtime,geometry,points,groups,prefix,q,cm)
        label=r['group']+'_S'+str(r['sensor'])
        save(out/(label+'.json'),dict(label=label,full=full,prefix_brake_hold=short))
        brief=lambda d:{k:v for k,v in d.items() if k!='rows'}
        results.append(dict(label=label,full=brief(full),prefix_brake_hold=brief(short)))
        print('path',label,'first40',clean(full['first_temporal_rejection']['0.04']),
              'visible',full['first_all_visible_s'],flush=True)
    assert hashes=={str(p):digest(p) for p in protected}
    assert 'rospy' not in sys.modules
    save(out/'summary.json',dict(batch=BATCH,snapshot='query21417000000',
        candidate_count=len(records),path_count=len(results),results=results,
        verification=dict(s5_reproduced_error_inf_rad=err,fk_cache_parity=True,
            exact_query_measured_join=True,complete_indexed_witness_response=True,
            input_hashes_unchanged=True,no_ros_import=True),inputs_sha256=hashes,
        script_sha256=digest(Path(__file__)),
        limitations=['Diagnostic quintic interpolation is not a production Sparse-SCP or tracker rollout.',
            'Recorded UNKNOWN points only; positive results cannot certify complete-map VBC/GCDF.',
            'Analytic FOV/self-LOS omits environment LOS, confidence accumulation and expiry.',
            'Sampled first visibility and shell times have finite resolution; no continuous-time guarantee.',
            'Counterfactual starts at rest; recorded velocity is retained for comparison.',
            'A failed straight path does not rule out a multisegment detour.']))


if __name__=='__main__':
    main()
