#!/usr/bin/env python3
"""Audit discrete Gazebo physics observations; never issues a command lease."""
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
from phase_e_stopping_contract import qualify

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();source=args.input_dir.resolve();out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    run=json.loads((source/'result.json').read_text())
    assert run['status']=='PASS' and run['protected_identity_equal'] and run['input_unchanged']
    assert not run['cleanup']['remaining'] and all(run['port_released'].values())
    steps=[]
    for line in (source/'physics_steps.jsonl').open():
        row=json.loads(line)
        steps.append((row['t_ns'],row['q_before'],row['q_after'],row['motor'],row['velocity']))
    ts=np.asarray([s[0] for s in steps],dtype=np.int64)
    before=np.asarray([s[1] for s in steps]);after=np.asarray([s[2] for s in steps])
    motor=np.asarray([s[3] for s in steps]);velocity=np.asarray([s[4] for s in steps]);del steps
    assert np.all(np.diff(ts)==1000000),'physics observation gap or changed step'
    assert np.all(np.isfinite(np.r_[before.ravel(),after.ravel(),motor.ravel(),velocity.ravel()]))
    state=[row for row in map(json.loads,(source/'controller_trace.jsonl').open()) if row['kind']=='state']
    by_stamp={int(t):i for i,t in enumerate(ts)}
    matched=[(row,by_stamp[row['t_ns']]) for row in state if row['t_ns'] in by_stamp]
    assert matched
    motor_alignment=max(float(np.max(np.abs(motor[i]-row['applied']))) for row,i in matched)
    pose_alignment=max(float(np.max(np.abs(before[i]-row['q']))) for row,i in matched)
    assert motor_alignment<1e-9 and pose_alignment<1e-9,'observer event order does not match controller state'
    trials=[];all_errors=[];zero_axis_drift=[]
    for trial in run['trials']:
        ids=np.flatnonzero((ts>=trial['baseline_ns'])&(ts<=trial['end_ns']))
        assert len(ids)>100
        q0=before[ids[0]]
        integral=np.cumsum(motor[ids]*.001,axis=0)
        error=after[ids]-q0-integral
        maximum=np.max(np.abs(error),axis=0);all_errors.append(maximum)
        absolute_integral=np.sum(np.abs(motor[ids])*.001,axis=0)
        zeros=np.flatnonzero(np.max(np.abs(motor[ids]),axis=1)<=1e-12)
        assert len(zeros),'no command zero in the physics trace'
        zero=int(zeros[0]);rest=None
        for k in range(zero,len(ids)-100):
            if np.max(np.abs(velocity[ids[k:k+101]]))<=.001:
                rest=k;break
        assert rest is not None,'no 100ms persistent measured rest'
        qanchor=next(row for row in state if row['t_ns']==trial['baseline_ns'])
        budget_checks=[]
        for age_s in [0.,.01,.04,.049]:
            now=qanchor['t_ns']*1e-9+age_s
            if now>=trial['fixed_until_ns']*1e-9:continue
            r=qualify(now_s=now,measured_s=qanchor['t_ns']*1e-9,applied=qanchor['applied'],
                acceleration=qanchor['acceleration_limits'],velocity_limits=qanchor['velocity_limits'],
                next_decision_s=.01,transport_bound_s=0.,update_gap_s=.001,
                lease_end_s=trial['fixed_until_ns']*1e-9,lease_enforced=True,
                physical_error_rad=None,physical_settle_s=None,snapshot=None,
                max_braking_step_s=qanchor['max_braking_step_s'],
                authorized_target_abs=qanchor.get('target_abs_velocity') if qanchor.get('target_range_enforced') else None)
            bound=r['budgets']['withdraw_now']
            assert np.all(absolute_integral<=np.asarray(bound['command_travel_rad'])+1e-9)
            assert ts[ids[zero]]*1e-9<=bound['required_until_s']+1e-9
            budget_checks.append(dict(receipt_age_s=age_s,budget_rad=bound['command_travel_rad'],
                command_integral_covered=True,command_zero_time_covered=True))
        inactive=[j for j,v in enumerate(trial['v']) if abs(v)<1e-12]
        if inactive:zero_axis_drift.append(float(np.max(np.abs(after[ids][:,inactive]-q0[inactive]))))
        trials.append(dict(label=trial['label'],physics_steps=len(ids),max_discrete_response_error_rad=maximum.tolist(),
            command_zero_ns=int(ts[ids[zero]]),measured_rest_ns=int(ts[ids[rest]]),
            measured_extra_settling_s=float((ts[ids[rest]]-ts[ids[zero]])*1e-9),
            absolute_command_integral_rad=absolute_integral.tolist(),budget_checks=budget_checks))
    counts=collections.Counter();reasons=collections.Counter();elapsed=[];points=[]
    for row in map(json.loads,(source/'shadow/rows.jsonl').open()):
        counts['attempts']+=1;elapsed.append(row['wall_elapsed_s'])
        reasons.update(row['reasons'])
        snapshot=row.get('map')
        if snapshot:
            if 'adaptive' in snapshot:
                counts['adaptive_attempts']+=1
                counts['adaptive_accepted']+=int(snapshot['adaptive']['accepted'])
                if not snapshot.get('complete'):
                    assert not snapshot['runtime_admission']
                    counts['adaptive_rejected']+=1
                    continue
            assert snapshot['enclosure_kind']=='command_only' and not snapshot['runtime_admission']
            counts['complete_union_queries']+=1;points.append(snapshot['points_queried'])
            counts['inside_map']+=int(snapshot['inside_map'])
            counts['all_centers_free']+=int(snapshot['all_centers_free'])
            counts['stable_version']+=int(snapshot['stable_version'])
            covered=all(snapshot[k] for k in ['complete','stable_version','inside_map','all_centers_free'])
            counts['conditional_point_coverage']+=int(covered)
            counts['coverage_and_no_observed_binding_change']+=int(covered and not row['reasons'])
            counts['queries_with_unknown_or_decayed']+=int(snapshot['evidence_counts']['unknown_or_decayed']>0)
            counts['queries_with_occupied']+=int(snapshot['evidence_counts']['occupied']>0)
    input_files=[source/'result.json',source/'physics_steps.jsonl',source/'controller_trace.jsonl',
        source/'shadow/rows.jsonl',Path(__file__).resolve(),Path(__file__).resolve().parent/'phase_e_stopping_contract.py']
    result=dict(status='PASS_OBSERVATIONAL_AUDIT',new_formal_windows=0,runtime_admission=False,
        physical_bound_certified=False,physics_steps=len(ts),observed_physics_gap_s=.001,
        controller_receipt_matches=len(matched),max_motor_to_receipt_error=motor_alignment,
        max_pose_to_receipt_error=pose_alignment,trials=trials,
        maximum_discrete_response_error_rad=np.max(all_errors,axis=0).tolist(),
        maximum_zero_command_axis_travel_rad=max(zero_axis_drift,default=0.),
        maximum_observed_extra_settling_s=max(t['measured_extra_settling_s'] for t in trials),
        map=dict(scope='command-only complete union; no qualified physical error',counts=dict(counts),reasons=dict(reasons),
            wall_ms=dict(median=float(np.median(elapsed)*1000),p95=float(np.percentile(elapsed,95)*1000),maximum=max(elapsed)*1000),
            query_points_min=min(points,default=0),query_points_max=max(points,default=0),observation_mode=run['map_observation_mode']),
        missing_qualification=['qualified_future_update_and_period_contract','full_operating_domain_response_bound',
            'full_physical_enclosure_map_evidence','qualified_future_environment_conditions'],
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files})
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['trials','input_sha256']},indent=2))

if __name__=='__main__':main()
