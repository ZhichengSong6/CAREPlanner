#!/usr/bin/env python3
"""Separate map-domain limits from conservative global stopping inflation.

Shorter deadlines are counterfactuals, not revocations of already issued leases.
Observed physical-step paths are diagnostics, not continuous path certificates.
"""
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
from phase_e_stop_shadow import PrimitiveEnclosure
from phase_e_stopping_contract import qualify
from audit_case001_motion_prior import frame_placements

ROOT=Path(__file__).resolve().parents[1]
LOW=np.array([-.95,-.95,0.]);HIGH=np.array([.95,.95,1.15])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();source=args.input_dir.resolve();out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    geometry=PrimitiveEnclosure(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf')
    def extent(q,travel):
        full=np.zeros(geometry.model.nq);full[geometry.qindices]=q
        placements=frame_placements(geometry.model,geometry.data,geometry.shapes,full)
        bounds=geometry.radii@travel
        minimum=np.full(3,np.inf);maximum=np.full(3,-np.inf)
        for i in geometry.active:
            shape,p=geometry.shapes[i],placements[i];rotation=np.asarray(p.rotation)
            if shape['kind']=='box':size=np.abs(rotation)@shape['half_size']
            elif shape['kind']=='cylinder':
                size=shape['radius']*np.sqrt(rotation[:,0]**2+rotation[:,1]**2)+shape['half_length']*np.abs(rotation[:,2])
            else:size=np.full(3,shape['radius'])
            size=size+.020+bounds[i]
            minimum=np.minimum(minimum,p.translation-size);maximum=np.maximum(maximum,p.translation+size)
        inside=bool(np.all(minimum>=LOW) and np.all(maximum<=HIGH))
        return dict(inside=inside,minimum=minimum.tolist(),maximum=maximum.tolist(),
            maximum_global_motion_bound_m=float(max(bounds[i] for i in geometry.active)))
    counts=collections.Counter();rows=[]
    for row in map(json.loads,(source/'shadow/rows.jsonl').open()):
        if 'map' not in row:continue
        state=row['state'];counts['complete_queries']+=1
        stationary=extent(state['q'],np.zeros(7));counts['measured_pose_inside']+=int(stationary['inside'])
        item=dict(state_ns=state['t_ns'],measured_pose_inside=stationary['inside'],shorter_deadline_inside={})
        for remaining in [0.,.01,.025,.05,.1]:
            now=row['attempt_ns']*1e-9
            result=qualify(now_s=now,measured_s=state['t_ns']*1e-9,applied=state['applied'],
                acceleration=state['acceleration_limits'],velocity_limits=state['velocity_limits'],
                next_decision_s=.01,transport_bound_s=0.,update_gap_s=.001,lease_end_s=now+remaining,
                lease_enforced=True,physical_error_rad=None,physical_settle_s=None,snapshot=None,
                max_braking_step_s=state['max_braking_step_s'])
            bound=result['budgets']['continue_then_stop']['command_travel_rad']
            box=extent(state['q'],np.asarray(bound))
            item['shorter_deadline_inside'][str(remaining)]=box['inside']
            counts['hypothetical_remaining_'+str(remaining)+'_inside']+=int(box['inside'])
        rows.append(item)
    run=json.loads((source/'result.json').read_text());states={}
    for row in map(json.loads,(source/'controller_trace.jsonl').open()):
        if row['kind']=='state':states[row['t_ns']]=row
    windows=[dict(start=t['baseline_ns'],end=t['end_ns'],q=[]) for t in run['trials']]
    for row in map(json.loads,(source/'physics_steps.jsonl').open()):
        for window in windows:
            if window['start']<=row['t_ns']<=window['end']:window['q'].append(row['q_after'])
    trials=[]
    for trial,window in zip(run['trials'],windows):
        state=states[trial['baseline_ns']]
        result=qualify(now_s=state['t_ns']*1e-9,measured_s=state['t_ns']*1e-9,applied=state['applied'],
            acceleration=state['acceleration_limits'],velocity_limits=state['velocity_limits'],
            next_decision_s=.01,transport_bound_s=0.,update_gap_s=.001,
            lease_end_s=trial['fixed_until_ns']*1e-9,lease_enforced=True,
            physical_error_rad=None,physical_settle_s=None,snapshot=None,max_braking_step_s=state['max_braking_step_s'])
        global_box=extent(state['q'],np.asarray(result['budgets']['withdraw_now']['command_travel_rad']))
        initial=extent(state['q'],np.zeros(7));actual_inside=True
        minimum=np.full(3,np.inf);maximum=np.full(3,-np.inf)
        for q in window['q']:
            box=extent(q,np.zeros(7));actual_inside=actual_inside and box['inside']
            minimum=np.minimum(minimum,box['minimum']);maximum=np.maximum(maximum,box['maximum'])
        trials.append(dict(label=trial['label'],measured_pose_inside=initial['inside'],
            global_command_enclosure=global_box,observed_discrete_path_inside=actual_inside,
            observed_discrete_path_minimum=minimum.tolist(),observed_discrete_path_maximum=maximum.tolist(),
            physics_steps=len(window['q'])))
    result=dict(scope='geometry-only counterfactuals and discrete observed paths; no map confidence or admission proof',
        runtime_admission=False,counts=dict(counts),rows=rows,trials=trials,
        explanation='Hypothetical deadline uses that remaining duration plus one more 10ms decision cycle; cannot shorten an outstanding real lease.',
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source/'result.json',
            source/'shadow/rows.jsonl',source/'physics_steps.jsonl',Path(__file__).resolve()]})
    (out/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(counts=dict(counts),trial_count=len(trials),
        trial_initial_inside=sum(t['measured_pose_inside'] for t in trials),
        trial_global_enclosure_inside=sum(t['global_command_enclosure']['inside'] for t in trials),
        trial_observed_path_inside=sum(t['observed_discrete_path_inside'] for t in trials),
        global_outside_but_observed_inside=[t['label'] for t in trials if not t['global_command_enclosure']['inside'] and t['observed_discrete_path_inside']]),indent=2))

if __name__=='__main__':main()
