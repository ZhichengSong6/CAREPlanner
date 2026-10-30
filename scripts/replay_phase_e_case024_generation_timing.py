#!/usr/bin/env python3
"""Recorded seed/execution binding and the production stability gate on fixed input.

The later recorded motions are not a counterfactual rollout of holding a grant.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/care_visibility_cdf/scripts'),str(ROOT/'scripts')]
from candidate_replacement_runtime import StableGenerationSeed
from diagnose_phase_e_case006_o6_recovery import strings
RUN=ROOT/'outputs/c5_5_vbc_gcdf_regime/phase_e_goal_024_r1_case024_owner_fallback_20261009/run'
OUT=ROOT/'outputs/phase_e_case024_stable_seed_20261009'

def main():
    inputs=[]
    def rows(name,raw=False):
        p=RUN/(name+'.csv');inputs.append(p)
        if not raw:return strings(p)
        with p.open() as f:return list(csv.DictReader(f))
    path=RUN/'projector_traces/candidate_replacement.jsonl';inputs.append(path)
    repl=[json.loads(l) for l in path.read_text().splitlines()]
    grants=rows('candidate_replacement_grant');requests=rows('candidate_replacement_request')
    js=rows('joint_states',True);tracker=rows('tracker_summary');outcomes=rows('verification_outcome')
    raw_tracker={int(r['%time']):r['field.data'] for r in rows('tracker_summary',True)}
    commits={r['execution_stamp_ns']:r for r in outcomes if r['committed']=='1'}
    samples=[dict(receipt_ns=int(r['%time']),stamp_ns=int(r['field.header.stamp']),
                  q=[float(r['field.position'+str(j)]) for j in range(7)]) for r in js]
    timeline=sorted([(r['receipt_ns'],0,r) for r in samples]+[(r['receipt_ns'],1,r) for r in tracker],key=lambda v:v[:2])
    results=[]
    for e in repl:
        if e['event']!='measured_state_changed':continue
        rid=str(e['request_id']);grant=next(g for g in grants if g['request_id']==rid)
        request=next(r for r in requests if r['request_id']==rid and r['action']=='reserve')
        end=round(e['ros_s']*1e9)
        matches=[s for s in samples if request['receipt_ns']-200000000<=s['receipt_ns']<=end and
                 np.max(abs(np.array(s['q'])-e['measured_seed']))<1e-12]
        assert len(matches)==1,[(s['stamp_ns'],s['receipt_ns']) for s in matches]
        seed=matches[0]
        tracking=next(t for t in reversed(tracker) if t['receipt_ns']<=seed['receipt_ns'])
        commit=commits[tracking['execution_stamp_ns']]
        assert tracking['active']=='1' and float(tracking['remaining_s'])>0
        assert commit['result']=='safe' and commit['committed']=='1'
        app=next(s for s in reversed(samples) if s['receipt_ns']<=end)
        delta=float(np.max(abs(np.array(app['q'])-e['measured_seed'])))
        assert delta>.01
        gate=StableGenerationSeed();first_ready=None;at_seed=None
        for t,kind,r in timeline:
            if t<request['receipt_ns']-300000000:continue
            if t>end+2500000000:break
            now=t/1e9
            if kind==0:gate.observe_measured(r['q'],r['stamp_ns']/1e9)
            else:gate.observe_tracker(raw_tracker[t],now)
            ready,detail=gate.ready(now,grant['receipt_ns']/1e9)
            if t==seed['receipt_ns'] and kind==0:at_seed=dict(ready=ready,detail=detail)
            if first_ready is None and ready:
                latest=next(s for s in reversed(samples) if s['receipt_ns']<=t)
                first_ready=dict(receipt_ns=t,measured=latest,detail=detail,
                    before_original_ros_deadline=t<int(grant['deadline_ns']))
        assert at_seed is not None and not at_seed['ready']
        assert first_ready is not None
        results.append(dict(request_id=e['request_id'],sensor=e['hybrid']['selected_sensor_id'],
            geometry_accepted=e['hybrid']['accepted'],request=request,grant=grant,
            exact_seed_sample=seed,tracker_before_seed=tracking,certified_execution_commit=commit,
            generation_finish_ros_s=e['ros_s'],nearest_application_sample=app,
            nearest_seed_delta_inf=delta,gate_at_recorded_seed=at_seed,
            first_ready_on_unchanged_recorded_stream=first_ready))
    assert [r['sensor'] for r in results]==[6,3]
    for path in [Path(__file__).resolve(),ROOT/'src/care_visibility_cdf/scripts/candidate_replacement_runtime.py']:
        inputs.append(path)
    report=dict(status='PASS',cases=results,limits=[
        'Generation start/seed header inferred from an exact unique recorded q match, not an original start log.',
        'Application sample uses latest CSV receipt before finish, not the internal callback identity.',
        'Readiness after holding a real reservation may be earlier: the unchanged recorded stream contains later commits that a pending reservation would prevent.',
        'No new hard QP, map, VBC or execution certificate in this fixed scheduling replay.'],
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (OUT/'timing_replay.json').write_text(json.dumps(report,indent=2)+'\n')
    for r in results:print('request',r['request_id'],'sensor',r['sensor'],'seed_s',r['exact_seed_sample']['stamp_ns']/1e9,
        'finish_s',r['generation_finish_ros_s'],'motion',r['nearest_seed_delta_inf'],
        'remaining',r['tracker_before_seed']['remaining_s'],'gate_ready_s',r['first_ready_on_unchanged_recorded_stream']['receipt_ns']/1e9)
if __name__=='__main__':main()
