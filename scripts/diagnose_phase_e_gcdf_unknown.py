#!/usr/bin/env python3
"""Recorded UNKNOWN -> observation -> explicit map disposition chronology.

No nearest-time identity joins; old seq-only associations are labelled weaker.
No missing row is interpreted as free, and no rejected trajectory is invented.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from analyze_phase_e_observation_events import string_rows, solver_point_key


def diagnose(run):
    run=Path(run)
    window=json.loads((run/'goal_stop_status.json').read_text())['benchmark_window']
    names=['goal_stop_status.json','verification_outcome.csv','final_gcdf_recovery_event.csv',
           'local_witness_selector_diagnostics.csv','local_planner_summary.csv',
           'projector_traces/observation_events.jsonl']
    hashes={n:hashlib.sha256((run/n).read_bytes()).hexdigest() for n in names}
    outcomes=string_rows(run/'verification_outcome.csv',window)
    events=[json.loads(l) for l in (run/names[-1]).read_text().splitlines()]
    events=[e for e in events if window['start_ros_s']<=e['ros_s']<=window['end_ros_s']]
    witnesses=string_rows(run/'local_witness_selector_diagnostics.csv',window)
    local=string_rows(run/'local_planner_summary.csv',window)
    recovery=[]
    with (run/'final_gcdf_recovery_event.csv').open() as f:
        # Variable-length Float64MultiArray: rostopic keeps only the first header.
        for row in list(csv.reader(f))[1:]:
            receipt=int(row[0])*1e-9
            if not window['start_ros_s']<=receipt<=window['end_ros_s']: continue
            values=list(map(float,row[2:])); count=int(values[5])
            assert len(values)==6+3*count
            recovery.append(dict(seq=str(int(values[0])),receipt_ros_s=receipt,
                stamp_ns=str(int(values[1])*10**9+int(values[2])),
                sweep_s=values[3],timestep=int(values[4]),
                points=[values[6+3*i:9+3*i] for i in range(count)]))
    groups={}
    for event in recovery:
        matched=[o for o in outcomes if o.get('seq')==event['seq'] and
                 o.get('result')=='unsafe' and o.get('safety_gate','').startswith('gcdf')]
        assert len(matched)==1, 'Ambiguous/missing per-run recovery seq'
        outcome=matched[0]; audited=outcome.get('audited_trajectory_stamp_ns','0')
        if audited!='0': assert audited==event['stamp_ns'], 'GCDF audit identity mismatch'
        for point in event['points']:
            key=solver_point_key(point)
            group=groups.setdefault(key,dict(point=point,rejections=[]))
            group['rejections'].append(dict(event,points=None,
                raw_candidate_stamp_ns=outcome['raw_candidate_stamp_ns'],
                identity='exact_audit_stamp' if audited!='0' else 'legacy_unique_seq_only'))
    for key,g in groups.items():
        schedules=[e for e in events if e['stage']=='scheduled' and
                   any(solver_point_key(p)==key for p in e.get('target_solver_points',e.get('points',[])))]
        # A shared target's ACTUAL_SEEN event certifies the queried member region,
        # not all target_solver_points. Match the exact original region token.
        regions={e['region_token'] for e in events if 'region_token' in e and
                 any(solver_point_key(p)==key for p in e.get('points',[]))}
        seen=[e for e in events if e.get('actual_seen') and e.get('region_token') in regions]
        wr=[r for r in witnesses if solver_point_key(json.loads(r['point']))==key]
        free=[r for r in wr if r['status']=='resolved_free']
        first_seen=min((e['ros_s'] for e in seen),default=None)
        g.update(rejection_count=len(g['rejections']),
            sweep_counts=dict(Counter(str(e['sweep_s']) for e in g['rejections'])),
            scheduled=[{k:e.get(k) for k in ['ros_s','observation_token','target_source','q_vis']} for e in schedules],
            first_actual_seen_ros_s=first_seen,
            first_resolved_free_ros_s=min((r['ros_s'] for r in free),default=None),
            resolved_free_rows=free, witness_status_counts=dict(Counter(r['status'] for r in wr)),
            post_seen_unknown_rows=sum(r['status']=='evaluated_unknown' and r['ros_s']>first_seen
                                      for r in wr) if first_seen is not None else None,
            post_seen_rejections=sum(r['receipt_ros_s']>first_seen for r in g['rejections'])
                                 if first_seen is not None else None)
    return dict(status='PASS',meaning='Recorded diagnostic consistency, not task success',
        source_run=str(run.resolve()),source_sha256=hashes,window=window,
        gcdf_rejections=sum(o.get('result')=='unsafe' and o.get('safety_gate','').startswith('gcdf') for o in outcomes),
        repair_cdf_horizons=sorted({r['cdf_horizon_steps'] for r in local if r.get('repair')=='1'}),
        points=sorted(groups.values(),key=lambda g:-g['rejection_count']),
        limitations=['Historical GCDF outcomes with audit stamp=0 have only unique per-run seq association.',
            'No full rejected q/t or map replay inferred from CSV chronology.',
            'FP32 point identity reflects actual solver inputs, not nearest-neighbour matching.',
            'UNKNOWN is not physical occupied collision evidence.'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args();result=diagnose(a.run)
    a.output_dir.mkdir(parents=True,exist_ok=False)
    with (a.output_dir/'result.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ['status','gcdf_rejections','repair_cdf_horizons']}))
    for p in result['points']:print(p['point'],p['rejection_count'],p['first_actual_seen_ros_s'],p['first_resolved_free_ros_s'])


if __name__=='__main__':main()
