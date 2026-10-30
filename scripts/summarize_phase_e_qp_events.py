#!/usr/bin/env python3
"""Collect bounded QP/observation runs; check startup ordering and exact joins."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from analyze_phase_e_observation_events import analyze


def collect(repo, batch, stamp, model):
    data=json.loads((batch/'qualification_summary.json').read_text())
    metadata=dict(line.split('=',1) for line in (batch/'qualification_metadata.txt').read_text().splitlines() if '=' in line)
    rows=[]
    for case in data['cases']:
        run_id=case['case_id']+'_empty_qualification_'+stamp+'_'+metadata['git_head'][:8]
        run=repo/'outputs/c5_5_vbc_gcdf_regime'/run_id/'run'
        log=(repo/'logs/c5_5_vbc_gcdf_regime'/run_id/'run/controlled.log').read_text(errors='replace')
        events=analyze(run,log)
        w=case['benchmark_window']
        with (run/'ee_target_pose.csv').open() as stream:
            goals=list(csv.DictReader(stream))
        goal_stamp=int(goals[0]['field.header.stamp'])*1e-9 if goals else None
        order_ok=bool(goal_stamp is not None and w['start_ros_s'] <= goal_stamp and
                      max(map(abs,w['first_measured_q'])) < 1e-4 and
                      w.get('protocol')=='pre_goal_arm_v1')
        rows.append(dict(case_id=case['case_id'],model=model,run=str(run),
            task_success=case['task_success'],overall_safe=case['overall_safe'],
            final_position_error_m=case['final_position_error_m'],
            final_orientation_error_rad=case['final_orientation_error_rad'],
            benchmark_window=w,first_goal_ros_s=goal_stamp,startup_order_pass=order_ok,
            normal_box_reseed_count=log.count('NORMAL_BOX_RESEED'),
            observation=events))
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--qp-stamp',required=True); ap.add_argument('--ab-stamp',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args(); repo=Path(__file__).resolve().parents[1]
    root=repo/'outputs/phase_e_empty_world_qualification'
    rows=collect(repo,root/('phase_e_qp_events_'+a.qp_stamp),a.qp_stamp,'qp_h9')
    for model in ('scalar','h9'):
        stamp=a.ab_stamp+'-'+model
        rows+=collect(repo,root/('phase_e_'+stamp),stamp,model)
    manifests=[json.loads((repo/'outputs/phase_e_qp_events_inputs'/
        (a.ab_stamp+'-'+model+'.source_sha256.json')).read_text()) for model in ('scalar','h9')]
    runtime_paths=[p for p in manifests[0] if p.startswith('src/') or p.endswith('.sh') or
                   p in ('scripts/wait_for_phase_d_goal.py','scripts/phase_e_benchmark_arm.py')]
    mismatch=[p for p in runtime_paths if manifests[0][p]!=manifests[1].get(p)]
    report=dict(schema_version=1,cases=rows,ab_runtime_source_equal=not mismatch,
                ab_runtime_source_mismatches=mismatch,
                startup_order_pass=all(r['startup_order_pass'] for r in rows),
                raw_model_success_counts={m:sum(r['task_success'] for r in rows if r['model']==m)
                                          for m in ('qp_h9','scalar','h9')})
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as stream: json.dump(report,stream,indent=2)
    for r in rows:
        o=r['observation']; print(r['case_id'],r['model'],'task',r['task_success'],'safe',r['overall_safe'],
            'startup',r['startup_order_pass'],'generation_errors',len(o['failed_generations']),
            'stages',o['stage_counts'],'unmatched',o['unmatched_verification_count'])
    print('A/B runtime source identical:',not mismatch)
    if mismatch or not report['startup_order_pass']: raise SystemExit(1)


if __name__=='__main__': main()
