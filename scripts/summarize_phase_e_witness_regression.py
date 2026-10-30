#!/usr/bin/env python3
"""Offline-only recovery/audit of the four witness regression simulations.

Never starts ROS, changes runtime, retries a case, or overwrites outputs.
"""
import argparse
import json
from pathlib import Path

from run_phase_e_repair_failure_regression import phase_audit
from run_phase_e_repeatability import REPO, fingerprint, save
from summarize_phase_e_qp_events import collect


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir',required=True)
    parser.add_argument('--baseline',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    batch=Path(args.batch_dir).resolve()
    stamp=batch.name
    frozen=json.loads((batch/'frozen_sha256.json').read_text())
    current=fingerprint()
    differences=sorted(p for p in frozen.keys() | current.keys()
                       if frozen.get(p)!=current.get(p))
    # These are the only permitted post-run changes for the empty-topic parser
    # recovery. Runtime/config/checkpoints/compiled artifacts must still match.
    analysis_only={
        'scripts/analyze_phase_e_observation_events.py',
        'scripts/test_phase_e_observation_events.py',
        'scripts/summarize_phase_e_witness_regression.py',
    }
    assert set(differences)<=analysis_only, ('unexpected frozen input changes',differences)
    baseline=json.loads(Path(args.baseline).read_text())
    rows=[]
    for model in ('scalar','h9'):
        assert json.loads((batch/(model+'_exit.json')).read_text())['return_code']==0
        group_stamp=stamp+'-'+model
        group=collect(REPO,REPO/'outputs/phase_e_empty_world_qualification'/('phase_e_'+group_stamp),group_stamp,model)
        assert {c['case_id'] for c in group}=={'phase_e_goal_007','phase_e_goal_026'}
        for row in group:
            row['repair_state']=phase_audit(row)
            old=next(c for c in baseline['cases'] if (c['model'],c['case_id'])==(model,row['case_id']))
            row['baseline']={k:old[k] for k in ('run','task_success','overall_safe','final_position_error_m','final_orientation_error_rad','repair_state')}
            row['witness_recording_bytes']={name:(Path(row['run'])/name).stat().st_size for name in (
                'local_witness_diagnostics.csv','local_witness_selector_diagnostics.csv')}
            rows.append(row)
    invariants=all(c['overall_safe'] and c['startup_order_pass'] and
        c['repair_state']['phase_fields_present'] and
        not c['repair_state']['retreat_candidate_gate_violations'] and
        not c['observation']['failed_generations'] and
        not c['observation']['unmatched_verification_count'] for c in rows)
    result=dict(cases=rows,completed_runs=len(rows),runtime_frozen_inputs_equal=True,
        post_run_analysis_changes=differences,invariants_pass=invariants,
        baseline=str(Path(args.baseline).resolve()),
        collection_note='Original orchestration failed during offline aggregation on a zero-byte optional ROS topic. '
                        'Both simulation subprocesses exited 0. No simulation was rerun. '
                        'Empty trace alone does not prove recorder health; inspect planner/log evidence.',
        caveat='One run per model/case; not a success-rate or causal improvement estimate.')
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=False)
    save(out/'summary.json',result)
    for c in rows:
        a=c['repair_state']
        print(c['model'],c['case_id'],'task',c['task_success'],'safe',c['overall_safe'],
              'phases',a['plans_by_phase'],'failures',a['failure_status_counts'],
              'witness',a['witness_reason_counts'],flush=True)
    print('invariants_pass',invariants,'post_run_analysis_changes',differences)
    if not invariants: raise SystemExit(1)


if __name__=='__main__': main()
