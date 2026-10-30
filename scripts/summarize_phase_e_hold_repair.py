#!/usr/bin/env python3
"""Archive only this batch's named snapshots; replay unchanged local QPs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from analyze_phase_e_observation_events import string_rows
from phase_e_evaluation_common import window_time_ns
from summarize_phase_e_fix34_regression import bound_conflicts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--summary', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--replay-binary', default='/tmp/care_replay_qp')
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    report = json.loads(Path(args.summary).read_text())
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=False)
    cases = []
    for case in report['cases']:
        run = Path(case['run']); w = case['benchmark_window']
        plans = string_rows(run/'local_planner_summary.csv', w)
        feedback = [p for p in plans if p.get('event','').startswith(('repair_qp_', 'repair_plan_'))]
        bounded = all(0 <= int(p['repair_failures']) <= 5 for p in feedback)
        log = (repo/'logs/c5_5_vbc_gcdf_regime'/run.parent.name/'run/controlled.log').read_text(errors='replace')
        snapshots = []
        for path in dict.fromkeys(re.findall(r'QP failure snapshot=(/tmp/care_qp_failure_[A-Za-z0-9]+)', log)):
            source = Path(path); data = json.loads(source.read_text())
            if not data.get('repair'): continue
            dest = out/(case['model']+'_'+case['case_id']+'_'+source.name+'.json')
            shutil.copyfile(source, dest)
            replay = subprocess.run([args.replay_binary,str(dest)],capture_output=True,text=True)
            result = json.loads(replay.stdout) if replay.stdout.strip().startswith('{') else dict(stdout=replay.stdout, stderr=replay.stderr)
            conflicts = bound_conflicts(data)
            stamp = data['query_stamp_ns']
            snapshots.append(dict(path=str(dest), sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                query_stamp_ns=stamp, within_window=window_time_ns(w,'start_ros') <= stamp <= window_time_ns(w,'end_ros'),
                plan_sequence=data['plan_sequence'], observation_token=data['observation_token'],
                repair_failures=data['repair_failures'], target_revision=data['target_revision'],
                progress_epoch=data['progress_epoch'], status=data['status'],
                selected_rows=len(data['selected_pairs']), conflicts=conflicts,
                conflicting_source_types=dict(Counter(str(c['source_type']) for c in conflicts)),
                replay_pass=replay.returncode == 0, replay=result))
        stop = json.loads((run/'goal_stop_status.json').read_text())
        cases.append(dict(model=case['model'], case_id=case['case_id'], task_success=case['task_success'],
            overall_safe=case['overall_safe'], startup_pass=case['startup_order_pass'],
            online_offline_agree=stop['goal_reached'] == case['task_success'],
            integer_window=w.get('schema_version') == 3,
            final_position_error_m=case['final_position_error_m'], final_orientation_error_rad=case['final_orientation_error_rad'],
            repair_feedback_counts=dict(Counter(p['event'] for p in feedback)),
            feedback=feedback, failure_counts_bounded=bounded,
            last_local_event=plans[-1] if plans else None, snapshots=snapshots))
        print(case['model'], case['case_id'], 'task',case['task_success'], 'feedback',cases[-1]['repair_feedback_counts'],
              'snapshots',len(snapshots), 'replay_pass', all(s['replay_pass'] for s in snapshots), flush=True)
    result = dict(cases=cases, tests_pass=all(c['online_offline_agree'] and c['integer_window'] and
        c['startup_pass'] and c['overall_safe'] and c['failure_counts_bounded'] and
        all(s['replay_pass'] for s in c['snapshots']) for c in cases),
        caveat='Per-target counters are diagnostic liveness, not global infeasibility. '
        'Independent row/box contradiction only certifies the stored linearized QP. '
        'Missing full-system success is not hidden; one run is not a success-rate estimate.')
    with (out/'report.json').open('x') as stream: json.dump(result,stream,indent=2)
    if not result['tests_pass']: raise SystemExit(1)


if __name__ == '__main__': main()
