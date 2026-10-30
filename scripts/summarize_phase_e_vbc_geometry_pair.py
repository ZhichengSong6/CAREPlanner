#!/usr/bin/env python3
"""Offline geometry-pair audit; preserves simulations and preliminary reports."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from run_phase_e_repeatability import REPO, fingerprint, save
from run_phase_e_vbc_geometry_pair import geometry_audit, distribution
from analyze_phase_e_observation_events import string_rows
from phase_e_evaluation_common import window_time_ns
from summarize_phase_e_fix34_regression import bound_conflicts


def replay_audit(case, out, binary):
    """Archive only log-named snapshots; include NORMAL/PROBE as well as REPAIR."""
    run, window = Path(case['run']), case['benchmark_window']
    log = (REPO/'logs/c5_5_vbc_gcdf_regime'/run.parent.name/'run/controlled.log').read_text(errors='replace')
    snapshots = []
    for path in dict.fromkeys(re.findall(r'QP failure snapshot=(/tmp/care_qp_failure_[A-Za-z0-9]+)', log)):
        source = Path(path)
        if not source.is_file():
            snapshots.append(dict(source=str(source), status='NOT_RUN', reason='snapshot missing'))
            continue
        data = json.loads(source.read_text())
        dest = out/(case['geometry_backend']+'_'+case['case_id']+'_'+source.name+'.json')
        shutil.copyfile(source, dest)
        replay = subprocess.run([str(binary), str(dest)], capture_output=True, text=True, timeout=60)
        result = json.loads(replay.stdout) if replay.stdout.strip().startswith('{') else dict(
            stdout=replay.stdout, stderr=replay.stderr)
        stamp = data['query_stamp_ns']
        snapshots.append(dict(path=str(dest), sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
            status='PASS' if replay.returncode == 0 else 'FAIL',
            within_window=window_time_ns(window, 'start_ros') <= stamp <= window_time_ns(window, 'end_ros'),
            query_stamp_ns=stamp,
            phase='REPAIR' if data.get('repair') else ('PROBE' if data.get('probe') else 'NORMAL'),
            original_status=data['status'], original_iterations=data['iterations'],
            original_primal_residual=data['primal_residual'], original_dual_residual=data['dual_residual'],
            settings={k: data[k] for k in ('max_iter', 'eps_abs', 'eps_rel', 'kkt_solver')},
            selected_rows=len(data['selected_pairs']), source_types=sorted(set(data['source_type'])),
            independent_row_box_conflicts=bound_conflicts(data), replay=result))
    rows = string_rows(run/'local_planner_summary.csv', window)
    failures = [r for r in rows if r.get('event') == 'scp_failed']
    return dict(status=('NOT_RUN' if not snapshots else
                       'PASS' if all(s['status'] == 'PASS' for s in snapshots) else 'FAIL'),
        snapshots=snapshots, scp_failed_status_counts=dict(Counter(r.get('status') for r in failures)),
        caveat='PASS means the original failure status was reproduced, not QP success. '
               'No row/box contradiction does not prove joint feasibility; snapshot logging may be capped.')


def source_audit(case):
    run, window = Path(case['run']), case['benchmark_window']
    result = {}
    for role in ('candidate', 'execution'):
        rows = string_rows(run/(role+'_vbc_summary.csv'), window)
        result[role] = {}
        for source in sorted({r.get('trajectory_source', 'missing') for r in rows}):
            group = [r for r in rows if r.get('trajectory_source', 'missing') == source]
            point_counts, links = Counter(), Counter()
            deadlines = []
            for row in group:
                evidence = json.loads(row['vbc_evidence'])
                for v in evidence['violations']:
                    point_counts[tuple(v['point'])] += 1
                    links[v['sweeping_link']] += 1
                    deadlines.append(v['latest_see_s'])
            result[role][source] = dict(records=len(group),
                unsafe=sum(r.get('has_violation') == '1' for r in group),
                unique_violation_points=len(point_counts), links=dict(links),
                top_points=[dict(point=list(p), records=n) for p, n in point_counts.most_common(8)],
                negative_deadlines=sum(t < 0 for t in deadlines),
                evaluation_triggers=dict(Counter(r.get('evaluation_trigger', 'missing') for r in group)),
                timing_ms={key: distribution([float(r[key]) for r in group if key in r]) for key in
                           ('trajectory_convert_ms', 'body_fk_ms', 'swept_voxel_build_ms',
                            'confidence_query_ms', 'candidate_filter_ms', 'sensor_fk_ms',
                            'visibility_scan_ms', 'cluster_ms', 'total_eval_ms')})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batch-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--replay-binary', type=Path, default=Path('/tmp/care_replay_qp'))
    args = ap.parse_args()
    batch, out = args.batch_dir.resolve(), args.output_dir.resolve()
    original = json.loads((batch/'summary.json').read_text())
    expected = json.loads((batch/'frozen_sha256.json').read_text())
    actual = fingerprint()
    differences = sorted(p for p in expected.keys() | actual.keys() if expected.get(p) != actual.get(p))
    assert set(differences) <= {'scripts/run_phase_e_vbc_geometry_pair.py',
                                'scripts/summarize_phase_e_vbc_geometry_pair.py'}, differences
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    for case in original['cases']:
        backend = case['geometry_backend']
        qualification = REPO/'outputs/phase_e_empty_world_qualification'/('phase_e_'+batch.name+'-'+backend)
        q = json.loads((qualification/'case_summaries'/(case['case_id']+'.json')).read_text())
        geometry = geometry_audit(case, backend)
        r, o = case['repair_state'], case['observation']
        witnesses = Counter(tuple(json.loads(w['point'])) for w in r['witness_diagnostics'] if 'point' in w)
        counters = {key: q[key] for key in ('commit_count', 'final_gcdf_unsafe_rejection_count',
            'exact_vbc_unsafe_rejection_count', 'verification_timeout_count', 'execution_vbc_unsafe_records')}
        seen = o['stage_counts'].get('ACTUAL_SEEN', 0)
        cases.append(dict(case_id=case['case_id'], model=case['model'], geometry_backend=backend,
            run=case['run'], benchmark_window=case['benchmark_window'],
            task_success=case['task_success'], overall_safe=case['overall_safe'],
            startup_order_pass=case['startup_order_pass'],
            final_position_error_m=case['final_position_error_m'],
            final_orientation_error_rad=case['final_orientation_error_rad'],
            counters=counters, verification_outcomes=geometry['verification'],
            plans_by_phase=r['plans_by_phase'], candidates_by_phase=r['candidates_by_phase'],
            qp_failure_statuses=r['failure_status_counts'], witness_max=r['max_retained_witnesses'],
            witness_reasons=r['witness_reason_counts'], witness_unique_points=len(witnesses),
            witness_points=[dict(point=list(p), records=n) for p, n in witnesses.most_common()],
            witness_missing_failures=len(r['witness_missing_failures']),
            last_regime={k: (r['last_regime'] or {}).get(k) for k in ('state', 'reason', 'tracker_active')},
            last_local_event={k: (r['last_local_event'] or {}).get(k) for k in
                              ('event', 'repair_phase', 'status', 'repair_failures', 'repair_witness_count')},
            observation_stages=o['stage_counts'], actual_seen_tokens=seen,
            failed_generations=len(o['failed_generations']), unfinished_generations=len(o['unfinished_generations']),
            unmatched_verifications=o['unmatched_verification_count'], hybrid_latency=o['hybrid_latency_ms'],
            vbc_by_source=source_audit(case), geometry_audit=geometry,
            qp_replay=replay_audit(case, out, args.replay_binary.resolve())))
    report = dict(status='FAIL' if any(c['qp_replay']['status'] == 'FAIL' for c in cases) else 'PASS',
        cases=cases, completed_runs=len(cases),
        replay_binary_sha256=hashlib.sha256(args.replay_binary.read_bytes()).hexdigest(),
        pairs=original['pairs'], runtime_frozen_inputs_equal=True, post_run_analysis_changes=differences,
        verification_counter_source='result:safety_gate from raw traces; corrects the legacy first-batch '
                                    'preliminary counter that used a nonexistent safe field',
        task_success={b: sum(c['task_success'] for c in cases if c['geometry_backend'] == b)
                      for b in ('samples', 'primitive')},
        safety_pass={b: sum(c['overall_safe'] for c in cases if c['geometry_backend'] == b)
                     for b in ('samples', 'primitive')},
        caveat='Single asynchronous run per case/backend. Bootstrap/rediscovery violations are not final '
               'candidate rejections. Repeated witness records are not unique constraints or obstacles.')
    save(out/'summary.json', report)
    for c in cases:
        print(c['geometry_backend'], c['case_id'], 'task', c['task_success'], 'safe', c['overall_safe'],
              'commits', c['counters']['commit_count'], 'verification', c['verification_outcomes'],
              'QP replay', c['qp_replay']['status'], len(c['qp_replay']['snapshots']),
              'seen', c['actual_seen_tokens'], flush=True)
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
