#!/usr/bin/env python3
"""Read-only diagnostic digest; never turn a missing exact join into a match."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from analyze_phase_e_observation_events import string_rows


def summarize(report):
    cases = []
    for case in report['cases']:
        observation = case['observation']
        run = Path(case['run']); window = case['benchmark_window']
        plans = string_rows(run / 'local_planner_summary.csv', window)
        regimes = string_rows(run / 'regime_summary.csv', window)
        obs = observation['observations']
        gate_counts = Counter()
        voxel_counts = Counter()
        critical = []
        for target in obs:
            rejected = [v for v in target['verification'] if v.get('committed') != '1']
            gate_counts.update(v.get('safety_gate', 'unknown') for v in rejected)
            exact = [v for v in rejected if v.get('vbc_detail', {}).get('violations')]
            if exact:
                latest = exact[-1]
                critical.append(dict(observation_token=target['observation_token'],
                    rejected_exact_count=len(exact), last_outcome=latest,
                    measured_q_distance_min_inf=target['measured_q_distance_min_inf'],
                    seen_query_count=target['seen_query_count'], last_stage=target['last_stage'],
                    scheduled=next((e for e in target['history'] if e['stage'] == 'scheduled'), None)))
            for outcome in exact:
                prefix = float(outcome.get('probe_effective_prefix_s', 'nan'))
                if float(outcome.get('cycle_recovery_active', 'nan')) == 1:
                    prefix = float(outcome.get('cycle_recovery_prefix_s', 'nan'))
                brake = float(outcome.get('brake_duration_s', 'nan'))
                for voxel in outcome['vbc_detail']['violations']:
                    voxel_counts['records'] += 1
                    voxel_counts['before_start_deadline'] += voxel['latest_see_s'] < 0
                    voxel_counts['never_nominally_visible'] += not voxel['nominally_visible']
                    membership = voxel.get('in_selected_solver_fp32_inputs')
                    voxel_counts['in_target' if membership is True else 'outside_target' if membership is False else 'target_unknown'] += 1
                    t = voxel['sweep_s']
                    segment = ('unknown' if not math.isfinite(prefix) else 'prefix' if t <= prefix + 1e-9
                               else 'brake' if math.isfinite(brake) and t <= prefix + brake + 1e-9
                               else 'hold' if math.isfinite(brake) else 'unknown')
                    voxel_counts['sweep_segment_' + segment] += 1
        critical.sort(key=lambda e: e['rejected_exact_count'], reverse=True)
        failures = [p for p in plans if p.get('event') in (
            'repair_no_candidate_wait_external_trigger', 'repair_qp_retry_scheduled',
            'repair_qp_exhausted', 'repair_qp_stale_progress_retry',
            'repair_plan_retry_scheduled', 'repair_plan_exhausted')]
        unique_failures = {p['plan_seq']: p for p in failures}
        cases.append(dict(case_id=case['case_id'], model=case['model'], task_success=case['task_success'],
            overall_safe=case['overall_safe'], startup_order_pass=case['startup_order_pass'],
            final_position_error_m=case['final_position_error_m'], final_orientation_error_rad=case['final_orientation_error_rad'],
            audit=observation['vbc_evidence_audit'], observation_reject_gates=dict(gate_counts),
            observation_voxel_records=dict(voxel_counts),
            repair_no_candidate_plans=list(unique_failures.values()),
            last_local_event=plans[-1] if plans else None, last_regime=regimes[-1] if regimes else None,
            critical_targets=critical))
    return dict(schema_version=1, cases=cases,
        caveat='Voxel counts are audit records, not unique voxels. Sweeping link is NOT an occlusion link. '
               'Missing exact bundle routes stay unknown; periodic same-trajectory refreshes are different audits. '
               'A negative latest_see_s means this fixed trajectory cannot create the required lead time from t=0. '
               'No path feasibility proof or causal sensor attribution.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = summarize(json.loads(Path(args.summary).read_text()))
    with Path(args.output).open('x') as stream:
        json.dump(report, stream, indent=2)
    for case in report['cases']:
        print(case['model'], case['case_id'], 'task', case['task_success'], 'safe', case['overall_safe'],
              'audit', case['audit'], 'repair_no_candidate_plans', len(case['repair_no_candidate_plans']))


if __name__ == '__main__': main()
