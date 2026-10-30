#!/usr/bin/env python3
"""Read-only exp76 first-step infeasibility certificate; no ROS or model calls.

The certificate relaxes all other constraints, including joint/trust bounds.
Infeasibility of this necessary subset proves infeasibility of the recorded QP;
it does not prove that the nonlinear robot planning problem has no safe path.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.optimize import linprog

from analyze_phase_e_observation_events import solver_point_key, string_rows

ROOT = Path(__file__).resolve().parents[1]
RID = 'case_001_r1_replacement_20260921_exp76'
RUN = ROOT / 'outputs/c5_5_vbc_gcdf_regime' / RID / 'run'
LOG = ROOT / 'logs/c5_5_vbc_gcdf_regime' / RID / 'run/controlled.log'
WINDOW = dict(start_ros_s=0., end_ros_s=100.)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = [RUN / (name + '.csv') for name in (
        'local_planner_summary', 'local_witness_selector_diagnostics',
        'local_gcdf_query', 'verification_outcome')]
    paths += [LOG, Path(__file__).resolve(), ROOT / 'scripts/analyze_phase_e_observation_events.py',
              ROOT / 'src/egocentric_arm_planner/src/local_sparse_scp_planner.cpp']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    manifest = json.loads((ROOT / 'outputs/phase_e_candidate_replacement_runtime/exp76_pre_run_sha256.json').read_text())
    changed = [p for p, digest in manifest.items() if sha(ROOT / p) != digest]
    assert not changed, changed
    log = LOG.read_text()
    prefix = '/local_sparse_scp_planner_node/'

    def parameter(key):
        return re.search(re.escape(prefix + key) + r': ([^\n]+)', log)[1].strip()

    joints = ['joint1', 'joint2', 'joint3', 'joint4', 'wrist_joint1', 'wrist_joint2', 'wrist_joint3']
    velocity = np.array([float(parameter('mpc/joint_velocity_limits/' + j)) for j in joints])
    dt = float(parameter('local_planner/horizon_duration')) / int(parameter('local_planner/num_intervals'))
    assert parameter('local_planner/cdf/slack_enabled') == 'False'
    base = float(parameter('local_planner/cdf/visibility_obligation_cdf_margin'))
    increment = float(parameter('local_planner/cdf/vbc_feedback_cdf_margin_step'))
    cap = float(parameter('local_planner/cdf/vbc_feedback_cdf_margin_max'))
    summaries = string_rows(paths[0], WINDOW)
    witness = string_rows(paths[1], WINDOW)
    with paths[2].open() as stream:
        queries = {r['field.header.stamp']: r for r in csv.DictReader(stream)}
    outcomes = string_rows(paths[3], WINDOW)
    margins = {}
    feedback = []
    for r in outcomes:
        if r.get('result') != 'unsafe' or r.get('safety_gate') != 'vbc':
            continue
        evidence = json.loads(r['vbc_evidence'])
        assert evidence['trajectory_stamp_ns'] == r['audited_trajectory_stamp_ns']
        points = {solver_point_key(v['point']): v['point'] for v in evidence['violations']}
        updates = []
        for key, point in points.items():
            old = margins.get(key, base)
            margins[key] = min(cap, old + increment)
            updates.append(dict(point=point, previous_margin=old, margin=margins[key]))
        feedback.append(dict(ros_s=r['ros_s'], raw=r['raw_candidate_stamp_ns'],
                             audited=r['audited_trajectory_stamp_ns'], updates=updates,
                             evidence=evidence))
    lines = [line for line in log.splitlines() if 'exact VBC queued-target feedback' in line]
    assert len(lines) == len(feedback) == 2
    assert not any('exact VBC active-target feedback' in line for line in log.splitlines())
    for event, line in zip(feedback, lines):
        assert abs(float(re.search(r', ([0-9.]+)\]', line)[1]) - event['ros_s']) < .005
        assert int(re.search(r' points=(\d+)', line)[1]) == len(event['updates'])
        assert abs(float(re.search(r'cdf_margin_max=([0-9.]+)', line)[1]) - max(u['margin'] for u in event['updates'])) < 1e-12

    critical_point = [-.15, 0., .9]
    # ROS's grid center has a nonzero ~1e-16 y coordinate. Use production's
    # 1e-5 point tolerance once to identify it, then retain its exact FP32 key.
    matching = [u['point'] for u in feedback[-1]['updates']
                if np.max(abs(np.array(u['point'])-critical_point)) <= 1e-5]
    assert len(matching) == 1
    key = solver_point_key(matching[0])
    assert abs(margins[key] - .22) < 1e-12
    failed = [r for r in summaries if r.get('event') == 'scp_failed' and r.get('repair') == '1']
    assert len(failed) == 8
    certificates = []
    for summary in failed:
        stamp = summary['query_stamp_ns']
        assert float(stamp) * 1e-9 > feedback[-1]['ros_s']
        rows = [r for r in witness if r.get('query_stamp_ns') == stamp]
        critical = [r for r in rows if r.get('timestep') == '1'
                    and solver_point_key(json.loads(r['point'])) == key]
        assert len(critical) == 1
        row = critical[0]
        assert row['status'] == 'evaluated_unknown'
        assert row['mode_epoch'] == '2'  # Feedback retained across same-mode target refresh.
        query = queries[stamp]
        qs = np.array([[float(query[f'field.points{k}.positions{j}']) for j in range(7)] for k in range(21)])
        q0 = qs[0]
        assert np.max(abs(qs - q0)) < 1e-12
        assert np.max(abs(np.array(json.loads(row['q_requested'])) - qs[1])) < 1e-12
        g = np.array(json.loads(row['gradient']))
        qlin = np.array(json.loads(row['q_linearization']))
        d = float(row['distance'])
        at_start = d + g @ (q0 - qlin)
        growth = dt * (abs(g) @ velocity)
        upper = at_start + growth
        gap = margins[key] - upper
        assert gap > .01
        # q1=q0+dt*u0 eliminates the equality. No other rows/bounds remain.
        bounds = list(zip(-velocity, velocity))
        lp = linprog(np.zeros(7), A_ub=[-dt*g], b_ub=[at_start-margins[key]],
                     bounds=bounds, method='highs')
        assert lp.status == 2
        previous_guard = linprog(np.zeros(7), A_ub=[-dt*g], b_ub=[at_start-.12],
                                 bounds=bounds, method='highs')
        assert previous_guard.success
        certificates.append(dict(summary=summary, row=row, measured_q0=q0.tolist(),
            qlin_error_inf=float(np.max(abs(qlin-qs[1]))),
            guard=margins[key], linearized_at_start=float(at_start),
            maximum_first_step_growth=float(growth), maximum_first_step_value=float(upper),
            infeasibility_gap=float(gap),
            minimum_time_for_this_single_linear_row_s=float((margins[key]-at_start)/(abs(g)@velocity)),
            lp_status=int(lp.status), lp_message=lp.message,
            previous_guard_single_row_feasible=bool(previous_guard.success),
            witness_rows_available=len(rows), selected_rows=int(summary['cdf_rows'])))
    # Check that the critical UNKNOWN witness was never declared resolved_free
    # after either feedback event. Its same-mode point-scoped margin persists.
    freed = [r for r in witness if r['ros_s'] > feedback[0]['ros_s']
             and r.get('status') == 'resolved_free' and 'point' in r
             and solver_point_key(json.loads(r['point'])) == key]
    assert not freed
    before = [r for r in summaries if r.get('event') == 'scp_solved'
              and r.get('query_stamp_ns') in ['10640000000', '10648000000']]
    assert len(before) == 2
    assert all(sha(ROOT / p) == digest for p, digest in hashes.items())
    result = dict(run_id=RID, verdict='LOCAL_LINEAR_QP_PROVEN_INFEASIBLE',
        method='One hard CDF row plus first-step dynamics and original velocity bounds; all other constraints relaxed.',
        dt_s=dt, velocity_limits_rad_s=velocity.tolist(), feedback=feedback,
        preceding_solved_summaries=before, certificates=certificates,
        limitations=['No complete saved QP matrix; subset certificate does not need the objective or other rows.',
                     'The .12 comparison is a single-row offline diagnostic, not an executable trajectory or proposed safety relaxation.',
                     'No proof that all nonlinear safe routes are impossible; no full map/VBC replay.',
                     'No causal counterfactual attributing the eventual state to the initial tracker abort or replacement enable flag.'],
        original_manifest_files_checked=len(manifest), changed_original_files=changed,
        source_sha256=hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(verdict=result['verdict'], certified_failed_queries=len(certificates),
                         last=certificates[-1], original_files_unchanged=len(manifest)), indent=2))


if __name__ == '__main__':
    main()
