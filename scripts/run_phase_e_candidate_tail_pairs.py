#!/usr/bin/env python3
"""Predeclared balanced horizon trials; retains every failure, no retries.

Private ROS/CUDA fixture only. No robot, planner, tracker or case modification.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from compare_phase_e_unified_primitives import compare
from phase_e_acceptance_scope import trajectory_workload
from phase_e_release_policy import POLICY, compare_release

ROOT = Path(__file__).resolve().parents[1]


def quantile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values)*fraction)-1)]


def aggregate(reports):
    result = copy.deepcopy(reports[0])
    result['latency_rows_ms'] = [row for r in reports for row in r['latency_rows_ms']]
    result['node_timing_rows_ms'] = [row for r in reports for row in r['node_timing_rows_ms']]
    result['status'] = 'PASS' if all(r['status'] == 'PASS' for r in reports) else 'FAIL'
    for r in reports:
        assert r['sha256'] == result['sha256'], 'inputs changed between rounds'
    result['latency_ms'] = {key: {label: quantile([r[key] for r in result['latency_rows_ms']], q)
        for label, q in [('p50', .5), ('p95', .95)]} for key in result['latency_ms']}
    result['candidate_segments_ms'] = {key: {label: quantile([r['candidate'][key] for r in result['node_timing_rows_ms']], q)
        for label, q in [('p50', .5), ('p95', .95)]} for key in result['node_timing_rows_ms'][0]['candidate']}
    if 'shadow' in result['node_timing_rows_ms'][0]:
        result['shadow_segments_ms'] = {key: {label: quantile([r['shadow'][key] for r in result['node_timing_rows_ms']], q)
            for label, q in [('p50', .5), ('p95', .95)]} for key in result['node_timing_rows_ms'][0]['shadow']}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--repetitions', type=int, default=80)
    ap.add_argument('--knots', type=int, choices=[2, 20, 21, 50], default=2,
                    help='21 = exact mainline horizon; 50 = stress, never a mainline gate')
    ap.add_argument('--persistent-confidence', action='store_true')
    ap.add_argument('--shadow-event-driven', action='store_true')
    ap.add_argument('--fixture-tcp-nodelay', action='store_true')
    ap.add_argument('--verification-tcp-nodelay', action='store_true')
    ap.add_argument('--install-prefix', type=Path)
    ap.add_argument('--input-period-s', type=float, default=0.)
    ap.add_argument('--release-policy', choices=['legacy', 'mainline21-v1'], default='legacy')
    args = ap.parse_args()
    if args.repetitions < 20:
        ap.error('at least 20 repetitions required')
    if args.release_policy != 'legacy' and args.knots != 21:
        ap.error('mainline release policy applies only to 21 states; 50 is stress only')
    if not math.isfinite(args.input_period_s) or not 0 <= args.input_period_s <= 1:
        ap.error('input period must be finite and within [0, 1] seconds')
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    # Declared before any result. Balanced backend order, four independent pairs.
    orders = [('samples', 'primitive'), ('primitive', 'samples'), ('primitive', 'samples'), ('samples', 'primitive')]
    plan = dict(orders=orders, repetitions=args.repetitions, knots=args.knots, warmup=2,
                workload=trajectory_workload(args.knots),
                tolerance=0, execution_event_driven=True, persistent_confidence=args.persistent_confidence, retry=False,
                shadow_event_driven=args.shadow_event_driven, input_period_s=args.input_period_s,
                fixture_tcp_nodelay=args.fixture_tcp_nodelay, install_prefix=str(args.install_prefix) if args.install_prefix else None,
                verification_tcp_nodelay=args.verification_tcp_nodelay,
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    plan['release_policy'] = dict(POLICY) if args.release_policy != 'legacy' else None
    policy_hash = hashlib.sha256((ROOT/'scripts/phase_e_release_policy.py').read_bytes()).hexdigest()
    plan['release_policy_sha256'] = policy_hash
    (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    reports = {'samples': [], 'primitive': []}; rounds = []; release_rounds = []
    for index, order in enumerate(orders, 1):
        pair = {}
        for backend in order:
            label = f'round{index}-{backend}'
            cmd = [sys.executable, str(ROOT/'scripts/test_phase_e_unified_primitives.py'),
                   '--output-dir', str(out/label), '--backend', backend,
                   '--knots', str(args.knots), '--execution-event-driven', '--repetitions', str(args.repetitions),
                   '--input-period-s', str(args.input_period_s)]
            if args.persistent_confidence:
                cmd.append('--persistent-confidence')
            if args.shadow_event_driven:
                cmd.append('--shadow-event-driven')
            if args.fixture_tcp_nodelay:
                cmd.append('--fixture-tcp-nodelay')
            if args.verification_tcp_nodelay:
                cmd.append('--verification-tcp-nodelay')
            if args.install_prefix:
                cmd += ['--install-prefix', str(args.install_prefix.resolve())]
            print('START '+label, flush=True)
            with (out/(label+'.log')).open('x') as handle:
                process = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT)
            path = out/label/'result.json'
            r = json.loads(path.read_text()) if path.exists() else dict(status='FAIL', error='missing result')
            if process.returncode != 0:
                r['status'] = 'FAIL'; r['exit_code'] = process.returncode
            pair[backend] = r; reports[backend].append(r)
            print(label+' '+r['status'], flush=True)
        try:
            verdict = compare(pair['samples'], pair['primitive'])
        except (AssertionError, KeyError) as exc:
            verdict = dict(status='FAIL', error=str(exc))
        rounds.append(verdict)
        if args.release_policy != 'legacy':
            try:
                release_rounds.append(compare_release(pair['samples'], pair['primitive']))
            except (AssertionError, KeyError) as exc:
                release_rounds.append(dict(status='FAIL', error=str(exc)))
        (out/f'comparison{index}.json').write_text(json.dumps(verdict, indent=2)+'\n')
        print('PAIR '+str(index)+' '+verdict['status'], flush=True)
    result = dict(plan=plan, rounds=rounds, status='FAIL', scope='Four fixed independent pairs; no statistical equivalence claim')
    result['workload'] = trajectory_workload(args.knots)
    result['functional_status'] = 'PASS' if all(r['status'] == 'PASS' for runs in reports.values() for r in runs) else 'FAIL'
    # Kept as a historical diagnostic, not the newly relaxed release policy.
    result['performance_policy'] = 'legacy_zero_tolerance_diagnostic'
    result['release_verdict'] = 'NOT_EVALUATED' if args.knots == 21 else 'NOT_APPLICABLE'
    try:
        assert all(r['status'] == 'PASS' for runs in reports.values() for r in runs), 'incomplete functional runs; pooled comparison NOT_RUN'
        combined = {key: aggregate(value) for key, value in reports.items()}
        result['pooled'] = compare(combined['samples'], combined['primitive'])
        result['candidate_segments'] = {key: r['candidate_segments_ms'] for key, r in combined.items()}
        result['shadow_segments'] = {key: r['shadow_segments_ms'] for key, r in combined.items()}
        result['status'] = 'PASS' if result['pooled']['status'] == 'PASS' and all(r['status'] == 'PASS' for r in rounds) else 'FAIL'
    except (AssertionError, KeyError) as exc:
        result['error'] = str(exc)
    if args.release_policy != 'legacy':
        result['legacy_diagnostic_status'] = result['status']
        result['performance_policy'] = POLICY['name']
        result['release_rounds'] = release_rounds
        result['release_verdict'] = 'FAIL'
        try:
            assert policy_hash == hashlib.sha256((ROOT/'scripts/phase_e_release_policy.py').read_bytes()).hexdigest(), 'policy changed during run'
            assert result['functional_status'] == 'PASS', 'functional failure'
            assert 'combined' in locals(), 'pooled aggregation failed'
            result['release_pooled'] = compare_release(combined['samples'], combined['primitive'])
            result['release_verdict'] = 'PASS' if result['release_pooled']['status'] == 'PASS' and all(r['status'] == 'PASS' for r in release_rounds) else 'FAIL'
        except (AssertionError, KeyError) as exc:
            result['release_error'] = str(exc)
        result['status'] = result['release_verdict']
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
