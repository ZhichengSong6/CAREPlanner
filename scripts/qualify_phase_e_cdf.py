#!/usr/bin/env python3
"""Compare collision CDF checkpoints on a frozen offline bundle; never deploy."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from cdf_qualification_common import (check_manifest, eligible_for_targeted_test, feasibility, in_box, matrix,
                                      new_output_dir, read, replace_cdf_rows, save, sha)


def query_id(point, q):
    return hashlib.sha256(json.dumps([point, q], separators=(',', ':')).encode()).hexdigest()


def verify_bundle(bundle_path):
    for relative, expected in read(bundle_path / 'artifact_sha256.json').items():
        if sha(bundle_path / relative) != expected:
            raise ValueError('Frozen bundle changed: ' + relative)


def evaluate(net, points, qs, batch_size=128):
    distances, gradients = [], []
    for start in range(0, len(points), batch_size):
        d, g = net.pair_distance_and_gradient(torch.tensor(points[start:start + batch_size], dtype=torch.float32),
                                            torch.tensor(qs[start:start + batch_size], dtype=torch.float32))
        distances.extend(d.cpu().tolist())
        gradients.extend(g.cpu().tolist())
    d, g = np.array(distances), np.array(gradients)
    if not np.isfinite(d).all() or not np.isfinite(g).all():
        raise ValueError('Model produced nonfinite values or gradients')
    return d, g


def derivative_check(net, points, qs, gradients, cfg):
    checks = []
    for scale in cfg['finite_difference_scales']:
        columns = []
        with torch.no_grad():
            p = torch.tensor(points, dtype=torch.float32)
            q = torch.tensor(qs, dtype=torch.float32)
            for j in range(7):
                plus, minus = q.clone(), q.clone()
                plus[:, j] += scale
                minus[:, j] -= scale
                value = (net.model(torch.cat((p, plus), dim=1)) - net.model(torch.cat((p, minus), dim=1))) / (2 * scale)
                columns.append(value.reshape(-1).cpu().numpy())
        fd = np.stack(columns, axis=1)
        if not np.isfinite(fd).all():
            raise ValueError('Finite-difference probes produced nonfinite values')
        error = np.abs(fd - gradients)
        allowed = cfg['derivative_atol'] + cfg['derivative_rtol'] * np.abs(gradients)
        checks.append(dict(scale=scale, max_absolute_error=float(error.max()),
                           failed_components=int(np.sum(error > allowed)), checked_components=int(error.size)))
    return checks


def compare_results(rows, qps, baseline):
    old = {r['id']: r for r in baseline['queries']}
    if set(old) != {r['id'] for r in rows}:
        raise ValueError('Baseline query identities do not match frozen corpus')
    clear_regressions, inside_positive, new_inside_positive = [], [], []
    group_changes = {}
    for row in rows:
        previous = old[row['id']]
        if previous['group'] != row['group'] or previous['point_m'] != row['point_m'] or previous['q_rad'] != row['q_rad']:
            raise ValueError('Baseline query payload mismatch')
        group = row['group']
        delta = group_changes.setdefault(group, dict(count=0, model_margin_pass_to_fail=0, model_margin_fail_to_pass=0))
        delta['count'] += 1
        was, now = previous['distance'] >= .05, row['distance'] >= .05
        delta['model_margin_pass_to_fail'] += int(was and not now)
        delta['model_margin_fail_to_pass'] += int(not was and now)
        if group == 'normal_clear' and was and not now:
            clear_regressions.append(row['id'])
        if group == 'inside_current_body' and row['distance'] > 0:
            inside_positive.append(row['id'])
            if previous['distance'] <= 0:
                new_inside_positive.append(row['id'])
    old_qps = {r['index']: r for r in baseline['qps']}
    if set(old_qps) != {r['index'] for r in qps}:
        raise ValueError('Baseline QP identities do not match')
    improved = sum(not old_qps[r['index']]['repeats'][0]['lp']['success'] and r['repeats'][0]['lp']['success'] for r in qps)
    worsened = sum(old_qps[r['index']]['repeats'][0]['lp']['success'] and not r['repeats'][0]['lp']['success'] for r in qps)
    return dict(group_changes=group_changes, new_normal_model_margin_blocks=clear_regressions,
                inside_current_body_positive_sign=inside_positive, newly_inside_positive_sign=new_inside_positive,
                qps_newly_feasible=int(improved), qps_newly_infeasible=int(worsened),
                limitation='Point signs use the current primitive geometry; model-margin changes are not metre-valued clearance errors')


def run(args, out):
    bundle_path = args.bundle.resolve()
    verify_bundle(bundle_path)
    cfg = read(bundle_path / 'bundle.json')
    manifest = read(args.manifest) if args.manifest else None
    checkpoint = args.checkpoint.resolve()
    contract = check_manifest(manifest, checkpoint, cfg, args.manifest.resolve().parent if args.manifest else out, args.activation)
    provenance = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), activation=args.activation,
                      checkpoint_key=args.checkpoint_key, bundle=str(bundle_path), bundle_sha256=sha(bundle_path / 'artifact_sha256.json'),
                      manifest_sha256=sha(args.manifest) if args.manifest else None,
                      tool_sha256=sha(Path(__file__)), helper_sha256=sha(Path(__file__).with_name('cdf_qualification_common.py')))
    save(out / 'contract.json', dict(contract=contract, provenance=provenance))
    if not args.diagnostic_only and not contract['compatible']:
        save(out / 'report.json', dict(status='BLOCKED_TRAINING_CONTRACT', contract=contract,
                                      provenance=provenance, ready_for_targeted_model_test=False))
        return 2
    runtime_source = bundle_path / 'runtime/collision_cdf_model.py'
    spec = importlib.util.spec_from_file_location('frozen_collision_cdf', runtime_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    net = module.CollisionCDF(str(checkpoint), device='cpu', checkpoint_key=args.checkpoint_key, activation=args.activation)
    if not net.architecture['nerf']:
        raise ValueError('Checkpoint encoding differs from the frozen raw x/sin(x)/cos(x) runtime contract')
    all_queries = read(bundle_path / 'queries.json')
    queries, rejected = [], []
    for row in all_queries:
        valid = in_box(row['point_m']) and np.asarray(row['q_rad']).shape == (7,) and np.isfinite(row['q_rad']).all()
        if valid != row['expected_input_valid']:
            raise ValueError('Unexpected invalid input: ' + row['id'])
        (queries if valid else rejected).append(row)
    save(out / 'rejected_inputs.json', rejected)
    points = np.array([r['point_m'] for r in queries])
    qs = np.array([r['q_rad'] for r in queries])
    predictions, times = [], []
    for repeat in range(cfg['repeats']):
        tic = time.perf_counter()
        prediction = evaluate(net, points, qs)
        times.append(1000 * (time.perf_counter() - tic))
        if predictions and not all(np.array_equal(a, b) for a, b in zip(predictions[0], prediction)):
            raise ValueError('Repeated fixed-input model inference differs')
        predictions.append(prediction)
    distances, gradients = predictions[0]
    rows = []
    historical_distance_error, historical_gradient_error = [], []
    for i, row in enumerate(queries):
        rows.append(dict(id=row['id'], group=row['group'], point_m=row['point_m'], q_rad=row['q_rad'],
                         distance=float(distances[i]), gradient=gradients[i].tolist(),
                         current_primitive_point_distance_m=row['current_primitive_point_distance_m']))
        if row['group'] == 'retained_fault':
            historical_distance_error.append(abs(distances[i] - row['old_model_distance']))
            historical_gradient_error.append(float(np.max(np.abs(gradients[i] - row['old_model_gradient']))))
    valid_index = {r['id']: i for i, r in enumerate(queries)}
    fd_rows = [valid_index[all_queries[i]['id']] for i in cfg['finite_difference_indices']]
    derivative = derivative_check(net, points[fd_rows], qs[fd_rows], gradients[fd_rows], cfg)
    if any(r['failed_components'] for r in derivative):
        raise ValueError('Autograd/finite-difference consistency failed: ' + str(derivative))
    lookup = {query_id(r['point_m'], r['q_rad']): r for r in rows if r['group'] == 'retained_fault'}
    qps, round_trip_errors = [], []
    native = bundle_path / 'runtime/replay_qp'
    for item in cfg['qps']:
        source = read(bundle_path / item['file'])['subproblem']
        # The recorded row transform must be an identity when fed its own old
        # values. This catches wrong knot, brake/control columns and offsets.
        identity = replace_cdf_rows(source, source['cdf_rows'])
        g_error = float(np.max(np.abs((matrix(identity['G']) - matrix(source['G'])).toarray())))
        h_error = float(np.max(np.abs(np.array(identity['h_l']) - source['h_l'])))
        if max(g_error, h_error) > 1e-8:
            raise ValueError('Recorded QP row round trip failed')
        round_trip_errors.append(dict(index=item['index'], matrix_error=g_error, lower_bound_error=h_error))
        values = [lookup[query_id(r['point'], r['q_linearization'])] for r in source['cdf_rows']]
        problem = replace_cdf_rows(source, values)
        path = out / 'qps' / f"{item['index']:02d}.json"
        save(path, problem)
        repeats = []
        for repeat in range(1, cfg['repeats'] + 1):
            proc = subprocess.run([str(native), str(path)], text=True, capture_output=True, timeout=30, check=True)
            nr = json.loads(proc.stdout)
            lp = feasibility(problem)
            # Preserve diagnostic solutions separately; none is executed.
            save(out / 'solutions' / f"{item['index']:02d}_r{repeat}.json", nr)
            repeats.append(dict(repeat=repeat, native={k: v for k, v in nr.items() if k != 'x'}, lp=lp))
        if len({(r['native']['status'], r['lp']['success']) for r in repeats}) != 1:
            raise ValueError('Repeated fixed-QP decisions differ')
        qps.append(dict(index=item['index'], source_sha256=item['sha256'], recorded_status=source['status'], repeats=repeats))
        print('QP', item['index'], repeats[0]['native']['status'], 'feasible', repeats[0]['lp']['success'], flush=True)
    baseline = read(args.baseline_report) if args.baseline_report else None
    comparison = None
    if baseline is not None:
        if baseline['provenance']['bundle_sha256'] != provenance['bundle_sha256']:
            raise ValueError('Baseline used a different bundle')
        if baseline['provenance']['checkpoint_sha256'] != cfg['baseline']['sha256']:
            raise ValueError('Comparison baseline is not the frozen old checkpoint')
        comparison = compare_results(rows, qps, baseline)
    is_old = provenance['checkpoint_sha256'] == cfg['baseline']['sha256']
    if is_old and (max(historical_distance_error) >= 1e-5 or max(historical_gradient_error) >= 2e-5):
        raise ValueError('Old checkpoint failed reproduction of historical values/gradients')
    ready = eligible_for_targeted_test(contract, comparison, args.diagnostic_only, is_old)
    groups = {}
    for group in cfg['groups']:
        subset = [r for r in rows if r['group'] == group]
        if subset:
            groups[group] = dict(count=len(subset), negative_sign=sum(r['distance'] < 0 for r in subset),
                                 at_or_above_model_margin=sum(r['distance'] >= .05 for r in subset))
    report = dict(status='DIAGNOSTIC_COMPLETE' if args.diagnostic_only else 'MODEL_COMPARISON_COMPLETE',
                  provenance=provenance, contract=contract, architecture=net.architecture,
                  selected_checkpoint=net.selected_checkpoint, diagnostic_only=args.diagnostic_only,
                  ready_for_targeted_model_test=ready, repeats=cfg['repeats'], input_queries=len(rows),
                  rejected_domain_probes=len(rejected), inference_wall_ms=times,
                  historical_distance_max_error=float(max(historical_distance_error)),
                  historical_gradient_max_error=float(max(historical_gradient_error)),
                  finite_difference_checks=derivative, qp_round_trip_errors=round_trip_errors,
                  groups=groups, queries=rows, qps=qps, comparison=comparison,
                  limitations=cfg['limitations'] + ['Ready means eligible for targeted closed-loop testing only; no deployment or physical safety certification',
                      'No automatic eligibility is granted to a different training geometry; its label/geometry equivalence needs review'])
    save(out / 'report.json', report)
    print(json.dumps(dict(status=report['status'], queries=len(rows), qps=len(qps),
                          feasible_qps=sum(r['repeats'][0]['lp']['success'] for r in qps),
                          contract=contract['status'], ready_for_targeted_model_test=ready)), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--activation', choices=('gelu', 'relu'), required=True)
    parser.add_argument('--checkpoint-key', default='latest')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--baseline-report', type=Path)
    parser.add_argument('--diagnostic-only', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not args.diagnostic_only and (args.manifest is None or args.baseline_report is None):
        parser.error('Model acceptance requires --manifest and --baseline-report; historical workflow checks use --diagnostic-only')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = new_output_dir(args.output_dir)
    try:
        return run(args, out)
    except Exception as exc:
        save(out / 'failure.json', dict(status='FAILED', error_type=type(exc).__name__, reason=str(exc)))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
