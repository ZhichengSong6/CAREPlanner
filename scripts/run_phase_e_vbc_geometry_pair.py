#!/usr/bin/env python3
"""Four canonical simulations: fixed H9 V1 x samples/primitive x 007/026.

Only orchestration and offline analysis. Preserve raw cases, runtime and logs.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

from run_phase_e_repeatability import (REPO, PYTHON, CHECKPOINTS, fingerprint,
    assert_frozen, assert_no_ros, capture, save)
from run_phase_e_repair_failure_regression import phase_audit
from summarize_phase_e_qp_events import collect
from analyze_phase_e_observation_events import string_rows


def metadata(path):
    return dict(line.split('=', 1) for line in path.read_text().splitlines() if '=' in line)


def validate_geometry_report(path, frozen):
    """Bind an explicitly selected passed validation, never bypass binary hashes."""
    raw = path.read_bytes()
    report = json.loads(raw)
    required = {'src/arm_description/urdf/Arm.urdf',
        'src/arm_description/urdf/Arm_with_self_filter_collision.urdf',
        'src/care_confidence_map/config/body_samples.yaml',
        'devel/lib/care_confidence_map/trajectory_vbc_selector_temporal_cluster_node',
        'devel/lib/libtrajectory_risk_evaluator.so'}
    assert report['status'] == 'PASS', 'geometry validation did not pass'
    assert required <= set(report['sha256']), 'incomplete geometry validation identity'
    assert all(frozen.get(p) == sha for p, sha in report['sha256'].items()), 'tested geometry input changed'
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                validated_artifacts=report['sha256'])


def validate_planner_report(path, frozen):
    raw = path.read_bytes()
    report = json.loads(raw)
    binary = 'devel/lib/egocentric_arm_planner/local_sparse_scp_planner_node'
    assert report['status'] == 'PASS', 'planner validation did not pass'
    assert frozen[binary] == report['binary_sha256'], 'tested planner binary changed'
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                binary_sha256=report['binary_sha256'])


def distribution(values):
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return dict(count=0, p50=None, p95=None, maximum=None)
    return dict(count=len(values), p50=values[int(.5*(len(values)-1))],
                p95=values[int(.95*(len(values)-1))], maximum=values[-1])


def geometry_audit(case, backend):
    run, window = Path(case['run']), case['benchmark_window']
    result = {}
    for role in ('candidate', 'execution'):
        rows = string_rows(run/(role+'_vbc_summary.csv'), window)
        backends = Counter(row.get('geometry_backend', 'MISSING') for row in rows)
        assert not set(backends) - {backend}, (role, backends)
        points, links, sources = Counter(), Counter(), Counter()
        negative_deadlines = 0
        for row in rows:
            evidence = json.loads(row['vbc_evidence'])
            assert evidence['geometry_backend'] == backend
            assert evidence['trajectory_stamp_ns'] == row['trajectory_stamp_ns']
            assert float(row['swept_volume_margin_m']) == 0
            for v in evidence['violations']:
                if backend == 'primitive':
                    assert v['sample_index'] == -1 and v['raw_radius_m'] is None
                    assert v['source_collision_name'] and v['primitive_signed_distance_m'] <= 1e-12
                points[tuple(v['point'])] += 1
                links[v['sweeping_link']] += 1
                sources[v.get('source_collision_name') or
                        (v['sweeping_link']+':'+str(v['source_collision_index']))] += 1
                negative_deadlines += v['latest_see_s'] < 0
        result[role] = dict(status='PASS' if rows else 'NOT_RUN', records=len(rows),
            backend_counts=dict(backends), violation_audits=sum(r.get('has_violation') == '1' for r in rows),
            unique_violation_points=len(points), links=dict(links), collisions=dict(sources),
            top_points=[dict(point=list(p), violation_records=n) for p, n in points.most_common(12)],
            negative_deadline_records=negative_deadlines,
            timing_ms={key: distribution([float(r[key]) for r in rows if key in r]) for key in
                       ('body_fk_ms', 'swept_voxel_build_ms', 'total_eval_ms')},
            sweep_time_s=distribution([float(r['sweep_time_s']) for r in rows if 'sweep_time_s' in r]))
    assert result['candidate']['records'], 'no actual candidate VBC evidence'
    outcomes = string_rows(run/'verification_outcome.csv', window)
    result['verification'] = dict(Counter(
        r.get('result', 'unknown')+':'+r.get('safety_gate', 'unknown')
        for r in outcomes))
    result['runtime_semantics'] = metadata(run/'runtime_semantics.txt')
    assert result['runtime_semantics']['vbc_geometry_backend'] == backend
    assert float(result['runtime_semantics']['vbc_swept_volume_margin_m']) == 0
    return result


def summarize(out, stamp):
    rows = []
    for backend in ('samples', 'primitive'):
        rows += json.loads((out/(backend+'_summary.json')).read_text())['cases']
    pairs = []
    for case_id in ('phase_e_goal_007', 'phase_e_goal_026'):
        a, b = [next(r for r in rows if r['case_id'] == case_id and r['geometry_backend'] == backend)
                for backend in ('samples', 'primitive')]
        am, bm = [r['geometry_audit']['runtime_semantics'] for r in (a, b)]
        differing = {k: [am.get(k), bm.get(k)] for k in am.keys() | bm.keys() if am.get(k) != bm.get(k)}
        assert set(differing) <= {'run_id', 'vbc_geometry_backend'}, differing
        pairs.append(dict(case_id=case_id, runtime_metadata_differences=differing,
                          task_success={r['geometry_backend']: r['task_success'] for r in (a, b)},
                          overall_safe={r['geometry_backend']: r['overall_safe'] for r in (a, b)}))
    report = dict(status='PASS', completed_runs=len(rows), cases=rows, pairs=pairs,
                  frozen_inputs_equal=True,
                  note='One run per case/backend. Task failure is retained, not a harness failure; '
                       'asynchronous simulation timings are descriptive, not causal paired trajectory timings.')
    save(out/'summary.json', report)
    print('COMPLETE', [(r['geometry_backend'], r['case_id'], r['task_success'], r['overall_safe']) for r in rows], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stamp', default=datetime.now().strftime('%Y%m%d-%H%M%S')+'-vbc-pair')
    ap.add_argument('--analyze-only', action='store_true')
    ap.add_argument('--state-test-report', type=Path,
                    default=REPO/'outputs/phase_e_witness_interface/planner_final/result.json')
    ap.add_argument('--enable-bounded-recovery', action='store_true')
    ap.add_argument('--geometry-validation-report', type=Path,
                    default=REPO/'outputs/phase_e_vbc_primitives/stage1-final/result.json',
                    help='Explicit passed geometry validation for the current binary; hashes remain mandatory')
    args = ap.parse_args()
    if not re.fullmatch('[A-Za-z0-9_-]+', args.stamp):
        ap.error('invalid stamp')
    os.chdir(REPO)
    out = REPO/'outputs/phase_e_vbc_geometry_pair'/args.stamp
    if args.analyze_only:
        summarize(out, args.stamp)
        return
    assert_no_ros()
    rviz = subprocess.run(['ps', '-C', 'rviz', '-o', 'pid='], capture_output=True, text=True)
    assert not rviz.stdout.strip(), 'existing RViz session: refuse canonical cleanup'
    short = capture(['git', 'rev-parse', '--short=8', 'HEAD'])
    for backend in ('samples', 'primitive'):
        stamp = args.stamp+'-'+backend
        batch = 'phase_e_'+stamp
        paths = [REPO/'outputs/phase_e_empty_world_qualification'/batch,
                 REPO/('CAREPlanner_PHASE_E_EMPTY_QUALIFICATION_'+batch+'.zip'),
                 REPO/'outputs/phase_e_qp_events_inputs'/(stamp+'.json'),
                 REPO/'outputs/phase_e_qp_events_inputs'/(stamp+'.source_sha256.json')]
        for case_id in ('007', '026'):
            run = 'phase_e_goal_'+case_id+'_empty_qualification_'+stamp+'_'+short
            paths += [REPO/'outputs/c5_5_vbc_gcdf_regime'/run,
                      REPO/'logs/c5_5_vbc_gcdf_regime'/run,
                      REPO/('CAREPlanner_C5_RESULT_'+run+'.zip')]
        assert not any(p.exists() for p in paths), 'run identity already exists'
    out.mkdir(parents=True, exist_ok=False)
    frozen = fingerprint()
    save(out/'frozen_sha256.json', frozen)
    assert all(frozen[p] == sha for p, sha in CHECKPOINTS.items()), 'fixed checkpoint mismatch'
    planner_binding = validate_planner_report(args.state_test_report, frozen)
    save(out/'planner_validation_binding.json', planner_binding)
    geometry_binding = validate_geometry_report(args.geometry_validation_report, frozen)
    save(out/'geometry_validation_binding.json', geometry_binding)
    cuda = capture([PYTHON, '-c', 'import json,torch; assert torch.cuda.is_available(); '
        'q=torch.ones(7,device="cuda",requires_grad=True); g=torch.autograd.grad((q*q).sum(),q)[0]; '
        'torch.cuda.synchronize(); print(json.dumps(dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),gradient=g.tolist())))'])
    save(out/'preflight.json', dict(cuda=json.loads(cuda), gpu=capture(['nvidia-smi']),
        git_head=capture(['git', 'rev-parse', 'HEAD']), dirty_status=capture(['git', 'status', '--short']),
        cases=['007', '026'], model='fixed H9 V1 + old scalar', planned_runs=4,
        watchdog_s=45, vbc_margin_m=0, order=['samples 007/026', 'primitive 007/026']))
    print('PREFLIGHT PASS', out, cuda, flush=True)
    original = json.loads((REPO/'outputs/phase_e_goal_sampling/phase_e_obstacle_goal_pool_30.json').read_text())
    for backend in ('samples', 'primitive'):
        assert_frozen(frozen)
        assert_no_ros()
        stamp = args.stamp+'-'+backend
        overrides = dict(BATCH_STAMP=stamp, BATCH_ID='phase_e_'+stamp,
            DIAGNOSTIC_CASES='007,026', PER_SENSOR_HYBRID_ENABLED='true',
            FINAL_VBC_RECOVERY_ENABLED='true' if args.enable_bounded_recovery else 'false',
            VBC_GEOMETRY_BACKEND=backend, VBC_SWEPT_VOLUME_MARGIN_M='0.0', KEEP_CASE_ZIPS='1')
        env = dict(os.environ, **overrides)
        save(out/(backend+'_environment.json'), overrides)
        print('START', backend, '007/026', flush=True)
        with (out/(backend+'.log')).open('x') as stream:
            proc = subprocess.run(['bash', 'scripts/run_phase_e_qp_events_regression.sh'],
                                  env=env, stdout=stream, stderr=subprocess.STDOUT)
        save(out/(backend+'_exit.json'), dict(return_code=proc.returncode))
        assert_frozen(frozen)
        assert_no_ros()
        assert proc.returncode == 0, 'runner failed; retain outputs, no automatic simulation retry'
        subset = json.loads((REPO/'outputs/phase_e_qp_events_inputs'/(stamp+'.json')).read_text())
        assert subset['cases'] == [next(c for c in original['cases'] if c['case_id'] == 'phase_e_goal_'+i)
                                   for i in ('007', '026')], 'case object changed'
        batch = REPO/'outputs/phase_e_empty_world_qualification'/('phase_e_'+stamp)
        meta = metadata(batch/'qualification_metadata.txt')
        assert meta['per_sensor_hybrid_enabled'] == 'true'
        assert meta['per_sensor_checkpoint_sha256'] == CHECKPOINTS['src/care_visibility_cdf/checkpoints/hierarchical9_scratch_seed0/final.pt']
        group = collect(REPO, batch, stamp, 'h9_'+backend)
        for row in group:
            row['geometry_backend'] = backend
            row['repair_state'] = phase_audit(row)
            row['geometry_audit'] = geometry_audit(row, backend)
        save(out/(backend+'_summary.json'), dict(cases=group, qualification_metadata=meta))
        print('END', backend, [(r['case_id'], r['task_success'], r['overall_safe']) for r in group], flush=True)
        assert all(r['overall_safe'] and r['startup_order_pass'] and
                   r['repair_state']['phase_fields_present'] and
                   not r['repair_state']['retreat_candidate_gate_violations'] for r in group), 'safety/startup/trace invariant failed'
    assert_frozen(frozen)
    assert_no_ros()
    summarize(out, args.stamp)


if __name__ == '__main__':
    main()
