#!/usr/bin/env python3
"""Query-bound exp74 candidate diagnostic. No ROS imports, publishers or simulation.

Exhaustive head enumeration is deliberately separate from production generate().
Saved witness linearizations and point/body distances are diagnostic proxies,
not execution certificates. Output directories must be new.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VIS = ROOT / 'src/care_visibility_cdf/scripts'
sys.path.insert(0, str(VIS))
from analyze_phase_e_observation_events import string_rows, solver_point_key
from per_sensor_visibility_runtime import PerSensorVisibilityRuntime
from evaluate_direct_vs_projection_ascent import (
    DEFAULT_JOINT_NAMES, build_model_from_checkpoint, torch_load_checkpoint,
    learned_projection_step, learned_ascent_step, model_value_and_grad_q,
)
from diagnose_phase_e_case001_exact_witness import Geometry

RID = 'case_001_dependency_lock_fix_20260921_exp74'
RUN = ROOT / 'outputs/c5_5_vbc_gcdf_regime' / RID / 'run'
WEIGHTS = ROOT / 'src/care_visibility_cdf/checkpoints'


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def save(path, value):
    with path.open('x') as f:
        json.dump(clean(value), f, indent=2, allow_nan=False)
        f.write('\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strings(name):
    return string_rows(RUN / (name + '.csv'),
                       dict(start_ros_s=0, end_ros_s=66.629))


def numeric(name):
    with (RUN / (name + '.csv')).open() as f:
        return [r for r in csv.DictReader(f) if r.get('%time', '').isdigit()]


def vector(row, prefix):
    return np.array([float(row[prefix + str(j)]) for j in range(7)])


def production_scalar(config):
    """Extract only pure production methods; do not import ROS node modules."""
    selections = {
        'vbc_deadline_waypoint_node.py': {
            '_clamp', '_learned', '_refine_root_bisection'},
        'vbc_deadline_waypoint_rolling_impl.py': {
            '_per_point_values', '_oracle_diag_set', '_generate_active_set_waypoint'},
    }
    methods = []
    defaults = {}
    for name, wanted in selections.items():
        tree = ast.parse((VIS / name).read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {'DEFAULT_Q_MIN', 'DEFAULT_Q_MAX'}:
                        defaults[target.id] = ast.literal_eval(node.value)
            if isinstance(node, ast.ClassDef):
                methods.extend(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in wanted)
    assert len(methods) == 6
    tree = ast.Module(body=[ast.ClassDef(name='ScalarReplay', bases=[], keywords=[],
                                      body=methods, decorator_list=[])], type_ignores=[])
    ns = dict(np=np, torch=torch, math=math, _finite=lambda x: np.isfinite(x).all(),
              model_value_and_grad_q=model_value_and_grad_q,
              learned_projection_step=learned_projection_step,
              learned_ascent_step=learned_ascent_step)
    exec(compile(ast.fix_missing_locations(tree), '<production scalar methods>', 'exec'), ns)
    obj = ns['ScalarReplay']()
    obj.device = torch.device('cpu')
    obj.model, _ = build_model_from_checkpoint(torch_load_checkpoint(
        str(WEIGHTS / 'exp1_yiming_k500_fov_signed/final.pt'), obj.device), obj.device)
    for p in obj.model.parameters():
        p.requires_grad_(False)
    for k, v in config.items():
        setattr(obj, k, v)
    obj.q_min = torch.tensor(defaults['DEFAULT_Q_MIN'])
    obj.q_max = torch.tensor(defaults['DEFAULT_Q_MAX'])
    obj.clamp_q = True
    obj.math_eps = 1e-8
    obj.safety_margin_s = .3
    obj.conservative_oracle = obj.nominal_oracle = None
    obj._maybe_apply_per_sensor_hybrid = lambda points, result: result
    return obj


def scalar_solve(obj, points, q):
    obj._sample_trajectory = lambda trajectory, t: np.asarray(q).copy()
    trajectory = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda: 1.)))
    # The dummy epoch is unused by projection; never represents an actual event.
    result = obj._generate_active_set_waypoint(points, trajectory, .1, None)
    for k in ['deadline_absolute_ros_s', 'deadline_epoch_source', 'nominal_sweep_time_s',
              'deadline_from_start_s', 'safety_margin_s']:
        result.pop(k)
    return result


def snapshots(traces):
    deps = []
    for line in (RUN / 'observation_dependency.csv').read_text().splitlines()[1:]:
        stamp, sep, payload = line.partition(',')
        if sep and stamp.isdigit():
            deps.append(dict(json.loads(payload), receipt_ns=int(stamp)))
    forward = [d for d in deps if d['observation_token'] == traces[1]['observation_token']]
    backward = [d for d in deps if d['observation_token'] == traces[6]['observation_token']]
    for events, owner in [(forward, 6), (backward, 1)]:
        keys = {solver_point_key(p) for p in traces[owner]['active_set_points_xyz']}
        assert all(solver_point_key(p) in keys for d in events for p in d['points'])
    query_rows = numeric('local_gcdf_query')
    joint_rows = numeric('joint_states')
    joint_q = np.array([vector(r, 'field.position') for r in joint_rows])
    joint_t = np.array([int(r['field.header.stamp']) for r in joint_rows])
    planners = strings('local_planner_summary')
    selectors = strings('local_witness_selector_diagnostics')
    checks = strings('local_witness_diagnostics')
    hold = next(r for r in planners if r.get('safe_stall_dependency_wait') == '1')
    out = []
    for label, d in [('parent_requests_child', forward[0]),
                     ('child_requests_parent', backward[0]),
                     ('last_dependency_before_budget_hold', backward[-1])]:
        stamp = str(d['query_stamp_ns'])
        qs = [r for r in query_rows if r.get('field.header.stamp') == stamp]
        assert len(qs) == 1, (stamp, len(qs))
        qr = qs[0]
        q = vector(qr, 'field.points0.positions')
        # Exact q matching against preceding measured samples; no interpolation.
        errors = np.max(np.abs(joint_q - q), axis=1)
        matching = np.flatnonzero((errors <= 1e-12) & (joint_t <= int(stamp)))
        assert len(matching), 'No measured sample for query q0: ' + stamp
        ji = matching[-1]
        p = [r for r in planners if r.get('query_stamp_ns') == stamp
             and r.get('plan_seq') == str(d['plan_seq'])
             and r.get('observation_token') == d['observation_token']]
        assert p
        rows = [r for r in selectors if r.get('query_stamp_ns') == stamp
                and r.get('plan_seq') == str(d['plan_seq'])
                and r.get('mode_epoch') == str(d['mode_epoch'])]
        assert rows and len({r['map_stamp_ns'] for r in rows}) == 1
        # One complete indexed response, never concatenate different SCP queries.
        assert sorted(int(r['witness_index']) for r in rows) == list(range(len(rows)))
        assert all(r['status'] == 'evaluated_unknown' for r in rows)
        for row in rows:
            k = int(row['timestep'])
            assert np.max(np.abs(np.array(json.loads(row['q_requested'])) -
                                 vector(qr, f'field.points{k}.positions'))) < 1e-12
            assert np.max(np.abs(np.array(json.loads(row['q_linearization'])) -
                                 np.array(json.loads(row['q_requested']), dtype=np.float32))) < 1e-12
        fresh = [r for r in checks if r.get('query_stamp_ns') == stamp
                 and r.get('plan_seq') == str(d['plan_seq']) and r.get('reason') == 'fresh']
        keys = {solver_point_key(json.loads(r['point'])) for r in rows}
        assert all(solver_point_key(x) in keys for x in d['points'])
        points = {solver_point_key(json.loads(r['point'])): json.loads(r['point']) for r in rows}
        out.append(dict(label=label, dependency=d, query_stamp_ns=int(stamp),
            measured_q=q.tolist(), measured_stamp_ns=int(joint_t[ji]),
            measured_age_at_query_ms=(int(stamp)-int(joint_t[ji]))/1e6,
            measured_query_q_error_inf=float(errors[ji]),
            query_receipt_ns=int(qr['%time']), joint_names=DEFAULT_JOINT_NAMES,
            planner_rows=p, witness_rows=rows, fresh_checks=fresh,
            unique_unknown_points=list(points.values()),
            map_stamp_ns=int(rows[0]['map_stamp_ns']),
            witness_receipt_interval_s=[min(r['ros_s'] for r in rows), max(r['ros_s'] for r in rows)],
            guard_scope='query-bound UNKNOWN witness rows; effective per-row margins and complete map absent',
            temporal_scope='query q0 from exact measured sample; response rows may arrive after dependency notification'))
    return out, dict(forward_requests=len(forward), backward_requests=len(backward),
                     point_ownership='exact canonical FP32', budget_hold_event=hold,
                     hold_is_distinct_from_last_query=True)


def union_geometry(runtime, points, q):
    all_heads = [runtime._candidate_geometry(points, q, s) for s in range(8)]
    point_visible = []
    for i in range(len(points)):
        point_visible.append(any(h['per_point'][i]['conservative_g'] >= 0 and
                                 not h['per_point'][i]['primitive_self_occluded'] for h in all_heads))
    return dict(heads=all_heads, all_points_visible_any_sensor=all(point_visible),
                visible_point_fraction=float(np.mean(point_visible)),
                best_single_sensor_g=max(h['min_conservative_g'] for h in all_heads))


def directional_proxies(geom, snapshot, q, old_q):
    q0 = np.array(snapshot['measured_q'])
    delta = np.array(q) - q0
    old_delta = np.array(old_q) - q0
    norm = float(np.linalg.norm(delta))
    denom = norm * float(np.linalg.norm(old_delta))
    # Fixed sampling resolution is diagnostic only, not a planner/safety setting.
    alphas = np.linspace(0, 1, 33)
    path = q0[None, :] + alphas[:, None] * delta
    points = snapshot['unique_unknown_points']
    distances = np.array([geom.minimum(qi, points)[0] for qi in path])
    start = distances[0]
    rows = snapshot['witness_rows']
    grads = np.array([json.loads(r['gradient']) for r in rows])
    qlin = np.array([json.loads(r['q_linearization']) for r in rows])
    ds = np.array([float(r['distance']) for r in rows])
    slopes = grads @ delta
    # Local .01-rad step is a probe, never a suggested executable motion.
    step = delta * min(1., .01 / max(float(np.max(np.abs(delta))), 1e-12))
    step_dist = geom.minimum(q0 + step, points)[0]
    witness_start = ds + np.sum(grads * (q0 - qlin), axis=1)
    w = int(np.argmin(slopes))
    return dict(q_delta=delta, q_distance_l2=norm,
        q_distance_inf=float(np.max(np.abs(delta))), cosine_to_recorded_target=(float(delta @ old_delta)/denom if denom else None),
        q_difference_from_recorded_target_inf=float(np.max(np.abs(np.array(q)-old_q))),
        local_probe_q=q0+step, local_probe_inf_rad=float(np.max(np.abs(step))),
        witness_model_direction=dict(min_slope=float(slopes.min()),
            decreasing_rows=int(np.sum(slopes < -1e-9)), row_count=len(rows),
            worst_row=dict(point=json.loads(rows[w]['point']), timestep=int(rows[w]['timestep']),
                           q_linearization=qlin[w], gradient=grads[w]),
            extrapolated_min_start=float(witness_start.min()),
            extrapolated_min_endpoint=float((witness_start+slopes).min()),
            qualification='directional linearization only; no margin/pass classification'),
        point_body_geometry=dict(start_min_m=float(start.min()),
            local_probe_min_m=float(step_dist.min()), endpoint_min_m=float(distances[-1].min()),
            sampled_line_min_m=float(distances.min()), samples=len(alphas),
            alphas=alphas, min_distance_by_alpha=distances.min(axis=1),
            start_point_distances_m=start, endpoint_point_distances_m=distances[-1],
            qualification='signed distance to recorded point centers, 21 active body primitives; no inflation, voxel extent or full map'),
        certified_safe_prefix='NOT_RUN', executed_observation_progress='NOT_RUN')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    trace_paths = {i: next((RUN/'projector_traces').glob(f'c46_obligation_{i:03d}_*.json')) for i in [1, 6]}
    traces = {i: json.loads(p.read_text()) for i, p in trace_paths.items()}
    assert traces[1]['config'] == traces[6]['config']
    config = traces[1]['config']
    protected = list(WEIGHTS.rglob('*.pt')) + list((ROOT/'src/care_collision_cdf/checkpoints').rglob('*.pt'))
    protected += list(VIS.glob('*.py')) + [ROOT/'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml']
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    sources = list(trace_paths.values()) + [RUN/(n+'.csv') for n in
        ['joint_states', 'local_gcdf_query', 'observation_dependency', 'local_planner_summary',
         'local_witness_selector_diagnostics', 'local_witness_diagnostics']]
    sources += [RUN/'runtime_semantics.txt', ROOT/'src/arm_description/urdf/Arm.urdf',
                ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf', Path(__file__)]
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    snaps, lineage = snapshots(traces)
    save(args.output_dir/'snapshots.json', dict(run_id=RID, snapshots=snaps, lineage=lineage,
                                              traces=traces, source_sha256=source_hashes))
    print('Snapshots recovered:', [(s['label'], s['measured_age_at_query_ms'], len(s['witness_rows'])) for s in snaps], flush=True)
    scalar = production_scalar(config)
    reproduction = {}
    for i, trace in traces.items():
        result = scalar_solve(scalar, trace['active_set_points_xyz'], trace['c4_6_measured_seed_q'])
        err = {k: float(np.max(np.abs(np.array(result[k])-trace[k]))) for k in ['q_zero', 'q_vis']}
        # Historical run may use CUDA; tolerate small FP32 arithmetic drift.
        assert max(err.values()) < 2e-5, err
        reproduction[i] = dict(max_abs_error=err, tolerance_rad=2e-5, result=result)
    geom = Geometry()
    runtimes = {}
    for label, folder in [('R1', 'hierarchical9_r1_scratch50k'), ('V1', 'hierarchical9_scratch_seed0')]:
        runtimes[label] = PerSensorVisibilityRuntime(
            checkpoint_path=str(WEIGHTS/folder/'final.pt'),
            reference_urdf_path=str(ROOT/'src/arm_description/urdf/Arm.urdf'),
            self_filter_urdf_path=str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
            device=torch.device('cpu'), q_min=scalar.q_min.tolist(), q_max=scalar.q_max.tolist(),
            **{k: config[k] for k in ['projection_iters', 'projection_damping', 'projection_epsilon_f',
                                     'projection_max_step_norm', 'root_refine_iters', 'root_tolerance_f']},
            branch_ascent_steps=config['ascent_steps'], branch_step_size=config['ascent_step_size'],
            branch_max_step_norm=config['ascent_max_step_norm'],
            branch_fallback_ascent_steps=config['shared_fallback_ascent_steps'])
    records = []
    groups = []
    for snap in snaps:
        for obligation in [1, 6]:
            trace = traces[obligation]
            points = np.array(trace['active_set_points_xyz'])
            q0 = np.array(snap['measured_q'])
            scalar_result = scalar_solve(scalar, points, q0)
            baseline = union_geometry(runtimes['R1'], points, q0)
            group = dict(snapshot=snap['label'], obligation=obligation,
                obligation_token=trace['observation_token'], points=points,
                target_scope='frozen generation point set; counterfactual when not the active obligation',
                scalar_recomputed=scalar_result, geometry_at_measured=baseline,
                recorded_target_geometry=union_geometry(runtimes['R1'], points, trace['q_vis']),
                scalar_target_geometry=union_geometry(runtimes['R1'], points, scalar_result['q_vis']),
                recorded_target_proxies=directional_proxies(geom, snap, trace['q_vis'], trace['q_vis']),
                scalar_target_proxies=directional_proxies(geom, snap, scalar_result['q_vis'], trace['q_vis']), models={})
            x = torch.tensor(points, dtype=torch.float32)
            seed = torch.tensor(q0.reshape(1, 7), dtype=torch.float32)
            for model_name, runtime in runtimes.items():
                scores, order = runtime.rank_sensors(x, torch.tensor([scalar_result['q_zero']], dtype=torch.float32))
                branches = []
                for sensor in range(8):
                    branch = runtime._optimize_branch(x, seed, sensor)
                    assert np.array_equal(np.array(branch['q_start'], dtype=np.float32), seed.numpy()[0])
                    value, grad = runtime.branch_value_and_grad(x, seed, sensor)
                    q_candidate = branch['q_candidate']
                    proxies = directional_proxies(geom, snap, q_candidate, trace['q_vis'])
                    geometry = runtime._candidate_geometry(points, q_candidate, sensor)
                    local_geometry = union_geometry(runtime, points, proxies['local_probe_q'])
                    row = dict(snapshot=snap['label'], obligation=obligation, model=model_name,
                        sensor=sensor, rank=int(np.flatnonzero(order == sensor)[0])+1,
                        seed_score=value, masked_q_gradient=grad.numpy()[0],
                        sensor_mask=runtime.sensor_masks_np[sensor], branch=branch,
                        geometry=geometry, union_geometry=union_geometry(runtime, points, q_candidate),
                        local_probe_union_geometry=local_geometry, proxies=proxies)
                    records.append(row)
                    branches.append(row)
                selected = next((int(s) for s in order[:runtime.max_branch_attempts]
                                 if branches[int(s)]['geometry']['accepted']), None)
                qs = np.array([r['branch']['q_candidate'] for r in branches])
                pairwise = np.max(np.abs(qs[:, None, :]-qs[None, :, :]), axis=2)
                group['models'][model_name] = dict(scores_at_recomputed_scalar_qzero=scores,
                    order=order, production_first_four_selected_sensor=selected,
                    exhaustive_accepted_sensors=[r['sensor'] for r in branches if r['geometry']['accepted']],
                    pairwise_q_inf=pairwise, checkpoint=runtime.checkpoint['runtime_adapter'])
                print(snap['label'], 'O'+str(obligation), model_name,
                      'accepted=', group['models'][model_name]['exhaustive_accepted_sensors'], flush=True)
            groups.append(group)
    # Determinism and direct generate parity on one complete representative case.
    sample = next(r for r in records if r['model']=='R1' and r['obligation']==6)
    snap = snaps[0]
    target = traces[6]['active_set_points_xyz']
    group = next(g for g in groups if g['snapshot']==snap['label'] and g['obligation']==6)
    runtime = runtimes['R1']
    direct = runtime.generate(target, group['scalar_recomputed']['q_zero'], snap['measured_q'])
    expected = group['models']['R1']['production_first_four_selected_sensor']
    assert direct['selected_sensor_id'] == (-1 if expected is None else expected)
    repeat = runtime._optimize_branch(torch.tensor(target, dtype=torch.float32),
                                     torch.tensor([snap['measured_q']], dtype=torch.float32), sample['sensor'])
    assert repeat == sample['branch']
    assert before == {str(p.relative_to(ROOT)): sha(p) for p in protected}
    assert source_hashes == {str(p.relative_to(ROOT)): sha(p) for p in sources}
    assert 'rospy' not in sys.modules
    report = dict(run_id=RID, qualification='OFFLINE_CANDIDATE_DIAGNOSTIC_ONLY',
        device='CPU FP32', tf32=False, amp=False, torch_version=torch.__version__,
        config=config, scalar_generation_reproduction=reproduction, groups=groups, candidates=records,
        lineage=lineage, checkpoint_and_source_sha256=before,
        verification=dict(snapshot_query_measured_and_witness_joins='PASS',
            production_generate_selection_parity='PASS', deterministic_repeat='PASS',
            protected_files_unchanged=True, input_sources_unchanged=True, rospy_not_imported=True),
        not_run=['full effective guard margin reconstruction', 'complete confidence/occupancy map replay',
            'new-candidate Sparse-SCP/GCDF/exact-VBC checks', 'certified prefix/brake/hold',
            'environment LOS', 'actual seen and parent resume', 'ROS/Gazebo', 'CUDA'],
        limits=['No production budget change: offline loops enumerate all eight heads.',
                'Point-center clearance is not voxel/full-map safety.',
                'Saved linearizations are not re-evaluated collision CDF and carry no pass verdict.',
                'Counterfactual O1/O6 endpoints do not require simultaneous observation.',
                'Final query precedes budget hold and is not relabelled as the hold snapshot.'])
    save(args.output_dir/'diagnostic.json', report)
    fields = ['snapshot','obligation','model','sensor','rank','geometry_pass','reject_reason',
              'conservative_g','self_occluded','q_distance_inf','cosine_old_direction',
              'start_clearance_m','local_probe_clearance_m','sampled_line_clearance_m',
              'endpoint_clearance_m','local_visible_fraction','endpoint_visible_fraction','min_witness_slope']
    with (args.output_dir/'candidates.csv').open('x') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            g, p = r['geometry'], r['proxies']
            body = p['point_body_geometry']
            writer.writerow(dict(snapshot=r['snapshot'], obligation=r['obligation'], model=r['model'],
                sensor=r['sensor'], rank=r['rank'], geometry_pass=g['accepted'], reject_reason=g['reject_reason'],
                conservative_g=g['min_conservative_g'], self_occluded=g['any_primitive_self_occluded'],
                q_distance_inf=p['q_distance_inf'], cosine_old_direction=p['cosine_to_recorded_target'],
                start_clearance_m=body['start_min_m'], local_probe_clearance_m=body['local_probe_min_m'],
                sampled_line_clearance_m=body['sampled_line_min_m'], endpoint_clearance_m=body['endpoint_min_m'],
                local_visible_fraction=r['local_probe_union_geometry']['visible_point_fraction'],
                endpoint_visible_fraction=r['union_geometry']['visible_point_fraction'],
                min_witness_slope=p['witness_model_direction']['min_slope']))
    print('PASS: wrote', args.output_dir, 'candidates=', len(records), flush=True)


if __name__ == '__main__':
    main()
