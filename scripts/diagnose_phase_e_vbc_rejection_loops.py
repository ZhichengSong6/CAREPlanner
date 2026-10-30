#!/usr/bin/env python3
"""Read-only paired-run forensics. CPU geometry only; no ROS or plan execution.

Consumes the exact-stamp/window joins in the paired summary, retaining source
hashes. Endpoint visibility is not trajectory feasibility or an execution permit.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def point_key(point):
    # Match solver input precision, not a spatial-nearest-neighbour heuristic.
    return struct.pack('!3f', *point).hex()


def in_window(rows, start, end):
    return [r for r in rows if start <= float(r['ros_s']) <= end]


def numeric_range(rows, key):
    import math
    values = [float(r[key]) for r in rows if key in r]
    values = [v for v in values if math.isfinite(v)]
    return [min(values), max(values)] if values else None


def endpoint_geometry(runtime, points, q):
    result = []
    for point in points:
        sensors = [dict(sensor_id=s, **runtime._candidate_geometry([point], q, s))
                   for s in range(8)]
        result.append(dict(point=point, sensors=sensors,
                           passing_sensors=[r['sensor_id'] for r in sensors if r['accepted']]))
    # Union coverage can use different sensors for different points.
    return dict(points=result, every_point_has_clear_sensor=all(r['passing_sensors'] for r in result))


def diagnose_case(case, runtime):
    import numpy as np
    from diagnose_phase_e_point_geometry import primitive_sdf, fk_transform, JOINTS
    observations = case['observation']['observations']
    obs = max(observations, key=lambda o: sum(v['result'] == 'unsafe' and
              v['safety_gate'] == 'vbc' for v in o['verification']))
    scheduled = [h for h in obs['history'] if h['stage'] == 'scheduled']
    if len(scheduled) != 1:
        raise ValueError('Dominant token must have one scheduled identity; split episodes first')
    target = scheduled[0]
    start, end = target['ros_s'], case['benchmark_window']['end_ros_s']
    verifications = in_window(obs['verification'], start, end)
    rejected = [v for v in verifications if v['result'] == 'unsafe' and v['safety_gate'] == 'vbc']
    assert rejected, 'No final VBC rejections for selected token'
    assert len({v['raw_candidate_stamp_ns'] for v in rejected}) == len(rejected)
    points = target['target_solver_points']
    point_keys = {point_key(p) for p in points}
    violations = []
    routes = []
    for v in rejected:
        detail = v['vbc_detail']
        assert str(detail['trajectory_stamp_ns']) == v['audited_trajectory_stamp_ns']
        for route in v['vbc_route_events']:
            identity = route['vbc_identity']
            assert str(identity['trajectory_stamp_ns']) == str(detail['trajectory_stamp_ns'])
            assert int(identity['bundle_seq']) == int(detail['bundle_seq'])
            routes.append(route)
        violations.extend(detail['violations'])
    grouped = {}
    for v in violations:
        key = point_key(v['point'])
        grouped.setdefault(key, []).append(v)
    seen = [h for h in in_window(obs['history'], start, end)
            if h['stage'] == 'seen_query' and h.get('measured_q') is not None]
    measured = np.asarray([h['measured_q'] for h in seen])
    local = in_window(obs['local_planning_events'], start, end)
    candidates = [r for r in local if r['event'] == 'candidate_published']
    upstream = in_window(case['repair_state']['upstream_witness_diagnostics'], start, end)
    witness_keys = {point_key(json.loads(r['point'])) for r in upstream}
    q_end = case['benchmark_window']['end_measured_q']
    qmap = dict(zip(JOINTS, q_end))
    boundary = []
    for key, rows in grouped.items():
        p = np.asarray(rows[-1]['point'])
        distances = []
        for primitive, chain in zip(runtime.primitives, runtime.collision_chains):
            if primitive['link'] in ('base_link', 'link1'):
                continue
            transform = fk_transform(chain, qmap) @ primitive['origin']
            local_p = transform[:3, :3].T @ (p - transform[:3, 3])
            g = primitive['geometry']
            size = list(g.size) if primitive['type'] == 'box' else (
                [g.radius, g.length] if primitive['type'] == 'cylinder' else [g.radius])
            distances.append(dict(link=primitive['link'], collision=primitive['name'],
                                  sdf_m=primitive_sdf(primitive['type'], local_p, size)))
        boundary.append(dict(point=p.tolist(), violation_records=len(rows),
            in_solver_fp32_points=key in point_keys,
            in_retained_witness_reevaluations=key in witness_keys,
            candidate_sdf_m=numeric_range(rows, 'primitive_signed_distance_m'),
            sweep_s=numeric_range(rows, 'sweep_s'), latest_see_s=numeric_range(rows, 'latest_see_s'),
            last_exact_violation=rows[-1],
            end_measured_nearest_primitive=min(distances, key=lambda r: r['sdf_m'])))
    range_keys = ['verification_age_s', 'dispatch_start_measured_mismatch_inf',
                  'handoff_rebase_shift_inf', 'cycle_recovery_active', 'cycle_recovery_prefix_s',
                  'raw_candidate_duration_s', 'constructed_executable_duration_s']
    return dict(case_id=case['case_id'], observation_token=obs['observation_token'],
        scheduled=target, window_ros_s=[start, end], raw_candidates=len(obs['raw_candidate_stamps']),
        final_vbc_rejections=len(rejected), committed=sum(v['committed'] == '1' for v in verifications),
        first_last_rejection_ros_s=[rejected[0]['ros_s'], rejected[-1]['ros_s']],
        tracker_records=obs['tracker_records'], first_actual_seen_ros_s=obs['first_actual_seen_ros_s'],
        measured_queries=len(seen), measured_q_inf_span=float(np.max(np.ptp(measured, axis=0))) if seen else None,
        measured_q_distance_min_inf=float(np.min(np.max(np.abs(measured-np.asarray(target['q_vis'])), axis=1))) if seen else None,
        q_end=q_end, endpoint=endpoint_geometry(runtime, points, target['q_vis']),
        measured_endpoint=endpoint_geometry(runtime, points, q_end), boundary_points=boundary,
        exact_routes=len(routes), route_modes=dict(Counter(x['route'] for r in routes for x in r['routes'])),
        routes_same_published_token=sum(r['published_observation_token'] == obs['observation_token'] for r in routes),
        routes_with_new_ids=sum(bool(r['new_ids']) for r in routes),
        verification_ranges={k: numeric_range(rejected, k) for k in range_keys},
        local_candidate_events=len(candidates), local_event_counts=dict(Counter(r['event'] for r in local)),
        local_candidate_ranges={k: numeric_range(candidates, k) for k in [
            'repair_failures', 'unknown_cdf_rows', 'repair_progress_epoch', 'repair_target_revision']},
        last_local_event=local[-1] if local else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batch-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Refusing to overwrite diagnostic directory')
    frozen_path = args.batch_dir / 'frozen_sha256.json'
    frozen = json.loads(frozen_path.read_text())
    checked = {p: sha(REPO/p) for p in frozen if p.startswith(('src/', 'devel/'))}
    assert all(checked[p] == frozen[p] for p in checked), 'Frozen runtime/input drift'
    summary_path = args.batch_dir / 'summary.json'
    data = json.loads(summary_path.read_text())
    from test_per_sensor_visibility_runtime_hierarchical9 import make_runtime, H9
    runtime = make_runtime(H9, 'cpu')  # geometry only: no projection, inference, or CUDA
    result = dict(schema_version=1, source_sha256={str(summary_path): sha(summary_path),
        str(frozen_path): sha(frozen_path), str(Path(__file__)): sha(__file__)},
        frozen_runtime_files_checked=len(checked), runtime_frozen_check='PASS',
        scope='Dominant VBC-rejected token in each primitive run; not all failure episodes',
        geometry_parameters=dict(conservative_hfov_deg=runtime.cons_hfov,
            conservative_vfov_deg=runtime.cons_vfov, conservative_z_min=runtime.cons_z_min,
            conservative_z_max=runtime.cons_z_max, conservative_delta=runtime.cons_delta,
            min_conservative_g=runtime.min_conservative_g, primitive_los=runtime.require_primitive_los),
        cases=[], baseline=[])
    for case in data['cases']:
        if case['geometry_backend'] != 'primitive':
            result['baseline'].append({k: case[k] for k in ['case_id', 'task_success', 'final_position_error_m']})
            continue
        for name in ['projector_traces/observation_events.jsonl', 'observation_candidate_identity.csv',
                     'verification_outcome.csv', 'runtime_semantics.txt']:
            path = Path(case['run']) / name
            result['source_sha256'][str(path)] = sha(path)
        semantics = (Path(case['run'])/'runtime_semantics.txt').read_text()
        assert 'per_sensor_min_conservative_g=0.0\n' in semantics
        assert 'per_sensor_require_primitive_los=true\n' in semantics
        result['cases'].append(diagnose_case(case, runtime))
    assert all(sha(REPO/p) == expected for p, expected in checked.items()), 'Runtime changed during analysis'
    result['limitations'] = [
        'No full rejected JointTrajectory: no exact knot-q old/new rasterizer or GCDF replay.',
        'End-measured q is not substituted for the rejected candidate sweep q.',
        'Retained witness absence does not prove absence from all GCDF query pairs.',
        'Endpoint analytic FOV/URDF LOS is not actual confidence, safe reachability, or global infeasibility.',
        'Existing paired summary supplies exact-stamp joins; raw inputs hashed, no simulation rerun.']
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for c in result['cases']:
        print(c['case_id'], 'rejected=', c['final_vbc_rejections'], 'committed=', c['committed'],
              'endpoint_clear=', c['endpoint']['every_point_has_clear_sensor'],
              'measured_span=', c['measured_q_inf_span'])
    print('Forensic checks PASS; original task failures retained. Output:', args.output_dir)


if __name__ == '__main__':
    main()
