#!/usr/bin/env python3
"""Freeze fault queries, QPs and geometry controls for a future CDF checkpoint."""
import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from cdf_qualification_common import BOX_MIN, BOX_MAX, JOINTS, in_box, new_output_dir, read, save, sha

ROOT = Path(__file__).resolve().parents[1]
FAULT = ROOT / 'outputs/phase_e_case029_task_rejoin_20261022'
TASK = ROOT / 'outputs/phase_e_rebase_task_ab_20261022'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    out = new_output_dir(args.bundle)
    sources = {}

    def track(path):
        sources[str(path.relative_to(ROOT))] = sha(path)
        return path

    # Reuse the exact active-body diagnostic that accompanied the fixed QPs.
    common_path = track(FAULT / 'common.py')
    spec = importlib.util.spec_from_file_location('frozen_cdf_geometry', common_path)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    geometry = common.Geometry()
    from validate_visibility_oracle import fk_transform
    primitive_urdf = track(common.RT / 'src/arm_description/urdf/Arm_with_self_filter_collision.urdf')
    for name in ('validate_visibility_oracle.py', 'check_visibility_self_occlusion.py'):
        track(common.RT / 'src/care_visibility_cdf/scripts' / name)
    corpus = read(track(FAULT / 'cdf_regression_queries.json'))
    assert corpus['unique_queries'] == 1248
    queries = []
    for query in corpus['queries']:
        assert in_box(query['point_m'])
        queries.append(dict(query, group='retained_fault', expected_input_valid=True))
    controls, poses = [], []
    runs = read(track(TASK / 'measurements.json'))
    for tag, run in sorted(runs.items()):
        if not run['metrics']['task_success']:
            continue
        path = track(TASK / 'runtime' / run['metrics']['run_directory'] / 'joint_states.csv')
        window = run['rejoin_timing']
        with path.open() as stream:
            rows = [r for r in csv.DictReader(stream)
                    if window['start_ros_ns'] <= int(r['%time']) <= window['end_ros_ns']]
        assert rows
        for label, index in [('start', 0), ('middle', len(rows) // 2), ('end', len(rows) - 1)]:
            row = rows[index]
            names = [row[f'field.name{j}'] for j in range(7)]
            assert names == JOINTS, names
            q = [float(row[f'field.position{j}']) for j in range(7)]
            assert np.isfinite(q).all()
            poses.append(dict(id=f'{tag}_{label}', q_rad=q, source=str(path.relative_to(ROOT)),
                              receipt_ros_ns=int(row['%time']), window=dict(start=window['start_ros_ns'], end=window['end_ros_ns'])))
    assert len(poses) == 24

    def append(group, point, pose, valid=True):
        point = np.asarray(point, dtype=float).tolist()
        d, ix = geometry.minimum(pose['q_rad'], [point])
        identity = json.dumps([group, pose['id'], point], separators=(',', ':'))
        controls.append(dict(id=hashlib.sha256(identity.encode()).hexdigest(), group=group,
                             point_m=point, q_rad=pose['q_rad'], pose_id=pose['id'],
                             current_primitive_point_distance_m=float(d[0]),
                             nearest_primitive=geometry.primitives[int(ix[0])]['name'],
                             expected_input_valid=valid))

    grid = np.array(list(itertools.product(np.linspace(-.8, .8, 9), np.linspace(-.8, .8, 9), np.linspace(0, 1, 6))))
    for pose in poses:
        distances, _ = geometry.minimum(pose['q_rad'], grid)
        eligible = np.flatnonzero(distances >= .05)
        assert len(eligible) >= 3
        eligible = eligible[np.argsort(distances[eligible], kind='stable')]
        for index in (0, len(eligible) // 2, len(eligible) - 1):
            append('normal_clear', grid[eligible[index]], pose)
        # Generate near-solid probes deterministically, then label by the union
        # distance. These are current-geometry diagnostics, not CDF distances.
        probes = []
        frames = {link: fk_transform(chain, dict(zip(JOINTS, pose['q_rad']))) for link, chain in geometry.chains.items()}
        for primitive in geometry.primitives:
            tf = frames[primitive['link']] @ primitive['origin']
            g = primitive['geometry']
            extents = (np.asarray(g.size) / 2 if primitive['type'] == 'box' else
                       np.array([g.radius] * 3) if primitive['type'] == 'sphere' else
                       np.array([g.radius, g.radius, g.length / 2]))
            for axis, sign, offset in itertools.product(range(3), (-1, 1), (-.005, .005, .015)):
                local = np.zeros(3)
                local[axis] = sign * (extents[axis] + offset)
                point = tf[:3, :3] @ local + tf[:3, 3]
                if in_box(point):
                    probes.append(point)
        ds, _ = geometry.minimum(pose['q_rad'], probes)
        for group, lower, upper in [('inside_current_body', -.006, -.001),
                                     ('clearance_5mm', .003, .007), ('clearance_15mm', .013, .017)]:
            found = np.flatnonzero((ds >= lower) & (ds <= upper))
            assert len(found), (pose['id'], group)
            append(group, probes[found[0]], pose)
        boundary = list(itertools.product(*zip(BOX_MIN, BOX_MAX)))
        center = (BOX_MIN + BOX_MAX) / 2
        for axis, sign in itertools.product(range(3), (-1, 1)):
            p = center.copy()
            p[axis] = BOX_MIN[axis] if sign < 0 else BOX_MAX[axis]
            boundary.append(p.copy())
            p[axis] -= sign * .025
            boundary.append(p)
        for point in boundary:
            assert in_box(point)
            append('workspace_boundary', point, pose)
    for axis, sign in itertools.product(range(3), (-1, 1)):
        point = (BOX_MIN + BOX_MAX) / 2
        point[axis] = (BOX_MIN[axis] - .001) if sign < 0 else (BOX_MAX[axis] + .001)
        assert not in_box(point)
        append('out_of_domain_probe', point, poses[0], valid=False)
    queries.extend(controls)
    assert len({q['id'] for q in queries}) == len(queries)
    save(out / 'queries.json', queries)
    save(out / 'control_poses.json', poses)
    shutil.copy2(track(FAULT / 'cdf_regression_queries.json'), out / 'original_fault_queries.json')
    plan = read(track(FAULT / 'offline_plan.json'))
    qps = []
    for index, item in enumerate(plan['snapshots']):
        source = track(ROOT / item['source'])
        assert sha(source) == item['sha256']
        target = out / 'qps' / f'{index:02d}.json'
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
        qps.append(dict(index=index, file=str(target.relative_to(out)), sha256=sha(target), source=item))
    assert len(qps) == 28
    runtime = out / 'runtime'
    runtime.mkdir()
    for source in (ROOT / 'src/care_collision_cdf/scripts/collision_cdf_model.py', primitive_urdf):
        shutil.copy2(track(source), runtime / source.name)
    executable = track(FAULT / 'replay_qp')
    shutil.copy2(executable, runtime / 'replay_qp')
    checkpoint = track(ROOT / 'src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt')
    config = read(track(ROOT / 'src/care_collision_cdf/config/cdf_training_contract.json'))
    groups = {group: sum(q['group'] == group for q in queries) for group in sorted({q['group'] for q in queries})}
    fd_indices = []
    for group in groups:
        indices = [i for i, q in enumerate(queries) if q['group'] == group and q['expected_input_valid']]
        if indices:
            fd_indices.extend(np.asarray(indices)[np.linspace(0, len(indices) - 1, min(12, len(indices)), dtype=int)].tolist())
    bundle = dict(schema=1, status='FROZEN_BEFORE_MODEL_COMPARISON', groups=groups, total_queries=len(queries),
                  finite_difference_indices=fd_indices, finite_difference_scales=[.001, .0005],
                  derivative_atol=.005, derivative_rtol=.02,
                  joint_order=JOINTS, target_domain=dict(xyz_min_m=BOX_MIN.tolist(), xyz_max_m=BOX_MAX.tolist()),
                  baseline=dict(checkpoint=str(checkpoint.relative_to(ROOT)), sha256=sha(checkpoint), activation='gelu'),
                  geometry=dict(urdf_sha256=sha(primitive_urdf), excluded_links=['base_link', 'link1'], primitives=21,
                                scope='Current active-body point geometry, not original configuration-space training truth'),
                  qps=qps, repeats=3, runtime_contract=config['runtime_input'],
                  output_units='configuration_space_model', model_margin_for_control_diagnostics=.05,
                  limitations=['Controls selected from eight successful development task windows, not an independent test distribution',
                               'Workspace boundary probes are finite coverage; grid spacing is not a neural error bound',
                               'Out-of-domain rejection is exercised in the offline input validator, not added to production',
                               'QP comparison only replaces saved selected CDF rows at the original linearization points; nonlinear SCP and any new active set are not rerun'])
    save(out / 'bundle.json', bundle)
    save(out / 'training_manifest_template.json', dict(
        schema=1, checkpoint_sha256=None, activation='gelu', runtime_input={k: v for k, v in config['runtime_input'].items() if k != 'activation'},
        output_units='configuration_space_model', positive_means='outside_collision', excluded_links=['base_link', 'link1'],
        training_urdf=dict(path=None, sha256=None), label_generator=dict(path=None, sha256=None),
        dataset_manifest=dict(path=None, sha256=None), label_definition=None,
        training_domain=bundle['target_domain'],
        instruction='Fill from actual training evidence. A different geometry triggers explicit review; do not replace its hashes with runtime geometry hashes.'))
    save(out / 'source_inputs_sha256.json', sources)
    inventory = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}
    save(out / 'artifact_sha256.json', inventory)
    print(json.dumps(dict(status='BUNDLE_FROZEN', groups=groups, qps=len(qps), fd_queries=len(fd_indices))))


if __name__ == '__main__':
    main()
