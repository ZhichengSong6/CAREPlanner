#!/usr/bin/env python3
"""Dense geometric follow-up of saved candidates; never certifies execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from diagnose_phase_e_r1_o1_o6_offline import (
    ROOT, WEIGHTS, PerSensorVisibilityRuntime, Geometry, production_scalar,
    union_geometry, save, sha,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paths = [args.input_dir/'diagnostic.json', args.input_dir/'snapshots.json']
    before = {str(p): sha(p) for p in paths}
    data, evidence = [json.loads(p.read_text()) for p in paths]
    scalar = production_scalar(data['config'])
    runtime = PerSensorVisibilityRuntime(
        checkpoint_path=str(WEIGHTS/'hierarchical9_r1_scratch50k/final.pt'),
        reference_urdf_path=str(ROOT/'src/arm_description/urdf/Arm.urdf'),
        self_filter_urdf_path=str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
        device=torch.device('cpu'), q_min=scalar.q_min.tolist(), q_max=scalar.q_max.tolist())
    geometry = Geometry()
    selected = [
        ('parent_requests_child', 1, 'R1', 5),
        ('child_requests_parent', 6, 'R1', 4),
        ('child_requests_parent', 6, 'R1', 7),
        ('last_dependency_before_budget_hold', 6, 'R1', 4),
        ('last_dependency_before_budget_hold', 6, 'R1', 7),
        ('last_dependency_before_budget_hold', 6, 'V1', 6),
    ]
    results = []
    for label, obligation, model, sensor in selected:
        candidate = next(r for r in data['candidates'] if
                         (r['snapshot'], r['obligation'], r['model'], r['sensor']) ==
                         (label, obligation, model, sensor))
        snap = next(s for s in evidence['snapshots'] if s['label'] == label)
        target = evidence['traces'][str(obligation)]['active_set_points_xyz']
        q0 = np.array(snap['measured_q'])
        q1 = np.array(candidate['branch']['q_candidate'])
        points = np.array(snap['unique_unknown_points'])
        rows = []
        for alpha in np.linspace(0, 1, 129):
            q = q0 + alpha*(q1-q0)
            distances = geometry.distances(q, points)
            point_i, primitive_i = np.unravel_index(np.argmin(distances), distances.shape)
            visibility = union_geometry(runtime, target, q)
            rows.append(dict(alpha=float(alpha), q=q.tolist(),
                min_point_body_distance_m=float(distances[point_i, primitive_i]),
                closest_point=points[point_i].tolist(),
                closest_primitive_link=geometry.primitives[primitive_i]['link'],
                closest_primitive_index=int(primitive_i),
                visible_fraction=visibility['visible_point_fraction'],
                all_target_points_visible=visibility['all_points_visible_any_sensor'],
                best_single_sensor_g=visibility['best_single_sensor_g'],
                per_head=[dict(sensor=i, accepted=g['accepted'],
                    conservative_g=g['min_conservative_g'], self_occluded=g['any_primitive_self_occluded'])
                          for i, g in enumerate(visibility['heads'])]))
        assert abs(rows[-1]['best_single_sensor_g'] -
                   candidate['union_geometry']['best_single_sensor_g']) < 1e-12
        assert rows[-1]['visible_fraction'] == candidate['union_geometry']['visible_point_fraction']
        contact = next((r for r in rows if r['min_point_body_distance_m'] <= 0), None)
        visible = next((r for r in rows if r['all_target_points_visible']), None)
        best = min(rows, key=lambda r: r['min_point_body_distance_m'])
        summary = dict(snapshot=label, obligation=obligation, model=model, sensor=sensor,
            rank=candidate['rank'], endpoint_selected_head_pass=candidate['geometry']['accepted'],
            q_distance_inf=candidate['proxies']['q_distance_inf'],
            min_sampled_point_body_distance_m=best['min_point_body_distance_m'],
            min_at_alpha=best['alpha'], min_point=best['closest_point'],
            min_primitive_link=best['closest_primitive_link'],
            first_sampled_point_contact_alpha=contact['alpha'] if contact else None,
            first_sampled_all_target_visible_alpha=visible['alpha'] if visible else None,
            first_visible_point_body_distance_m=visible['min_point_body_distance_m'] if visible else None,
            visible_before_first_sampled_contact=(bool(visible and (not contact or visible['alpha'] < contact['alpha']))),
            qualification='129 straight-line samples, recorded point centers, analytic FOV and primitive self-LOS only',
            execution_safety='NOT_RUN', actual_seen='NOT_RUN', rows=rows)
        results.append(summary)
        print({k: v for k, v in summary.items() if k != 'rows'}, flush=True)
    assert before == {str(p): sha(p) for p in paths}
    save(args.output, dict(input_sha256=before, script_sha256=sha(Path(__file__)), results=results,
        limits=['Interpolation is a diagnostic straight line, not Sparse-SCP output or a timed trajectory.',
                'Positive sampled point-center distances do not certify voxels, continuous motion, full map or margins.',
                'Analytic FOV and primitive self-LOS omit environment LOS and confidence updates.',
                'Selected head failure may coexist with another sensor seeing the target; production selection remains unchanged.']))


if __name__ == '__main__':
    main()
