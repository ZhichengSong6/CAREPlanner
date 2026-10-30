#!/usr/bin/env python3
"""Replay every head at saved failed REPAIR replacement states; diagnostic only."""

import ast
import glob
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
VIS = ROOT / 'src/care_visibility_cdf/scripts'
sys.path.insert(0, str(VIS))
from per_sensor_visibility_runtime import PerSensorVisibilityRuntime  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def joint_limits():
    tree = ast.parse((VIS / 'vbc_deadline_waypoint_node.py').read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('DEFAULT_Q_MIN', 'DEFAULT_Q_MAX'):
                    values[target.id] = ast.literal_eval(node.value)
    return values['DEFAULT_Q_MIN'], values['DEFAULT_Q_MAX']


def main():
    batch = 'r1_margin20_normal_progress_4_20260923'
    out = ROOT / 'outputs/phase_e_replacement_score_gap/offline_prior_candidates.json'
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / 'src/care_visibility_cdf/checkpoints/hierarchical9_r1_scratch50k/final.pt'
    assert sha(checkpoint) == '4f395926fa79c29474be8748cef4733ec400d155cd8fadb76c632c2838864002'
    q_min, q_max = joint_limits()
    torch.set_num_threads(1)
    runtime = PerSensorVisibilityRuntime(
        str(checkpoint), str(ROOT / 'src/arm_description/urdf/Arm.urdf'),
        str(ROOT / 'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
        torch.device('cpu'), q_min, q_max)
    rows = []
    inputs = {}
    for case in ('005', '018'):
        paths = glob.glob(str(ROOT / 'outputs/c5_5_vbc_gcdf_regime' /
            f'phase_e_goal_{case}_{batch}/run/projector_traces/candidate_replacement.jsonl'))
        assert len(paths) == 1
        path = Path(paths[0])
        inputs[str(path.relative_to(ROOT))] = sha(path)
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event['event'] != 'no_geometric_candidate':
                continue
            hybrid = event['hybrid']
            cfg = hybrid['config']
            assert cfg['checkpoint_sha256'] == sha(checkpoint)
            assert cfg['min_conservative_g'] == 0.0
            threshold = hybrid['attempts'][0]['geometry']['minimum_conservative_g_required']
            points = torch.tensor(event['points'], dtype=torch.float32)
            seed = torch.tensor(hybrid['branch_seed_q'], dtype=torch.float32).reshape(1, 7)
            branches = []
            for sid in range(8):
                branch = runtime._optimize_branch(points, seed, sid)
                geometry = runtime._candidate_geometry(
                    event['points'], branch['q_candidate'], sid,
                    minimum_conservative_g=threshold)
                branches.append(dict(sensor_id=sid, score_at_measured_seed=
                    hybrid['scores_at_branch_seed'][sid],
                    final_score=branch['final_score'], root_found=branch['root_found'],
                    geometry_accepted=geometry['accepted'],
                    geometry_reject_reason=geometry['reject_reason'],
                    min_conservative_g=geometry['min_conservative_g'],
                    min_nominal_margin=geometry['min_nominal_margin']))
            attempted = hybrid['attempts'][0]
            actual = branches[attempted['sensor_id']]
            assert actual['geometry_reject_reason'] == attempted['geometry']['reject_reason']
            assert abs(actual['min_conservative_g'] - attempted['geometry']['min_conservative_g']) < 1e-4
            rows.append(dict(case_id=f'phase_e_goal_{case}', request_id=event['request_id'],
                ros_s=event['ros_s'], points=event['points'],
                attempted_sensor_id=attempted['sensor_id'],
                attempted_geometry=attempted['geometry']['reject_reason'],
                geometrically_accepted_sensor_ids=[
                    row['sensor_id'] for row in branches if row['geometry_accepted']],
                branches=branches))
    result = dict(schema='phase_e_replacement_all_heads_offline_v1',
        inputs_sha256=inputs, checkpoint_sha256=sha(checkpoint),
        qualification='Recorded measured seed, exact production branch solver and analytic FOV/primitive self-LOS',
        limit='No final VBC, GCDF, map line-of-sight, trajectory or execution certification',
        rows=rows)
    out.write_text(json.dumps(result, indent=2) + '\n')
    for row in rows:
        print(row['case_id'], row['request_id'], row['attempted_sensor_id'],
              row['geometrically_accepted_sensor_ids'])
    print(out)


if __name__ == '__main__':
    main()
