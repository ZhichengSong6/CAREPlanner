#!/usr/bin/env python3
"""Offline checks of runtime GCDF/FK contracts at two real rejected poses.

This cannot establish the training geometry or training-label accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/care_collision_cdf/scripts'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = ap.parse_args()
    import numpy as np
    import pinocchio as pin
    import torch
    from collision_cdf_model import CollisionCDF, _looks_like_state_dict
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = ROOT/'src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt'
    net = CollisionCDF(str(checkpoint), device=args.device)
    payload = torch.load(checkpoint, map_location='cpu')
    meta = [] if _looks_like_state_dict(payload) else [str(k) for k, v in payload.items()
        if not _looks_like_state_dict(v)]
    urdfs = [ROOT/'src/arm_description/urdf'/name for name in
             ['Arm.urdf', 'Arm_with_self_filter_collision.urdf']]
    models = [pin.buildModelFromUrdf(str(p)) for p in urdfs]
    joints = ['joint1', 'joint2', 'joint3', 'joint4', 'wrist_joint1', 'wrist_joint2', 'wrist_joint3']
    assert all(list(m.names)[1:] == joints for m in models)
    cases = [
        ('phase_e_goal_026_empty_qualification_20260914-unknown-recovery-01_dcde98b2',
         'rejection_04_da6b653bba73.json', [.10000000149011612, .10000000149011612, .30000001192092896], 5),
        ('phase_e_goal_007_empty_qualification_20260914-unknown-perf-pair-02-primitive_dcde98b2',
         'rejection_05_588ef5ded6d8.json', [-.1499999999999999, -.19999999999999996, .8500000000000001], 5)]
    rows = []
    sources = {}
    for run, file, point, k in cases:
        path = ROOT/'outputs/c5_5_vbc_gcdf_regime'/run/'run/rejection_snapshots'/file
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        record = json.loads(path.read_text())
        assert record['audited_trajectory']['joint_names'] == joints
        q = np.array(record['audited_trajectory']['points'][k]['positions'])
        pt = torch.tensor([point], device=args.device, dtype=torch.float32)
        qt = torch.tensor(q[None], device=args.device, dtype=torch.float32)
        distance, grad = net.pair_distance_and_gradient(pt, qt)
        finite_differences = []
        h = .001
        with torch.no_grad():
            for j in range(7):
                plus, minus = qt.clone(), qt.clone()
                plus[0, j] += h; minus[0, j] -= h
                fplus = net.model(torch.cat((pt, plus), dim=1)).item()
                fminus = net.model(torch.cat((pt, minus), dim=1)).item()
                finite_differences.append((fplus-fminus)/(2*h))
        error = float(np.max(np.abs(grad.cpu().numpy()[0]-finite_differences)))
        assert error < .002, '7D finite-difference discrepancy'
        data = [m.createData() for m in models]
        for m, d in zip(models, data):
            assert np.all(q >= m.lowerPositionLimit) and np.all(q <= m.upperPositionLimit)
            pin.framesForwardKinematics(m, d, q)
        # Self-filter URDF intentionally omits fixed camera/EE child frames.
        # Compare every self-filter body link, not nonexistent sensor frames.
        links = [link.attrib['name'] for link in ET.parse(urdfs[1]).getroot().findall('link')]
        assert all(m.existFrame(link) for m in models for link in links)
        fk_error = max(float(np.max(np.abs(data[0].oMf[models[0].getFrameId(link)].homogeneous -
            data[1].oMf[models[1].getFrameId(link)].homogeneous))) for link in links)
        assert fk_error < 1e-10
        rows.append(dict(snapshot=str(path.relative_to(ROOT)), point=point, q=q.tolist(),
            learned_distance=distance.item(), gradient_7d=grad.cpu().numpy()[0].tolist(),
            finite_difference_7d=finite_differences, max_gradient_error=error,
            current_urdf_fk_max_error=fk_error, compared_body_links=links, joint_limits_pass=True))
    sources.update({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [checkpoint]+urdfs})
    report = dict(status='PASS', scope='runtime contract only, NOT training-label validation',
        architecture=net.architecture, checkpoint_selected=net.selected_checkpoint,
        checkpoint_metadata_keys=meta, joint_names=joints, input='raw xyz + raw q7 (radians)',
        training_geometry_status='NOT_AVAILABLE: exact training URDF/label generator not supplied',
        activation_status='GELU runtime configuration; activation cannot be inferred from weights',
        cases=rows, sha256=sources)
    (args.output_dir/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
