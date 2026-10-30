#!/usr/bin/env python3
"""Matched small development fine-tunes, isolated from all runtime checkpoints."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src/care_visibility_cdf/scripts'), str(ROOT/'scripts')]
from per_sensor_visibility_runtime import build_per_sensor_model
from visibility_boundary_supervision import interval_loss
from train_signed_visibility_cdf_pairwise_replace import (PinocchioFOVOracle, DEFAULT_JOINT_NAMES,
    DEFAULT_SENSOR_FRAMES, decode_per_sensor_distance_and_grad)
from train_per_sensor_visibility_cdf import per_sensor_signed_targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--interval-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=200)
    args = ap.parse_args()
    assert args.steps == 200, 'predeclared development budget'
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    checkpoint = ROOT/'src/care_visibility_cdf/checkpoints/hierarchical9_r1_scratch50k/final.pt'
    original_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    view, _ = build_per_sensor_model(str(checkpoint), torch.device('cpu'))
    frozen = view.full_model.eval().requires_grad_(False)
    samples = json.loads((args.interval_dir/'interval_samples.json').read_text())
    with np.load(ROOT/'outputs/phase_e_candidate_network_20261019/label_subset.npz') as src:
        z = {k:src[k] for k in src.files}
    lookup = {int(v):i for i,v in enumerate(z['indices'])}
    oracle = PinocchioFOVOracle(str(ROOT/'src/arm_description/urdf/Arm.urdf'),
        DEFAULT_JOINT_NAMES, DEFAULT_SENSOR_FRAMES, 50., 66., .2, .7, .01)
    masks = torch.tensor(z['sensor_chain_masks'], dtype=torch.float32)
    legacy = []
    for row in samples:
        i, sid = lookup[row['point_index']], row['sensor_id']
        q = torch.tensor([row['q']], dtype=torch.float32)
        _, sign = oracle.signed_fov_margins(torch.tensor(z['x'][i:i+1]), q)
        ds, dg, has = decode_per_sensor_distance_and_grad(torch.tensor(z['q'][i:i+1]),
            torch.tensor(z['valid_fov'][i:i+1]), q, masks)
        target, _, valid = per_sensor_signed_targets(ds, dg, sign, has)
        assert valid[0,0,sid]
        legacy.append(float(target[0,0,sid]))
    inputs = torch.tensor([r['x']+r['q'] for r in samples], dtype=torch.float32)
    sid = torch.tensor([r['sensor_id']+1 for r in samples], dtype=torch.long)
    lo, hi = [torch.tensor([r[k] for r in samples]) for k in ('lower_rad', 'upper_rad')]
    legacy = torch.tensor(legacy)
    train = np.array([i for i,r in enumerate(samples) if r['development_split']=='train'])
    validation = np.array([i for i,r in enumerate(samples) if r['development_split']=='validation'])
    assert set(samples[i]['point_index'] for i in train).isdisjoint(samples[i]['point_index'] for i in validation)
    rng = np.random.default_rng(0)
    points = sorted({samples[i]['point_index'] for i in train})
    retention = torch.tensor([list(z['all_x'][int(rng.choice(points))])+
        list(rng.uniform(z['q_min'], z['q_max'])) for _ in range(512)], dtype=torch.float32)
    validation_points = sorted({samples[i]['point_index'] for i in validation})
    retention_validation = torch.tensor([list(z['all_x'][int(rng.choice(validation_points))])+
        list(rng.uniform(z['q_min'], z['q_max'])) for _ in range(256)], dtype=torch.float32)
    with torch.no_grad():
        retention_targets = frozen(retention)
        retention_validation_targets = frozen(retention_validation)
    batches = [rng.choice(train, 64, replace=True) for _ in range(args.steps)]
    retention_batches = [rng.choice(len(retention), 64, replace=True) for _ in range(args.steps)]

    def evaluate(model):
        result = {}
        for name, ids in [('train', train), ('validation', validation)]:
            x = inputs[ids].clone().requires_grad_(True)
            p = model(x).gather(1, sid[ids,None]).flatten()
            gradient = torch.autograd.grad(p.sum(), x)[0][:,3:]*masks[sid[ids]-1]
            violation = torch.relu(lo[ids]-p)+torch.relu(p-hi[ids])
            endpoints = torch.tensor([samples[i]['origin']!='root' for i in ids])
            expected_positive = lo[ids] >= 0.
            result[name] = dict(samples=len(ids), interval_violation_mae_rad=float(violation.mean()),
                within_interval_fraction=float((violation <= 1e-6).float().mean()),
                bracket_sign_accuracy=float(((p[endpoints]>=0.)==expected_positive[endpoints]).float().mean()),
                active_gradient_norm_mean=float(gradient.norm(dim=1).mean()))
        with torch.no_grad():
            result['retention_validation_drift_mae_rad'] = float(
                (model(retention_validation)-retention_validation_targets).abs().mean())
        return result

    summary = dict(scope='development-only local-bracket fine-tune, not a repeat of the original 50k training',
        initial_checkpoint_sha256=original_sha, seed=0, device='cpu', steps=200, batch_size=64,
        learning_rate=3e-5, retention_weight=1., eikonal_weight=.01,
        mainline_checkpoint_changed=False, runtime_weights_promoted=False, baseline=evaluate(frozen), variants={})
    states = {'frozen':copy.deepcopy(frozen.state_dict())}
    for mode in ['legacy_point_targets', 'boundary_intervals']:
        model = copy.deepcopy(frozen).train().requires_grad_(True)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
        history = []
        started = time.monotonic()
        for step, (ids, replay_ids) in enumerate(zip(batches, retention_batches)):
            x = inputs[ids].clone().requires_grad_(True)
            p = model(x).gather(1, sid[ids,None]).flatten()
            supervised = ((p-legacy[ids]).square().mean() if mode=='legacy_point_targets'
                          else interval_loss(p, lo[ids], hi[ids]))
            gradient = torch.autograd.grad(p.sum(), x, create_graph=True)[0][:,3:]*masks[sid[ids]-1]
            eikonal = (gradient.norm(dim=1)-1.).square().mean()
            retention_loss = (model(retention[replay_ids])-retention_targets[replay_ids]).square().mean()
            loss = supervised+retention_loss+.01*eikonal
            optimizer.zero_grad();loss.backward();optimizer.step()
            if step % 20 == 0:
                history.append(dict(step=step, supervised=float(supervised), retention=float(retention_loss), eikonal=float(eikonal)))
        model.eval().requires_grad_(False)
        summary['variants'][mode] = dict(metrics=evaluate(model), history=history,
                                         elapsed_wall_s=time.monotonic()-started)
        states[mode] = copy.deepcopy(model.state_dict())
        torch.save(dict(format='care_boundary_interval_development_v1', source_checkpoint_sha256=original_sha,
                        model_state=states[mode], production_compatible=False), out/(mode+'.pt'))
    # Original fixed-budget geometry adjudication, with identical heads/seeds.
    spec = importlib.util.spec_from_file_location('immutable_network_study_common',
        ROOT/'outputs/phase_e_candidate_network_20261019/common.py')
    common = importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    runtime = common.runtime()
    rows = common.rows()
    for mode, state in states.items():
        runtime.model.full_model.load_state_dict(state, strict=True)
        runtime.model.eval().requires_grad_(False)
        results = []
        for row in rows:
            common.configure(runtime, row)
            threshold = row['attempt']['geometry']['minimum_conservative_g_required']
            branch = runtime._optimize_branch(torch.tensor(row['points'], dtype=torch.float32),
                torch.tensor([row['measured_seed']], dtype=torch.float32), row['attempt']['sensor_id'],
                geometry_guided=True, geometry_threshold=threshold)
            geometry = runtime._candidate_geometry(np.asarray(row['points']), branch['q_candidate'],
                row['attempt']['sensor_id'], minimum_conservative_g=threshold)
            results.append(dict(key=row['key'], cohort=row['cohort'], accepted=bool(geometry['accepted']),
                                reject_reason=geometry['reject_reason']))
        summary.setdefault('fixed_branch_replays', {})[mode] = results
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == original_sha
    (out/'result.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['fixed_branch_replays']}), flush=True)


if __name__ == '__main__':
    main()
