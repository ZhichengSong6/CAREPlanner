#!/usr/bin/env python3
"""Generate immutable TRAIN-only calibration anchors and separate held-out monitoring anchors."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch

from common import (REPO, AUDIT, SCRIPTS, V1_REL, V1_SHA, DATA_REL, URDF, CACHE_FORMAT,
                    load_v1, module, setup_paths, source_fingerprints, sha256, write_json,
                    validate_array_split)


def build_split(dataset, oracle, split, points_per_sensor, anchors_per_point, seed, device, cfg, log_file):
    from core import refine_boundary, qualify_boundary, tangent_seed, bank_label
    lo, hi = dataset.q_limits(device)
    masks = dataset.sensor_masks(device)
    pool = dataset.train_indices_np if split == 'train' else dataset.val_indices_np
    records, attempts = [], Counter()
    for s in range(8):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0 if split == 'train' else 1, s]))
        supported = np.asarray([i for i in pool if bool(dataset.valid_cpu[int(i), :, s].any())])
        chosen = rng.choice(supported, min(points_per_sensor, len(supported)), replace=False)
        for position, index in enumerate(chosen):
            index = int(index)
            x = dataset.x_cpu[index].to(device)
            slots = torch.where(dataset.valid_cpu[index, :, s])[0].numpy()
            bank = dataset.qlib_cpu[index, slots, :, s].to(device)
            if not torch.isfinite(bank).all():
                raise ValueError('Nonfinite q0 marked valid')
            selected = rng.choice(len(slots), min(anchors_per_point, len(slots)), replace=False)
            for k in selected:
                identity = {'split': split, 's': s, 'x_index': index, 'source_slot': int(slots[k])}
                refined = refine_boundary(lambda q: oracle.value_grad(x, q, s), bank[k], masks[s], lo, hi, cfg)
                # Always qualify the result. A failed refinement is not a zero-label anchor.
                for kind in (0, 1):
                    if kind == 1:
                        if not bank_qualified:
                            break
                        qseed = tangent_seed(bank_q, bank_normal, masks[s], rng, cfg.tangent_radius_rad)
                        if qseed is None:
                            attempts[f'S{s}/offbank/degenerate_tangent'] += 1
                            log_file.write(json.dumps({**identity, 'kind': kind, 'accepted': False, 'reason': 'degenerate_tangent'})+'\n')
                            continue
                        refined = refine_boundary(lambda q: oracle.value_grad(x, q, s), qseed, masks[s], lo, hi, cfg)
                    q = refined['q']
                    qual = qualify_boundary(oracle, x, q, s, masks[s], lo, hi, cfg)
                    gap = bank_label(q, bank, masks[s], 1)['distance_rad']
                    reasons = list(qual['reasons'])
                    if not refined['accepted']:
                        reasons.append('refinement:' + refined['reason'])
                    if kind == 1 and gap < cfg.offbank_min_rad:
                        reasons.append('offbank_too_close_to_bank')
                    accepted = not reasons
                    origin = 'bank' if kind == 0 else 'offbank'
                    attempts[f'S{s}/{origin}/attempted'] += 1
                    attempts[f'S{s}/{origin}/accepted'] += int(accepted)
                    for reason in reasons:
                        attempts[f'S{s}/{origin}/reject:{reason}'] += 1
                    log_file.write(json.dumps({**identity, 'kind': kind, 'accepted': accepted,
                        'reasons': reasons, 'g_m': qual['g_m'], 'bank_gap_rad': gap,
                        'q': q.tolist()}, allow_nan=False)+'\n')
                    if kind == 0:
                        bank_qualified, bank_q, bank_normal = accepted, q.clone(), qual['normal']
                    if accepted:
                        records.append({'x': x.cpu().numpy(), 'q': q.cpu().numpy(),
                            'normal': qual['normal'].cpu().numpy(), 's': s, 'kind': kind,
                            'x_index': index, 'g': qual['g_m'], 'plane': qual['active_plane'],
                            'bank_gap': gap, 'source_slot': int(slots[k])})
            if (position + 1) % 16 == 0 or position + 1 == len(chosen):
                log_file.flush()
                print(f'[cache] {split} S{s} x={position+1}/{len(chosen)} accepted_total={len(records)}', flush=True)
    if not records:
        raise ValueError('No qualified anchors')
    integer = {'s', 'kind', 'x_index', 'plane', 'source_slot'}
    arrays = {k: np.asarray([r[k] for r in records], dtype=np.int64 if k in integer else np.float32)
              for k in records[0]}
    counts = validate_array_split(arrays, pool, masks.cpu().numpy(), lo.cpu().numpy(), hi.cpu().numpy())
    return arrays, {'attempts': dict(attempts), 'stratum_counts': counts.tolist(),
                    'distinct_spatial_points': int(len(np.unique(arrays['x_index'])))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train-points', type=int, default=512)
    p.add_argument('--val-points', type=int, default=64)
    p.add_argument('--train-anchors', type=int, default=4)
    p.add_argument('--val-anchors', type=int, default=2)
    p.add_argument('--seed', type=int, default=314159)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = p.parse_args()
    if min(args.train_points, args.val_points, args.train_anchors, args.val_anchors) < 1 or args.seed < 0:
        p.error('Sample counts must be positive, seed nonnegative')
    setup_paths()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('No GPU allocated')
    device = torch.device(args.device)
    root, out = args.artifact_root.resolve(), args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite cache: {out}')
    _, ckpt = load_v1(root / V1_REL)
    data_path = root / DATA_REL
    if sha256(URDF) != ckpt['metadata']['urdf_sha256'] or data_path.stat().st_size != ckpt['metadata']['data_bytes']:
        raise ValueError('Geometry/dataset metadata differs from V1')
    api = module('train_signed_visibility_cdf_pairwise_replace', SCRIPTS)
    dataset = api.VisibilityQ0Dataset(str(data_path), val_count=1000, seed=0)
    oracle = module('oracle', AUDIT).SensorOracle(URDF, device, api.DEFAULT_JOINT_NAMES, api.DEFAULT_SENSOR_FRAMES)
    checks = module('audit', AUDIT).preflight(dataset, oracle, {}, device)
    print('[preflight]', json.dumps(checks), flush=True)
    cfg = module('core', AUDIT).Config()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'format': CACHE_FORMAT, 'status': 'RUNNING', 'parent_sha256': V1_SHA,
        'generation_seed': args.seed, 'geometry_config': asdict(cfg), 'urdf_sha256': sha256(URDF),
        'sensor_masks': dataset.sensor_masks_cpu.tolist(), 'q_min': dataset.q_min_cpu.tolist(),
        'q_max': dataset.q_max_cpu.tolist(), 'source_sha256': source_fingerprints(), 'preflight': checks,
        'data_identity': {'path': str(data_path), 'bytes': data_path.stat().st_size,
                          'mtime_ns': data_path.stat().st_mtime_ns, 'content_hash': 'NOT_COMPUTED'},
        'split': {'seed': 0, 'train_indices': dataset.train_indices_np.tolist(),
                  'val_indices': dataset.val_indices_np.tolist(), 'independent_final_test': False},
        'args': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        'semantics': 'Only sensor s has zero and unit-normal labels at (x,q,s); no union-zero label. '
                     'Tangent-refined roots are not nearest-boundary solutions. Val anchors never train.', 'files': {}}
    write_json(out / 'manifest.json', manifest)
    start = time.perf_counter()
    try:
        for split, points, anchors in (('train', args.train_points, args.train_anchors), ('val', args.val_points, args.val_anchors)):
            with (out / (split + '_generation.jsonl')).open('w') as log:
                arrays, info = build_split(dataset, oracle, split, points, anchors, args.seed, device, cfg, log)
            np.savez_compressed(out / (split + '.npz'), **arrays)
            manifest[split] = info
            manifest['files'][split + '.npz'] = sha256(out / (split + '.npz'))
        manifest.update(status='COMPLETE', elapsed_seconds=time.perf_counter()-start)
        write_json(out / 'manifest.json', manifest)
    except Exception as exc:
        manifest.update(status='FAILED', error=repr(exc))
        write_json(out / 'manifest.json', manifest)
        raise
    print(f'[done] calibration_cache_complete {out}', flush=True)


if __name__ == '__main__':
    main()
