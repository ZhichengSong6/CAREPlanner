"""Explicit P2 lineage; old P0/P1 code and artifacts remain read-only."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LEGACY = REPO / 'experiments/hierarchical9_calibration_pilot_v1'
sys.path.insert(0, str(LEGACY))
import common as old
if Path(old.__file__).resolve() != LEGACY / 'common.py':
    raise ImportError('Wrong legacy common module')
old.setup_paths()
calibration = old.module('calibration', LEGACY)
FORMAT = 'care_h9_p2_no_boundary_eikonal_v1'
BASE_COMMIT = '0c50a7998a32c9484f78681731da4239f5c10f6c'
WEIGHTS = {'zero': 5.0, 'normal': 0.1, 'eikonal': 0.0}
CONTROL_WEIGHTS = {'zero': 5.0, 'normal': 0.1, 'eikonal': 0.1}
FIXED = dict(steps=2000, seed=0, lr=1e-4, global_batch_x=4000, batch_q=100,
             microbatch_x=250, decode_x_chunk=64, boundary_count=4096,
             boundary_microbatch=256, boundary_warmup=500, val_global_batch_x=512,
             val_batch_q=100, val_microbatch_x=128, val_every=500, log_every=100,
             amp='fp16', max_amp_retries=16)


def weights():
    return calibration.BoundaryWeights(**WEIGHTS)


def alpha(step: int, warmup: int = 500) -> float:
    if step < 0 or warmup <= 0:
        raise ValueError('Invalid P2 update/warmup')
    return min(step / warmup, 1.0)


def fingerprints():
    paths = sorted(p for p in HERE.iterdir() if p.suffix in ('.py', '.sh', '.sbatch'))
    return {str(p.relative_to(REPO)): old.sha256(p) for p in paths}


def load_saved(path: Path):
    """Only locally trusted project checkpoints; hash-before-unpickle is integrity, not trust."""
    path = Path(path).resolve()
    run = json.loads((path.parent / 'run.json').read_text())
    digest = old.sha256(path)
    if path.name != 'final.pt' or run.get('status') != 'COMPLETE' or run.get('final_sha256') != digest:
        raise ValueError(f'Incomplete/changed checkpoint: {path}')
    return torch.load(path, map_location='cpu', weights_only=False), digest


def validate_controls(c0: dict, c1: dict, cache_identity: str, current_sources: dict):
    for arm, cp in (('P0', c0), ('P1', c1)):
        expected = dict(format=old.PILOT_FORMAT, arm=arm, completed=True, parent_updates=50000,
                        parent_sha256=old.V1_SHA, pilot_updates=2000, total_updates=52000,
                        cache_manifest_sha256=cache_identity, out_dim=9, frozen_parameters=0,
                        initialization='V1_weights_only_fresh_Adam',
                        optimizer_policy='fresh_Adam_constant_lr_no_scheduler_no_clipping')
        for key, value in expected.items():
            if cp.get(key) != value:
                raise ValueError(f'{arm} reference mismatch: {key}')
        if cp.get('source_sha256') != current_sources:
            raise ValueError(f'{arm}: reused engine/model/solver sources changed; do not bypass this gate')
        if cp.get('boundary_weights') != CONTROL_WEIGHTS:
            raise ValueError(f'{arm}: wrong reference boundary weights')
        for key, value in FIXED.items():
            if cp['args'].get(key) != value:
                raise ValueError(f'{arm}: fixed reference config {key} differs')
        if cp['args'].get('arm') != arm:
            raise ValueError('Control arg/arm mismatch')
        if len(cp.get('sample_stream_sha256_by_rank', [])) != 4:
            raise ValueError('Expected four rank-specific sample hashes')
    if c0['sample_stream_sha256_by_rank'] != c1['sample_stream_sha256_by_rank']:
        raise ValueError('Original P0/P1 streams do not match')
    for key in ('architecture', 'output_layout'):
        if c0[key] != c1[key]:
            raise ValueError(f'Control {key} mismatch')
    if set(c0['args']) != set(c1['args']):
        raise ValueError('Control argument keys differ')
    for key in c0['args']:
        if key not in ('arm', 'output') and c0['args'][key] != c1['args'][key]:
            raise ValueError(f'Control arguments differ: {key}')


def load_references(root: Path):
    root = Path(root).resolve()
    cache = old.BoundaryCache(root / 'cache')
    c0, h0 = load_saved(root / 'P0/final.pt')
    c1, h1 = load_saved(root / 'P1/final.pt')
    validate_controls(c0, c1, cache.identity, old.source_fingerprints())
    if Path(c0['args']['cache']).resolve() != cache.path:
        raise ValueError('Use the original immutable reference cache path')
    return cache, c0, c1, {'P0': h0, 'P1': h1}


def make_args(control: dict, output: Path, mode: str):
    if mode not in ('smoke', 'pilot'):
        raise ValueError(mode)
    args = dict(control['args'])
    args.update(arm='P2', output=str(Path(output).resolve()), steps=2 if mode == 'smoke' else 2000)
    return argparse.Namespace(**args)


def assert_p2(cp: dict, control: dict, cache_identity: str, references: dict,
              *, require_pilot: bool):
    expected = dict(format=FORMAT, arm='P2', completed=True, parent_sha256=old.V1_SHA,
                    parent_updates=50000, cache_manifest_sha256=cache_identity,
                    boundary_weights=WEIGHTS, reference_sha256=references, out_dim=9,
                    frozen_parameters=0, initialization='V1_weights_only_fresh_Adam',
                    optimizer_policy='fresh_Adam_constant_lr_no_scheduler_no_clipping')
    for key, value in expected.items():
        if cp.get(key) != value:
            raise ValueError(f'P2 identity mismatch: {key}')
    if cp.get('mode') not in ('smoke', 'pilot'):
        raise ValueError('Missing P2 mode')
    steps = 2 if cp['mode'] == 'smoke' else 2000
    if require_pilot and steps != 2000:
        raise ValueError('Smoke cannot be compared as a 2000-step P2')
    if cp.get('pilot_updates') != steps or cp.get('total_updates') != 50000 + steps:
        raise ValueError('P2 successful-update count mismatch')
    expected_args = vars(make_args(control, Path(cp['args']['output']), cp['mode']))
    if cp['args'] != expected_args:
        raise ValueError('P2 scientific config differs beyond arm/output/smoke steps')
    for key in ('source_sha256', 'architecture', 'output_layout'):
        if cp.get(key) != control[key]:
            raise ValueError(f'P2 unchanged reference metadata differs: {key}')
    if cp.get('p2_source_sha256') != fingerprints():
        raise ValueError('P2 extension sources changed since training')
    if len(cp.get('sample_stream_sha256_by_rank', [])) != 4:
        raise ValueError('Missing P2 per-rank streams')
    if steps == 2000:
        if cp['sample_stream_sha256_by_rank'] != control['sample_stream_sha256_by_rank']:
            raise ValueError('P2 and P0/P1 sample streams differ')
        if cp.get('pair_sample_streams') != 'MATCH':
            raise ValueError('P2 full stream verification missing')
    elif cp.get('pair_sample_streams') != 'NOT_COMPARABLE_2_VS_2000':
        raise ValueError('Smoke must not claim a full 2000-step stream match')


def model_from(cp, device):
    model = old.module('model', old.SCRATCH).HierarchicalVisibilityCDF()
    model.load_state_dict(cp['model_state'], strict=True)
    return model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
