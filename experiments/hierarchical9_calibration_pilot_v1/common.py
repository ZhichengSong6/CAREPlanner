"""Pinned-model I/O, immutable boundary cache, and explicit repository imports."""
from __future__ import annotations
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRATCH = REPO / 'experiments/hierarchical9_scratch_v1'
AUDIT = REPO / 'experiments/hierarchical9_boundary_audit_v1'
SCRIPTS = REPO / 'src/care_visibility_cdf/scripts'
V1_SHA = '979552db20bc7e20775758b273613532921c5dbf11c480b13597127683c4c199'
V1_REL = 'src/care_visibility_cdf/checkpoints/hierarchical9_scratch_seed0/final.pt'
DATA_REL = 'src/care_visibility_cdf/data/visibility_yiming_style_grid30_q20000_k500_fovonly.npz'
URDF = REPO / 'src/arm_description/urdf/Arm.urdf'
CACHE_FORMAT = 'care_h9_calibration_cache_v1'
PILOT_FORMAT = 'care_h9_boundary_calibration_pilot_v1'


def module(name: str, directory: Path):
    """Avoid silently importing a same-named module from a different worktree."""
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    result = importlib.import_module(name)
    if Path(result.__file__).resolve() != (directory / (name + '.py')).resolve():
        raise ImportError(f'Wrong {name} module: {result.__file__}')
    return result


def setup_paths():
    for directory in (SCRIPTS, SCRATCH, AUDIT):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def array_hash(a) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str((a.dtype.str, a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def write_json(path: Path, value):
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def load_v1(path: Path, device='cpu', trainable=False):
    path = Path(path).resolve()
    if path.name != 'final.pt' or sha256(path) != V1_SHA:
        raise ValueError('Expected the fixed evaluated V1 final.pt; refusing another checkpoint')
    # Hash is verified BEFORE unpickling this locally trusted training artifact.
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if ckpt.get('format') != 'careplanner_hierarchical9_scratch_v1' or ckpt.get('step') != 50000:
        raise ValueError('Wrong V1 format or update count')
    if ckpt.get('out_dim') != 9 or ckpt.get('initialization') != 'random_from_scratch':
        raise ValueError('Wrong V1 architecture/initialization')
    for key, expected in {'seed': 0, 'val_count': 1000, 'horizontal_fov_deg': 50.,
                           'vertical_fov_deg': 66., 'z_min': .2, 'z_max': .7, 'delta': .01}.items():
        if ckpt['args'].get(key) != expected:
            raise ValueError(f'Wrong V1 scientific definition: {key}')
    model = module('model', SCRATCH).HierarchicalVisibilityCDF()
    model.load_state_dict(ckpt['model_state'], strict=True)
    model.to(device=device, dtype=torch.float32).train(trainable).requires_grad_(trainable)
    return model, ckpt


def source_fingerprints():
    paths = sorted(HERE.glob('*.py'))
    paths += [SCRATCH / n for n in ('model.py', 'objective.py', 'train.py', 'evaluate.py')]
    paths += [AUDIT / n for n in ('core.py', 'oracle.py', 'runtime_probe.py', 'audit.py')]
    paths += [SCRIPTS / n for n in ('per_sensor_visibility_runtime.py',
        'train_signed_visibility_cdf_pairwise_replace.py', 'train_per_sensor_visibility_cdf.py',
        'train_per_sensor_visibility_cdf_ddp.py', 'extract_visibility_zero_level_sets.py',
        'compare_scalar_vs_per_sensor_apples_to_apples.py')]
    return {str(p.relative_to(REPO)): sha256(p) for p in paths if p.is_file()}


def validate_array_split(data: dict, allowed_indices, masks, lo, hi):
    """Validate cache membership/geometry metadata; normals are full original-q 7D."""
    n = len(data['s'])
    expected = {'x': (n, 3), 'q': (n, 7), 'normal': (n, 7), 's': (n,), 'kind': (n,),
                'x_index': (n,), 'g': (n,), 'plane': (n,), 'bank_gap': (n,)}
    for key, shape in expected.items():
        if key not in data or data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f'Invalid boundary array {key}, expected {shape}')
    if not n or not np.isin(data['x_index'], allowed_indices).all():
        raise ValueError('Empty cache or train/validation spatial leakage')
    for key in ('s', 'kind', 'x_index', 'plane'):
        if not np.issubdtype(data[key].dtype, np.integer):
            raise ValueError(f'{key} must be integer')
    if not np.isin(data['s'], np.arange(8)).all() or not np.isin(data['kind'], [0, 1]).all():
        raise ValueError('Invalid sensor/origin mapping')
    if not np.isin(data['plane'], np.arange(6)).all():
        raise ValueError('Invalid FOV plane')
    if not ((data['q'] >= lo).all() and (data['q'] <= hi).all()):
        raise ValueError('Boundary q outside unchanged joint limits')
    if np.max(np.abs(data['g'])) > 1.0001e-5:
        raise ValueError('Unqualified boundary residual')
    if not np.allclose(np.linalg.norm(data['normal'], axis=1), 1, atol=2e-5):
        raise ValueError('Boundary normals are not unit vectors')
    if np.max(np.abs(data['normal'] * (1 - masks[data['s']]))) > 1e-6:
        raise ValueError('Normal contains inactive-joint components')
    group = 2 * data['s'] + data['kind']
    counts = np.bincount(group, minlength=16)
    if np.any(counts == 0):
        raise ValueError(f'Missing sensor/origin strata; no resampling to hide failure: {counts.tolist()}')
    return counts


class BoundaryCache:
    def __init__(self, directory: Path):
        self.path = Path(directory).resolve()
        self.manifest = json.loads((self.path / 'manifest.json').read_text())
        m = self.manifest
        if m.get('format') != CACHE_FORMAT or m.get('status') != 'COMPLETE' or m.get('parent_sha256') != V1_SHA:
            raise ValueError('Not a complete V1 calibration cache')
        self.identity = sha256(self.path / 'manifest.json')
        self.masks = np.asarray(m['sensor_masks'], dtype=np.float32)
        self.lo = np.asarray(m['q_min'], dtype=np.float32)
        self.hi = np.asarray(m['q_max'], dtype=np.float32)
        if self.masks.shape != (8, 7) or not np.isin(self.masks, [0, 1]).all():
            raise ValueError('Invalid sensor masks')
        if np.any(self.masks.sum(1) == 0) or self.lo.shape != (7,) or self.hi.shape != (7,) or not (self.lo < self.hi).all():
            raise ValueError('Invalid masks/joint limits')
        train_ids, val_ids = m['split']['train_indices'], m['split']['val_indices']
        if set(train_ids) & set(val_ids) or len(val_ids) != 1000:
            raise ValueError('Invalid canonical spatial split')
        self.arrays, self.groups = {}, {}
        for split, pool in (('train', train_ids), ('val', val_ids)):
            file = self.path / (split + '.npz')
            if sha256(file) != m['files'][file.name]:
                raise ValueError(f'Cache checksum mismatch: {file}')
            with np.load(file, allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
            validate_array_split(data, pool, self.masks, self.lo, self.hi)
            self.arrays[split] = data
            group = 2 * data['s'] + data['kind']
            self.groups[split] = [np.flatnonzero(group == k) for k in range(16)]

    def verify_dataset(self, dataset, path: Path):
        m = self.manifest
        for key, ids in (('train_indices', dataset.train_indices_np), ('val_indices', dataset.val_indices_np)):
            if not np.array_equal(np.asarray(m['split'][key]), ids):
                raise ValueError('Cache does not match original seed-0 spatial split')
        if Path(path).stat().st_size != m['data_identity']['bytes']:
            raise ValueError('Dataset size changed (size is NOT a content hash)')
        if not np.array_equal(dataset.sensor_masks_cpu.numpy(), self.masks):
            raise ValueError('Sensor masks changed')
        if not np.array_equal(dataset.q_min_cpu.numpy(), self.lo) or not np.array_equal(dataset.q_max_cpu.numpy(), self.hi):
            raise ValueError('Joint limits changed')
        for data in self.arrays.values():
            if not np.array_equal(dataset.x_cpu[data['x_index']].numpy(), data['x']):
                raise ValueError('Cache (x_index,x) does not match dataset')

    def draw_indices(self, seed: int, step: int, count: int, rank=0, world=1):
        if count < 16 or count % 16 or count % world or not 0 <= rank < world:
            raise ValueError('Boundary count must be divisible by 16 and world size')
        # Independent of Torch's uniform-query RNG and of model/AMP behavior.
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, 91021]))
        indices = np.concatenate([rng.choice(g, count // 16, replace=True) for g in self.groups['train']])
        rng.shuffle(indices)
        return indices[rank * (count // world):(rank + 1) * (count // world)]

    def tensors(self, split, indices, device):
        d = self.arrays[split]
        x = torch.as_tensor(d['x'][indices], device=device, dtype=torch.float32)
        q = torch.as_tensor(d['q'][indices], device=device, dtype=torch.float32)
        s = torch.as_tensor(d['s'][indices], device=device, dtype=torch.long)
        normals = torch.as_tensor(d['normal'][indices], device=device, dtype=torch.float32)
        group = 2 * s + torch.as_tensor(d['kind'][indices], device=device, dtype=torch.long)
        return torch.cat((x, q), 1), s, normals, group


def seed_for_update(seed, step):
    # Explicit successful-update stream: P0/P1 do not depend on validation or AMP retries.
    return int(np.random.SeedSequence([seed, step, 411]).generate_state(1)[0])
