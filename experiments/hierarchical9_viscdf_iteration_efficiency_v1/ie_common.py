"""Input/source guards and immutable result stages for the iteration experiment."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ('R1', 'B_keep')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    with Path(path).open() as f:
        for number, text in enumerate(f, 1):
            if text.strip():
                try:
                    yield json.loads(text)
                except ValueError as exc:
                    raise ValueError(f'Malformed JSONL {path}:{number}') from exc


def write(path, value):
    """Publish complete JSON once, without overwriting any previous result."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + f'.tmp.{os.getpid()}')
    with temp.open('x') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    try:
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def finish(folder, **metadata):
    folder = Path(folder)
    files = {str(p.relative_to(folder)): sha(p) for p in folder.rglob('*')
             if p.is_file() and not p.name.startswith('.') and p.name != 'complete.json'}
    write(folder / 'complete.json', dict(status='COMPLETE', files=files, **metadata))


def stage(folder):
    folder = Path(folder).resolve(); result = read(folder / 'complete.json')
    require(result.get('status') == 'COMPLETE', f'Incomplete stage: {folder}')
    for rel, digest in result['files'].items():
        f = (folder / rel).resolve()
        require(f.is_relative_to(folder), f'Unsafe stage path: {rel}')
        require(sha(f) == digest, f'Corrupt stage: {f}')
    return result


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f'Cannot load {path}')
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def code_hashes():
    return {p.name: sha(p) for p in HERE.iterdir() if p.suffix in ('.py', '.sh', '.json')}


def verify(job):
    require(job['cfg'] == read(HERE / 'protocol.json'), 'Protocol changed')
    require(job['code_hashes'] == code_hashes(), 'Frozen evaluation source changed')
    out = Path(job['out'])
    for rel, digest in job['dep_hashes'].items():
        require(sha(out / 'dependencies' / rel) == digest, f'Dependency changed: {rel}')
    for rel, digest in job['frozen_hashes'].items():
        require(sha(out / rel) == digest, f'Frozen input changed: {rel}')
    for name in ARMS:
        require(sha(job['models'][name]['path']) == job['models'][name]['sha256'], f'Weight changed: {name}')


def model_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def validate_starts(starts, cfg, lo, hi):
    require(len(starts) == cfg['case_count'], 'Wrong case count')
    require([r['case_id'] for r in starts] == list(range(len(starts))), 'Duplicate/missing case IDs')
    points = {}
    for row in starts:
        pid, s = row['point_id'], row['sensor']
        require(type(pid) is int and 0 <= pid < cfg['point_count'], 'Invalid point ID')
        require(type(s) is int and 0 <= s < 8, 'Invalid sensor ID')
        require(row['cohort'] in ('local', 'uniform'), 'Unknown cohort')
        x = np.asarray(row['x']); q = np.asarray(row['q_init']); w = np.asarray(row['witness_q'])
        require(x.shape == (3,) and q.shape == w.shape == (7,), 'Bad input dimensions')
        require(np.isfinite(x).all() and np.isfinite(q).all() and np.isfinite(w).all(), 'Nonfinite input')
        require((q >= lo).all() and (q <= hi).all() and (w >= lo).all() and (w <= hi).all(), 'Input violates limits')
        require(row['initial_g_m'] < 0 and row['witness_g_m'] > 0, 'Expected outside start and positive witness')
        points.setdefault(pid, row['x'])
        require(points[pid] == row['x'], 'Same point ID has different position')
    require(len(points) == cfg['point_count'], 'Incomplete point coverage')


def load_runtime(job, device):
    import torch
    deps = Path(job['out']) / 'dependencies'
    scripts = str(deps / 'src/care_visibility_cdf/scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    arch = load('ie_original_arch', deps / 'experiments/hierarchical9_scratch50k_r012_v1/model.py')
    pa = load('ie_legacy_probe', deps / 'experiments/hierarchical9_boundary_audit_v1/runtime_probe.py')
    oa = load('ie_oracle', deps / 'experiments/hierarchical9_boundary_audit_v1/oracle.py')
    from train_signed_visibility_cdf_pairwise_replace import DEFAULT_JOINT_NAMES, DEFAULT_SENSOR_FRAMES
    from hierarchical_visibility_cdf_model import HierarchicalSensorView
    oracle = oa.SensorOracle(deps / 'src/arm_description/urdf/Arm.urdf', device,
                             DEFAULT_JOINT_NAMES, DEFAULT_SENSOR_FRAMES)
    models, probes = {}, {}
    lo, hi, masks = (torch.tensor(job[k], dtype=torch.float32, device=device) for k in ('lo', 'hi', 'masks'))
    for name in ARMS:
        spec = job['models'][name]
        require(sha(spec['path']) == spec['sha256'], 'Weight hash mismatch')
        cp = torch.load(spec['path'], map_location='cpu', weights_only=False)
        require(cp.get('arm') == 'R1', 'Wrong architecture arm')
        if name == 'R1':
            require(cp.get('step') == 50000 and cp.get('initialization') == 'random_from_scratch', 'Wrong R1 lineage')
        else:
            require(cp.get('variant') == 'B_keep' and cp.get('stage_step') == 20000 and cp.get('step') == 70000
                    and cp.get('parent_sha256') == job['cfg']['parent_sha256']
                    and cp.get('initialization') == 'finetune_from_solver_goal_50000', 'Wrong B lineage')
        model = arch.build_model('R1')
        model.load_state_dict(cp['model_state'], strict=True)
        model.to(device).eval().requires_grad_(False)
        models[name] = model; probes[name] = pa.make_probe(HierarchicalSensorView(model), masks, lo, hi)
        p = probes[name]
        require((p.projection_iters, p.projection_damping, p.projection_epsilon_f,
                 p.root_refine_iters, p.branch_ascent_steps, p.branch_fallback_ascent_steps)
                == (10, .5, .03, 12, 1, 8), 'Legacy solver configuration changed')
    return models, probes, oracle
