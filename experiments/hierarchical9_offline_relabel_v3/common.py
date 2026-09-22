"""Immutable paired-cache plumbing. No training, checkpoint loading or runtime writes."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import numpy as np
import scipy
import torch

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / 'hierarchical9_offline_relabel_v1'
V2 = HERE.parent / 'hierarchical9_offline_relabel_v2'
for folder in (V1, V2):
    sys.path.insert(0, str(folder))
import verified_core as core
import cache_io as old_io
from repo_oracle import RepoOracle, sha256_file
if Path(core.__file__).resolve() != V2 / 'verified_core.py':
    raise ImportError('Wrong v2 verified_core module')
FORMAT = 'careplanner_paired_offline_labels_v3'


def versions():
    return dict(python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, torch=torch.__version__)


def code_identity():
    files = list(HERE.glob('*.py')) + list(V1.glob('*.py')) + [V2/'verified_core.py']
    return {str(p.relative_to(HERE.parent)): sha256_file(p) for p in sorted(files)}


def fingerprint(value):
    return hashlib.sha256(json.dumps(old_io.clean_json(value), sort_keys=True,
                                     allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def source_bank(source):
    source, manifest, bank, queries = old_io.open_job(source)
    spec = old_io.read_json(source/'run_spec.json')
    if bank.manifest['source_sha256'] != manifest['original_data_sha256']:
        raise ValueError('Original dataset identity mismatch')
    return source, manifest, spec, bank, queries


def geometry_identity(identity):
    # An unrelated git commit is not a geometry change. Actual files MUST match.
    return {k: identity[k] for k in ('urdf_sha256', 'source_sha256')}


def check_output_location(out, source, bank):
    out, source, bank = [Path(p).resolve() for p in (out, source, bank)]
    for protected in (source, bank):
        if out == protected or out.is_relative_to(protected) or protected.is_relative_to(out):
            raise ValueError('Output must not overlap original smoke or original bank')


@contextmanager
def lock(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Already being written: {path}') from exc
        yield


def read_record(folder, spec_sha, task=None):
    folder = Path(folder)
    meta = old_io.read_json(folder/'complete.json')
    if meta['spec_sha256'] != spec_sha:
        raise ValueError(f'Different run spec: {folder}')
    if task is not None and meta['task'] != task:
        raise ValueError(f'Different query/sensor: {folder}')
    if sha256_file(folder/'result.json.gz') != meta['result_sha256']:
        raise ValueError(f'Corrupt record: {folder}')
    with gzip.open(folder/'result.json.gz', 'rt', encoding='utf-8') as f:
        result = json.load(f)
    if result['task'] != meta['task']:
        raise ValueError(f'Record identity mismatch: {folder}')
    return result, meta


def write_record(folder, result, spec_sha):
    """Atomic directory installation. Call under the per-task lock; never overwrite."""
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError(folder)
    folder.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.'+folder.name+'.', dir=folder.parent))
    try:
        with gzip.open(tmp/'result.json.gz', 'wt', encoding='utf-8') as f:
            json.dump(old_io.clean_json(result), f, ensure_ascii=False, allow_nan=False)
        old_io.write_json(tmp/'complete.json', dict(spec_sha256=spec_sha, task=result['task'],
            result_sha256=sha256_file(tmp/'result.json.gz')))
        os.rename(tmp, folder)
    finally:
        if tmp.exists(): shutil.rmtree(tmp)


def record_path(out, stage, qid, sensor):
    return Path(out)/stage/f'q{int(qid):07d}_s{int(sensor)}'


def open_run(out, check_code=True):
    out = Path(out).resolve(); spec = old_io.read_json(out/'batch_spec.json')
    if spec.get('format') != FORMAT:
        raise ValueError('Not a v3 paired-label run')
    if check_code and (spec['code_sha256'] != code_identity() or spec['versions'] != versions()):
        raise ValueError('Code/library version changed; do not mix results in this output')
    for name, digest in spec['frozen_files_sha256'].items():
        if sha256_file(out/name) != digest:
            raise ValueError(f'Frozen input changed: {name}')
    source, m, previous, bank, _ = source_bank(spec['source'])
    if sha256_file(source/'manifest.json') != spec['source_manifest_sha256']:
        raise ValueError('Source manifest changed')
    if sha256_file(source/'run_spec.json') != spec['source_run_spec_sha256']:
        raise ValueError('Source run specification changed')
    if geometry_identity(previous['geometry_identity']) != spec['geometry']:
        raise ValueError('Source geometry changed')
    q = load_npz(out/'queries.npz')
    split = load_npz(out/'spatial_splits.npz')
    tr, va = set(split['train_x_indices'].tolist()), set(split['val_x_indices'].tolist())
    if tr & va or len(set(q['query_id'].tolist())) != len(q['query_id']):
        raise ValueError('Split leakage or duplicate query id')
    for xi, s in zip(q['x_index'], q['split']):
        if int(s) not in (0, 1) or int(xi) not in (tr if s == 0 else va):
            raise ValueError('Query escaped original spatial split')
    if not np.isfinite(q['q_query']).all() or np.any(q['q_query']<bank.lo) or np.any(q['q_query']>bank.hi):
        raise ValueError('Invalid stored query')
    return out, spec, bank, q, sha256_file(out/'batch_spec.json')


def reference_gate(oracle, x, s, q, result, cfg):
    # Same FP32 check used in the accepted v2 review; no runtime margin changes.
    if result['selected'] is None: return result
    gm = float(oracle.reference_margins(x, np.asarray(result['q_star'])[None])[0, s])
    gq = float(oracle.reference_margins(x, np.asarray(q)[None])[0, s])
    ok = abs(gm)<=cfg.boundary_tol_m and (gq>=0)==(result['query_g_m']>=0) and (abs(gq)>cfg.sign_guard_m or result['value']==0)
    result['reference_check'] = dict(boundary_g_m=gm, query_g_m=gq, passed=bool(ok))
    if not ok:
        result['value_valid'] = result['grad_valid'] = False
        result['gradient_reasons'].append('UPSTREAM_GEOMETRY_OR_SIGN_MISMATCH')
    return result


def distribution(values):
    a = np.asarray(values, float); a = a[np.isfinite(a)]
    return dict(count=len(a), median=float(np.median(a)) if len(a) else None,
                p95=float(np.quantile(a,.95)) if len(a) else None,
                total=float(a.sum()), maximum=float(a.max()) if len(a) else None)
