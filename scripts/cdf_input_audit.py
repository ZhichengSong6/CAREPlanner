"""Passive, bounded summaries of every CQ01 worker input (no ROS dependency)."""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

LOW = np.array([-.8, -.8, 0.], dtype=np.float64)
HIGH = np.array([.8, .8, 1.], dtype=np.float64)
TOL = 1e-7


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def exclusive_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, sort_keys=True)
        stream.write('\n')


def domain_summary(rows):
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != 10 or not len(rows):
        raise ValueError('Expected nonempty [B,10] worker rows')
    xyz = rows[:, :3].astype(np.float64)
    finite = np.isfinite(xyz).all(axis=1)
    qfinite = np.isfinite(rows[:, 3:]).all(axis=1)
    strict = finite & ((xyz < LOW) | (xyz > HIGH)).any(axis=1)
    outside = finite & ((xyz < LOW-TOL) | (xyz > HIGH+TOL)).any(axis=1)
    valid = xyz[finite]
    return dict(pairs=len(rows), nonfinite_xyz_rows=int((~finite).sum()),
                nonfinite_q_rows=int((~qfinite).sum()),
                strict_outside_rows=int(strict.sum()), outside_tolerance_rows=int(outside.sum()),
                xyz_min_m=valid.min(axis=0).tolist() if len(valid) else None,
                xyz_max_m=valid.max(axis=0).tolist() if len(valid) else None,
                outside_faces={axis+side: int((finite & (xyz[:, i] < LOW[i]-TOL if side == '_lower'
                                                     else xyz[:, i] > HIGH[i]+TOL)).sum())
                               for i, axis in enumerate('xyz') for side in ('_lower', '_upper')},
                outside_examples=[dict(row=int(i), point_m=xyz[i].tolist())
                                  for i in np.flatnonzero(outside)[:8]])


class InputAudit:
    def __init__(self, path, attestation_path):
        self.stream = Path(path).open('x', buffering=1)
        self.sequence = 0
        self.active = None
        self.started_pairs = self.finished_pairs = self.outside_pairs = self.failed = 0
        self.emit(dict(event='session', schema=1, pid=os.getpid(),
                       attestation_sha256=sha_file(attestation_path),
                       domain_min_m=LOW.tolist(), domain_max_m=HIGH.tolist(), tolerance_m=TOL,
                       scope='All received CQ01 batches; channel identity is absent from CQ01. Warmup excluded.'))

    def emit(self, value):
        self.stream.write(json.dumps(dict(wall_time_ns=time.time_ns(), **value),
                                     allow_nan=False, sort_keys=True, separators=(',', ':'))+'\n')
        self.stream.flush()

    def begin(self, rows, payload):
        if self.active is not None:
            raise RuntimeError('Unfinished previous request')
        t0 = time.perf_counter()
        summary = domain_summary(rows)
        self.sequence += 1
        self.active = self.sequence
        self.started_pairs += summary['pairs']
        self.outside_pairs += summary['outside_tolerance_rows']
        self.emit(dict(event='request', sequence=self.active,
                       payload_sha256=hashlib.sha256(payload).hexdigest(), **summary))
        return (time.perf_counter()-t0)*1000

    def finish(self, distance, gradient, pair_count, input_audit_ms, worker_total_ms):
        if self.active is None:
            raise RuntimeError('No active request')
        finite = bool(np.isfinite(distance).all() and np.isfinite(gradient).all())
        self.emit(dict(event='result', sequence=self.active, pairs=pair_count,
                       finite_output=finite, input_audit_ms=input_audit_ms,
                       worker_total_ms_before_result_log=worker_total_ms))
        self.finished_pairs += pair_count
        self.active = None

    def error(self, exc):
        self.failed += 1
        self.emit(dict(event='error', sequence=self.active, error_type=type(exc).__name__, error=str(exc)))
        self.active = None

    def close(self):
        self.emit(dict(event='close', requests=self.sequence, started_pairs=self.started_pairs,
                       finished_pairs=self.finished_pairs, outside_pairs=self.outside_pairs,
                       failed_requests=self.failed, unfinished_sequence=self.active))
        self.stream.close()


def attest_model(args, cdf, before_sha, worker_path, loader_path):
    after = sha_file(args.checkpoint)
    if before_sha != after or after != args.expected_checkpoint_sha256:
        raise ValueError('Checkpoint changed or differs from declared SHA256')
    if cdf.architecture['activation'] != args.activation:
        raise ValueError('Loaded activation differs from declaration')
    # Bind the actually loaded tensors as well as the checkpoint container.
    state = cdf.model.state_dict()
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        h.update(json.dumps([key, str(array.dtype), list(array.shape)], separators=(',', ':')).encode())
        h.update(array.tobytes())
    data = dict(schema=1, pid=os.getpid(), checkpoint=str(Path(args.checkpoint).resolve()),
                checkpoint_sha256=after, checkpoint_key=args.checkpoint_key,
                selected_checkpoint=cdf.selected_checkpoint, activation=cdf.architecture['activation'],
                loaded_tensor_sha256=h.hexdigest(), architecture=cdf.architecture,
                device=str(cdf.device), diagnostic_cpu=bool(getattr(args, 'diagnostic_cpu', False)),
                worker_sha256=sha_file(worker_path), loader_sha256=sha_file(loader_path),
                audit_helper_sha256=sha_file(__file__))
    exclusive_json(args.model_attestation, data)
    return data
