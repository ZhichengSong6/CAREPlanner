#!/usr/bin/env python3
"""Bounded asynchronous diagnostic sink. No ROS publishers or safety decisions."""
from collections import OrderedDict
import copy
import hashlib
import json
from pathlib import Path
import queue
import threading
import time


def ros_value(value):
    # genpy decodes uint8[] (including empty source_type) as bytes, not list.
    if isinstance(value, (bytes, bytearray)):
        return list(value)
    if hasattr(value, 'to_nsec'):
        return str(value.to_nsec())
    if hasattr(value, '__slots__'):
        return {s: ros_value(getattr(value, s)) for s in value.__slots__}
    if isinstance(value, (list, tuple)):
        return [ros_value(v) for v in value]
    return value


def trajectory_dict(msg):
    return dict(stamp_ns=str(msg.header.stamp.to_nsec()), frame_id=msg.header.frame_id,
        joint_names=list(msg.joint_names), points=[dict(time_ns=str(p.time_from_start.to_nsec()),
            positions=list(p.positions), velocities=list(p.velocities),
            accelerations=list(p.accelerations), effort=list(p.effort)) for p in msg.points])


def failure_signature(record, context):
    gate = record.get('safety_gate', 'vbc')
    evidence = record['vbc_evidence'] if gate == 'vbc' else record['gcdf_evidence']
    # Transaction stamps differ on each retry. Keep distinct geometry/link/
    # deadline signatures, but coalesce repeated candidates in the same episode.
    points = sorted(json.dumps({k: v.get(k) for k in [
        'point', 'sweeping_link', 'source_collision_index', 'sweep_eval_k', 'latest_see_s',
        'source_type', 'timestep']},
        sort_keys=True) for v in evidence['violations'])
    identity = [context.get(k) for k in ['observation_token', 'mode_epoch', 'target_revision', 'progress_epoch']]
    # Missing identity must never be accidentally joined/coalesced with another target.
    if not context:
        identity = [record['raw_candidate_stamp_ns']]
    return hashlib.sha256(json.dumps([gate, identity, points], sort_keys=True).encode()).hexdigest()


def gcdf_evidence(batch, trajectory, margin):
    """Exact final batch rows, not evidence of physical occupied matter.

    Keep a malformed rejection explicit; never invent point/q/gradient values.
    Geometry/map replay requires additional inputs and is not implied here.
    """
    rows = []
    source = ros_value(batch.source_type)
    for i, d in enumerate(batch.distance):
        if d >= margin:
            continue
        k = int(batch.original_timestep[i]) if i < len(batch.original_timestep) else None
        rows.append(dict(pair=i, timestep=k,
            point=list(batch.point_flat[3*i:3*i+3]),
            source_type=int(source[i]) if i < len(source) else None,
            distance=float(d), gradient=list(batch.gradient_flat[7*i:7*i+7]),
            q_linearization=list(batch.q_linearization_flat[7*i:7*i+7]),
            candidate_q=(list(trajectory.points[k].positions)
                         if k is not None and 0 <= k < len(trajectory.points) else None),
            time_ns=(str(trajectory.points[k].time_from_start.to_nsec())
                     if k is not None and 0 <= k < len(trajectory.points) else None)))
    return dict(schema=1, trajectory_stamp_ns=str(trajectory.header.stamp.to_nsec()),
        batch_stamp_ns=str(batch.header.stamp.to_nsec()), safety_margin=margin,
        violations=rows, semantics='UNKNOWN is not occupied collision evidence')


class RejectionSnapshotSink:
    max_record_bytes = 8 * 1024 * 1024

    def __init__(self, directory, max_files=16, max_bytes=32*1024*1024, warn=lambda s: None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.max_files, self.max_bytes = max_files, max_bytes
        self.warn = warn
        self.contexts = OrderedDict()
        self.lock = threading.Lock()
        self.jobs = queue.Queue(maxsize=4)
        self.stop = threading.Event()
        self.seen = set()
        self.written = self.bytes_written = self.coalesced = self.dropped = self.errors = 0
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def add_context(self, text):
        if len(text) > 2*1024*1024:
            return
        try:
            context = json.loads(text)
            stamp = str(int(context['raw_candidate_stamp_ns']))
            if int(stamp) <= 0 or context.get('schema') != 1:
                return
        except (ValueError, KeyError, TypeError):
            return
        with self.lock:
            self.contexts[stamp] = context
            while len(self.contexts) > 8:
                self.contexts.popitem(last=False)

    def submit(self, record):
        if self.stop.is_set():
            return 'closed'
        # Avoid copying a multi-MiB record when the bounded sink is already
        # saturated. put_nowait below remains the authoritative race-safe check.
        if self.jobs.full():
            self.dropped += 1
            return 'queue_full'
        try:
            self.jobs.put_nowait(copy.deepcopy(record))
            return 'queued'
        except queue.Full:
            self.dropped += 1
            return 'queue_full'

    def _run(self):
        while not self.stop.is_set() or not self.jobs.empty():
            try:
                record = self.jobs.get(timeout=.05)
            except queue.Empty:
                continue
            try:
                # Independent ROS topics may arrive in either order. Wait only
                # on this writer thread, never in the admission callback.
                deadline = time.monotonic()+.5
                context = None
                while context is None:
                    with self.lock:
                        context = self.contexts.get(str(record['raw_candidate_stamp_ns']))
                    if context is not None or time.monotonic() >= deadline or self.stop.is_set():
                        break
                    self.stop.wait(.01)
                signature = failure_signature(record, context or {})
                if signature in self.seen:
                    self.coalesced += 1
                    continue
                if self.written >= self.max_files:
                    self.dropped += 1
                    continue
                record['local_context'] = context
                record['context_status'] = 'exact_raw_stamp' if context else 'NOT_RECORDED'
                record['failure_signature'] = signature
                payload = json.dumps(record, indent=2, allow_nan=False).encode()+b'\n'
                if len(payload) > self.max_record_bytes or self.bytes_written+len(payload) > self.max_bytes:
                    self.dropped += 1
                    continue
                path = self.directory / ('rejection_%02d_%s.json' % (self.written, signature[:12]))
                with path.open('xb') as f:
                    f.write(payload)
                self.bytes_written += len(payload)
                self.written += 1
                self.seen.add(signature)
            except Exception as exc:
                self.errors += 1
                self.warn('rejection snapshot failed (safety unaffected): '+str(exc))
            finally:
                self.jobs.task_done()

    def close(self):
        self.stop.set()
        self.worker.join(timeout=2)
