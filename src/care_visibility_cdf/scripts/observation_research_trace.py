"""Optional bounded experiment records; never observation or motion authority."""
import json
import math
import os
from pathlib import Path
import queue
import threading
import time


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, 'tolist'):
        return _plain(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class ObservationResearchTrace:
    def __init__(self, filename, capacity=512):
        self.stream = open(filename, 'x', buffering=1)
        self.pending = queue.Queue(maxsize=capacity)
        self.lock = threading.Lock()
        self.sequence = self.dropped = 0
        self.closed = False
        self.worker_error = None
        self.worker = threading.Thread(target=self._write, daemon=True)
        self.worker.start()

    def emit(self, stage, ros_s, **fields):
        with self.lock:
            if self.closed:
                return
            self.sequence += 1
            item = dict(stage=stage, sequence=self.sequence,
                        monotonic_s=time.monotonic(), wall_unix_s=time.time(),
                        ros_s=ros_s, fields=_plain(fields))
            try:
                self.pending.put_nowait(item)
            except queue.Full:
                self.dropped += 1

    def _write(self):
        try:
            while True:
                item = self.pending.get()
                try:
                    if item is None:
                        break
                    self.stream.write(json.dumps(item, allow_nan=False) + '\n')
                finally:
                    self.pending.task_done()
        except Exception as exc:
            self.worker_error = repr(exc)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
        if self.worker.is_alive():
            try:
                self.pending.put(None, timeout=.2)
                self.worker.join(timeout=.5)
            except queue.Full:
                pass
        # Do not race a still-running writer or claim a complete trace.
        if not self.worker.is_alive():
            self.stream.close()
        Path(self.stream.name + '.status.json').write_text(json.dumps(dict(
            emitted=self.sequence, dropped=self.dropped,
            drained=not self.worker.is_alive() and self.pending.empty(),
            worker_error=self.worker_error), indent=2) + '\n')


def record_observation(node, stage, fields):
    filename = os.environ.get('CARE_OBSERVATION_TRACE_FILE')
    if not filename:
        return
    # Construction is serialized even when subscriber callbacks first arrive together.
    with _creation_lock:
        if not hasattr(node, '_research_trace'):
            try:
                node._research_trace = ObservationResearchTrace(filename)
                import rospy
                rospy.on_shutdown(node._research_trace.close)
            except Exception as exc:
                node._research_trace = None
                node._research_trace_error = repr(exc)
        trace = node._research_trace
    if trace is not None:
        import rospy
        trace.emit(stage, rospy.Time.now().to_sec(), **fields)


_creation_lock = threading.Lock()


def publish_task_window(start_monotonic_s, timeout_s, ros_ns, arm_token):
    """Expose the existing watchdog origin without changing it or its scoring."""
    filename = os.environ.get('CARE_TASK_WINDOW_FILE')
    if not filename:
        return
    value = dict(schema=1, arm_token=arm_token, start_ros_ns=ros_ns,
                 start_monotonic_s=start_monotonic_s,
                 deadline_monotonic_s=start_monotonic_s + timeout_s,
                 budget_wall_s=timeout_s)
    path = Path(filename)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False) + '\n')
    temporary.replace(path)


def completion_context(now=None):
    """Return an explicitly enabled, calibrated proposal-ranking context."""
    if os.environ.get('CARE_COMPLETION_RANKING_ENABLED', 'false').lower() != 'true':
        return None
    try:
        window = json.loads(Path(os.environ['CARE_TASK_WINDOW_FILE']).read_text())
        profile = json.loads(Path(os.environ['CARE_COMPLETION_PROFILE_FILE']).read_text())
        now = time.monotonic() if now is None else now
        if window['schema'] != 1 or profile['schema'] != 1 or not profile['training_episode_ids']:
            return None
        start, deadline = float(window['start_monotonic_s']), float(window['deadline_monotonic_s'])
        if not all(math.isfinite(v) for v in (start, deadline, now)) or not start <= now < deadline:
            return None
        keys = ('approach_scale', 'rejoin_scale', 'acquisition_wall_s', 'tail_ros_s', 'hold_ros_s')
        values = {k: float(profile[k]) for k in keys}
        values['return_overhead_wall_s'] = float(profile.get('return_overhead_wall_s', 0.))
        if (not all(math.isfinite(v) and v >= 0. for v in values.values()) or
                min(values['approach_scale'], values['rejoin_scale']) < 1.):
            return None
        return dict(values, deadline_monotonic_s=deadline, measured_monotonic_s=now,
                    profile_id=profile['profile_id'], arm_token=window['arm_token'])
    except (OSError, KeyError, ValueError, TypeError):
        return None
