"""Clock evidence for passive CDF records; never changes task deadlines."""
import hashlib
import math
import os
from pathlib import Path
import time


def clock_identity():
    info = time.get_clock_info('monotonic')
    if not info.monotonic or info.adjustable:
        raise ValueError('A non-adjustable monotonic clock is required')
    return dict(clock='CLOCK_MONOTONIC',
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                time_namespace=os.readlink('/proc/self/ns/time'))


def clock_source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def clock_stamp():
    # Capture before serialization. Wall time is retained only for diagnostics.
    return dict(monotonic_ns=time.monotonic_ns(), wall_time_ns=time.time_ns())


def annotate_status(payload):
    """Add evidence to a copy of the final score window, keeping all old values."""
    window = payload.get('benchmark_window')
    if not window or not window.get('valid'):
        return payload
    start, end = window['start_monotonic_s'], window['end_monotonic_s']
    if not all(math.isfinite(x) for x in (start, end)) or not 0 <= start <= end:
        raise ValueError('Invalid original monotonic score bounds')
    result = dict(payload)
    result['benchmark_window'] = dict(window,
        monotonic_clock_identity=clock_identity(),
        start_monotonic_ns=round(start*1e9), end_monotonic_ns=round(end*1e9),
        monotonic_bounds_source='rounded_existing_score_window_seconds')
    return result


def validate_clock_records(records):
    if not records or records[0].get('schema') != 2:
        raise ValueError('Schema2 monotonic evidence required; legacy logs cannot be reconstructed')
    identity = records[0].get('monotonic_clock_identity')
    if not isinstance(identity, dict) or identity.get('clock') != 'CLOCK_MONOTONIC' or not all(
            isinstance(identity.get(k), str) and identity[k] for k in ('boot_id', 'time_namespace')):
        raise ValueError('Missing clock identity')
    previous = -1
    for row in records:
        value = row.get('monotonic_ns')
        if type(value) is not int or value < previous or value < 0:
            raise ValueError('Missing, invalid or reversed monotonic timestamp')
        previous = value
    return identity


def task_bounds(records, window):
    identity = validate_clock_records(records)
    if not window.get('valid') or window.get('monotonic_clock_identity') != identity:
        raise ValueError('Invalid task or different boot/time namespace')
    lo, hi = window.get('start_monotonic_ns'), window.get('end_monotonic_ns')
    if type(lo) is not int or type(hi) is not int or not 0 <= lo <= hi:
        raise ValueError('Explicit monotonic bounds required')
    for name, value in [('start', lo), ('end', hi)]:
        seconds = window.get(name+'_monotonic_s')
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or abs(round(seconds*1e9)-value) > 1:
            raise ValueError('Integer bounds differ from the original score window')
    if records[0]['monotonic_ns'] > lo or records[-1]['monotonic_ns'] < hi:
        raise ValueError('Worker session does not cover the original task window')
    return lo, hi
