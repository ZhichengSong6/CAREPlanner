"""ROS-independent pose and benchmark-window helpers shared by watcher/evaluator."""
import json
import math
import os
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_EVEN

import numpy as np


def seconds_to_ns(seconds, duration=False):
    """Legacy seconds/config adapter; measured stamps must use integer ns.

    Sub-ns positive durations round UP, never shorten the required hold.
    Historical float timestamps round to the nearest representable nanosecond.
    """
    value = Decimal(str(seconds))
    if not value.is_finite() or value < 0:
        raise ValueError('time must be finite and nonnegative')
    return int((value * 1000000000).to_integral_value(
        rounding=ROUND_CEILING if duration else ROUND_HALF_EVEN))


def parse_stamp_ns(value):
    value = Decimal(str(value))
    if not value.is_finite() or value < 0 or value != value.to_integral_value():
        raise ValueError('timestamp must be nonnegative integer nanoseconds')
    return int(value)


def window_time_ns(window, name):
    exact = name + '_ns'
    if exact in window:
        return parse_stamp_ns(window[exact])
    return seconds_to_ns(window[name + '_s'])


def quat_rot(q):
    q = np.asarray(q, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if not np.isfinite(q).all() or not math.isfinite(n) or n <= 0:
        raise ValueError("quaternion must be finite and nonzero")
    x, y, z, w = q / n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def pose_error(T, p_goal, R_goal):
    if not all(np.isfinite(x).all() for x in (T, p_goal, R_goal)):
        return math.inf, math.inf
    pos = float(np.linalg.norm(T[:3, 3] - p_goal))
    cosine = (float(np.trace(R_goal.T @ T[:3, :3])) - 1) / 2
    return pos, math.acos(max(-1.0, min(1.0, cosine)))


def load_window(run, required=False):
    path = os.path.join(run, "goal_stop_status.json")
    status = {}
    if os.path.isfile(path):
        with open(path) as f:
            status = json.load(f)
    window = status.get("benchmark_window")
    if window is None:
        if required:
            raise ValueError("required benchmark_window is missing")
        return None
    start, end = window.get("start_ros_s"), window.get("end_ros_s")
    if (not window.get("valid") or start is None or end is None or
            not math.isfinite(start) or not math.isfinite(end) or end < start):
        raise ValueError("invalid benchmark_window; refusing unbounded evaluation")
    q = window.get("end_measured_q")
    if q is None or len(q) != 7 or not all(math.isfinite(v) for v in q):
        raise ValueError("benchmark end measured q must contain seven finite values")
    if window_time_ns(window, 'end_ros') < window_time_ns(window, 'start_ros'):
        raise ValueError('invalid integer benchmark window')
    return window


def window_rows(rows, window, carry_state=False):
    if window is None:
        return rows
    start, end = window["start_ros_s"], window["end_ros_s"]
    selected = [r for r in rows if start <= r.get("_t", math.nan) <= end]
    if carry_state:
        before = [r for r in rows if r.get("_t", math.nan) < start]
        if before:
            selected.insert(0, dict(before[-1], _t=start))
    return selected


def window_joint_samples(samples, window, timestamps_ns=False):
    if window is None:
        return samples
    start, end = ((window_time_ns(window, 'start_ros'), window_time_ns(window, 'end_ros'))
                  if timestamps_ns else (window['start_ros_s'], window['end_ros_s']))
    selected = {t:q for t,q in samples if start <= t <= end}
    # Boundary samples were actually measured by the watcher. Include both
    # even if the independent recorder missed a frame, preserving the hold.
    first = (window_time_ns(window, 'first_sample_ros') if timestamps_ns and
             ('first_sample_ros_ns' in window or 'first_sample_ros_s' in window)
             else window.get('first_sample_ros_s'))
    for t, q in ((first, window.get("first_measured_q")),
                 (end, window.get("end_measured_q"))):
        if t is None or not start <= t <= end:
            continue
        if q is None or len(q) != 7 or not all(math.isfinite(v) for v in q):
            raise ValueError("invalid measured boundary q")
        selected[t] = q
    return sorted(selected.items())


def rejection_counts(gcdf_unsafe, gcdf_timeout, combined_unsafe, combined_timeout):
    if combined_unsafe < gcdf_unsafe or combined_timeout < gcdf_timeout:
        raise ValueError("GCDF counters exceed combined verification counters")
    return {"commit_gate_rejection_count": combined_unsafe + combined_timeout,
            "exact_vbc_unsafe_rejection_count": combined_unsafe - gcdf_unsafe,
            "post_gcdf_timeout_count": combined_timeout - gcdf_timeout}
