"""Choose another proposal after a future visibility rejection; grants no motion."""
import json
import math


def next_task_prefix(evidence, audited_stamp, current_prefix, step_s, attempts):
    """At most three shorter proposals, rounded to an existing motion knot.

    The first offending sweep only guides the proposal. The new braking tail
    changes the swept path, so both final GCDF and VBC must run again.
    """
    try:
        if attempts >= 3 or not all(math.isfinite(x) and x > 0 for x in [current_prefix, step_s]):
            return None
        value = json.loads(evidence)
        if value.get('schema') != 1 or str(value.get('trajectory_stamp_ns')) != str(audited_stamp):
            return None
        points = value.get('violations')
        if not isinstance(points, list) or not points:
            return None
        times = [float(p['sweep_s']) for p in points]
        if not all(math.isfinite(t) and t > 0 for t in times):
            return None  # The present body cannot be repaired by truncation.
        target = min(.5 * current_prefix, .75 * min(times))
        prefix = math.floor((target + 1e-10) / step_s) * step_s
        if prefix < step_s - 1e-10 or prefix >= current_prefix - 1e-10:
            return None
        return prefix
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
