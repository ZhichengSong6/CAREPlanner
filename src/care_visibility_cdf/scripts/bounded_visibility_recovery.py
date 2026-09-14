"""Bounded steering refresh policy; never an execution/safety certificate."""
import math
import struct


def region_key(points):
    # Same actual FP32 solver inputs, so float64/float32 map transport changes
    # cannot buy another attempt. Keep exact ordering-independent point sets.
    return tuple(sorted({struct.pack('!3f', *p) for p in points}))


class VisibilityRecoveryBudget:
    def __init__(self):
        self.anchor = None
        self.attempted = set()

    def reserve(self, points, q):
        if len(q) != 7 or not all(math.isfinite(float(v)) for v in q):
            return False
        key = region_key(points)
        if not key:
            return False
        if self.anchor is None or max(abs(float(a)-float(b)) for a,b in zip(q,self.anchor)) > .01:
            self.anchor = tuple(float(v) for v in q)
            self.attempted.clear()
        if key in self.attempted or len(self.attempted) >= 64:
            return False
        self.attempted.add(key)  # Includes failed generation: no infinite retry.
        return True


def certified_distinct_q(result, previous):
    hybrid = result.get('per_sensor_hybrid', {})
    q = result.get('q_vis', [])
    if (hybrid.get('accepted') is not True or len(q) != 7 or len(previous) != 7 or
            not all(math.isfinite(float(v)) for v in q) or
            list(q) != list(hybrid.get('selected_q_vis', []))):
        return False
    # Existing measured-progress scale: microscopic optimizer drift is not an
    # alternative target and must not reopen the planner's liveness budget.
    return max(abs(float(a)-float(b)) for a,b in zip(q,previous)) > .01
