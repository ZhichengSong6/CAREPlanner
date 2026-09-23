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
    hybrid = result.get('per_sensor_hybrid')
    q = result.get('q_vis', [])
    if (len(q) != 7 or len(previous) != 7 or
            not all(math.isfinite(float(v)) for v in q) or
            not all(math.isfinite(float(v)) for v in previous)):
        return False
    if hybrid is not None:
        # When the optional per-sensor proposer ran, only its explicitly
        # accepted branch is a valid bounded refresh. A rejected branch must
        # not silently fall back to the scalar pose in this path.
        if (hybrid.get('accepted') is not True or
                list(q) != list(hybrid.get('selected_q_vis', []))):
            return False
    else:
        # The scalar proposer is the normal CASE001 path when the optional
        # per-sensor runtime is disabled. It is still only a steering proposal;
        # final GCDF and exact VBC remain the execution authorities.
        try:
            if not math.isfinite(float(result.get('final_f_min'))):
                return False
        except (TypeError, ValueError):
            return False
    # Existing measured-progress scale: microscopic optimizer drift is not an
    # alternative target and must not reopen the planner's liveness budget.
    return max(abs(float(a)-float(b)) for a,b in zip(q,previous)) > .01
