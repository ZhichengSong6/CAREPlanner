#!/usr/bin/env python3
"""Validate the exact executable curve before either safety audit.

Uses the same quadratic/legacy-linear interpolation as trajectory_motion.hpp.
No clipping, path changes, or post-certification retiming takes place here.
"""
import math


def effective_limits(names, configured, tracker_cap):
    cap = float(tracker_cap)
    if not math.isfinite(cap) or cap <= 0 or not names or len(set(names)) != len(names):
        raise ValueError('invalid_execution_limits')
    limits = [float(configured[n]) for n in names]
    if any(not math.isfinite(v) or v <= 0 or v > cap for v in limits):
        raise ValueError('planner_tracker_velocity_contract_mismatch')
    return limits


def violation(trajectory, velocity, acceleration, tolerance=1e-6):
    names = list(trajectory.joint_names)
    n = len(names)
    if not n or len(set(names)) != n or len(velocity) != n or len(acceleration) != n:
        return 'motion_limit_dimensions'
    if any(not math.isfinite(x) or x <= 0 for x in list(velocity)+list(acceleration)):
        return 'invalid_motion_limits'
    if not trajectory.points:
        return 'empty_motion'
    previous = None
    for point in trajectory.points:
        t = point.time_from_start.to_sec()
        q, v = list(point.positions), list(point.velocities)
        if (not math.isfinite(t) or t < 0 or len(q) != n or len(v) != n or
                not all(math.isfinite(x) for x in q+v)):
            return 'invalid_motion_state'
        if any(abs(x) > limit+tolerance for x, limit in zip(v, velocity)):
            return 'reference_velocity_limit'
        if previous is not None:
            t0, q0, v0 = previous
            h = t-t0
            if h <= 1e-12:
                return 'nonincreasing_motion_time'
            ramp = all(abs(y-x-.5*h*(u+w)) <= 1e-9
                       for x, y, u, w in zip(q0, q, v0, v))
            for j in range(n):
                curve = h*(v0[j]-v[j])/8. if ramp else 0.
                secant = (q[j]-q0[j])/h
                if max(abs(secant+4*curve/h), abs(secant-4*curve/h)) > velocity[j]+tolerance:
                    return 'continuous_velocity_limit'
                if abs(8*curve/(h*h)) > acceleration[j]+tolerance:
                    return 'continuous_acceleration_limit'
                # A chord's derivatives may differ on either side of a knot.
                # Reject such a discontinuity at an acceleration-limited actuator.
                if max(abs(secant+4*curve/h-v0[j]), abs(secant-4*curve/h-v[j])) > tolerance:
                    return 'motion_velocity_discontinuity'
        elif abs(t) > 1e-12:
            return 'motion_missing_measured_start'
        previous = t, q, v
    if any(abs(v) > tolerance for v in previous[2]):
        return 'motion_missing_terminal_stop'
    return None
