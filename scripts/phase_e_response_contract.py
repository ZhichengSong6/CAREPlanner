"""Conditional response algebra and strict step auditing, without an authorizer.

Observed maxima never become future bounds. Even a perfectly fitting trace
does not establish solver, actuator, lifecycle, or environmental qualification.
"""
import math
import numpy as np


def seven(value, name, nonnegative=False):
    x = np.asarray(value, dtype=float)
    if x.shape != (7,) or not np.all(np.isfinite(x)) or (nonnegative and np.any(x < 0)):
        raise ValueError('invalid_' + name)
    return x


def conditional_error_budget(*, steps, step_s, velocity_residual_bound,
                             pose_increment_residual_bound, initial_error=None):
    """Triangle inequality for q[k+1]-q[k]-h*u[k].

If every future step has |v[k+1]-u[k]| <= eps and
|q[k+1]-q[k]-h*v[k+1]| <= eta, error <= E0+N*(h*eps+eta).
Neither premise, nor finite settling after the command reaches zero, is
established by this function. The output is explicitly not qualified evidence.
"""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError('invalid_step_count')
    if not math.isfinite(step_s) or step_s <= 0:
        raise ValueError('invalid_step_size')
    eps = seven(velocity_residual_bound, 'velocity_residual_bound', True)
    eta = seven(pose_increment_residual_bound, 'pose_increment_residual_bound', True)
    e0 = np.zeros(7) if initial_error is None else seven(initial_error, 'initial_error', True)
    return dict(error_rad=(e0 + steps * (step_s * eps + eta)).tolist(),
                physical_bound_certified=False, runtime_admission=False,
                premises='uniform future per-step bounds, not sample maxima',
                settling_bound_established=False)


def audit_steps(rows, expected_step_s=.001, allow_discontinuities=False):
    """Audit pairing, actual motors and integration defects at every logged step."""
    if not rows:
        raise ValueError('empty_physics_trace')
    t = np.asarray([r['t_ns'] for r in rows], dtype=np.int64)
    index = np.asarray([r['index'] for r in rows], dtype=np.int64)
    h = np.asarray([r['step_s'] for r in rows], dtype=float)
    if not np.all(np.isfinite(h)) or np.any(h <= 0):
        raise ValueError('invalid_physics_step')
    if not np.all(np.diff(index) == 1):
        raise ValueError('missing_observer_record')
    values = {}
    for key in ('q_before', 'q_after', 'v_before', 'velocity', 'motor', 'motor_after', 'fmax'):
        a = np.asarray([r[key] for r in rows], dtype=float)
        if a.shape != (len(rows), 7) or not np.all(np.isfinite(a)):
            raise ValueError('invalid_physics_' + key)
        values[key] = a
    if np.any(values['fmax'] < 0):
        raise ValueError('negative_motor_force_limit')
    if np.max(np.abs(values['motor'] - values['motor_after'])) > 1e-12:
        raise ValueError('motor_changed_during_physics_step')
    # World::UpdateBegin and UpdateEnd share a stamp; physics integrates h.
    if any(abs(float(r['end_time_ns']) - int(r['t_ns'])) > 100 for r in rows):
        raise ValueError('observer_begin_end_stamp_mismatch')
    gaps = np.diff(t)
    nominal = int(round(expected_step_s * 1e9))
    nonnominal = np.flatnonzero((gaps != nominal) | (np.abs(h[1:] - expected_step_s) > 1e-12))
    if not allow_discontinuities and (len(nonnominal) or abs(h[0] - expected_step_s) > 1e-12):
        raise ValueError('physics_domain_changed_or_observation_gap')
    contiguous = gaps > 0
    # Changes of q by reset/set-model-configuration are never silently folded
    # into a response error: report the boundary and exclude it from continuity.
    boundary_error = np.max(np.abs(values['q_after'][:-1] - values['q_before'][1:]), axis=1)
    boundaries = np.flatnonzero((~contiguous) | (boundary_error > 1e-8))
    if not allow_discontinuities and len(boundaries):
        raise ValueError('pose_discontinuity_between_steps')
    velocity_defect = values['velocity'] - values['motor']
    increment_defect = values['q_after'] - values['q_before'] - h[:, None] * values['velocity']
    command_defect = values['q_after'] - values['q_before'] - h[:, None] * values['motor']
    if not np.allclose(command_defect, h[:, None] * velocity_defect + increment_defect,
                       atol=2e-16, rtol=1e-12):
        raise ValueError('response_algebra_mismatch')
    return dict(rows=len(rows), t=t, h=h, values=values,
                velocity_defect=velocity_defect, increment_defect=increment_defect,
                command_defect=command_defect, nonnominal_indices=nonnominal,
                discontinuity_indices=boundaries, physical_bound_certified=False)


def persistent_rest_index(velocity, start, hold_steps=101, tolerance=.001):
    a = np.asarray(velocity, dtype=float)
    if a.ndim != 2 or a.shape[1] != 7 or not np.all(np.isfinite(a)):
        raise ValueError('invalid_rest_velocity')
    if start < 0 or hold_steps < 1 or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('invalid_rest_window')
    good = np.max(np.abs(a), axis=1) <= tolerance
    for k in range(start, len(a) - hold_steps + 1):
        if np.all(good[k:k + hold_steps]):
            return k
    return None
