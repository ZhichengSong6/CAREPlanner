"""Admission scheduling only; the full curve still needs both safety audits."""
import math

from recovery_motion import residual_motion_bound


def terminal_hold_start(points):
    """Recognize a real, constant-position, zero-velocity terminal interval."""
    if len(points) < 2:
        return None
    end = points[-1]
    n = len(end.positions)
    if not n or len(end.velocities) != n:
        return None
    start = None
    next_t = end.time_from_start.to_sec()
    for point in reversed(points):
        t = point.time_from_start.to_sec()
        if (len(point.positions) != n or len(point.velocities) != n or
                not all(math.isfinite(x) for x in point.positions + point.velocities) or
                any(abs(v) > 1e-9 for v in point.velocities) or
                any(abs(a-b) > 1e-9 for a, b in zip(point.positions, end.positions))):
            break
        if t < next_t:
            start = t
        next_t = t
    return start


def state_variation_ready(previous, current, actuator, certificate, acceleration, gain):
    """Use the existing residual formula to avoid rebuilding during fast change.

    Project the observed change over the last measured build-to-commit delay.
    This is a scheduling hint, not a certificate of future tracking.
    """
    if previous is None:
        return False
    stamp, q, v = current
    old_stamp, old_q, old_v = previous
    if not 0 < stamp-old_stamp <= 50000000:
        return False
    try:
        radii, horizon, reserve = (certificate[k] for k in
                                  ('joint_motion_radii', 'motion_horizon_s', 'certified_residual_m'))
        if not math.isfinite(reserve) or reserve <= 0:
            return False
        dt = (stamp-old_stamp)*1e-9
        latency = certificate.get('handoff_latency_s', dt)
        if not math.isfinite(latency) or latency < 0:
            return False
        variation = residual_motion_bound(old_q, old_v, q, v, radii, acceleration, horizon, gain)
        actuator_delta = residual_motion_bound(q, v, q, actuator, radii, acceleration, horizon, gain)
        return max(variation*max(1., latency/dt), actuator_delta) < reserve-1e-12
    except (KeyError, ValueError, TypeError, OverflowError):
        return False


def predict_held_motion(previous, current, actuator, certificate, acceleration, gain):
    """Propose a start at the observed audit delay during a stationary reference.

    The tracker applies v = gain*(q_hold-q), so its low-speed hold approaches
    q_hold exponentially. Two measured samples and the applied velocity must
    support that model before it may seed a proposal. This estimate grants no
    execution authority: the constructed curve still needs fresh full audits
    and the unchanged live initial/actuator check.
    """
    try:
        if previous is None or not math.isfinite(gain) or gain <= 0.:
            return None
        old_stamp, old_q, old_v = previous
        stamp, q, v = current
        dt = (stamp-old_stamp)*1e-9
        observed_delay = float(certificate['handoff_latency_s'])
        delay = min(.2, 1.25*observed_delay)
        radii = certificate['joint_motion_radii']
        horizon = float(certificate['motion_horizon_s'])
        reserve = float(certificate['certified_residual_m'])
        if (not 0. < dt <= .05 or not 0. < observed_delay <= .2 or
                not math.isfinite(reserve) or reserve <= 0. or
                any(len(a) != 7 for a in (old_q, old_v, q, v, actuator, radii, acceleration)) or
                not all(math.isfinite(x) for a in (old_q, old_v, q, v, actuator) for x in a) or
                max(abs(x) for x in list(v)+list(actuator)) > .01):
            return None
        decay = math.exp(-gain*dt)
        expected_q = [x+w*(1.-decay)/gain for x,w in zip(old_q, old_v)]
        expected_v = [w*decay for w in old_v]
        fit = residual_motion_bound(expected_q, expected_v, q, v,
                                    radii, acceleration, horizon, gain)
        applied = residual_motion_bound(q, v, q, actuator,
                                        radii, acceleration, horizon, gain)
        error = fit*max(1., delay/dt)+applied
        if not math.isfinite(error) or error >= reserve-1e-12:
            return None
        decay = math.exp(-gain*delay)
        predicted_q = [x+w*(1.-decay)/gain for x,w in zip(q, v)]
        predicted_v = [w*decay for w in v]
        if max(abs(a-b) for a,b in zip(predicted_q,q)) > .03:
            return None
        return dict(q=predicted_q, velocity=predicted_v, source_stamp_ns=stamp,
                    latency_s=delay, observed_latency_s=observed_delay, estimated_error_m=error)
    except (KeyError, ValueError, TypeError, OverflowError):
        return None
