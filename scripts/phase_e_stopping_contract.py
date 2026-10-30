#!/usr/bin/env python3
"""Conditional stopping envelopes, including one more decision cycle.

This is an offline qualification gate, not a command authorizer. Arbitrary
bounded command updates may occur up to an ENFORCED fixed lease deadline.
Unlike the old no-more-commands model, renewal cannot silently extend a proof.
Physical response and scheduler/transport bounds must come from a separate
qualified contract; finite replay observations are never promoted into bounds.
"""
import math
import numpy as np

from phase_e_stop_envelope import vector


def bounded_update_travel(applied, acceleration, velocity_limits, hold_s, update_gap_s,
                          max_braking_step_s=.02, authorized_target_abs=None):
    """Absolute command travel from the update that supplied ``applied``.

    The conditional model requires monotone simulation time, a future update
    gap <= update_gap_s, and period == elapsed at every update. Output is held
    between updates. hold_s starts at that SAME update, not at receipt time.
    """
    u = np.abs(vector(applied, 'applied'))
    a = vector(acceleration, 'acceleration', True)
    vmax = vector(velocity_limits, 'velocity limits', True)
    if (any(not math.isfinite(x) or x < 0 for x in [hold_s, update_gap_s, max_braking_step_s]) or
            update_gap_s == 0 or max_braking_step_s == 0 or np.any(a <= 0) or
            np.any(vmax <= 0) or np.any(u > vmax+1e-12)):
        raise ValueError('invalid_bounded_update_contract')
    cap=vmax
    if authorized_target_abs is not None:
        target=vector(authorized_target_abs,'authorized target range',True)
        if np.any(target>vmax):raise ValueError('target_range_exceeds_controller_limit')
        # A newly smaller target range cannot erase the current output. Ignore
        # its tracking decay until expiry, then enclose full rate-limited braking.
        cap=np.maximum(u,target)
    # Any target sequence is bounded by min(vmax, |u|+a*t). One update gap
    # covers deadline detection; a second covers zero-order discrete braking.
    before_brake = hold_s+update_gap_s
    accelerating = np.minimum(before_brake, (cap-u)/a)
    peak = np.minimum(cap, u+a*before_brake)
    forward = u*accelerating+.5*a*accelerating**2+cap*(before_brake-accelerating)
    brake_a = a*min(1., max_braking_step_s/update_gap_s)
    travel = forward+peak**2/(2*brake_a)+peak*update_gap_s
    stop_s = before_brake+float(np.max(peak/brake_a))+update_gap_s
    return travel, stop_s


def qualify(*, now_s, measured_s, applied, acceleration, velocity_limits,
            next_decision_s, transport_bound_s, update_gap_s, lease_end_s,
            lease_enforced, physical_error_rad, physical_settle_s,
            snapshot, max_state_age_s=.05, max_braking_step_s=.02,
            authorized_target_abs=None):
    """Enclose travel about the measured pose, including receipt latency.

    snapshot must refer to the COMPLETE geometry of the returned enclosure,
    bind a stable version, and cover its full horizon. This routine does not
    invent that snapshot from the last selected VBC blockers.
    applied and the enclosure's reference pose both belong to measured_s.
    command_stop_s/physical_stop_s are remaining times from now_s, whereas
    command_stop_from_measurement_s starts at the reference measurement.
    """
    result = dict(qualified=False, runtime_admission=False, reasons=[], budgets={})
    reasons = result['reasons']
    def reject(reason):
        if reason not in reasons:
            reasons.append(reason)
    if lease_enforced is not True:
        reject('missing_enforced_command_lease')
    if physical_error_rad is None or physical_settle_s is None:
        reject('missing_physical_response_bound')
    if not snapshot:
        reject('missing_complete_stopping_snapshot')
    values = [now_s, measured_s, next_decision_s, transport_bound_s, update_gap_s, lease_end_s]
    if any(x is None or not math.isfinite(x) for x in values):
        reject('missing_or_invalid_time_bounds')
        return result
    age = now_s-measured_s
    # ROS timestamps have nanosecond resolution; subtraction in seconds may
    # round an exactly-on-limit age upward by a few floating-point ulps.
    if age < 0 or age > max_state_age_s+1e-12:
        reject('stale_or_invalid_measured_state')
        return result
    if next_decision_s < 0 or transport_bound_s < 0 or update_gap_s <= 0 or lease_end_s < now_s:
        reject('invalid_command_deadline')
        return result
    physical = physical_error_rad is not None and physical_settle_s is not None
    if not physical:
        reject('missing_physical_response_bound')
    else:
        error = vector(physical_error_rad, 'physical error', True)
        if not math.isfinite(physical_settle_s) or physical_settle_s < 0:
            raise ValueError('invalid_physical_settling_bound')
    for mode, extra in [('withdraw_now', 0.), ('continue_then_stop', next_decision_s)]:
        # During receipt latency the actuator may already have accelerated.
        # Keep one time origin for that acceleration, subsequent commands and
        # braking; merely adding vmax*age leaves the braking speed too small.
        hold_from_measurement = lease_end_s-measured_s+extra+transport_bound_s
        travel, horizon = bounded_update_travel(applied, acceleration, velocity_limits,
            hold_from_measurement, update_gap_s, max_braking_step_s,authorized_target_abs)
        remaining_stop = horizon-age
        b = dict(command_travel_rad=travel.tolist(), command_stop_s=remaining_stop,
                 command_stop_from_measurement_s=horizon, measured_state_s=measured_s,
                 required_until_s=measured_s+horizon, fixed_lease_end_s=lease_end_s+extra)
        if authorized_target_abs is not None:b['authorized_target_abs']=list(authorized_target_abs)
        if physical:
            b['physical_travel_rad'] = (travel+error).tolist()
            b['physical_stop_s'] = remaining_stop+physical_settle_s
            b['required_until_s'] += physical_settle_s
        result['budgets'][mode] = b
    if not snapshot:
        reject('missing_complete_stopping_snapshot')
    else:
        required = max(b['required_until_s'] for b in result['budgets'].values())
        if (snapshot.get('complete') is not True or snapshot.get('stable_version') is not True or
                not snapshot.get('enclosure_sha256') or snapshot.get('bound_modes') != ['withdraw_now', 'continue_then_stop']):
            reject('unbound_or_incomplete_stopping_snapshot')
        if snapshot.get('enclosure_kind','physical')!='physical':
            reject('command_only_snapshot_not_physical_evidence')
        if snapshot.get('inside_map') is not True or snapshot.get('all_centers_free') is not True:
            reject('stopping_enclosure_not_covered')
        valid_until = snapshot.get('valid_until_s')
        if valid_until is None or not math.isfinite(valid_until) or valid_until < required:
            reject('evidence_expires_before_stop')
    result['qualified'] = not reasons
    return result
