"""Bind recovery to the exact audited executable and fresh measured state."""
import json
import math
import os
from recovery_motion import residual_motion_bound


def context(candidate, deadline_ns, eligible, acceleration, brake_dt, brake_steps,
            request_session='standalone', request_sequence=None, planned_prefix=None,
            geometric_restore=None):
    # Header.seq is owned by the ROS publisher and changes during serialization.
    # The immutable audit stamp + complete content bind the two topic messages.
    # Business identity is explicit metadata, independent of the wire counter.
    sequence = int(candidate.header.seq) if request_sequence is None else int(request_sequence)
    data = dict(schema=3, trajectory_stamp_ns=str(candidate.header.stamp.to_nsec()),
        request_session=str(request_session), request_sequence=sequence, frame=candidate.header.frame_id,
        deadline_ns=str(deadline_ns), kind='repair_prefix_brake_hold' if eligible else 'ordinary',
        joint_names=list(candidate.joint_names), q=[list(p.positions) for p in candidate.points],
        v=[list(p.velocities) for p in candidate.points],
        times_s=[p.time_from_start.to_sec() for p in candidate.points],
        acceleration=list(acceleration), brake_dt_s=brake_dt, brake_steps=brake_steps)
    if planned_prefix is not None:
        data['planned_prefix'] = planned_prefix
    if geometric_restore is not None:
        from geometric_restore import validate
        data['geometric_restore'] = validate(geometric_restore)
        data['kind'] = 'geometric_restore'
    return json.dumps(data, separators=(',', ':'), allow_nan=False)


def unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('Duplicate recovery evidence key')
        out[key] = value
    return out


def commit_check(raw, stamp_ns, now_ns, measured_stamp_ns, names, measured, velocity,
                 acceleration, brake_dt, brake_steps, execution_margin, diagnostics=None,
                 feedback_gain=1., actuator_velocity=None, certificate_max_age_ns=50000000):
    """Bound FK snapshot changes plus braking, without claiming future tracking."""
    def reject(reason, failure_kind=None):
        if diagnostics is not None:
            diagnostics['reason'] = reason
            diagnostics['failure_kind'] = failure_kind or reason
        return False

    try:
        if not 50000000 <= certificate_max_age_ns <= 250000000:
            return reject('invalid_certificate_age_policy')
        if len(raw) > 256*1024:
            return reject('oversized_evidence')
        e = json.loads(raw, object_pairs_hook=unique_object)
        if (e['schema'] not in (1,2) or int(e['trajectory_stamp_ns']) != stamp_ns or
                e['execution_floor_m'] != execution_margin or not e['points'] or
                len(e['points']) > 32 or len(names) != 7 or len(set(names)) != 7):
            return reject('identity_or_shape')
        previous_stamp = int(e['measured_stamp_ns'])
        if diagnostics is not None:
            diagnostics.update(certificate_state_age_ms=(now_ns-previous_stamp)/1e6,
                               live_state_age_ms=(now_ns-measured_stamp_ns)/1e6,
                               map_age_ms=(now_ns-int(e['map_snapshot_stamp_ns']))/1e6)
        if min(previous_stamp, measured_stamp_ns) <= 0 or measured_stamp_ns < previous_stamp:
            return reject('measured_snapshot_stale', 'state_order_invalid')
        if not 0 <= now_ns-measured_stamp_ns <= 50000000:
            return reject('measured_snapshot_stale', 'live_state_stale')
        # Static-scene research may spend up to 250 ms computing the proof.
        # A fresh live sample and the full displacement/braking/residual bounds
        # below still determine whether its geometry applies to the live robot.
        if diagnostics is not None:
            diagnostics['certificate_max_age_ms'] = certificate_max_age_ns/1e6
        if not 0 <= now_ns-previous_stamp <= certificate_max_age_ns:
            return reject('measured_snapshot_stale', 'certificate_state_stale')
        reference, radii = list(e['measured_q']), list(e['joint_motion_radii'])
        if any(len(x) != 7 for x in [reference, radii, acceleration]):
            return reject('joint_dimensions')
        if (brake_dt != .05 or brake_steps != 20 or not math.isfinite(execution_margin) or execution_margin < 0):
            return reject('brake_contract')
        total = 0.
        for j, name in enumerate(names):
            q, v, a, r = float(measured[name]), float(velocity[name]), float(acceleration[j]), float(radii[j])
            if not all(math.isfinite(x) for x in [q, v, a, r, reference[j]]) or a <= 0. or r < 0.:
                return reject('nonfinite_or_missing_state')
            displacement = abs(q-reference[j])
            for _ in range(brake_steps):
                if abs(v) <= 1e-6:
                    break
                nv = math.copysign(max(0., abs(v)-a*brake_dt), v)
                displacement += .5*brake_dt*(abs(v)+abs(nv))
                v = nv
            if abs(v) > 1e-5:
                return reject('brake_steps_exhausted')
            total += r*displacement
        if diagnostics is not None:
            diagnostics['motion_bound_m'] = total if math.isfinite(total) else None
        actual_min = float(e['actual_min_distance_m'])
        if not math.isfinite(actual_min):
            return reject('nonfinite_clearance')
        if e['schema']==2:
            reserve=float(e['certified_residual_m'])
            horizon=float(e['motion_horizon_s'])
            if not math.isfinite(reserve) or reserve<=0 or not 0<=horizon<=2.:
                return reject('invalid_geometric_certificate')
            if len(e['measured_velocity'])!=7:return reject('joint_dimensions')
            measured_horizon=horizon
            # New certificates distinguish the measured emergency stop from
            # the proposed prefix/brake/hold. Only the latter uses its full
            # motion horizon. Recompute the stop duration independently; an
            # arbitrary short number cannot weaken admission. The linear
            # measured->initial bridge needs only the position-displacement
            # part, which the same bound also encloses.
            if 'measured_stop_horizon_s' in e:
                stop_v=list(e['measured_velocity']);stop_time=0.
                for _ in range(brake_steps):
                    if max(abs(float(x)) for x in stop_v)<=1e-6:break
                    stop_v=[math.copysign(max(0.,abs(float(x))-a*brake_dt),x)
                            for x,a in zip(stop_v,acceleration)]
                    stop_time+=brake_dt
                if (max(abs(float(x)) for x in stop_v)>1e-5 or
                        e.get('measured_stop_contract')!='ramp_stop_and_linear_start_bridge_v1' or
                        not math.isfinite(float(e['measured_stop_horizon_s'])) or
                        abs(float(e['measured_stop_horizon_s'])-stop_time)>1e-12):
                    return reject('invalid_measured_stop_contract')
                measured_horizon=stop_time
            elif 'measured_stop_contract' in e:
                return reject('invalid_measured_stop_contract')
            measured_bound=residual_motion_bound(reference,e['measured_velocity'],
                [measured[name] for name in names],[velocity[name] for name in names],
                radii,acceleration,measured_horizon,feedback_gain)
            initial_bound=residual_motion_bound(e['initial_q'],e['initial_velocity'],
                [measured[name] for name in names],[velocity[name] for name in names],
                radii,acceleration,horizon,feedback_gain)
            if actuator_velocity is None:return reject('missing_actuator_state')
            actuator_bound=residual_motion_bound(e['initial_q'],e['initial_velocity'],
                [measured[name] for name in names],[actuator_velocity[name] for name in names],
                radii,acceleration,horizon,feedback_gain)
            total=max(measured_bound,initial_bound,actuator_bound)
            measured_actuator_bound=0.
            if 'measured_stop_horizon_s' in e:
                measured_actuator_bound=residual_motion_bound(reference,e['measured_velocity'],
                    [measured[name] for name in names],[actuator_velocity[name] for name in names],
                    radii,acceleration,measured_horizon,feedback_gain)
                total=max(total,measured_actuator_bound)
            numerical_tolerance = 1e-5 if os.environ.get('CARE_TASK_FIRST_RESEARCH') == 'true' else 0.
            if diagnostics is not None:
                diagnostics.update(motion_bound_m=total,certified_residual_m=reserve,
                    research_numerical_tolerance_m=numerical_tolerance,
                    measured_motion_bound_m=measured_bound,initial_motion_bound_m=initial_bound,
                    actuator_motion_bound_m=actuator_bound,onset_clearance_budget_m=float(e['onset_clearance_budget_m']),
                    requires_reconstruction=max(initial_bound,actuator_bound)>=float(e['onset_clearance_budget_m']),
                    initial_state_changed=max(initial_bound,actuator_bound)>=reserve+numerical_tolerance-1e-12,
                    measured_motion_horizon_s=measured_horizon,
                    measured_actuator_motion_bound_m=measured_actuator_bound,
                    live_stamp_ns=str(measured_stamp_ns),decision_ros_ns=str(now_ns),
                    live_q=[measured[name] for name in names],live_velocity=[velocity[name] for name in names],
                    actuator_velocity=[actuator_velocity[name] for name in names],
                    certification_method='continuous_geometry_residual')
            if total>=reserve+numerical_tolerance-1e-12:return reject('geometric_certificate_state_changed')
        elif actual_min-total <= execution_margin+1e-12:
            return reject('actual_clearance_consumed')
        budget = float(e['onset_clearance_budget_m'])
        map_stamp = int(e['map_snapshot_stamp_ns'])
        if not math.isfinite(budget) or budget<=total+1e-12:
            return reject('onset_budget_consumed')
        if not 0<=now_ns-map_stamp<=50000000:
            return reject('map_snapshot_stale')
        points = set()
        for point in e['points']:
            p = point['point']
            if len(p) != 3 or not all(math.isfinite(x) for x in p) or tuple(p) in points:
                return reject('point_identity')
            points.add(tuple(p))
            if not all(math.isfinite(point[k]) for k in ['floor_m', 'start_m', 'end_m']):
                return reject('nonfinite_floor')
            if point['floor_m'] < execution_margin or point['start_m'] <= point['floor_m']:
                return reject('invalid_floor')
        if diagnostics is not None:
            diagnostics.update(reason='accepted', motion_bound_m=total, actual_min_distance_m=actual_min, onset_clearance_budget_m=budget)
        return True
    except (KeyError, ValueError, TypeError, OverflowError):
        return reject('malformed_evidence_or_state')
