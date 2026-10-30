"""Complete reference intervals for the actual quadratic tracker interpolator.

These are observation planning inputs, never physical or execution authority.
The header stamp identifies a trajectory; its execution origin is separate.
"""
import hashlib
import json
import math
import numpy as np


def trajectory_record(trajectory):
    names = list(trajectory.joint_names)
    if len(names) != 7 or len(set(names)) != 7:
        raise ValueError('invalid_trajectory_joint_names')
    points = []
    if not 1 <= len(trajectory.points) <= 4096:
        raise ValueError('invalid_trajectory_point_count')
    for p in trajectory.points:
        row = dict(time_ns=int(p.time_from_start.to_nsec()))
        for key in ('positions', 'velocities', 'accelerations', 'effort'):
            values = list(getattr(p, key, []))
            if (len(values) not in ((7,) if key == 'positions' else (0, 7)) or
                    any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
                raise ValueError('invalid_trajectory_' + key)
            row[key] = values
        if row['time_ns'] < 0 or (points and row['time_ns'] <= points[-1]['time_ns']):
            raise ValueError('nonincreasing_trajectory_time')
        points.append(row)
    if points[0]['time_ns'] != 0 or trajectory.header.stamp.to_nsec() <= 0:
        raise ValueError('missing_trajectory_start_or_identity')
    return dict(stamp_ns=int(trajectory.header.stamp.to_nsec()),
                frame_id=str(trajectory.header.frame_id), seq=int(trajectory.header.seq),
                joint_names=names, points=points)


def trajectory_sha256(trajectory):
    record = trajectory_record(trajectory)
    # rospy serializes Header.seq from its per-publisher counter on EACH
    # publish. It is transport metadata, not part of trajectory business identity.
    record.pop('seq')
    return hashlib.sha256(json.dumps(record, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _curvature(a, b, mapping):
    h = (b['time_ns'] - a['time_ns']) * 1e-9
    # C++ trajectoryCurvature falls back to linear for ALL joints if even one
    # mapped velocity is missing or its integral disagrees with displacement.
    if not a['velocities'] or not b['velocities']:
        return [0.] * 7
    for j in mapping:
        integral = .5 * h * (a['velocities'][j] + b['velocities'][j])
        if not math.isfinite(integral) or abs(b['positions'][j] - a['positions'][j] - integral) > 1e-9:
            return [0.] * 7
    return [h * (a['velocities'][j] - b['velocities'][j]) / 8. for j in mapping]


def _bounds(q0, q1, c, lo, hi):
    delta = q1 - q0
    def q(s):
        return (1. - s) * q0 + s * q1 + 4. * s * (1. - s) * c
    samples = [q(lo), q(hi)]
    if c:
        critical = (delta + 4. * c) / (8. * c)
        if lo < critical < hi:
            samples.append(q(critical))
    # Cover evaluation roundoff as well as endpoint and stationary-point error.
    pad = 32. * 2.220446049250313e-16 * (1. + abs(q0) + abs(q1) + 4. * abs(c))
    return float(np.nextafter(min(samples) - pad, -math.inf)), float(np.nextafter(max(samples) + pad, math.inf))


def build_trajectory_prefix(*, trajectory, joint_names, start_window_ns,
        horizon_s, target_abs_velocity, applied_velocity, stopping_budget,
        phase_s=0., max_segments=32):
    """Enclose every clipped interpolation piece, including interior extrema.

    start_window_ns is an explicit earliest/latest start, NOT header.stamp.
    stopping_budget(worst_applied, proposed_target_range) returns travel and
    duration under its declared conditional command model. The proposed range
    is a planning assumption; each actual grant must independently check all
    active/pending rights and its complete stopping set.
    """
    record = trajectory_record(trajectory)
    if len(joint_names) != 7 or set(joint_names) != set(record['joint_names']):
        raise ValueError('unexpected_trajectory_joints')
    if type(max_segments) is not int or max_segments < 1:
        raise ValueError('invalid_trajectory_segment_capacity')
    if (not isinstance(start_window_ns, (list, tuple)) or len(start_window_ns) != 2 or
            any(type(t) is not int or t <= 0 for t in start_window_ns) or
            start_window_ns[0] > start_window_ns[1] or
            not all(math.isfinite(x) for x in (horizon_s, phase_s)) or horizon_s <= 0 or phase_s < 0):
        raise ValueError('invalid_trajectory_start_window')
    for values in (target_abs_velocity, applied_velocity):
        if len(values) != 7 or any(not math.isfinite(x) for x in values):
            raise ValueError('invalid_planning_velocity_range')
    if any(v < 0 for v in target_abs_velocity):
        raise ValueError('invalid_planning_velocity_range')
    worst = [max(abs(u), cap) for u, cap in zip(applied_velocity, target_abs_velocity)]
    travel, stop_s = stopping_budget(worst, list(target_abs_velocity))
    travel = list(map(float, travel))
    if len(travel) != 7 or any(not math.isfinite(t) or t < 0 for t in travel) or not math.isfinite(stop_s) or stop_s < 0:
        raise ValueError('invalid_planning_stopping_budget')
    mapping = [record['joint_names'].index(n) for n in joint_names]
    end_s = min(phase_s + horizon_s, record['points'][-1]['time_ns'] * 1e-9)
    segments = []
    for index, (a, b) in enumerate(zip(record['points'], record['points'][1:])):
        t0, t1 = a['time_ns'] * 1e-9, b['time_ns'] * 1e-9
        left, right = max(phase_s, t0), min(end_s, t1)
        if right <= left:
            continue
        c = _curvature(a, b, mapping)
        bounds = [_bounds(a['positions'][j], b['positions'][j], cc,
                          (left-t0)/(t1-t0), (right-t0)/(t1-t0)) for j, cc in zip(mapping, c)]
        segments.append(dict(q_low=[v[0] for v in bounds], q_high=[v[1] for v in bounds],
            stop_travel_rad=travel, first_use_s=start_window_ns[0]*1e-9+left,
            required_until_s=start_window_ns[1]*1e-9+right+stop_s,
            phase_begin_s=left, phase_end_s=right, source_piece=index,
            interpolation='tracker_quadratic_v1'))
    # At/after the final knot the tracker holds its position and copies the
    # terminal feed-forward. A nonzero terminal velocity must not be hidden.
    duration_s = record['points'][-1]['time_ns'] * 1e-9
    if phase_s+horizon_s > duration_s or not segments:
        if not segments and len(record['points']) > 1 and phase_s < end_s:
            raise ValueError('missing_trajectory_prefix_piece')
        p = record['points'][-1]
        q = [p['positions'][j] for j in mapping]
        hold_begin = max(phase_s, duration_s)
        segments.append(dict(q_low=q, q_high=q, stop_travel_rad=travel,
            first_use_s=start_window_ns[0]*1e-9+hold_begin,
            required_until_s=start_window_ns[1]*1e-9+phase_s+horizon_s+stop_s,
            phase_begin_s=hold_begin, phase_end_s=phase_s+horizon_s,
            source_piece=len(record['points'])-1, interpolation='terminal_hold'))
    if len(segments) > max_segments:
        raise ValueError('trajectory_prefix_segment_capacity')
    digest = trajectory_sha256(trajectory)
    return dict(schema=1, planning_only=True, runtime_admission=False,
        reference_only=True, trajectory_sha256=digest,
        trajectory_stamp_ns=record['stamp_ns'], path_id=str(record['stamp_ns'])+':'+digest,
        start_window_ns=list(start_window_ns), phase_s=phase_s,
        requested_horizon_s=horizon_s, covered_until_phase_s=phase_s+horizon_s,
        proposed_target_abs_velocity=list(target_abs_velocity), worst_applied_velocity=worst,
        conditional_stop_s=stop_s, segments=segments)


def bind_report(report, construction):
    if report['trajectory_stamp_ns'] != construction['trajectory_stamp_ns'] or report['path_id'] != construction['path_id']:
        raise ValueError('trajectory_report_construction_mismatch')
    report['trajectory_sha256'] = construction['trajectory_sha256']
    report['trajectory_construction'] = {k: v for k, v in construction.items() if k != 'segments'}
    return report


def validate_trajectory_report(report, trajectory, now_s):
    from stopping_observation_demands import validate_report
    if report.get('evidence_kind') == 'context_observation_demand':
        raise ValueError('context_report_is_not_candidate_evidence')
    validate_report(report, now_s, trajectory.header.stamp.to_nsec())
    if report.get('trajectory_sha256') != trajectory_sha256(trajectory):
        raise ValueError('stopping_observation_trajectory_content_mismatch')
    return report


def apply_absolute_observation_deadline(result, region, source):
    """Keep prefix deadlines absolute through the legacy waypoint generator."""
    if source != 'stopping_prefix_lookahead':
        return result
    deadline = region.get('stopping_observation_deadline_abs_s', math.nan)
    epoch = region.get('stopping_observation_epoch_s', math.nan)
    epoch_source = region.get('stopping_observation_epoch_source', 'stopping_observation_start_window')
    if not math.isfinite(deadline) or not math.isfinite(epoch):
        raise ValueError('missing_absolute_stopping_observation_deadline')
    if epoch_source not in ('stopping_observation_start_window', 'observation_context_started'):
        raise ValueError('invalid_stopping_observation_epoch_source')
    return dict(result, deadline_absolute_ros_s=deadline,
        deadline_from_start_s=deadline-epoch,
        deadline_epoch_source=epoch_source, deadline_epoch_ros_s=epoch)
