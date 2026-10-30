"""Bounded observation proposal ranking. No map, completion, or safety authority."""
import math
import os
import time
import numpy as np


def task_rejoin_context(q_start, q_task, velocity, acceleration, source=None):
    arrays = [np.asarray(a, dtype=float).reshape(7)
              for a in (q_start, q_task, velocity, acceleration)]
    if (not all(np.all(np.isfinite(a)) for a in arrays) or
            np.any(arrays[2] <= 0.) or np.any(arrays[3] <= 0.)):
        raise ValueError('invalid observation task-return context')
    return dict(q_start=arrays[0].tolist(), q_task=arrays[1].tolist(),
                velocity=arrays[2].tolist(), acceleration=arrays[3].tolist(),
                source=source)


def _rest_time(q0, q1, context):
    distance = np.abs(np.asarray(q1)-np.asarray(q0))
    v, a = np.asarray(context['velocity']), np.asarray(context['acceleration'])
    return float(np.max(np.where(distance <= v*v/a,
        2.*np.sqrt(distance/a), distance/v+v/a)))


def task_pose_ranking_enabled():
    return os.environ.get('CARE_TASK_POSE_RANKING') == 'true'


def task_departure_penalty(cost):
    """Soft time-equivalent preference; neither reachability nor completion proof.

    Fixed research weights: 0.20 m and 1 rad each cost one second in ranking.
    A necessary retreat may still win through shorter travel or observation order.
    """
    position = float(cost['task_position_m'])
    angle = float(cost['task_angle_rad'])
    if not all(math.isfinite(x) and x >= 0. for x in (position, angle)):
        raise ValueError('invalid task pose cost')
    return position / .20 + angle


def ranking_total(cost, completion=False):
    total = (cost['completion']['estimate_wall_s'] if completion else cost['total_s'])
    return total + (task_departure_penalty(cost) if task_pose_ranking_enabled() else 0.)


def candidate_cost(q, context, pose, task_pose):
    q = np.asarray(q, dtype=float).reshape(7)
    if not np.all(np.isfinite(q)):
        raise ValueError('nonfinite observation candidate')
    approach = _rest_time(context['q_start'], q, context)
    rejoin = _rest_time(q, context['q_task'], context)
    rate=float(context.get('ros_per_wall_s',1.))
    if not math.isfinite(rate) or rate<=0.:raise ValueError('invalid task-return clock rate')
    approach/=rate;rejoin/=rate
    position = float(np.linalg.norm(pose[:3, 3]-task_pose[:3, 3]))
    cosine = float((np.trace(task_pose[:3, :3].T @ pose[:3, :3])-1.)*.5)
    angle = math.acos(min(1., max(-1., cosine)))
    result=dict(approach_s=approach, rejoin_s=rejoin, total_s=approach+rejoin,
                task_position_m=position, task_angle_rad=angle,
                task_pose_error=max(position/.02, angle/.2))
    evaluator=context.get('corridor_evaluator')
    if evaluator is not None:
        try:preview=evaluator(q,context)
        except Exception:preview=dict(status='unavailable')
        result['corridor_preview']=preview
        result['corridor_used']=preview.get('status')=='valid'
        if result['corridor_used']:
            penalty=approach*preview['approach']['risk_fraction']+rejoin*preview['rejoin']['risk_fraction']
            result['new_observation_proxy_s']=penalty
            result['total_s']+=penalty
    completion = context.get('completion')
    if completion is not None:
        # This remains a calibrated ranking estimate, never a feasibility bound.
        estimate = (approach * completion['approach_scale'] +
                    rejoin * completion['rejoin_scale'] +
                    result.get('new_observation_proxy_s', 0.) +
                    completion['acquisition_wall_s'] +
                    completion.get('return_overhead_wall_s', 0.) +
                    (completion['tail_ros_s'] + completion['hold_ros_s']) / rate)
        result['completion'] = dict(estimate_wall_s=estimate,
            deadline_monotonic_s=completion['deadline_monotonic_s'],
            profile_id=completion['profile_id'], arm_token=completion['arm_token'])
    if task_pose_ranking_enabled():
        result['task_departure_penalty_s'] = task_departure_penalty(result)
        result['ranking_total_s'] = ranking_total(result)
        result['ranking_weights'] = dict(position_m_per_s=.20, angle_rad_per_s=1.)
    return result


def alternative_can_improve(baseline, context, extra_compute_s, now=None):
    """Prune only when even a zero-risk direct-return proxy cannot win.

    The stop-to-stop time is subadditive: travel via any observation q cannot
    cost less than direct travel. This bounds the existing ranking proxy, not
    the physical path. A first candidate needing a detour remains searchable
    because an alternative can win by observation order alone.
    """
    if os.environ.get('CARE_TASK_FIRST_RESEARCH') == 'true':
        # One additional original branch, including unsuccessful geometry.
        # The first accepted proposal remains available when search stops.
        calls = context.get('_task_first_comparisons', 0)
        context['_task_first_comparisons'] = calls + 1
        if calls >= 1 or extra_compute_s >= 1.5:
            return False
    if baseline.get('approach_requires_detour') is not False:
        return True
    try:
        rate = float(context.get('ros_per_wall_s', 1.))
        elapsed = float(extra_compute_s)
        total = float(ranking_total(baseline))
        win_fraction = 1. if task_pose_ranking_enabled() else .95
        if not all(math.isfinite(x) for x in (rate, elapsed, total)) or rate <= 0. or min(elapsed, total) < 0.:
            return True
        lower = _rest_time(context['q_start'], context['q_task'], context) / rate
        completion = context.get('completion')
        if completion is None:
            return lower + elapsed < win_fraction * total
        if baseline.get('completion') is None:
            return True
        lower = (lower * min(completion['approach_scale'], completion['rejoin_scale']) +
                 completion['acquisition_wall_s'] +
                 completion.get('return_overhead_wall_s', 0.) +
                 (completion['tail_ros_s'] + completion['hold_ros_s']) / rate)
        total = ranking_total(baseline, completion=True)
        remaining = completion['deadline_monotonic_s'] - (time.monotonic() if now is None else now)
        if remaining <= 0.:
            return False
        if lower + elapsed <= remaining < total:
            return True
        return lower + elapsed < win_fraction * total
    except (KeyError, ValueError, TypeError, FloatingPointError):
        return True


def improves_candidate(candidate, baseline, extra_compute_s):
    """Compare total task-return proxies after observation geometry passes.

    Times are stop-to-stop joint proxies, not collision-free travel times.
    Prefer sampled observation before body entry when both paths were checked.
    Within that class, charge extra computation and require a five-percent
    proxy gain before replacing the original fallback.
    """
    keys = ('approach_s', 'rejoin_s', 'total_s', 'task_pose_error')
    values = [float(c[k]) for c in (candidate, baseline) for k in keys]
    if not all(math.isfinite(v) and v >= 0. for v in values):
        return False
    if not math.isfinite(extra_compute_s) or extra_compute_s < 0.:
        return False
    research = os.environ.get('CARE_TASK_FIRST_RESEARCH') == 'true'
    if research:
        # Geometry order is independent of the optional corridor RPC.
        if ('approach_requires_detour' in candidate and 'approach_requires_detour' in baseline and
                candidate['approach_requires_detour'] != baseline['approach_requires_detour']):
            return not candidate['approach_requires_detour']
        comparable = (candidate.get('corridor_used', False) == baseline.get('corridor_used', False) and
            (not candidate.get('corridor_used', False) or
             candidate['corridor_preview']['map_identity'] == baseline['corridor_preview']['map_identity']))
        if not comparable:
            # Use common information on BOTH sides, never missing=free for one.
            candidate, baseline = dict(candidate), dict(baseline)
            for cost in (candidate, baseline):
                cost['total_s'] -= cost.get('new_observation_proxy_s', 0.)
                cost['corridor_used'] = False
                if cost.get('completion') is not None:
                    cost['completion'] = dict(cost['completion'])
                    cost['completion']['estimate_wall_s'] -= cost.get('new_observation_proxy_s', 0.)
    # Missing information must not look cheaper than a queried corridor.
    if candidate.get('corridor_used',False)!=baseline.get('corridor_used',False):return False
    if candidate.get('corridor_used',False) and (
            candidate['corridor_preview']['map_identity']!=baseline['corridor_preview']['map_identity']):return False
    # Within the original candidate pool, prefer an approach that can observe
    # the actual target points before crossing the execution floor. Keep the
    # first accepted proposal when no such alternative exists. Sampling only
    # ranks proposals; every path still needs the original complete audits.
    if ('approach_requires_detour' in candidate and 'approach_requires_detour' in baseline and
            candidate['approach_requires_detour']!=baseline['approach_requires_detour']):
        return not candidate['approach_requires_detour']
    c, b = candidate.get('completion'), baseline.get('completion')
    if c is not None or b is not None:
        if c is None or b is None:
            return False
        if any(c.get(k) != b.get(k) for k in ('deadline_monotonic_s', 'profile_id', 'arm_token')):
            return False
        cv, bv = c.get('estimate_wall_s'), b.get('estimate_wall_s')
        if any(not isinstance(x, (float, int)) or not math.isfinite(x) or x < 0.
               for x in (cv, bv, c.get('deadline_monotonic_s'))):
            return False
        remaining = c['deadline_monotonic_s'] - time.monotonic()
        if remaining <= 0.:
            return False
        # Candidate computation has already elapsed. Charge it as opportunity
        # cost when comparing with the original immediately executable fallback.
        if (cv + extra_compute_s <= remaining) != (bv <= remaining):
            return cv + extra_compute_s <= remaining
        if task_pose_ranking_enabled():
            return ranking_total(candidate, True) + extra_compute_s < ranking_total(baseline, True) - 1e-6
        return cv + extra_compute_s < .95 * bv
    # FK endpoint distance is not a path cost or a safety constraint. Likewise
    # one longer leg may be outweighed by the other leg. Full execution audits
    # remain mandatory for every selected proposal.
    if task_pose_ranking_enabled():
        return ranking_total(candidate)+extra_compute_s < ranking_total(baseline)-1e-6
    return candidate['total_s']+extra_compute_s < .95*baseline['total_s']
