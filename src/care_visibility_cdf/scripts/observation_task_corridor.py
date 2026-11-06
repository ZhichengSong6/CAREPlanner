"""Ordered upcoming-task risk estimates, without execution/map authority."""
import numpy as np
from observation_corridor_cost import snapshot_cost
from validate_visibility_oracle import DEFAULT_JOINT_NAMES, fk_transform


def bounded_task_reference(q_rows, times, q_start, source, horizon_s=1., limit=8):
    """A shared nearest-reference anchor and a bounded ordered future segment.

    The anchor is only an association for proposal scoring. It does not advance
    the planner's nominal-progress cursor or assert that intervening work is done.
    """
    q = np.asarray(q_rows, dtype=float).reshape(-1, 7)
    t = np.asarray(times, dtype=float).reshape(-1)
    start = np.asarray(q_start, dtype=float).reshape(7)
    if (len(q) < 2 or len(t) != len(q) or not np.isfinite(q).all() or
            not np.isfinite(t).all() or not np.isfinite(start).all() or
            np.any(np.diff(t) <= 0.) or not 0. < horizon_s <= 1. or not 2 <= limit <= 8):
        raise ValueError('invalid upcoming task reference')
    anchor = int(np.argmin(np.sum((q-start)**2, axis=1)))
    end = min(float(t[-1]), float(t[anchor])+horizon_s)
    sample_t = np.linspace(t[anchor], end, limit) if end > t[anchor] else np.array([end])
    sample_q = np.stack([np.interp(sample_t, t, q[:, j]) for j in range(7)], axis=1)
    return dict(schema=1, source=source, anchor_index=anchor,
        anchor_kind='nearest_original_task_reference', horizon_s=horizon_s,
        times=sample_t.tolist(), q=sample_q.tolist())


def ordered_body_probes(runtime, q_rows, times, resolution=.05, limit=96):
    """Keep first-touch path order while deduplicating bounded RPC points."""
    q_rows = np.asarray(q_rows, dtype=float).reshape(-1, 7)
    times = np.asarray(times, dtype=float).reshape(-1)
    if (not len(q_rows) or len(q_rows) > 12 or len(q_rows) != len(times) or
            not np.isfinite(q_rows).all() or not np.isfinite(times).all() or
            np.any(np.diff(times) < 0.) or resolution != .05 or not 1 <= limit <= 96):
        raise ValueError('invalid ordered body probe path')
    first = {}
    for sample, (q, stamp) in enumerate(zip(q_rows, times)):
        qmap = dict(zip(DEFAULT_JOINT_NAMES, q))
        frames = {}
        for primitive, chain in zip(runtime.primitives, runtime.collision_chains):
            link = primitive['link']
            if link in ('base_link', 'link1'):
                continue
            if link not in frames:
                frames[link] = fk_transform(chain, qmap)
            transform = frames[link] @ primitive['origin']
            geometry = primitive['geometry']
            half = (np.asarray(geometry.size)*.5 if primitive['type'] == 'box' else
                np.asarray([geometry.radius, geometry.radius, geometry.length*.5])
                if primitive['type'] == 'cylinder' else np.full(3, geometry.radius))
            local = [np.zeros(3)]
            for axis in range(3):
                delta = np.zeros(3)
                delta[axis] = half[axis]
                local.extend([delta, -delta])
            for point in local:
                p = transform[:3, :3] @ point + transform[:3, 3]
                if not np.isfinite(p).all():
                    raise ValueError('nonfinite body preview')
                cell = tuple(np.rint(p/resolution).astype(int))
                first.setdefault(cell, (float(stamp), sample))
    cells = list(first)
    if len(cells) > limit:
        cells = [cells[i] for i in np.linspace(0, len(cells)-1, limit, dtype=int)]
    return dict(points=[tuple(float(x)*resolution for x in cell) for cell in cells],
        first_time_s=[first[cell][0] for cell in cells],
        first_sample=[first[cell][1] for cell in cells], samples=len(q_rows))


def candidate_corridors(runtime, q_candidate, context):
    """Approach the view, then return to the same original task reference.

    q_candidate is the completed-view proposal, not a short prefix's stopped q.
    Real observation completion remains owned by confidence-map feedback.
    """
    from observation_task_rejoin import _rest_time
    reference = context['upcoming_task_reference']
    q = np.asarray(q_candidate, dtype=float).reshape(7)
    task_q = np.asarray(reference['q'], dtype=float).reshape(-1, 7)
    task_t = np.asarray(reference['times'], dtype=float).reshape(-1)
    if not 1 <= len(task_q) <= 8 or len(task_t) != len(task_q):
        raise ValueError('invalid bounded task corridor')
    approach_s = _rest_time(context['q_start'], q, context)
    return_s = _rest_time(q, task_q[0], context)
    approach = ordered_body_probes(runtime, np.linspace(context['q_start'], q, 4),
        np.linspace(0., approach_s, 4))
    # Four points on the return-to-reference leg, then the captured ordered
    # task segment. Omit the duplicate anchor shared by the two pieces.
    return_q = np.vstack([np.linspace(q, task_q[0], 4), task_q[1:]])
    return_t = np.r_[np.linspace(0., return_s, 4), return_s+task_t[1:]-task_t[0]]
    rejoin = ordered_body_probes(runtime, return_q, return_t)
    return dict(approach=approach, rejoin=rejoin, reference=reference)


def paired_corridor_costs(runtime, candidates, context, query, now_ns, threshold):
    """Both alternatives use one exact map snapshot or both lose this feature."""
    if len(candidates) != 2 or 'upcoming_task_reference' not in context:
        return [dict(status='unavailable', reason='no_bounded_pair_or_reference') for _ in candidates]
    paths = [candidate_corridors(runtime, q, context) for q in candidates]
    points = list(dict.fromkeys(p for path in paths for leg in ('approach', 'rejoin')
        for p in path[leg]['points']))
    if not points:
        return [dict(status='unavailable', reason='no_body_probes') for _ in candidates]
    response = query(points)
    captured_now = now_ns()
    costs = [snapshot_cost(response, points, path['approach']['points'], path['rejoin']['points'],
        captured_now, threshold) for path in paths]
    if any(cost.get('status') != 'valid' for cost in costs):
        return [dict(status='unavailable', reason='common_snapshot_unavailable') for _ in candidates]
    for cost, path in zip(costs, paths):
        cost.update(reference_source=path['reference']['source'],
            reference_anchor=path['reference']['anchor_index'],
            corridor_kind='completed_view_to_upcoming_original_task',
            path_order={leg: {k: path[leg][k] for k in ('first_time_s', 'first_sample', 'samples')}
                for leg in ('approach', 'rejoin')},
            motion_authorized=False, predicted_map_update=False)
    return costs


def replace_corridor_costs(costs, previews):
    """Replace, never add to, the existing corridor penalty on both candidates.

    A partially missing/stale response drops the map feature for both sides.
    Approach/return travel estimates and task-pose weights remain unchanged.
    """
    import copy
    result = copy.deepcopy(costs)
    valid = len(result) == len(previews) == 2
    identities = []
    if valid:
        for preview in previews:
            try:
                valid = valid and preview['status'] == 'valid'
                identities.append((tuple(preview['map_identity']), preview['map_revision'],
                                   preview['snapshot_stamp_ns']))
                for leg in ('approach', 'rejoin'):
                    risk = float(preview[leg]['risk_fraction'])
                    valid = valid and np.isfinite(risk) and 0. <= risk <= 1.
            except (KeyError, TypeError, ValueError):
                valid = False
        valid = valid and len(identities) == 2 and identities[0] == identities[1]
    for index, cost in enumerate(result):
        old = float(cost.pop('new_observation_proxy_s', 0.))
        cost['total_s'] -= old
        if cost.get('completion') is not None:
            cost['completion']['estimate_wall_s'] -= old / cost.get('ros_per_wall_s', 1.)
        cost['corridor_used'] = bool(valid)
        cost['corridor_preview'] = (copy.deepcopy(previews[index]) if valid else
            dict(status='unavailable', reason='paired_common_information_fallback'))
        if valid:
            penalty = (cost['approach_s']*previews[index]['approach']['risk_fraction'] +
                       cost['rejoin_s']*previews[index]['rejoin']['risk_fraction'])
            cost['new_observation_proxy_s'] = penalty
            cost['total_s'] += penalty
            if cost.get('completion') is not None:
                cost['completion']['estimate_wall_s'] += penalty / cost.get('ros_per_wall_s', 1.)
        if 'ranking_total_s' in cost:
            cost['ranking_total_s'] = cost['total_s'] + cost.get('task_departure_penalty_s', 0.)
    return result
