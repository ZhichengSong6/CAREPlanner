#!/usr/bin/env python3
"""Look ahead over complete candidate prefix boxes; produce steering requests.

This module never authorizes motion. A current complete stopping check and its
commit-time recheck remain mandatory, even after a requested observation.
"""
import copy
import hashlib
import json
import math
import time


def _finite(values, size):
    return (isinstance(values, (list, tuple)) and len(values) == size and
            all(type(x) in (int, float) and math.isfinite(x) for x in values))


def _geometry(e):
    return [e[k] for k in ('frame', 'resolution', 'map_min', 'map_max', 'grid_size')]


def validate_report(report, now_s, trajectory_stamp_ns=None, max_age_s=.5):
    if (report.get('schema') != 1 or report.get('runtime_admission') is not False or
            report.get('planning_only') is not True or report.get('stable_version') is not True or
            not isinstance(report.get('path_id'), str) or not report['path_id'] or
            type(report.get('trajectory_stamp_ns')) is not int or report['trajectory_stamp_ns'] <= 0 or
            not math.isfinite(now_s) or type(report.get('snapshot_ns')) is not int or
            not 0 <= int(round(now_s*1e9))-report['snapshot_ns'] <= int(round(max_age_s*1e9)) or
            (trajectory_stamp_ns is not None and report['trajectory_stamp_ns'] != trajectory_stamp_ns)):
        raise ValueError('stale_or_unbound_stopping_observation_report')
    identity = report.get('map_identity')
    if not isinstance(identity, (list, tuple)) or len(identity) != 3 or not identity[0]:
        raise ValueError('missing_stopping_observation_map_identity')
    points = report.get('cells')
    if not isinstance(points, list) or not 1 <= len(points) <= 250000:
        raise ValueError('invalid_stopping_observation_cells')
    seen = set()
    for cell in points:
        gid = cell.get('grid_index')
        if (type(gid) is not int or gid < 0 or gid in seen or not _finite(cell.get('point'), 3) or
                not all(math.isfinite(cell.get(k, math.nan)) for k in ('first_use_s', 'required_until_s')) or
                cell['required_until_s'] < cell['first_use_s']):
            raise ValueError('invalid_stopping_observation_cell')
        seen.add(gid)
        if cell.get('action') in ('first_observation', 'refresh_observation'):
            if (not all(math.isfinite(cell.get(k, math.nan)) for k in ('observation_deadline_s', 'latest_start_s')) or
                    cell['latest_start_s'] > cell['observation_deadline_s']):
                raise ValueError('invalid_stopping_observation_deadline')
    return report


def query_prefix_observation_demands(*, segments, geometry, query, now_ns, path_id,
        trajectory_stamp_ns, observation_timing, wall_budget_s=.5, wall_clock=time.monotonic,
        retained_cells=None, retained_map_scope=None):
    """Query the union of supplied whole segment boxes and their stopping sets.

    q_low/q_high must enclose the complete proposed segment, not just endpoints
    of an unspecified interpolator. stop_travel_rad must cover every state in
    it. These are candidate construction inputs, never qualified motion proof.
    Geometry is constructed before binding a content version. The subsequent
    point query and version recheck bind fresh content without caching freedom.
    """
    if (not isinstance(path_id, str) or not path_id or type(trajectory_stamp_ns) is not int or
            trajectory_stamp_ns <= 0 or not 0 <= len(segments) <= 32 or
            (not segments and not retained_cells) or
            not math.isfinite(wall_budget_s) or wall_budget_s <= 0):
        raise ValueError('invalid_stopping_prefix_request')
    timing_keys = ('motion_s', 'acquisition_s', 'verification_s', 'stop_reserve_s')
    if any(not math.isfinite(observation_timing.get(k, math.nan)) or observation_timing[k] < 0 for k in timing_keys):
        raise ValueError('missing_observation_route_timing')
    lead = sum(observation_timing[k] for k in timing_keys)
    started = wall_clock()
    def checkpoint():
        if wall_clock()-started >= wall_budget_s:
            raise ValueError('stopping_prefix_planning_budget_exhausted')
    def call(*args):
        checkpoint()
        remaining = wall_budget_s-(wall_clock()-started)
        result = query.call_with_timeout(remaining, *args) if hasattr(query, 'call_with_timeout') else query(*args)
        checkpoint()
        return result
    meta = call([], 0., None)
    if not meta.get('valid') or meta.get('clock_fault'):
        raise ValueError('invalid_stopping_prefix_metadata')
    if retained_map_scope is not None and retained_map_scope != [
            [meta.get('instance'), meta.get('epoch')], _geometry(meta)]:
        raise ValueError('observation_context_map_changed')
    cells = {}
    for segment in segments:
        checkpoint()
        lo, hi, travel = [segment.get(k) for k in ('q_low', 'q_high', 'stop_travel_rad')]
        first, until = segment.get('first_use_s'), segment.get('required_until_s')
        if (not all(_finite(v, 7) for v in (lo, hi, travel)) or
                any(a > b for a, b in zip(lo, hi)) or any(v < 0 for v in travel) or
                not all(type(v) in (int, float) and math.isfinite(v) for v in (first, until)) or first > until):
            raise ValueError('invalid_stopping_prefix_segment')
        limits=getattr(geometry,'configuration_limits',None)
        if limits is None or len(limits)!=7:
            raise ValueError('missing_stopping_prefix_joint_limits')
        if any(a<bound[0] or b>bound[1] for a,b,bound in zip(lo,hi,limits)):
            raise ValueError('stopping_prefix_joint_limit')
        center = [(a+b)/2 for a, b in zip(lo, hi)]
        expanded = [(b-a)/2+t for a, b, t in zip(lo, hi, travel)]
        enclosure = geometry.raster(center, expanded, meta, .020, subdivisions=1, check_budget=checkpoint)
        if not enclosure['inside']:
            raise ValueError('stopping_prefix_may_leave_map')
        for gid, point in zip(enclosure['grid_indices'], enclosure['points']):
            point = list(map(float, point))
            if gid not in cells:
                cells[gid] = dict(grid_index=gid, point=point, first_use_s=first, required_until_s=until)
            else:
                if cells[gid]['point'] != point:
                    raise ValueError('stopping_prefix_grid_identity_changed')
                cells[gid]['first_use_s'] = min(first, cells[gid]['first_use_s'])
                cells[gid]['required_until_s'] = max(until, cells[gid]['required_until_s'])
    # Requery unresolved cells omitted by a new hypothetical candidate. These
    # retained planning points carry no candidate execution evidence.
    if retained_cells is not None:
        if len(retained_cells) > 256:
            raise ValueError('observation_context_demand_capacity')
        for cell in retained_cells:
            gid = cell['grid_index']
            value = {k: copy.deepcopy(cell[k]) for k in
                     ('grid_index', 'point', 'first_use_s', 'required_until_s')}
            if gid in cells:
                if cells[gid]['point'] != value['point']:
                    raise ValueError('observation_context_grid_changed')
                cells[gid]['first_use_s'] = min(cells[gid]['first_use_s'], value['first_use_s'])
                cells[gid]['required_until_s'] = max(cells[gid]['required_until_s'], value['required_until_s'])
            else:
                cells[gid] = value
    ordered = [cells[g] for g in sorted(cells)]
    evidence = call([c['point'] for c in ordered], 0., None)
    identity = tuple(evidence.get(k) for k in ('instance', 'epoch', 'revision'))
    now = now_ns()
    keys = ('inside_map', 'provenance', 'confidence_now', 'occupancy', 'last_seen_s', 'model_free_at_horizon')
    if (not evidence.get('valid') or evidence.get('clock_fault') or
            evidence.get('instance') != meta.get('instance') or _geometry(evidence) != _geometry(meta) or
            not 0 <= now-evidence.get('snapshot_ns', 0) <= 500000000 or
            any(len(evidence.get(k, [])) != len(ordered) for k in keys)):
        raise ValueError('invalid_stopping_prefix_evidence')
    tau = evidence['temporal_decay_time']
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError('invalid_stopping_prefix_memory')
    for i, cell in enumerate(ordered):
        p, last = evidence['provenance'][i], evidence['last_seen_s'][i]
        cell.update(provenance=p, last_seen_s=last, confidence_now=evidence['confidence_now'][i],
                    inside_map=evidence['inside_map'][i], occupancy=evidence['occupancy'][i])
        good = (p in (1, 2, 3) and cell['inside_map'] == 1 and
                math.isfinite(cell['confidence_now']) and cell['confidence_now'] >= .8 and
                math.isfinite(cell['occupancy']) and 0 <= cell['occupancy'] <= .5 and
                evidence['model_free_at_horizon'][i] == 1)
        expiry = None
        if p in (1, 3):
            if not math.isfinite(last) or last < 0 or last > evidence['snapshot_ns']*1e-9:
                raise ValueError('future_or_invalid_prefix_observation')
            expiry = last-tau*math.log(.8)
        cell['sensor_expiry_s'] = expiry
        if math.isfinite(cell['occupancy']) and cell['occupancy'] > .5:
            action = 'blocked_occupied'
        elif not good:
            action = 'refresh_observation' if p == 1 and expiry is not None else 'first_observation'
        elif p == 1 and expiry < cell['required_until_s']:
            action = 'refresh_observation'
        else:
            action = 'covered'
        cell['action'] = action
        if action in ('first_observation', 'refresh_observation'):
            deadline = (cell['first_use_s'] if action == 'first_observation' else
                        max(cell['first_use_s'], expiry-observation_timing['stop_reserve_s']))
            cell.update(observation_deadline_s=deadline, latest_start_s=deadline-lead,
                        observation_timing=copy.deepcopy(observation_timing),
                        scheduling_overdue=deadline-lead <= now*1e-9)
    recheck = call([], 0., identity)
    if (not recheck.get('valid') or recheck.get('clock_fault') or
            tuple(recheck.get(k) for k in ('instance', 'epoch', 'revision')) != identity or
            _geometry(recheck) != _geometry(evidence) or
            not evidence['snapshot_ns'] <= recheck.get('snapshot_ns', 0) <= now_ns()):
        raise ValueError('stopping_prefix_version_changed')
    report = dict(schema=1, planning_only=True, runtime_admission=False, stable_version=True,
        path_id=path_id, trajectory_stamp_ns=trajectory_stamp_ns, map_identity=list(identity),
        snapshot_ns=evidence['snapshot_ns'], map_geometry=_geometry(evidence), cells=ordered,
        construction_sha256=hashlib.sha256(json.dumps(segments, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        wall_elapsed_s=wall_clock()-started)
    checkpoint()
    return validate_report(report, now_ns()*1e-9)


class ObservationDemandQueue:
    """Keep unresolved spatial requests until an actual later map query covers them.

    Disappearing from one hypothetical prefix does not resolve a request. An
    explicit path retirement is separate from observing its cells.
    """
    def __init__(self, capacity=256):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('invalid_observation_demand_capacity')
        self.capacity = capacity
        self.pending = {}
        self.last_snapshot = {}

    def update(self, report, now_s, trajectory_stamp_ns=None):
        validate_report(report, now_s, trajectory_stamp_ns)
        path = report['path_id']
        if report['snapshot_ns'] < self.last_snapshot.get(path, 0):
            raise ValueError('stopping_observation_report_regressed')
        updated = copy.deepcopy(self.pending)
        for cell in report['cells']:
            key = (path, cell['grid_index'])
            previous = updated.get(key)
            if previous and (previous['map_identity'][:2] != report['map_identity'][:2] or
                             previous['map_geometry'] != report['map_geometry'] or previous['point'] != cell['point']):
                raise ValueError('observation_demand_map_identity_changed')
            if cell['action'] == 'covered':
                # Resolution needs a real later sensor observation, adequate
                # horizon, and no replacement of a missing source by bootstrap.
                if previous and (cell['provenance'] in (1, 3) and
                        cell['last_seen_s'] > previous['last_seen_s'] and
                        cell['sensor_expiry_s'] >= previous['required_until_s'] and
                        cell['confidence_now'] >= .8 and 0 <= cell['occupancy'] <= .5 and cell['inside_map'] == 1):
                    del updated[key]
            elif cell['action'] in ('first_observation', 'refresh_observation', 'blocked_occupied'):
                value = dict(copy.deepcopy(cell), path_id=path, map_identity=report['map_identity'],
                             map_geometry=report['map_geometry'], requested_snapshot_ns=report['snapshot_ns'])
                if previous:
                    value['required_until_s'] = max(previous['required_until_s'], value['required_until_s'])
                    value['first_use_s'] = min(previous['first_use_s'], value['first_use_s'])
                    if value['last_seen_s'] <= previous['last_seen_s']:
                        for field in ('observation_deadline_s', 'latest_start_s'):
                            if field in previous and field in value:
                                value[field] = min(previous[field], value[field])
                updated[key] = value
        if len(updated) > self.capacity:
            raise ValueError('observation_demand_capacity_exhausted')
        self.pending = updated
        self.last_snapshot[path] = report['snapshot_ns']
        return self.ordered(path)

    def ordered(self, path_id=None):
        return sorted([copy.deepcopy(v) for (p, _), v in self.pending.items() if path_id is None or p == path_id],
            key=lambda v: (v.get('latest_start_s', -math.inf), v['grid_index']))

    def retire(self, path_id):
        self.pending = {k: v for k, v in self.pending.items() if k[0] != path_id}
        self.last_snapshot.pop(path_id, None)


def exact_region_owner(obligations, points):
    """A nearby region cannot own a stopping demand for a different cell."""
    needed={tuple(map(float,p)) for p in points}
    return next((ob for ob in obligations if needed.issubset({tuple(map(float,p)) for p in ob.get('points',[])})),None)


def inherit_stopping_request(parent, child):
    request=parent.get('stopping_prefix_request')
    if not request:return child
    required={tuple(p) for p in request['points']}
    overlap=[list(map(float,p)) for p in child['points'] if tuple(map(float,p)) in required]
    if overlap:
        child['stopping_prefix_request']=dict(copy.deepcopy(request),points=overlap)
        child['stopping_prefix_owned']=parent.get('stopping_prefix_owned',False)
        child['deadline_abs_s']=min(child['deadline_abs_s'],parent['deadline_abs_s'])
    return child
