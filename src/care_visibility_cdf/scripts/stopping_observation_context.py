"""Bounded, explicitly owned observation work across candidate replacements.

Contexts and reports in this module are steering inputs only. In particular a
context report must never pass a candidate/execution evidence validator.
"""
import copy
import hashlib
import json
import math

from stopping_observation_demands import validate_report


def context_key(context):
    return (context['owner_session'], context['generation'], context['context_id'])


def context_path(context):
    return 'observation-context:' + hashlib.sha256(json.dumps(
        context_key(context), separators=(',', ':')).encode()).hexdigest()


class ObservationContext:
    """Single-owner ordered events, with heartbeat expiry and terminal epochs.

    The owner supplies a relevance box, not the candidate observer. Changing
    that box or resuming after cancellation/expiry requires a new generation.
    Events use ROS time; a backwards clock closes the current generation.
    """
    def __init__(self, max_heartbeat_s=2., max_sessions=16):
        self.max_heartbeat_ns = round(max_heartbeat_s * 1e9)
        self.max_sessions = max_sessions
        self.current = None
        self.sessions = set()
        self.closed = False
        self.last_now = 0
        self.reason = None

    def snapshot(self, now_ns):
        if now_ns < self.last_now:
            self.invalidate('context_clock_reversed')
        self.last_now = now_ns
        if self.current and now_ns > self.current['valid_until_ns']:
            self.invalidate('context_heartbeat_expired')
        return copy.deepcopy(self.current) if self.current and not self.closed else None

    def invalidate(self, reason):
        self.closed = True
        self.reason = reason

    def update(self, event, now_ns):
        self.snapshot(now_ns)
        e = copy.deepcopy(event)
        if (e.get('schema') != 1 or type(e.get('active')) is not bool or
                any(not isinstance(e.get(k), str) or not 1 <= len(e[k]) <= 128
                    for k in ('owner_session', 'context_id')) or
                any(type(e.get(k)) is not int or e[k] <= 0
                    for k in ('generation', 'event_seq', 'stamp_ns', 'valid_until_ns')) or
                not e['stamp_ns'] <= now_ns <= e['valid_until_ns'] or
                not 0 < e['valid_until_ns']-e['stamp_ns'] <= self.max_heartbeat_ns):
            raise ValueError('invalid_observation_context_event')
        lo, hi = e.get('region_min'), e.get('region_max')
        if (not all(isinstance(v, list) and len(v) == 3 and all(
                type(x) in (int, float) and math.isfinite(x) for x in v) for v in (lo, hi)) or
                any(a > b for a, b in zip(lo, hi))):
            raise ValueError('invalid_observation_context_region')
        old = self.current
        if old:
            if e['stamp_ns'] < old['stamp_ns']:
                raise ValueError('observation_context_event_regressed')
            if e['owner_session'] == old['owner_session']:
                if e['event_seq'] <= old['event_seq'] or e['generation'] < old['generation']:
                    raise ValueError('observation_context_event_regressed')
                if e['generation'] == old['generation']:
                    if (context_key(e) != context_key(old) or lo != old['region_min'] or hi != old['region_max']):
                        raise ValueError('observation_context_epoch_mutated')
                    if self.closed and e['active']:
                        raise ValueError('observation_context_epoch_closed')
            elif e['owner_session'] in self.sessions or e['stamp_ns'] <= old['stamp_ns']:
                raise ValueError('observation_context_old_owner')
        if e['owner_session'] not in self.sessions and len(self.sessions) >= self.max_sessions:
            raise ValueError('observation_context_session_capacity')
        same = old is not None and context_key(e) == context_key(old)
        supplied_start = e.get('started_ns', old['started_ns'] if same else e['stamp_ns'])
        if (type(supplied_start) is not int or not 0 < supplied_start <= e['stamp_ns'] or
                (same and supplied_start != old['started_ns'])):
            raise ValueError('invalid_observation_context_start')
        # A late subscriber may first receive a heartbeat. Its owner's explicit
        # start is immutable and must not become this heartbeat's timestamp.
        e['started_ns'] = supplied_start
        self.sessions.add(e['owner_session'])
        self.current = e
        self.closed = not e['active']
        self.reason = None if e['active'] else 'context_cancelled'
        return self.snapshot(now_ns)


def validate_context_report(report, context, now_s):
    validate_report(report, now_s)
    supplied = report.get('observation_context')
    if (context is None or not isinstance(supplied, dict) or
            context_key(supplied) != context_key(context) or
            supplied.get('started_ns') != context['started_ns'] or
            supplied.get('region_min') != context['region_min'] or
            supplied.get('region_max') != context['region_max'] or
            report.get('evidence_kind') != 'context_observation_demand' or
            report.get('execution_eligible') is not False or
            not isinstance(report.get('trajectory_sha256'), str) or len(report['trajectory_sha256']) != 64 or
            report['trajectory_stamp_ns'] < context['started_ns']):
        raise ValueError('unbound_observation_context_report')
    if context.get('candidate_binding'):
        geometry = report.get('map_geometry')
        if (report.get('task_context') != context['candidate_binding'] or
                not isinstance(geometry, list) or not geometry or geometry[0] != context['frame_id']):
            raise ValueError('unbound_task_observation_report')
        scope = context.get('map_scope')
        if scope is not None and scope != dict(instance=report['map_identity'][0],
                epoch=report['map_identity'][1], geometry=geometry):
            raise ValueError('task_observation_map_scope_changed')
    return report


class ContextObservationDemandQueue:
    """Keep a finite spatial union until new sensor evidence or explicit exit.

    Revision changes do not reset work. Instance/epoch/geometry changes close
    this ledger, requiring a new owner generation. Sources are bounded samples
    with an exact total count; they are never execution certificates.
    """
    def __init__(self, capacity=256):
        self.capacity = capacity
        self.reset()

    def reset(self):
        self.pending = {}
        self.owner = None
        self.map_scope = None
        self.last_snapshot = 0
        self.invalidated = False
        self.resolved = 0

    def update(self, report, context, now_s):
        validate_context_report(report, context, now_s)
        owner = context_key(context)
        if owner != self.owner:
            self.reset()
            self.owner = owner
        scope = [report['map_identity'][:2], report['map_geometry']]
        if self.invalidated or (self.map_scope is not None and scope != self.map_scope):
            self.invalidated = True
            self.pending = {}
            raise ValueError('observation_context_map_changed')
        if report['snapshot_ns'] < self.last_snapshot:
            raise ValueError('observation_context_snapshot_regressed')
        pending = copy.deepcopy(self.pending)
        resolved = 0
        for cell in report['cells']:
            if not all(a <= p <= b for a, p, b in zip(context['region_min'], cell['point'], context['region_max'])):
                continue
            gid = cell['grid_index']
            old = pending.get(gid)
            if old and old['point'] != cell['point']:
                raise ValueError('observation_context_grid_changed')
            until = max(cell['required_until_s'], old['required_until_s']) if old else cell['required_until_s']
            # Coverage must include both previously promised and newly requested
            # horizons. Bootstrap or replayed source time cannot complete work.
            if cell['action'] == 'covered':
                if old and (cell['provenance'] in (1, 3) and cell['last_seen_s'] > old['last_seen_s'] and
                        cell['sensor_expiry_s'] is not None and cell['sensor_expiry_s'] >= until and
                        cell['confidence_now'] >= .8 and 0 <= cell['occupancy'] <= .5 and cell['inside_map'] == 1):
                    del pending[gid]
                    resolved += 1
                elif old:
                    old['required_until_s'] = until
                continue
            if cell['action'] not in ('first_observation', 'refresh_observation', 'blocked_occupied'):
                raise ValueError('invalid_observation_context_action')
            value = copy.deepcopy(cell)
            value.update(path_id=context_path(context), map_identity=report['map_identity'],
                map_geometry=report['map_geometry'], requested_snapshot_ns=report['snapshot_ns'],
                required_until_s=until, source_count=1, source_paths=[report['path_id']],
                first_requested_ns=report['snapshot_ns'])
            if old:
                value['first_use_s'] = min(old['first_use_s'], cell['first_use_s'])
                value['first_requested_ns'] = old['first_requested_ns']
                for field in ('observation_deadline_s', 'latest_start_s'):
                    if field in old:
                        value[field] = min(old[field], value.get(field, old[field]))
                value['source_paths'] = (old['source_paths'] + [report['path_id']])[-8:]
                value['source_count'] = old['source_count'] + 1
                # Insufficient new observations cannot move the completion
                # baseline forward and make a replay count as later evidence.
                value['last_seen_s'] = max(old['last_seen_s'], cell['last_seen_s'])
            value['scheduling_overdue'] = value.get('latest_start_s', -math.inf) <= now_s
            pending[gid] = value
        if len(pending) > self.capacity:
            raise ValueError('observation_context_demand_capacity')
        self.pending = pending
        self.map_scope = scope
        self.last_snapshot = report['snapshot_ns']
        self.resolved += resolved
        return self.ordered()

    def ordered(self):
        return sorted(copy.deepcopy(list(self.pending.values())),
                      key=lambda c: (c.get('latest_start_s', -math.inf), c['grid_index']))
