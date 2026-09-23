"""Opt-in, bounded steering replacement for a dependency or repeated final VBC.

The C++ planner owns the attempt/segment ledger. This module neither commits
trajectories nor removes obligations/witnesses. All replacement poses must pass
the existing live-map solver, final GCDF, exact VBC and tracker path.
"""
import json
import threading
import time
from pathlib import Path

import numpy as np
from bounded_visibility_recovery import region_key
from observation_identity import observation_region_token, observation_token


SENSOR_MOUNT_GROUPS = ((0, 1), (2, 3), (4, 5), (6,), (7,))
HIGH_WITNESS_SENSOR_PRIORITY = (7, 6)
SENSOR_MOUNT_GROUP = {
    sensor_id: group_id
    for group_id, group in enumerate(SENSOR_MOUNT_GROUPS)
    for sensor_id in group
}


def sensor_mount_group_id(sensor_id):
    sensor_id = int(sensor_id)
    if sensor_id not in SENSOR_MOUNT_GROUP:
        raise ValueError("replacement sensor id outside [0,7]")
    return SENSOR_MOUNT_GROUP[sensor_id]


def high_witness_sensor_preference(points_xyz, enabled, minimum_z):
    """Return the wrist-mounted sensor order for an all-high witness set."""
    points = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    minimum_z = float(minimum_z)
    if points.shape[0] < 1 or not np.all(np.isfinite(points)):
        raise ValueError("high-witness points must be finite xyz rows")
    if not np.isfinite(minimum_z):
        raise ValueError("high-witness minimum z must be finite")
    if bool(enabled) and bool(np.all(points[:, 2] >= minimum_z)):
        return list(HIGH_WITNESS_SENSOR_PRIORITY)
    return []


def replacement_mount_group_pool(ranked, excluded, ee_sensor_priority_first=False):
    """Prefer an untried mount group, then an opposite-facing mate if one exists."""
    excluded = {int(sensor_id) for sensor_id in excluded}
    if any(sensor_id < 0 or sensor_id >= 8 for sensor_id in excluded):
        raise ValueError("replacement sensor id outside [0,7]")
    remaining = [(int(rank), int(sensor_id)) for rank, sensor_id in ranked
                 if int(sensor_id) not in excluded]
    tried_groups = {sensor_mount_group_id(sensor_id) for sensor_id in excluded}
    untried_groups = [
        row for row in remaining
        if sensor_mount_group_id(row[1]) not in tried_groups]
    ee_priority = [
        row for row in remaining if row[1] in HIGH_WITNESS_SENSOR_PRIORITY]
    phase = ('high_witness_ee_priority'
             if ee_sensor_priority_first and ee_priority else
             'untried_mount_group' if untried_groups else 'paired_sensor_fallback')
    pool = (ee_priority if ee_sensor_priority_first and ee_priority else
            untried_groups if untried_groups else remaining)
    return pool, {
        'phase': phase,
        'mount_groups': [list(group) for group in SENSOR_MOUNT_GROUPS],
        'tried_sensor_ids': sorted(excluded),
        'tried_mount_groups': sorted(tried_groups),
        'eligible_sensor_ids': [sensor_id for _, sensor_id in pool],
    }


class CandidateReplacement:
    def __init__(self, node, now, send, log):
        self.node, self.now, self.send, self.log = node, now, send, log
        self.lock = threading.RLock()
        self.pending = None
        self.grant = None
        self.serial = 0
        self.families = {}
        self.poses = {}
        # A live blocker obligation may preempt the current target once per
        # parent/child lineage.  Do not evict entries within a mode epoch:
        # eviction would reopen the O4<->O5 scheduling loop this ledger closes.
        self.owner_promotion_epoch = None
        self.owner_promotions = set()
        # A planner-authenticated QP failure owns its recovery target through
        # reservation, generation and the replacement target's hard-QP result.
        # New blockers remain live, but cannot invalidate the in-flight sensor
        # experiment by changing the stack top before it is evaluated.
        self.qp_owner_lock = None

    def locked_qp_owner_id(self):
        with self.lock:
            lock = self.qp_owner_lock
            if lock is None:
                return None
            if time.monotonic() >= lock['expires_wall']:
                self._release_qp_owner_locked('owner_lock_timeout')
                return None
            return int(lock['owner_id'])

    def _release_qp_owner_locked(self, reason):
        lock = self.qp_owner_lock
        if lock is None:
            return
        self.qp_owner_lock = None
        self.log('qp_replacement_owner_released', reason=reason,
                 active_id=int(lock['owner_id']), stage=lock['stage'],
                 observation_token=lock['observation_token'])

    def _arm_qp_owner_locked(self, event, restored_from=None):
        owner_id = int(event['obligation_id'])
        current = self.qp_owner_lock
        if current is not None and int(current['owner_id']) != owner_id:
            return False
        self.qp_owner_lock = dict(
            owner_id=owner_id,
            mode_epoch=int(event['mode_epoch']),
            observation_token=str(event['observation_token']),
            stage='reservation',
            expires_wall=time.monotonic()+4.0)
        if restored_from is not None:
            self.log('qp_replacement_owner_restored', active_id=owner_id,
                     displaced_id=int(restored_from),
                     observation_token=event['observation_token'])
        return True

    def observe_planner_summary(self, text):
        """Release only after the replacement q_vis hard-QP candidate exists."""
        try:
            fields = {}
            for word in text.split():
                if '=' not in word:
                    continue
                key, value = word.split('=', 1)
                if key in fields:
                    return
                fields[key] = value
            event = fields['event']
            token = fields['observation_token']
        except (KeyError, ValueError):
            return
        with self.lock:
            lock = self.qp_owner_lock
            if (lock is None or lock['stage'] != 'hard_qp' or
                    token != lock['observation_token']):
                return
            if event == 'candidate_published':
                self._release_qp_owner_locked(
                    'replacement_hard_qp_candidate_published')
            elif event in ('repair_qp_candidate_exhausted_hold',
                           'repair_candidate_replacement_exhausted_hold'):
                self._release_qp_owner_locked(event)

    def _queue_locked(self, event, ob, stack, published, reason, owners):
        lineage = (int(event['mode_epoch']), int(ob['id']),
                   ob.get('generation_event_id', 'legacy'))
        if lineage not in self.families and len(self.families) >= 64:
            return False
        tried = self.families.setdefault(lineage, set())
        sensor = int(ob.get('per_sensor_selected_sensor_id', -1))
        if sensor >= 0:
            tried.add(sensor)
        if len(tried) >= 8:
            return False
        self.serial += 1
        tracked_ids = sorted(set(stack) | set(owners))
        self.pending = dict(operation='replace', event=event.copy(), request_id=self.serial,
            lineage=lineage, stack=stack, active_id=int(ob['id']),
            region=region_key(ob['points']), points=np.asarray(ob['points']).copy(),
            regions={int(oid): observation_region_token(next(
                item for item in self.node._obligations if int(item['id']) == int(oid)))
                for oid in tracked_ids},
            previous_q=np.asarray(published['q_vis']).copy(),
            offered_wall=time.monotonic(), reason=reason)
        trigger_raw = int(event.get('raw_candidate_stamp_ns', 0))
        self.send(
            'action=reserve request_id={} reason={} mode_epoch={} query_stamp_ns={} '
            'observation_token={} obligation_id={} trigger_raw_candidate_stamp_ns={}'.format(
                self.serial, reason, event['mode_epoch'], event['query_stamp_ns'],
                event['observation_token'], int(ob['id']), trigger_raw))
        self.log('requested', request_id=self.serial, reason=reason,
                 active_id=int(ob['id']), owners=sorted(owners),
                 query_stamp_ns=event['query_stamp_ns'],
                 trigger_raw_candidate_stamp_ns=trigger_raw)
        return True

    def _queue_owner_promotion_locked(self, event, ob, stack, published,
                                      owner_id, point, promotion_key,
                                      ancestor=False):
        dependency_key = (region_key(ob['points']), region_key([point])[0])
        attempts = getattr(self.node, '_dependency_attempts', {})
        if dependency_key not in attempts and len(attempts) >= 64:
            return False
        self.serial += 1
        tracked_ids = sorted(set(stack) | {int(owner_id)})
        self.pending = dict(operation=('restore_ancestor' if ancestor else 'promote_owner'),
            event=event.copy(),
            request_id=self.serial, stack=stack, active_id=int(ob['id']),
            owner_id=int(owner_id), blocker_point=np.asarray(point, dtype=float).copy(),
            promotion_key=promotion_key, dependency_key=dependency_key,
            region=region_key(ob['points']),
            regions={int(oid): observation_region_token(next(
                item for item in self.node._obligations if int(item['id']) == int(oid)))
                for oid in tracked_ids},
            previous_q=np.asarray(published['q_vis']).copy(),
            offered_wall=time.monotonic(), reason='final_vbc_owner_promotion')
        trigger_raw = int(event.get('raw_candidate_stamp_ns', 0))
        self.send(
            'action=reserve_owner request_id={} reason=final_vbc_owner_promotion '
            'mode_epoch={} query_stamp_ns={} observation_token={} obligation_id={} '
            'trigger_raw_candidate_stamp_ns={}'.format(
                self.serial, event['mode_epoch'], event['query_stamp_ns'],
                event['observation_token'], int(ob['id']), trigger_raw))
        self.log('owner_promotion_requested', request_id=self.serial,
                 active_id=int(ob['id']), owner_id=int(owner_id),
                 point=np.asarray(point, dtype=float).tolist(),
                 query_stamp_ns=event['query_stamp_ns'],
                 trigger_raw_candidate_stamp_ns=trigger_raw)
        return True

    def offer(self, event):
        """Called only after exact ownership identified a live ancestor cycle."""
        n = self.node
        with self.lock:
            if self.pending is not None:
                return
            with n._schedule_publish_lock, n._obligation_lock:
                stack = tuple(n._repair_stack)
                by_id = {int(o['id']): o for o in n._obligations}
                published = getattr(n, '_trace_published_target', None)
                if len(stack) < 2 or stack[-1] not in by_id or not published:
                    return
                ob = by_id[stack[-1]]
                if (published['observation_token'] != event['observation_token'] or
                        published['region_token'] != observation_region_token(ob)):
                    return
                owners = {oid for oid in stack[:-1] if oid in by_id and
                    any(region_key([p])[0] in region_key(by_id[oid]['points']) for p in event['points'])}
                if not owners:
                    return
                self._queue_locked(event, ob, stack, published,
                                   'observation_dependency', owners)

    def offer_final_vbc(self, text):
        """Route a planner-authenticated repeated exact-VBC point to a new head."""
        try:
            fields = {}
            for word in text.split():
                key, value = word.split('=', 1)
                if key in fields:
                    return
                fields[key] = value
            if fields['version'] != '1' or fields['reason'] not in ('final_vbc_repeat', 'repair_qp_failure'):
                return
            event = dict(
                reason=fields['reason'],
                mode_epoch=int(fields['mode_epoch']),
                query_stamp_ns=int(fields['query_stamp_ns']),
                query_ros_s=float(fields['query_ros_s']),
                observation_token=fields['observation_token'],
                obligation_id=int(fields['obligation_id']),
                raw_candidate_stamp_ns=int(fields['raw_candidate_stamp_ns']),
                audited_trajectory_stamp_ns=int(fields['audited_trajectory_stamp_ns']),
                points=[[float(value) for value in point.split(',')]
                        for point in fields['repeated_points'].split(';')])
            points = np.asarray(event['points'], dtype=float)
            if (event['mode_epoch'] < 0 or event['query_stamp_ns'] <= 0 or
                    (event['reason'] == 'final_vbc_repeat' and (
                        event['raw_candidate_stamp_ns'] <= 0 or event['audited_trajectory_stamp_ns'] <= 0)) or
                    (event['reason'] == 'repair_qp_failure' and (
                        event['raw_candidate_stamp_ns'] != 0 or event['audited_trajectory_stamp_ns'] != 0)) or
                    not event['observation_token'].startswith('care_obs_v1_') or
                    points.ndim != 2 or points.shape[1] != 3 or
                    not 1 <= len(points) <= 32 or not np.all(np.isfinite(points)) or
                    not 0. <= self.now()-event['query_ros_s'] <=
                        (1.0 if event['reason'] == 'repair_qp_failure' else .5)):
                return
        except (ValueError, KeyError, AttributeError, OverflowError):
            return
        n = self.node
        with self.lock:
            if self.pending is not None:
                return
            restored_from = None
            with n._schedule_publish_lock:
                with n._obligation_lock:
                    stack = tuple(n._repair_stack)
                    by_id = {int(o['id']): o for o in n._obligations}
                    published = getattr(n, '_trace_published_target', None)
                    owner_id = int(event['obligation_id'])
                    if (not stack or owner_id not in stack or owner_id not in by_id):
                        return
                    ob = by_id[owner_id]
                    exact_live_token = observation_token(ob)
                    current = stack[-1] == owner_id
                    if event['reason'] == 'repair_qp_failure':
                        # The trigger is planner-authenticated and still fresh.
                        # If an urgent blocker changed only the stack top, put
                        # the exact live owner back before reserving its sensor
                        # replacement. Geometry/token drift still fails closed.
                        if exact_live_token != event['observation_token']:
                            return
                        if not current:
                            restored_from = int(stack[-1])
                            n._repair_stack = [
                                int(oid) for oid in stack if int(oid) != owner_id]
                            n._repair_stack.append(owner_id)
                            n._path_co_plan_ids = [owner_id]
                            n._last_switch_reason = (
                                'qp_replacement_owner_restored')
                            n._path_association_reason = (
                                'qp_replacement_owner_restored')
                            n._progressive_shared_cache_key = None
                            n._progressive_shared_cache = None
                            stack = tuple(n._repair_stack)
                        if not self._arm_qp_owner_locked(event, restored_from):
                            return
                    elif not current:
                        return
                if restored_from is not None:
                    n._publish_schedule()
                with n._obligation_lock:
                    by_id = {int(o['id']): o for o in n._obligations}
                    stack = tuple(n._repair_stack)
                    published = getattr(n, '_trace_published_target', None)
                    if (not stack or stack[-1] != owner_id or
                            owner_id not in by_id or not published):
                        self._release_qp_owner_locked('owner_restore_context_changed')
                        return
                    ob = by_id[owner_id]
                    if (published['observation_token'] != event['observation_token'] or
                            published['region_token'] != observation_region_token(ob)):
                        self._release_qp_owner_locked('owner_restore_publish_mismatch')
                        return
                # A trajectory for the active target may repeatedly cross a
                # point owned by another live obligation. The planner already
                # authenticated target/raw continuity; retain only the weaker
                # live-point check here and keep every matched owner in the
                # generation context so stale/resolved blockers abort safely.
                repeated_keys = {region_key([point])[0]
                                 for point in event['points']}
                owners = {oid for oid, live in by_id.items()
                          if repeated_keys.intersection(region_key(live['points']))}
                if not owners:
                    if event['reason'] == 'repair_qp_failure':
                        self._release_qp_owner_locked('replacement_points_not_live')
                    return
                # The trigger token belongs to the published active target,
                # while the rejected point may belong to an ancestor. Never
                # spend the active child's sensor ledger on an ancestor point.
                # An authenticated promotion grant can reorder the live stack
                # without charging a sensor direction.
                ancestor = []
                external = []
                for point in event['points']:
                    point_key = region_key([point])[0]
                    for owner_id in sorted(owners):
                        if (owner_id != int(ob['id']) and
                                point_key in region_key(by_id[owner_id]['points'])):
                            if owner_id in stack[:-1]:
                                ancestor.append((owner_id, point))
                            elif owner_id not in stack:
                                external.append((owner_id, point))
                epoch = int(event['mode_epoch'])
                if self.owner_promotion_epoch != epoch:
                    self.owner_promotion_epoch = epoch
                    self.owner_promotions.clear()
                if (int(ob['id']) not in owners and (ancestor or external) and
                        event['reason'] == 'final_vbc_repeat'):
                    owner_id, point = (ancestor or external)[0]
                    is_ancestor = bool(ancestor)
                    promotion_key = (epoch, int(ob['id']),
                        ob.get('generation_event_id', 'legacy'), int(owner_id))
                    if (promotion_key not in self.owner_promotions and
                            len(self.owner_promotions) < 64):
                        # Consume before transport. A stale/denied request must
                        # not reopen an unbounded parent/child oscillation.
                        self.owner_promotions.add(promotion_key)
                        if self._queue_owner_promotion_locked(
                                event, ob, stack, published, owner_id, point,
                                promotion_key, ancestor=is_ancestor):
                            return
                    self.log('point_owner_promotion_exhausted',
                             active_id=int(ob['id']), owner_id=int(owner_id),
                             ancestor=is_ancestor)
                    return
                if int(ob['id']) not in owners:
                    self.log('point_owner_mismatch_unroutable',
                             active_id=int(ob['id']), owners=sorted(owners))
                    return
                queued = self._queue_locked(event, ob, stack, published,
                                            event['reason'], owners)
                if not queued and event['reason'] == 'repair_qp_failure':
                    self._release_qp_owner_locked('replacement_queue_rejected')

    def receive(self, text):
        try:
            fields = {}
            for word in text.split():
                key, value = word.split('=', 1)
                if key in fields:
                    return
                fields[key] = value
            rid = int(fields['request_id'])
            if fields['granted'] not in ('0', '1'):
                return
        except (ValueError, KeyError):
            return
        with self.lock:
            if self.pending is not None and rid == self.pending['request_id'] and self.grant is None:
                self.grant = fields

    def _current(self, p):
        n = self.node
        by_id = {int(o['id']): o for o in n._obligations}
        target = getattr(n, '_trace_published_target', {}) or {}
        return (tuple(n._repair_stack) == p['stack'] and
            all(oid in by_id and observation_region_token(by_id[oid]) == region
                for oid, region in p['regions'].items()) and
            target.get('observation_token') == p['event']['observation_token'])

    def process(self):
        # Queue-only transport callback; generation runs on the existing node timer.
        with self.lock:
            if self.qp_owner_lock is not None:
                owner = int(self.qp_owner_lock['owner_id'])
                live = {int(ob['id']) for ob in self.node._obligations}
                if owner not in live:
                    self._release_qp_owner_locked('owner_obligation_resolved')
                elif time.monotonic() >= self.qp_owner_lock['expires_wall']:
                    self._release_qp_owner_locked('owner_lock_timeout')
            if self.pending is None:
                return
            p = self.pending
            if self.grant is None:
                if time.monotonic()-p['offered_wall'] > 3.:
                    self.pending = None
                    self.log('reservation_timeout', request_id=p['request_id'])
                    if p['reason'] == 'repair_qp_failure':
                        self._release_qp_owner_locked('reservation_timeout')
                return
            grant, self.grant = self.grant, None
            if grant['granted'] != '1':
                self.pending = None
                self.log('reservation_denied', **grant)
                if p['reason'] == 'repair_qp_failure':
                    self._release_qp_owner_locked('reservation_denied')
                return
            if p['operation'] in ('promote_owner', 'restore_ancestor'):
                self._promote_owner(p, grant)
            else:
                self._generate(p, grant)

    def _promote_owner(self, p, grant):
        """Push one exact live-point owner; observation completion pops it."""
        n = self.node
        new_token = 'none'
        reason = 'owner_promotion_context_changed'
        detail = dict(active_id=p['active_id'], owner_id=p['owner_id'],
                      point=p['blocker_point'].tolist())
        try:
            deadline = int(grant['deadline_ns'])*1e-9
            if (grant.get('owner_promotion') != '1' or
                    int(grant['mode_epoch']) != int(p['event']['mode_epoch']) or
                    int(grant['query_stamp_ns']) != int(p['event']['query_stamp_ns']) or
                    not 0 <= int(grant['attempts']) <= 5 or
                    not 0 <= int(grant['segments']) < 6):
                raise ValueError('invalid owner-promotion grant')
            if self.now() >= deadline or time.monotonic()-p['offered_wall'] >= 2.5:
                reason = 'owner_promotion_expired'
                return
            with n._schedule_publish_lock:
                with n._obligation_lock:
                    if not self._current(p):
                        return
                    live = {int(ob['id']): ob for ob in n._obligations}
                    pid, cid = int(p['active_id']), int(p['owner_id'])
                    ancestor = p['operation'] == 'restore_ancestor'
                    if (pid not in live or cid not in live or
                            (ancestor and cid not in n._repair_stack[:-1]) or
                            (not ancestor and (
                                cid in n._repair_stack or
                                cid in n._dependency_edges))):
                        return
                    point_key = region_key([p['blocker_point']])[0]
                    if point_key not in region_key(live[cid]['points']):
                        return
                    parent_q = np.asarray(p['previous_q'], dtype=float)
                    owner_pin = n._dependency_pins.get(cid)
                    child_q = np.asarray(
                        owner_pin['q_vis'] if owner_pin is not None and
                        owner_pin['region'] == region_key(live[cid]['points'])
                        else live[cid]['q_vis'], dtype=float)
                    if (parent_q.shape != (7,) or child_q.shape != (7,) or
                            not np.all(np.isfinite(parent_q)) or
                            not np.all(np.isfinite(child_q))):
                        reason = 'owner_promotion_invalid_target'
                        return
                    if not ancestor:
                        n._dependency_attempts[p['dependency_key']] = max(
                            2, n._dependency_attempts.get(p['dependency_key'], 0))
                    n._dependency_pins[pid] = dict(
                        region=region_key(live[pid]['points']), q_vis=parent_q.copy())
                    n._dependency_pins[cid] = dict(
                        region=region_key(live[cid]['points']), q_vis=child_q.copy())
                    if ancestor:
                        n._repair_stack = [oid for oid in n._repair_stack
                                           if int(oid) != cid] + [cid]
                        n._stack_cycle_block_count += 1
                        n._dependency_reason = n._last_switch_reason = (
                            'final_vbc_ancestor_owner_restored')
                        n._path_co_plan_ids = [cid]
                    else:
                        n._dependency_edges[cid] = pid
                        n._repair_stack.append(cid)
                        n._stack_push_count += 1
                        n._dependency_pushes += 1
                        n._dependency_reason = n._last_switch_reason = (
                            'final_vbc_owner_promotion_push')
                        n._path_co_plan_ids = []
                    n._pending_blocker_id = None
                    n._pending_blocker_count = 0
                    n._progressive_shared_cache_key = None
                    n._progressive_shared_cache = None
                    n._trace_observation(
                        ('final_vbc_ancestor_owner_restored' if ancestor else
                         'final_vbc_owner_promotion_push'), parent_id=pid,
                        child_id=cid, query_stamp_ns=p['event']['query_stamp_ns'],
                        points=[p['blocker_point'].tolist()],
                        observation_token=p['event']['observation_token'])
                n._publish_schedule()
                new_token = n._trace_published_target['observation_token']
                if ancestor:
                    self.qp_owner_lock = dict(
                        owner_id=cid, mode_epoch=int(p['event']['mode_epoch']),
                        observation_token=new_token, stage='hard_qp',
                        expires_wall=time.monotonic()+4.0)
                    reason = 'ancestor_owner_restored_requires_existing_gates'
                else:
                    reason = 'owner_promotion_applied_requires_existing_gates'
        except Exception as exc:
            reason = 'owner_promotion_error'
            detail['error'] = str(exc)
        finally:
            self.pending = None
            self.send('action=finish request_id={} new_token={}'.format(
                p['request_id'], new_token))
            self.log(reason, request_id=p['request_id'], grant=grant,
                     new_token=new_token, **detail)

    def _generate(self, p, grant):
        n = self.node
        new_token = 'none'
        reason = 'context_changed'
        detail = {}
        try:
            deadline = int(grant['deadline_ns'])*1e-9
            if (int(grant['mode_epoch']) != int(p['event']['mode_epoch']) or
                    int(grant['query_stamp_ns']) != int(p['event']['query_stamp_ns']) or
                    not 1 <= int(grant['attempts']) <= 5 or not 0 <= int(grant['segments']) < 6):
                raise ValueError('invalid grant')
            with n._schedule_publish_lock:
                with n._obligation_lock:
                    if not self._current(p):
                        return
                measured, stamp = getattr(n, '_trace_measured', (None, float('nan')))
                if measured is None or not 0. <= self.now()-stamp <= .2:
                    reason = 'stale_measured_state'; return
                seed = np.asarray(measured, dtype=float).copy()
                if seed.shape != (7,) or not np.all(np.isfinite(seed)):
                    reason = 'invalid_measured_state'; return
                with n._lock:
                    trajectory, received, _ = n._preferred_trajectory_locked()
                if trajectory is None:
                    reason = 'no_trajectory'; return
                with n._progressive_shared_lock:
                    n._seed_override = seed
                    replacement_override = dict(
                        excluded=sorted(self.families[p['lineage']]),
                        tried_mount_groups=sorted({
                            sensor_mount_group_id(sensor_id)
                            for sensor_id in self.families[p['lineage']]}),
                        previous_q=p['previous_q'].tolist(),
                        minimum_conservative_g=float(getattr(
                            n, 'per_sensor_replacement_min_conservative_g', 0.0)))
                    preferred = high_witness_sensor_preference(
                        p['event']['points'],
                        getattr(n, 'per_sensor_high_witness_priority_enabled', False),
                        getattr(n, 'per_sensor_high_witness_z_min', .85))
                    if preferred:
                        replacement_override['preferred_sensor_ids'] = preferred
                        replacement_override['ee_sensor_priority_first'] = True
                        replacement_override['preference_reason'] = (
                            'repeated_exact_vbc_high_witness')
                    n._candidate_replacement_override = replacement_override
                    try:
                        result = n._generate_active_set_waypoint(p['points'], trajectory, .1, received)
                    finally:
                        n._seed_override = None
                        n._candidate_replacement_override = None
                hybrid = result.get('per_sensor_hybrid', {})
                for attempt in hybrid.get('attempts', []):
                    self.families[p['lineage']].add(int(attempt['sensor_id']))
                detail = dict(hybrid=hybrid, points=p['points'].tolist(), measured_seed=seed.tolist())
                if hybrid.get('accepted') is not True:
                    reason = 'no_geometric_candidate'; return
                q = np.asarray(hybrid['selected_q_vis'], dtype=float)
                if q.shape != (7,) or not np.all(np.isfinite(q)):
                    reason = 'invalid_candidate'; return
                poses = self.poses.setdefault(p['lineage'], [])
                if any(np.max(np.abs(q-old)) <= .01 for old in [p['previous_q']]+poses):
                    reason = 'candidate_not_distinct'; return
                if self.now() >= deadline or time.monotonic()-p['offered_wall'] >= 2.5:
                    reason = 'generation_expired'; return
                live, live_stamp = getattr(n, '_trace_measured', (None, float('nan')))
                if (live is None or not 0. <= self.now()-live_stamp <= .2 or
                        np.max(np.abs(np.asarray(live)-seed)) > .01):
                    reason = 'measured_state_changed'; return
                with n._obligation_lock:
                    if not self._current(p):
                        return
                    ob = next(o for o in n._obligations if int(o['id']) == p['active_id'])
                    # Preserve identity, safety points, ancestor stack and edges.
                    ob['q_vis'] = q.copy()
                    ob['q_vis_joint_mask'] = np.asarray(
                        hybrid.get('selected_joint_mask', [1.0] * 7),
                        dtype=float).reshape(7).copy()
                    ob['q_zero'] = np.asarray(result['q_zero'], dtype=float).copy()
                    ob['per_sensor_selected_sensor_id'] = int(hybrid['selected_sensor_id'])
                    ob['per_sensor_selected_sensor_frame'] = hybrid['selected_sensor_frame']
                    ob['per_sensor_selected_rank'] = int(hybrid['selected_rank'])
                    n._dependency_pins[p['active_id']] = dict(region=p['region'], q_vis=q.copy(),
                        preserve_on_growth=True)
                    n._progressive_shared_cache_key = None
                    n._progressive_shared_cache = None
                    poses.append(q.copy())
                n._publish_schedule()  # NEVER publish while holding obligation Lock.
                new_token = n._trace_published_target['observation_token']
                reason = 'applied_requires_existing_gates'
        except Exception as exc:
            reason = 'generation_error'
            detail['error'] = str(exc)
        finally:
            self.pending = None
            if p['reason'] == 'repair_qp_failure':
                if new_token.startswith('care_obs_v1_') and self.qp_owner_lock is not None:
                    self.qp_owner_lock['observation_token'] = new_token
                    self.qp_owner_lock['stage'] = 'hard_qp'
                    self.qp_owner_lock['expires_wall'] = time.monotonic()+4.0
                else:
                    self._release_qp_owner_locked(reason)
            self.send('action=finish request_id={} new_token={}'.format(p['request_id'],new_token))
            self.log(reason, request_id=p['request_id'], grant=grant, new_token=new_token, **detail)


def attach_candidate_replacement(node):
    import rospy
    from std_msgs.msg import String
    if not rospy.get_param('~candidate_replacement_enabled', False):
        return None
    if not node.per_sensor_hybrid_enabled or node._per_sensor_runtime is None:
        raise ValueError('candidate replacement requires per-sensor hybrid')
    publisher = rospy.Publisher('/care_planner/local_planner/candidate_replacement_request', String, queue_size=8)
    path = Path(node.output_root) / 'candidate_replacement.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    def log(event, **data):
        row = dict(event=event, ros_s=rospy.Time.now().to_sec(), **data)
        with path.open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        rospy.logwarn('[candidate_replacement] %s request=%s', event, data.get('request_id'))
    runtime = CandidateReplacement(node, lambda: rospy.Time.now().to_sec(),
        lambda text: publisher.publish(String(data=text)), log)
    runtime.subscriber = rospy.Subscriber('/care_planner/local_planner/candidate_replacement_grant',
        String, lambda msg: runtime.receive(msg.data), queue_size=8)
    runtime.trigger_subscriber = rospy.Subscriber(
        '/care_planner/local_planner/candidate_replacement_trigger', String,
        lambda msg: runtime.offer_final_vbc(msg.data), queue_size=8)
    runtime.summary_subscriber = rospy.Subscriber(
        '/care_planner/local_planner/summary', String,
        lambda msg: runtime.observe_planner_summary(msg.data), queue_size=16)
    return runtime
