"""Opt-in, bounded steering replacement for a dependency or repeated final VBC.

The C++ planner owns the attempt/segment ledger. This module neither commits
trajectories nor removes obligations/witnesses. All replacement poses must pass
the existing live-map solver, final GCDF, exact VBC and tracker path.
"""
import json
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from bounded_visibility_recovery import region_key
from observation_identity import observation_region_token, observation_token
from observation_identity import observation_target_state


# All three replace the current observation target, preserving its full region.
# Path counterexamples additionally identify the real rejected raw/audit pair.
PATH_TARGET_REPLACEMENT_REASON = 'geometric_restore_path_counterexample'
AUDITED_TARGET_REPLACEMENT_REASONS = (
    PATH_TARGET_REPLACEMENT_REASON, 'geometric_restore_proof_resource_exhausted',
    'geometric_restore_handoff_residual_insufficient')
PLANNER_TARGET_REPLACEMENT_REASONS = (
    'repair_qp_failure', 'observation_no_progress') + AUDITED_TARGET_REPLACEMENT_REASONS
FINAL_POINT_REPLACEMENT_REASONS = ('final_vbc_repeat', 'final_gcdf_repeat')

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


class StableGenerationSeed:
    """Read-only execution/measurement gate; never commands a stop or motion.

    The reservation already pauses planning. Wait for the existing certified
    execution to finish or enter an explicit zero-command safety hold, and
    for a fresh 100-ms measured window within 1 mrad.
    This is a generation scheduling check, not an execution certificate.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.samples = deque(maxlen=64)
        self.tracker = None
        self.seen_executions = deque(maxlen=64)
        self.commit_ack = None

    def observe_commit_ack(self, text, received):
        try:
            words = text.split()
            fields = dict(word.split('=', 1) for word in words)
            if len(fields) != len(words) or fields['confirmed'] != '1':
                return
            ack = {k: int(fields[k]) for k in ('request_id', 'mode_epoch',
                'query_stamp_ns', 'deadline_ns', 'last_execution_stamp_ns')}
            if any(v < 0 for v in ack.values()) or not np.isfinite(received):
                return
            ack['received'] = float(received)
        except (KeyError, ValueError):
            return
        with self.lock:
            if self.commit_ack is None or ack['request_id'] > self.commit_ack['request_id']:
                self.commit_ack = ack

    def confirmed(self, grant):
        with self.lock:
            ack = self.commit_ack
            if ack is None or any(ack[k] != int(grant[k]) for k in
                                 ('request_id', 'mode_epoch', 'query_stamp_ns', 'deadline_ns')):
                return False, 'waiting_for_commit_confirmation', None
            execution = ack['last_execution_stamp_ns']
            if execution and execution not in self.seen_executions:
                return False, 'waiting_for_tracker_commit_receipt', ack
            return True, 'commit_confirmed', ack

    def observe_measured(self, q, stamp):
        q = np.asarray(q, dtype=float)
        with self.lock:
            if q.shape != (7,) or not np.all(np.isfinite(q)) or not np.isfinite(stamp):
                self.samples.clear()
                return
            if self.samples and stamp <= self.samples[-1][0]:
                if stamp == self.samples[-1][0] and np.array_equal(q, self.samples[-1][1]):
                    return  # duplicate delivery is not additional stability evidence
                self.samples.clear()
            self.samples.append((float(stamp), q.copy()))

    def observe_tracker(self, text, received):
        try:
            fields = {}
            words = text.split()
            if words and words[0] == 'TRACKER':
                words = words[1:]
            for word in words:
                key, value = word.split('=', 1)
                if key in fields:
                    raise ValueError('duplicate tracker field')
                fields[key] = value
            if fields['active'] not in ('0', '1') or not np.isfinite(received):
                raise ValueError('invalid tracker state')
            execution = int(fields['execution_stamp_ns'])
            if execution < 0:
                raise ValueError('invalid execution identity')
            tracker = dict(received=float(received), active=fields['active']=='1',
                           execution_stamp_ns=execution, source=fields.get('source', 'unknown'))
        except (KeyError, ValueError, AttributeError):
            tracker = None
        with self.lock:
            self.tracker = tracker
            if tracker is not None and execution > 0 and execution not in self.seen_executions:
                self.seen_executions.append(execution)

    def ready(self, now, granted_at):
        with self.lock:
            tracker = self.tracker
            samples = list(self.samples)
        detail = dict(tracker=tracker)
        if tracker is None or not 0. <= now-tracker['received'] <= .2:
            return False, dict(detail, reason='tracker_missing_or_stale')
        # The tracker freezes phase during these safety holds and keeps
        # active=1 for the suspended trajectory, while commanding zero dq.
        # Readiness still requires measured settling below; a hold message
        # alone is not evidence that the actuator has already stopped.
        holding = tracker['source'] in ('external_vbc_hold', 'external_gcdf_hard_hold')
        if tracker['active'] and not holding:
            return False, dict(detail, reason='certified_execution_active')
        if not samples or not 0. <= now-samples[-1][0] <= .05:
            return False, dict(detail, reason='measured_missing_or_stale')
        end = samples[-1][0]
        # Include the sample just before the 100-ms boundary. All evidence
        # must postdate the grant; an old stationary window cannot cover a
        # just-committed trajectory whose measured motion has not begun yet.
        eligible = [(t, q) for t, q in samples if t >= granted_at]
        before = [i for i, (t, _) in enumerate(eligible) if t <= end-.1+1e-9]
        if not before:
            return False, dict(detail, reason='stable_window_incomplete')
        window = eligible[before[-1]:]
        if len(window) < 3 or max(b[0]-a[0] for a, b in zip(window, window[1:])) > .05+1e-9:
            return False, dict(detail, reason='measured_window_gap')
        span = float(np.max(np.ptp(np.array([q for _, q in window]), axis=0)))
        detail.update(window_start=window[0][0], window_end=end,
                      window_samples=len(window), measured_span_inf=span)
        if span > .001:
            return False, dict(detail, reason='measured_still_moving')
        return True, dict(detail, reason='stable_seed_ready')


class CandidateReplacement:
    def __init__(self, node, now, send, log):
        self.node, self.now, self.send, self.log = node, now, send, log
        self.lock = threading.RLock()
        self.pending = None
        self.waiting_blocker = None
        # One planner-authenticated QP trigger may arrive before the denial
        # of an older observation reservation. Retain its original deadline
        # and identity; waiting never reserves or replenishes a direction.
        self.waiting_qp_trigger = None
        # Bounded transport deduplication, valid only for the original trigger
        # lifetime. It does not grant a reservation or replenish the ledger.
        self.recent_triggers = {}
        self.grant = None
        self.serial = 0
        self.families = {}
        self.poses = {}
        # Independent lock: joint/tracker callbacks must keep updating while
        # process() holds the replacement lock during model inference.
        self.seed_gate = StableGenerationSeed()
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

    @staticmethod
    def _trigger_identity(event):
        return tuple(event[k] for k in (
            'reason', 'mode_epoch', 'query_stamp_ns', 'query_ros_s',
            'observation_token', 'obligation_id', 'raw_candidate_stamp_ns',
            'audited_trajectory_stamp_ns')) + (tuple(sorted(
                tuple(float(x) for x in p) for p in event['points'])),)

    def _prune_triggers_locked(self):
        now, wall = self.now(), time.monotonic()
        self.recent_triggers = {key: value for key, value in self.recent_triggers.items()
            if 0. <= now-value['query_ros_s'] <= value['max_age'] and
            wall < value['expires_wall']}

    def _remember_trigger_locked(self, event, received_wall=None):
        if event.get('reason') not in PLANNER_TARGET_REPLACEMENT_REASONS + FINAL_POINT_REPLACEMENT_REASONS:
            return
        identity = self._trigger_identity(event)
        if identity not in self.recent_triggers:
            self.recent_triggers[identity] = dict(query_ros_s=event['query_ros_s'],
                max_age=1.0 if event['reason'] in PLANNER_TARGET_REPLACEMENT_REASONS else .5,
                expires_wall=(time.monotonic() if received_wall is None else received_wall)+1.25)

    def _can_defer_trigger_locked(self, event):
        p = self.pending
        # Preserve the existing denied-dependency handoff, which can restore
        # an owner even when the obsolete dependency used an older epoch.
        if p['reason'] == 'observation_dependency':
            return True
        if (p['operation'] != 'replace' or
                p['reason'] not in PLANNER_TARGET_REPLACEMENT_REASONS + FINAL_POINT_REPLACEMENT_REASONS or
                int(p['active_id']) != event['obligation_id'] or
                int(p['event']['mode_epoch']) != event['mode_epoch'] or
                p['event']['observation_token'] != event['observation_token']):
            return False
        n = self.node
        with n._schedule_publish_lock, n._obligation_lock:
            return self._current(p)

    def _expire_waiting_trigger_locked(self):
        waiting = self.waiting_qp_trigger
        if waiting is None:
            return
        event = waiting['event']
        if (not 0. <= self.now()-event['query_ros_s'] <= 1.0 or
                time.monotonic() >= waiting['expires_wall']):
            self.waiting_qp_trigger = None
            self.log('qp_trigger_deferred_expired',
                query_stamp_ns=event['query_stamp_ns'],
                observation_token=event['observation_token'],
                age_ros_s=self.now()-event['query_ros_s'],
                expires_wall=waiting['expires_wall'])

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
            if event in ('candidate_published', 'geometric_restore_candidate_published'):
                self._release_qp_owner_locked(
                    'replacement_hard_qp_candidate_published' if event=='candidate_published'
                    else 'replacement_geometric_restore_proposal_published')
            elif event in ('repair_qp_candidate_exhausted_hold',
                           'repair_candidate_replacement_exhausted_hold'):
                self._release_qp_owner_locked(event)

    def _queue_locked(self, event, ob, stack, published, reason, owners):
        lineage = (int(event['mode_epoch']), int(ob['id']),
                   ob.get('generation_event_id', 'legacy'))
        if lineage not in self.families and len(self.families) >= 64:
            return False
        tried = self.families.setdefault(lineage, set())
        # The trigger was matched to this published target, which may be a
        # shared or pinned pose from a different head than the stored owner.
        # Charge the same snapshot used for previous_q/mask below; a union
        # target (-1) must not consume the owner's stale sensor branch.
        sensor = int(published.get('per_sensor_selected_sensor_id', -1))
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
            previous_target=observation_target_state(published),
            offered_wall=time.monotonic(), reason=reason)
        self._remember_trigger_locked(event)
        trigger_raw = int(event.get('raw_candidate_stamp_ns', 0))
        self.send(
            'action=reserve request_id={} reason={} mode_epoch={} query_stamp_ns={} '
            'observation_token={} obligation_id={} trigger_raw_candidate_stamp_ns={}'.format(
                self.serial, reason, event['mode_epoch'], event['query_stamp_ns'],
                event['observation_token'], int(ob['id']), trigger_raw))
        self.log('requested', request_id=self.serial, reason=reason,
                 active_id=int(ob['id']), owners=sorted(owners),
                 failed_sensor_id=sensor,
                 stored_sensor_id=int(ob.get('per_sensor_selected_sensor_id', -1)),
                 query_stamp_ns=event['query_stamp_ns'],
                 trigger_raw_candidate_stamp_ns=trigger_raw)
        return True

    def _queue_owner_promotion_locked(self, event, ob, stack, published,
                                      owner_id, point, promotion_key,
                                      ancestor=False, reason='final_vbc_owner_promotion'):
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
            previous_target=observation_target_state(published),
            offered_wall=time.monotonic(), reason=reason)
        trigger_raw = int(event.get('raw_candidate_stamp_ns', 0))
        self.send(
            'action=reserve_owner request_id={} reason={} '
            'mode_epoch={} query_stamp_ns={} observation_token={} obligation_id={} '
            'trigger_raw_candidate_stamp_ns={}'.format(
                self.serial, reason, event['mode_epoch'], event['query_stamp_ns'],
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
            if self.pending is not None or self.waiting_qp_trigger is not None:
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
                epoch=int(event['mode_epoch'])
                if self.owner_promotion_epoch != epoch:
                    self.owner_promotion_epoch=epoch
                    self.owner_promotions.clear()
                owner_id=min(owners)
                point=next(p for p in event['points']
                           if region_key([p])[0] in region_key(by_id[owner_id]['points']))
                promotion_key=(epoch,int(ob['id']),ob.get('generation_event_id','legacy'),owner_id)
                if (getattr(n,'rolling_guidance_mode','off') == 'drive' and
                        promotion_key not in self.owner_promotions and len(self.owner_promotions)<64):
                    self.owner_promotions.add(promotion_key)
                    if self._queue_owner_promotion_locked(event,ob,stack,published,owner_id,
                            point,promotion_key,ancestor=True,reason='observation_dependency_owner'):
                        return
                self._queue_locked(event, ob, stack, published,
                                   'observation_dependency', owners)

    def offer_final_vbc(self, text, _deferred=None):
        """Route a planner-authenticated repeated exact-VBC point to a new head."""
        # A rejected trigger otherwise leaves the planner waiting for a
        # replacement without any explanation in the request/grant streams.
        def reject(reason):
            pending = self.pending or {}
            self.log('trigger_ignored', reason=reason,
                     trigger_reason=fields.get('reason', 'unknown'),
                     query_stamp_ns=fields.get('query_stamp_ns', 'unknown'),
                     mode_epoch=fields.get('mode_epoch', 'unknown'),
                     pending_request_id=pending.get('request_id'),
                     pending_reason=pending.get('reason'),
                     pending_active_id=pending.get('active_id'),
                     observation_token=fields.get('observation_token', 'unknown'),
                     obligation_id=fields.get('obligation_id', 'unknown'))

        received_wall = time.monotonic() if _deferred is None else _deferred['received_wall']
        fields = {}
        try:
            for word in text.split():
                key, value = word.split('=', 1)
                if key in fields:
                    reject('duplicate_field')
                    return
                fields[key] = value
            if fields['version'] != '1' or fields['reason'] not in FINAL_POINT_REPLACEMENT_REASONS + PLANNER_TARGET_REPLACEMENT_REASONS:
                reject('unsupported_trigger')
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
                    (event['reason'] in FINAL_POINT_REPLACEMENT_REASONS and (
                        event['raw_candidate_stamp_ns'] <= 0 or event['audited_trajectory_stamp_ns'] <= 0)) or
                    (event['reason'] in AUDITED_TARGET_REPLACEMENT_REASONS and (
                        event['raw_candidate_stamp_ns'] <= 0 or event['audited_trajectory_stamp_ns'] <= 0)) or
                    (event['reason'] in PLANNER_TARGET_REPLACEMENT_REASONS and
                     event['reason'] not in AUDITED_TARGET_REPLACEMENT_REASONS and (
                        event['raw_candidate_stamp_ns'] != 0 or event['audited_trajectory_stamp_ns'] != 0)) or
                    not event['observation_token'].startswith('care_obs_v1_') or
                    points.ndim != 2 or points.shape[1] != 3 or
                    not 1 <= len(points) <= 32 or not np.all(np.isfinite(points)) or
                    not 0. <= self.now()-event['query_ros_s'] <=
                        (1.0 if event['reason'] in PLANNER_TARGET_REPLACEMENT_REASONS else .5)):
                reject('invalid_or_stale_trigger')
                return
        except (ValueError, KeyError, AttributeError, OverflowError):
            reject('malformed_trigger')
            return
        n = self.node
        with self.lock:
            identity = self._trigger_identity(event)
            self._prune_triggers_locked()
            self._expire_waiting_trigger_locked()
            if not 0. <= self.now()-event['query_ros_s'] <= (
                    1.0 if event['reason'] in PLANNER_TARGET_REPLACEMENT_REASONS else .5):
                reject('invalid_or_stale_trigger')
                return
            if _deferred is None and identity in self.recent_triggers:
                self.log('trigger_duplicate', query_stamp_ns=event['query_stamp_ns'],
                         observation_token=event['observation_token'])
                return
            if (_deferred is None and self.pending is not None and
                    self.pending['event'].get('reason') in
                        PLANNER_TARGET_REPLACEMENT_REASONS + FINAL_POINT_REPLACEMENT_REASONS and
                    self._trigger_identity(self.pending['event']) == identity):
                return
            if _deferred is None and self.waiting_qp_trigger is not None:
                reject('different_qp_trigger_already_waiting')
                return
            if time.monotonic() >= received_wall+1.25:
                reject('trigger_transport_expired')
                return
            if len(self.recent_triggers) >= 64 and identity not in self.recent_triggers:
                reject('trigger_history_capacity')
                return
            self.log('trigger_received', query_stamp_ns=event['query_stamp_ns'],
                     mode_epoch=event['mode_epoch'], obligation_id=event['obligation_id'],
                     observation_token=event['observation_token'], trigger_reason=event['reason'],
                     received_wall=received_wall, deferred_revalidation=_deferred is not None)
            if self.pending is not None:
                if (event['reason'] in PLANNER_TARGET_REPLACEMENT_REASONS and
                        self._can_defer_trigger_locked(event)):
                    waiting = self.waiting_qp_trigger
                    if waiting is None:
                        self.waiting_qp_trigger = dict(
                            text=text, event=event, identity=identity,
                            received_wall=received_wall,
                            expires_wall=received_wall+1.25)
                        self._remember_trigger_locked(event, received_wall)
                        self.log('qp_trigger_deferred',
                                 request_id=self.pending['request_id'],
                                 pending_reason=self.pending['reason'],
                                 query_stamp_ns=event['query_stamp_ns'],
                                 observation_token=event['observation_token'],
                                 received_wall=received_wall,
                                 age_ros_s=self.now()-event['query_ros_s'])
                    elif waiting['identity'] != identity:
                        reject('different_qp_trigger_already_waiting')
                    # Duplicate delivery must not renew the deadline.
                    return
                reject('reservation_already_pending')
                return
            restored_from = None
            with n._schedule_publish_lock:
                with n._obligation_lock:
                    stack = tuple(n._repair_stack)
                    by_id = {int(o['id']): o for o in n._obligations}
                    published = getattr(n, '_trace_published_target', None)
                    owner_id = int(event['obligation_id'])
                    if (not stack or owner_id not in stack or owner_id not in by_id):
                        reject('owner_not_live_in_stack')
                        return
                    ob = by_id[owner_id]
                    live_region = observation_region_token(ob)
                    current = stack[-1] == owner_id
                    if event['reason'] in PLANNER_TARGET_REPLACEMENT_REASONS:
                        # The trigger is planner-authenticated and still fresh.
                        # If an urgent blocker changed only the stack top, put
                        # the exact live owner back before reserving its sensor
                        # replacement. The planner solves the published target,
                        # which can be a pinned/shared q_vis rather than ob's
                        # original q_vis. Match that exact published token and
                        # the live region; stale geometry still fails closed.
                        published_matches = bool(published and
                            published['observation_token'] == event['observation_token'] and
                            published['region_token'] == live_region)
                        # If a newer blocker preempted the owner, its target is
                        # now published. The owner's original token can still
                        # be restored; the post-publication check below must
                        # then reproduce the exact trigger token.
                        restorable_original = (not current and
                            observation_token(ob) == event['observation_token'])
                        if not (published_matches or restorable_original):
                            reject('live_observation_token_changed')
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
                            reject('different_qp_owner_locked')
                            return
                    elif not current:
                        reject('final_vbc_owner_not_current')
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
                        reject('published_target_missing_or_owner_changed')
                        return
                    ob = by_id[owner_id]
                    if (published['observation_token'] != event['observation_token'] or
                            published['region_token'] != observation_region_token(ob)):
                        self._release_qp_owner_locked('owner_restore_publish_mismatch')
                        reject('published_target_identity_changed')
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
                    if event['reason'] in PLANNER_TARGET_REPLACEMENT_REASONS:
                        self._release_qp_owner_locked('replacement_points_not_live')
                    elif event['reason'] in FINAL_POINT_REPLACEMENT_REASONS:
                        # The coherent bundle and trigger use different topics.
                        # Wait for registration without reserving a direction.
                        # Retrying runs every original identity/freshness check.
                        self.waiting_blocker = dict(text=text, event=event)
                        self.log('trigger_waiting_for_blocker',
                                 observation_token=event['observation_token'],
                                 audit_stamp_ns=event['audited_trajectory_stamp_ns'])
                        return
                    reject('replacement_points_not_live')
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
                        event['reason'] in FINAL_POINT_REPLACEMENT_REASONS):
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
                    # The once-only handoff guard does not prove this target's
                    # other heads infeasible. Keep its observation points and
                    # lineage, while tracking the external blocker owners for
                    # freshness. The ordinary authenticated reserve spends the
                    # existing C++ ledger; all hard rows/gates still apply.
                    if self._queue_locked(event, ob, stack, published,
                                          event['reason'], owners):
                        self.log('owner_exhausted_replacement_requested',
                                 request_id=self.serial, active_id=int(ob['id']),
                                 owners=sorted(owners))
                    else:
                        # No reservation exists to finish. Explicitly answer
                        # this exact trigger so the planner can leave its wait.
                        self.serial += 1
                        self.send(
                            'action=decline request_id={} reason={} '
                            'mode_epoch={} query_stamp_ns={} observation_token={} '
                            'obligation_id={} trigger_raw_candidate_stamp_ns={} '
                            'decline_reason=no_untried_sensor_or_family_capacity'.format(
                                self.serial, event['reason'], event['mode_epoch'], event['query_stamp_ns'],
                                event['observation_token'], int(ob['id']),
                                event['raw_candidate_stamp_ns']))
                        self.log('owner_exhausted_replacement_unavailable',
                                 request_id=self.serial, active_id=int(ob['id']))
                    return
                if int(ob['id']) not in owners:
                    self.log('point_owner_mismatch_unroutable',
                             active_id=int(ob['id']), owners=sorted(owners))
                    reject('point_owner_mismatch_unroutable')
                    return
                queued = self._queue_locked(event, ob, stack, published,
                                            event['reason'], owners)
                if not queued and event['reason'] in PLANNER_TARGET_REPLACEMENT_REASONS:
                    self._release_qp_owner_locked('replacement_queue_rejected')
                    reject('replacement_queue_rejected')

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
                if 'generation_grant' not in self.pending:
                    self.grant = fields
                    self.pending['grant_received_ros_s'] = self.now()
                    self.pending['grant_received_wall'] = time.monotonic()
                    self.log('grant_received', request_id=rid, grant=fields,
                             received_wall=self.pending['grant_received_wall'])

    def _current(self, p):
        n = self.node
        by_id = {int(o['id']): o for o in n._obligations}
        target = getattr(n, '_trace_published_target', {}) or {}
        return (tuple(n._repair_stack) == p['stack'] and
            all(oid in by_id and observation_region_token(by_id[oid]) == region
                for oid, region in p['regions'].items()) and
            target.get('observation_token') == p['event']['observation_token'])

    def _process_waiting_blocker(self):
        with self.lock:
            waiting = self.waiting_blocker
            if waiting is None:
                return
            event = waiting['event']
            age = self.now()-event['query_ros_s']
            if not 0. <= age <= .5:
                self.waiting_blocker = None
                self.log('waiting_blocker_expired',
                         observation_token=event['observation_token'])
                return
            n = self.node
            with n._obligation_lock:
                keys = {region_key([p])[0] for p in event['points']}
                materialized = any(keys.intersection(region_key(ob['points']))
                                   for ob in n._obligations)
                live_owner = any(int(ob['id']) == event['obligation_id']
                                 for ob in n._obligations)
            target = getattr(n, '_trace_published_target', None) or {}
            if self.pending is not None or not live_owner or target.get('observation_token') != event['observation_token']:
                self.waiting_blocker = None
                self.log('waiting_blocker_context_changed',
                         observation_token=event['observation_token'])
                return
            if materialized:
                self.waiting_blocker = None
                self.log('waiting_blocker_revalidate',
                         observation_token=event['observation_token'])
                self.offer_final_vbc(waiting['text'])

    def _process_waiting_qp_trigger(self):
        with self.lock:
            self._expire_waiting_trigger_locked()
            waiting = self.waiting_qp_trigger
            if waiting is None:
                return
            event = waiting['event']
            if self.pending is not None:
                return
            self.waiting_qp_trigger = None
            self.log('qp_trigger_deferred_revalidate',
                     query_stamp_ns=event['query_stamp_ns'],
                     observation_token=event['observation_token'])
            # Re-run the complete live identity/region/owner validation.
            # The native planner remains the only reservation/budget owner.
            self.offer_final_vbc(waiting['text'], _deferred=waiting)

    def process(self):
        with self.lock:
            try:
                self._process_pending()
            finally:
                # Retire and hand off atomically before other offers can
                # occupy the slot. Joint/tracker callbacks use seed_gate.lock.
                self._process_waiting_qp_trigger()
                self._process_waiting_blocker()

    def _process_pending(self):
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
            if 'first_process_wall' not in p:
                p['first_process_wall'] = time.monotonic()
                self.log('reservation_first_processed', request_id=p['request_id'],
                         processing_delay_wall_s=p['first_process_wall']-p['offered_wall'],
                         grant_received_wall=p.get('grant_received_wall'))
            if self.grant is None and 'generation_grant' not in p:
                if time.monotonic()-p['offered_wall'] > 3.:
                    self.pending = None
                    self.log('reservation_timeout', request_id=p['request_id'])
                    if p['reason'] in PLANNER_TARGET_REPLACEMENT_REASONS:
                        self._release_qp_owner_locked('reservation_timeout')
                return
            grant = p.get('generation_grant', self.grant)
            self.grant = None
            if grant['granted'] != '1':
                self.pending = None
                self.log('reservation_denied', **grant)
                if p['reason'] in PLANNER_TARGET_REPLACEMENT_REASONS:
                    self._release_qp_owner_locked('reservation_denied')
                return
            if p['operation'] in ('promote_owner', 'restore_ancestor'):
                self._promote_owner(p, grant)
            else:
                try:
                    self._validate_generation_grant(p, grant)
                except (ValueError, KeyError, OverflowError) as exc:
                    self._finish_generation(p, grant, 'generation_error', dict(error=str(exc)))
                    return
                if self.now() >= int(grant['deadline_ns'])*1e-9 or time.monotonic()-p['offered_wall'] >= 2.5:
                    self._finish_generation(p, grant, 'generation_expired', dict(stage='waiting_for_stable_seed'))
                    return
                with self.node._schedule_publish_lock, self.node._obligation_lock:
                    current = self._current(p)
                if not current:
                    self._finish_generation(p, grant, 'context_changed', dict(stage='waiting_for_stable_seed'))
                    return
                confirmed, reason, ack = self.seed_gate.confirmed(grant)
                if confirmed:
                    ready, detail = self.seed_gate.ready(
                        self.now(), max(p['grant_received_ros_s'], ack['received']))
                    detail['commit_ack'] = ack
                else:
                    ready, detail = False, dict(reason=reason, commit_ack=ack)
                if not ready:
                    p['generation_grant'] = grant
                    if not p.get('seed_wait_logged'):
                        p['seed_wait_logged'] = True
                        self.log('generation_waiting_for_stable_seed', request_id=p['request_id'],
                                 grant=grant, **detail)
                    return
                self.log('generation_stable_seed_ready', request_id=p['request_id'], **detail)
                self._generate(p, grant)

    @staticmethod
    def _validate_generation_grant(p, grant):
        if (int(grant['mode_epoch']) != int(p['event']['mode_epoch']) or
                int(grant['query_stamp_ns']) != int(p['event']['query_stamp_ns']) or
                not 1 <= int(grant['attempts']) <= 5 or not 0 <= int(grant['segments']) < 6):
            raise ValueError('invalid grant')
        if int(grant['deadline_ns']) <= 0:
            raise ValueError('invalid deadline')

    def _finish_generation(self, p, grant, reason, detail, new_token='none'):
        self.pending = None
        if p['reason'] in PLANNER_TARGET_REPLACEMENT_REASONS:
            if new_token.startswith('care_obs_v1_') and self.qp_owner_lock is not None:
                self.qp_owner_lock['observation_token'] = new_token
                self.qp_owner_lock['stage'] = 'hard_qp'
                self.qp_owner_lock['expires_wall'] = time.monotonic()+4.0
            else:
                self._release_qp_owner_locked(reason)
        self.send('action=finish request_id={} new_token={} generation_result={}'.format(
            p['request_id'], new_token, reason))
        self.log(reason, request_id=p['request_id'], grant=grant, new_token=new_token,
                 lifetime_wall_s=time.monotonic()-p['offered_wall'],
                 grant_received_wall=p.get('grant_received_wall'),
                 first_process_wall=p.get('first_process_wall'), **detail)

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
                    child_target = (owner_pin if owner_pin is not None and
                        owner_pin['region'] == region_key(live[cid]['points'])
                        else live[cid])
                    child_q = np.asarray(child_target['q_vis'], dtype=float)
                    if (parent_q.shape != (7,) or child_q.shape != (7,) or
                            not np.all(np.isfinite(parent_q)) or
                            not np.all(np.isfinite(child_q))):
                        reason = 'owner_promotion_invalid_target'
                        return
                    if not ancestor:
                        n._dependency_attempts[p['dependency_key']] = max(
                            2, n._dependency_attempts.get(p['dependency_key'], 0))
                    n._dependency_pins[pid] = dict(
                        region=region_key(live[pid]['points']), **p['previous_target'])
                    n._dependency_pins[cid] = dict(
                        region=region_key(live[cid]['points']),
                        **observation_target_state(child_target))
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
        generation_started_wall = time.monotonic()
        try:
            deadline = int(grant['deadline_ns'])*1e-9
            self._validate_generation_grant(p, grant)
            if self.now() >= deadline or time.monotonic()-p['offered_wall'] >= 2.5:
                reason = 'generation_expired'; return
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
                        # One reservation still solves one branch.  When the
                        # proposed diverse head scores far below an untried
                        # head at the measured seed, try the plausible head.
                        score_gap_override=0.75,
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
                        inference_started_wall = time.monotonic()
                        result = n._generate_active_set_waypoint(p['points'], trajectory, .1, received)
                        inference_ended_wall = time.monotonic()
                    finally:
                        n._seed_override = None
                        n._candidate_replacement_override = None
                hybrid = result.get('per_sensor_hybrid', {})
                for attempt in hybrid.get('attempts', []):
                    self.families[p['lineage']].add(int(attempt['sensor_id']))
                detail = dict(hybrid=hybrid, points=p['points'].tolist(), measured_seed=seed.tolist(),
                              measured_seed_stamp=stamp, generation_end_ros_s=self.now(),
                              model_compute_wall_s=inference_ended_wall-inference_started_wall,
                              generation_started_wall=generation_started_wall)
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
                detail.update(application_measured_stamp=live_stamp,
                              application_measured_q=None if live is None else np.asarray(live).tolist(),
                              seed_delta_inf=None if live is None else float(np.max(np.abs(np.asarray(live)-seed))))
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
                    n._dependency_pins[p['active_id']] = dict(region=p['region'],
                        **observation_target_state(ob), preserve_on_growth=True)
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
            self._finish_generation(p, grant, reason, detail, new_token)


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
        row = dict(event=event, ros_s=rospy.Time.now().to_sec(),
                   monotonic_s=time.monotonic(), **data)
        with path.open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        rospy.logwarn('[candidate_replacement] %s request=%s', event, data.get('request_id'))
    import os
    runtime_type = CandidateReplacement
    if os.environ.get('CARE_REPLACEMENT_EXECUTION_V2') == 'true':
        from replacement_execution_runtime import ReplacementExecution
        runtime_type = ReplacementExecution
    runtime = runtime_type(node, lambda: rospy.Time.now().to_sec(),
        lambda text: publisher.publish(String(data=text)), log)
    runtime.subscriber = rospy.Subscriber('/care_planner/local_planner/candidate_replacement_grant',
        String, lambda msg: runtime.receive(msg.data), queue_size=8)
    runtime.trigger_subscriber = rospy.Subscriber(
        '/care_planner/local_planner/candidate_replacement_trigger', String,
        lambda msg: runtime.offer_final_vbc(msg.data), queue_size=8)
    runtime.summary_subscriber = rospy.Subscriber(
        '/care_planner/local_planner/summary', String,
        lambda msg: runtime.observe_planner_summary(msg.data), queue_size=16)
    runtime.tracker_subscriber = rospy.Subscriber(
        '/care_planner/execution/tracker_summary', String,
        lambda msg: runtime.seed_gate.observe_tracker(msg.data, rospy.Time.now().to_sec()), queue_size=8)
    runtime.commit_ack_subscriber = rospy.Subscriber(
        '/care_planner/local_planner/candidate_replacement_commit_ack', String,
        lambda msg: runtime.seed_gate.observe_commit_ack(msg.data, rospy.Time.now().to_sec()), queue_size=8)
    return runtime
