"""Task-bound planning events. These messages never authorize execution."""
import copy
import json
import math
import re
import uuid

from task_observation_binding import canonical, validate_binding

RECOVERY_TOPIC = '/care_planner/active_sensing/task_recovery_snapshot'
CONTROL_TOPIC = '/care_planner/c4_4/task_control'
VERIFICATION_TOPIC = '/care_planner/verification_outcome/task_context'
PLANNER_TOPIC = '/care_planner/local_planner/task_event'
MAX_EVENT_BYTES = 65536
MAX_EVENT_NS = 250000000


class TaskEventSource:
    def __init__(self, session=None):
        self.session = session or uuid.uuid4().hex
        self.sequence = 0

    def event(self, kind, context, payload, now_ns, ttl_ns=MAX_EVENT_NS):
        binding = context['candidate_binding']
        validate_binding(binding, context)
        if not context.get('active') or not context['stamp_ns'] <= now_ns <= context['valid_until_ns']:
            raise ValueError('inactive_task_event_source')
        if type(ttl_ns) is not int or not 0 < ttl_ns <= MAX_EVENT_NS:
            raise ValueError('invalid_task_event_lifetime')
        self.sequence += 1
        value = dict(schema=1, kind=kind, source_session=self.session, event_seq=self.sequence,
                     task_context=binding, stamp_ns=now_ns,
                     valid_until_ns=min(now_ns+ttl_ns, context['valid_until_ns']),
                     payload=copy.deepcopy(payload))
        raw = canonical(value)
        if len(raw.encode('utf-8')) > MAX_EVENT_BYTES:
            raise ValueError('task_event_too_large')
        return value


class TaskEventReceiver:
    """Bounded replay protection, independent for each kind on ROS connections.

    Callers serialize this object with their context/publication lock. Invalid
    payloads must be rejected before calling remember(), so a malformed message
    cannot consume the sequence of a later valid event.
    """
    def __init__(self):
        self.sequences = {}
        self.sessions = set()

    def parse(self, raw, context, now_ns, kind):
        if not isinstance(raw, str) or len(raw.encode('utf-8')) > MAX_EVENT_BYTES:
            raise ValueError('task_event_too_large')
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('ambiguous_task_event')
                result[key] = value
            return result
        value = json.loads(raw, object_pairs_hook=unique)
        canonical(value)  # Reject NaN/Infinity, including unused payload fields.
        if (not isinstance(value, dict) or set(value) != {
                'schema', 'kind', 'source_session', 'event_seq', 'task_context',
                'stamp_ns', 'valid_until_ns', 'payload'} or type(value['schema']) is not int or value['schema'] != 1 or
                value['kind'] != kind or
                not isinstance(value['payload'], dict) or
                not isinstance(value['source_session'], str) or
                re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', value['source_session']) is None or
                any(type(value[k]) is not int or not 0 < value[k] < 2**64
                    for k in ('event_seq', 'stamp_ns', 'valid_until_ns'))):
            raise ValueError('invalid_task_event')
        validate_binding(value['task_context'], context)
        if (not context.get('active') or not context['stamp_ns'] <= now_ns <= context['valid_until_ns'] or
                not context['started_ns'] <= value['stamp_ns'] <= now_ns <= value['valid_until_ns'] or
                not 0 <= value['valid_until_ns']-value['stamp_ns'] <= MAX_EVENT_NS):
            raise ValueError('expired_or_future_task_event')
        key = (value['source_session'], kind)
        if value['event_seq'] <= self.sequences.get(key, 0):
            raise ValueError('replayed_task_event')
        if value['source_session'] not in self.sessions and len(self.sessions) >= 16:
            raise ValueError('task_event_source_capacity')
        if key not in self.sequences and len(self.sequences) >= 128:
            raise ValueError('task_event_kind_capacity')
        return value

    def remember(self, event):
        self.sessions.add(event['source_session'])
        self.sequences[(event['source_session'], event['kind'])] = event['event_seq']


def validate_recovery_snapshot(payload, binding=None):
    fields = {'episode_id', 'started', 'complete', 'pending_count', 'target'}
    if (set(payload) not in (fields, fields | {'selection'}) or
            type(payload['started']) is not bool or type(payload['complete']) is not bool or
            type(payload['pending_count']) is not int or not 0 <= payload['pending_count'] <= 256 or
            not isinstance(payload['episode_id'], str) or
            (payload['started'] and re.fullmatch(r'[0-9a-f]{32}', payload['episode_id']) is None) or
            (not payload['started'] and (payload['episode_id'] or payload['complete']))):
        raise ValueError('invalid_task_recovery_snapshot')
    target = payload['target']
    if payload['complete'] and (not payload['started'] or payload['pending_count'] or target is not None):
        raise ValueError('unproven_task_recovery_completion')
    if 'selection' in payload:
        from recovery_policy import validate_selection
        validate_selection(payload['selection'], target, binding)
    if target is None:
        return payload
    if (not isinstance(target, dict) or set(target) != {
            'observation_token', 'region_token', 'obligation_id', 'q_vis', 'joint_mask',
            'points', 'deadline_ns', 'sensor_id'} or type(target['obligation_id']) is not int or
            type(target['sensor_id']) is not int or not -1 <= target['sensor_id'] <= 7 or
            target['obligation_id'] < 0 or type(target['deadline_ns']) is not int or
            not 0 < target['deadline_ns'] < 2**64 or not payload['started'] or
            payload['pending_count'] <= 0):
        raise ValueError('invalid_task_recovery_target')
    for key, prefix in [('observation_token', 'care_obs_v1_'), ('region_token', 'care_region_v1_')]:
        if (not isinstance(target[key], str) or re.fullmatch(
                prefix+str(target['obligation_id'])+r'_[0-9a-f]{24}', target[key]) is None):
            raise ValueError('invalid_task_recovery_identity')
    def vector(values, size):
        return (isinstance(values, list) and len(values) == size and
                all(type(v) in (int, float) and math.isfinite(v) for v in values))
    if (not vector(target['q_vis'], 7) or not vector(target['joint_mask'], 7) or
            not all(0 <= v <= 1 for v in target['joint_mask']) or not any(target['joint_mask']) or
            not isinstance(target['points'], list) or not 0 < len(target['points']) <= 256 or
            not all(vector(point, 3) for point in target['points'])):
        raise ValueError('invalid_task_recovery_geometry')
    return payload


def validate_regime_control(payload):
    fields = {'mode', 'mode_revision', 'ready', 'clear', 'replan_seq',
              'reference_refresh_seq', 'force_bootstrap', 'qp_retry'}
    if (set(payload) not in (fields, fields | {'recovery_target_sha256'}) or
            ('recovery_target_sha256' in payload and
             re.fullmatch(r'([0-9a-f]{64})?', payload['recovery_target_sha256']) is None) or
            payload['mode'] not in ('NORMAL', 'REPAIR', 'PROBE_NORMAL') or
            any(type(payload[k]) is not bool for k in ('ready', 'clear', 'force_bootstrap')) or
            type(payload['mode_revision']) is not int or not 0 < payload['mode_revision'] < 2**53 or
            not isinstance(payload['qp_retry'], str) or len(payload['qp_retry']) > 256 or
            (payload['qp_retry'] and re.fullmatch(r'[A-Za-z0-9_.:-]+', payload['qp_retry']) is None) or
            any(type(payload[k]) is not int or not 0 <= payload[k] < 2**53
                for k in ('replan_seq', 'reference_refresh_seq'))):
        raise ValueError('invalid_task_regime_control')
    return payload


def regime_identity(event):
    payload = validate_regime_control(event['payload'])
    return event['source_session']+':'+str(payload['mode_revision'])+':'+payload['mode']


class RecoverySnapshotSource:
    """A completion is a later empty ledger in the SAME task/episode.

    The caller supplies only current-task obligations and the actual context
    demand ledger. Cancellation resets this object rather than completing it.
    """
    def __init__(self):
        self.events = TaskEventSource()
        self.binding = None
        self.episode = ''
        self.complete = False

    def reset(self):
        self.binding = None
        self.episode = ''
        self.complete = False

    def snapshot(self, context, pending_count, target, now_ns, policy=None):
        if self.binding != context['candidate_binding']:
            self.reset()
            self.binding = context['candidate_binding']
        if pending_count and (not self.episode or self.complete):
            self.episode = uuid.uuid4().hex
        self.complete = bool(self.episode and not pending_count and target is None)
        payload = dict(episode_id=self.episode, started=bool(self.episode), complete=self.complete,
                       pending_count=pending_count, target=copy.deepcopy(target))
        from recovery_policy import selection_record
        payload['selection'] = selection_record(target, policy)
        validate_recovery_snapshot(payload, context['candidate_binding'])
        return self.events.event('recovery_snapshot', context, payload, now_ns)
