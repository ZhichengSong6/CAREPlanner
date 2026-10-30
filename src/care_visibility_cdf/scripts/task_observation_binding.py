"""Task ingress and immutable candidate lineage; no execution authority."""
import copy
import json
import math
import re
import uuid

from stopping_observation_context import ObservationContext, context_key


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def binding_of(event):
    value = {k: event[k] for k in (
        'schema', 'goal_request_id', 'owner_session', 'generation', 'context_id',
        'started_ns', 'frame_id', 'region_min', 'region_max', 'target')}
    value['map_scope'] = event.get('map_scope')
    return canonical(value)


def validate_binding(raw, context, frame_id=None):
    if not isinstance(raw, str) or not 0 < len(raw) <= 16384:
        raise ValueError('missing_task_candidate_binding')
    value = json.loads(raw)
    if context is None or raw != binding_of(context):
        raise ValueError('task_candidate_context_mismatch')
    if frame_id is not None and value['frame_id'] != frame_id:
        raise ValueError('task_candidate_frame_mismatch')
    return value


class TaskContextReceiver:
    def __init__(self):
        self.context = ObservationContext()
        self.pending_candidate = None

    def update(self, event, now_ns):
        # Validate the immutable envelope before accepting its heartbeat.
        if event.get('candidate_binding') != binding_of(event):
            raise ValueError('task_context_binding_mismatch')
        old = self.context.current
        if old and context_key(old) == context_key(event) and binding_of(old) != binding_of(event):
            raise ValueError('task_context_identity_mutated')
        return self.context.update(event, now_ns)

    def snapshot(self, now_ns):
        return self.context.snapshot(now_ns)

    def accepts(self, raw, now_ns, frame_id=None):
        try:
            validate_binding(raw, self.snapshot(now_ns), frame_id)
            return True
        except (ValueError, KeyError, TypeError):
            return False

    def offer_candidate(self, trajectory, raw, now_ns, metadata=None):
        """One fresh envelope may arrive before its context; it cannot run yet."""
        from stopping_trajectory_prefix import trajectory_sha256
        if not isinstance(raw, str) or not 0 < len(raw) <= 16384:
            raise ValueError('missing_task_candidate_binding')
        value = json.loads(raw)
        if (not isinstance(value, dict) or value.get('schema') != 1 or canonical(value) != raw or
                type(value.get('generation')) is not int or value['generation'] <= 0 or
                not isinstance(value.get('owner_session'), str) or not value['owner_session'] or
                type(value.get('started_ns')) is not int or value['started_ns'] <= 0):
            raise ValueError('invalid_task_candidate_binding')
        stamp = trajectory.header.stamp.to_nsec()
        if stamp < value['started_ns'] or stamp-now_ns > 50000000 or now_ns-stamp > 800000000:
            raise ValueError('stale_task_candidate')
        current = self.context.current
        if current and value['owner_session'] == current['owner_session']:
            if value['generation'] < current['generation'] or (
                    value['generation'] == current['generation'] and
                    (raw != current.get('candidate_binding') or self.snapshot(now_ns) is None)):
                raise ValueError('task_candidate_context_mismatch')
        elif current and value['owner_session'] in self.context.sessions:
            raise ValueError('old_task_candidate_owner')
        digest = trajectory_sha256(trajectory)
        metadata = copy.deepcopy(metadata or {})
        if len(canonical(metadata)) > 32768:
            raise ValueError('task_candidate_metadata_too_large')
        old = self.pending_candidate
        if (old is None or old['digest'] != digest or old['binding'] != raw or
                old.get('metadata', {}) != metadata):
            self.pending_candidate = dict(trajectory=copy.deepcopy(trajectory),binding=raw,
                                          digest=digest,received_ns=now_ns,metadata=metadata)

    def candidate(self, now_ns):
        candidate = self.candidate_envelope(now_ns)
        if candidate is None:
            return None
        return (candidate['trajectory'], candidate['binding'], candidate['received_ns'])

    def candidate_envelope(self, now_ns):
        candidate = self.pending_candidate
        if candidate is None:
            return None
        if not 0 <= now_ns-candidate['received_ns'] <= 800000000:
            self.pending_candidate = None
            return None
        if candidate['trajectory'].header.stamp.to_nsec() > now_ns:
            return None  # Wait for an out-of-order /clock update; never execute early.
        if not self.accepts(candidate['binding'],now_ns,candidate['trajectory'].header.frame_id):
            return None
        return copy.deepcopy(candidate)


class ObservationTaskOwner:
    """One explicit request stream owns goal contexts and map replacement.

    IDs cannot be reused after termination. Ordinary mode changes and candidate
    replacements are not task events. Capacity exhaustion rejects new requests.
    """
    def __init__(self, session=None, heartbeat_s=.8, capacity=256):
        self.session = session or uuid.uuid4().hex
        self.heartbeat_ns = round(heartbeat_s*1e9)
        if not 0 < self.heartbeat_ns <= 2000000000:
            raise ValueError('invalid_task_heartbeat')
        self.capacity = capacity
        self.requests = {}
        self.current = None
        self.generation = self.sequence = 0
        self.last_now = 0

    @staticmethod
    def _request(command):
        request_id = command.get('request_id')
        if not isinstance(request_id, str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', request_id):
            raise ValueError('invalid_task_request_id')
        target = copy.deepcopy(command.get('target'))
        lo, hi = command.get('region_min'), command.get('region_max')
        if not isinstance(target, dict) or not isinstance(target.get('frame_id'), str) or not target['frame_id']:
            raise ValueError('invalid_task_target_frame')
        vectors = (target.get('position'), target.get('orientation'), lo, hi)
        if any(not isinstance(v, list) or len(v) != n or any(
                type(x) not in (int, float) or not math.isfinite(x) for x in v)
               for v, n in zip(vectors, (3, 4, 3, 3))):
            raise ValueError('invalid_task_geometry')
        if any(a >= b for a, b in zip(lo, hi)):
            raise ValueError('invalid_task_region')
        norm = math.sqrt(sum(x*x for x in target['orientation']))
        if not math.isfinite(norm) or norm < 1e-12:
            raise ValueError('invalid_task_orientation')
        target['orientation'] = [x/norm for x in target['orientation']]
        # q and -q describe the same target; select one representation.
        first = next((x for x in reversed(target['orientation']) if abs(x) > 1e-15), 1.)
        if first < 0: target['orientation'] = [-x for x in target['orientation']]
        return request_id, dict(target=target, region_min=list(lo), region_max=list(hi))

    def _event(self, now_ns, active, reason):
        self.sequence += 1
        self.current.update(event_seq=self.sequence, stamp_ns=now_ns,
                            valid_until_ns=now_ns+self.heartbeat_ns,
                            active=active, reason=reason)
        self.current['candidate_binding'] = binding_of(self.current)
        return copy.deepcopy(self.current)

    def _start(self, request_id, spec, now_ns, reason):
        self.generation += 1
        self.current = dict(schema=1, goal_request_id=request_id, owner_session=self.session,
            generation=self.generation, context_id=uuid.uuid4().hex, started_ns=now_ns,
            frame_id=spec['target']['frame_id'], **copy.deepcopy(spec))
        return self._event(now_ns, True, reason)

    def tick(self, now_ns):
        if not self.current or not self.current['active']:
            self.last_now = now_ns
            return []
        # A missed owner heartbeat cannot revive the same generation.
        if now_ns < self.last_now or now_ns > self.current['valid_until_ns']:
            self.last_now = now_ns
            return [self._event(now_ns, False, 'owner_clock_or_heartbeat_expired')]
        self.last_now = now_ns
        return [self._event(now_ns, True, 'heartbeat')]

    def command(self, command, now_ns):
        if command.get('schema') != 1:
            raise ValueError('invalid_task_command_schema')
        if now_ns <= 0 or now_ns < self.last_now:
            raise ValueError('invalid_task_clock')
        self.last_now = now_ns
        action = command.get('action')
        if action == 'submit':
            request_id, spec = self._request(command)
            if request_id in self.requests:
                if self.requests[request_id] != spec:
                    raise ValueError('task_request_id_conflict')
                if not self.current or self.current['goal_request_id'] != request_id or not self.current['active']:
                    raise ValueError('task_request_already_terminated')
                if now_ns > self.current['valid_until_ns']:
                    return [self._event(now_ns, False, 'owner_heartbeat_expired')]
                return [self._event(now_ns, True, 'duplicate_submit')]
            if len(self.requests) >= self.capacity:
                raise ValueError('task_request_capacity')
            events = []
            if self.current and self.current['active']:
                events.append(self._event(now_ns, False, 'goal_replaced'))
            self.requests[request_id] = spec
            return events + [self._start(request_id, spec, now_ns, 'goal_submitted')]
        if action not in ('cancel', 'complete', 'replace_route'):
            raise ValueError('invalid_task_action')
        if (not self.current or command.get('request_id') != self.current['goal_request_id'] or
                command.get('context_id') != self.current['context_id']):
            raise ValueError('stale_task_terminal_event')
        if not self.current['active']:
            return []
        spec = None
        if action == 'replace_route':
            _, spec = self._request(dict(command, target=self.current['target']))
            spec['map_scope'] = self.current.get('map_scope')
        event = self._event(now_ns, False, action)
        if action != 'replace_route':
            return [event]
        return [event, self._start(self.current['goal_request_id'], spec, now_ns, 'route_replaced')]

    def map_invalidated(self, message, now_ns):
        old = message.get('observation_context')
        if (message.get('kind') != 'observation_context_invalidated' or
                message.get('reason') != 'observation_context_map_changed' or
                not self.current or not self.current['active'] or not isinstance(old, dict) or
                context_key(old) != context_key(self.current)):
            return []
        # Only close here. Reopening requires an explicit fresh-map event.
        return [self._event(now_ns, False, 'map_invalidated')]

    def resume_map(self, command, now_ns):
        if (command.get('schema') != 1 or command.get('action') != 'resume_map' or
                not self.current or self.current['active'] or self.current.get('reason') != 'map_invalidated' or
                command.get('context_id') != self.current['context_id'] or
                command.get('request_id') != self.current['goal_request_id'] or
                not isinstance(command.get('fresh_map_scope'), dict) or
                not command['fresh_map_scope'].get('instance')):
            raise ValueError('invalid_fresh_map_resume')
        spec = {k: self.current[k] for k in ('target', 'region_min', 'region_max')}
        spec['map_scope'] = copy.deepcopy(command['fresh_map_scope'])
        return [self._start(self.current['goal_request_id'], spec, now_ns, 'fresh_map_resumed')]
