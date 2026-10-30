"""Validate immutable preparations and explain stopping-qualification gaps.

Staging is permitted without motion authority. This module cannot issue a
lease, activate a curve, or turn a diagnostic result into qualification.
"""
import copy
import json
import math
import re
import time

from candidate_admission import payload_sha256
from execution_motion_limits import violation
from stopping_trajectory_prefix import trajectory_record
from task_observation_binding import TaskContextReceiver, canonical
from task_recovery_events import TaskEventReceiver, regime_identity, validate_regime_control

QUALIFICATION_GAPS = {
    'qualified_future_update_and_period_contract':
        'No enforced future update-gap and period==elapsed contract for the selected runtime.',
    'full_operating_domain_response_bound':
        'No qualified mechanical error and settling bounds over the selected operating domain.',
    'full_physical_enclosure_map_evidence':
        'No complete physical withdraw-now and continue-then-stop map certificate for this proposal.',
    'qualified_future_environment_conditions':
        'ideal_fov and finite observations do not establish future physical visibility/environment conditions.'}


class ExecutionHandoffShadow:
    def __init__(self, names, velocity, acceleration, runtime_sha256='', wall_clock=time.monotonic):
        if len(names) != 7 or len(set(names)) != 7 or any(
                len(v) != 7 or any(not math.isfinite(x) or x <= 0 for x in v)
                for v in (velocity, acceleration)):
            raise ValueError('invalid_handoff_motion_contract')
        self.names, self.velocity, self.acceleration = list(names), list(velocity), list(acceleration)
        self.runtime_sha256 = runtime_sha256
        self.context = TaskContextReceiver()
        self.events = TaskEventReceiver()
        self.control = None
        self.pending = None
        self.source_sequences = {}
        self.controller = None
        self.retired_sessions = set()
        self.last_now = 0
        self.clock_fault = False
        self.wall_clock = wall_clock
        self.reason = 'waiting_preparation'

    def cancel(self, reason):
        self.pending = None
        self.reason = reason

    def _clock(self, now):
        if type(now) is not int or now <= 0:
            raise ValueError('invalid_handoff_clock')
        if now < self.last_now:
            self.clock_fault = True
            self.cancel('handoff_clock_reversed')
        self.last_now = max(self.last_now, now)
        if self.clock_fault:
            raise ValueError('handoff_clock_reversed')

    def update_context(self, value, now):
        before = self.context.context.current
        before = before.get('candidate_binding') if before else None
        self.context.update(value, now)
        current = self.context.snapshot(now)
        if not current or current['candidate_binding'] != before:
            self.cancel('task_context_changed')
            self.control = None

    def update_control(self, raw, now):
        event = self.events.parse(raw, self.context.snapshot(now), now, 'regime_control')
        validate_regime_control(event['payload'])
        if self.control and event['source_session'] == self.control['source_session']:
            if event['payload']['mode_revision'] < self.control['payload']['mode_revision']:
                raise ValueError('retired_handoff_regime')
            if (event['payload']['mode_revision'] == self.control['payload']['mode_revision'] and
                    regime_identity(event) != regime_identity(self.control)):
                raise ValueError('mutated_handoff_regime')
        self.events.remember(event)
        if (not self.control or regime_identity(event) != regime_identity(self.control) or
                not event['payload']['ready']):
            self.cancel('task_regime_changed')
        self.control = event

    def observe_controller(self, state, now):
        if (not state.controller_session or not state.execution_lease_required or state.clock_fault or
                not 0 <= now-state.header.stamp.to_nsec() <= 50000000 or
                list(state.velocity_limits) != self.velocity or
                list(state.acceleration_limits) != self.acceleration or
                any(not math.isfinite(x) for values in (
                    state.applied_velocity, state.measured_velocity, state.measured_position) for x in values)):
            return False
        old = self.controller
        if old and state.controller_session == old.controller_session:
            if state.header.stamp <= old.header.stamp:
                return False
        elif old:
            if state.controller_session in self.retired_sessions or len(self.retired_sessions) >= 16:
                return False
            self.retired_sessions.add(old.controller_session)
            self.cancel('controller_session_changed')
        self.controller = copy.deepcopy(state)
        if state.mode >= 2:
            self.cancel('controller_stopping')
        return True

    def current(self, now):
        self._clock(now)
        context = self.context.snapshot(now)
        control = self.control
        if (not context or not control or not control['payload']['ready'] or
                not control['stamp_ns'] <= now < control['valid_until_ns'] or
                control['task_context'] != context['candidate_binding']):
            self.cancel('handoff_context_or_control_expired')
            return None
        if self.pending and (now >= self.pending['expires_ns'] or self.wall_clock() >= self.pending['wall_until']):
            self.cancel('execution_preparation_expired')
        return context

    def accept(self, message, now):
        context = self.current(now)
        if context is None:
            raise ValueError('handoff_context_or_control_expired')
        if (message.action != 'execution_prepare' or
                not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', message.session) or
                type(message.sequence) is not int or not 0 < message.sequence < 2**64 or
                message.task_context != context['candidate_binding'] or
                message.task_regime != regime_identity(self.control) or
                message.trajectory.header.frame_id != context['frame_id'] or
                len(message.planned_prefix_context) > 32768 or
                message.payload_sha256 != payload_sha256(message.trajectory,
                    message.planned_prefix_context, message.task_context, message.task_regime)):
            raise ValueError('invalid_handoff_identity_or_digest')
        meta = json.loads(message.planned_prefix_context)
        if (not isinstance(meta, dict) or canonical(meta) != message.planned_prefix_context or
                set(meta) != {'schema', 'phase', 'verified_ns', 'expires_ns', 'verification_seq',
                    'source_session', 'source_sequence', 'source_payload_sha256', 'execution_qualification'} or
                type(meta['schema']) is not int or meta['schema'] != 1 or
                meta['phase'] != 'execution_prepare' or meta['execution_qualification'] != 'UNMET' or
                any(type(meta[k]) is not int or not 0 < meta[k] < 2**64 for k in (
                    'verified_ns', 'expires_ns', 'verification_seq', 'source_sequence')) or
                not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', meta['source_session']) or
                not re.fullmatch(r'[0-9a-f]{64}', meta['source_payload_sha256']) or
                not context['started_ns'] <= meta['verified_ns'] <= now < meta['expires_ns'] or
                meta['expires_ns']-meta['verified_ns'] > 250000000):
            raise ValueError('invalid_handoff_metadata_or_deadline')
        record = trajectory_record(message.trajectory)
        if len(record['points']) > 257 or record['joint_names'] != self.names:
            raise ValueError('invalid_handoff_motion_dimensions')
        failure = violation(message.trajectory, self.velocity, self.acceleration)
        if failure:
            raise ValueError(failure)
        identity = message.session, message.sequence
        if self.pending and identity == self.pending['identity']:
            if message.payload_sha256 != self.pending['message'].payload_sha256:
                raise ValueError('handoff_payload_mutated')
            return None  # Neither forwarding nor deadlines restart on repetition.
        if message.sequence <= self.source_sequences.get(message.session, 0):
            raise ValueError('retired_handoff_preparation')
        if message.session not in self.source_sequences and len(self.source_sequences) >= 16:
            raise ValueError('handoff_source_capacity')
        state = self.controller
        if (not state or not 0 <= now-state.header.stamp.to_nsec() <= 50000000 or
                state.mode != 0 or message.trajectory.header.stamp.to_nsec() <= state.retired_execution_through_ns or
                any(v != 0. for v in state.applied_velocity) or
                any(abs(v) > .001 for v in state.measured_velocity)):
            raise ValueError('handoff_requires_fresh_waiting_rest')
        self.source_sequences[message.session] = message.sequence
        # Same-regime heartbeats can arrive on different ROS connections in a
        # different order. Keep the immutable upstream deadline, and enforce
        # the EARLIEST locally known deadline without relabelling its payload.
        local_until = min(meta['expires_ns'], context['valid_until_ns'], self.control['valid_until_ns'])
        self.pending = dict(identity=identity, message=copy.deepcopy(message), expires_ns=local_until,
            wall_until=self.wall_clock()+(local_until-now)*1e-9,
            controller_session=state.controller_session)
        self.reason = 'staged_waiting_stopping_qualification'
        return copy.deepcopy(message)

    def report(self, now):
        try:
            self.current(now)
        except ValueError:
            pass
        pending = self.pending
        return dict(state=self.reason, staged=bool(pending), runtime_admission=False,
            production_issuer_enabled=False, execution_authorized=False,
            missing_qualification=list(QUALIFICATION_GAPS), qualification_gaps=copy.deepcopy(QUALIFICATION_GAPS),
            runtime_sha256=self.runtime_sha256,
            task_regime=regime_identity(self.control) if self.control else '',
            preparation_sha256=pending['message'].payload_sha256 if pending else '',
            execution_stamp_ns=pending['message'].trajectory.header.stamp.to_nsec() if pending else 0,
            expires_ns=pending['expires_ns'] if pending else 0)
