"""Immutable geometry-verified proposals awaiting stopping qualification.

This module never publishes a lease or velocity. A preparation cannot be used
as execution permission; the downstream qualified issuance/application
handshake must precede any motion.
"""
import copy
import json
import re
import uuid

from candidate_admission import payload_sha256, transfer_message
from task_observation_binding import canonical


class ExecutionPreparationLedger:
    def __init__(self):
        self.session = uuid.uuid4().hex
        self.sequence = 0
        self.pending = None
        self.source_sequences = {}
        self.last_now = 0
        self.clock_fault = False
        self.reason = 'waiting_geometry_verification'

    def cancel(self, reason):
        self.pending = None
        self.reason = reason

    def _clock(self, now_ns):
        if type(now_ns) is not int or now_ns <= 0:
            raise ValueError('invalid_preparation_clock')
        if now_ns < self.last_now:
            self.clock_fault = True
            self.cancel('preparation_clock_reversed')
        self.last_now = max(self.last_now, now_ns)
        if self.clock_fault:
            raise ValueError('preparation_clock_reversed')

    def prepare(self, trajectory, task_context, task_regime, source_key,
                source_digest, verification_seq, now_ns, expires_ns):
        self._clock(now_ns)
        if (type(expires_ns) is not int or not now_ns < expires_ns <= now_ns+250000000 or
                type(verification_seq) is not int or verification_seq <= 0 or
                not re.fullmatch(r'[0-9a-f]{64}', source_digest or '') or
                not isinstance(task_context, str) or not 0 < len(task_context) <= 16384 or
                not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}:[1-9][0-9]*:(NORMAL|REPAIR|PROBE_NORMAL)', task_regime or '') or
                not trajectory.points or trajectory.header.stamp.to_nsec() <= 0):
            raise ValueError('invalid_execution_preparation')
        context = json.loads(task_context)
        if context.get('schema') != 1 or canonical(context) != task_context:
            raise ValueError('invalid_preparation_task_binding')
        source_session, source_sequence = source_key
        if (not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', source_session or '') or
                type(source_sequence) is not int or not 0 < source_sequence < 2**64):
            raise ValueError('invalid_preparation_source')
        identity = (source_session, source_sequence, source_digest, task_context, task_regime)
        motion_digest = payload_sha256(trajectory, '', task_context, task_regime)
        if self.pending and self.pending['identity'] == identity:
            if self.pending['motion_digest'] != motion_digest:
                raise ValueError('preparation_payload_mutated')
            # Repetition preserves both the reserved token and FIRST deadline.
            message = self.snapshot(now_ns, task_context, task_regime)
            if message is None:
                raise ValueError('preparation_already_expired')
            return message
        if source_sequence <= self.source_sequences.get(source_session, 0):
            raise ValueError('retired_preparation_source')
        if source_session not in self.source_sequences and len(self.source_sequences) >= 16:
            raise ValueError('preparation_source_capacity')
        metadata = canonical(dict(schema=1, phase='execution_prepare', verified_ns=now_ns,
            expires_ns=expires_ns, verification_seq=verification_seq,
            source_session=source_session, source_sequence=source_sequence,
            source_payload_sha256=source_digest, execution_qualification='UNMET'))
        self.sequence += 1
        digest = payload_sha256(trajectory, metadata, task_context, task_regime)
        message = transfer_message('execution_prepare', self.session, self.sequence, raw=trajectory,
            digest=digest, remaining=(expires_ns-now_ns)*1e-9, prefix_context=metadata,
            task_context=task_context, task_regime=task_regime)
        self.pending = dict(identity=identity, motion_digest=motion_digest, message=message,
                            expires_ns=expires_ns)
        self.source_sequences[source_session] = source_sequence
        self.reason = 'waiting_stopping_qualification_and_application_handshake'
        return copy.deepcopy(message)

    def snapshot(self, now_ns, task_context, task_regime):
        self._clock(now_ns)
        pending = self.pending
        if pending is None:
            return None
        message = pending['message']
        if now_ns >= pending['expires_ns']:
            self.cancel('execution_preparation_expired')
            return None
        if message.task_context != task_context or message.task_regime != task_regime:
            self.cancel('execution_preparation_context_changed')
            return None
        result = copy.deepcopy(message)
        result.remaining_budget_s = (pending['expires_ns']-now_ns)*1e-9
        return result

    def status(self):
        pending = self.pending
        return dict(state='WAITING_STOPPING_QUALIFICATION' if pending else 'IDLE',
            reason=self.reason, runtime_admission=False, execution_authorized=False,
            preparation_session=self.session, preparation_sequence=self.sequence,
            execution_stamp_ns=pending['message'].trajectory.header.stamp.to_nsec() if pending else 0,
            expires_ns=pending['expires_ns'] if pending else 0)
