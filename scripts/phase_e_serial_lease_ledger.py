#!/usr/bin/env python3
"""Isolated command-domain proposal/check/commit ledger, with no ROS publisher.

This exercises issuance ordering and liveness. It cannot qualify physical
response, future scheduling or environment conditions, or enable admission.
"""
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import threading
import time

from phase_e_stopping_contract import qualify
from phase_e_leased_stop_evidence import query_adaptive_stopping_snapshot


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


class LedgerReject(ValueError):
    pass


@dataclass(frozen=True)
class Proposal:
    serial: int
    payload: bytes

    @property
    def sha256(self):
        return hashlib.sha256(self.payload).hexdigest()

    def record(self):
        return json.loads(self.payload)


@dataclass(frozen=True)
class CheckedProposal:
    proposal: Proposal
    payload: bytes

    def record(self):
        return json.loads(self.payload)


class SerialLeaseLedger:
    """One in-flight proposal; every emitted, unexpired right remains enclosed.

    Returned grants are diagnostic records only. Acknowledgement loss never
    removes an emitted grant. Ordinary source updates within its allowed range
    may continue while a proposal is checked.
    """
    def __init__(self,runtime_sha256,update_gap_s=.001,next_decision_s=.01):
        if not isinstance(runtime_sha256,str) or len(runtime_sha256)!=64:
            raise ValueError('invalid_runtime_fingerprint')
        if not math.isfinite(update_gap_s) or update_gap_s<=0 or not math.isfinite(next_decision_s) or next_decision_s<0:
            raise ValueError('invalid_diagnostic_time_assumptions')
        self.runtime_sha256=runtime_sha256
        self.gap=update_gap_s;self.next_decision=next_decision_s
        self.lock=threading.RLock();self.state=None;self.rights={}
        self.proposal=None;self.proof=None;self.serial=0;self.sequence=0
        self.checking=None;self.committing=None
        self.retired=0;self.epoch=0;self.last_now=0;self.fault=None
        self.events=[]

    def _event(self,kind,**fields):
        self.events.append(dict(kind=kind,wall_s=time.monotonic(),runtime_admission=False,**copy.deepcopy(fields)))

    def drain_events(self):
        """Transfer diagnostics without retaining history in the live ledger."""
        with self.lock:
            values,self.events=self.events,[]
            return values

    def _invalidate(self,reason):
        self.epoch+=1;self.proposal=None;self.proof=None
        self._event('invalidate',reason=reason)

    def _time(self,now):
        if not isinstance(now,int) or now<=0 or now<self.last_now:
            self.fault='ledger_clock_regressed';self._invalidate(self.fault)
            raise LedgerReject(self.fault)
        self.last_now=now

    @staticmethod
    def _finite7(value,positive=False):
        return (isinstance(value,(list,tuple)) and len(value)==7 and
                all(isinstance(x,(float,int)) and math.isfinite(x) and
                    (not positive or x>0) for x in value))

    def observe(self,state):
        """Accept real-shaped receipts; unknown applied rights fault the session."""
        s=copy.deepcopy(state)
        with self.lock:
            if (not isinstance(s.get('controller_session'),int) or s['controller_session']<=0 or
                    not isinstance(s.get('t_ns'),int) or s['t_ns']<=0 or
                    s.get('mode') not in (0,1,2,3) or not s.get('lease_required') or
                    s.get('clock_fault') or not s.get('integral_valid') or
                    not all(self._finite7(s.get(k)) for k in ('q','applied','measured')) or
                    not all(self._finite7(s.get(k),True) for k in ('velocity_limits','acceleration_limits')) or
                    not math.isfinite(s.get('max_update_gap',math.inf)) or s['max_update_gap']>self.gap+1e-12 or
                    not math.isfinite(s.get('max_braking_step_s',0.)) or s['max_braking_step_s']<=0 or
                    not math.isfinite(s.get('max_execution_lease_s',0.)) or s['max_execution_lease_s']<=0 or
                    any(abs(u)>v+1e-12 for u,v in zip(s['applied'],s['velocity_limits']))):
                self.fault='invalid_controller_receipt';self._invalidate(self.fault)
                raise LedgerReject(self.fault)
            new_session=self.state is None or s['controller_session']!=self.state['controller_session']
            if new_session:
                self._invalidate('controller_session_changed')
                self.rights={};self.sequence=0;self.retired=0;self.fault=None;self.last_now=0
            elif s['t_ns']<self.state['t_ns']:
                self.fault='controller_receipt_time_regressed';self._invalidate(self.fault)
                raise LedgerReject(self.fault)
            if s['mode']==1:
                grant=self.rights.get(s.get('lease_sequence'))
                fields=('controller_session','execution_stamp_ns','valid_until_ns',
                        'target_range_enforced','target_abs_velocity','certificate_sha256')
                if grant is None or any(s.get(k)!=grant[k] for k in fields):
                    self.fault='unrecorded_applied_authorization';self._invalidate(self.fault)
                    raise LedgerReject(self.fault)
            if s['mode'] in (2,3):
                token=s.get('execution_stamp_ns',0)
                if not isinstance(token,int) or token<0:raise LedgerReject('invalid_retired_token')
                changed=(new_session or self.state['mode']!=s['mode'] or token>self.retired)
                self.retired=max(self.retired,token)
                # Stop/revocation wins even while a map query is in flight.
                if changed:self._invalidate('controller_braking_or_latched')
            self.state=s

    def revoke(self):
        with self.lock:
            self.retired=max([self.retired]+[g['execution_stamp_ns'] for g in self.rights.values()])
            self.fault='session_revoked';self._invalidate(self.fault)

    def _fresh(self,state,now):
        if not state:raise LedgerReject('missing_controller_receipt')
        if now<state['t_ns'] or now-state['t_ns']>50000000:
            raise LedgerReject('stale_or_future_controller_receipt')
        if self.fault:raise LedgerReject(self.fault)

    def begin(self,*,now_ns,token,target_abs_velocity,duration_ns=100000000):
        with self.lock:
            self._time(now_ns)
            if self.checking is not None or self.committing is not None:
                raise LedgerReject('query_or_commit_still_in_flight')
            if self.proposal is not None:raise LedgerReject('proposal_already_in_flight')
            s=self.state;self._fresh(s,now_ns)
            if not isinstance(token,int) or token<=self.retired:raise LedgerReject('retired_or_invalid_token')
            if (not self._finite7(target_abs_velocity) or
                    any(c<0 or c>v for c,v in zip(target_abs_velocity,s['velocity_limits']))):
                raise LedgerReject('invalid_proposed_target_range')
            if (not isinstance(duration_ns,int) or duration_ns<=0 or
                    duration_ns>int(s['max_execution_lease_s']*1e9)):
                raise LedgerReject('invalid_proposed_duration')
            phase={0:'initial_zero',1:'tracking',2:'braking',3:'rearm_zero'}[s['mode']]
            if phase=='braking':raise LedgerReject('controller_braking')
            if phase=='tracking':
                if token!=s['execution_stamp_ns']:raise LedgerReject('tracking_token_change')
                if now_ns>=s['valid_until_ns']:raise LedgerReject('active_authorization_expired')
            elif (any(target_abs_velocity) or any(abs(v)>1e-12 for v in s['applied']) or
                  any(abs(v)>.001 for v in s['measured'])):
                raise LedgerReject('initial_or_rearm_requires_rest_and_zero_range')
            self.serial+=1
            candidate=dict(controller_session=s['controller_session'],execution_stamp_ns=token,
                lease_sequence=self.sequence+1,issued_at_ns=now_ns,valid_until_ns=now_ns+duration_ns,
                target_range_enforced=True,target_abs_velocity=list(target_abs_velocity))
            rights=[g for g in self.rights.values() if g['valid_until_ns']>now_ns]
            union=rights+[candidate]
            cap=[max(g['target_abs_velocity'][j] for g in union) for j in range(7)]
            end=max(g['valid_until_ns'] for g in union)
            result=qualify(now_s=now_ns*1e-9,measured_s=s['t_ns']*1e-9,applied=s['applied'],
                acceleration=s['acceleration_limits'],velocity_limits=s['velocity_limits'],
                next_decision_s=self.next_decision,transport_bound_s=0.,update_gap_s=self.gap,
                lease_end_s=end*1e-9,lease_enforced=True,physical_error_rad=None,physical_settle_s=None,
                snapshot=None,max_braking_step_s=s['max_braking_step_s'],authorized_target_abs=cap)
            if not result['budgets']:raise LedgerReject('invalid_proposal_stopping_budget')
            payload=dict(serial=self.serial,epoch=self.epoch,phase=phase,state=s,candidate=candidate,
                created_wall_s=time.monotonic(),
                outstanding=rights,enclosed_until_ns=end,enclosed_target_abs_velocity=cap,
                budgets=result['budgets'],runtime_sha256=self.runtime_sha256,
                diagnostic_assumptions=dict(update_gap_s=self.gap,next_decision_s=self.next_decision),
                admission_deadline_ns=min(s['t_ns']+50000000,
                    s['valid_until_ns'] if phase=='tracking' else candidate['valid_until_ns']),
                runtime_admission=False)
            self.proposal=Proposal(self.serial,encoded(payload));self.proof=None
            self._event('propose',proposal_sha256=self.proposal.sha256,proposal=payload)
            return self.proposal

    def abort(self,proposal,reason='caller_aborted'):
        with self.lock:
            if proposal is self.proposal:
                self._event('abort',proposal_sha256=proposal.sha256,reason=reason)
                self.proposal=None;self.proof=None

    def check(self,proposal,*,geometry,query,now_ns,evidence_sink=None,wall_budget_s=.05):
        with self.lock:
            if proposal is not self.proposal:raise LedgerReject('proposal_not_current')
            if self.checking is not None or self.proof is not None:
                raise LedgerReject('proposal_check_already_started')
            self.checking=proposal
            p=proposal.record()
        try:
            deadline=p['created_wall_s']+wall_budget_s
            remaining=deadline-time.monotonic()
            if remaining<=0:raise LedgerReject('proposal_wall_budget_exhausted_before_check')
            result=query_adaptive_stopping_snapshot(budgets=p['budgets'],q=p['state']['q'],
                geometry=geometry,query=query,now_ns=now_ns,measured_ns=p['state']['t_ns'],
                authorization_until_ns=p['admission_deadline_ns'],enclosure_field='command_travel_rad',
                evidence_sink=evidence_sink,wall_budget_s=remaining)
            result['protocol_wall_deadline_s']=deadline
            proof=CheckedProposal(proposal,encoded(result))
            with self.lock:
                if proposal is not self.proposal:raise LedgerReject('proposal_invalidated_during_check')
                self.proof=proof
                self._event('check',proposal_sha256=proposal.sha256,result=result)
            return proof
        finally:
            with self.lock:
                if self.checking is proposal:self.checking=None

    def commit(self,proof,*,query,now_ns):
        """Re-read the map version, then atomically validate and record emission.

        Callers receive a diagnostic grant. No transport/publication occurs.
        Failure consumes the proposal and leaves every old deadline unchanged.
        """
        proposal=proof.proposal;p=proposal.record();result=proof.record()
        try:
            with self.lock:
                if proof is not self.proof or proposal is not self.proposal:
                    raise LedgerReject('checked_proposal_not_current')
                if self.committing is not None:raise LedgerReject('proposal_commit_already_started')
                if not result.get('diagnostic_accepted'):raise LedgerReject(result['adaptive']['reason'])
                if time.monotonic()>=result['protocol_wall_deadline_s']:
                    raise LedgerReject('proposal_wall_budget_exhausted_at_commit')
                self.committing=proof
            version=tuple(result['map_identity'])
            if hasattr(query,'call_with_timeout'):
                checked=query.call_with_timeout(result['protocol_wall_deadline_s']-time.monotonic(),[],0.,version)
            else:checked=query([],0.,version)
            with self.lock:
                now=now_ns();self._time(now)
                if time.monotonic()>=result['protocol_wall_deadline_s']:
                    raise LedgerReject('proposal_wall_budget_exhausted_at_commit')
                if proof is not self.proof or proposal is not self.proposal:
                    raise LedgerReject('proposal_invalidated_during_commit')
                self._fresh(self.state,now)
                if now>=p['admission_deadline_ns']:raise LedgerReject('proposal_commit_deadline_expired')
                if (p['epoch']!=self.epoch or p['runtime_sha256']!=self.runtime_sha256 or
                        self.state['controller_session']!=p['state']['controller_session']):
                    raise LedgerReject('proposal_binding_changed')
                if self.state['mode']!=p['state']['mode']:raise LedgerReject('controller_mode_changed')
                if any(self.state[k]!=p['state'][k] for k in
                       ('velocity_limits','acceleration_limits','max_braking_step_s','max_execution_lease_s')):
                    raise LedgerReject('controller_contract_changed')
                if p['phase']!='tracking' and (any(abs(v)>1e-12 for v in self.state['applied']) or
                        any(abs(v)>.001 for v in self.state['measured'])):
                    raise LedgerReject('controller_no_longer_at_rest')
                if p['candidate']['execution_stamp_ns']<=self.retired:raise LedgerReject('proposal_token_retired')
                if p['phase']=='tracking' and now>=self.state['valid_until_ns']:
                    raise LedgerReject('active_authorization_expired')
                if (not checked.get('valid') or checked.get('clock_fault') or
                        tuple(checked.get(k) for k in ('instance','epoch','revision'))!=version or
                        [checked.get(k) for k in ('frame','resolution','map_min','map_max','grid_size')]!=result['map_geometry'] or
                        checked.get('snapshot_ns',0)<result['snapshot_ns'] or checked['snapshot_ns']>now):
                    raise LedgerReject('map_version_or_time_changed_at_commit')
                # A first/rearm grant only authorizes zero; movement after a
                # latch additionally requires actual new-token zero + rearm.
                grant=dict(p['candidate'],certificate_sha256=hashlib.sha256(proof.payload).hexdigest(),
                    proposal_sha256=proposal.sha256,requires_explicit_rearm=p['phase']=='rearm_zero')
                self.rights[grant['lease_sequence']]=grant
                self.sequence=grant['lease_sequence'];self.proposal=None;self.proof=None
                # Bound storage only by expiration; ACK supersession alone is
                # insufficient. Stopping travel from existing output survives.
                self.rights={k:g for k,g in self.rights.items() if g['valid_until_ns']>now}
                self._event('diagnostic_commit',grant=grant,commit_ns=now,map_version_recheck=checked)
                return dict(grant=copy.deepcopy(grant),runtime_admission=False,
                    production_qualified=False,scope='isolated_command_domain_protocol')
        except (ValueError,RuntimeError):
            self.abort(proposal,'commit_rejected')
            raise
        finally:
            with self.lock:
                if self.committing is proof:self.committing=None
