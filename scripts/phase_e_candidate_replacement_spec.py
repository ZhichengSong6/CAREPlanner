"""OFFLINE SPEC ONLY: bounded replacement of an active dependency target.

Not imported by production. Tickets permit changing a steering proposal only;
they never authorize execution, release UNKNOWN guards, or complete obligations.
The ledger must belong to the original obligation lineage, not its mutable token.
"""
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/care_visibility_cdf/scripts'))
from bounded_visibility_recovery import region_key
from observation_identity import observation_token


def native_limits():
    text = (ROOT/'src/egocentric_arm_planner/include/egocentric_arm_planner/safe_frontier_recovery.hpp').read_text()
    return tuple(int(re.search(r'max_'+name+r'\s*=\s*(\d+)',text)[1])
                 for name in ['attempts','segments'])


def vector(q):
    values = tuple(float(v) for v in q)
    if len(values)!=7 or not all(math.isfinite(v) for v in values):
        raise ValueError('invalid q')
    return values


@dataclass(frozen=True)
class Context:
    mode_epoch: int
    query_stamp_ns: int
    map_stamp_ns: int
    active_id: int
    parent_id: int
    token: str
    points_key: tuple
    measured_q: tuple


@dataclass(frozen=True)
class Ticket:
    serial: int
    context: Context
    family: tuple


@dataclass
class Ledger:
    """One ledger survives token/q changes and temporary task preemption."""
    attempts: int
    segments: int
    attempted_families: set = field(default_factory=set)
    attempted_poses: list = field(default_factory=list)


class ReplacementSpec:
    def __init__(self, context, ledger):
        self.lineage=(context.mode_epoch,context.active_id,context.points_key)
        self.ledger=ledger
        self.max_attempts,self.max_segments=native_limits()
        if ledger.attempts<0 or ledger.segments<0: raise ValueError('invalid inherited budget')
        self.pending=None
        self.serial=0
        self.state='HOLD'
        self.events=[]

    def event(self, reason):
        self.events.append(dict(reason=reason,state=self.state,
                                attempts=self.ledger.attempts,segments=self.ledger.segments))
        return reason

    def reserve(self, context, stack, edges, family, cycle_query_identity):
        if self.pending is not None:
            return self.event('generation_in_flight'),None
        if (context.mode_epoch,context.active_id,context.points_key)!=self.lineage:
            return self.event('wrong_lineage'),None
        if (context.query_stamp_ns<=0 or context.map_stamp_ns<=0 or
            not context.token or context.parent_id==context.active_id or
            not stack or stack[-1]!=context.active_id or context.parent_id not in stack[:-1] or
            edges.get(context.active_id)!=context.parent_id or
            cycle_query_identity!=(context.mode_epoch,context.query_stamp_ns,context.token)):
            return self.event('no_current_ancestor_cycle'),None
        vector(context.measured_q)
        family=tuple(family)
        if len(family)!=2 or not family[0] or not isinstance(family[1],int) or not 0<=family[1]<8:
            return self.event('invalid_family'),None
        if family in self.ledger.attempted_families:
            return self.event('family_already_attempted'),None
        if self.ledger.attempts>=self.max_attempts or self.ledger.segments>=self.max_segments:
            self.state='EXHAUSTED'
            return self.event('budget_exhausted'),None
        # Charge before async generation; errors, stale replies and rejections cost an attempt.
        self.ledger.attempts+=1
        self.ledger.attempted_families.add(family)
        self.serial+=1
        self.pending=Ticket(self.serial,context,family)
        self.state='RESERVED'
        return self.event('reserved'),self.pending

    def reject(self,ticket,reason):
        if ticket!=self.pending:return self.event('stale_ticket')
        self.pending=None;self.state='HOLD'
        return self.event(reason)

    def apply(self,ticket,current,candidate,node):
        """Apply to an INERT test node; production integration is intentionally absent.

        Caller supplies the current query/map/measured context. Lock order follows
        production publication: schedule RLock -> obligation Lock -> release -> publish.
        """
        if ticket!=self.pending:return self.event('stale_ticket')
        if current!=ticket.context:return self.reject(ticket,'generation_context_changed')
        if self.ledger.segments>=self.max_segments or self.ledger.attempts>self.max_attempts:
            return self.reject(ticket,'budget_exhausted_during_generation')
        if (candidate.get('checkpoint_sha'),candidate.get('sensor'))!=ticket.family:
            return self.reject(ticket,'candidate_family_mismatch')
        if candidate.get('geometry_accepted') is not True or candidate.get('local_progress_evidence') is not True:
            return self.reject(ticket,'candidate_not_supported')
        try:
            q=vector(candidate['q_vis']);zero=vector(candidate['q_zero'])
        except (KeyError,ValueError,TypeError):return self.reject(ticket,'invalid_candidate')
        if any(max(abs(a-b) for a,b in zip(q,old))<=.01 for old in self.ledger.attempted_poses):
            return self.reject(ticket,'pose_already_attempted')
        with node._schedule_publish_lock:
            with node._obligation_lock:
                by_id={int(ob['id']):ob for ob in node._obligations}
                ob=by_id.get(current.active_id)
                if (ob is None or current.parent_id not in by_id or
                    node._repair_stack[-1:]!=[current.active_id] or
                    node._dependency_edges.get(current.active_id)!=current.parent_id):
                    return self.reject(ticket,'active_dependency_changed')
                pin=node._dependency_pins.get(current.active_id)
                published=getattr(node,'_trace_published_target',{})
                if vector(node._latest_measured_q)!=current.measured_q:
                    return self.reject(ticket,'measured_state_changed')
                if (region_key(ob['points'])!=current.points_key or
                    published.get('observation_token')!=current.token or
                    observation_token(ob)!=current.token or
                    pin is None or pin['region']!=current.points_key or
                    vector(pin['q_vis'])!=vector(ob['q_vis'])):
                    return self.reject(ticket,'active_target_changed')
                if max(abs(a-b) for a,b in zip(q,pin['q_vis']))<=.01:
                    return self.reject(ticket,'candidate_not_distinct')
                # Preserve id, generation, points, parent pin, stack and edge identity.
                import numpy as np
                ob['q_vis']=np.array(q);ob['q_zero']=np.array(zero)
                node._dependency_pins[current.active_id]=dict(region=current.points_key,q_vis=np.array(q))
                node._progressive_shared_cache_key=None;node._progressive_shared_cache=None
                self.ledger.attempted_poses.append(q)
                self.pending=None;self.state='PLANNING'
                self.event('steering_replaced_requires_existing_gates')
            # Publishes q_vis to an inert collector, never an executable trajectory.
            node._publish_schedule()
        return 'steering_replaced_requires_existing_gates'

    def outcome(self,kind):
        if kind not in {'unsafe','timeout','aborted','partial_offline_check','safe_candidate','motion'}:
            raise ValueError('unsupported event')
        if kind in {'unsafe','timeout','aborted'}:self.state='HOLD'
        elif kind in {'safe_candidate','motion'}:self.state='WAIT_SEEN'
        # Completion and guard release are exclusively owned by existing perception/gates.
        return self.event(kind+'_does_not_complete_obligation')

    def inherit_progress(self,attempts,segments):
        if attempts<0 or segments<0:raise ValueError('invalid progress')
        self.ledger.attempts=max(self.ledger.attempts,attempts)
        self.ledger.segments=max(self.ledger.segments,segments)
        if self.ledger.attempts>=self.max_attempts or self.ledger.segments>=self.max_segments:
            self.state='EXHAUSTED'
        return self.event('budget_merged_without_reset')
