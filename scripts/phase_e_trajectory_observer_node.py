#!/usr/bin/env python3
"""Private diagnostic candidate->whole-prefix->observation-scheduler adapter.

No lease/command publishers. Proposed start windows and command-model timing
are observation planning assumptions, not qualified execution promises.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from urllib.parse import urlparse

sys.path.append('/usr/lib/python3/dist-packages')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/care_visibility_cdf/scripts'))
from stopping_trajectory_prefix import build_trajectory_prefix, bind_report, trajectory_sha256
from stopping_observation_demands import query_prefix_observation_demands
from stopping_observation_context import ObservationContext, ContextObservationDemandQueue, context_key
from phase_e_stopping_contract import bounded_update_travel
from task_observation_binding import TaskContextReceiver, validate_binding


class TrajectoryPrefixObserver:
    """One latest candidate and one worker; context results only request sensing."""
    def __init__(self, *, geometry, query, now_ns, joint_names, config, publish, audit):
        self.geometry, self.query, self.now_ns = geometry, query, now_ns
        self.joint_names, self.config = list(joint_names), copy.deepcopy(config)
        self.publish, self.audit = publish, audit
        self.lock, self.worker = threading.RLock(), threading.Lock()
        self.latest, self.state, self.leases = None, None, {}
        self.context_enabled = bool(config.get('enable_observation_context', False))
        self.require_task_binding = bool(config.get("require_task_binding", False))
        if self.require_task_binding and not self.context_enabled:
            raise ValueError("Task binding requires observation context")
        self.task_receiver = TaskContextReceiver()
        self.observation_context = self.task_receiver.context if self.require_task_binding else ObservationContext()
        self.demands = ContextObservationDemandQueue()
        self.lease_overflow_until_ns = 0
        for k in ('horizon_s', 'admission_wait_s', 'input_max_age_s', 'wall_budget_s',
                  'lease_s', 'update_gap_s', 'max_braking_step_s'):
            if not math.isfinite(config[k]) or config[k] <= 0:
                raise ValueError('invalid_observer_' + k)
        if (len(config['target_abs_velocity']) != 7 or
                any(not math.isfinite(v) or v < 0 for v in config['target_abs_velocity'])):
            raise ValueError('invalid_observer_target_range')

    def offer_context(self, event):
        with self.lock:
            before = self.observation_context.snapshot(self.now_ns())
            after = (self.task_receiver.update(event, self.now_ns()) if self.require_task_binding
                     else self.observation_context.update(event, self.now_ns()))
            if before is None or after is None or context_key(before) != context_key(after):
                self.demands.reset()
            if self.require_task_binding and after is not None:
                pending = self.task_receiver.candidate(self.now_ns())
                if pending is not None:
                    self.offer(pending[0], pending[1])

    def offer(self, trajectory, task_context=""):
        digest = trajectory_sha256(trajectory)
        with self.lock:
            received_ns = self.now_ns()
            if self.require_task_binding:
                self.task_receiver.offer_candidate(trajectory, task_context, received_ns)
                pending = self.task_receiver.candidate(received_ns)
                if pending is None:
                    return
                trajectory, task_context, received_ns = pending
            context = self.observation_context.snapshot(self.now_ns()) if self.context_enabled else None
            owner = context_key(context) if context else None
            if self.require_task_binding:
                validate_binding(task_context, context, trajectory.header.frame_id)
            if self.context_enabled and (context is None or trajectory.header.stamp.to_nsec() < context['started_ns']):
                return
            # A republished identical candidate cannot slide its start window.
            if self.latest is None or digest != self.latest['sha256'] or self.latest.get('context_key') != owner:
                self.latest = dict(trajectory=copy.deepcopy(trajectory), sha256=digest,
                                   received_ns=received_ns, context_key=owner, task_context=task_context)

    def offer_state(self, state):
        with self.lock:
            self.state = copy.deepcopy(state)

    def offer_lease(self, lease):
        with self.lock:
            now = self.now_ns()
            self.leases = {k: v for k, v in self.leases.items() if v['valid_until_ns'] > now}
            key = (lease['controller_session'], lease['execution_stamp_ns'], lease['lease_sequence'])
            if key in self.leases and self.leases[key] != lease:
                self.lease_overflow_until_ns = max(self.lease_overflow_until_ns,
                    lease['valid_until_ns'], self.leases[key]['valid_until_ns'])
                raise ValueError('conflicting_observer_lease_identity')
            if key not in self.leases and len(self.leases) >= 512:
                self.lease_overflow_until_ns = max(self.lease_overflow_until_ns, lease['valid_until_ns'])
                raise ValueError('observer_outstanding_lease_capacity')
            self.leases[key] = copy.deepcopy(lease)

    def _context(self, now):
        with self.lock:
            s = copy.deepcopy(self.state)
            grants = copy.deepcopy(list(self.leases.values()))
            if now < self.lease_overflow_until_ns:
                raise ValueError('observer_outstanding_lease_capacity')
        if not s or not 0 <= now-s['t_ns'] <= 50000000 or s['clock_fault']:
            raise ValueError('observer_stale_or_invalid_controller_state')
        for field in ('applied', 'velocity_limits', 'acceleration_limits'):
            if len(s[field]) != 7 or any(not math.isfinite(v) for v in s[field]):
                raise ValueError('observer_invalid_controller_limits')
        if any(v <= 0 for v in s['velocity_limits']+s['acceleration_limits']):
            raise ValueError('observer_invalid_controller_limits')
        if not math.isfinite(s.get('max_braking_step_s', math.nan)) or s['max_braking_step_s'] <= 0:
            raise ValueError('observer_invalid_controller_braking_step')
        cap = list(self.config['target_abs_velocity'])
        hold_s = self.config['lease_s']
        # Include the applied receipt and EVERY unexpired active/pending grant.
        rights = grants + [s]
        for g in rights:
            if g['controller_session'] == s['controller_session'] and g['valid_until_ns'] > now:
                hold_s = max(hold_s, (g['valid_until_ns']-now)*1e-9)
                other = g['target_abs_velocity'] if g['target_range_enforced'] else s['velocity_limits']
                if len(other) != 7 or any(not math.isfinite(v) or v < 0 for v in other):
                    raise ValueError('observer_invalid_outstanding_range')
                cap = [max(a, b) for a, b in zip(cap, other)]
        if any(c > v for c, v in zip(cap, s['velocity_limits'])):
            raise ValueError('observer_range_exceeds_controller_limit')
        s['planning_hold_s'] = hold_s
        return s, cap

    def tick(self):
        if not self.worker.acquire(blocking=False):
            return
        started = time.monotonic()
        row = dict(kind='prefix_attempt', wall_start_s=started, runtime_admission=False, status='REJECT')
        try:
            with self.lock:
                entry = copy.deepcopy(self.latest)
                context = self.observation_context.snapshot(self.now_ns()) if self.context_enabled else None
                retained = self.demands.ordered() if context and self.demands.owner == context_key(context) else []
                map_scope = copy.deepcopy(self.demands.map_scope) if retained else None
            if entry is None:
                return
            now = self.now_ns()
            if self.context_enabled and (context is None or entry.get('context_key') != context_key(context)):
                raise ValueError('observer_context_inactive_or_replaced')
            row.update(trajectory_sha256=entry['sha256'], received_ns=entry['received_ns'])
            c = self.config
            refresh_only = bool(context and retained and now-entry['received_ns'] > round(c['input_max_age_s']*1e9))
            if not refresh_only and not 0 <= now-entry['received_ns'] <= round(c['input_max_age_s']*1e9):
                raise ValueError('observer_candidate_stale_or_clock_reversed')
            window = [entry['received_ns'], entry['received_ns']+round(c['admission_wait_s']*1e9)]
            if not refresh_only and now > window[1]:
                raise ValueError('observer_candidate_start_window_expired')
            state, cap = self._context(now)
            def budget(applied, target):
                return bounded_update_travel(applied, state['acceleration_limits'], state['velocity_limits'],
                    state['planning_hold_s'], c['update_gap_s'], state['max_braking_step_s'], target)
            construction = build_trajectory_prefix(trajectory=entry['trajectory'], joint_names=self.joint_names,
                start_window_ns=window, horizon_s=c['horizon_s'], target_abs_velocity=cap,
                applied_velocity=state['applied'], stopping_budget=budget)
            construction['conditional_command_model'] = dict(hold_s=state['planning_hold_s'],
                update_gap_s=c['update_gap_s'], max_braking_step_s=state['max_braking_step_s'],
                qualified=False)
            row['construction'] = construction
            report = query_prefix_observation_demands(segments=[] if refresh_only else construction['segments'], geometry=self.geometry,
                query=self.query, now_ns=self.now_ns, path_id=construction['path_id'],
                trajectory_stamp_ns=construction['trajectory_stamp_ns'], observation_timing=c['observation_timing'],
                wall_budget_s=c['wall_budget_s'], retained_cells=retained, retained_map_scope=map_scope)
            report = bind_report(report, construction)
            end = self.now_ns()
            state_end, cap_end = self._context(end)
            if (state_end['controller_session'] != state['controller_session'] or
                    state_end['velocity_limits'] != state['velocity_limits'] or
                    state_end['acceleration_limits'] != state['acceleration_limits'] or
                    state_end['max_braking_step_s'] != state['max_braking_step_s'] or
                    state_end['planning_hold_s'] > state['planning_hold_s']+1e-12 or
                    any(a > b for a, b in zip(cap_end, cap)) or
                    any(abs(a) > b+1e-12 for a, b in zip(state_end['applied'], construction['worst_applied_velocity']))):
                raise ValueError('observer_controller_context_changed')
            with self.lock:
                candidate_current = (self.latest is not None and self.latest['sha256'] == entry['sha256'] and
                                     self.latest['received_ns'] == entry['received_ns'])
                current_context = self.observation_context.snapshot(end) if self.context_enabled else None
                if self.context_enabled:
                    if current_context is None or context_key(current_context) != context_key(context):
                        raise ValueError('observer_context_changed_during_query')
                    if self.require_task_binding:
                        validate_binding(entry["task_context"], current_context, entry["trajectory"].header.frame_id)
                        report["task_context"] = entry["task_context"]
                    report.update(observation_context=context, evidence_kind='context_observation_demand',
                        execution_eligible=False, candidate_current_at_publish=candidate_current,
                        retained_refresh_only=refresh_only)
                elif not candidate_current:
                    raise ValueError('observer_trajectory_replaced_during_query')
                if not refresh_only and (not 0 <= end-entry['received_ns'] <= round(c['input_max_age_s']*1e9) or end > window[1]):
                    raise ValueError('observer_candidate_aged_during_query')
                if self.context_enabled:
                    self.demands.update(report, current_context, end*1e-9)
                self.publish(report)
            row.update(status='PUBLISHED', report=report)
        except Exception as exc:
            row['reason'] = str(exc)
            if self.context_enabled and str(exc) in (
                    'observation_context_map_changed', 'task_observation_map_scope_changed'):
                with self.lock:
                    current = self.observation_context.snapshot(self.now_ns())
                    if current is not None and context is not None and context_key(current) == context_key(context):
                        self.observation_context.invalidate(str(exc))
                        self.demands.reset()
                        self.publish(dict(kind='observation_context_invalidated', observation_context=current,
                            reason='observation_context_map_changed', runtime_admission=False))
        finally:
            row.update(wall_end_s=time.monotonic(), wall_elapsed_s=time.monotonic()-started)
            self.audit(row)
            self.worker.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic-private-master', action='store_true', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    uri = urlparse(os.environ.get('ROS_MASTER_URI', ''))
    if uri.hostname not in ('127.0.0.1', 'localhost') or not uri.port or not 12000 <= uri.port <= 12998:
        parser.error('An explicit isolated localhost diagnostic master is required')
    config = json.loads(args.config.read_text())
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    import rospy
    from trajectory_msgs.msg import JointTrajectory
    from std_msgs.msg import String
    from geometry_msgs.msg import Point
    from care_velocity_controller.msg import CommandStatus, ExecutionLease
    from care_confidence_map.srv import QueryConfidenceSnapshot, QueryConfidenceSnapshotRequest
    from phase_e_stop_shadow_node import BoundedQuery, PersistentServiceTransport, snapshot_dict
    from phase_e_local_stop_geometry import LocalPrimitiveEnclosure
    from phase_e_leased_stop_evidence import status_record, JOINTS
    rospy.init_node('trajectory_stopping_observer', disable_signals=True)
    pub = rospy.Publisher(config['output_topic'], String, queue_size=1)
    def encode(points, horizon, expected):
        req = QueryConfidenceSnapshotRequest(points=[Point(*p) for p in points], horizon_s=horizon, minimum_confidence=.8)
        if expected:
            req.expected_instance, req.expected_epoch, req.expected_revision = expected
        return req
    query = BoundedQuery(PersistentServiceTransport(lambda: rospy.ServiceProxy(config['query_service'],
        QueryConfidenceSnapshot, persistent=True), encode, snapshot_dict), .3)
    with (out/'attempts.jsonl').open('x') as stream:
        audit_lock = threading.Lock()
        def audit(row):
            with audit_lock:
                stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
        observer = TrajectoryPrefixObserver(geometry=LocalPrimitiveEnclosure(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
            query=query, now_ns=lambda: rospy.Time.now().to_nsec(), joint_names=JOINTS, config=config,
            publish=lambda r: pub.publish(String(data=json.dumps(r, allow_nan=False))), audit=audit)
        def candidate(msg, task_context=""):
            try:
                observer.offer(msg, task_context)
            except Exception as exc:
                audit(dict(kind='input_reject', reason=str(exc), runtime_admission=False))
        def grant(msg):
            try:
                observer.offer_lease(dict(controller_session=msg.controller_session,
                    execution_stamp_ns=msg.execution_stamp_ns, lease_sequence=msg.lease_sequence,
                    valid_until_ns=msg.valid_until.to_nsec(), target_range_enforced=msg.target_range_enforced,
                    target_abs_velocity=list(msg.target_abs_velocity)))
            except Exception as exc:
                audit(dict(kind='lease_reject', reason=str(exc), runtime_admission=False))
        if observer.require_task_binding:
            from egocentric_arm_planner.msg import CandidateTransfer
            rospy.Subscriber(config['trajectory_topic']+'/planned', CandidateTransfer,
                lambda m: candidate(m.trajectory, m.task_context) if m.action=='planned' else None, queue_size=1)
        else:
            rospy.Subscriber(config['trajectory_topic'], JointTrajectory, candidate, queue_size=1)
        if observer.context_enabled:
            def context_event(msg):
                try:
                    observer.offer_context(json.loads(msg.data))
                except Exception as exc:
                    audit(dict(kind='context_reject', reason=str(exc), runtime_admission=False))
            rospy.Subscriber(config['context_topic'], String, context_event, queue_size=20)
        rospy.Subscriber(config['state_topic'], CommandStatus, lambda m: observer.offer_state(status_record(m)), queue_size=1)
        rospy.Subscriber(config['lease_topic'], ExecutionLease, grant, queue_size=128)
        # A wall loop remains responsive when simulated ROS time is paused.
        while not rospy.is_shutdown():
            started = time.monotonic()
            observer.tick()
            # Processing is part of the 20ms poll period; adding another 20ms
            # after a ~40ms query would itself miss a 20Hz candidate cadence.
            time.sleep(max(.001, .02-(time.monotonic()-started)))


if __name__ == '__main__':
    main()
