"""Wall-clock replacement lease; inference never holds publication/owner locks.

Produces a steering pose only. The existing fresh-state QP and execution
checks still decide whether any prefix of the motion can be committed.
"""
import time
import os
import numpy as np
from candidate_replacement_runtime import (
    CandidateReplacement, PLANNER_TARGET_REPLACEMENT_REASONS,
    high_witness_sensor_preference, sensor_mount_group_id)
from observation_identity import observation_region_token, observation_target_state


def remaining_points(points, token, region, feedback, now):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if (feedback is not None and feedback['token'] == token and
            feedback['region'] == region and 0 <= now-feedback['query_time'] <= .5 and
            np.array_equal(feedback['points'], points) and
            len(feedback['point_seen']) == len(points)):
        return points[~np.asarray(feedback['point_seen'], dtype=bool)].copy()
    return points.copy()


class ReplacementExecution(CandidateReplacement):
    def __init__(self, *args):
        super().__init__(*args)
        self.inflight = None

    def receive(self, text):
        super().receive(text)
        with self.lock:
            if self.grant is not None and self.grant.get('granted') == '1':
                try:
                    self._validate_clock(self.grant)
                    if self.qp_owner_lock is not None:
                        self.qp_owner_lock['expires_wall'] = time.monotonic() + max(
                            0., int(self.grant['deadline_wall_ns'])*1e-9-time.time())
                except (ValueError, KeyError, OverflowError):
                    pass  # process() sends a classified error for this exact request

    @staticmethod
    def _validate_clock(grant):
        if (grant.get('lease_timing') != 'wall_v2' or
                float(grant['wait_budget_s']) != 3. or
                float(grant['generation_budget_s']) != 6. or
                int(grant['deadline_wall_ns']) <= 0):
            raise ValueError('inconsistent replacement lease')

    @staticmethod
    def _expired(grant, compute_start=None):
        return (time.time() >= int(grant['deadline_wall_ns'])*1e-9 or
                (compute_start is not None and time.monotonic()-compute_start >=
                 float(grant['generation_budget_s'])))

    def process(self):
        work = None
        with self.lock:
            if self.inflight is not None:
                return
            p = self.pending
            if p is None:
                self._process_waiting_qp_trigger(); self._process_waiting_blocker()
                return
            if 'first_process_wall' not in p:
                p['first_process_wall'] = time.monotonic()
                self.log('reservation_first_processed', request_id=p['request_id'],
                         processing_delay_wall_s=p['first_process_wall']-p['offered_wall'])
            if self.grant is None and 'generation_grant' not in p:
                if time.monotonic()-p['offered_wall'] >= 3.:
                    self.pending = None
                    self._release_qp_owner_locked('reservation_timeout')
                    self.log('reservation_timeout', request_id=p['request_id'])
                return
            grant = p.get('generation_grant', self.grant)
            self.grant = None
            if grant['granted'] != '1':
                self.pending = None
                self._release_qp_owner_locked('reservation_denied')
                self.log('reservation_denied', **grant)
                return
            if p['operation'] in ('promote_owner', 'restore_ancestor'):
                self._promote_owner(p, grant)
                return
            try:
                self._validate_generation_grant(p, grant)
                self._validate_clock(grant)
            except (ValueError, KeyError, OverflowError) as exc:
                self._finish_generation(p, grant, 'generation_error', dict(error=str(exc)))
                return
            if (self._expired(grant) or time.monotonic()-p['grant_received_wall'] >=
                    float(grant['wait_budget_s'])):
                self._finish_generation(p, grant, 'generation_expired', dict(stage='seed_wait'))
                return
            with self.node._schedule_publish_lock, self.node._obligation_lock:
                current = self._current(p)
            if not current:
                self._finish_generation(p, grant, 'context_changed', dict(stage='seed_wait'))
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
            self.inflight = p['request_id']
            self.log('generation_stable_seed_ready', request_id=p['request_id'], **detail)
            work = p, grant
        # Joint/tracker/owner callbacks and schedule publication remain runnable.
        try:
            self._generate(*work)
        finally:
            with self.lock:
                self.inflight = None
                self._process_waiting_qp_trigger(); self._process_waiting_blocker()

    def _finish_generation(self, p, grant, reason, detail, new_token='none'):
        with self.lock:
            super()._finish_generation(p, grant, reason, detail, new_token)

    def _points(self, p):
        ob = next(o for o in self.node._obligations if int(o['id']) == p['active_id'])
        return remaining_points(p['points'], p['event']['observation_token'],
            observation_region_token(ob), getattr(self.node, '_guidance_feedback', None), self.now())

    def _generate(self, p, grant):
        n = self.node
        reason, new_token, detail = 'context_changed', 'none', {}
        started = time.monotonic()
        try:
            with n._schedule_publish_lock, n._obligation_lock:
                if not self._current(p):
                    return
                points = self._points(p)
                measured, stamp = getattr(n, '_trace_measured', (None, float('nan')))
                if measured is None or not 0 <= self.now()-stamp <= .2:
                    reason = 'stale_measured_state'; return
                seed = np.asarray(measured, dtype=float).copy()
            if seed.shape != (7,) or not np.all(np.isfinite(seed)):
                reason = 'invalid_measured_state'; return
            if len(points) == 0:
                reason = 'observation_already_seen'; return
            with n._lock:
                trajectory, received, _ = n._preferred_trajectory_locked()
            if trajectory is None:
                reason = 'no_trajectory'; return
            with self.lock:
                excluded = sorted(self.families[p['lineage']])
            override = dict(excluded=excluded,
                tried_mount_groups=sorted({sensor_mount_group_id(s) for s in excluded}),
                previous_q=p['previous_q'].tolist(), score_gap_override=.75,
                minimum_conservative_g=float(getattr(n, 'per_sensor_replacement_min_conservative_g', 0.)))
            preferred = high_witness_sensor_preference(points,
                getattr(n, 'per_sensor_high_witness_priority_enabled', False),
                getattr(n, 'per_sensor_high_witness_z_min', .85))
            if preferred:
                override.update(preferred_sensor_ids=preferred, ee_sensor_priority_first=True,
                                preference_reason='repeated_exact_vbc_high_witness')
            with n._progressive_shared_lock:
                if self._expired(grant):
                    reason = 'generation_expired'; return
                n._seed_override, n._candidate_replacement_override = seed, override
                try:
                    compute_start = time.monotonic()
                    override['compute_deadline_monotonic_s'] = min(compute_start+6.,
                        compute_start+int(grant['deadline_wall_ns'])*1e-9-time.time())
                    result = n._generate_active_set_waypoint(points, trajectory, .1, received)
                    compute_end = time.monotonic()
                finally:
                    n._seed_override = n._candidate_replacement_override = None
            hybrid = result.get('per_sensor_hybrid', {})
            detail = dict(hybrid=hybrid, points=points.tolist(), full_points=p['points'].tolist(),
                measured_seed=seed.tolist(), measured_seed_stamp=stamp,
                model_compute_wall_s=compute_end-compute_start, generation_started_wall=started,
                generation_end_ros_s=self.now(), clock_policy='wall_v2')
            # A timed-out result is neither a geometric rejection nor a tried head.
            if self._expired(grant, compute_start):
                reason = 'generation_expired'; return
            # Match offer()/publication lock order; never schedule-lock -> owner-lock.
            with self.lock, n._schedule_publish_lock:
                with n._obligation_lock:
                    if not self._current(p):
                        return
                    current_points = self._points(p)
                if not len(current_points):
                    reason = 'observation_already_seen'; return
                if hybrid.get('accepted') is not True:
                    with self.lock:
                        self.families[p['lineage']].update(int(a['sensor_id']) for a in hybrid.get('attempts', []))
                    reason = 'no_geometric_candidate'; return
                q = np.asarray(hybrid['selected_q_vis'], dtype=float)
                if q.shape != (7,) or not np.all(np.isfinite(q)):
                    reason = 'invalid_candidate'; return
                live, live_stamp = getattr(n, '_trace_measured', (None, float('nan')))
                live = None if live is None else np.asarray(live, dtype=float)
                live_age = self.now()-live_stamp
                if (live is None or live.shape != (7,) or not np.all(np.isfinite(live)) or
                        not np.isfinite(live_age) or live_age < 0.):
                    reason = 'stale_measured_state'; return
                # q_vis is an absolute steering pose. Its geometry is checked
                # below without live q; native QP reconstructs and audits motion
                # from its own fresh measured state. A delayed diagnostic joint
                # callback must not discard an otherwise valid new viewpoint.
                if live_age > .2 and os.environ.get('CARE_OBSERVATION_ACTION_PROGRESS') != 'true':
                    reason = 'stale_measured_state'; return
                detail.update(application_measured_stamp=live_stamp, application_measured_q=live.tolist(),
                              application_measured_age_s=live_age,
                              application_requires_native_fresh_start=True,
                              seed_delta_inf=float(np.max(np.abs(live-seed))))
                # Recheck the proposal against the current unseen set / current
                # occupied evidence. Its path is rebuilt from live q by native QP.
                application_points = current_points
                if os.environ.get('CARE_OBSERVATION_ACTION_PROGRESS') == 'true':
                    # This absolute viewpoint was solved for this remaining
                    # subset, not for every historic point simultaneously.
                    # A stale feedback cache may expand current_points, but
                    # cannot retroactively change the proposal's objective.
                    # All other points stay in the obligation and live audits.
                    remaining_keys = {tuple(row) for row in current_points}
                    application_points = np.asarray([row for row in points
                        if tuple(row) in remaining_keys], dtype=float).reshape(-1, 3)
                    if not len(application_points):
                        reason = 'observation_already_seen'; return
                detail.update(application_points=application_points.tolist(),
                    application_full_remaining_count=len(current_points),
                    application_scope='generated_remaining_subset' if os.environ.get(
                        'CARE_OBSERVATION_ACTION_PROGRESS') == 'true' else 'fresh_full_remaining')
                geometry = n._per_sensor_runtime._candidate_geometry(application_points, q,
                    int(hybrid['selected_sensor_id']), minimum_conservative_g=override['minimum_conservative_g'])
                detail['application_geometry'] = geometry
                if not geometry['accepted']:
                    reason = 'application_geometry_changed'; return
                if self._expired(grant, compute_start):
                    reason = 'generation_expired'; return
                with self.lock:
                    poses = self.poses.setdefault(p['lineage'], [])
                    if any(np.max(np.abs(q-old)) <= .01 for old in [p['previous_q']]+poses):
                        self.families[p['lineage']].add(int(hybrid['selected_sensor_id']))
                        reason = 'candidate_not_distinct'; return
                with n._obligation_lock:
                    if not self._current(p):
                        return
                    ob = next(o for o in n._obligations if int(o['id']) == p['active_id'])
                    ob.update(q_vis=q.copy(), q_vis_joint_mask=np.asarray(
                        hybrid.get('selected_joint_mask', [1.]*7), dtype=float).reshape(7).copy(),
                        q_zero=np.asarray(result['q_zero'], dtype=float).copy(),
                        per_sensor_selected_sensor_id=int(hybrid['selected_sensor_id']),
                        per_sensor_selected_sensor_frame=hybrid['selected_sensor_frame'],
                        per_sensor_selected_rank=int(hybrid['selected_rank']))
                    n._dependency_pins[p['active_id']] = dict(region=p['region'],
                        **observation_target_state(ob), preserve_on_growth=True)
                    n._progressive_shared_cache_key = n._progressive_shared_cache = None
                n._publish_schedule()
                new_token = n._trace_published_target['observation_token']
                with self.lock:
                    poses.append(q.copy())
                    self.families[p['lineage']].add(int(hybrid['selected_sensor_id']))
                reason = 'applied_requires_existing_gates'
        except Exception as exc:
            reason = 'generation_error'; detail['error'] = str(exc)
        finally:
            self._finish_generation(p, grant, reason, detail, new_token)
