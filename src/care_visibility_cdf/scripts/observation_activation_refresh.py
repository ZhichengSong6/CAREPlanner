"""One optional repricing of an existing candidate pool before activation.

This changes proposal preference only. It cannot complete an observation,
reset a retry/deadline ledger, or authorize motion. Projection is deliberately
absent: both alternatives must already have passed the original generator.
"""
import copy
from contextlib import nullcontext
import math
import threading
import time

import numpy as np

from observation_identity import observation_region_token, observation_token


def accepted_candidate_pool(runtime, hybrid):
    """Original selection first, at most two distinct accepted sensor heads."""
    if runtime is None or not hybrid.get('accepted'):
        return []
    pool = []
    def add(sid, q, mask, frame, rank):
        sid = int(sid)
        if len(pool) >= 2 or any(row['sensor_id'] == sid for row in pool):
            return
        q = np.asarray(q, dtype=float).reshape(7)
        mask = np.asarray(mask, dtype=float).reshape(7)
        if (not 0 <= sid < 8 or not np.all(np.isfinite(q)) or
                not np.array_equal(mask, runtime.sensor_masks_np[sid])):
            raise ValueError('invalid accepted observation candidate')
        pool.append(dict(sensor_id=sid, q=q.tolist(), mask=mask.tolist(),
            sensor_frame=str(frame), rank=int(rank)))
    add(hybrid['selected_sensor_id'], hybrid['selected_q_vis'],
        hybrid['selected_joint_mask'], hybrid['selected_sensor_frame'],
        hybrid.get('selected_rank', -1))
    for other in hybrid.get('native_alternatives', []):
        if other.get('accepted'):
            add(other['selected_sensor_id'], other['selected_q_vis'],
                other['selected_joint_mask'], other['selected_sensor_frame'],
                other.get('selected_rank', -1))
    for attempt in hybrid.get('attempts', []):
        if attempt.get('geometry', {}).get('accepted'):
            sid = int(attempt['sensor_id'])
            add(sid, attempt['q_candidate'], runtime.sensor_masks_np[sid],
                attempt['sensor_frame'], attempt.get('rank', -1))
    return pool


def reprice_candidates(runtime, points, q, pool, context, budget_s=.45,
                       clock=time.monotonic):
    """Evaluate both candidates at one measured state, using existing weights."""
    from observation_corridor_cost import approach_visibility_order
    from observation_task_rejoin import candidate_cost, improves_candidate
    from validate_visibility_oracle import DEFAULT_JOINT_NAMES, fk_transform

    started = clock()
    diagnostic = dict(status='original_fallback', selected_index=0, candidates=[],
        projection_calls=0, motion_authorized=False)
    if len(pool) != 2 or context is None:
        diagnostic['reason'] = 'no_bounded_pair_or_task'
        return diagnostic
    q = np.asarray(q, dtype=float).reshape(7)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not len(points) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(points)):
        diagnostic['reason'] = 'invalid_activation_input'
        return diagnostic
    context = dict(context, q_start=q.tolist())
    # A isolates activation state from C's map-corridor change. Both sides use
    # identical available information; no missing-map == zero-risk advantage.
    context.pop('corridor_evaluator', None)
    def pose(value):
        return fk_transform(runtime.task_ee_chain, dict(zip(DEFAULT_JOINT_NAMES, value)))
    task_pose = pose(context['q_task'])
    for candidate in pool:
        if clock()-started >= budget_s:
            diagnostic['reason'] = 'activation_budget'
            return diagnostic
        mask = np.asarray(candidate['mask'], dtype=float).reshape(7)
        # A sensor does not command its free joints. Use the captured pose for
        # body LOS and task-return scoring, never a stale generation seed.
        goal = np.where(mask > .5, np.asarray(candidate['q']), q)
        geometry = runtime._candidate_geometry(points, goal, candidate['sensor_id'])
        cost = candidate_cost(goal, context, pose(goal), task_pose)
        order = approach_visibility_order(runtime, points, q, goal, candidate['sensor_id'])
        cost['approach_visibility_order'] = order
        if order.get('status') == 'valid':
            cost['approach_requires_detour'] = order['requires_detour']
        diagnostic['candidates'].append(dict(candidate=copy.deepcopy(candidate),
            evaluated_goal=goal.tolist(), geometry=geometry, cost=cost))
    diagnostic['elapsed_s'] = clock()-started
    if diagnostic['elapsed_s'] >= budget_s:
        diagnostic['reason'] = 'activation_budget'
        return diagnostic
    first, other = diagnostic['candidates']
    if other['geometry'].get('accepted') and (
            not first['geometry'].get('accepted') or
            improves_candidate(other['cost'], first['cost'], 0.)):
        diagnostic.update(status='alternative_preferred', selected_index=1,
            reason='activation_geometry' if not first['geometry'].get('accepted')
            else 'activation_task_return')
    else:
        diagnostic['reason'] = 'original_still_preferred'
    return diagnostic


class ActivationRefresh:
    """Serialize optional work; compare ownership again before publishing it."""
    def __init__(self, node, mode='off', budget_s=.45):
        if mode not in ('off', 'shadow', 'select'):
            raise ValueError('activation refresh mode must be off, shadow, or select')
        self.node = node
        self.mode = mode
        self.budget_s = budget_s
        self.worker = threading.Lock()
        self.task = None
        self.claimed = set()

    def _snapshot_locked(self):
        node = self.node
        if getattr(node, '_require_task_binding', False):
            return None  # The separate task-bound root protocol owns selection.
        by_id = {int(ob['id']): ob for ob in node._obligations}
        if not by_id:
            return None
        stack = [oid for oid in node._repair_stack if oid in by_id]
        oid = stack[-1] if stack else min(by_id)
        ob = by_id[oid]
        # A parent restored from a dependency and a recovery proposal have
        # their own identity contracts. Activation repricing cannot replace them.
        co_plan = ({int(v) for v in node._path_co_plan_ids} | {oid}
                   if node.progressive_shared_path_association_only else set(by_id))
        if (oid in getattr(node, '_dependency_pins', {}) or
                ob.get('final_recovery_individual_region') is not None or
                len(co_plan.intersection(by_id)) > 1):
            return None
        replacement = getattr(node, '_candidate_replacement', None)
        # Caller owns replacement.lock before the publication locks. Inspect
        # without expiring/releasing any owner or changing its retry ledger.
        if replacement is not None and (replacement.qp_owner_lock is not None or
                                        replacement.pending is not None):
            return None
        region = observation_region_token(ob)
        published = getattr(node, '_trace_published_target', None)
        if published is not None and published.get('region_token') == region:
            return None  # Already active: never reset its motion/progress owner.
        pool = ob.get('activation_candidate_pool', [])
        if len(pool) != 2 or pool[0]['sensor_id'] != ob.get('per_sensor_selected_sensor_id'):
            return None
        if not np.array_equal(pool[0]['q'], ob['q_vis']):
            return None  # A replacement/recovery has already changed the target.
        trajectory = node._trajectory
        if trajectory is None:
            return None
        task = int(trajectory.header.stamp.to_nsec())
        owner = (task, region, observation_token(ob),
                 published.get('observation_token') if published else None,
                 tuple(stack))
        return dict(task=task, owner=owner, region=region,
            ob=copy.deepcopy(ob), pool=copy.deepcopy(pool))

    def before_publish(self):
        if self.mode == 'off':
            return
        node = self.node
        # Recovery callbacks can republish while already holding these locks.
        # They retain their original target and must not run optional FK/R1.
        owned = getattr(node._schedule_publish_lock, '_is_owned', None)
        if ((owned is not None and owned()) or
                (owned is None and node._schedule_publish_lock.locked()) or
                node._obligation_lock.locked()):
            return
        if not self.worker.acquire(False):
            return
        started = time.monotonic()
        snapshot = None
        claimed = False
        diagnostic = dict(status='original_fallback', motion_authorized=False)
        replacement = getattr(node, '_candidate_replacement', None)
        replacement_lock = replacement.lock if replacement is not None else nullcontext()
        try:
            # These locks protect only copies and comparison. FK/R1 and optional
            # task context work execute after releasing all publication locks.
            with replacement_lock, node._schedule_publish_lock, node._obligation_lock, node._lock:
                snapshot = self._snapshot_locked()
            if snapshot is None:
                return
            if self.task != snapshot['task']:
                self.task = snapshot['task']
                self.claimed.clear()
            if snapshot['region'] in self.claimed:
                return
            self.claimed.add(snapshot['region'])
            claimed = True
            measured, stamp = getattr(node, '_trace_measured', (None, math.nan))
            now = node._activation_ros_now()
            if measured is None or not 0. <= now-stamp <= .2:
                diagnostic['reason'] = 'stale_measurement'
                return
            q = np.asarray(measured, dtype=float).reshape(7).copy()
            context = node._task_rejoin_context(q)
            remaining = self.budget_s-(time.monotonic()-started)
            if remaining <= 0.:
                diagnostic['reason'] = 'activation_budget'
                return
            diagnostic = reprice_candidates(node._per_sensor_runtime,
                snapshot['ob']['points'], q, snapshot['pool'], context, remaining)
            diagnostic['measured_q'] = q.tolist()
            diagnostic['measured_stamp_s'] = stamp
            if self.mode != 'select' or diagnostic['selected_index'] == 0:
                return
            selected = diagnostic['candidates'][diagnostic['selected_index']]
            goal = np.asarray(selected['evaluated_goal'], dtype=float)
            union_min = node._activation_union_min(snapshot['ob']['points'], goal)
            if not math.isfinite(union_min) or time.monotonic()-started >= self.budget_s:
                diagnostic.update(status='original_fallback', reason='activation_budget_or_field')
                return
            with replacement_lock, node._schedule_publish_lock, node._obligation_lock, node._lock:
                current = self._snapshot_locked()
                current_q, current_stamp = getattr(node, '_trace_measured', (None, math.nan))
                if (current is None or current['owner'] != snapshot['owner'] or
                        current_q is None or not 0. <= node._activation_ros_now()-current_stamp <= .2 or
                        np.max(np.abs(np.asarray(current_q)-q)) > .05):
                    diagnostic.update(status='original_fallback', reason='activation_superseded')
                    return
                ob = next(ob for ob in node._obligations if int(ob['id']) == snapshot['ob']['id'])
                candidate = selected['candidate']
                ob.update(q_vis=goal.copy(), q_vis_joint_mask=np.asarray(candidate['mask']).copy(),
                    per_sensor_selected_sensor_id=candidate['sensor_id'],
                    per_sensor_selected_sensor_frame=candidate['sensor_frame'],
                    per_sensor_selected_rank=candidate['rank'], per_sensor_hybrid_used=True,
                    final_f_min=union_min,
                    shared_solution_mode=str(ob['shared_solution_mode'])+'+activation_pool',
                    activation_refresh_source_token=snapshot['owner'][2])
                diagnostic.update(status='selected_before_activation',
                    new_observation_token=observation_token(ob))
        except Exception as exc:
            diagnostic.update(status='original_fallback', reason='activation_error', error=str(exc))
        finally:
            self.worker.release()
            if claimed:
                diagnostic.update(mode=self.mode, elapsed_s=time.monotonic()-started,
                    region_token=snapshot['region'], source_observation_token=snapshot['owner'][2])
                node._trace_observation('activation_refresh', **diagnostic)
