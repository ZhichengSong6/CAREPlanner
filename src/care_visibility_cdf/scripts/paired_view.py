"""Choose a reachable R1 view together with the task segment it can open."""
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import rospy
import torch
from care_confidence_map.srv import QueryConfidenceSnapshotRequest
from geometry_msgs.msg import Point
from validate_visibility_oracle import DEFAULT_JOINT_NAMES

from optical_view import TaskRouteComparison as OpticalView, intersects_cell


class TaskRouteComparison(OpticalView):
    def __init__(self, node):
        super().__init__(node)
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.work = None
        self.pending = False
        self.attempt = 0

    def choose(self, blocked, reference, identity, previous_view=None, excluded=()):
        self.pending = False
        runtime = self.node._per_sensor_runtime
        measured = getattr(self.node, '_activation_measured_state', None)
        if runtime is None or measured is None or measured[1] is None or not reference or not reference.points:
            return None
        start, velocity, stamp = measured
        if not 0. <= rospy.Time.now().to_sec() - stamp <= .15:
            return None
        points = np.asarray(blocked, dtype=float).reshape(-1, 3)
        if not len(points) or not np.all(np.isfinite(points)):
            return None
        mapping = self.node._trajectory_mapping(reference)
        goal = np.array([reference.points[-1].positions[i] for i in mapping])
        request = tuple(sorted(identity.items()))
        if self.work is not None:
            key, seed, old_goal, future = self.work
            same_context = key == request or (
                previous_view is None and key[0][0] != 'observation_token')
            if not future.done():
                self.pending = same_context
                return None
            self.work = None
            result = future.result()
            if same_context and np.max(abs(old_goal - goal)) < .10 and np.max(abs(start - seed)) < .06:
                return result
        limits = rospy.get_param(self.node.task_rejoin_limits_namespace + '/joint_acceleration_limits')
        accel = np.array([limits[joint] for joint in DEFAULT_JOINT_NAMES])
        if not np.isfinite(np.r_[start, velocity, goal, accel]).all() or np.any(accel <= 0.):
            return None
        stopped = np.clip(start + velocity * abs(velocity) / (2. * accel),
                          runtime.q_min_np, runtime.q_max_np)
        self.attempt += 1
        future = self.worker.submit(self._search, runtime, start.copy(), stopped, goal, points,
                                    accel, previous_view, tuple(excluded), self.attempt, dict(identity))
        self.work = request, start.copy(), goal, future
        self.pending = True
        return None

    def _edge_geometry(self, runtime, origin, starts, ends, accel, *, samples=9):
        grid, memberships = self._geometry(runtime)
        fractions = np.linspace(0., 1., samples)
        delta = ends - starts
        poses = starts[:, None, :] + delta[:, None, :] * fractions[None, :, None]
        speed = np.clip(delta, -.25, .25)
        stop_offset = speed * abs(speed) / (2. * accel)
        brakes = np.clip(poses + stop_offset[:, None, :] * np.sin(np.pi * fractions)[None, :, None],
                         runtime.q_min_np, runtime.q_max_np)
        flat = poses.reshape(-1, 7)
        body, *samples = memberships(np.vstack([origin, flat, brakes.reshape(-1, 7)]))
        sweeps = [np.setdiff1d(np.union1d(a, b), body, assume_unique=True)
                  for a, b in zip(samples[:len(flat)], samples[len(flat):])]
        routes = [sweeps[i:i+len(fractions)] for i in range(0, len(sweeps), len(fractions))]
        return grid, poses, routes

    def _edges(self, runtime, origin, starts, ends, accel, *, query=None, samples=9):
        grid, poses, routes = self._edge_geometry(runtime, origin, starts, ends, accel, samples=samples)
        ids = np.unique(np.concatenate([cells for route in routes for cells in route]))
        # A compact map snapshot covers every edge in this batch.
        unknown = np.zeros(len(grid), dtype=bool)
        occupied = unknown.copy()
        if len(ids):
            req = QueryConfidenceSnapshotRequest(points=[Point(*p) for p in grid[ids]],
                minimum_confidence=float(self.node.seen_threshold), horizon_s=0.)
            reply = (self.query if query is None else query)(req)
            if (reply is None or not reply.valid or reply.clock_fault or reply.frame != 'base_link'
                    or not reply.instance or not 0. <= (rospy.Time.now() - reply.snapshot_stamp).to_sec() <= .1
                    or any(len(a) != len(ids) for a in (reply.inside_map, reply.confidence_now, reply.occupancy))):
                return None
            inside = np.asarray(list(reply.inside_map), dtype=bool)
            confidence = np.asarray(reply.confidence_now, dtype=float)
            occupancy = np.asarray(reply.occupancy, dtype=float)
            occupied[ids] = ~inside | ~np.isfinite(confidence) | ~np.isfinite(occupancy) | (occupancy > .5)
            unknown[ids] = ~occupied[ids] & (confidence < req.minimum_confidence)
        return poses, routes, unknown, occupied

    @staticmethod
    def _deduplicate(poses, spacing=.065):
        unique = []
        for pose in poses:
            if all(np.max(abs(pose - old)) >= spacing for old in unique):
                unique.append(pose)
        return unique

    def _search(self, runtime, start, stopped, goal, blocked, accel, previous_view, excluded,
                attempt, identity):
        began = time.perf_counter()
        grid, _ = self._geometry(runtime)
        clip = lambda q: np.clip(q, runtime.q_min_np, runtime.q_max_np)
        delta = goal - stopped
        direct = delta * (.55 / max(.55, np.max(abs(delta))))
        side = np.array([.32, -.09, -.26, .15, 0., 0., 0.])
        elbow = np.array([0., .24, -.30, -.30, .25, 0., 0.])
        task = clip(stopped + np.array([direct, direct + side, direct - side,
                                        direct + elbow, direct - elbow]))
        rng = np.random.default_rng(attempt)
        independent = rng.uniform(-1., 1., (12, 7))
        independent *= np.array([.68, .32, .62, .45, .48, .33, .35])
        views = clip(stopped + independent)
        # Process a small batch of starts at a time. The running local controller
        # continues to use its current audited trajectory during this search.
        safe_views, entries = [], {}
        snapshot_failures = 0
        for begin in range(0, len(task) + len(views), 5):
            ends = np.vstack((task, views))[begin:begin+5]
            sweep = self._edges(runtime, start, np.tile(stopped, (len(ends), 1)), ends, accel)
            if sweep is None:
                snapshot_failures += 1
                continue
            poses, routes, unknown, occupied = sweep
            for i, (path, route) in enumerate(zip(poses, routes)):
                reachable = []
                for k, (pose, cells) in enumerate(zip(path, route)):
                    if occupied[cells].any():
                        break
                    missing = cells[unknown[cells]]
                    if len(missing):
                        if begin + i < len(task):
                            # Prefer an entry far enough past the boundary for the
                            # normal waypoint completion tolerance to matter less.
                            if k+1 < len(route) and not occupied[route[k+1]].any():
                                next_missing = route[k+1][unknown[route[k+1]]]
                                if np.array_equal(next_missing, missing):
                                    pose = path[k+1]
                            if np.max(abs(pose - stopped)) >= .10:
                                entries.setdefault(tuple(missing), pose)
                        break
                    if np.max(abs(pose - stopped)) >= .08:
                        reachable.append(pose)
                safe_views.extend(reachable[-2:][::-1])
        entry_list = [(grid[list(ids)], pose) for ids, pose in entries.items()]
        options = entry_list + [(p[None, :], None) for p in blocked[:2]]
        # R1 contributes reachable gradient directions as well as the score
        # used to rank these and independently seeded observation poses.
        gradients = self._propose(runtime, stopped, options, previous_view, excluded)
        refined = self._deduplicate(c['view'] for c in gradients)[:8]
        if refined:
            sweep = self._edges(runtime, start, np.tile(stopped, (len(refined), 1)), np.array(refined), accel)
            if sweep is None:
                snapshot_failures += 1
            else:
                _, routes, unknown, occupied = sweep
                safe_views.extend(q for q, route in zip(refined, routes)
                                  if all(not (unknown[cells] | occupied[cells]).any() for cells in route))
        safe_views = self._deduplicate(safe_views)
        if previous_view is not None:
            safe_views = [q for q in safe_views if np.max(abs(q - previous_view)) >= .08]

        pairs, exploration = self._rank_views(runtime, stopped, safe_views, options, excluded)
        choice, rejected = None, 0
        for offset in range(0, len(pairs), 4):
            batch = pairs[offset:offset+4]
            sweep = self._edges(runtime, start, np.array([p['view'] for p in batch]),
                                np.array([p['target'] for p in batch]), accel)
            if sweep is None:
                snapshot_failures += 1
                continue
            _, routes, unknown, occupied = sweep
            for proposal, route in zip(batch, routes):
                ids = np.unique(np.concatenate(route))
                remaining = grid[ids[unknown[ids]]]
                if (occupied[ids].any() or not len(remaining) or
                        not self._covers(runtime, proposal['view'], proposal['sensor'], remaining)):
                    rejected += 1
                    continue
                choice = dict(proposal, points=remaining)
                break
            if choice is not None:
                break
        if choice is None and exploration is not None:
            choice = exploration
        if choice is not None:
            choice.update(start=stopped, goal=goal)
        self.node._trace_observation('task_route_preview', vbc_identity=identity,
            direct_points=blocked.tolist(), task_entries=len(entries), reachable_views=len(safe_views),
            proposed_pairs=len(pairs), rejected_connections=rejected, snapshot_failures=snapshot_failures,
            selected_view=None if choice is None else choice['view'].tolist(),
            selected_points=None if choice is None else choice['points'].tolist(),
            selected_route_target=None if choice is None or choice['target'] is None else choice['target'].tolist(),
            selected_sensor=None if choice is None else int(choice['sensor']),
            wall_ms=1000. * (time.perf_counter() - began))
        return choice

    def _covers(self, runtime, pose, sensor, points):
        planes, margins, upper = self._optics(runtime, pose, sensor, points)
        return upper >= 0. and all(intersects_cell(planes, m) for m in margins)

    def _rank_views(self, runtime, stopped, views, options, excluded):
        if not views or len(excluded) >= 8:
            return [], None
        sensors = [sensor for sensor in range(8) if sensor not in excluded]
        q = np.array([stopped, *views])
        entries = []
        for points, target in options:
            batch = np.column_stack((np.tile(points, (len(q), 1)), np.repeat(q, len(points), axis=0)))
            with torch.no_grad():
                output = runtime.model(torch.as_tensor(batch, device=runtime.device, dtype=torch.float32))
            scores = output.detach().cpu().numpy().reshape(len(q), len(points), 8).min(axis=1)
            travel = np.max(abs(q[1:] - stopped), axis=1)
            for i, view in enumerate(views):
                for sensor in sensors:
                    planes, margins, upper = self._optics(runtime, view, sensor, points)
                    terminal = upper >= 0. and all(intersects_cell(planes, m) for m in margins)
                    if not terminal and upper <= self._optics(runtime, stopped, sensor, points)[2] + .01:
                        continue
                    score = float(scores[i+1, sensor])
                    entries.append(dict(view=view, points=points, target=target if terminal else None,
                        sensor=sensor, margin=upper, score=score, nominal_terminal=terminal,
                        rank=int(np.sum(scores[i+1] > score)) + 1,
                        quality=upper + .035 * score - .02 * travel[i]))
        entries.sort(key=lambda entry: entry['quality'], reverse=True)
        pairs = [entry for entry in entries if entry['target'] is not None][:12]
        exploration = next((p for p in entries if p['target'] is None), None)
        return pairs, exploration

    def waypoint(self, choice, trajectory, sweep, received):
        result = super().waypoint(choice, trajectory, sweep, received)
        result['per_sensor_hybrid']['proposal_kind'] = (
            'paired_task_entry' if choice['target'] is not None else 'exploration_step')
        return result
