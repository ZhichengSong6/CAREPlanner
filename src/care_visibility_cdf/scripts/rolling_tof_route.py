"""Propose a safe first move and reconsider the following sweep after new ToF returns."""
from collections import deque
import time
from types import SimpleNamespace

import numpy as np
import rospy
import torch

from live_tof_route import TaskRouteComparison as LiveToFRoute


class TaskRouteComparison(LiveToFRoute):
    def __init__(self, node):
        self.recent_targets = deque(maxlen=4)
        self.searches = 0
        super().__init__(node)

    def choose(self, *args, **kwargs):
        # The timer proposes task segments. The ordinary R1 waypoint fallback
        # can continue when a segment has not yet been handed to SCP.
        return None

    def publish_ready(self, route):
        self.recent_targets.append(route['target'].copy())
        super().publish_ready(route)

    def _entry(self, runtime, packets, start, stopped, goal, accel):
        from ray_prediction import predict

        began = time.monotonic()
        goal_pose = self.goal_pose
        if goal_pose is None:
            return None
        previous_targets = tuple(self.recent_targets)
        self.searches += 1
        side = np.array([.25, -.07, -.20, .12, 0., 0., 0.])
        elbow = np.array([0., .18, -.23, -.24, .18, 0., 0.])
        turn = 1 if self.searches % 2 else -1

        def toward(q):
            delta = goal - q
            return delta * min(1., .38 / max(.38, np.max(abs(delta))))

        def clip(q):
            return np.clip(q, runtime.q_min_np, runtime.q_max_np)

        def cost(q):
            return self._pose_cost(runtime, q, *goal_pose)

        direct = toward(stopped)
        warm = previous_targets[-1] - stopped if previous_targets else np.zeros(7)
        third = warm if .075 <= np.max(abs(warm)) <= .30 else -turn * (side + elbow)
        directions = np.asarray([turn * side, turn * elbow, third])
        ends = clip(stopped + np.vstack([direct, .28 * direct + directions]))
        first = self._sweeps(start, stopped, ends, accel, runtime)
        if first is None:
            return None
        grid, paths, sweeps, unknown, occupied = first
        prefix = [self._blocked_prefix(p, s, unknown, occupied) for p, s in zip(paths, sweeps)]
        direct_end, _, _ = prefix[0]
        direct_cost = cost(direct_end if direct_end is not None else stopped)
        if direct_end is not None and np.allclose(direct_end, ends[0]) and cost(stopped) - direct_cost >= .5:
            return None

        ready = [(i, q) for i, (q, _, _) in enumerate(prefix[1:]) if q is not None
                 and np.max(abs(q - stopped)) >= .055]
        if not ready:
            return None
        via = np.array([q for _, q in ready])

        def outgoing(poses):
            ends = clip(poses + [toward(q) + .20 * (q - stopped) for q in poses])
            edges = self._edges(runtime, start, poses, ends, accel)
            if edges is None:
                return None
            paths, sweeps, unknown, occupied = edges
            return [self._blocked_prefix(p, s, unknown, occupied) for p, s in zip(paths, sweeps)]

        following = outgoing(via)
        if following is None:
            return None
        shifted = via.copy()
        guides = {}
        for j, (last, _, missing) in enumerate(following):
            if missing is None:
                continue
            points = grid[missing[:3]]
            q = via[j]
            _, order = runtime.rank_sensors(
                torch.as_tensor(points, dtype=torch.float32, device=runtime.device),
                torch.as_tensor(q[None], dtype=torch.float32, device=runtime.device))
            sensor = int(order[0])
            _, gradient = runtime.branch_value_and_grad_numpy(points, q, sensor)
            norm = np.max(abs(gradient))
            if norm > 1e-8:
                shifted[j] = clip(q + .095 * gradient / norm)
                guides[j] = (sensor, points)

        guided_checks = self._sweeps(start, stopped, shifted, accel, runtime) if guides else None
        guided_moves = set()
        if guided_checks is not None:
            _, paths, sweeps, unknown, occupied = guided_checks
            for j in guides:
                q, _, _ = self._blocked_prefix(paths[j], sweeps[j], unknown, occupied)
                if q is not None and np.max(abs(q - stopped)) >= .055 and np.max(abs(q - via[j])) >= .015:
                    via[j] = q
                    guided_moves.add(j)
            if guided_moves:
                revised = outgoing(via)
                if revised is None:
                    return None
                following = revised

        now = packets[-1].header.stamp.to_sec()
        with self.measurements_lock:
            surfaces = {sensor: xyz for sensor, (stamp, xyz) in self.surfaces.items()
                        if abs(now - stamp) <= 1.}
        occupied = np.vstack(list(surfaces.values())) if surfaces else np.empty((0, 3))
        optics = SimpleNamespace(primitives=runtime.primitives, chains=runtime.collision_chains,
                                 sensors=runtime.sensor_chains)
        deadline = time.monotonic() + .05
        candidates = []
        for j, (q, (last, waiting, missing)) in enumerate(zip(via, following)):
            if any(np.max(abs(q - previous)) < .045 for previous in previous_targets):
                continue
            if last is None and missing is None:
                continue
            # Compare progress on the outgoing sweep. Return prediction ranks
            # proposals and never contributes free cells to the map.
            frontier = grid[missing[:3]] if missing is not None else None
            finite = 0
            sensor = guides[j][0] if j in guides else None
            if sensor is not None and frontier is not None and len(occupied) and time.monotonic() < deadline:
                if self._optics(runtime, q, sensor, frontier)[2] >= -.025:
                    shot = predict(optics, q, frontier, occupied, sensor_ids=[sensor],
                                   measured_surfaces=surfaces, deadline=deadline, grid_indices=self._grid_indices)
                    finite = len(shot['finite_ids']) if shot else 0
            reachable_cost = cost(last if last is not None else q)
            future_cost = cost(waiting) if waiting is not None else reachable_cost
            score = .75 * reachable_cost + .25 * future_cost - .55 * finite
            if missing is None and reachable_cost >= direct_cost:
                continue
            candidates.append((score, j, finite, reachable_cost, frontier))

        self.node._trace_observation('task_route_map_search', alternatives=len(ready),
            candidates=len(candidates), gradient_adjustments=len(guided_moves),
            wall_ms=1000. * (time.monotonic() - began))
        if not candidates:
            return None
        _, j, finite, reachable_cost, frontier = min(candidates, key=lambda item: item[0])
        return dict(ready=True, start=start, target=via[j], goal=goal,
                    cost=reachable_cost, direct_cost=direct_cost,
                    identity=dict(next_unknown=None if frontier is None else frontier.tolist(),
                                  r1_sensor=guides.get(j, (None,))[0], r1_adjusted=j in guided_moves,
                                  predicted_finite_cells=finite))
