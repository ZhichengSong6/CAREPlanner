"""Generate R1 sensing motion for the missing part of a whole-body task sweep."""
import time
from types import SimpleNamespace

import numpy as np

from rolling_tof_route import TaskRouteComparison as RollingToFRoute


class TaskRouteComparison(RollingToFRoute):
    def _entry(self, runtime, packets, start, stopped, goal, accel):
        from ray_prediction import predict

        began = time.monotonic()
        goal_pose = self.goal_pose
        if goal_pose is None:
            return None

        def clip(q):
            return np.clip(q, runtime.q_min_np, runtime.q_max_np)

        def cost(q):
            return self._pose_cost(runtime, q, *goal_pose)

        delta = goal - stopped
        direct = stopped + delta * (.38 / max(.38, np.max(abs(delta))))
        elbow = np.array([.22, .13, -.24, -.16, .18, .07, 0.])
        ends = clip(np.array([direct, direct + elbow, direct - elbow]))
        first = self._sweeps(start, stopped, ends, accel, runtime)
        if first is None:
            return None
        grid, paths, sweeps, unknown, occupied = first
        prefixes = [self._blocked_prefix(p, cells, unknown, occupied)
                    for p, cells in zip(paths, sweeps)]
        last, _, _ = prefixes[0]
        direct_cost = cost(stopped if last is None else last)
        if last is not None and np.allclose(last, ends[0]) and direct_cost < cost(stopped):
            return None

        # Propose the observation from the stopped state and the sweep's
        # actual map obligations. Separate points can use different heads.
        frontiers = [missing for _, _, missing in prefixes]
        views = []
        for route_index, (endpoint, group) in enumerate(zip(ends, frontiers)):
            if group is None:
                continue
            points = grid[group]
            spread = np.linalg.norm(points - points.mean(axis=0), axis=1)
            anchors = [spread.argmin()]
            if len(points) > 1:
                anchors.append(np.linalg.norm(points - points[anchors[0]], axis=1).argmax())
            seen = set()
            for proposal in self._propose(runtime, stopped, [(points[[k]], endpoint) for k in anchors],
                                          self.last_ready, ()):
                sensor = int(proposal['sensor'])
                if sensor in seen:
                    continue
                seen.add(sensor)
                views.append((route_index, proposal))
                if len(seen) == 2:
                    break

        proposals = [dict(view=prefix[0], sensor=None, score=0.)
                     for prefix in prefixes[1:] if prefix[0] is not None]
        drafts = views + [(None, p) for p in proposals]
        if not drafts:
            self.node._trace_observation('task_route_frontier_search', frontiers=sum(g is not None for g in frontiers),
                                         r1_views=0, supported_views=0, finite_views=0)
            return None
        checks = self._sweeps(start, stopped, np.array([p['view'] for _, p in drafts]), accel, runtime)
        if checks is None:
            return None
        _, paths, sweeps, unknown, occupied = checks
        available = []
        for (route_index, proposal), path, cells in zip(drafts, paths, sweeps):
            view, _, _ = self._blocked_prefix(path, cells, unknown, occupied)
            if view is None or np.max(abs(view - stopped)) < .045:
                continue
            if any(np.max(abs(view - last)) < .045 for last in self.recent_targets):
                continue
            if route_index is None:
                # Rejoin the original task from an already known side sweep.
                route_index = 0
            available.append((view, route_index, proposal))
        if not available:
            self.node._trace_observation('task_route_frontier_search', frontiers=sum(g is not None for g in frontiers),
                                         r1_views=len(views), supported_views=0, finite_views=0)
            return None

        poses = np.array([v for v, _, _ in available])
        destination = np.array([ends[i] for _, i, _ in available])
        outgoing = self._edges(runtime, start, poses, destination, accel)
        if outgoing is None:
            return None
        paths, sweeps, unknown, occupied = outgoing
        stamp = packets[-1].header.stamp.to_sec()
        with self.measurements_lock:
            surfaces = {sensor: xyz for sensor, (t, xyz) in self.surfaces.items()
                        if abs(stamp - t) <= 1.}
        observed = np.vstack(list(surfaces.values())) if surfaces else np.empty((0, 3))
        optics = SimpleNamespace(primitives=runtime.primitives, chains=runtime.collision_chains,
                                 sensors=runtime.sensor_chains)
        deadline = time.monotonic() + .06
        candidates = []
        for (view, route_index, proposal), path, cells in zip(available, paths, sweeps):
            last, _, missing = self._blocked_prefix(path, cells, unknown, occupied)
            current_cost = cost(view if last is None else last)
            sensor = proposal['sensor']
            finite = set()
            if missing is not None:
                points = grid[missing]
                for head in dict.fromkeys((sensor, *range(8))):
                    if head is None or time.monotonic() >= deadline:
                        continue
                    visible = np.array([self._optics(runtime, view, head, p[None])[2] >= 0. for p in points])
                    if np.any(visible):
                        shot = predict(optics, view, points[visible], observed, sensor_ids=[head],
                                       measured_surfaces=surfaces, deadline=deadline, grid_indices=self._grid_indices)
                        if shot:
                            finite.update(shot['finite_ids'])
            forecast = unknown.copy()
            if finite:
                forecast[list(finite)] = False
            future, _, _ = self._blocked_prefix(path, cells, forecast, occupied)
            future_cost = cost(view if future is None else future)
            extra = np.max(abs((view if future is None else future) - (view if last is None else last)))
            opportunity = False
            if sensor is not None and missing is not None:
                before = self._optics(runtime, stopped, sensor, points)[2]
                after = self._optics(runtime, view, sensor, points)[2]
                opportunity = after > before + .005
            gain = current_cost - future_cost
            if extra <= 0. and current_cost >= direct_cost and not opportunity:
                continue
            # An unknown suffix receives credit only when modeled returns
            # would extend its contiguous supported sweep. The real map is
            # unchanged and SCP receives only the checked preparation segment.
            priority = (extra > 0., -min(current_cost, future_cost), extra, len(finite),
                        -np.max(abs(view - stopped)))
            identity = dict(sweep=route_index, r1_sensor=sensor, r1_generated=sensor is not None,
                            next_unknown=None if missing is None else grid[missing].tolist(),
                            predicted_finite_cells=len(finite), forecast_goal_gain=float(gain),
                            forecast_extra_rad=float(extra),
                            pose_cost_at_view=float(cost(view)), sight_improves=opportunity)
            candidates.append((priority, view, current_cost, identity))
        self.node._trace_observation('task_route_frontier_search', frontiers=sum(g is not None for g in frontiers),
                                     r1_views=len(views), supported_views=len(available),
                                     finite_views=sum(c[-1]['predicted_finite_cells'] > 0 for c in candidates),
                                     wall_ms=1000. * (time.monotonic() - began))
        if not candidates:
            return None
        _, view, value, identity = max(candidates, key=lambda item: item[0])
        return dict(ready=True, start=start, target=view, goal=goal,
                    cost=value, direct_cost=direct_cost, identity=identity)
