"""Generate whole-arm motion that can use free space measured by the onboard ToFs."""
import time

import numpy as np
from rospy import Time
from care_confidence_map.srv import QueryConfidenceSnapshotRequest
from geometry_msgs.msg import Point

from retained_frontier_endpoint import TaskRouteComparison as RetainedRoute


class TaskRouteComparison(RetainedRoute):
    def __init__(self, node):
        rng = np.random.default_rng(20261120)
        envelope = np.array([.50, .15, .46, .44, .48, .35, .46])
        self.body_offsets = rng.uniform(-1., 1., size=(2048, 7)) * envelope
        self.body_offsets[:512] *= .5
        self.entry_offsets = rng.uniform(-.075, .075, size=(128, 7))
        self.last_body_search = -np.inf
        super().__init__(node)

    def _entry(self, runtime, packets, start, stopped, goal, accel):
        resumed = self._resume(runtime, start, stopped, goal, accel)
        if resumed is not None:
            return resumed
        stamp = packets[-1].header.stamp.to_sec()
        if stamp - self.last_body_search >= 1.:
            self.last_body_search = stamp
            entry = self._enter_free_space(runtime, start, stopped, goal, accel)
            if entry is not None:
                return entry
        return self._new_view(runtime, packets, start, stopped, goal, accel)

    def _entry_support(self, grid, ids):
        threshold = float(self.node.seen_threshold)
        request = QueryConfidenceSnapshotRequest(points=[Point(*p) for p in grid[ids]],
                                                 minimum_confidence=threshold, horizon_s=0.)
        reply = self.query(request)
        if (reply is None or not reply.valid or reply.clock_fault or reply.frame != 'base_link'
                or not reply.instance or not 0. <= (Time.now() - reply.snapshot_stamp).to_sec() <= .1
                or any(len(a) != len(ids) for a in (reply.inside_map, reply.confidence_now,
                                                   reply.occupancy, reply.provenance))):
            return None
        flags = np.asarray(list(reply.provenance), dtype=int) & 7
        measured = flags == 1
        free = (np.asarray(list(reply.inside_map), dtype=bool) &
                (np.asarray(reply.confidence_now) >= threshold) &
                (np.asarray(reply.occupancy) <= .5) & ((flags & 3) != 0) & ((flags & 4) == 0))
        traversable = np.zeros(len(grid), dtype=bool)
        sensor_only = traversable.copy()
        traversable[ids] = free
        sensor_only[ids] = measured & free
        return traversable, sensor_only

    def _enter_free_space(self, runtime, start, stopped, goal, accel):
        began = time.monotonic()
        if self.goal_pose is None:
            return None
        grid, body = self._geometry(runtime)
        ends = np.clip(stopped + self.body_offsets, runtime.q_min_np, runtime.q_max_np)
        # Work in small kinematic batches to bound each array.
        endpoint_body = [cells for low in range(0, len(ends), 96)
                         for cells in body(ends[low:low+96])]
        origin, = body(start[None])
        added_body = [np.setdiff1d(cells, origin, assume_unique=True) for cells in endpoint_body]
        ids = np.unique(np.concatenate(added_body))
        if not len(ids):
            return None
        support = self._entry_support(grid, ids)
        if support is None:
            return None
        free, sensor_only = support
        candidates = []
        for q, added in zip(ends, added_body):
            known = added[sensor_only[added]]
            displacement = np.max(abs(q - stopped))
            if len(known) and free[added].all() and displacement >= .045:
                candidates.append((displacement, q, known))
        if not candidates:
            self.node._trace_observation('task_route_body_entry_search', measured_candidates=0,
                                         wall_ms=1000. * (time.monotonic() - began))
            return None
        # Refine the first sensor-supported endpoints so the body enters the
        # cell before the tracker reaches its nominal stopping tolerance.
        candidates.sort(key=lambda item: item[0])
        seeds = np.array([row[1] for row in candidates[:6]])
        refined = np.clip((seeds[:, None, :] + self.entry_offsets[None]).reshape(-1, 7),
                          runtime.q_min_np, runtime.q_max_np)
        refined = np.vstack([seeds, refined])
        distances = np.max(abs(refined - stopped), axis=1)
        keep = distances >= .045
        refined, distances = refined[keep], distances[keep]
        near = refined - (refined - stopped) * (.03 / distances[:, None])
        refinement = []
        for low in range(0, len(refined), 96):
            near_body = body(near[low:low+96])
            end_body = body(refined[low:low+96])
            for q, distance, before, after in zip(refined[low:low+96], distances[low:low+96],
                                                    near_body, end_body):
                added = np.setdiff1d(after, origin, assume_unique=True)
                known = np.intersect1d(added[sensor_only[added]], before)
                if len(known) and free[added].all():
                    refinement.append((distance, q, known))

        # Reserve candidate slots for different observed clusters.
        shortlist = []
        per_cell = {}
        for _, q, cells in sorted(refinement, key=lambda item: item[0]):
            primary = min(cells, key=lambda cell: per_cell.get(cell, 0))
            if per_cell.get(primary, 0) >= 12:
                continue
            if any(np.max(abs(q - old)) < .05 for old in self.recent_targets):
                continue
            per_cell[primary] = per_cell.get(primary, 0) + 1
            shortlist.append((q, cells))
            if len(shortlist) >= 20:
                break
        if not shortlist:
            return None

        delta = goal - stopped
        direct = stopped + delta * (.38 / max(.38, np.max(abs(delta))))
        proposals = np.vstack([direct, *[q for q, _ in shortlist]])
        preview = self._sweeps(start, stopped, proposals, accel, runtime)
        if preview is None:
            return None
        _, paths, swept, unknown, occupied = preview
        direct_last, _, _ = self._blocked_prefix(paths[0], swept[0], unknown, occupied)
        pose_cost = lambda q: self._pose_cost(runtime, q, *self.goal_pose)
        direct_cost = pose_cost(stopped if direct_last is None else direct_last)
        ready = []
        for (q, cells), poses, path in zip(shortlist, paths[1:], swept[1:]):
            last, _, _ = self._blocked_prefix(poses, path, unknown, occupied)
            if last is not None and np.allclose(last, q):
                ready.append((q, cells))
        self.node._trace_observation('task_route_body_entry_search', measured_candidates=len(candidates),
                                     usable_candidates=len(ready), checks=len(shortlist),
                                     wall_ms=1000. * (time.monotonic() - began))
        if not ready:
            return None
        if direct_last is not None and np.allclose(direct_last, direct) and direct_cost < pose_cost(stopped):
            return None

        starts = np.array([q for q, _ in ready])
        displacement = goal - starts
        next_pose = starts + displacement * (.38 / np.maximum(.38, np.max(abs(displacement), axis=1)))[:, None]
        outgoing = self._edges(runtime, start, starts, next_pose, accel)
        if outgoing is None:
            return None
        paths, swept, unknown, occupied = outgoing
        ranked = []
        for (q, cells), path, sets in zip(ready, paths, swept):
            last, _, missing = self._blocked_prefix(path, sets, unknown, occupied)
            value = pose_cost(q if last is None else last)
            ranked.append((value, q, cells, path, missing))
        value, q, cells, path, missing = min(ranked, key=lambda item: item[0])
        return dict(ready=True, start=start, target=q, goal=goal, cost=value, direct_cost=direct_cost,
                    continuation=path if missing is not None else None, missing=missing,
                    identity=dict(measured_body_entry=True, used_cells_m=grid[cells].tolist(),
                                  next_unknown=None if missing is None else grid[missing].tolist()))
