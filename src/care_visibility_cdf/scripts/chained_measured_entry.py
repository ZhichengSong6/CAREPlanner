"""Continue an evidence-supported body entry through the existing SCP interface."""
import numpy as np

from targeted_measured_entry import TaskRouteComparison as TargetedEntry


class TaskRouteComparison(TargetedEntry):
    def __init__(self, node):
        self.active_entry = None
        self.last_entry_gap = None
        self.entry_stalls = 0
        super().__init__(node)

    def publish_ready(self, route):
        if route['identity'].get('entry_chain_prefix'):
            self.candidate_to_recheck = None
        super().publish_ready(route)

    def _entry(self, runtime, packets, start, stopped, goal, accel):
        if self.active_entry is not None:
            if np.max(abs(self.active_entry['goal'] - goal)) < .02:
                next_segment = self._next_entry_segment(runtime, start, stopped, accel)
                if next_segment is not None:
                    return next_segment
                if self.active_entry is not None:
                    # A current snapshot can be temporarily unavailable.
                    return None
            self.active_entry = None
        candidate = super()._entry(runtime, packets, start, stopped, goal, accel)
        if candidate is None or not candidate['identity'].get('measured_body_entry'):
            return candidate
        self.candidate_to_recheck = None
        self.active_entry = candidate
        self.last_entry_gap = None
        self.entry_stalls = 0
        return self._next_entry_segment(runtime, start, stopped, accel)

    def _next_entry_segment(self, runtime, start, stopped, accel):
        route = self.active_entry
        gap = np.max(abs(route['target'] - stopped))
        grid, body = self._geometry(runtime)
        ids = self._grid_indices(np.asarray(route['identity']['used_cells_m']))
        current, = body(start[None])
        used = np.intersect1d(current, ids)
        if used.size:
            self.node._trace_observation('task_route_body_entry_reached', measured_cells=grid[used].tolist(),
                                         residual_joint_gap_rad=float(gap))
            if route.get('continuation') is not None:
                self.deferred_routes.append(route)
            self.active_entry = None
            return None
        if gap < .035 or self.entry_stalls >= 3:
            self.node._trace_observation('task_route_body_entry_released', residual_joint_gap_rad=float(gap),
                                         retries=self.entry_stalls)
            self.active_entry = None
            return None
        support = self._entry_support(grid, ids)
        if support is None:
            return None
        if not support[1][ids].any():
            self.active_entry = None
            return None

        aims = stopped + np.minimum(1., np.array([.22, .14, .09]) / gap)[:, None] * (route['target'] - stopped)
        preview = self._sweeps(start, stopped, aims, accel, runtime)
        if preview is None:
            return None
        _, paths, swept, unknown, occupied = preview
        if self.last_entry_gap is not None and gap >= self.last_entry_gap - .008:
            self.entry_stalls += 1
        else:
            self.entry_stalls = 0
        self.last_entry_gap = gap
        for path, cells in zip(paths, swept):
            target, _, _ = self._blocked_prefix(path, cells, unknown, occupied)
            if target is None or np.max(abs(target - stopped)) < .06:
                continue
            identity = dict(route['identity'], entry_chain_prefix=True,
                            retained_goal=route['target'].tolist(), joint_gap_rad=float(gap))
            return dict(route, start=start, target=target, identity=identity, continuation=None)
        return None
