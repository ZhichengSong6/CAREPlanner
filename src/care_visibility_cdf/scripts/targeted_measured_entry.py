"""Prefilter body poses by the nearby free cells coming from live ToF."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from online_measured_entry import TaskRouteComparison as OnlineEntry


class TaskRouteComparison(OnlineEntry):
    def __init__(self, node):
        self.entry_body_model = None
        self.candidate_to_recheck = None
        super().__init__(node)

    def publish_ready(self, route):
        if route['identity'].get('measured_body_entry'):
            self.candidate_to_recheck = None
        super().publish_ready(route)

    def _entry(self, runtime, packets, start, stopped, goal, accel):
        pending = self.candidate_to_recheck
        self.candidate_to_recheck = None
        if pending is not None and np.max(abs(pending['goal'] - goal)) < .02:
            refreshed = self._recheck(runtime, pending, start, stopped, accel)
            if refreshed is not None:
                self.candidate_to_recheck = refreshed
                return refreshed
        route = super()._entry(runtime, packets, start, stopped, goal, accel)
        if route is not None and route['identity'].get('measured_body_entry'):
            self.candidate_to_recheck = route
        return route

    def _recheck(self, runtime, route, start, stopped, accel):
        grid, memberships = self._geometry(runtime)
        ids = self._grid_indices(np.asarray(route['identity']['used_cells_m']))
        support = self._entry_support(grid, ids)
        if support is None or not support[1][ids].all():
            return None
        motion = route['target'] - stopped
        span = np.max(abs(motion))
        if span < .045:
            return None
        near, = memberships((route['target'] - motion * (.03 / span))[None])
        current, = memberships(start[None])
        if not np.intersect1d(np.setdiff1d(near, current), ids).size:
            return None
        preview = self._sweeps(start, stopped, route['target'][None], accel, runtime)
        if preview is None:
            return None
        _, paths, swept, unknown, occupied = preview
        last, _, _ = self._blocked_prefix(paths[0], swept[0], unknown, occupied)
        if last is None or not np.allclose(last, route['target']):
            return None
        return dict(route, start=start, identity=dict(route['identity'], rechecked_latest_state=True))

    def _enter_free_space(self, runtime, start, stopped, goal, accel):
        grid, body = self._geometry(runtime)
        current, = body(start[None])
        if not current.size:
            return None
        local = grid[current]
        bound = .18
        ids = np.flatnonzero(np.all((grid >= local.min(axis=0) - bound)
                                   & (grid <= local.max(axis=0) + bound), axis=1))
        ids = np.setdiff1d(ids, current, assume_unique=True)
        if not ids.size:
            return None
        support = self._entry_support(grid, ids)
        if support is None:
            return None
        targets = grid[ids[support[1][ids]]]
        if not len(targets):
            return None

        from batched_body import memberships
        if self.entry_body_model is None:
            from known_ground import load_robot_frames
            root = Path(__file__).resolve().parents[3]
            _, _, frames, rotations = load_robot_frames(str(
                root / 'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'))
            kin = SimpleNamespace(frames=frames, rotations=rotations)
            self.entry_body_model = SimpleNamespace(task_model=SimpleNamespace(kinematics=kin),
                                                    primitives=runtime.primitives)

        original = self.body_offsets
        poses = np.clip(stopped + original, runtime.q_min_np, runtime.q_max_np)
        matches = [cells for k in range(0, len(poses), 160)
                   for cells in memberships(self.entry_body_model, poses[k:k+160], grid=targets)]
        shortlist = np.flatnonzero([bool(len(cells)) for cells in matches])
        self.node._trace_observation('task_route_body_prefilter', measured_cells=len(targets),
                                     poses_sampled=len(poses),
                                     forwarded_poses=len(shortlist))
        if not shortlist.size:
            return None
        self.body_offsets = original[shortlist]
        try:
            return super()._enter_free_space(runtime, start, stopped, goal, accel)
        finally:
            self.body_offsets = original
