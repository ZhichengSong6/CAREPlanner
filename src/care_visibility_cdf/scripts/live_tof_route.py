"""Propose local motion from recent finite ToF free space using the existing SCP."""
from collections import deque
import time

import numpy as np
import rospy
from care_confidence_map.srv import QueryConfidenceSnapshotRequest
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from validate_visibility_oracle import DEFAULT_JOINT_NAMES, fk_transform

from measured_task_route import TaskRouteComparison as MeasuredTaskRoute
from paired_view import TaskRouteComparison as PairedView


class TaskRouteComparison(MeasuredTaskRoute):
    def __init__(self, node):
        super().__init__(node)
        self.packets = deque(maxlen=4)
        self.sensor_free = None
        self.last_packet = -1.
        self.last_request = -1.
        self.route_future = None
        self.regime = ''
        self.acceleration = None
        self.regime_subscriber = rospy.Subscriber('/care_planner/c4_4/regime_summary',
                                                 String, self._regime, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(.1), self._from_map)

    def _regime(self, message):
        self.regime = message.data.partition('state=')[2].split(' ', 1)[0]

    def _returns(self, message):
        with self.measurements_lock:
            self.packets.append(message)
        super()._returns(message)

    def choose(self, *args, **kwargs):
        with self.selection_lock:
            if self.route_future is not None:
                self.pending = False
                return None
            return PairedView.choose(self, *args, **kwargs)

    def _from_map(self, _timer):
        with self.selection_lock:
            runtime = getattr(self.node, '_per_sensor_runtime', None)
            if runtime is None:
                return
            if self.return_subscriber is None:
                self._geometry(runtime)
                self.return_subscriber = rospy.Subscriber(
                    '/care_planner/perception/valid_interval_tof_rays',
                    PointCloud2, self._returns, queue_size=1, buff_size=4 << 20)
            self._update_route(runtime)

    def _update_route(self, runtime):
        state = getattr(self.node, '_activation_measured_state', None)
        now = rospy.Time.now().to_sec()
        if self.route_future is not None:
            if not self.route_future.done():
                return
            future, self.route_future = self.route_future, None
            try:
                route = future.result()
            except Exception as exc:
                self.node._trace_observation('task_route_map_error', error=repr(exc))
                route = None
            if (route is not None and self.regime == 'REPAIR' and state is not None
                    and 0. <= now - state[2] <= .15
                    and np.max(np.abs(state[0] - route['start'])) < .06):
                self.publish_ready(route)
                return
            if (route is not None and getattr(self, 'task_sweep_handoff', False)
                    and not route.get('rechecked_start') and self.regime == 'REPAIR' and state is not None
                    and state[1] is not None and self.acceleration is not None
                    and 0. <= now-state[2] <= .15):
                drift = np.max(abs(state[0] - route['start']))
                if .06 <= drift <= .24:
                    self.route_future = self.worker.submit(self._recheck_route_start, runtime, state, route)
                    self.node._trace_observation('task_route_recheck_start_requested',
                        drift_rad=float(drift), target=route['target'].tolist())
                    return

        reference = self.node._trajectory
        if (self.regime != 'REPAIR' or self.goal_pose is None or state is None
                or state[1] is None or not 0. <= now - state[2] <= .15
                or not reference or not reference.points or now - self.last_request < .35
                or (self.work is not None and not self.work[-1].done())):
            return
        with self.measurements_lock:
            packets = [p for p in self.packets if p.header.stamp.to_sec() > self.last_packet]
        if not packets:
            return
        self.last_packet = packets[-1].header.stamp.to_sec()
        if self.acceleration is None:
            limits = rospy.get_param(self.node.task_rejoin_limits_namespace + '/joint_acceleration_limits')
            self.acceleration = np.array([limits[name] for name in DEFAULT_JOINT_NAMES])
        start, velocity, _ = state
        mapping = self.node._trajectory_mapping(reference)
        if mapping is None:
            return
        goal = np.array([reference.points[-1].positions[i] for i in mapping])
        accel = self.acceleration
        if not np.isfinite(np.r_[goal, accel]).all() or np.any(accel <= 0.):
            return
        stopped = np.clip(start + velocity * np.abs(velocity) / (2. * accel),
                          runtime.q_min_np, runtime.q_max_np)
        self.last_request = now
        self.route_future = self.worker.submit(self._entry, runtime, packets, start, stopped, goal, accel)

    def _recheck_route_start(self, runtime, state, route):
        start, velocity, _ = state
        accel = self.acceleration
        stopped = np.clip(start + velocity*abs(velocity)/(2.*accel), runtime.q_min_np, runtime.q_max_np)
        target = route['target']
        if self._conflict(start, target):
            return None
        steps = max(14, int(np.ceil(np.max(abs(target-stopped))/.035)))
        preview = self._sweeps(start, stopped, target[None], accel, runtime, samples=steps)
        if preview is None:
            return None
        _, _, sweeps, unknown, occupied = preview
        if any((unknown[ids] | occupied[ids]).any() for ids in sweeps[0]):
            return None
        identity = route['identity']
        if identity.get('route_goal_view_return'):
            observed = self._grid_indices(np.asarray(identity['points']))
            if not np.intersect1d(observed, np.unique(np.concatenate(sweeps[0]))).size:
                return None
        self.node._trace_observation('task_route_recheck_start_supported',
            target=target.tolist(), measured_q=start.tolist())
        return dict(route, start=start, rechecked_start=True)

    def _finite_free_hints(self, packets, size):
        from ray_evidence import array

        if self.sensor_free is None:
            self.sensor_free = np.full(size, -np.inf)
        for packet in packets:
            rays = array(packet)
            origin = np.column_stack([rays['origin_' + axis] for axis in 'xyz'])
            end = np.column_stack([rays[axis] for axis in 'xyz'])
            direction = end - origin
            length = np.linalg.norm(direction, axis=1)
            valid = ((rays['hit'] != 0) & (rays['sensor_id'] >= 0) & (rays['sensor_id'] < 8)
                     & np.isfinite(origin).all(axis=1)
                     & np.isfinite(end).all(axis=1) & (length >= .05) & (length <= 1.5))
            if not np.any(valid):
                continue
            origin, direction, length = origin[valid], direction[valid], length[valid]
            # Origins and endpoints already use the adapter's measured optically valid interval.
            distances = np.arange(0., length.max(), .025)
            ray, step = np.nonzero(distances[None, :] < length[:, None] - .025)
            points = origin[ray] + direction[ray] * (distances[step] / length[ray])[:, None]
            self.sensor_free[self._grid_indices(points)] = packet.header.stamp.to_sec()
        return np.flatnonzero(self.sensor_free >= packets[-1].header.stamp.to_sec() - 12.)

    @staticmethod
    def _surface(primitive, points):
        geometry = primitive['geometry']
        if primitive['type'] == 'box':
            half = .5 * np.asarray(geometry.size)
            return np.clip(points, -half, half)
        radius = geometry.radius
        if primitive['type'] == 'sphere':
            return points * (radius / np.maximum(radius, np.linalg.norm(points, axis=1)))[:, None]
        surface = points.copy()
        radial = np.linalg.norm(points[:, :2], axis=1)
        surface[:, :2] *= (radius / np.maximum(radius, radial))[:, None]
        surface[:, 2] = np.clip(points[:, 2], -.5 * geometry.length, .5 * geometry.length)
        return surface

    def _entry(self, runtime, packets, start, stopped, goal, accel):
        began = time.monotonic()
        grid, membership = self.geometry
        ids = np.setdiff1d(self._finite_free_hints(packets, len(grid)), membership([stopped])[0])
        if not len(ids):
            return None
        cost = lambda q: self._pose_cost(runtime, q, *self.goal_pose)
        clip = lambda q: np.clip(q, runtime.q_min_np, runtime.q_max_np)
        toward = lambda q: (goal - q) * (.32 / max(.32, np.max(np.abs(goal - q))))

        offsets = np.eye(7) * .0001
        joints = dict(zip(DEFAULT_JOINT_NAMES, stopped))
        neighbors = [dict(zip(DEFAULT_JOINT_NAMES, q)) for q in stopped + offsets]
        frames, local_points, deltas = [], [], []
        for primitive, chain in zip(runtime.primitives, runtime.collision_chains):
            if primitive['link'] in ('base_link', 'link1'):
                continue
            frame = fk_transform(chain, joints) @ primitive['origin']
            local = (grid[ids] - frame[:3, 3]) @ frame[:3, :3]
            surface = self._surface(primitive, local)
            frames.append((primitive, chain, frame))
            local_points.append(surface)
            deltas.append((local - surface) @ frame[:3, :3].T)
        if not frames:
            return None
        gaps = np.linalg.norm(np.array(deltas), axis=2)
        nearest = gaps.argmin(axis=0)
        separation = gaps[nearest, np.arange(len(ids))]
        near = np.flatnonzero((.012 < separation) & (separation < .16))
        order = near[np.argsort(separation[near], kind='stable')[:96]]
        if not len(order):
            return None
        points = grid[ids[order]]
        threshold = float(self.node.seen_threshold)
        reply = self.query(QueryConfidenceSnapshotRequest(points=[Point(*p) for p in points],
                           minimum_confidence=threshold, horizon_s=0.))
        if (reply is None or not reply.valid or reply.clock_fault or reply.frame != 'base_link'
                or not 0. <= (rospy.Time.now() - reply.snapshot_stamp).to_sec() <= .1
                or any(len(a) != len(order) for a in
                       (reply.provenance, reply.inside_map, reply.confidence_now, reply.occupancy))):
            return None
        live = (np.asarray(list(reply.provenance), dtype=int) & 7) == 1
        live &= np.asarray(list(reply.inside_map), dtype=bool)
        live &= (np.asarray(reply.confidence_now) >= threshold) & (np.asarray(reply.occupancy) <= .5)

        ends = [clip(stopped + toward(stopped))]
        cells, links, selected = [], [], []
        for index in order[live]:
            point = grid[ids[index]]
            if any(np.linalg.norm(point - previous) < .065 for previous in selected):
                continue
            primitive, chain, frame = frames[nearest[index]]
            surface = np.r_[local_points[nearest[index]][index], 1.]
            jacobian = np.column_stack([
                ((fk_transform(chain, q) @ primitive['origin'] - frame) @ surface)[:3] / .0001
                for q in neighbors])
            delta = deltas[nearest[index]][index] * (1. + .012 / separation[index])
            inverse = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + np.eye(3) * .025**2)
            step = inverse @ delta
            if np.max(np.abs(step)) > .42:
                continue
            # The free cell specifies the link motion. A null-space term also
            # lets the remaining axes prepare for the original task.
            steering = toward(stopped) - inverse @ (jacobian @ toward(stopped))
            ends.extend((clip(stopped + step), clip(stopped + step + .50 * steering)))
            cells.extend((ids[index], ids[index]))
            links.extend((primitive['link'], primitive['link']))
            selected.append(point)
            if len(selected) == 5:
                break
        if not cells:
            return None

        first = self._sweeps(start, stopped, np.asarray(ends), accel, runtime)
        if first is None:
            return None
        _, paths, sweeps, unknown, occupied = first
        direct, _, _ = self._blocked_prefix(paths[0], sweeps[0], unknown, occupied)
        direct_cost = cost(direct if direct is not None else stopped)
        options = []
        for path, sweep, cell, link in zip(paths[1:], sweeps[1:], cells, links):
            for pose, touched in zip(path, sweep):
                if (unknown[touched] | occupied[touched]).any():
                    break
                if (cell in touched and np.max(np.abs(pose - stopped)) >= .055
                        and (self.last_ready is None or np.max(np.abs(pose - self.last_ready)) > .045)
                        and cell in membership([pose])[0]):
                    options.append((pose, cell, link))
                    break
        if not options or (direct is not None and np.allclose(direct, paths[0][-1])
                           and cost(stopped) - direct_cost >= 1.):
            return None

        via = np.array([q for q, _, _ in options])
        second = self._edges(runtime, start, via, clip(via + [toward(q) for q in via]), accel)
        if second is None:
            return None
        paths, sweeps, unknown, occupied = second
        candidates = []
        for (q, cell, link), path, sweep in zip(options, paths, sweeps):
            last, _, missing = self._blocked_prefix(path, sweep, unknown, occupied)
            score = cost(last if last is not None else q)
            if score < direct_cost or (cost(stopped) - direct_cost < .5 and missing is not None):
                score += .15 * np.max(np.abs(q - stopped))
                candidates.append((score, q, cell, link))
        self.node._trace_observation('task_route_map_search', options=len(options),
            candidates=len(candidates), nearby_measured_cells=int(live.sum()),
            wall_ms=1000. * (time.monotonic() - began))
        if not candidates:
            return None
        score, target, cell, link = min(candidates, key=lambda route: route[0])
        return dict(ready=True, start=start, target=target, goal=goal, cost=score,
                    direct_cost=direct_cost, identity=dict(measured_free_cell=grid[cell].tolist(),
                                                          moving_link=link))
