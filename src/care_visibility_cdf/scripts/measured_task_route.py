"""Rank short, measured-map-supported task segments and R1 sensing motions."""
from threading import Lock
import time
from types import SimpleNamespace

import numpy as np
import rospy
import torch
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from tf.transformations import quaternion_matrix
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from validate_visibility_oracle import DEFAULT_JOINT_NAMES, fk_transform

from paired_view import TaskRouteComparison as PairedView


class TaskRouteComparison(PairedView):
    def __init__(self, node):
        super().__init__(node)
        self.selection_lock = Lock()
        self.measurements_lock = Lock()
        self.surfaces = {}
        self.goal_pose = None
        self.return_subscriber = None
        self.last_ready = None
        self.goal_subscriber = rospy.Subscriber(
            '/care_planner/ee_target_pose', PoseStamped, self._goal, queue_size=1)

    def choose(self, *args, **kwargs):
        with self.selection_lock:
            return super().choose(*args, **kwargs)

    def _goal(self, message):
        if message.header.frame_id.lstrip('/') != 'base_link':
            return
        p, q = message.pose.position, message.pose.orientation
        position = np.array([p.x, p.y, p.z])
        quaternion = np.array([q.x, q.y, q.z, q.w])
        if np.isfinite(np.r_[position, quaternion]).all() and np.linalg.norm(quaternion) > 0.:
            self.goal_pose = position, quaternion_matrix(quaternion)[:3, :3]

    def _returns(self, message):
        from ray_evidence import array

        coordinates, _ = self.geometry

        rays = array(message)
        xyz = np.column_stack([rays[axis] for axis in 'xyz'])
        valid = (rays['hit'] != 0) & np.isfinite(xyz).all(axis=1)
        valid &= (rays['sensor_id'] >= 0) & (rays['sensor_id'] < 8)
        sensors = rays['sensor_id'][valid]
        stamp = message.header.stamp.to_sec()
        with self.measurements_lock:
            self.surfaces.update({
                int(sensor): (stamp, coordinates[np.unique(self._grid_indices(xyz[valid][sensors == sensor]))])
                for sensor in np.unique(sensors)
            })

    def _grid_indices(self, points):
        origin = self.geometry[0][0]
        shape = np.rint((self.geometry[0][-1] - origin) / .05).astype(int) + 1
        ijk = np.floor((points - origin) / .05 + .5).astype(int)
        ijk = ijk[np.all((ijk >= 0) & (ijk < shape), axis=1)]
        return np.ravel_multi_index(ijk.T, tuple(shape))

    def _pose_cost(self, runtime, joints, position, rotation):
        pose = fk_transform(runtime.task_ee_chain, dict(zip(DEFAULT_JOINT_NAMES, joints)))
        distance = np.linalg.norm(pose[:3, 3] - position)
        angle = np.arccos(np.clip((np.trace(rotation.T @ pose[:3, :3]) - 1.) / 2., -1., 1.))
        return max(distance / .02, angle / .2)

    @staticmethod
    def _blocked_prefix(path, sweep, unknown, occupied):
        last = None
        for pose, cells in zip(path, sweep):
            if occupied[cells].any():
                return last, None, None
            missing = cells[unknown[cells]]
            if len(missing):
                return last, pose, missing
            last = pose
        return last, None, None

    def publish_ready(self, route):
        self.last_ready = route['target'].copy()
        message = JointTrajectory()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = 'measured_free_task_route'
        message.joint_names = list(DEFAULT_JOINT_NAMES)
        message.points = [JointTrajectoryPoint(positions=route[key].tolist())
                          for key in ('target', 'goal', 'start')]
        self.publisher.publish(message)
        self.node._trace_observation('task_route_ready_proposed',
            task_target=route['target'].tolist(), pose_cost=route['cost'],
            direct_pose_cost=route['direct_cost'], vbc_identity=route['identity'])

    def _search(self, runtime, start, stopped, goal, blocked, accel, previous_view, excluded,
                attempt, identity):
        began = time.monotonic()
        goal_pose = self.goal_pose
        if goal_pose is None:
            return None
        grid, _ = self._geometry(runtime)
        if self.return_subscriber is None:
            self.return_subscriber = rospy.Subscriber(
                '/care_planner/perception/valid_interval_tof_rays', PointCloud2,
                self._returns, queue_size=1, buff_size=4 << 20)

        def cost(q):
            return self._pose_cost(runtime, q, *goal_pose)

        def clip(q):
            return np.clip(q, runtime.q_min_np, runtime.q_max_np)

        def toward(q):
            delta = goal - q
            return delta * (.32 / max(.32, np.max(np.abs(delta))))

        direct = toward(stopped)
        side = np.array([.25, -.07, -.20, .12, 0., 0., 0.])
        elbow = np.array([0., .18, -.23, -.24, .18, 0., 0.])
        ends = clip(stopped + np.vstack([
            direct, .5 * direct + side, .5 * direct - side,
            .5 * direct + elbow, .5 * direct - elbow, -.6 * direct,
        ]))
        first = self._sweeps(start, stopped, ends, accel, runtime)
        if first is None:
            return None
        _, poses, sweeps, unknown, occupied = first
        available, frontiers = [], []
        for index, (path, sweep) in enumerate(zip(poses, sweeps)):
            last, waiting, missing = self._blocked_prefix(path, sweep, unknown, occupied)
            if last is not None and np.max(np.abs(last - stopped)) >= .055:
                available.append((index, last))
            if missing is not None:
                frontiers.append((grid[missing], waiting))
        ready = []
        current_cost = cost(stopped)
        direct_cost = next((cost(q) for index, q in available if index == 0), current_cost)
        side_routes = [(index, q) for index, q in available if index != 0]
        if side_routes:
            via = np.array([q for _, q in side_routes])
            second = self._edges(runtime, start, via, clip(via + [toward(q) for q in via]), accel)
            if second is not None:
                paths, sweeps, unknown, occupied = second
                for q, path, sweep in zip(via, paths, sweeps):
                    last, waiting, missing = self._blocked_prefix(path, sweep, unknown, occupied)
                    reachable_cost = cost(last if last is not None else q)
                    if self.last_ready is None or np.max(np.abs(q - self.last_ready)) > .035:
                        if reachable_cost + .10 < min(current_cost, direct_cost):
                            ready.append((reachable_cost, q))
                    if missing is not None:
                        frontiers.append((grid[missing], waiting))
        if ready:
            best_cost, target = min(ready, key=lambda entry: entry[0])
            return dict(ready=True, start=start, target=target, goal=goal, cost=best_cost,
                        direct_cost=direct_cost, identity=identity)

        # Keep the earliest VBC blocker available alongside a different outgoing segment.
        options = sorted(frontiers, key=lambda item: cost(item[1]))
        targets = [(blocked[:2], None)]
        if options:
            targets.append((options[0][0], options[0][1]))
        views = [q for _, q in available]
        guided = []
        joints = torch.as_tensor(stopped[None, :], device=runtime.device, dtype=torch.float32)
        for points, _ in targets:
            xyz = torch.as_tensor(points, device=runtime.device, dtype=torch.float32)
            _, order = runtime.rank_sensors(xyz, joints)
            sensor = next((int(s) for s in order if int(s) not in excluded), None)
            if sensor is None:
                continue
            _, gradient = runtime.branch_value_and_grad_numpy(points, stopped, sensor)
            norm = np.max(abs(gradient))
            if np.isfinite(norm) and norm > 1e-8:
                guided.extend(clip(stopped + step * gradient / norm) for step in (.14, .30))
        if guided:
            proposed = np.asarray(guided)
            refined = self._sweeps(start, stopped, proposed, accel, runtime)
            if refined is not None:
                _, _, routes, unknown, occupied = refined
                views.extend(q for q, route in zip(proposed, routes)
                             if all(not (unknown[c] | occupied[c]).any() for c in route))
        views = self._deduplicate(views, .035)
        if previous_view is not None:
            views = [q for q in views if np.max(np.abs(q - previous_view)) >= .075]
        choices = self._rank_for_returns(runtime, views, targets, stopped, goal, excluded)
        selected = choices[0] if choices else None
        self.node._trace_observation('task_route_preview', vbc_identity=identity,
            reachable_views=len(views), supported_task_routes=len(available),
            selected_points=None if selected is None else selected['points'].tolist(),
            selected_view=None if selected is None else selected['view'].tolist(),
            selected_finite_cells=None if selected is None else selected['finite_cells'],
            wall_ms=1000. * (time.monotonic() - began))
        return selected

    def _rank_for_returns(self, runtime, views, targets, stopped, goal, excluded):
        from ray_prediction import predict

        if not views:
            return []
        choices = []
        for points, target in targets:
            features = np.concatenate([np.c_[points, np.tile(q, (len(points), 1))] for q in views])
            with torch.no_grad():
                predictions = runtime.model(torch.as_tensor(
                    features, device=runtime.device, dtype=torch.float32)).detach().cpu().numpy()
            head_scores = predictions.reshape(len(views), len(points), 8).min(axis=1)
            for q, scores in zip(views, head_scores):
                order = np.argsort(-scores)
                available = [int(sensor) for sensor in order if int(sensor) not in excluded]
                for rank, sensor in enumerate(available[:2]):
                    _, _, margin = self._optics(runtime, q, sensor, points)
                    score = float(scores[sensor])
                    quality = margin + .02 * np.clip(score, -1., 1.) - .015 * np.max(abs(q-stopped))
                    choices.append(dict(view=q, start=stopped, points=points, target=target,
                                        goal=goal, sensor=sensor, margin=margin, score=score,
                                        rank=rank + 1, quality=quality, finite_cells=0,
                                        nominal_terminal=margin >= 0.))
        choices.sort(key=lambda entry: entry['quality'], reverse=True)
        choices = choices[:3]
        now = rospy.Time.now().to_sec()
        with self.measurements_lock:
            surfaces = {sensor: points for sensor, (stamp, points) in self.surfaces.items()
                        if 0. <= now - stamp <= 1.}
        occupied = np.vstack(list(surfaces.values())) if surfaces else np.empty((0, 3))
        optics = SimpleNamespace(primitives=runtime.primitives, chains=runtime.collision_chains,
                                 sensors=runtime.sensor_chains)
        deadline = time.monotonic() + .05
        for entry in choices:
            if entry['margin'] >= -.025:
                shot = predict(optics, entry['view'], entry['points'], occupied,
                               sensor_ids=[entry['sensor']], measured_surfaces=surfaces,
                               deadline=deadline, grid_indices=self._grid_indices)
                entry['finite_cells'] = len(shot['finite_ids']) if shot else 0
            if not entry['nominal_terminal'] and not entry['finite_cells']:
                entry['target'] = None
        choices.sort(key=lambda entry: (entry['finite_cells'], entry['quality']), reverse=True)
        return choices
