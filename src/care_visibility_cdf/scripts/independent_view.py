"""Batch short task-entry and independent sensing candidates from the live map."""
import time
from itertools import product

import numpy as np
import rospy
import torch
from care_confidence_map.srv import QueryConfidenceSnapshot, QueryConfidenceSnapshotRequest
from geometry_msgs.msg import Point
from validate_visibility_oracle import DEFAULT_JOINT_NAMES, fk_transform

from observation_corridor_cost import BoundedCorridorQuery
from task_route_comparison import TaskRouteComparison as PreviousComparison


class TaskRouteComparison(PreviousComparison):
    def __init__(self, node):
        super().__init__(node)
        self.generation = 0
        self.query = BoundedCorridorQuery(rospy.ServiceProxy(
            node.confidence_query_service + '_snapshot', QueryConfidenceSnapshot), timeout_s=.06)
        self.cell_samples = .024 * np.asarray([(0, 0, 0), *product((-1, 1), repeat=3)])

    def choose(self, blocked, reference, identity, previous_view=None, excluded=()):
        node = self.node
        runtime = node._per_sensor_runtime
        measured = getattr(node, '_activation_measured_state', None)
        if runtime is None or measured is None or measured[1] is None or not reference or not reference.points:
            return None
        start, velocity, stamp = measured
        now = rospy.Time.now().to_sec()
        if not 0 <= now - stamp <= .15:
            return None
        points = np.asarray(blocked, dtype=float).reshape(-1, 3)
        key = tuple(sorted(node._cell_key(point) for point in points))
        if previous_view is None and self.last is not None:
            old_key, old_q, old_stamp = self.last
            if now - old_stamp < .5 or (key == old_key and np.max(np.abs(start - old_q)) < .08):
                return None
        self.last = key, start.copy(), now
        began = time.perf_counter()
        mapping = node._trajectory_mapping(reference)
        goal = np.asarray([reference.points[-1].positions[j] for j in mapping])
        limits = rospy.get_param(node.task_rejoin_limits_namespace + '/joint_acceleration_limits')
        accel = np.array([limits[name] for name in DEFAULT_JOINT_NAMES])
        lower, upper = runtime.q_min_np, runtime.q_max_np
        stopped = np.clip(start + velocity * np.abs(velocity) / (2. * accel), lower, upper)
        if not np.all(np.isfinite([start, velocity, goal, stopped])) or np.any(accel <= 0.):
            return None

        grid, memberships = self._geometry(runtime)
        direct = goal - stopped
        direct *= .5 / max(.5, np.max(np.abs(direct)))
        rng = np.random.default_rng(self.generation)
        self.generation += 1
        directions = rng.uniform(-1., 1., (16, 7))
        directions *= .54 / np.max(np.abs(directions), axis=1, keepdims=True)
        # Separate pure sensing moves from moves biased toward the task.
        motions = np.vstack([directions, direct, .60 * direct + directions[:8]])
        ends = np.clip(stopped + motions, lower, upper)
        fractions = np.linspace(0.125, 1., 8)
        poses = stopped + fractions[None, :, None] * (ends - stopped)[:, None, :]
        speed = np.clip(ends - stopped, -.25, .25)
        brake = speed * np.abs(speed) / (2. * accel)
        braking_poses = np.clip(poses + brake[:, None, :] * np.sin(np.pi * fractions)[None, :, None],
                                lower, upper)
        swept = memberships(np.vstack([np.linspace(start, stopped, 4), poses.reshape(-1, 7),
                                        braking_poses.reshape(-1, 7)]))
        body = swept[0]
        approach = np.setdiff1d(np.unique(np.concatenate(swept[1:4])), body)
        count = poses.shape[0] * poses.shape[1]
        steps = [np.setdiff1d(np.union1d(a, b), body) for a, b in zip(
            swept[4:4+count], swept[4+count:])]
        ids = np.unique(np.concatenate([approach, *steps]))
        if not len(ids):
            return None
        request = QueryConfidenceSnapshotRequest(points=[Point(*p) for p in grid[ids]],
            minimum_confidence=float(node.seen_threshold), horizon_s=0.)
        reply = self.query(request)
        if (reply is None or not reply.valid or reply.clock_fault or reply.frame != 'base_link'
                or not reply.instance or not 0 <= (rospy.Time.now() - reply.snapshot_stamp).to_sec() <= .1):
            node._trace_observation('task_route_preview', reason='map_snapshot_unavailable',
                vbc_identity=identity, wall_ms=1000. * (time.perf_counter() - began))
            return None
        inside = np.asarray(list(reply.inside_map), dtype=bool)
        confidence = np.asarray(list(reply.confidence_now), dtype=float)
        occupancy = np.asarray(list(reply.occupancy), dtype=float)
        if not (len(inside) == len(confidence) == len(occupancy) == len(ids)):
            return None
        valid = np.isfinite(confidence) & np.isfinite(occupancy)
        occupied = ids[~inside | ~valid | (occupancy > .5)]
        unknown = ids[inside & valid & (confidence < request.minimum_confidence) & (occupancy <= .5)]
        if np.intersect1d(approach, np.union1d(occupied, unknown)).size:
            return None

        views, entries = [], []
        for route, samples in enumerate(poses):
            for step, cells in enumerate(steps[route * len(samples):(route + 1) * len(samples)]):
                if np.intersect1d(cells, occupied).size:
                    break
                missing = np.intersect1d(cells, unknown)
                if len(missing):
                    if len(missing) <= 3:
                        entries.append((grid[missing], samples[step]))
                    break
                if step == len(samples) - 1 or step % 2 == 1:
                    views.append(samples[step])
        if previous_view is not None:
            views = [q for q in views if np.max(np.abs(q - previous_view)) >= .10]
            entries = [(points, None)]
        elif len(points) <= 3:
            entries.append((points, None))
        choice = self._rank(views, entries, stopped, runtime, excluded)
        node._trace_observation('task_route_preview', vbc_identity=identity,
            direct_points=points.tolist(), reachable_views=len(views), task_entries=len(entries),
            selected_points=choice['points'].tolist() if choice else None,
            selected_view=choice['view'].tolist() if choice else None,
            selected_nominal_margin_m=choice['margin'] if choice else None,
            selected_r1_score=choice['score'] if choice else None,
            map_revision=int(reply.revision), wall_ms=1000. * (time.perf_counter() - began))
        if choice is not None:
            choice['goal'] = goal
        return choice

    def _rank(self, views, entries, stopped, runtime, excluded):
        if not views:
            return None
        centers, groups = [], []
        for cells, target in entries:
            keys = [tuple(p) for p in cells]
            for key in keys:
                if key not in centers:
                    centers.append(key)
            groups.append(([centers.index(key) for key in keys], target))
        xyz = np.asarray(centers)[:, None, :] + self.cell_samples
        poses = np.vstack([stopped, views])
        points = xyz.reshape(-1, 3)
        inputs = np.concatenate([np.broadcast_to(points, (len(poses), *points.shape)),
                                 np.broadcast_to(poses[:, None], (len(poses), len(points), 7))], axis=2)
        with torch.no_grad():
            batch = torch.as_tensor(inputs.reshape(-1, 10), dtype=torch.float32, device=runtime.device)
            scores = runtime.model(batch).detach().cpu().numpy().reshape(len(poses), *xyz.shape[:2], 8)

        hx, hy = np.tan(np.deg2rad([runtime.nom_hfov, runtime.nom_vfov]) * .5)
        margins = np.empty_like(scores)
        for i, q in enumerate(poses):
            joints = dict(zip(DEFAULT_JOINT_NAMES, q))
            for sid, chain in enumerate(runtime.sensor_chains):
                transform = fk_transform(chain, joints)
                x, y, z = ((xyz - transform[:3, 3]) @ transform[:3, :3]).transpose(2, 0, 1)
                margins[i, :, :, sid] = np.minimum.reduce([
                    (hx*z - abs(x)) / np.hypot(1., hx), (hy*z - abs(y)) / np.hypot(1., hy),
                    z - runtime.nom_z_min, runtime.nom_z_max - z])
        # Rank the eight R1 heads and finite-volume samples jointly. The map
        # release remains tied to actual ToF evidence at the cell centers.
        cost = margins + .03 * np.clip(scores, -1., 1.)
        best_sample = cost.argmax(axis=2, keepdims=True)
        values = np.take_along_axis(cost, best_sample, axis=2)[:, :, 0]
        fov = np.take_along_axis(margins, best_sample, axis=2)[:, :, 0]
        joint_cost = .015 * np.max(abs(poses - stopped), axis=1)
        best, choice = -np.inf, None
        sample_scores = np.take_along_axis(scores, best_sample, axis=2)[:, :, 0]
        for ids, target in groups:
            score = values[:, ids].min(axis=1) - joint_cost[:, None]
            margin = fov[:, ids].min(axis=1)
            score[(margin < -.025) | (margin < margin[0] + .008)] = -np.inf
            score[0] = -np.inf
            score[:, sorted(excluded)] = -np.inf
            pose, sensor = np.unravel_index(score.argmax(), score.shape)
            if score[pose, sensor] > best:
                best = score[pose, sensor]
                head_scores = sample_scores[pose, ids].min(axis=0)
                choice = dict(view=poses[pose], start=stopped, points=np.asarray(centers)[ids],
                    target=target, sensor=int(sensor), margin=float(margin[pose, sensor]),
                    score=float(head_scores[sensor]), rank=1 + int((head_scores > head_scores[sensor]).sum()))
        return choice

    def waypoint(self, choice, trajectory, sweep, received):
        epoch = trajectory.header.stamp.to_sec()
        if epoch <= 0:
            epoch = rospy.Time.now().to_sec() if received is None else received.to_sec()
        sensor = choice['sensor']
        frame = self.node._per_sensor_runtime.sensor_chains[sensor][-1].child
        q_vis = choice['view'].tolist()
        return dict(q_vis=q_vis, q_zero=choice['start'].tolist(),
            q_vis_joint_mask=[1.] * 7, final_f_min=choice['score'],
            deadline_absolute_ros_s=epoch + max(0., sweep - self.node.safety_margin_s),
            shared_solution_mode='independent_task_view_r1', per_sensor_hybrid_used=True,
            per_sensor_selected_sensor_id=sensor, per_sensor_selected_sensor_frame=frame,
            per_sensor_selected_rank=choice['rank'],
            per_sensor_hybrid=dict(enabled=True, accepted=True, selected_q_vis=q_vis,
                proposal_kind='independent_task_view',
                selected_sensor_id=sensor, nominal_margin_m=choice['margin']))
