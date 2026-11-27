"""Rank two task approaches before starting the existing VisCDF acquisition."""
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import rospy
import torch
from care_confidence_map.srv import QueryConfidenceSnapshotRequest
from geometry_msgs.msg import Point
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from validate_visibility_oracle import DEFAULT_JOINT_NAMES, fk_transform, sensor_margin


class TaskRouteComparison:
    def __init__(self, node):
        self.node = node
        self.last = None
        self.geometry = None
        self.publisher = rospy.Publisher('/care_planner/rolling_task/route_proposal',
                                         JointTrajectory, queue_size=1, latch=True)

    def _geometry(self, runtime):
        if self.geometry is None:
            root = Path(__file__).resolve().parents[3]
            sys.path.extend([str(root / 'scripts'), str(root / 'src/egocentric_arm_planner/scripts')])
            from batched_body import memberships
            from known_ground import load_robot_frames
            _, _, frames, rotations = load_robot_frames(str(
                root / 'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'))
            model = SimpleNamespace(kinematics=SimpleNamespace(frames=frames, rotations=rotations))
            body = SimpleNamespace(task_model=model, primitives=runtime.primitives)
            origin = np.array([-.95, -.95, 0.])
            upper = np.array([.95, .95, 1.15])
            pitch = .05
            shape = np.rint((upper - origin) / pitch).astype(int) + 1
            grid = origin + pitch * np.indices(shape).reshape(3, -1).T
            from moving_body_footprint import moving_footprint
            self.roadmap_body=body
            motion,_=moving_footprint(body,grid)
            self.geometry=grid,motion.membership
        return self.geometry

    def choose(self, blocked, reference, identity):
        node = self.node
        runtime = node._per_sensor_runtime
        measured = getattr(node, '_activation_measured_state', None)
        if runtime is None or measured is None or measured[1] is None or not reference.points:
            return None
        start, velocity, stamp = measured
        now = rospy.Time.now().to_sec()
        if not 0 <= now - stamp <= .15:
            return None
        points = np.asarray(blocked, dtype=float).reshape(-1, 3)
        key = tuple(sorted(node._cell_key(point) for point in points))
        if self.last is not None:
            previous_key, previous_pose, previous_time = self.last
            if now - previous_time < 1. or (
                    key == previous_key and np.max(np.abs(start - previous_pose)) < .08):
                return None
        self.last = key, start.copy(), now
        began = time.perf_counter()
        mapping = node._trajectory_mapping(reference)
        goal = np.asarray([reference.points[-1].positions[j] for j in mapping])
        if not np.isfinite(goal).all():
            return None
        limits = node.task_rejoin_limits_namespace
        accel_param = rospy.get_param(limits + '/joint_acceleration_limits')
        accel = np.array([accel_param[name] for name in DEFAULT_JOINT_NAMES])
        if not np.isfinite(accel).all() or np.any(accel <= 0):
            return None

        grid, memberships = self._geometry(runtime)
        stopped = np.clip(start + velocity * np.abs(velocity) / (2. * accel),
                          runtime.q_min_np, runtime.q_max_np)
        direct = goal - stopped
        span = np.max(np.abs(direct))
        if span < .1:
            return None
        direct *= min(1., .44 / span)
        # Pick one side direction from the existing proximal-joint perturbations.
        # Piecewise hypotheses include positions during a conservative braking offset.
        deltas = [direct + .36 * sign * np.eye(7)[axis]
                  for axis in (0, 2) for sign in (-1., 1.)]
        end = np.clip(stopped + np.asarray(deltas), runtime.q_min_np, runtime.q_max_np)
        poses = np.stack([np.linspace(stopped, target, 13) for target in end])
        speed = np.clip(end - stopped, -.25, .25)
        offset = speed * np.abs(speed) / (2. * accel)
        brake = np.clip(poses + offset[:, None, :] * np.sin(np.linspace(0., np.pi, 13))[None, :, None],
                        runtime.q_min_np, runtime.q_max_np)
        groups = memberships(np.concatenate(([start], [stopped], poses.reshape(-1, 7),
                                            brake.reshape(-1, 7))))
        body_at_start = groups[0]
        base_unknown = np.setdiff1d(groups[1], body_at_start)
        n = poses.shape[0] * poses.shape[1]
        sweeps = [np.setdiff1d(np.union1d(a, b), body_at_start)
                  for a, b in zip(groups[2:2+n], groups[2+n:])]
        ids = np.unique(np.concatenate([base_unknown, *sweeps]))
        if not len(ids):
            return None
        req = QueryConfidenceSnapshotRequest(points=[Point(*p) for p in grid[ids]],
                                             minimum_confidence=float(node.seen_threshold), horizon_s=0.)
        reply = node._corridor_query(req)
        if (reply is None or not reply.valid or reply.clock_fault or reply.frame != 'base_link'
                or not reply.instance or not 0 <= (rospy.Time.now() - reply.snapshot_stamp).to_sec() <= .1):
            node._trace_observation('task_route_preview', reason='map_snapshot_unavailable', vbc_identity=identity)
            return None
        inside = np.asarray(list(reply.inside_map), dtype=bool)
        confidence = np.asarray(reply.confidence_now, dtype=float)
        occupancy = np.asarray(reply.occupancy, dtype=float)
        if (not (len(inside) == len(confidence) == len(occupancy) == len(ids))
                or not np.isfinite(confidence).all() or not np.isfinite(occupancy).all()):
            return None
        unknown = ids[inside & (confidence < req.minimum_confidence) & (occupancy <= .5)]
        occupied = ids[~inside | (occupancy > .5)]
        if np.intersect1d(base_unknown, np.union1d(unknown, occupied)).size:
            return None

        options = [(point, stopped, None, 0) for point in points]
        for route, samples in enumerate(poses):
            swept = sweeps[route * len(samples):(route + 1) * len(samples)]
            for i, cells in enumerate(swept):
                if np.intersect1d(cells, occupied).size:
                    break
                missing = np.intersect1d(cells, unknown)
                if not len(missing):
                    continue
                distinct = [p for p in grid[missing] if node._cell_key(p) not in key]
                if len(distinct) == len(missing) and i >= 2:
                    for p in distinct[:3]:
                        for j in sorted({0, i // 2, i - 1}):
                            options.append((p, samples[j], samples[i], i))
                break

        tensor = torch.as_tensor(np.asarray([np.r_[p, q] for p, q, _, _ in options]),
                                 device=runtime.device, dtype=torch.float32)
        with torch.no_grad():
            scores = runtime.model(tensor).detach().cpu().numpy()
        if scores.shape != (len(options), 8) or not np.isfinite(scores).all():
            return None

        def assess(option, heads):
            point, q, _, _ = option
            joint_map = dict(zip(DEFAULT_JOINT_NAMES, q))
            for sensor in np.argsort(heads)[-2:][::-1]:
                transform = fk_transform(runtime.sensor_chains[int(sensor)], joint_map)
                margin = sensor_margin(point, transform, runtime.nom_hfov, runtime.nom_vfov,
                                       runtime.nom_z_min, runtime.nom_z_max)[0]
                yield float(margin), float(heads[sensor]), int(sensor)

        evaluated = [max(assess(option, heads)) for option, heads in zip(options, scores)]
        first_alt = len(points)
        direct_score = max(evaluated[:first_alt])
        selected = max(range(first_alt, len(options)), key=evaluated.__getitem__, default=None)
        choice = None
        if selected is not None:
            preview = evaluated[selected]
            if preview[0] > direct_score[0] and preview[1] > direct_score[1]:
                point, view, target, first_sample = options[selected]
                choice = dict(point=point, view=view, target=target, goal=goal,
                              first_sample=first_sample, rank=preview)
        node._trace_observation('task_route_preview', vbc_identity=identity,
            direct_points=points.tolist(), direct_rank=direct_score,
            alternate_samples=len(options)-first_alt, map_revision=int(reply.revision),
            selected_point=choice['point'].tolist() if choice else None,
            selected_view=choice['view'].tolist() if choice else None,
            selected_route_target=choice['target'].tolist() if choice else None,
            first_unknown_sample=choice['first_sample'] if choice else None,
            alternate_rank=choice['rank'] if choice else None,
            wall_ms=1000.*(time.perf_counter()-began))
        return choice

    def publish(self, token, choice):
        message = JointTrajectory()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = token
        message.joint_names = list(DEFAULT_JOINT_NAMES)
        message.points = [JointTrajectoryPoint(positions=choice[key].tolist())
                          for key in ('target', 'goal')]
        self.publisher.publish(message)
