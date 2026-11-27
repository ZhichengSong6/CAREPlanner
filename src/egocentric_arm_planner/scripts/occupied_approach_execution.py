"""Carry an occupied veto across reference changes in the existing execution gate."""
from threading import Lock
from types import SimpleNamespace

import numpy as np
import rospy
from care_confidence_map.srv import QueryConfidenceSnapshot, QueryConfidenceSnapshotRequest
from geometry_msgs.msg import Point
from std_msgs.msg import Bool, String

from occupied_route_geometry import joint_segment
from occupied_veto_execution import OccupiedVeto


def stop_samples(poses, velocities, acceleration):
    duration = np.abs(velocities) / acceleration
    time = np.minimum(np.linspace(0., 1., 9)[None, :, None] * duration.max(axis=1)[:, None, None],
                      duration[:, None, :])
    return (poses[:, None, :] + velocities[:, None, :] * time
            - .5 * np.sign(velocities[:, None, :]) * acceleration * time**2).reshape(-1, poses.shape[1])


def approaches_veto(body, measured, sweep, cells, contact_margin, near_distance, tolerance):
    from batched_body import point_clearances

    distance = point_clearances(body, np.vstack([measured, sweep]), cells) - contact_margin
    floor = np.minimum(distance[0], near_distance)
    return bool(np.any(distance[1:] < floor - tolerance))


def approaching_wrist_cells(body, measured, swept, grid, tree, margin):
    from batched_body import _primitive_distances

    poses = np.vstack((measured, np.unique(swept, axis=0)))
    initial = np.full(len(grid), np.inf)
    later = initial.copy()
    for ids, distances in _primitive_distances(body, poses, grid, tree, search_margin=margin):
        np.minimum.at(initial, ids, distances[0])
        np.minimum.at(later, ids, distances[1:].min(axis=0))
    crossing = np.flatnonzero(later < np.minimum(initial, margin)-1e-5)
    return crossing[np.argsort(later[crossing], kind='stable')]


class OccupiedApproach(OccupiedVeto):
    def __init__(self, node):
        super().__init__(node)
        self.lock = Lock()
        self.conflicts = {}
        self.stop_veto_stamp = 0
        self.completed_held_token = 0
        self.stop_pub = rospy.Publisher('/care_planner/rolling_task/occupied_stop_confirmed',
                                        String, queue_size=4)
        self.near_distance = .5 * node.final_occupied_resolution_m
        self.tolerance = max(node.final_occupied_hard_margin_m, 1e-7)
        self.snapshot_query = rospy.ServiceProxy('/care_planner/confidence_map/query_snapshot',
                                                 QueryConfidenceSnapshot)
        self.refresh_timer = rospy.Timer(rospy.Duration(.1), self._refresh_observed_free)
        from path_seed_geometry import COORDS, TREE
        self.wrist_grid, self.wrist_tree = COORDS, TREE
        self.wrist_query = rospy.ServiceProxy('/care_planner/confidence_map/query_snapshot',
                                              QueryConfidenceSnapshot)
        self.wrist = SimpleNamespace(task_model=self.body.task_model,
            primitives=[part for part in self.body.primitives if part['link'].startswith('wrist_link')])

    def approaching_measured_wrist(self, trajectory):
        sweep = self._sweep(trajectory, 0., rospy.Time.now())
        if sweep is None:
            rospy.logwarn_throttle(1., '[MEASURED_WRIST_CLEARANCE] measured state unavailable')
            return None
        measured, poses = sweep
        tracking_margin_m = .020
        margin = self.margin + tracking_margin_m
        ids = approaching_wrist_cells(self.wrist, measured, poses, self.wrist_grid, self.wrist_tree, margin)
        if not len(ids):
            return []
        points = self.wrist_grid[ids]
        request = QueryConfidenceSnapshotRequest(points=[Point(*p) for p in points],
                                                 minimum_confidence=.5, horizon_s=0.)
        try:
            reply = self.wrist_query(request)
        except rospy.ServiceException as exc:
            rospy.logwarn_throttle(1., '[MEASURED_WRIST_CLEARANCE] snapshot request failed: %s', exc)
            return None
        if (not reply.valid or reply.clock_fault or reply.frame != 'base_link' or not reply.instance
                or reply.observation_mode != 'tof_ray'
                or not 0. <= (rospy.Time.now()-reply.snapshot_stamp).to_sec() <= .1
                or len(reply.inside_map) != len(points) or len(reply.occupancy) != len(points)):
            rospy.logwarn_throttle(1., '[MEASURED_WRIST_CLEARANCE] snapshot invalid: %s, clock_fault=%d',
                                   reply.reason, reply.clock_fault)
            return None
        hits = np.asarray(reply.inside_map, dtype=bool) & (np.asarray(reply.occupancy) > .5)
        return points[hits][:12].reshape(-1).tolist()

    def _refresh_observed_free(self, _event):
        with self.lock:
            conflicts = tuple(self.conflicts.items())
        if not conflicts:
            return
        request = QueryConfidenceSnapshotRequest(
            points=[Point(*p) for p, _ in conflicts], minimum_confidence=.5)
        try:
            reply = self.snapshot_query(request)
        except rospy.ServiceException:
            return
        if (not reply.valid or reply.clock_fault or reply.frame != 'base_link'
                or not 0. <= (rospy.Time.now() - reply.snapshot_stamp).to_sec() <= .1):
            return
        with self.lock:
            for (point, stamp), inside, confidence, occupancy, seen, provenance in zip(
                    conflicts, reply.inside_map, reply.confidence_now, reply.occupancy,
                    reply.last_seen_s, reply.provenance):
                if (inside and confidence >= .5 and occupancy <= .5 and provenance == 1
                        and seen > stamp and self.conflicts.get(point) == stamp):
                    del self.conflicts[point]
                    rospy.loginfo('[OCCUPIED_VETO] cleared_by_new_tof cell=%s', point)

    def _sweep(self, trajectory, phase, now):
        node = self.node
        if not 0 <= now.to_nsec() - node._latest_measured_stamp_ns <= 100_000_000:
            return None
        names = list(trajectory.joint_names)
        order = [names.index(n) for n in self.names]
        try:
            measured = np.asarray([node._latest_measured_q_by_name[n] for n in self.names])
            speed = np.asarray([node._latest_measured_velocity_by_name[n] for n in self.names])
        except KeyError:
            return None
        acceleration = np.asarray(node.repair_acceleration_limits)[order]
        end = node._duration(trajectory)
        timestamps = np.unique(np.r_[np.linspace(phase, end, 1 + max(1, int(np.ceil((end-phase)/.025)))),
                            [p.time_from_start.to_sec() for p in trajectory.points
                             if p.time_from_start.to_sec() >= phase]])
        points = [node._point_at_time(trajectory, t) for t in timestamps]
        positions = np.array([p.positions for p in points])[:, order]
        if not np.isfinite(np.r_[measured, speed, positions.ravel()]).all():
            return None
        velocities = np.array([p.velocities if len(p.velocities) == len(names)
                               else np.zeros(len(names)) for p in points])[:, order]
        if not np.isfinite(velocities).all():
            return None
        swept = np.vstack([joint_segment(measured, positions[0]), positions,
                           stop_samples(np.vstack([measured, positions]), np.vstack([speed, velocities]), acceleration)])
        return measured, swept

    def _approaches(self, trajectory, phase, now):
        with self.lock:
            cells = np.array(tuple(self.conflicts))
        if not cells.size:
            return False
        sweep = self._sweep(trajectory, phase, now)
        if sweep is None:
            return True
        measured, path = sweep
        return approaches_veto(self.body, measured, path, cells, self.margin, self.near_distance, self.tolerance)

    def reject(self, candidate, raw_stamp, points, overflow, now):
        with self.lock:
            for point in map(tuple, np.asarray(points, dtype=float).reshape(-1, 3)):
                self.conflicts.pop(point, None)
                self.conflicts[point] = now.to_sec()
            while len(self.conflicts) > 32:
                del self.conflicts[next(iter(self.conflicts))]
        was_held = bool(self.hold_token)
        super().reject(candidate, raw_stamp, points, overflow, now)
        if not was_held and self.hold_token:
            self.stop_veto_stamp = candidate.header.stamp.to_nsec()
        node = self.node
        master = node._committed_master
        state = node._execution_tracker_state
        if (self.hold_token or master is None or state is None or state[0] != master.header.stamp.to_nsec()
                or node._handoff_tracker_fields.get('source') != 'active_trajectory'):
            return
        if self._approaches(master, min(state[1], node._duration(master)), now):
            self.hold_token = state[0]
            self.stop_veto_stamp = candidate.header.stamp.to_nsec()
            self.hold_pub.publish(Bool(data=True))
            rospy.logwarn('[OCCUPIED_VETO] stop_approaching execution=%d candidate=%d', state[0], raw_stamp)

    def observe_tracker(self, state, fields, now):
        held = self.hold_token
        finished_tail = False
        if held:
            node = self.node
            master = node._committed_master
            if (master is not None and state[0] == held == master.header.stamp.to_nsec()
                    and state[1] >= node._duration(master)-1e-9
                    and fields.get('complete') == '1' and fields.get('execution_aborted') == '0'
                    and fields.get('source') == 'trajectory_complete_hold'):
                self.completed_held_token = held
            completed = held in (self.completed_held_token, node._handoff_completed_token)
            latest = node._latest_measured_velocity_by_name
            if (completed and fields.get('active') == '0' and state[0] in (0, held)
                    and 0 <= now.to_nsec()-node._latest_measured_stamp_ns <= 100_000_000
                    and all(abs(latest.get(name, float('inf'))) <= .05 for name in self.names)):
                self.release_stamp = now.to_nsec()
                self.hold_token = 0
                self.hold_pub.publish(Bool(data=False))
                finished_tail = True
                rospy.loginfo('[OCCUPIED_VETO] stop_confirmed_completed_tail execution=%d', held)
        if not finished_tail:
            super().observe_tracker(state, fields, now)
        if held and not self.hold_token:
            self.stop_pub.publish(String(data=(f'execution_stamp_ns={held} '
                f'veto_stamp_ns={self.stop_veto_stamp} completed_tail={int(finished_tail)}')))
            self.stop_veto_stamp = self.completed_held_token = 0

    def permits_commit(self, candidate):
        if not super().permits_commit(candidate):
            return False
        if self._approaches(candidate, 0., rospy.Time.now()):
            rospy.loginfo('[OCCUPIED_VETO] veto_retained_for_commit candidate=%d', candidate.header.stamp.to_nsec())
            return False
        return True
