"""Use the existing controlled stop if an occupied veto overlaps active motion."""
import math

import numpy as np
import rospy
from care_collision_cdf.msg import CollisionCDFRejectionFeedback
from std_msgs.msg import Bool

from occupied_route_geometry import joint_segment, load_body, occupied_conflict


class OccupiedVeto:
    def __init__(self, node):
        self.node = node
        self.names, self.body = load_body()
        self.margin = (.5 * math.sqrt(3.) * node.final_occupied_resolution_m
                       + node.final_occupied_hard_margin_m)
        self.hold_token = 0
        self.release_stamp = 0
        self.feedback_pub = rospy.Publisher('/care_planner/rolling_task/occupied_rejection',
                                            CollisionCDFRejectionFeedback, queue_size=4)
        self.hold_pub = rospy.Publisher('/care_planner/execution_gcdf/hard_hold',
                                        Bool, queue_size=4, latch=True)

    def reject(self, candidate, raw_stamp, points, overflow, now):
        self.feedback_pub.publish(CollisionCDFRejectionFeedback(
            header=candidate.header, raw_candidate_stamp=rospy.Time(0, raw_stamp),
            point_flat=points, overflow=overflow))
        node = self.node
        master = node._committed_master
        state, received = node._execution_tracker_state, node._execution_tracker_received
        if (master is None or state is None or received is None or not points or self.hold_token
                or state[0] != master.header.stamp.to_nsec()
                or node._handoff_tracker_fields.get('source') != 'active_trajectory'
                or not 0. <= (now - received).to_sec() <= .15
                or not 0 <= now.to_nsec() - node._latest_measured_stamp_ns <= 100_000_000):
            return
        positions = node._latest_measured_q_by_name
        if any(name not in positions or name not in master.joint_names for name in self.names):
            return
        measured = np.asarray([positions[name] for name in self.names])
        if not np.isfinite(measured).all():
            return
        idx = [master.joint_names.index(name) for name in self.names]
        phase = min(state[1], node._duration(master))
        end = node._duration(master)
        times = np.unique(np.r_[np.linspace(phase, end, max(2, 1 + math.ceil((end-phase) / .025))),
                                [p.time_from_start.to_sec() for p in master.points
                                 if p.time_from_start.to_sec() >= phase]])
        future = [np.asarray(node._point_at_time(master, t).positions)[idx] for t in times]
        poses = np.vstack([joint_segment(measured, future[0]), future])
        conflict = occupied_conflict(self.body, poses, np.asarray(points).reshape(-1, 3), self.margin)
        rospy.loginfo('[OCCUPIED_VETO] checked_execution=%d candidate=%d conflict=%d', state[0], raw_stamp, conflict)
        if conflict:
            self.hold_token = state[0]
            self.hold_pub.publish(Bool(data=True))
            rospy.logwarn('[OCCUPIED_VETO] controlled_stop execution=%d candidate=%d cells=%s',
                          self.hold_token, raw_stamp, points)

    def observe_tracker(self, state, fields, now):
        if not self.hold_token or state[0] != self.hold_token or fields.get('source') != 'stopped_safety_hold':
            return
        rospy.loginfo('[OCCUPIED_VETO] stop_confirmed execution=%d', self.hold_token)
        self.release_stamp = now.to_nsec()
        self.hold_token = 0
        self.hold_pub.publish(Bool(data=False))

    def permits_commit(self, candidate):
        return not self.hold_token and candidate.header.stamp.to_nsec() > self.release_stamp
