#!/usr/bin/env python3
import math
import threading

import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class MeasuredStateTrajectoryNode:
    def __init__(self):
        self.joint_states_topic = str(rospy.get_param(
            "~joint_states_topic", "/care_arm/joint_states"))
        self.output_topic = str(rospy.get_param(
            "~output_topic", "/care_planner/execution_gcdf/measured_trajectory"))
        self.rate_hz = float(rospy.get_param("~rate", 20.0))
        self.max_age_s = float(rospy.get_param("~max_age_s", 0.15))
        self.frame_id = str(rospy.get_param("~frame_id", "base_link"))
        self.joint_names = list(rospy.get_param("~joint_names", []))
        if not self.joint_names:
            raise RuntimeError("~joint_names is empty")
        if self.rate_hz <= 0.0 or self.max_age_s <= 0.0:
            raise RuntimeError("invalid measured-state trajectory timing")

        self._lock = threading.Lock()
        self._latest = None
        self._received = rospy.Time(0)
        self._seq = 0

        self.pub = rospy.Publisher(
            self.output_topic, JointTrajectory, queue_size=1)
        rospy.Subscriber(
            self.joint_states_topic, JointState, self._joint_cb, queue_size=1)
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.rate_hz), self._timer_cb)

        rospy.loginfo(
            "[Phase E5 measured trajectory] joint_states=%s output=%s rate=%.1fHz",
            self.joint_states_topic, self.output_topic, self.rate_hz)

    def _joint_cb(self, msg):
        if msg is None:
            return
        with self._lock:
            self._latest = msg
            self._received = rospy.Time.now()

    def _timer_cb(self, _event):
        now = rospy.Time.now()
        with self._lock:
            msg = self._latest
            received = self._received
        if msg is None:
            return
        if (now - received).to_sec() > self.max_age_s:
            rospy.logwarn_throttle(
                1.0, "[Phase E5 measured trajectory] joint state stale")
            return

        idx = {name: i for i, name in enumerate(msg.name)}
        q = []
        for name in self.joint_names:
            i = idx.get(name)
            if i is None or i >= len(msg.position):
                rospy.logwarn_throttle(
                    1.0,
                    "[Phase E5 measured trajectory] missing joint %s", name)
                return
            v = float(msg.position[i])
            if not math.isfinite(v):
                return
            q.append(v)

        out = JointTrajectory()
        self._seq += 1
        out.header.seq = self._seq
        out.header.stamp = now
        out.header.frame_id = self.frame_id
        out.joint_names = list(self.joint_names)

        pt = JointTrajectoryPoint()
        pt.positions = q
        pt.velocities = [0.0] * len(q)
        pt.time_from_start = rospy.Duration(0.0)
        out.points = [pt]
        self.pub.publish(out)


if __name__ == "__main__":
    rospy.init_node("execution_gcdf_measured_state_trajectory")
    MeasuredStateTrajectoryNode()
    rospy.spin()
