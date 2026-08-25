#!/usr/bin/env python3
"""Publish the committed CAREPlanner trajectory with short-horizon plan memory.

The MPC produces a fresh receding-horizon solution when all optimization inputs
are ready.  A learned q_vis refresh can temporarily suppress a fresh MPC solve.
The low-level controller should not interpret that planner-side refresh latency
as an instruction to stop.  This node therefore owns the *committed optimized
trajectory* stream:

* every fresh MPC horizon is forwarded immediately;
* during a short input gap, the remaining suffix of the last valid horizon is
  re-timed and republished;
* after a bounded continuation timeout, publication stops so downstream stale
  reference / VBC freshness logic remains fail-safe.

The output is the trajectory that should be both audited and tracked.  It is not
an actuator controller and never publishes joint commands.
"""

from __future__ import annotations

import copy
import math
import threading

import rospy
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class OptimizedTrajectoryContinuityNode:
    def __init__(self) -> None:
        self._lock = threading.Lock()

        self.input_topic = str(rospy.get_param(
            "~input_topic", "/care_planner/mpc/predicted_trajectory"))
        self.output_topic = str(rospy.get_param(
            "~output_topic", "/care_planner/optimized_trajectory"))
        self.summary_topic = str(rospy.get_param(
            "~summary_topic", "/care_planner/optimized_trajectory_summary"))
        self.rate = float(rospy.get_param("~rate", 20.0))
        self.continuation_start_delay_s = float(rospy.get_param(
            "~continuation_start_delay_s", 0.065))
        self.continuation_timeout_s = float(rospy.get_param(
            "~continuation_timeout_s", 0.50))

        if self.rate <= 0.0:
            raise ValueError("~rate must be positive")
        if self.continuation_start_delay_s <= 0.0:
            raise ValueError("~continuation_start_delay_s must be positive")
        if self.continuation_timeout_s <= self.continuation_start_delay_s:
            raise ValueError("~continuation_timeout_s must exceed start delay")

        self._trajectory = None
        self._received = None
        self._last_output_time = None
        self._last_input_gap_s = math.nan
        self._max_input_gap_s = 0.0
        self._input_count = 0
        self._output_count = 0
        self._continuation_count = 0
        self._last_source = "waiting"
        self._last_age_s = math.nan

        self.pub = rospy.Publisher(
            self.output_topic, JointTrajectory, queue_size=1)
        self.summary_pub = rospy.Publisher(
            self.summary_topic, String, queue_size=1, latch=True)
        self.sub = rospy.Subscriber(
            self.input_topic, JointTrajectory, self._trajectory_cb, queue_size=1)
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.rate), self._timer_cb)

        self._publish_summary()
        rospy.logwarn(
            "[optimized_trajectory_continuity] raw=%s committed=%s "
            "rate=%.1fHz continuation_delay=%.3fs timeout=%.3fs",
            self.input_topic, self.output_topic, self.rate,
            self.continuation_start_delay_s, self.continuation_timeout_s)

    @staticmethod
    def _duration(msg: JointTrajectory) -> float:
        if msg is None or not msg.points:
            return -1.0
        return float(msg.points[-1].time_from_start.to_sec())

    @staticmethod
    def _lerp_array(a, b, alpha):
        if len(a) == 0 and len(b) == 0:
            return []
        if len(a) != len(b):
            return list(a) if alpha < 0.5 else list(b)
        return [
            (1.0 - alpha) * float(a[i]) + alpha * float(b[i])
            for i in range(len(a))
        ]

    @classmethod
    def _interpolate_point(cls, p0, p1, alpha):
        out = JointTrajectoryPoint()
        out.positions = cls._lerp_array(p0.positions, p1.positions, alpha)
        out.velocities = cls._lerp_array(p0.velocities, p1.velocities, alpha)
        out.accelerations = cls._lerp_array(
            p0.accelerations, p1.accelerations, alpha)
        out.effort = cls._lerp_array(p0.effort, p1.effort, alpha)
        out.time_from_start = rospy.Duration(0.0)
        return out

    @classmethod
    def _suffix_from_phase(cls, master, phase_s):
        if master is None or not master.points:
            return None

        duration = cls._duration(master)
        if not math.isfinite(duration) or duration < 0.0:
            return None
        phase = max(0.0, min(float(phase_s), duration))

        out = JointTrajectory()
        out.header = copy.deepcopy(master.header)
        out.header.stamp = rospy.Time.now()
        out.joint_names = list(master.joint_names)

        points = master.points
        start_point = None
        if len(points) == 1 or phase <= points[0].time_from_start.to_sec():
            start_point = copy.deepcopy(points[0])
        elif phase >= points[-1].time_from_start.to_sec():
            start_point = copy.deepcopy(points[-1])
        else:
            hi = 1
            while (hi < len(points) and
                   points[hi].time_from_start.to_sec() < phase):
                hi += 1
            lo = hi - 1
            t0 = points[lo].time_from_start.to_sec()
            t1 = points[hi].time_from_start.to_sec()
            alpha = 0.0 if t1 <= t0 else (phase - t0) / (t1 - t0)
            start_point = cls._interpolate_point(points[lo], points[hi], alpha)

        start_point.time_from_start = rospy.Duration(0.0)
        out.points.append(start_point)
        for point in points:
            t = point.time_from_start.to_sec()
            if t <= phase + 1e-9:
                continue
            p = copy.deepcopy(point)
            p.time_from_start = rospy.Duration(t - phase)
            out.points.append(p)

        # Preserve a usable short horizon even when phase reaches the endpoint.
        if len(out.points) == 1:
            endpoint = copy.deepcopy(out.points[0])
            endpoint.time_from_start = rospy.Duration(0.05)
            if endpoint.velocities:
                endpoint.velocities = [0.0 for _ in endpoint.velocities]
            if endpoint.accelerations:
                endpoint.accelerations = [0.0 for _ in endpoint.accelerations]
            out.points.append(endpoint)
        return out

    def _publish_trajectory(self, msg, source, age_s):
        if msg is None or not msg.points:
            return
        self.pub.publish(msg)
        with self._lock:
            self._output_count += 1
            if source == "continuation":
                self._continuation_count += 1
            self._last_output_time = rospy.Time.now()
            self._last_source = source
            self._last_age_s = float(age_s)
        self._publish_summary()

    def _trajectory_cb(self, msg):
        if msg is None or not msg.points:
            return
        now = rospy.Time.now()
        with self._lock:
            if self._received is not None:
                gap = (now - self._received).to_sec()
                if gap >= 0.0:
                    self._last_input_gap_s = gap
                    self._max_input_gap_s = max(self._max_input_gap_s, gap)
            self._trajectory = copy.deepcopy(msg)
            self._trajectory.header.stamp = now
            self._received = now
            self._input_count += 1
            fresh = copy.deepcopy(self._trajectory)
        # Forward a fresh planner solution immediately; do not add a timer cycle
        # of latency between planner and low-level tracker.
        self._publish_trajectory(fresh, "fresh_mpc", 0.0)

    def _timer_cb(self, _event):
        now = rospy.Time.now()
        with self._lock:
            if self._trajectory is None or self._received is None:
                return
            age = (now - self._received).to_sec()
            if age < self.continuation_start_delay_s:
                return
            if age > self.continuation_timeout_s:
                self._last_source = "stale_stop_publish"
                self._last_age_s = age
                self._publish_summary_locked()
                return
            master = copy.deepcopy(self._trajectory)

        suffix = self._suffix_from_phase(master, age)
        self._publish_trajectory(suffix, "continuation", age)

    def _publish_summary_locked(self):
        msg = String()
        msg.data = (
            "source={} input_count={} output_count={} continuation_count={} "
            "raw_age_s={} last_input_gap_s={} max_input_gap_s={} "
            "continuation_timeout_s={:.6f}"
        ).format(
            self._last_source,
            self._input_count,
            self._output_count,
            self._continuation_count,
            "nan" if not math.isfinite(self._last_age_s)
            else "{:.6f}".format(self._last_age_s),
            "nan" if not math.isfinite(self._last_input_gap_s)
            else "{:.6f}".format(self._last_input_gap_s),
            self._max_input_gap_s,
            self.continuation_timeout_s,
        )
        self.summary_pub.publish(msg)

    def _publish_summary(self):
        with self._lock:
            self._publish_summary_locked()


def main():
    rospy.init_node("optimized_trajectory_continuity")
    OptimizedTrajectoryContinuityNode()
    rospy.spin()


if __name__ == "__main__":
    main()
