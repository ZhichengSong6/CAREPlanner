#!/usr/bin/env python3
"""Final low-level joint-velocity envelope for CAREPlanner.

TrajectoryExecutionManager produces the desired tracking velocity.  This node is
an actuator-side rate limiter: it enforces the same per-joint velocity and
acceleration limits used by the planner before publishing the command to the
robot.  It contains no planning or VBC logic.
"""

import math
import threading

import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray, String


class JointVelocityRateLimiter:
    def __init__(self):
        self._lock = threading.Lock()

        self.rate = float(rospy.get_param("~rate", 100.0))
        self.input_topic = str(rospy.get_param(
            "~input_topic", "/care_planner/execution/tracker_velocity_desired"))
        self.output_topic = str(rospy.get_param(
            "~output_topic", "/care_arm/arm_group_velocity_controller/command"))
        self.summary_topic = str(rospy.get_param(
            "~summary_topic", "/care_planner/execution/rate_limiter_summary"))
        self.input_timeout = float(rospy.get_param("~input_timeout", 0.10))

        self.joint_names = list(rospy.get_param("~joint_names", []))
        if not self.joint_names:
            raise RuntimeError("~joint_names is required")
        self.dof = len(self.joint_names)

        vel_cfg = rospy.get_param("~mpc/joint_velocity_limits", {})
        acc_cfg = rospy.get_param("~mpc/joint_acceleration_limits", {})
        self.velocity_limits = self._joint_vector(
            vel_cfg, "mpc/joint_velocity_limits", fallback=2.0)
        self.acceleration_limits = self._joint_vector(
            acc_cfg, "mpc/joint_acceleration_limits", fallback=4.0)

        if self.rate <= 0.0:
            raise RuntimeError("~rate must be positive")
        if self.input_timeout <= 0.0:
            raise RuntimeError("~input_timeout must be positive")
        if np.any(self.velocity_limits <= 0.0):
            raise RuntimeError("velocity limits must be positive")
        if np.any(self.acceleration_limits <= 0.0):
            raise RuntimeError("acceleration limits must be positive")

        self._desired = np.zeros(self.dof, dtype=np.float64)
        self._last_output = np.zeros(self.dof, dtype=np.float64)
        self._desired_received = None
        self._last_tick = None
        self._input_count = 0
        self._output_count = 0
        self._limited_cycle_count = 0
        self._max_accel_ratio = 0.0
        self._last_status = "waiting_input"

        self.pub = rospy.Publisher(
            self.output_topic, Float64MultiArray, queue_size=1)
        self.summary_pub = rospy.Publisher(
            self.summary_topic, String, queue_size=1, latch=True)
        self.sub = rospy.Subscriber(
            self.input_topic, Float64MultiArray, self._desired_cb, queue_size=1)
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.rate), self._timer_cb)

        rospy.logwarn(
            "[joint_velocity_rate_limiter] desired=%s actuator=%s rate=%.1fHz "
            "vel=%s accel=%s",
            self.input_topic, self.output_topic, self.rate,
            np.array2string(self.velocity_limits, precision=3),
            np.array2string(self.acceleration_limits, precision=3))
        self._publish_summary("waiting_input", 0.0, 0.0)

    def _joint_vector(self, cfg, name, fallback):
        if isinstance(cfg, (int, float)):
            return np.full(self.dof, float(cfg), dtype=np.float64)
        if isinstance(cfg, (list, tuple)):
            if len(cfg) != self.dof:
                raise RuntimeError("{} must have {} entries".format(name, self.dof))
            return np.asarray(cfg, dtype=np.float64)
        if isinstance(cfg, dict):
            return np.asarray(
                [float(cfg.get(j, fallback)) for j in self.joint_names],
                dtype=np.float64)
        return np.full(self.dof, float(fallback), dtype=np.float64)

    def _desired_cb(self, msg):
        if msg is None or len(msg.data) != self.dof:
            rospy.logwarn_throttle(
                1.0, "[joint_velocity_rate_limiter] malformed desired command")
            return
        desired = np.asarray(msg.data, dtype=np.float64)
        if not np.all(np.isfinite(desired)):
            rospy.logwarn_throttle(
                1.0, "[joint_velocity_rate_limiter] non-finite desired command")
            return
        with self._lock:
            self._desired = desired
            self._desired_received = rospy.Time.now()
            self._input_count += 1

    def _timer_cb(self, _event):
        now = rospy.Time.now()
        with self._lock:
            desired = self._desired.copy()
            received = self._desired_received
            last = self._last_output.copy()
            last_tick = self._last_tick

        if last_tick is None:
            dt = 1.0 / self.rate
        else:
            dt = max(1e-4, min(2.0 / self.rate, (now - last_tick).to_sec()))

        fresh = received is not None and 0.0 <= (now - received).to_sec() <= self.input_timeout
        if not fresh:
            desired = np.zeros(self.dof, dtype=np.float64)
            status = "stale_ramp_to_zero" if received is not None else "waiting_input"
        else:
            status = "tracking"

        desired = np.clip(desired, -self.velocity_limits, self.velocity_limits)
        max_delta = self.acceleration_limits * dt
        raw_delta = desired - last
        limited_delta = np.clip(raw_delta, -max_delta, max_delta)
        output = last + limited_delta
        output = np.clip(output, -self.velocity_limits, self.velocity_limits)

        limited = bool(np.any(np.abs(raw_delta - limited_delta) > 1e-10))
        accel = np.abs(output - last) / max(dt, 1e-9)
        ratio = float(np.max(accel / self.acceleration_limits)) if self.dof else 0.0

        msg = Float64MultiArray()
        msg.data = output.astype(float).tolist()
        self.pub.publish(msg)

        with self._lock:
            self._last_output = output
            self._last_tick = now
            self._output_count += 1
            if limited:
                self._limited_cycle_count += 1
                status += "_limited"
            self._max_accel_ratio = max(self._max_accel_ratio, ratio)
            self._last_status = status
            max_accel_ratio = self._max_accel_ratio

        self._publish_summary(status, dt, max_accel_ratio)

    def _publish_summary(self, status, dt, max_accel_ratio):
        with self._lock:
            text = (
                "status={} input_count={} output_count={} limited_cycle_count={} "
                "dt_s={:.6f} max_accel_ratio={:.6f}"
            ).format(
                status,
                self._input_count,
                self._output_count,
                self._limited_cycle_count,
                float(dt),
                float(max_accel_ratio),
            )
        self.summary_pub.publish(String(data=text))


def main():
    rospy.init_node("joint_velocity_rate_limiter")
    JointVelocityRateLimiter()
    rospy.spin()


if __name__ == "__main__":
    main()
