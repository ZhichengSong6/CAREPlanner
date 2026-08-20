#!/usr/bin/env python3

import math
import threading

import rospy
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import Bool, Float32, String


class PhaseB2ControlledTrialNode:
    """Publish one fixed EE goal and freeze the first VBC target + sweep time.

    The first goal generates the frozen Phase-B2/C2 case. If deadline-miss
    recovery later succeeds, the MPC emits a one-shot recovery-complete event.
    This node then republishes the *same original EE goal* exactly once, asking
    RecedingHorizonPlanner to replan from the robot's measured post-recovery q.

    The first VBC target/sweep remains frozen throughout the trial; recovery
    replanning does not silently select a different benchmark target.
    """

    def __init__(self):
        self._lock = threading.Lock()

        self.base_frame = str(rospy.get_param("~base_frame", "base_link"))
        self.goal_topic = str(rospy.get_param("~goal_topic", "/care_planner/ee_target_pose"))
        self.candidate_topic = str(rospy.get_param(
            "~candidate_topic", "/care_planner/active_sensing/target_candidate"))
        self.candidate_sweep_time_topic = str(rospy.get_param(
            "~candidate_sweep_time_topic",
            "/care_planner/trajectory_risk/vbc_selected_sweep_time_s"))
        self.frozen_target_topic = str(rospy.get_param(
            "~frozen_target_topic", "/care_planner/active_sensing/target_point"))
        self.frozen_sweep_time_topic = str(rospy.get_param(
            "~frozen_sweep_time_topic",
            "/care_planner/active_sensing/frozen_sweep_time_s"))
        self.recovery_complete_topic = str(rospy.get_param(
            "~recovery_complete_topic",
            "/care_planner/execution/visibility_recovery_complete"))
        self.require_initial_prior_ready = bool(rospy.get_param("~require_initial_prior_ready", True))
        self.initial_prior_ready_topic = str(rospy.get_param(
            "~initial_prior_ready_topic",
            "/care_planner/confidence_map/initial_prior_ready"))
        self.publish_rate = float(rospy.get_param("~publish_rate", 20.0))
        self.goal_delay = float(rospy.get_param("~goal_delay", 0.5))
        self.pair_timeout = float(rospy.get_param("~candidate_pair_timeout", 0.20))

        self.goal_x = float(rospy.get_param("~goal/x", 0.286881))
        self.goal_y = float(rospy.get_param("~goal/y", 0.0))
        self.goal_z = float(rospy.get_param("~goal/z", 0.560765))
        self.goal_qx = float(rospy.get_param("~goal/qx", 0.0))
        self.goal_qy = float(rospy.get_param("~goal/qy", 0.0))
        self.goal_qz = float(rospy.get_param("~goal/qz", 0.70710678))
        self.goal_qw = float(rospy.get_param("~goal/qw", 0.70710678))

        if self.publish_rate <= 0.0:
            raise ValueError("publish_rate must be positive")
        if self.goal_delay < 0.0:
            raise ValueError("goal_delay must be non-negative")
        if self.pair_timeout <= 0.0:
            raise ValueError("candidate_pair_timeout must be positive")

        self._startup_time = rospy.Time.now()
        self._goal_sent = False
        self._replan_goal_sent = False
        self._replan_count = 0
        self._frozen_target = None
        self._frozen_sweep_time = None
        self._pending_candidate = None
        self._pending_candidate_received = None
        self._initial_prior_ready = not self.require_initial_prior_ready

        self.goal_pub = rospy.Publisher(self.goal_topic, PoseStamped, queue_size=1, latch=True)
        self.target_pub = rospy.Publisher(self.frozen_target_topic, PointStamped, queue_size=1)
        self.sweep_pub = rospy.Publisher(
            self.frozen_sweep_time_topic, Float32, queue_size=1, latch=True)
        self.frozen_pub = rospy.Publisher("~target_frozen", Bool, queue_size=1, latch=True)
        self.summary_pub = rospy.Publisher("~summary", String, queue_size=1, latch=True)

        rospy.Subscriber(
            self.candidate_topic, PointStamped, self._candidate_callback, queue_size=1)
        rospy.Subscriber(
            self.candidate_sweep_time_topic, Float32, self._candidate_sweep_callback, queue_size=1)
        rospy.Subscriber(
            self.initial_prior_ready_topic, Bool, self._initial_prior_ready_callback, queue_size=1)
        rospy.Subscriber(
            self.recovery_complete_topic, Bool, self._recovery_complete_callback, queue_size=1)

        self.goal_timer = rospy.Timer(rospy.Duration(0.05), self._goal_timer_callback)
        self.target_timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate), self._target_timer_callback)

        self._publish_frozen_state(False)
        rospy.logwarn(
            "[phase_b2_trial] CONTROLLED MODE: fixed EE goal + first VBC target/sweep deadline frozen")
        rospy.loginfo(
            "[phase_b2_trial] fixed goal p=[%.6f, %.6f, %.6f] q=[%.8f, %.8f, %.8f, %.8f]",
            self.goal_x, self.goal_y, self.goal_z,
            self.goal_qx, self.goal_qy, self.goal_qz, self.goal_qw)
        rospy.loginfo(
            "[phase_b2_trial] recovery_complete=%s; successful recovery republishes the original EE goal once",
            self.recovery_complete_topic)

    def _publish_frozen_state(self, frozen):
        msg = Bool(); msg.data = bool(frozen); self.frozen_pub.publish(msg)

    def _goal_message(self):
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = self.base_frame
        goal.pose.position.x = self.goal_x
        goal.pose.position.y = self.goal_y
        goal.pose.position.z = self.goal_z
        goal.pose.orientation.x = self.goal_qx
        goal.pose.orientation.y = self.goal_qy
        goal.pose.orientation.z = self.goal_qz
        goal.pose.orientation.w = self.goal_qw
        return goal

    def _publish_summary_locked(self):
        summary = String()
        if self._frozen_target is None or self._frozen_sweep_time is None:
            summary.data = "frozen=0 replan_count={}".format(self._replan_count)
        else:
            summary.data = (
                "frozen_target=[{:.9f},{:.9f},{:.9f}] frame={} "
                "frozen_sweep_time_s={:.9f} replan_count={}"
            ).format(
                self._frozen_target.point.x,
                self._frozen_target.point.y,
                self._frozen_target.point.z,
                self.base_frame,
                self._frozen_sweep_time,
                self._replan_count)
        self.summary_pub.publish(summary)

    def _initial_prior_ready_callback(self, msg):
        if msg is None or not bool(msg.data):
            return
        with self._lock:
            first = not self._initial_prior_ready
            self._initial_prior_ready = True
        if first:
            rospy.logwarn(
                "[phase_b2_trial] initial trusted-free prior READY; controlled motion may begin")

    def _goal_timer_callback(self, _event):
        with self._lock:
            if self._goal_sent:
                return
            if not self._initial_prior_ready:
                rospy.logwarn_throttle(
                    1.0, "[phase_b2_trial] waiting for one-shot initial trusted-free prior")
                return
            elapsed = (rospy.Time.now() - self._startup_time).to_sec()
            if elapsed < self.goal_delay:
                return
            if self.goal_pub.get_num_connections() <= 0:
                rospy.logwarn_throttle(1.0, "[phase_b2_trial] waiting for EE-goal subscriber")
                return
            goal = self._goal_message()
            self.goal_pub.publish(goal)
            self._goal_sent = True
            rospy.logwarn(
                "[phase_b2_trial] FIXED EE GOAL PUBLISHED exactly once: p=[%.6f, %.6f, %.6f]",
                self.goal_x, self.goal_y, self.goal_z)

    def _recovery_complete_callback(self, msg):
        if msg is None or not bool(msg.data):
            return
        with self._lock:
            if not self._goal_sent or self._replan_goal_sent:
                return
            if self.goal_pub.get_num_connections() <= 0:
                rospy.logerr("[phase_b2_trial] recovery complete but no EE-goal subscriber; cannot request replan")
                return
            goal = self._goal_message()
            self.goal_pub.publish(goal)
            self._replan_goal_sent = True
            self._replan_count += 1
            self._publish_summary_locked()
        rospy.logwarn(
            "[phase_b2_trial] RECOVERY COMPLETE -> REPUBLISHED ORIGINAL EE GOAL for one measured-state replan")

    def _candidate_callback(self, msg):
        if msg is None:
            return
        with self._lock:
            if not self._goal_sent or self._frozen_target is not None:
                return
            if msg.header.frame_id and msg.header.frame_id.lstrip("/") != self.base_frame.lstrip("/"):
                rospy.logwarn_throttle(
                    1.0,
                    "[phase_b2_trial] ignoring candidate in frame '%s' (expected '%s')",
                    msg.header.frame_id, self.base_frame)
                return

            pending = PointStamped()
            pending.header.frame_id = self.base_frame
            pending.header.stamp = rospy.Time.now()
            pending.point.x = msg.point.x
            pending.point.y = msg.point.y
            pending.point.z = msg.point.z
            self._pending_candidate = pending
            self._pending_candidate_received = rospy.Time.now()

    def _candidate_sweep_callback(self, msg):
        if msg is None or not math.isfinite(float(msg.data)) or float(msg.data) < 0.0:
            return

        with self._lock:
            if not self._goal_sent or self._frozen_target is not None:
                return
            if self._pending_candidate is None or self._pending_candidate_received is None:
                return
            age = (rospy.Time.now() - self._pending_candidate_received).to_sec()
            if age < 0.0 or age > self.pair_timeout:
                self._pending_candidate = None
                self._pending_candidate_received = None
                rospy.logwarn_throttle(
                    1.0, "[phase_b2_trial] dropped stale unpaired VBC candidate (age=%.3f s)", age)
                return

            self._frozen_target = self._pending_candidate
            self._frozen_target.header.stamp = rospy.Time.now()
            self._frozen_sweep_time = float(msg.data)
            self._pending_candidate = None
            self._pending_candidate_received = None

            sweep_msg = Float32(); sweep_msg.data = self._frozen_sweep_time
            self.sweep_pub.publish(sweep_msg)
            self._publish_frozen_state(True)
            self._publish_summary_locked()

            rospy.logwarn(
                "[phase_b2_trial] VBC TARGET+SWEEP FROZEN: x=%.9f y=%.9f z=%.9f sweep=%.6f s",
                self._frozen_target.point.x,
                self._frozen_target.point.y,
                self._frozen_target.point.z,
                self._frozen_sweep_time)

    def _target_timer_callback(self, _event):
        with self._lock:
            if self._frozen_target is None:
                return
            msg = PointStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = self.base_frame
            msg.point.x = self._frozen_target.point.x
            msg.point.y = self._frozen_target.point.y
            msg.point.z = self._frozen_target.point.z
        self.target_pub.publish(msg)


def main():
    rospy.init_node("phase_b2_controlled_trial")
    PhaseB2ControlledTrialNode()
    rospy.spin()


if __name__ == "__main__":
    main()
