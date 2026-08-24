#!/usr/bin/env python3

import math
import threading

import rospy
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import Bool, Float32, Float64, String


class PhaseB2ControlledTrialNode:
    """Coordinate the fixed task goal and the VBC target lifecycle.

    Legacy mode preserves C4.2: the first target+sweep pair stays frozen and a
    completed Recovery requests at most one measured-state task replan.

    C4.3 rolling mode keeps that same frozen bootstrap until execution T0.  Once
    released, selector target+sweep pairs may roll.  The current pair is locked
    throughout Recovery and RECOVERY_HOLD and is released only after the gate
    installs the measured-state replan.  Multiple Recovery/replan episodes are
    allowed in rolling mode.
    """

    def __init__(self):
        self._lock = threading.Lock()

        self.base_frame = str(rospy.get_param("~base_frame", "base_link"))
        self.goal_topic = str(rospy.get_param(
            "~goal_topic", "/care_planner/ee_target_pose"))
        self.candidate_topic = str(rospy.get_param(
            "~candidate_topic", "/care_planner/active_sensing/target_candidate"))
        self.candidate_active_topic = str(rospy.get_param(
            "~candidate_active_topic",
            "/care_planner/active_sensing/target_candidate_active"))
        self.candidate_sweep_time_topic = str(rospy.get_param(
            "~candidate_sweep_time_topic",
            "/care_planner/trajectory_risk/vbc_selected_sweep_time_s"))

        # Historical topic names remain for C4.2 compatibility.  In C4.3 they
        # carry the current selected/locked pair rather than a trial-long pair.
        self.target_topic = str(rospy.get_param(
            "~frozen_target_topic", "/care_planner/active_sensing/target_point"))
        self.sweep_time_topic = str(rospy.get_param(
            "~frozen_sweep_time_topic",
            "/care_planner/active_sensing/frozen_sweep_time_s"))
        self.selected_active_topic = str(rospy.get_param(
            "~selected_active_topic",
            "/care_planner/active_sensing/target_selection_active"))
        self.rolling_deadline_topic = str(rospy.get_param(
            "~rolling_deadline_topic",
            "/care_planner/active_sensing/visibility_waypoint_deadline_rolling"))

        self.recovery_active_topic = str(rospy.get_param(
            "~recovery_active_topic",
            "/care_planner/execution/visibility_recovery_active"))
        self.recovery_complete_topic = str(rospy.get_param(
            "~recovery_complete_topic",
            "/care_planner/execution/visibility_recovery_complete"))
        self.replan_ready_topic = str(rospy.get_param(
            "~replan_ready_topic",
            "/care_planner/execution/visibility_replan_ready"))
        self.execution_ready_topic = str(rospy.get_param(
            "~execution_ready_topic", "/care_planner/execution/ready"))
        self.execution_start_topic = str(rospy.get_param(
            "~execution_start_topic", "/care_planner/execution/start_time"))

        self.rolling_target_mode = bool(rospy.get_param(
            "~rolling_target_mode", False))
        self.safety_margin_s = float(rospy.get_param("~safety_margin_s", 0.30))
        self.require_initial_prior_ready = bool(rospy.get_param(
            "~require_initial_prior_ready", True))
        self.initial_prior_ready_topic = str(rospy.get_param(
            "~initial_prior_ready_topic",
            "/care_planner/confidence_map/initial_prior_ready"))
        self.publish_rate = float(rospy.get_param("~publish_rate", 20.0))
        self.goal_delay = float(rospy.get_param("~goal_delay", 0.5))
        self.pair_timeout = float(rospy.get_param(
            "~candidate_pair_timeout", 0.20))

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
        if self.safety_margin_s < 0.0:
            raise ValueError("safety_margin_s must be non-negative")

        self._startup_time = rospy.Time.now()
        self._goal_sent = False
        self._legacy_replan_goal_sent = False
        self._replan_count = 0

        self._selected_target = None
        self._selected_sweep_time = None
        self._selected_active = False
        self._candidate_active = False
        self._pending_candidate = None
        self._pending_candidate_received = None

        self._initial_prior_ready = not self.require_initial_prior_ready
        self._execution_released = False
        self._execution_start_s = None
        self._target_lock = False
        self._rolling_deadline_s = None

        self.goal_pub = rospy.Publisher(
            self.goal_topic, PoseStamped, queue_size=1, latch=True)
        self.target_pub = rospy.Publisher(
            self.target_topic, PointStamped, queue_size=1)
        self.sweep_pub = rospy.Publisher(
            self.sweep_time_topic, Float32, queue_size=1, latch=True)
        self.selected_active_pub = rospy.Publisher(
            self.selected_active_topic, Bool, queue_size=1, latch=True)
        self.rolling_deadline_pub = rospy.Publisher(
            self.rolling_deadline_topic, Float64, queue_size=1, latch=True)
        self.frozen_pub = rospy.Publisher(
            "~target_frozen", Bool, queue_size=1, latch=True)
        self.summary_pub = rospy.Publisher(
            "~summary", String, queue_size=1, latch=True)

        rospy.Subscriber(
            self.candidate_topic, PointStamped,
            self._candidate_callback, queue_size=1)
        rospy.Subscriber(
            self.candidate_active_topic, Bool,
            self._candidate_active_callback, queue_size=1)
        rospy.Subscriber(
            self.candidate_sweep_time_topic, Float32,
            self._candidate_sweep_callback, queue_size=1)
        rospy.Subscriber(
            self.initial_prior_ready_topic, Bool,
            self._initial_prior_ready_callback, queue_size=1)
        rospy.Subscriber(
            self.execution_ready_topic, Bool,
            self._execution_ready_callback, queue_size=1)
        rospy.Subscriber(
            self.execution_start_topic, Float64,
            self._execution_start_callback, queue_size=1)
        rospy.Subscriber(
            self.recovery_active_topic, Bool,
            self._recovery_active_callback, queue_size=1)
        rospy.Subscriber(
            self.recovery_complete_topic, Bool,
            self._recovery_complete_callback, queue_size=1)
        rospy.Subscriber(
            self.replan_ready_topic, Bool,
            self._replan_ready_callback, queue_size=1)

        self.goal_timer = rospy.Timer(
            rospy.Duration(0.05), self._goal_timer_callback)
        self.target_timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate),
            self._target_timer_callback)

        self._publish_selected_active(False)
        self._publish_frozen_state(False)
        self._publish_summary_locked()

        if self.rolling_target_mode:
            rospy.logwarn(
                "[phase_b2_trial] C4.3 ROLLING MODE: bootstrap frozen to T0; "
                "target rolls in NORMAL and locks through Recovery/replan")
        else:
            rospy.logwarn(
                "[phase_b2_trial] C4.2 CONTROLLED MODE: first VBC pair frozen")

    def _publish_frozen_state(self, frozen):
        msg = Bool()
        msg.data = bool(frozen)
        self.frozen_pub.publish(msg)

    def _publish_selected_active(self, active):
        msg = Bool()
        msg.data = bool(active)
        self.selected_active_pub.publish(msg)

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
        base = (
            "rolling={} selected_active={} candidate_active={} target_lock={} "
            "released={} replan_count={}"
        ).format(
            int(self.rolling_target_mode), int(self._selected_active),
            int(self._candidate_active), int(self._target_lock),
            int(self._execution_released), self._replan_count)

        if self._selected_target is None or self._selected_sweep_time is None:
            summary.data = "selected=0 " + base
        else:
            deadline = (
                "nan" if self._rolling_deadline_s is None
                else "{:.9f}".format(self._rolling_deadline_s))
            summary.data = (
                "selected=1 {} target=[{:.9f},{:.9f},{:.9f}] frame={} "
                "sweep_time_s={:.9f} rolling_deadline_s={}"
            ).format(
                base,
                self._selected_target.point.x,
                self._selected_target.point.y,
                self._selected_target.point.z,
                self.base_frame,
                self._selected_sweep_time,
                deadline)
        self.summary_pub.publish(summary)

    def _publish_pair_locked(self):
        if self._selected_target is None or self._selected_sweep_time is None:
            return

        target = PointStamped()
        target.header.frame_id = self.base_frame
        target.header.stamp = (
            self._selected_target.header.stamp
            if self.rolling_target_mode else rospy.Time.now())
        target.point.x = self._selected_target.point.x
        target.point.y = self._selected_target.point.y
        target.point.z = self._selected_target.point.z
        self.target_pub.publish(target)

        sweep = Float32()
        sweep.data = float(self._selected_sweep_time)
        self.sweep_pub.publish(sweep)

    def _update_rolling_deadline_locked(self):
        if (not self.rolling_target_mode or not self._execution_released or
                self._selected_target is None or
                self._selected_sweep_time is None):
            return

        target_epoch = float(self._selected_target.header.stamp.to_sec())
        if not math.isfinite(target_epoch) or target_epoch <= 0.0:
            target_epoch = rospy.Time.now().to_sec()

        epoch = target_epoch
        if self._execution_start_s is not None:
            epoch = max(epoch, float(self._execution_start_s))

        self._rolling_deadline_s = (
            epoch + max(
                0.0,
                float(self._selected_sweep_time) - self.safety_margin_s))
        msg = Float64()
        msg.data = self._rolling_deadline_s
        self.rolling_deadline_pub.publish(msg)

    def _set_selected_active_locked(self, active):
        self._selected_active = bool(active)
        self._publish_selected_active(self._selected_active)
        self._publish_frozen_state(
            self._selected_target is not None and
            (not self.rolling_target_mode or
             not self._execution_released or self._target_lock))
        self._publish_summary_locked()

    def _clear_selected_locked(self):
        self._selected_target = None
        self._selected_sweep_time = None
        self._rolling_deadline_s = None
        self._pending_candidate = None
        self._pending_candidate_received = None
        self._set_selected_active_locked(False)

    def _initial_prior_ready_callback(self, msg):
        if msg is None or not bool(msg.data):
            return
        with self._lock:
            first = not self._initial_prior_ready
            self._initial_prior_ready = True
        if first:
            rospy.logwarn(
                "[phase_b2_trial] initial trusted-free prior READY")

    def _goal_timer_callback(self, _event):
        with self._lock:
            if self._goal_sent:
                return
            if not self._initial_prior_ready:
                rospy.logwarn_throttle(
                    1.0,
                    "[phase_b2_trial] waiting for one-shot initial trusted-free prior")
                return
            if (rospy.Time.now() - self._startup_time).to_sec() < self.goal_delay:
                return
            if self.goal_pub.get_num_connections() <= 0:
                rospy.logwarn_throttle(
                    1.0, "[phase_b2_trial] waiting for EE-goal subscriber")
                return
            self.goal_pub.publish(self._goal_message())
            self._goal_sent = True
            rospy.logwarn(
                "[phase_b2_trial] FIXED EE GOAL PUBLISHED: p=[%.6f, %.6f, %.6f]",
                self.goal_x, self.goal_y, self.goal_z)

    def _candidate_active_callback(self, msg):
        if msg is None:
            return
        active = bool(msg.data)
        with self._lock:
            self._candidate_active = active
            if not self.rolling_target_mode or self._target_lock:
                return

            if not self._execution_released:
                # The first pair is immutable before T0, but the selector sends
                # target+sweep before its active Bool.  Therefore the later true
                # message must be allowed to activate the already frozen pair.
                if self._selected_target is not None:
                    if active:
                        self._set_selected_active_locked(True)
                    return
                if not active:
                    self._set_selected_active_locked(False)
                return

            if not active:
                self._clear_selected_locked()
                rospy.loginfo_throttle(
                    0.5,
                    "[phase_b2_trial] rolling selector safe -> target inactive")
            elif self._selected_target is not None:
                self._set_selected_active_locked(True)

    def _candidate_callback(self, msg):
        if msg is None:
            return
        with self._lock:
            if not self._goal_sent or self._target_lock:
                return
            if (msg.header.frame_id and
                    msg.header.frame_id.lstrip("/") != self.base_frame.lstrip("/")):
                rospy.logwarn_throttle(
                    1.0,
                    "[phase_b2_trial] ignoring candidate frame '%s' (expected '%s')",
                    msg.header.frame_id, self.base_frame)
                return
            if not self.rolling_target_mode and self._selected_target is not None:
                return
            if (self.rolling_target_mode and not self._execution_released and
                    self._selected_target is not None):
                return

            pending = PointStamped()
            pending.header.frame_id = self.base_frame
            pending.header.stamp = (
                msg.header.stamp if self.rolling_target_mode else rospy.Time.now())
            pending.point.x = msg.point.x
            pending.point.y = msg.point.y
            pending.point.z = msg.point.z
            self._pending_candidate = pending
            self._pending_candidate_received = rospy.Time.now()

    def _candidate_sweep_callback(self, msg):
        if (msg is None or not math.isfinite(float(msg.data)) or
                float(msg.data) < 0.0):
            return

        with self._lock:
            if not self._goal_sent or self._target_lock:
                return
            if not self.rolling_target_mode and self._selected_target is not None:
                return
            if (self.rolling_target_mode and not self._execution_released and
                    self._selected_target is not None):
                return
            if (self._pending_candidate is None or
                    self._pending_candidate_received is None):
                return

            age = (rospy.Time.now() - self._pending_candidate_received).to_sec()
            if age < 0.0 or age > self.pair_timeout:
                self._pending_candidate = None
                self._pending_candidate_received = None
                rospy.logwarn_throttle(
                    1.0,
                    "[phase_b2_trial] dropped stale unpaired VBC candidate (age=%.3f s)",
                    age)
                return

            self._selected_target = self._pending_candidate
            self._selected_sweep_time = float(msg.data)
            self._pending_candidate = None
            self._pending_candidate_received = None

            # Target is intentionally published before sweep.  The rolling
            # waypoint node clears its old sweep on target change.
            self._publish_pair_locked()
            self._update_rolling_deadline_locked()

            if self.rolling_target_mode:
                self._set_selected_active_locked(self._candidate_active)
                rospy.loginfo_throttle(
                    0.25,
                    "[phase_b2_trial] rolling pair x=%.3f y=%.3f z=%.3f "
                    "sweep=%.3f active=%d",
                    self._selected_target.point.x,
                    self._selected_target.point.y,
                    self._selected_target.point.z,
                    self._selected_sweep_time,
                    int(self._selected_active))
            else:
                self._set_selected_active_locked(True)
                rospy.logwarn(
                    "[phase_b2_trial] VBC TARGET+SWEEP FROZEN: "
                    "x=%.9f y=%.9f z=%.9f sweep=%.6f s",
                    self._selected_target.point.x,
                    self._selected_target.point.y,
                    self._selected_target.point.z,
                    self._selected_sweep_time)

    def _execution_ready_callback(self, msg):
        if msg is None or not bool(msg.data):
            return
        with self._lock:
            if self._execution_released:
                return
            self._execution_released = True
            self._update_rolling_deadline_locked()
            self._publish_frozen_state(self._target_lock)
            self._publish_summary_locked()
        if self.rolling_target_mode:
            rospy.logwarn(
                "[phase_b2_trial] T0 released -> rolling target selection ENABLED")

    def _execution_start_callback(self, msg):
        if msg is None or not math.isfinite(float(msg.data)):
            return
        with self._lock:
            self._execution_start_s = float(msg.data)
            self._update_rolling_deadline_locked()
            self._publish_summary_locked()

    def _recovery_active_callback(self, msg):
        if msg is None or not self.rolling_target_mode or not bool(msg.data):
            return
        with self._lock:
            self._target_lock = True
            if self._selected_target is not None:
                self._set_selected_active_locked(True)
            else:
                self._publish_summary_locked()
        rospy.logwarn(
            "[phase_b2_trial] Recovery active -> rolling target LOCKED")

    def _recovery_complete_callback(self, msg):
        if msg is None or not bool(msg.data):
            return
        with self._lock:
            if not self._goal_sent:
                return
            if not self.rolling_target_mode and self._legacy_replan_goal_sent:
                return
            if self.goal_pub.get_num_connections() <= 0:
                rospy.logerr(
                    "[phase_b2_trial] recovery complete but no EE-goal subscriber")
                return

            self.goal_pub.publish(self._goal_message())
            self._replan_count += 1
            if self.rolling_target_mode:
                self._target_lock = True
            else:
                self._legacy_replan_goal_sent = True
            self._publish_summary_locked()

        rospy.logwarn(
            "[phase_b2_trial] RECOVERY COMPLETE -> measured-state replan #%d",
            self._replan_count)

    def _replan_ready_callback(self, msg):
        if msg is None or not bool(msg.data) or not self.rolling_target_mode:
            return
        with self._lock:
            self._target_lock = False
            # The locked target has been resolved.  Wait for a fresh selector
            # pair instead of reviving the old one after replan.
            self._clear_selected_locked()
            self._publish_frozen_state(False)
            self._publish_summary_locked()
        rospy.logwarn(
            "[phase_b2_trial] replan ready -> rolling target selection UNLOCKED")

    def _target_timer_callback(self, _event):
        with self._lock:
            should_publish = (
                self._selected_target is not None and
                (not self.rolling_target_mode or
                 self._selected_active or self._target_lock))
            if not should_publish:
                return
            target = PointStamped()
            target.header.frame_id = self.base_frame
            target.header.stamp = (
                self._selected_target.header.stamp
                if self.rolling_target_mode else rospy.Time.now())
            target.point.x = self._selected_target.point.x
            target.point.y = self._selected_target.point.y
            target.point.z = self._selected_target.point.z
        self.target_pub.publish(target)


def main():
    rospy.init_node("phase_b2_controlled_trial")
    PhaseB2ControlledTrialNode()
    rospy.spin()


if __name__ == "__main__":
    main()
