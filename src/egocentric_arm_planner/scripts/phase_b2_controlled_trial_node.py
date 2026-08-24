#!/usr/bin/env python3

import math
import threading

import rospy
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import Bool, Float32, Float64, String


class PhaseB2ControlledTrialNode:
    """Coordinate the fixed task goal and the active-sensing target lifecycle.

    Legacy mode preserves the validated C2/C4.2 behavior: the first VBC
    target+sweep pair is frozen for the whole trial and a successful Recovery
    requests one measured-state task replan.

    C4.3 rolling mode keeps the same pre-execution bootstrap semantics, then
    releases target selection after T0.  During normal execution it forwards
    the selector's current critical target+sweep pair.  During Recovery and
    RECOVERY_HOLD the current pair is locked so q_vis cannot move underneath
    the validated Recovery controller.  The lock is released only when the
    gate reports replan-ready, after which the selector may choose the next
    critical target.  Rolling mode supports multiple Recovery/replan episodes.
    """

    def __init__(self):
        self._lock = threading.Lock()

        self.base_frame = str(rospy.get_param("~base_frame", "base_link"))
        self.goal_topic = str(rospy.get_param("~goal_topic", "/care_planner/ee_target_pose"))
        self.candidate_topic = str(rospy.get_param(
            "~candidate_topic", "/care_planner/active_sensing/target_candidate"))
        self.candidate_active_topic = str(rospy.get_param(
            "~candidate_active_topic",
            "/care_planner/active_sensing/target_candidate_active"))
        self.candidate_sweep_time_topic = str(rospy.get_param(
            "~candidate_sweep_time_topic",
            "/care_planner/trajectory_risk/vbc_selected_sweep_time_s"))

        # Historical topic names are retained for compatibility.  In rolling
        # mode they carry the currently selected/locked pair rather than a
        # trial-long frozen pair.
        self.frozen_target_topic = str(rospy.get_param(
            "~frozen_target_topic", "/care_planner/active_sensing/target_point"))
        self.frozen_sweep_time_topic = str(rospy.get_param(
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

        self.rolling_target_mode = bool(rospy.get_param("~rolling_target_mode", False))
        self.safety_margin_s = float(rospy.get_param("~safety_margin_s", 0.30))
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

        self.goal_pub = rospy.Publisher(self.goal_topic, PoseStamped, queue_size=1, latch=True)
        self.target_pub = rospy.Publisher(self.frozen_target_topic, PointStamped, queue_size=1)
        self.sweep_pub = rospy.Publisher(
            self.frozen_sweep_time_topic, Float32, queue_size=1, latch=True)
        self.selected_active_pub = rospy.Publisher(
            self.selected_active_topic, Bool, queue_size=1, latch=True)
        self.rolling_deadline_pub = rospy.Publisher(
            self.rolling_deadline_topic, Float64, queue_size=1, latch=True)
        self.frozen_pub = rospy.Publisher("~target_frozen", Bool, queue_size=1, latch=True)
        self.summary_pub = rospy.Publisher("~summary", String, queue_size=1, latch=True)

        rospy.Subscriber(
            self.candidate_topic, PointStamped, self._candidate_callback, queue_size=1)
        rospy.Subscriber(
            self.candidate_active_topic, Bool, self._candidate_active_callback, queue_size=1)
        rospy.Subscriber(
            self.candidate_sweep_time_topic, Float32, self._candidate_sweep_callback, queue_size=1)
        rospy.Subscriber(
            self.initial_prior_ready_topic, Bool, self._initial_prior_ready_callback, queue_size=1)
        rospy.Subscriber(
            self.execution_ready_topic, Bool, self._execution_ready_callback, queue_size=1)
        rospy.Subscriber(
            self.execution_start_topic, Float64, self._execution_start_callback, queue_size=1)
        rospy.Subscriber(
            self.recovery_active_topic, Bool, self._recovery_active_callback, queue_size=1)
        rospy.Subscriber(
            self.recovery_complete_topic, Bool, self._recovery_complete_callback, queue_size=1)
        rospy.Subscriber(
            self.replan_ready_topic, Bool, self._replan_ready_callback, queue_size=1)

        self.goal_timer = rospy.Timer(rospy.Duration(0.05), self._goal_timer_callback)
        self.target_timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate), self._target_timer_callback)

        self._publish_selected_active(False)
        self._publish_frozen_state(False)
        self._publish_summary_locked()

        if self.rolling_target_mode:
            rospy.logwarn(
                "[phase_b2_trial] C4.3 ROLLING MODE: freeze bootstrap pair until T0, then roll; lock target during Recovery/replan")
        else:
            rospy.logwarn(
                "[phase_b2_trial] CONTROLLED MODE: fixed EE goal + first VBC target/sweep frozen")
        rospy.loginfo(
            "[phase_b2_trial] fixed goal p=[%.6f, %.6f, %.6f] q=[%.8f, %.8f, %.8f, %.8f]",
            self.goal_x, self.goal_y, self.goal_z,
            self.goal_qx, self.goal_qy, self.goal_qz, self.goal_qw)

    def _publish_frozen_state(self, frozen):
        msg = Bool(); msg.data = bool(frozen); self.frozen_pub.publish(msg)

    def _publish_selected_active(self, active):
        msg = Bool(); msg.data = bool(active); self.selected_active_pub.publish(msg)

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
        if self._selected_target is None or self._selected_sweep_time is None:
            summary.data = (
                "rolling={} selected=0 selected_active={} target_lock={} "
                "released={} replan_count={}"
            ).format(
                int(self.rolling_target_mode), int(self._selected_active),
                int(self._target_lock), int(self._execution_released),
                self._replan_count)
        else:
            summary.data = (
                "rolling={} selected=1 selected_active={} target_lock={} "
                "released={} target=[{:.9f},{:.9f},{:.9f}] frame={} "
                "sweep_time_s={:.9f} rolling_deadline_s={} replan_count={}"
            ).format(
                int(self.rolling_target_mode), int(self._selected_active),
                int(self._target_lock), int(self._execution_released),
                self._selected_target.point.x,
                self._selected_target.point.y,
                self._selected_target.point.z,
                self.base_frame,
                self._selected_sweep_time,
                "nan" if self._rolling_deadline_s is None
                else "{:.9f}".format(self._rolling_deadline_s),
                self._replan_count)
        self.summary_pub.publish(summary)

    def _publish_pair_locked(self):
        if self._selected_target is None or self._selected_sweep_time is None:
            return

        target = PointStamped()
        target.header.frame_id = self.base_frame
        if self.rolling_target_mode:
            target.header.stamp = self._selected_target.header.stamp
        else:
            target.header.stamp = rospy.Time.now()
        target.point.x = self._selected_target.point.x
        target.point.y = self._selected_target.point.y
        target.point.z = self._selected_target.point.z
        self.target_pub.publish(target)

        sweep = Float32(); sweep.data = float(self._selected_sweep_time)
        self.sweep_pub.publish(sweep)

    def _update_rolling_deadline_locked(self):
        if (not self.rolling_target_mode or not self._execution_released or
                self._selected_target is None or self._selected_sweep_time is None):
            return

        target_epoch = float(self._selected_target.header.stamp.to_sec())
        if not math.isfinite(target_epoch) or target_epoch <= 0.0:
            target_epoch = rospy.Time.now().to_sec()
        epoch = target_epoch
        if self._execution_start_s is not None:
            epoch = max(epoch, float(self._execution_start_s))

        self._rolling_deadline_s = (
            epoch + max(0.0, float(self._selected_sweep_time) - self.safety_margin_s))
        msg = Float64(); msg.data = self._rolling_deadline_s
        self.rolling_deadline_pub.publish(msg)

    def _set_selected_active_locked(self, active):
        self._selected_active = bool(active)
        self._publish_selected_active(self._selected_active)
        self._publish_frozen_state(
            self._selected_target is not None and
            (not self.rolling_target_mode or not self._execution_released or self._target_lock))
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
            if not self.rolling_target_mode:
                return
            if self._target_lock:
                return
            if not self._execution_released:
                # The first bootstrap pair remains fixed until the gate releases
                # T0.  This preserves the validated pre-execution synchronization.
                if self._selected_target is not None:
                    return
                if not active:
                    self._set_selected_active_locked(False)
                return

            if not active:
                self._clear_selected_locked()
                rospy.loginfo_throttle(
                    0.5, "[phase_b2_trial] rolling selector has no VBC violation -> target inactive")
            elif self._selected_target is not None:
                self._set_selected_active_locked(True)

    def _candidate_callback(self, msg):
        if msg is None:
            return
        with self._lock:
            if not self._goal_sent:
                return
            if msg.header.frame_id and msg.header.frame_id.lstrip("/") != self.base_frame.lstrip("/"):
                rospy.logwarn_throttle(
                    1.0,
                    "[phase_b2_trial] ignoring candidate in frame '%s' (expected '%s')",
                    msg.header.frame_id, self.base_frame)
                return
            if self._target_lock:
                return
            if not self.rolling_target_mode and self._selected_target is not None:
                return
            if self.rolling_target_mode and not self._execution_released and self._selected_target is not None:
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
        if msg is None or not math.isfinite(float(msg.data)) or float(msg.data) < 0.0:
            return

        with self._lock:
            if not self._goal_sent or self._target_lock:
                return
            if not self.rolling_target_mode and self._selected_target is not None:
                return
            if self.rolling_target_mode and not self._execution_released and self._selected_target is not None:
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

            self._selected_target = self._pending_candidate
            self._selected_sweep_time = float(msg.data)
            self._pending_candidate = None
            self._pending_candidate_received = None

            # Publish target first, then its paired sweep.  The rolling waypoint
            # generator clears its old sweep on target change, so it cannot
            # accidentally generate a new q_vis from a mismatched pair.
            self._publish_pair_locked()
            self._update_rolling_deadline_locked()

            if self.rolling_target_mode:
                self._set_selected_active_locked(self._candidate_active)
                rospy.loginfo_throttle(
                    0.25,
                    "[phase_b2_trial] rolling target pair: x=%.3f y=%.3f z=%.3f sweep=%.3f active=%d",
                    self._selected_target.point.x,
                    self._selected_target.point.y,
                    self._selected_target.point.z,
                    self._selected_sweep_time,
                    int(self._selected_active))
            else:
                self._set_selected_active_locked(True)
                rospy.logwarn(
                    "[phase_b2_trial] VBC TARGET+SWEEP FROZEN: x=%.9f y=%.9f z=%.9f sweep=%.6f s",
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
            rospy.logwarn("[phase_b2_trial] T0 released -> rolling target selection ENABLED")

    def _execution_start_callback(self, msg):
        if msg is None or not math.isfinite(float(msg.data)):
            return
        with self._lock:
            self._execution_start_s = float(msg.data)
            self._update_rolling_deadline_locked()
            self._publish_summary_locked()

    def _recovery_active_callback(self, msg):
        if msg is None or not self.rolling_target_mode:
            return
        if not bool(msg.data):
            # Do not unlock here: false is published when Recovery enters
            # RECOVERY_HOLD.  The target remains locked until replan-ready.
            return
        with self._lock:
            self._target_lock = True
            if self._selected_target is not None:
                self._set_selected_active_locked(True)
            else:
                self._publish_summary_locked()
        rospy.logwarn("[phase_b2_trial] Recovery active -> rolling target LOCKED")

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
                    "[phase_b2_trial] recovery complete but no EE-goal subscriber; cannot request replan")
                return

            self.goal_pub.publish(self._goal_message())
            self._replan_count += 1
            if self.rolling_target_mode:
                self._target_lock = True
            else:
                self._legacy_replan_goal_sent = True
            self._publish_summary_locked()

        rospy.logwarn(
            "[phase_b2_trial] RECOVERY COMPLETE -> REPUBLISHED ORIGINAL EE GOAL for measured-state replan #%d",
            self._replan_count)

    def _replan_ready_callback(self, msg):
        if msg is None or not bool(msg.data) or not self.rolling_target_mode:
            return
        with self._lock:
            self._target_lock = False
            # The locked target has just been resolved.  Do not revive it from
            # stale state; wait for the next fresh selector target pair.
            self._clear_selected_locked()
            self._publish_frozen_state(False)
            self._publish_summary_locked()
        rospy.logwarn("[phase_b2_trial] replan ready -> rolling target selection UNLOCKED")

    def _target_timer_callback(self, _event):
        with self._lock:
            should_publish = (
                self._selected_target is not None and
                (not self.rolling_target_mode or self._selected_active or self._target_lock))
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
