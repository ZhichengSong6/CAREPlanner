#!/usr/bin/env python3

import math
import re
import threading

import rospy
from std_msgs.msg import Bool, Float64, String


_TOKEN_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


def _tokens(text):
    return {key: value for key, value in _TOKEN_RE.findall(text or "")}


def _as_bool(value):
    if value in ("1", "true", "True"):
        return True
    if value in ("0", "false", "False"):
        return False
    return None


def _as_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


class PredictedVbcRecoveryGuard:
    """Supervise predicted-trajectory visibility-before-contact safety.

    Two input modes are supported.

    C4.2 selected-target mode (default):
      visibility_waypoint_active + single-target predicted-VBC auditor Bool
      -> N consecutive violations -> direct Recovery trigger.
      Recovery completion remains the legacy selected-target behavior in the MPC.

    C4.3 global-set mode:
      trajectory_vbc_selector summary over the complete Primary+Secondary
      candidate set -> N consecutive *predicted* global violations -> Recovery.
      Once Recovery is active, N consecutive fresh global-safe evaluations
      publish recovery_clear=true.  The currently selected x* remains a steering
      target only; it no longer defines either Recovery entry or Recovery exit.

    In both modes the physical steering-target deadline is passed through
    unchanged to the MPC. In global mode a separate safety deadline is derived
    from the selector's currently selected violating candidate for diagnostics;
    it never replaces the q_vis timing deadline.

    Global mode is fail-safe with respect to evaluator freshness: after T0, a
    missing/stale predicted global summary requests verification hold. Recovery
    episodes are explicitly disarmed at recovery_complete and re-armed only at
    replan_ready so an old latched trigger/clear cannot leak across replans.
    """

    def __init__(self):
        self._lock = threading.RLock()

        self._enabled = bool(rospy.get_param("~enabled", True))
        self._rate = float(rospy.get_param("~rate", 20.0))
        self._input_timeout = float(rospy.get_param("~input_timeout", 0.25))
        self._consecutive_required = int(
            rospy.get_param("~consecutive_violations_required", 2))
        self._consecutive_safe_required = int(
            rospy.get_param("~consecutive_safe_required", 2))
        self._safety_margin_s = float(rospy.get_param("~safety_margin_s", 0.30))

        self._use_global_summary = bool(rospy.get_param(
            "~use_global_summary_input", False))
        self._episode_rearm_enabled = bool(rospy.get_param(
            "~episode_rearm_enabled", False))

        self._active_topic = str(rospy.get_param(
            "~active_topic",
            "/care_planner/active_sensing/visibility_waypoint_active"))
        self._violation_topic = str(rospy.get_param(
            "~violation_topic",
            "/care_planner/execution/predicted_vbc_violation"))
        self._global_summary_topic = str(rospy.get_param(
            "~global_summary_topic",
            "/care_planner/trajectory_risk/vbc_summary"))
        self._execution_ready_topic = str(rospy.get_param(
            "~execution_ready_topic", "/care_planner/execution/ready"))
        self._recovery_complete_topic = str(rospy.get_param(
            "~recovery_complete_topic",
            "/care_planner/execution/visibility_recovery_complete"))
        self._replan_ready_topic = str(rospy.get_param(
            "~replan_ready_topic",
            "/care_planner/execution/visibility_replan_ready"))

        self._physical_deadline_topic = str(rospy.get_param(
            "~physical_deadline_topic",
            "/care_planner/active_sensing/visibility_waypoint_deadline_synced"))
        self._effective_deadline_topic = str(rospy.get_param(
            "~effective_deadline_topic",
            "/care_planner/active_sensing/visibility_waypoint_deadline_effective"))
        self._verification_hold_topic = str(rospy.get_param(
            "~verification_hold_topic",
            "/care_planner/execution/predicted_vbc_verification_hold"))
        self._triggered_topic = str(rospy.get_param(
            "~triggered_topic",
            "/care_planner/execution/predicted_vbc_recovery_triggered"))
        self._recovery_clear_topic = str(rospy.get_param(
            "~recovery_clear_topic",
            "/care_planner/execution/predicted_vbc_recovery_clear"))
        self._summary_topic = str(rospy.get_param(
            "~summary_topic",
            "/care_planner/execution/predicted_vbc_recovery_summary"))

        if self._rate <= 0.0:
            raise ValueError("~rate must be > 0")
        if self._input_timeout <= 0.0:
            raise ValueError("~input_timeout must be > 0")
        if self._consecutive_required < 1:
            raise ValueError("~consecutive_violations_required must be >= 1")
        if self._consecutive_safe_required < 1:
            raise ValueError("~consecutive_safe_required must be >= 1")
        if self._safety_margin_s < 0.0:
            raise ValueError("~safety_margin_s must be non-negative")

        # Selected-target C4.2 input state.
        self._active = False
        self._active_received = None
        self._violation_received = None
        self._last_violation = False

        # Global C4.3 input state.
        self._execution_ready = False
        self._execution_ready_received = None
        self._global_summary_received = None
        self._global_trajectory_source = "none"
        self._global_candidate_count = 0
        self._global_primary_count = 0
        self._global_secondary_count = 0
        self._global_violation_count = 0
        self._global_primary_violation_count = 0
        self._global_secondary_violation_count = 0
        self._global_selected_group = "none"
        self._global_safety_deadline_abs = None
        self._episode_armed = True

        # The physical deadline belongs to the current steering target. It is
        # intentionally distinct from the global safety deadline above.
        self._physical_deadline_abs = None

        self._violation_streak = 0
        self._max_violation_streak = 0
        self._safe_streak = 0
        self._max_safe_streak = 0
        self._triggered = False
        self._recovery_clear = False
        self._trigger_count_total = 0
        self._clear_count_total = 0
        self._trigger_time_abs = None
        self._last_trigger_lead_s = float("nan")
        self._last_status = "waiting"

        self._deadline_pub = rospy.Publisher(
            self._effective_deadline_topic, Float64, queue_size=1)
        self._verification_hold_pub = rospy.Publisher(
            self._verification_hold_topic, Bool, queue_size=1, latch=True)
        self._triggered_pub = rospy.Publisher(
            self._triggered_topic, Bool, queue_size=1, latch=True)
        self._recovery_clear_pub = rospy.Publisher(
            self._recovery_clear_topic, Bool, queue_size=1, latch=True)
        self._summary_pub = rospy.Publisher(
            self._summary_topic, String, queue_size=1, latch=True)

        # Keep the C4.2 subscribers alive in both modes for backward-compatible
        # launch wiring. Their callbacks are ignored in global mode.
        rospy.Subscriber(
            self._active_topic, Bool, self._active_cb, queue_size=1)
        rospy.Subscriber(
            self._violation_topic, Bool, self._violation_cb, queue_size=1)
        rospy.Subscriber(
            self._physical_deadline_topic, Float64,
            self._deadline_cb, queue_size=1)

        if self._use_global_summary:
            rospy.Subscriber(
                self._global_summary_topic, String,
                self._global_summary_cb, queue_size=1)
            rospy.Subscriber(
                self._execution_ready_topic, Bool,
                self._execution_ready_cb, queue_size=1)
            if self._episode_rearm_enabled:
                rospy.Subscriber(
                    self._recovery_complete_topic, Bool,
                    self._recovery_complete_cb, queue_size=1)
                rospy.Subscriber(
                    self._replan_ready_topic, Bool,
                    self._replan_ready_cb, queue_size=1)

        self._triggered_pub.publish(Bool(data=False))
        self._recovery_clear_pub.publish(Bool(data=False))
        self._verification_hold_pub.publish(Bool(data=False))
        self._timer = rospy.Timer(
            rospy.Duration(1.0 / self._rate), self._timer_cb)

        rospy.logwarn(
            "[predicted_vbc_recovery_guard] mode=%s enabled=%d "
            "unsafe_consecutive=%d safe_consecutive=%d timeout=%.3fs "
            "episode_rearm=%d",
            "global_set" if self._use_global_summary else "selected_target",
            int(self._enabled), self._consecutive_required,
            self._consecutive_safe_required, self._input_timeout,
            int(self._episode_rearm_enabled))

    def _set_recovery_clear_locked(self, clear):
        clear = bool(clear)
        if clear == self._recovery_clear:
            return
        self._recovery_clear = clear
        self._recovery_clear_pub.publish(Bool(data=clear))
        if clear:
            self._clear_count_total += 1
            rospy.logwarn(
                "[predicted_vbc_recovery_guard] GLOBAL VBC RECOVERY CLEAR: "
                "safe_streak=%d",
                self._safe_streak)

    def _reset_decision_locked(self, publish=True):
        self._violation_streak = 0
        self._max_violation_streak = 0
        self._safe_streak = 0
        self._max_safe_streak = 0
        self._triggered = False
        self._trigger_time_abs = None
        self._last_violation = False
        self._recovery_clear = False
        if publish:
            self._triggered_pub.publish(Bool(data=False))
            self._recovery_clear_pub.publish(Bool(data=False))
            self._verification_hold_pub.publish(Bool(data=False))

    def _active_cb(self, msg):
        if msg is None or self._use_global_summary:
            return
        now = rospy.Time.now()
        with self._lock:
            was_active = self._active
            self._active = bool(msg.data)
            self._active_received = now
            if was_active and not self._active:
                self._reset_decision_locked()
                self._last_status = "inactive_reset"
            elif not self._active:
                self._last_status = "inactive"

    def _deadline_cb(self, msg):
        if (msg is None or not math.isfinite(float(msg.data)) or
                float(msg.data) <= 0.0):
            rospy.logwarn_throttle(
                1.0,
                "[predicted_vbc_recovery_guard] ignoring invalid physical deadline")
            return
        with self._lock:
            self._physical_deadline_abs = float(msg.data)

    def _execution_ready_cb(self, msg):
        if msg is None or not self._use_global_summary:
            return
        now = rospy.Time.now()
        with self._lock:
            self._execution_ready = bool(msg.data)
            self._execution_ready_received = now
            if not self._execution_ready:
                self._reset_decision_locked()
                self._last_status = "waiting_execution_release"

    def _recovery_complete_cb(self, msg):
        if (msg is None or not bool(msg.data) or
                not self._use_global_summary or
                not self._episode_rearm_enabled):
            return
        with self._lock:
            # Controller is already in RECOVERY_HOLD at this point. Clear both
            # edge signals before replan_ready can release the hold.
            self._episode_armed = False
            self._reset_decision_locked()
            self._global_summary_received = None
            self._global_safety_deadline_abs = None
            self._last_status = "recovery_complete_disarmed_waiting_replan"
            rospy.logwarn(
                "[predicted_vbc_recovery_guard] Recovery complete -> global "
                "supervisor DISARMED until replan_ready")

    def _replan_ready_cb(self, msg):
        if (msg is None or not bool(msg.data) or
                not self._use_global_summary or
                not self._episode_rearm_enabled):
            return
        with self._lock:
            self._episode_armed = True
            self._reset_decision_locked()
            # Require a genuinely fresh post-replan predicted selector result.
            # Re-anchor the freshness grace period at the new episode boundary.
            self._global_summary_received = None
            self._global_safety_deadline_abs = None
            self._execution_ready_received = rospy.Time.now()
            self._last_status = "rearmed_waiting_fresh_global_summary"
            rospy.logwarn(
                "[predicted_vbc_recovery_guard] replan_ready -> global "
                "supervisor RE-ARMED")

    def _active_fresh_locked(self, now):
        if self._active_received is None:
            return False
        age = (now - self._active_received).to_sec()
        return 0.0 <= age <= self._input_timeout

    def _audit_fresh_locked(self, now):
        if self._violation_received is None:
            return False
        age = (now - self._violation_received).to_sec()
        return 0.0 <= age <= self._input_timeout

    def _global_fresh_locked(self, now):
        if self._global_summary_received is None:
            return False
        age = (now - self._global_summary_received).to_sec()
        return 0.0 <= age <= self._input_timeout

    def _current_safety_deadline_locked(self):
        if self._use_global_summary:
            return self._global_safety_deadline_abs
        return self._physical_deadline_abs

    def _process_violation_locked(self, violation, now, source_label):
        self._last_violation = bool(violation)

        if not self._enabled:
            self._last_status = "disabled"
            return

        if violation:
            self._safe_streak = 0
            self._set_recovery_clear_locked(False)

            if self._triggered:
                self._last_status = source_label + "_triggered_unsafe"
                return

            self._violation_streak += 1
            self._max_violation_streak = max(
                self._max_violation_streak, self._violation_streak)
            self._last_status = source_label + "_violation_streak"

            if self._violation_streak < self._consecutive_required:
                return

            self._triggered = True
            self._trigger_time_abs = now.to_sec()
            self._trigger_count_total += 1
            deadline = self._current_safety_deadline_locked()
            if deadline is not None:
                self._last_trigger_lead_s = deadline - self._trigger_time_abs
            else:
                self._last_trigger_lead_s = float("nan")
            self._last_status = source_label + "_recovery_triggered"
            self._triggered_pub.publish(Bool(data=True))
            rospy.logerr(
                "[predicted_vbc_recovery_guard] PREDICTED VBC RECOVERY TRIGGER: "
                "mode=%s streak=%d safety_deadline_lead=%s",
                source_label,
                self._violation_streak,
                "nan" if not math.isfinite(self._last_trigger_lead_s)
                else "%.3fs" % self._last_trigger_lead_s)
            return

        # Safe global evaluations are relevant to Recovery exit only after a
        # Recovery trigger has been latched. Before that, they simply reset the
        # entry streak. In C4.2 selected-target mode Recovery exit remains legacy.
        self._violation_streak = 0
        self._last_status = source_label + "_safe"
        if not (self._use_global_summary and self._triggered):
            self._safe_streak = 0
            return

        self._safe_streak += 1
        self._max_safe_streak = max(self._max_safe_streak, self._safe_streak)
        self._last_status = source_label + "_safe_streak"
        if self._safe_streak >= self._consecutive_safe_required:
            self._set_recovery_clear_locked(True)
            self._last_status = source_label + "_recovery_clear"

    def _violation_cb(self, msg):
        if msg is None or self._use_global_summary:
            return
        now = rospy.Time.now()
        with self._lock:
            self._violation_received = now
            if not self._active or not self._active_fresh_locked(now):
                self._violation_streak = 0
                self._last_violation = bool(msg.data)
                self._last_status = "inactive_or_stale_active"
                return
            self._process_violation_locked(
                bool(msg.data), now, "selected_target")

    def _global_summary_cb(self, msg):
        if msg is None or not self._use_global_summary:
            return
        now = rospy.Time.now()
        fields = _tokens(msg.data)

        source = fields.get("trajectory_source", "none")
        violation = _as_bool(fields.get("has_violation"))
        candidate_count = _as_int(fields.get("candidate_count"), 0)
        if violation is None:
            return

        with self._lock:
            self._global_trajectory_source = source
            self._global_candidate_count = candidate_count
            self._global_primary_count = _as_int(
                fields.get("primary_count"), 0)
            self._global_secondary_count = _as_int(
                fields.get("secondary_count"), 0)
            self._global_violation_count = _as_int(
                fields.get("violation_count"), 0)
            self._global_primary_violation_count = _as_int(
                fields.get("primary_violation_count"), 0)
            self._global_secondary_violation_count = _as_int(
                fields.get("secondary_violation_count"), 0)
            self._global_selected_group = fields.get(
                "selected_group", "none")

            # Bootstrap summaries remain useful for steering/gating, but global
            # closed-loop safety is deliberately based only on the current MPC
            # prediction. Do not refresh the global watchdog with bootstrap data.
            if source != "predicted":
                self._last_status = "waiting_predicted_global_summary"
                return

            self._global_summary_received = now
            sweep_time_s = _as_float(fields.get("sweep_time_s"))
            if violation and sweep_time_s is not None:
                self._global_safety_deadline_abs = (
                    now.to_sec() +
                    max(0.0, sweep_time_s - self._safety_margin_s))
            else:
                self._global_safety_deadline_abs = None

            if not self._execution_ready:
                self._last_violation = bool(violation)
                self._violation_streak = 0
                self._safe_streak = 0
                self._last_status = "global_waiting_execution_release"
                return
            if self._episode_rearm_enabled and not self._episode_armed:
                self._last_violation = bool(violation)
                self._violation_streak = 0
                self._safe_streak = 0
                self._last_status = "global_disarmed_waiting_replan"
                return

            self._process_violation_locked(
                bool(violation), now, "global_set")

    def _verification_hold_locked(self, now):
        if not self._enabled:
            return False, "disabled"

        if self._use_global_summary:
            if not self._execution_ready:
                return False, "waiting_execution_release"
            if self._episode_rearm_enabled and not self._episode_armed:
                return False, "replan_hold_owned_by_controller"
            if self._triggered:
                return False, "triggered"
            if self._global_fresh_locked(now):
                return False, "global_verified"

            # Allow one timeout window after T0/rearm for the first predicted
            # selector result before entering fail-safe verification hold.
            if self._global_summary_received is None:
                anchor = self._execution_ready_received
                if anchor is not None:
                    age = (now - anchor).to_sec()
                    if 0.0 <= age <= self._input_timeout:
                        return False, "waiting_first_global_summary"
            return True, "global_summary_stale_verification_hold"

        if not self._active:
            return False, "inactive"
        if self._physical_deadline_abs is None:
            return False, "missing_physical_deadline"
        if now.to_sec() < self._physical_deadline_abs:
            return False, "predeadline"
        if self._triggered:
            return False, "triggered"

        fresh = self._active_fresh_locked(now) and self._audit_fresh_locked(now)
        if not fresh:
            return True, "postdeadline_verification_hold"
        return False, "postdeadline_verified"

    def _publish_locked(self, now):
        # The MPC's q_vis timing always receives the steering target's physical
        # deadline unchanged. Global safety timing never overwrites it.
        if self._physical_deadline_abs is not None:
            self._deadline_pub.publish(
                Float64(data=self._physical_deadline_abs))

        verification_hold, routing_status = self._verification_hold_locked(now)
        self._verification_hold_pub.publish(
            Bool(data=verification_hold))

        if self._use_global_summary:
            source_age = float("nan")
            if self._global_summary_received is not None:
                source_age = (
                    now - self._global_summary_received).to_sec()
            active = self._execution_ready and (
                not self._episode_rearm_enabled or self._episode_armed)
        else:
            source_age = float("nan")
            if self._violation_received is not None:
                source_age = (now - self._violation_received).to_sec()
            active = self._active

        physical_remaining = float("nan")
        if self._physical_deadline_abs is not None:
            physical_remaining = (
                self._physical_deadline_abs - now.to_sec())
        safety_remaining = float("nan")
        safety_deadline = self._current_safety_deadline_locked()
        if safety_deadline is not None:
            safety_remaining = safety_deadline - now.to_sec()

        text = " ".join([
            "mode=%s" % (
                "global_set" if self._use_global_summary
                else "selected_target"),
            "enabled=%d" % int(self._enabled),
            "active=%d" % int(active),
            "episode_armed=%d" % int(self._episode_armed),
            "triggered=%d" % int(self._triggered),
            "recovery_clear=%d" % int(self._recovery_clear),
            "verification_hold=%d" % int(verification_hold),
            "trigger_count_total=%d" % self._trigger_count_total,
            "clear_count_total=%d" % self._clear_count_total,
            "status=%s" % self._last_status,
            "routing=%s" % routing_status,
            "violation=%d" % int(self._last_violation),
            "violation_streak=%d" % self._violation_streak,
            "max_violation_streak=%d" % self._max_violation_streak,
            "required_streak=%d" % self._consecutive_required,
            "safe_streak=%d" % self._safe_streak,
            "max_safe_streak=%d" % self._max_safe_streak,
            "required_safe_streak=%d" % self._consecutive_safe_required,
            "source_age=%s" % (
                "nan" if not math.isfinite(source_age)
                else "%.6f" % source_age),
            "trajectory_source=%s" % self._global_trajectory_source,
            "candidate_count=%d" % self._global_candidate_count,
            "primary_count=%d" % self._global_primary_count,
            "secondary_count=%d" % self._global_secondary_count,
            "violation_count=%d" % self._global_violation_count,
            "primary_violation_count=%d" % (
                self._global_primary_violation_count),
            "secondary_violation_count=%d" % (
                self._global_secondary_violation_count),
            "selected_group=%s" % self._global_selected_group,
            "physical_deadline_remaining=%s" % (
                "nan" if not math.isfinite(physical_remaining)
                else "%.6f" % physical_remaining),
            "safety_deadline_remaining=%s" % (
                "nan" if not math.isfinite(safety_remaining)
                else "%.6f" % safety_remaining),
            "last_trigger_lead_s=%s" % (
                "nan" if not math.isfinite(self._last_trigger_lead_s)
                else "%.6f" % self._last_trigger_lead_s),
        ])
        self._summary_pub.publish(String(data=text))
        rospy.loginfo_throttle(
            0.5, "[predicted_vbc_recovery_guard] %s", text)

    def _timer_cb(self, _event):
        now = rospy.Time.now()
        with self._lock:
            if self._use_global_summary:
                if (self._enabled and self._execution_ready and
                        (not self._episode_rearm_enabled or self._episode_armed) and
                        self._global_summary_received is not None and
                        not self._global_fresh_locked(now)):
                    self._violation_streak = 0
                    self._safe_streak = 0
                    self._set_recovery_clear_locked(False)
                    if not self._triggered:
                        self._last_status = "stale_global_summary"
            else:
                if (self._enabled and not self._triggered and self._active and
                        self._violation_received is not None and
                        not self._audit_fresh_locked(now)):
                    self._violation_streak = 0
                    self._last_status = "stale_audit"
            self._publish_locked(now)


def main():
    rospy.init_node("predicted_vbc_recovery_guard")
    try:
        PredictedVbcRecoveryGuard()
    except Exception as exc:
        rospy.logfatal(
            "[predicted_vbc_recovery_guard] initialization failed: %s", exc)
        raise
    rospy.spin()


if __name__ == "__main__":
    main()
