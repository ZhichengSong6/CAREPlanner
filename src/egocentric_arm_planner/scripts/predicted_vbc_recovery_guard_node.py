#!/usr/bin/env python3

import math
import threading

import rospy
from std_msgs.msg import Bool, Float64, String


class PredictedVbcRecoveryGuard:
    """C4.2 recovery supervisor driven by predicted-trajectory VBC audit.

    Keep the logic intentionally small:
      * before the physical deadline: pass it through unchanged;
      * after the physical deadline with a fresh audit: keep soft steering alive;
      * after N consecutive violation audits: force the effective deadline past
        so the validated Recovery controller takes over;
      * if the audit is temporarily unavailable while the target is still
        active: publish verification_hold=true. The MPC only decelerates until
        a fresh audit returns. This is not a new persistent Recovery state.

    Streaks are updated only by new auditor messages, so N=2 means two distinct
    MPC-prediction audits.
    """

    def __init__(self):
        self._lock = threading.RLock()

        self._enabled = bool(rospy.get_param("~enabled", True))
        self._rate = float(rospy.get_param("~rate", 20.0))
        self._input_timeout = float(rospy.get_param("~input_timeout", 0.25))
        self._consecutive_required = int(
            rospy.get_param("~consecutive_violations_required", 2)
        )
        self._force_past_s = float(rospy.get_param("~force_past_s", 0.01))
        self._post_deadline_hold_future_s = float(
            rospy.get_param("~post_deadline_hold_future_s", 0.025)
        )

        self._active_topic = str(rospy.get_param(
            "~active_topic",
            "/care_planner/active_sensing/visibility_waypoint_active"))
        self._violation_topic = str(rospy.get_param(
            "~violation_topic",
            "/care_planner/execution/predicted_vbc_violation"))
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
        self._summary_topic = str(rospy.get_param(
            "~summary_topic",
            "/care_planner/execution/predicted_vbc_recovery_summary"))

        if self._rate <= 0.0:
            raise ValueError("~rate must be > 0")
        if self._input_timeout <= 0.0:
            raise ValueError("~input_timeout must be > 0")
        if self._consecutive_required < 1:
            raise ValueError("~consecutive_violations_required must be >= 1")
        if self._force_past_s <= 0.0:
            raise ValueError("~force_past_s must be > 0")
        if self._post_deadline_hold_future_s <= 0.0:
            raise ValueError("~post_deadline_hold_future_s must be > 0")

        # The effective deadline is republished at guard rate. Keep it far
        # enough ahead to survive one full publication period plus jitter;
        # otherwise a 20 Hz MPC can observe the short 25 ms hold as already
        # expired and accidentally enter legacy deadline Recovery.
        self._post_deadline_hold_future_s = max(
            self._post_deadline_hold_future_s, 1.5 / self._rate)

        self._active = False
        self._active_received = None
        self._physical_deadline_abs = None
        self._violation_received = None
        self._last_violation = False

        self._violation_streak = 0
        self._max_violation_streak = 0
        self._triggered = False
        self._trigger_count_total = 0
        self._trigger_time_abs = None
        self._last_trigger_lead_s = float("nan")
        self._last_status = "waiting"

        self._deadline_pub = rospy.Publisher(
            self._effective_deadline_topic, Float64, queue_size=1)
        self._verification_hold_pub = rospy.Publisher(
            self._verification_hold_topic, Bool, queue_size=1, latch=True)
        self._triggered_pub = rospy.Publisher(
            self._triggered_topic, Bool, queue_size=1, latch=True)
        self._summary_pub = rospy.Publisher(
            self._summary_topic, String, queue_size=1, latch=True)

        rospy.Subscriber(
            self._active_topic, Bool, self._active_cb, queue_size=1)
        rospy.Subscriber(
            self._violation_topic, Bool, self._violation_cb, queue_size=1)
        rospy.Subscriber(
            self._physical_deadline_topic, Float64, self._deadline_cb, queue_size=1)

        self._triggered_pub.publish(Bool(data=False))
        self._verification_hold_pub.publish(Bool(data=False))
        self._timer = rospy.Timer(rospy.Duration(1.0 / self._rate), self._timer_cb)

        rospy.logwarn(
            "[predicted_vbc_recovery_guard] C4.2 enabled=%d consecutive=%d "
            "timeout=%.3fs hold_future=%.3fs force_past=%.3fs",
            int(self._enabled), self._consecutive_required, self._input_timeout,
            self._post_deadline_hold_future_s, self._force_past_s)

    def _reset_decision_locked(self):
        self._violation_streak = 0
        self._max_violation_streak = 0
        self._triggered = False
        self._trigger_time_abs = None
        self._last_violation = False
        self._triggered_pub.publish(Bool(data=False))
        self._verification_hold_pub.publish(Bool(data=False))

    def _active_cb(self, msg):
        if msg is None:
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
        if msg is None or not math.isfinite(float(msg.data)) or float(msg.data) <= 0.0:
            rospy.logwarn_throttle(
                1.0, "[predicted_vbc_recovery_guard] ignoring invalid physical deadline")
            return
        with self._lock:
            self._physical_deadline_abs = float(msg.data)

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

    def _violation_cb(self, msg):
        if msg is None:
            return
        now = rospy.Time.now()
        with self._lock:
            self._violation_received = now
            self._last_violation = bool(msg.data)

            if not self._enabled:
                self._last_status = "disabled_passthrough"
                return
            if not self._active or not self._active_fresh_locked(now):
                self._violation_streak = 0
                self._last_status = "inactive_or_stale_active"
                return
            if self._triggered:
                self._last_status = "triggered"
                return

            if bool(msg.data):
                self._violation_streak += 1
                self._max_violation_streak = max(
                    self._max_violation_streak, self._violation_streak)
                self._last_status = "predicted_vbc_violation_streak"
            else:
                self._violation_streak = 0
                self._last_status = "predicted_vbc_safe"

            if self._violation_streak >= self._consecutive_required:
                self._triggered = True
                self._trigger_time_abs = now.to_sec()
                self._trigger_count_total += 1
                if self._physical_deadline_abs is not None:
                    self._last_trigger_lead_s = (
                        self._physical_deadline_abs - self._trigger_time_abs)
                else:
                    self._last_trigger_lead_s = float("nan")
                self._last_status = "predicted_vbc_recovery_triggered"
                self._triggered_pub.publish(Bool(data=True))
                rospy.logerr(
                    "[predicted_vbc_recovery_guard] PREDICTED VBC RECOVERY TRIGGER: "
                    "streak=%d physical_deadline_lead=%s",
                    self._violation_streak,
                    "nan" if not math.isfinite(self._last_trigger_lead_s)
                    else "%.3fs" % self._last_trigger_lead_s)

    def _routing_locked(self, now):
        now_s = now.to_sec()
        if self._physical_deadline_abs is None:
            return None, False, "missing_physical_deadline"

        physical = float(self._physical_deadline_abs)
        if not self._enabled:
            return physical, False, "disabled_passthrough"
        if self._triggered:
            return min(physical, now_s - self._force_past_s), False, "trigger_forced_past"
        if physical > now_s:
            return physical, False, "predeadline_passthrough"

        # Target already seen/inactive: never turn missing audits into Recovery.
        if not self._active:
            return now_s + self._post_deadline_hold_future_s, False, "postdeadline_inactive"

        fresh = self._active_fresh_locked(now) and self._audit_fresh_locked(now)
        if not fresh:
            # Lightweight fail-safe: suppress the stale nominal deadline and ask
            # the MPC to decelerate until a fresh C4 verdict arrives.
            return now_s + self._post_deadline_hold_future_s, True, "postdeadline_verification_hold"

        return now_s + self._post_deadline_hold_future_s, False, "postdeadline_c4_verified"

    def _publish_locked(self, now):
        effective, verification_hold, routing_status = self._routing_locked(now)
        if effective is not None and math.isfinite(effective):
            self._deadline_pub.publish(Float64(data=effective))
        self._verification_hold_pub.publish(Bool(data=bool(verification_hold)))

        violation_age = float("nan")
        if self._violation_received is not None:
            violation_age = (now - self._violation_received).to_sec()
        physical_remaining = float("nan")
        if self._physical_deadline_abs is not None:
            physical_remaining = self._physical_deadline_abs - now.to_sec()
        effective_remaining = float("nan")
        if effective is not None:
            effective_remaining = effective - now.to_sec()

        text = " ".join([
            "enabled=%d" % int(self._enabled),
            "active=%d" % int(self._active),
            "triggered=%d" % int(self._triggered),
            "verification_hold=%d" % int(verification_hold),
            "trigger_count_total=%d" % self._trigger_count_total,
            "status=%s" % self._last_status,
            "routing=%s" % routing_status,
            "violation=%d" % int(self._last_violation),
            "violation_streak=%d" % self._violation_streak,
            "max_violation_streak=%d" % self._max_violation_streak,
            "required_streak=%d" % self._consecutive_required,
            "violation_age=%s" % (
                "nan" if not math.isfinite(violation_age) else "%.6f" % violation_age),
            "physical_deadline_remaining=%s" % (
                "nan" if not math.isfinite(physical_remaining)
                else "%.6f" % physical_remaining),
            "effective_deadline_remaining=%s" % (
                "nan" if not math.isfinite(effective_remaining)
                else "%.6f" % effective_remaining),
            "last_trigger_lead_s=%s" % (
                "nan" if not math.isfinite(self._last_trigger_lead_s)
                else "%.6f" % self._last_trigger_lead_s),
        ])
        self._summary_pub.publish(String(data=text))
        rospy.loginfo_throttle(0.5, "[predicted_vbc_recovery_guard] %s", text)

    def _timer_cb(self, _event):
        now = rospy.Time.now()
        with self._lock:
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
