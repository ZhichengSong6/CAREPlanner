#!/usr/bin/env python3

import math
import threading

import rospy
from std_msgs.msg import Bool, Float64, Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory


class PredictiveRecoveryGuard:
    """Advance the controller recovery deadline when soft MPC predicts a miss.

    The input deadline remains the physical VBC deadline.  This node publishes
    an *effective controller deadline*: normally it is identical to the physical
    deadline; after a stable predicted miss it is advanced to the present time
    so the existing waypoint MPC enters its already-validated recovery mode on
    the next control cycle.

    Keeping the trigger outside the MPC makes Phase-C3 attribution clean: the
    recovery controller/state machine is unchanged, while the trigger policy is
    independently observable and can later be folded into the MPC if desired.
    """

    def __init__(self):
        self._lock = threading.RLock()

        self._enabled = rospy.get_param("~enabled", True)
        self._rate = float(rospy.get_param("~rate", 20.0))
        self._input_timeout = float(rospy.get_param("~input_timeout", 0.25))
        self._horizon_slack = float(rospy.get_param("~horizon_slack", 0.025))
        self._error_threshold_inf = float(
            rospy.get_param("~prediction_error_threshold_inf", 0.10)
        )
        self._min_improvement_inf = float(
            rospy.get_param("~min_prediction_improvement_inf", 0.002)
        )
        self._consecutive_misses_required = int(
            rospy.get_param("~consecutive_misses_required", 3)
        )
        self._target_q_change_tolerance_inf = float(
            rospy.get_param("~target_q_change_tolerance_inf", 1.0e-3)
        )
        self._target_deadline_change_tolerance = float(
            rospy.get_param("~target_deadline_change_tolerance", 1.0e-3)
        )
        self._force_past_s = float(rospy.get_param("~force_past_s", 0.01))

        self._active_topic = rospy.get_param(
            "~active_topic",
            "/care_planner/active_sensing/visibility_waypoint_active",
        )
        self._q_topic = rospy.get_param(
            "~q_topic",
            "/care_planner/active_sensing/visibility_waypoint_q",
        )
        self._physical_deadline_topic = rospy.get_param(
            "~physical_deadline_topic",
            "/care_planner/active_sensing/visibility_waypoint_deadline_synced",
        )
        self._prediction_topic = rospy.get_param(
            "~prediction_topic",
            "/care_planner/mpc/predicted_trajectory",
        )
        self._effective_deadline_topic = rospy.get_param(
            "~effective_deadline_topic",
            "/care_planner/active_sensing/visibility_waypoint_deadline_effective",
        )
        self._summary_topic = rospy.get_param(
            "~summary_topic",
            "/care_planner/execution/predictive_recovery_summary",
        )
        self._triggered_topic = rospy.get_param(
            "~triggered_topic",
            "/care_planner/execution/predictive_recovery_triggered",
        )

        if self._rate <= 0.0:
            raise ValueError("~rate must be > 0")
        if self._input_timeout <= 0.0:
            raise ValueError("~input_timeout must be > 0")
        if self._horizon_slack < 0.0:
            raise ValueError("~horizon_slack must be >= 0")
        if self._error_threshold_inf < 0.0:
            raise ValueError("~prediction_error_threshold_inf must be >= 0")
        if self._min_improvement_inf < 0.0:
            raise ValueError("~min_prediction_improvement_inf must be >= 0")
        if self._consecutive_misses_required < 1:
            raise ValueError("~consecutive_misses_required must be >= 1")
        if self._force_past_s <= 0.0:
            raise ValueError("~force_past_s must be > 0")

        self._active = False
        self._q_vis = None
        self._physical_deadline_abs = None
        self._prediction = None

        self._active_received = None
        self._q_received = None
        self._deadline_received = None
        self._prediction_received = None

        self._tracked_q_vis = None
        self._tracked_deadline_abs = None
        self._previous_pred_error_inf = None
        self._miss_streak = 0
        self._triggered = False
        self._trigger_time_abs = None
        self._trigger_lead_s = float("nan")

        self._last_status = "waiting"
        self._last_error_inf = float("nan")
        self._last_improvement_inf = float("nan")
        self._last_waypoint_k = -1
        self._last_deadline_remaining = float("nan")

        self._deadline_pub = rospy.Publisher(
            self._effective_deadline_topic, Float64, queue_size=1
        )
        self._summary_pub = rospy.Publisher(
            self._summary_topic, String, queue_size=1, latch=True
        )
        self._triggered_pub = rospy.Publisher(
            self._triggered_topic, Bool, queue_size=1, latch=True
        )

        rospy.Subscriber(self._active_topic, Bool, self._active_cb, queue_size=1)
        rospy.Subscriber(self._q_topic, Float64MultiArray, self._q_cb, queue_size=1)
        rospy.Subscriber(
            self._physical_deadline_topic,
            Float64,
            self._deadline_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            self._prediction_topic,
            JointTrajectory,
            self._prediction_cb,
            queue_size=1,
        )

        self._triggered_pub.publish(Bool(data=False))
        self._timer = rospy.Timer(rospy.Duration(1.0 / self._rate), self._timer_cb)

        rospy.logwarn(
            "[predictive_recovery_guard] enabled=%s error_inf>%.4f, "
            "min_improvement=%.4f, consecutive=%d, horizon_slack=%.3fs",
            self._enabled,
            self._error_threshold_inf,
            self._min_improvement_inf,
            self._consecutive_misses_required,
            self._horizon_slack,
        )
        rospy.loginfo(
            "[predictive_recovery_guard] physical deadline=%s -> effective deadline=%s",
            self._physical_deadline_topic,
            self._effective_deadline_topic,
        )

    @staticmethod
    def _finite_vector(values):
        return bool(values) and all(math.isfinite(float(v)) for v in values)

    @staticmethod
    def _inf_distance(a, b):
        if a is None or b is None or len(a) != len(b) or not a:
            return float("inf")
        return max(abs(float(x) - float(y)) for x, y in zip(a, b))

    def _reset_decision_locked(self, clear_target):
        self._previous_pred_error_inf = None
        self._miss_streak = 0
        self._triggered = False
        self._trigger_time_abs = None
        self._trigger_lead_s = float("nan")
        self._last_error_inf = float("nan")
        self._last_improvement_inf = float("nan")
        self._last_waypoint_k = -1
        if clear_target:
            self._tracked_q_vis = None
            self._tracked_deadline_abs = None
        self._triggered_pub.publish(Bool(data=False))

    def _ensure_target_identity_locked(self):
        if self._q_vis is None or self._physical_deadline_abs is None:
            return

        is_new = self._tracked_q_vis is None or self._tracked_deadline_abs is None
        if not is_new:
            q_delta = self._inf_distance(self._q_vis, self._tracked_q_vis)
            deadline_delta = abs(
                float(self._physical_deadline_abs) - float(self._tracked_deadline_abs)
            )
            is_new = (
                q_delta > self._target_q_change_tolerance_inf
                or deadline_delta > self._target_deadline_change_tolerance
            )

        if is_new:
            self._reset_decision_locked(clear_target=False)
            self._tracked_q_vis = list(self._q_vis)
            self._tracked_deadline_abs = float(self._physical_deadline_abs)
            self._last_status = "new_target"

    def _active_cb(self, msg):
        now = rospy.Time.now()
        with self._lock:
            self._active = bool(msg.data)
            self._active_received = now
            if not self._active:
                self._reset_decision_locked(clear_target=True)
                self._last_status = "inactive"

    def _q_cb(self, msg):
        values = [float(v) for v in msg.data]
        if not self._finite_vector(values):
            rospy.logwarn_throttle(
                1.0, "[predictive_recovery_guard] ignoring malformed/non-finite q_vis"
            )
            return
        with self._lock:
            self._q_vis = values
            self._q_received = rospy.Time.now()
            self._ensure_target_identity_locked()

    def _effective_deadline_locked(self, now_s):
        if self._physical_deadline_abs is None:
            return None
        if self._enabled and self._triggered:
            return min(float(self._physical_deadline_abs), now_s - self._force_past_s)
        return float(self._physical_deadline_abs)

    def _publish_effective_deadline_locked(self, now):
        if self._physical_deadline_abs is None or self._deadline_received is None:
            return
        age = (now - self._deadline_received).to_sec()
        if age < 0.0 or age > self._input_timeout:
            return
        effective = self._effective_deadline_locked(now.to_sec())
        if effective is not None and math.isfinite(effective):
            self._deadline_pub.publish(Float64(data=effective))

    def _deadline_cb(self, msg):
        if not math.isfinite(msg.data) or msg.data <= 0.0:
            rospy.logwarn_throttle(
                1.0, "[predictive_recovery_guard] ignoring invalid physical deadline"
            )
            return
        now = rospy.Time.now()
        with self._lock:
            self._physical_deadline_abs = float(msg.data)
            self._deadline_received = now
            self._ensure_target_identity_locked()
            self._publish_effective_deadline_locked(now)

    def _prediction_cb(self, msg):
        with self._lock:
            self._prediction = msg
            self._prediction_received = rospy.Time.now()

    def _inputs_fresh_locked(self, now):
        stamps = (
            self._active_received,
            self._q_received,
            self._deadline_received,
            self._prediction_received,
        )
        if any(stamp is None for stamp in stamps):
            return False
        for stamp in stamps:
            age = (now - stamp).to_sec()
            if age < 0.0 or age > self._input_timeout:
                return False
        return True

    def _prediction_point_for_deadline_locked(self, deadline_remaining):
        msg = self._prediction
        if msg is None or len(msg.points) < 2:
            return None, -1, 0.0

        horizon = msg.points[-1].time_from_start.to_sec()
        if not math.isfinite(horizon) or horizon <= 0.0:
            return None, -1, horizon
        if deadline_remaining > horizon + self._horizon_slack:
            return None, -1, horizon

        # Match the waypoint MPC convention: choose the latest grid point not
        # after the physical deadline, but never use k=0 for intervention.
        k = 1
        if deadline_remaining > msg.points[1].time_from_start.to_sec():
            for i in range(1, len(msg.points)):
                t = msg.points[i].time_from_start.to_sec()
                if t <= deadline_remaining + 1.0e-9:
                    k = i
                else:
                    break
        k = max(1, min(len(msg.points) - 1, k))
        return msg.points[k], k, horizon

    def _evaluate_locked(self, now):
        now_s = now.to_sec()
        self._last_deadline_remaining = float("nan")

        if not self._enabled:
            self._last_status = "disabled_passthrough"
            return
        if not self._active:
            self._last_status = "inactive"
            return
        if not self._inputs_fresh_locked(now):
            self._last_status = "stale_or_missing_input"
            return
        if self._q_vis is None or not self._finite_vector(self._q_vis):
            self._last_status = "invalid_q_vis"
            return
        if self._physical_deadline_abs is None:
            self._last_status = "missing_deadline"
            return

        self._ensure_target_identity_locked()
        deadline_remaining = float(self._physical_deadline_abs) - now_s
        self._last_deadline_remaining = deadline_remaining

        if self._triggered:
            self._last_status = "triggered"
            return
        if deadline_remaining <= 0.0:
            # Preserve the original deadline-miss recovery as the fallback.
            self._last_status = "physical_deadline_elapsed"
            return

        point, k, horizon = self._prediction_point_for_deadline_locked(
            deadline_remaining
        )
        if point is None:
            self._miss_streak = 0
            self._previous_pred_error_inf = None
            self._last_waypoint_k = -1
            if horizon > 0.0 and deadline_remaining > horizon + self._horizon_slack:
                self._last_status = "deadline_outside_prediction_horizon"
            else:
                self._last_status = "invalid_prediction"
            return

        if len(point.positions) != len(self._q_vis):
            self._miss_streak = 0
            self._previous_pred_error_inf = None
            self._last_status = "prediction_dimension_mismatch"
            return

        pred_q = [float(v) for v in point.positions]
        if not self._finite_vector(pred_q):
            self._miss_streak = 0
            self._previous_pred_error_inf = None
            self._last_status = "nonfinite_prediction"
            return

        error_inf = self._inf_distance(pred_q, self._q_vis)
        improvement = float("nan")
        if self._previous_pred_error_inf is not None:
            improvement = self._previous_pred_error_inf - error_inf

        high_error = error_inf > self._error_threshold_inf
        insufficient_progress = (
            self._previous_pred_error_inf is None
            or improvement < self._min_improvement_inf
        )

        if high_error and insufficient_progress:
            self._miss_streak += 1
            self._last_status = "predicted_miss_streak"
        else:
            self._miss_streak = 0
            if high_error:
                self._last_status = "high_error_but_improving"
            else:
                self._last_status = "predicted_reachable"

        self._previous_pred_error_inf = error_inf
        self._last_error_inf = error_inf
        self._last_improvement_inf = improvement
        self._last_waypoint_k = k

        if self._miss_streak >= self._consecutive_misses_required:
            self._triggered = True
            self._trigger_time_abs = now_s
            self._trigger_lead_s = deadline_remaining
            self._last_status = "predictive_recovery_triggered"
            self._triggered_pub.publish(Bool(data=True))
            rospy.logwarn(
                "[predictive_recovery_guard] PREDICTIVE RECOVERY TRIGGER: "
                "physical_deadline_lead=%.3fs pred_error_inf=%.4f "
                "improvement_inf=%s streak=%d waypoint_k=%d",
                deadline_remaining,
                error_inf,
                "nan" if not math.isfinite(improvement) else "%.4f" % improvement,
                self._miss_streak,
                k,
            )

    def _publish_summary_locked(self, now):
        effective = self._effective_deadline_locked(now.to_sec())
        effective_remaining = float("nan")
        if effective is not None:
            effective_remaining = effective - now.to_sec()

        fields = [
            "enabled=%d" % int(self._enabled),
            "active=%d" % int(self._active),
            "triggered=%d" % int(self._triggered),
            "status=%s" % self._last_status,
            "miss_streak=%d" % self._miss_streak,
            "required_streak=%d" % self._consecutive_misses_required,
            "error_threshold_inf=%.6f" % self._error_threshold_inf,
            "min_improvement_inf=%.6f" % self._min_improvement_inf,
            "pred_error_inf=%s"
            % ("nan" if not math.isfinite(self._last_error_inf) else "%.6f" % self._last_error_inf),
            "pred_improvement_inf=%s"
            % (
                "nan"
                if not math.isfinite(self._last_improvement_inf)
                else "%.6f" % self._last_improvement_inf
            ),
            "physical_deadline_remaining=%s"
            % (
                "nan"
                if not math.isfinite(self._last_deadline_remaining)
                else "%.6f" % self._last_deadline_remaining
            ),
            "effective_deadline_remaining=%s"
            % (
                "nan"
                if not math.isfinite(effective_remaining)
                else "%.6f" % effective_remaining
            ),
            "trigger_lead_s=%s"
            % (
                "nan"
                if not math.isfinite(self._trigger_lead_s)
                else "%.6f" % self._trigger_lead_s
            ),
            "waypoint_k=%d" % self._last_waypoint_k,
        ]
        self._summary_pub.publish(String(data=" ".join(fields)))

    def _timer_cb(self, _event):
        now = rospy.Time.now()
        with self._lock:
            self._evaluate_locked(now)
            self._publish_effective_deadline_locked(now)
            self._publish_summary_locked(now)


def main():
    rospy.init_node("predictive_recovery_guard")
    try:
        PredictiveRecoveryGuard()
    except Exception as exc:
        rospy.logfatal("[predictive_recovery_guard] initialization failed: %s", exc)
        raise
    rospy.spin()


if __name__ == "__main__":
    main()
