#!/usr/bin/env python3
"""Phase B.2-v2 confidence-gated normalized visibility acquisition adapter.

This node deliberately sits between the frozen NCDF observer and the C++
velocity QP-MPC.

The NCDF observer remains a faithful diagnostic module and continues to publish
raw gradients df/dq.  This adapter gives those gradients planner semantics:

  1) normalize each horizon gradient so the neural-network gradient magnitude
     does not directly set the strength of the MPC visibility objective;
  2) query the confidence map for the active target x*;
  3) publish acquisition_active=True only while the target has not yet been
     confirmed seen by the confidence map and the NCDF observer is healthy;
  4) latch "seen" for the current target.  Confidence decay therefore cannot
     reactivate the same acquisition task during a trial.  The latch resets only
     when the target coordinates change by more than target_change_tolerance.

No actuation command is published here.  The output topics are consumed by the
existing C++ VelocityQPMPC linear visibility objective.
"""

from __future__ import annotations

import math
import threading
from typing import Optional, Tuple

import numpy as np
import rospy
from care_confidence_map.srv import QueryConfidence, QueryConfidenceRequest
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool, Float32, Float64MultiArray, String


class PhaseB2VisibilityAcquisitionAdapter:
    def __init__(self) -> None:
        self._lock = threading.Lock()

        self.target_topic = str(
            rospy.get_param(
                "~target_topic", "/care_planner/active_sensing/target_point"
            )
        )
        self.observer_active_topic = str(
            rospy.get_param(
                "~observer_active_topic", "/ncdf_horizon_observer/active"
            )
        )
        self.raw_gradient_topic = str(
            rospy.get_param(
                "~raw_gradient_topic", "/ncdf_horizon_observer/gradient_q"
            )
        )
        self.confidence_query_service = str(
            rospy.get_param(
                "~confidence_query_service", "/care_planner/confidence_map/query"
            )
        )

        self.acquisition_active_topic = str(
            rospy.get_param(
                "~acquisition_active_topic",
                "/care_planner/active_sensing/acquisition_active",
            )
        )
        self.normalized_gradient_topic = str(
            rospy.get_param(
                "~normalized_gradient_topic",
                "/care_planner/active_sensing/normalized_gradient_q",
            )
        )

        self.rate = float(rospy.get_param("~rate", 20.0))
        self.target_timeout = float(rospy.get_param("~target_timeout", 0.25))
        self.observer_timeout = float(rospy.get_param("~observer_timeout", 0.15))
        self.service_wait_timeout = float(
            rospy.get_param("~service_wait_timeout", 0.01)
        )
        self.seen_threshold = float(rospy.get_param("~seen_threshold", 0.50))
        self.target_change_tolerance = float(
            rospy.get_param("~target_change_tolerance", 1e-4)
        )
        self.normalization_eps = float(
            rospy.get_param("~normalization_eps", 1e-6)
        )
        self.dof = int(rospy.get_param("~dof", 7))

        self._validate_params()

        self._latest_target: Optional[PointStamped] = None
        self._target_received: Optional[rospy.Time] = None
        self._target_xyz: Optional[np.ndarray] = None
        self._observer_active = False
        self._observer_active_received: Optional[rospy.Time] = None

        self._seen_latched = False
        self._last_confidence = math.nan
        self._last_current_visibility = math.nan
        self._last_inside_map = False
        self._last_raw_grad_min = math.nan
        self._last_raw_grad_max = math.nan
        self._last_norm_grad_min = math.nan
        self._last_norm_grad_max = math.nan
        self._sequence = 0

        self._confidence_client = rospy.ServiceProxy(
            self.confidence_query_service, QueryConfidence, persistent=False
        )

        self.active_pub = rospy.Publisher(
            self.acquisition_active_topic, Bool, queue_size=1
        )
        self.normalized_gradient_pub = rospy.Publisher(
            self.normalized_gradient_topic, Float64MultiArray, queue_size=1
        )
        self.target_seen_pub = rospy.Publisher(
            "~target_seen", Bool, queue_size=1
        )
        self.target_confidence_pub = rospy.Publisher(
            "~target_confidence", Float32, queue_size=1
        )
        self.target_current_visibility_pub = rospy.Publisher(
            "~target_current_visibility", Float32, queue_size=1
        )
        self.summary_pub = rospy.Publisher("~summary", String, queue_size=1)

        self.target_sub = rospy.Subscriber(
            self.target_topic, PointStamped, self._target_callback, queue_size=1
        )
        self.observer_active_sub = rospy.Subscriber(
            self.observer_active_topic,
            Bool,
            self._observer_active_callback,
            queue_size=1,
        )
        self.gradient_sub = rospy.Subscriber(
            self.raw_gradient_topic,
            Float64MultiArray,
            self._gradient_callback,
            queue_size=1,
        )

        self.timer = rospy.Timer(
            rospy.Duration.from_sec(1.0 / self.rate), self._timer_callback
        )

        rospy.logwarn(
            "[phase_b2_v2_adapter] ENABLED: confidence-gated acquisition + per-horizon normalized NCDF gradients."
        )
        rospy.loginfo(
            "[phase_b2_v2_adapter] target=%s raw_gradient=%s observer_active=%s",
            self.target_topic,
            self.raw_gradient_topic,
            self.observer_active_topic,
        )
        rospy.loginfo(
            "[phase_b2_v2_adapter] output active=%s normalized_gradient=%s seen_threshold=%.3f",
            self.acquisition_active_topic,
            self.normalized_gradient_topic,
            self.seen_threshold,
        )

    def _validate_params(self) -> None:
        if self.rate <= 0.0:
            raise ValueError("~rate must be positive")
        if self.target_timeout <= 0.0 or self.observer_timeout <= 0.0:
            raise ValueError("timeouts must be positive")
        if self.service_wait_timeout <= 0.0:
            raise ValueError("~service_wait_timeout must be positive")
        if not 0.0 <= self.seen_threshold <= 1.0:
            raise ValueError("~seen_threshold must be in [0,1]")
        if self.target_change_tolerance < 0.0:
            raise ValueError("~target_change_tolerance must be non-negative")
        if self.normalization_eps <= 0.0:
            raise ValueError("~normalization_eps must be positive")
        if self.dof <= 0:
            raise ValueError("~dof must be positive")

    @staticmethod
    def _target_array(msg: PointStamped) -> np.ndarray:
        return np.asarray(
            [msg.point.x, msg.point.y, msg.point.z], dtype=np.float64
        )

    def _target_callback(self, msg: PointStamped) -> None:
        if msg is None:
            return
        xyz = self._target_array(msg)
        if not np.all(np.isfinite(xyz)):
            rospy.logwarn_throttle(
                1.0, "[phase_b2_v2_adapter] ignoring non-finite target"
            )
            return

        with self._lock:
            changed = (
                self._target_xyz is None
                or np.linalg.norm(xyz - self._target_xyz)
                > self.target_change_tolerance
            )
            if changed:
                self._seen_latched = False
                self._last_confidence = math.nan
                self._last_current_visibility = math.nan
                self._last_inside_map = False
                rospy.loginfo(
                    "[phase_b2_v2_adapter] new target -> reset seen latch: [%.6f, %.6f, %.6f]",
                    xyz[0],
                    xyz[1],
                    xyz[2],
                )
            self._latest_target = msg
            self._target_xyz = xyz
            self._target_received = rospy.Time.now()

    def _observer_active_callback(self, msg: Bool) -> None:
        if msg is None:
            return
        with self._lock:
            self._observer_active = bool(msg.data)
            self._observer_active_received = rospy.Time.now()

    def _gradient_callback(self, msg: Float64MultiArray) -> None:
        if msg is None or not msg.data:
            return

        values = np.asarray(msg.data, dtype=np.float64)
        if values.size % self.dof != 0:
            rospy.logwarn_throttle(
                1.0,
                "[phase_b2_v2_adapter] ignoring raw gradient with %d values; not divisible by dof=%d",
                values.size,
                self.dof,
            )
            return
        if not np.all(np.isfinite(values)):
            rospy.logwarn_throttle(
                1.0, "[phase_b2_v2_adapter] ignoring non-finite raw gradient"
            )
            return

        matrix = values.reshape(-1, self.dof)
        raw_norm = np.linalg.norm(matrix, axis=1)
        normalized = np.zeros_like(matrix)
        valid = raw_norm > self.normalization_eps
        normalized[valid] = matrix[valid] / raw_norm[valid, None]
        norm_after = np.linalg.norm(normalized, axis=1)

        out = Float64MultiArray()
        out.layout = msg.layout
        out.data = [float(v) for v in normalized.reshape(-1)]
        self.normalized_gradient_pub.publish(out)

        with self._lock:
            self._last_raw_grad_min = float(np.min(raw_norm))
            self._last_raw_grad_max = float(np.max(raw_norm))
            self._last_norm_grad_min = float(np.min(norm_after))
            self._last_norm_grad_max = float(np.max(norm_after))

    def _snapshot(
        self,
    ) -> Tuple[
        Optional[PointStamped],
        Optional[rospy.Time],
        bool,
        Optional[rospy.Time],
        bool,
    ]:
        with self._lock:
            return (
                self._latest_target,
                self._target_received,
                self._observer_active,
                self._observer_active_received,
                self._seen_latched,
            )

    def _query_confidence(self, target: PointStamped):
        try:
            rospy.wait_for_service(
                self.confidence_query_service,
                timeout=self.service_wait_timeout,
            )
            request = QueryConfidenceRequest()
            request.points = [target.point]
            response = self._confidence_client(request)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logwarn_throttle(
                1.0,
                "[phase_b2_v2_adapter] confidence query unavailable: %s",
                exc,
            )
            return None

        if (
            len(response.confidence) != 1
            or len(response.current_visibility) != 1
            or len(response.inside_map) != 1
        ):
            rospy.logwarn_throttle(
                1.0,
                "[phase_b2_v2_adapter] malformed confidence query response",
            )
            return None
        return response

    def _publish_state(
        self,
        active: bool,
        seen: bool,
        confidence: float,
        current_visibility: float,
        inside_map: bool,
        reason: str,
    ) -> None:
        active_msg = Bool()
        active_msg.data = bool(active)
        self.active_pub.publish(active_msg)

        seen_msg = Bool()
        seen_msg.data = bool(seen)
        self.target_seen_pub.publish(seen_msg)

        confidence_msg = Float32()
        confidence_msg.data = float(confidence) if math.isfinite(confidence) else math.nan
        self.target_confidence_pub.publish(confidence_msg)

        current_visibility_msg = Float32()
        current_visibility_msg.data = (
            float(current_visibility)
            if math.isfinite(current_visibility)
            else math.nan
        )
        self.target_current_visibility_pub.publish(current_visibility_msg)

        with self._lock:
            raw_min = self._last_raw_grad_min
            raw_max = self._last_raw_grad_max
            norm_min = self._last_norm_grad_min
            norm_max = self._last_norm_grad_max
            xyz = None if self._target_xyz is None else self._target_xyz.copy()

        target_text = (
            "none"
            if xyz is None
            else "[{:.3f},{:.3f},{:.3f}]".format(xyz[0], xyz[1], xyz[2])
        )
        summary_text = (
            f"seq={self._sequence} active={int(active)} seen={int(seen)} "
            f"inside_map={int(inside_map)} confidence={confidence:.4f} "
            f"current_visibility={current_visibility:.4f} target={target_text} "
            f"raw_grad_norm=[{raw_min:.4e},{raw_max:.4e}] "
            f"normalized_grad_norm=[{norm_min:.4e},{norm_max:.4e}] "
            f"reason={reason}"
        )
        summary = String()
        summary.data = summary_text
        self.summary_pub.publish(summary)
        rospy.loginfo_throttle(
            0.5, "[phase_b2_v2_adapter] %s", summary_text
        )

    def _timer_callback(self, _event) -> None:
        self._sequence += 1
        now = rospy.Time.now()
        (
            target,
            target_received,
            observer_active,
            observer_received,
            seen_latched,
        ) = self._snapshot()

        if target is None or target_received is None:
            self._publish_state(False, False, math.nan, math.nan, False, "waiting_target")
            return
        if (now - target_received).to_sec() > self.target_timeout:
            self._publish_state(False, seen_latched, math.nan, math.nan, False, "stale_target")
            return
        if observer_received is None:
            self._publish_state(False, seen_latched, math.nan, math.nan, False, "waiting_observer")
            return
        if (now - observer_received).to_sec() > self.observer_timeout:
            self._publish_state(False, seen_latched, math.nan, math.nan, False, "stale_observer")
            return
        if not observer_active:
            self._publish_state(False, seen_latched, math.nan, math.nan, False, "observer_inactive")
            return

        response = self._query_confidence(target)
        if response is None:
            self._publish_state(False, seen_latched, math.nan, math.nan, False, "confidence_unavailable")
            return

        confidence = float(response.confidence[0])
        current_visibility = float(response.current_visibility[0])
        inside_map = bool(response.inside_map[0])
        if not math.isfinite(confidence) or not math.isfinite(current_visibility):
            self._publish_state(False, seen_latched, confidence, current_visibility, inside_map, "invalid_confidence")
            return
        if not inside_map:
            self._publish_state(False, seen_latched, confidence, current_visibility, False, "target_outside_map")
            return

        seen_now = confidence >= self.seen_threshold
        if seen_now and not seen_latched:
            rospy.loginfo(
                "[phase_b2_v2_adapter] target CONFIRMED SEEN: confidence=%.4f >= %.4f",
                confidence,
                self.seen_threshold,
            )
            with self._lock:
                self._seen_latched = True
            seen_latched = True

        with self._lock:
            self._last_confidence = confidence
            self._last_current_visibility = current_visibility
            self._last_inside_map = inside_map

        if seen_latched:
            self._publish_state(
                False,
                True,
                confidence,
                current_visibility,
                inside_map,
                "seen_latched",
            )
            return

        self._publish_state(
            True,
            False,
            confidence,
            current_visibility,
            inside_map,
            "acquiring",
        )


def main() -> None:
    rospy.init_node("phase_b2_visibility_acquisition_adapter")
    try:
        PhaseB2VisibilityAcquisitionAdapter()
    except Exception as exc:
        rospy.logfatal("[phase_b2_v2_adapter] initialization failed: %s", exc)
        raise
    rospy.spin()


if __name__ == "__main__":
    main()
