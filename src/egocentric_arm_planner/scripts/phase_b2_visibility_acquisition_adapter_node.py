#!/usr/bin/env python3
"""Confidence-gated visibility steering adapter for CAREPlanner Phase B.2.

Two steering modes are supported on exactly the same observer/model data:

  normalized_ascent (legacy B.2-v2)
      r_k = grad f_k / ||grad f_k||

  projection_ascent (B.2-v3)
      if f_k < -epsilon_f:
          r_k = -damping * f_k * grad f_k / (||grad f_k||^2 + eps)
          ||r_k|| <= projection_max_step_norm
      else:
          r_k = ascent_step_size * grad f_k / ||grad f_k||
          ||r_k|| <= ascent_max_step_norm

For projection_ascent, q_target,k = q_observer,k + r_k is also published for
diagnostics. The C++ condensed QP consumes r=[r_1;...;r_K] as the gradient of
a first-order sequential linearization of a soft configuration-target cost.
Thus the magnitude information from zero-level projection is retained instead
of reducing every far-outside state to a unit direction.

The observer remains non-actuating. This adapter publishes only soft steering
information and an acquisition-active gate. Actual sensing completion is always
determined by the confidence map and latched for the frozen target.
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

        self.target_topic = str(rospy.get_param(
            "~target_topic", "/care_planner/active_sensing/target_point"))
        self.observer_active_topic = str(rospy.get_param(
            "~observer_active_topic", "/ncdf_horizon_observer/active"))
        self.horizon_q_topic = str(rospy.get_param(
            "~horizon_q_topic", "/ncdf_horizon_observer/horizon_q"))
        self.learned_values_topic = str(rospy.get_param(
            "~learned_values_topic", "/ncdf_horizon_observer/learned_values"))
        self.raw_gradient_topic = str(rospy.get_param(
            "~raw_gradient_topic", "/ncdf_horizon_observer/gradient_q"))
        self.confidence_query_service = str(rospy.get_param(
            "~confidence_query_service", "/care_planner/confidence_map/query"))

        self.acquisition_active_topic = str(rospy.get_param(
            "~acquisition_active_topic",
            "/care_planner/active_sensing/acquisition_active"))
        self.steering_topic = str(rospy.get_param(
            "~steering_topic",
            "/care_planner/active_sensing/configuration_target_residual_q"))
        self.configuration_target_topic = str(rospy.get_param(
            "~configuration_target_topic",
            "/care_planner/active_sensing/configuration_target_q"))
        # Kept as a diagnostic output so the old raw-gradient geometry can still
        # be inspected without affecting the v3 control input.
        self.normalized_gradient_topic = str(rospy.get_param(
            "~normalized_gradient_topic",
            "/care_planner/active_sensing/normalized_gradient_q"))

        self.rate = float(rospy.get_param("~rate", 20.0))
        self.target_timeout = float(rospy.get_param("~target_timeout", 0.25))
        self.observer_timeout = float(rospy.get_param("~observer_timeout", 0.15))
        self.service_wait_timeout = float(rospy.get_param("~service_wait_timeout", 0.01))
        self.seen_threshold = float(rospy.get_param("~seen_threshold", 0.50))
        self.target_change_tolerance = float(rospy.get_param("~target_change_tolerance", 1e-4))
        self.normalization_eps = float(rospy.get_param("~normalization_eps", 1e-8))
        self.dof = int(rospy.get_param("~dof", 7))

        self.steering_mode = str(rospy.get_param(
            "~steering_mode", "projection_ascent")).strip().lower()
        self.projection_epsilon_f = float(rospy.get_param("~projection_epsilon_f", 0.03))
        self.projection_damping = float(rospy.get_param("~projection_damping", 0.5))
        self.projection_max_step_norm = float(rospy.get_param("~projection_max_step_norm", 0.25))
        self.ascent_step_size = float(rospy.get_param("~ascent_step_size", 0.05))
        self.ascent_max_step_norm = float(rospy.get_param("~ascent_max_step_norm", 0.25))
        self.legacy_ascent_scale = float(rospy.get_param("~legacy_ascent_scale", 1.0))

        self._validate_params()

        self._latest_target: Optional[PointStamped] = None
        self._target_received: Optional[rospy.Time] = None
        self._target_xyz: Optional[np.ndarray] = None
        self._observer_active = False
        self._observer_active_received: Optional[rospy.Time] = None

        self._latest_horizon_q: Optional[np.ndarray] = None
        self._horizon_q_received: Optional[rospy.Time] = None
        self._latest_f: Optional[np.ndarray] = None
        self._f_received: Optional[rospy.Time] = None
        self._latest_grad: Optional[np.ndarray] = None
        self._grad_received: Optional[rospy.Time] = None
        self._gradient_layout = None

        self._seen_latched = False
        self._last_confidence = math.nan
        self._last_current_visibility = math.nan
        self._last_inside_map = False
        self._last_raw_grad_min = math.nan
        self._last_raw_grad_max = math.nan
        self._last_norm_grad_min = math.nan
        self._last_norm_grad_max = math.nan
        self._last_residual_min = math.nan
        self._last_residual_max = math.nan
        self._last_f_min = math.nan
        self._last_f_max = math.nan
        self._last_project_count = 0
        self._last_ascent_count = 0
        self._last_stage = "none"
        self._sequence = 0

        self._confidence_client = rospy.ServiceProxy(
            self.confidence_query_service, QueryConfidence, persistent=False)

        self.active_pub = rospy.Publisher(self.acquisition_active_topic, Bool, queue_size=1)
        self.steering_pub = rospy.Publisher(self.steering_topic, Float64MultiArray, queue_size=1)
        self.configuration_target_pub = rospy.Publisher(
            self.configuration_target_topic, Float64MultiArray, queue_size=1)
        self.normalized_gradient_pub = rospy.Publisher(
            self.normalized_gradient_topic, Float64MultiArray, queue_size=1)
        self.target_seen_pub = rospy.Publisher("~target_seen", Bool, queue_size=1)
        self.target_confidence_pub = rospy.Publisher("~target_confidence", Float32, queue_size=1)
        self.target_current_visibility_pub = rospy.Publisher(
            "~target_current_visibility", Float32, queue_size=1)
        self.summary_pub = rospy.Publisher("~summary", String, queue_size=1)

        self.target_sub = rospy.Subscriber(
            self.target_topic, PointStamped, self._target_callback, queue_size=1)
        self.observer_active_sub = rospy.Subscriber(
            self.observer_active_topic, Bool, self._observer_active_callback, queue_size=1)
        self.horizon_q_sub = rospy.Subscriber(
            self.horizon_q_topic, Float64MultiArray, self._horizon_q_callback, queue_size=1)
        self.learned_values_sub = rospy.Subscriber(
            self.learned_values_topic, Float64MultiArray, self._learned_values_callback, queue_size=1)
        self.gradient_sub = rospy.Subscriber(
            self.raw_gradient_topic, Float64MultiArray, self._gradient_callback, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration.from_sec(1.0 / self.rate), self._timer_callback)

        rospy.logwarn(
            "[phase_b2_adapter] ENABLED steering_mode=%s projection(eps_f=%.3f,damping=%.3f,max=%.3f) ascent(step=%.3f,max=%.3f)",
            self.steering_mode,
            self.projection_epsilon_f,
            self.projection_damping,
            self.projection_max_step_norm,
            self.ascent_step_size,
            self.ascent_max_step_norm,
        )
        rospy.loginfo(
            "[phase_b2_adapter] observer q=%s f=%s grad=%s -> steering=%s target_q=%s",
            self.horizon_q_topic,
            self.learned_values_topic,
            self.raw_gradient_topic,
            self.steering_topic,
            self.configuration_target_topic,
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
        if self.steering_mode not in ("normalized_ascent", "projection_ascent"):
            raise ValueError("~steering_mode must be normalized_ascent or projection_ascent")
        if self.projection_epsilon_f <= 0.0:
            raise ValueError("~projection_epsilon_f must be positive")
        if not 0.0 < self.projection_damping <= 1.0:
            raise ValueError("~projection_damping must be in (0,1]")
        if self.projection_max_step_norm <= 0.0:
            raise ValueError("~projection_max_step_norm must be positive")
        if self.ascent_step_size <= 0.0 or self.ascent_max_step_norm <= 0.0:
            raise ValueError("ascent step/max must be positive")
        if self.legacy_ascent_scale <= 0.0:
            raise ValueError("~legacy_ascent_scale must be positive")

    @staticmethod
    def _target_array(msg: PointStamped) -> np.ndarray:
        return np.asarray([msg.point.x, msg.point.y, msg.point.z], dtype=np.float64)

    @staticmethod
    def _matrix_from_msg(msg: Float64MultiArray, dof: int) -> Optional[np.ndarray]:
        if msg is None or not msg.data:
            return None
        values = np.asarray(msg.data, dtype=np.float64)
        if values.size % dof != 0 or not np.all(np.isfinite(values)):
            return None
        return values.reshape(-1, dof)

    @staticmethod
    def _matrix_msg(values: np.ndarray, source_layout=None) -> Float64MultiArray:
        out = Float64MultiArray()
        if source_layout is not None:
            out.layout = source_layout
        out.data = [float(v) for v in np.asarray(values, dtype=np.float64).reshape(-1)]
        return out

    def _target_callback(self, msg: PointStamped) -> None:
        if msg is None:
            return
        xyz = self._target_array(msg)
        if not np.all(np.isfinite(xyz)):
            return
        with self._lock:
            changed = self._target_xyz is None or np.linalg.norm(xyz - self._target_xyz) > self.target_change_tolerance
            if changed:
                self._seen_latched = False
                self._last_confidence = math.nan
                self._last_current_visibility = math.nan
                self._last_inside_map = False
                rospy.loginfo(
                    "[phase_b2_adapter] new target -> reset seen latch: [%.6f, %.6f, %.6f]",
                    xyz[0], xyz[1], xyz[2])
            self._latest_target = msg
            self._target_xyz = xyz
            self._target_received = rospy.Time.now()

    def _observer_active_callback(self, msg: Bool) -> None:
        if msg is None:
            return
        with self._lock:
            self._observer_active = bool(msg.data)
            self._observer_active_received = rospy.Time.now()

    def _horizon_q_callback(self, msg: Float64MultiArray) -> None:
        matrix = self._matrix_from_msg(msg, self.dof)
        if matrix is None:
            rospy.logwarn_throttle(1.0, "[phase_b2_adapter] invalid horizon_q")
            return
        with self._lock:
            self._latest_horizon_q = matrix
            self._horizon_q_received = rospy.Time.now()

    def _learned_values_callback(self, msg: Float64MultiArray) -> None:
        if msg is None or not msg.data:
            return
        values = np.asarray(msg.data, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(values)):
            rospy.logwarn_throttle(1.0, "[phase_b2_adapter] invalid learned_values")
            return
        with self._lock:
            self._latest_f = values
            self._f_received = rospy.Time.now()

    def _gradient_callback(self, msg: Float64MultiArray) -> None:
        matrix = self._matrix_from_msg(msg, self.dof)
        if matrix is None:
            rospy.logwarn_throttle(1.0, "[phase_b2_adapter] invalid gradient_q")
            return
        raw_norm = np.linalg.norm(matrix, axis=1)
        normalized = np.zeros_like(matrix)
        valid = raw_norm > self.normalization_eps
        normalized[valid] = matrix[valid] / raw_norm[valid, None]
        norm_after = np.linalg.norm(normalized, axis=1)
        self.normalized_gradient_pub.publish(self._matrix_msg(normalized, msg.layout))
        with self._lock:
            self._latest_grad = matrix
            self._grad_received = rospy.Time.now()
            self._gradient_layout = msg.layout
            self._last_raw_grad_min = float(np.min(raw_norm))
            self._last_raw_grad_max = float(np.max(raw_norm))
            self._last_norm_grad_min = float(np.min(norm_after))
            self._last_norm_grad_max = float(np.max(norm_after))

    def _snapshot(self):
        with self._lock:
            return (
                self._latest_target,
                self._target_received,
                self._observer_active,
                self._observer_active_received,
                self._seen_latched,
                None if self._latest_horizon_q is None else self._latest_horizon_q.copy(),
                self._horizon_q_received,
                None if self._latest_f is None else self._latest_f.copy(),
                self._f_received,
                None if self._latest_grad is None else self._latest_grad.copy(),
                self._grad_received,
                self._gradient_layout,
            )

    def _query_confidence(self, target: PointStamped):
        try:
            rospy.wait_for_service(self.confidence_query_service, timeout=self.service_wait_timeout)
            request = QueryConfidenceRequest()
            request.points = [target.point]
            return self._confidence_client(request)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logwarn_throttle(1.0, "[phase_b2_adapter] confidence query unavailable: %s", exc)
            return None

    @staticmethod
    def _clip_rows(vectors: np.ndarray, max_norm: float, eps: float) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1)
        scale = np.ones_like(norms)
        mask = norms > max_norm
        scale[mask] = max_norm / np.maximum(norms[mask], eps)
        return vectors * scale[:, None]

    def _build_steering(
        self,
        horizon_q: np.ndarray,
        f_values: np.ndarray,
        grad: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, str, int, int]:
        if horizon_q.shape != grad.shape or horizon_q.shape[0] != f_values.size:
            raise ValueError(
                f"observer shape mismatch q={horizon_q.shape} f={f_values.shape} grad={grad.shape}")

        grad_norm = np.linalg.norm(grad, axis=1)
        valid = grad_norm > self.normalization_eps
        unit = np.zeros_like(grad)
        unit[valid] = grad[valid] / grad_norm[valid, None]

        if self.steering_mode == "normalized_ascent":
            residual = self.legacy_ascent_scale * unit
            project_count = 0
            ascent_count = int(np.count_nonzero(valid))
            stage = "legacy_ascent"
        else:
            residual = np.zeros_like(grad)
            project_mask = valid & (f_values < -self.projection_epsilon_f)
            ascent_mask = valid & ~project_mask

            if np.any(project_mask):
                denom = np.maximum(grad_norm[project_mask] ** 2, self.normalization_eps)
                residual[project_mask] = (
                    -self.projection_damping
                    * f_values[project_mask, None]
                    * grad[project_mask]
                    / denom[:, None]
                )
                residual[project_mask] = self._clip_rows(
                    residual[project_mask], self.projection_max_step_norm, self.normalization_eps)

            if np.any(ascent_mask):
                residual[ascent_mask] = self.ascent_step_size * unit[ascent_mask]
                residual[ascent_mask] = self._clip_rows(
                    residual[ascent_mask], self.ascent_max_step_norm, self.normalization_eps)

            project_count = int(np.count_nonzero(project_mask))
            ascent_count = int(np.count_nonzero(ascent_mask))
            if project_count > 0 and ascent_count > 0:
                stage = "mixed"
            elif project_count > 0:
                stage = "project"
            elif ascent_count > 0:
                stage = "ascent"
            else:
                stage = "zero_gradient"

        q_target = horizon_q + residual
        return residual, q_target, stage, project_count, ascent_count

    def _publish_state(
        self,
        active: bool,
        seen: bool,
        confidence: float,
        current_visibility: float,
        inside_map: bool,
        reason: str,
    ) -> None:
        active_msg = Bool(); active_msg.data = bool(active); self.active_pub.publish(active_msg)
        seen_msg = Bool(); seen_msg.data = bool(seen); self.target_seen_pub.publish(seen_msg)
        c_msg = Float32(); c_msg.data = float(confidence) if math.isfinite(confidence) else math.nan; self.target_confidence_pub.publish(c_msg)
        v_msg = Float32(); v_msg.data = float(current_visibility) if math.isfinite(current_visibility) else math.nan; self.target_current_visibility_pub.publish(v_msg)

        with self._lock:
            raw_min, raw_max = self._last_raw_grad_min, self._last_raw_grad_max
            norm_min, norm_max = self._last_norm_grad_min, self._last_norm_grad_max
            res_min, res_max = self._last_residual_min, self._last_residual_max
            f_min, f_max = self._last_f_min, self._last_f_max
            project_count, ascent_count = self._last_project_count, self._last_ascent_count
            stage = self._last_stage
            xyz = None if self._target_xyz is None else self._target_xyz.copy()

        target_text = "none" if xyz is None else "[{:.3f},{:.3f},{:.3f}]".format(*xyz)
        summary_text = (
            f"seq={self._sequence} active={int(active)} seen={int(seen)} "
            f"inside_map={int(inside_map)} confidence={confidence:.4f} "
            f"current_visibility={current_visibility:.4f} target={target_text} "
            f"steering_mode={self.steering_mode} stage={stage} "
            f"project_count={project_count} ascent_count={ascent_count} "
            f"f=[{f_min:.4e},{f_max:.4e}] "
            f"raw_grad_norm=[{raw_min:.4e},{raw_max:.4e}] "
            f"normalized_grad_norm=[{norm_min:.4e},{norm_max:.4e}] "
            f"residual_norm=[{res_min:.4e},{res_max:.4e}] reason={reason}"
        )
        summary = String(); summary.data = summary_text; self.summary_pub.publish(summary)
        rospy.loginfo_throttle(0.5, "[phase_b2_adapter] %s", summary_text)

    def _timer_callback(self, _event) -> None:
        self._sequence += 1
        now = rospy.Time.now()
        (
            target, target_received, observer_active, observer_received, seen_latched,
            horizon_q, horizon_received, f_values, f_received, grad, grad_received, layout,
        ) = self._snapshot()

        if target is None or target_received is None:
            self._publish_state(False, False, math.nan, math.nan, False, "waiting_target"); return
        if (now - target_received).to_sec() > self.target_timeout:
            self._publish_state(False, seen_latched, math.nan, math.nan, False, "stale_target"); return
        if observer_received is None:
            self._publish_state(False, seen_latched, math.nan, math.nan, False, "waiting_observer"); return
        if (now - observer_received).to_sec() > self.observer_timeout:
            self._publish_state(False, seen_latched, math.nan, math.nan, False, "stale_observer"); return
        if not observer_active:
            self._publish_state(False, seen_latched, math.nan, math.nan, False, "observer_inactive"); return
        if horizon_q is None or f_values is None or grad is None:
            self._publish_state(False, seen_latched, math.nan, math.nan, False, "waiting_observer_fields"); return
        field_times = [horizon_received, f_received, grad_received]
        if any(t is None for t in field_times):
            self._publish_state(False, seen_latched, math.nan, math.nan, False, "waiting_observer_fields"); return
        field_age = max((now - t).to_sec() for t in field_times if t is not None)
        if field_age > self.observer_timeout:
            self._publish_state(False, seen_latched, math.nan, math.nan, False, "stale_observer_fields"); return

        try:
            residual, q_target, stage, project_count, ascent_count = self._build_steering(
                horizon_q, f_values, grad)
        except ValueError as exc:
            rospy.logwarn_throttle(1.0, "[phase_b2_adapter] %s", exc)
            self._publish_state(False, seen_latched, math.nan, math.nan, False, "shape_mismatch"); return

        residual_norm = np.linalg.norm(residual, axis=1)
        with self._lock:
            self._last_residual_min = float(np.min(residual_norm))
            self._last_residual_max = float(np.max(residual_norm))
            self._last_f_min = float(np.min(f_values))
            self._last_f_max = float(np.max(f_values))
            self._last_project_count = project_count
            self._last_ascent_count = ascent_count
            self._last_stage = stage

        # Publish steering before active=True so the C++ MPC never observes a
        # newly active gate without a matching fresh steering vector.
        self.steering_pub.publish(self._matrix_msg(residual, layout))
        self.configuration_target_pub.publish(self._matrix_msg(q_target, layout))

        response = self._query_confidence(target)
        if response is None:
            self._publish_state(False, seen_latched, math.nan, math.nan, False, "confidence_unavailable"); return
        if len(response.confidence) != 1 or len(response.current_visibility) != 1 or len(response.inside_map) != 1:
            self._publish_state(False, seen_latched, math.nan, math.nan, False, "malformed_confidence"); return

        confidence = float(response.confidence[0])
        current_visibility = float(response.current_visibility[0])
        inside_map = bool(response.inside_map[0])
        if not math.isfinite(confidence) or not math.isfinite(current_visibility):
            self._publish_state(False, seen_latched, confidence, current_visibility, inside_map, "invalid_confidence"); return
        if not inside_map:
            self._publish_state(False, seen_latched, confidence, current_visibility, False, "target_outside_map"); return

        seen_now = confidence >= self.seen_threshold
        if seen_now and not seen_latched:
            rospy.loginfo(
                "[phase_b2_adapter] target CONFIRMED SEEN: confidence=%.4f >= %.4f",
                confidence, self.seen_threshold)
            with self._lock:
                self._seen_latched = True
            seen_latched = True

        with self._lock:
            self._last_confidence = confidence
            self._last_current_visibility = current_visibility
            self._last_inside_map = inside_map

        if seen_latched:
            self._publish_state(False, True, confidence, current_visibility, inside_map, "seen_latched"); return

        self._publish_state(
            True, False, confidence, current_visibility, inside_map,
            "acquiring_" + stage)


def main() -> None:
    rospy.init_node("phase_b2_visibility_acquisition_adapter")
    try:
        PhaseB2VisibilityAcquisitionAdapter()
    except Exception as exc:
        rospy.logfatal("[phase_b2_adapter] initialization failed: %s", exc)
        raise
    rospy.spin()


if __name__ == "__main__":
    main()
