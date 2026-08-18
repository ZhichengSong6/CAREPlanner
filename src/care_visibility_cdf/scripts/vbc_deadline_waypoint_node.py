#!/usr/bin/env python3
"""Explicit VisCDF deadline-waypoint generator for CAREPlanner.

This is the runtime counterpart of vbc_deadline_projection_smoke_node.py.
It does NOT actuate the robot.  For one frozen VBC target x* it:

  1) receives the frozen nominal sweep time t_sweep;
  2) defines t_deadline = max(0, t_sweep - safety_margin);
  3) samples q_nom(t_deadline) from the one-shot nominal task trajectory;
  4) explicitly iterates the learned zero-level projection, re-evaluating
     f(x*,q) and df/dq after every update;
  5) once a sign-changing projection segment is found, refines the learned root
     on that segment with bisection to avoid the Newton oscillation observed in
     the smoke experiment;
  6) takes a small positive-gradient crossing/ascent step from the learned root;
  7) publishes q_vis together with an ABSOLUTE ROS deadline timestamp.

Only learned f and df/dq are used to construct q_vis.  Analytic FOV oracles are
optional diagnostics only and never influence projection, root refinement,
ascent, active gating, or the published waypoint.

The downstream VelocityQPMPC tracks q_vis at the rolling horizon index whose
absolute time is at-or-before the published deadline.
"""

from __future__ import annotations

import json
import math
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rospkg
import rospy
import torch
from care_confidence_map.srv import QueryConfidence, QueryConfidenceRequest
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool, Float32, Float64, Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory


PACKAGE_DIR = Path(rospkg.RosPack().get_path("care_visibility_cdf")).resolve()
SCRIPT_DIR = PACKAGE_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_direct_vs_projection_ascent import (  # noqa: E402
    DEFAULT_JOINT_NAMES,
    DEFAULT_SENSOR_FRAMES,
    PinocchioFOVOracle,
    build_model_from_checkpoint,
    learned_ascent_step,
    learned_projection_step,
    model_value_and_grad_q,
    oracle_visibility_g,
    torch_load_checkpoint,
)


DEFAULT_CHECKPOINT = (
    PACKAGE_DIR / "checkpoints" / "exp1_yiming_k500_fov_signed" / "final.pt"
)
DEFAULT_URDF = Path(rospkg.RosPack().get_path("arm_description")) / "urdf" / "Arm.urdf"
DEFAULT_OUTPUT_ROOT = (
    Path(rospkg.RosPack().get_path("egocentric_arm_planner")).resolve().parents[1]
    / "outputs"
    / "phase_b2_vbc_waypoint"
)
DEFAULT_Q_MIN = [-3.14, -2.30, -3.14, -2.65, -3.14, -3.14, -1.20]
DEFAULT_Q_MAX = [3.14, 2.30, 3.14, 2.65, 3.14, 3.14, 1.20]


def _finite(values: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(values)))


def _to_float_list(value, default: Sequence[float]) -> List[float]:
    if value is None:
        return [float(v) for v in default]
    if isinstance(value, str):
        return [float(v) for v in value.replace(",", " ").split() if v]
    return [float(v) for v in value]


def _vector_msg(values: np.ndarray) -> Float64MultiArray:
    msg = Float64MultiArray()
    msg.data = [float(v) for v in np.asarray(values, dtype=np.float64).reshape(-1)]
    return msg


def _fmt(values: Sequence[float], precision: int = 5) -> str:
    return "[" + ", ".join(f"{float(v):+.{precision}f}" for v in values) + "]"


class VbcDeadlineWaypointNode:
    def __init__(self) -> None:
        self._lock = threading.Lock()

        self.target_topic = str(rospy.get_param(
            "~target_topic", "/care_planner/active_sensing/target_point"))
        self.frozen_sweep_topic = str(rospy.get_param(
            "~frozen_sweep_time_topic",
            "/care_planner/active_sensing/frozen_sweep_time_s"))
        self.trajectory_topic = str(rospy.get_param(
            "~trajectory_topic", "/care_planner/task_trajectory"))
        self.confidence_query_service = str(rospy.get_param(
            "~confidence_query_service", "/care_planner/confidence_map/query"))

        self.active_topic = str(rospy.get_param(
            "~active_topic", "/care_planner/active_sensing/visibility_waypoint_active"))
        self.waypoint_topic = str(rospy.get_param(
            "~waypoint_topic", "/care_planner/active_sensing/visibility_waypoint_q"))
        self.zero_topic = str(rospy.get_param(
            "~zero_topic", "/care_planner/active_sensing/visibility_zero_q"))
        self.deadline_topic = str(rospy.get_param(
            "~deadline_topic", "/care_planner/active_sensing/visibility_waypoint_deadline"))
        self.summary_topic = str(rospy.get_param(
            "~summary_topic", "/care_planner/active_sensing/visibility_waypoint_summary"))

        self.rate = float(rospy.get_param("~rate", 20.0))
        self.safety_margin_s = float(rospy.get_param("~safety_margin_s", 0.30))
        self.seen_threshold = float(rospy.get_param("~seen_threshold", 0.50))
        self.target_change_tolerance = float(rospy.get_param("~target_change_tolerance", 1e-4))
        self.service_wait_timeout = float(rospy.get_param("~service_wait_timeout", 0.01))

        # Strictly match the successful explicit smoke defaults.
        self.projection_iters = int(rospy.get_param("~projection_iters", 10))
        self.projection_damping = float(rospy.get_param("~projection_damping", 0.5))
        self.projection_epsilon_f = float(rospy.get_param("~projection_epsilon_f", 0.03))
        self.projection_max_step_norm = float(rospy.get_param("~projection_max_step_norm", 0.25))
        self.root_refine_iters = int(rospy.get_param("~root_refine_iters", 12))
        self.root_tolerance_f = float(rospy.get_param("~root_tolerance_f", 0.002))
        self.ascent_steps = int(rospy.get_param("~ascent_steps", 1))
        self.ascent_step_size = float(rospy.get_param("~ascent_step_size", 0.05))
        self.ascent_max_step_norm = float(rospy.get_param("~ascent_max_step_norm", 0.25))
        self.math_eps = float(rospy.get_param("~math_eps", 1e-8))
        self.clamp_q = bool(rospy.get_param("~clamp_q", True))

        self.device_name = str(rospy.get_param("~device", "cpu"))
        self.checkpoint_path = Path(rospy.get_param(
            "~checkpoint", str(DEFAULT_CHECKPOINT))).expanduser().resolve()
        self.urdf_path = Path(rospy.get_param(
            "~urdf", str(DEFAULT_URDF))).expanduser().resolve()
        self.output_root = Path(rospy.get_param(
            "~output_root", str(DEFAULT_OUTPUT_ROOT))).expanduser().resolve()
        self.q_min_list = _to_float_list(rospy.get_param("~q_min", DEFAULT_Q_MIN), DEFAULT_Q_MIN)
        self.q_max_list = _to_float_list(rospy.get_param("~q_max", DEFAULT_Q_MAX), DEFAULT_Q_MAX)

        self.enable_oracle_diagnostics = bool(rospy.get_param("~enable_oracle_diagnostics", True))
        self.nominal_hfov = float(rospy.get_param("~nominal_horizontal_fov_deg", 55.0))
        self.nominal_vfov = float(rospy.get_param("~nominal_vertical_fov_deg", 72.0))
        self.nominal_z_min = float(rospy.get_param("~nominal_z_min", 0.15))
        self.nominal_z_max = float(rospy.get_param("~nominal_z_max", 0.75))

        self._validate_params()
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(self.device_name)
        rospy.loginfo("[vbc_waypoint] loading frozen VisCDF model...")
        checkpoint = torch_load_checkpoint(str(self.checkpoint_path), self.device)
        self.model, self.checkpoint_args = build_model_from_checkpoint(checkpoint, self.device)
        self.q_min = torch.tensor(self.q_min_list, device=self.device, dtype=torch.float32)
        self.q_max = torch.tensor(self.q_max_list, device=self.device, dtype=torch.float32)
        self._initialize_optional_oracles()
        rospy.logwarn("[vbc_waypoint] frozen model READY: %s", self.checkpoint_path)

        self._target: Optional[PointStamped] = None
        self._target_xyz: Optional[np.ndarray] = None
        self._trajectory: Optional[JointTrajectory] = None
        self._trajectory_received: Optional[rospy.Time] = None
        self._sweep_time_s: Optional[float] = None
        self._generation_key = None
        self._q_zero: Optional[np.ndarray] = None
        self._q_vis: Optional[np.ndarray] = None
        self._deadline_abs_s: Optional[float] = None
        self._deadline_from_start_s: Optional[float] = None
        self._seen_latched = False
        self._generation_success = False
        self._last_confidence = math.nan
        self._last_current_visibility = math.nan
        self._summary = "waiting_inputs"

        self.confidence_client = rospy.ServiceProxy(
            self.confidence_query_service, QueryConfidence, persistent=False)

        self.active_pub = rospy.Publisher(self.active_topic, Bool, queue_size=1)
        self.waypoint_pub = rospy.Publisher(self.waypoint_topic, Float64MultiArray, queue_size=1, latch=True)
        self.zero_pub = rospy.Publisher(self.zero_topic, Float64MultiArray, queue_size=1, latch=True)
        self.deadline_pub = rospy.Publisher(self.deadline_topic, Float64, queue_size=1, latch=True)
        self.summary_pub = rospy.Publisher(self.summary_topic, String, queue_size=1, latch=True)

        self.target_sub = rospy.Subscriber(
            self.target_topic, PointStamped, self._target_callback, queue_size=1)
        self.sweep_sub = rospy.Subscriber(
            self.frozen_sweep_topic, Float32, self._sweep_callback, queue_size=1)
        self.trajectory_sub = rospy.Subscriber(
            self.trajectory_topic, JointTrajectory, self._trajectory_callback, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate), self._timer_callback)
        rospy.logwarn(
            "[vbc_waypoint] explicit projection + root refinement + %d ascent step(s); diagnostic-only oracle=%d",
            self.ascent_steps, int(self.enable_oracle_diagnostics))

    def _validate_params(self) -> None:
        if self.device_name not in ("cpu", "cuda"):
            raise ValueError("~device must be cpu or cuda")
        if self.device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if not self.checkpoint_path.is_file() or not self.urdf_path.is_file():
            raise FileNotFoundError("checkpoint or URDF does not exist")
        if self.rate <= 0.0 or self.service_wait_timeout <= 0.0:
            raise ValueError("invalid rate/service timeout")
        if self.safety_margin_s < 0.0:
            raise ValueError("safety margin must be non-negative")
        if not 0.0 <= self.seen_threshold <= 1.0:
            raise ValueError("seen threshold must be in [0,1]")
        if self.projection_iters <= 0 or self.root_refine_iters < 0 or self.ascent_steps <= 0:
            raise ValueError("iteration counts are invalid")
        if not 0.0 < self.projection_damping <= 1.0:
            raise ValueError("projection damping must be in (0,1]")
        if self.projection_epsilon_f <= 0.0 or self.root_tolerance_f <= 0.0:
            raise ValueError("projection/root tolerances must be positive")
        if self.projection_max_step_norm <= 0.0:
            raise ValueError("projection max-step norm must be positive")
        if self.ascent_step_size <= 0.0 or self.ascent_max_step_norm <= 0.0:
            raise ValueError("ascent step/max must be positive")
        if len(self.q_min_list) != 7 or len(self.q_max_list) != 7:
            raise ValueError("q limits must contain seven values")

    def _initialize_optional_oracles(self) -> None:
        self.conservative_oracle = None
        self.nominal_oracle = None
        if not self.enable_oracle_diagnostics:
            return
        hfov = float(self.checkpoint_args.get("horizontal_fov_deg", 50.0))
        vfov = float(self.checkpoint_args.get("vertical_fov_deg", 66.0))
        z_min = float(self.checkpoint_args.get("z_min", 0.20))
        z_max = float(self.checkpoint_args.get("z_max", 0.70))
        delta = float(self.checkpoint_args.get("delta", 0.01))
        self.conservative_oracle = PinocchioFOVOracle(
            urdf_path=str(self.urdf_path), joint_names=list(DEFAULT_JOINT_NAMES),
            sensor_frames=list(DEFAULT_SENSOR_FRAMES), horizontal_fov_deg=hfov,
            vertical_fov_deg=vfov, z_min=z_min, z_max=z_max, delta=delta,
            base_frame="base_link")
        self.nominal_oracle = PinocchioFOVOracle(
            urdf_path=str(self.urdf_path), joint_names=list(DEFAULT_JOINT_NAMES),
            sensor_frames=list(DEFAULT_SENSOR_FRAMES), horizontal_fov_deg=self.nominal_hfov,
            vertical_fov_deg=self.nominal_vfov, z_min=self.nominal_z_min,
            z_max=self.nominal_z_max, delta=0.0, base_frame="base_link")

    @staticmethod
    def _target_array(msg: PointStamped) -> np.ndarray:
        return np.asarray([msg.point.x, msg.point.y, msg.point.z], dtype=np.float64)

    def _target_callback(self, msg: PointStamped) -> None:
        if msg is None:
            return
        xyz = self._target_array(msg)
        if not _finite(xyz):
            return
        with self._lock:
            changed = self._target_xyz is None or np.linalg.norm(xyz - self._target_xyz) > self.target_change_tolerance
            self._target = msg
            self._target_xyz = xyz
            if changed:
                self._seen_latched = False
                self._generation_success = False
                self._generation_key = None
                self._q_zero = None
                self._q_vis = None
                self._deadline_abs_s = None
                self._summary = "new_target_waiting_generation"
                rospy.logwarn("[vbc_waypoint] new frozen target x*=%s", _fmt(xyz, 6))

    def _sweep_callback(self, msg: Float32) -> None:
        if msg is None or not math.isfinite(float(msg.data)) or float(msg.data) < 0.0:
            return
        with self._lock:
            self._sweep_time_s = float(msg.data)

    def _trajectory_callback(self, msg: JointTrajectory) -> None:
        if msg is None or not msg.points:
            return
        with self._lock:
            self._trajectory = msg
            self._trajectory_received = rospy.Time.now()

    @staticmethod
    def _trajectory_mapping(trajectory: JointTrajectory) -> List[int]:
        index = {name: i for i, name in enumerate(trajectory.joint_names)}
        missing = [name for name in DEFAULT_JOINT_NAMES if name not in index]
        if missing:
            raise ValueError("task trajectory missing joints: " + ",".join(missing))
        return [index[name] for name in DEFAULT_JOINT_NAMES]

    @classmethod
    def _sample_trajectory(cls, trajectory: JointTrajectory, t: float) -> np.ndarray:
        mapping = cls._trajectory_mapping(trajectory)
        times = np.asarray([p.time_from_start.to_sec() for p in trajectory.points], dtype=np.float64)
        if not _finite(times) or np.any(np.diff(times) < -1e-9):
            raise ValueError("task trajectory has invalid timestamps")

        def q_of(point) -> np.ndarray:
            if len(point.positions) < len(trajectory.joint_names):
                raise ValueError("task trajectory point has incomplete positions")
            return np.asarray([point.positions[i] for i in mapping], dtype=np.float64)

        if t <= times[0]:
            return q_of(trajectory.points[0])
        if t >= times[-1]:
            return q_of(trajectory.points[-1])
        hi = int(np.searchsorted(times, t, side="right"))
        lo = hi - 1
        alpha = float((t - times[lo]) / max(times[hi] - times[lo], 1e-12))
        return (1.0 - alpha) * q_of(trajectory.points[lo]) + alpha * q_of(trajectory.points[hi])

    def _clamp(self, q: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        if not self.clamp_q:
            return q, False
        out = torch.maximum(torch.minimum(q, self.q_max[None, :]), self.q_min[None, :])
        return out, bool(torch.any(torch.abs(out - q) > 1e-9).item())

    def _learned(self, x: torch.Tensor, q: torch.Tensor):
        f, grad, _ = model_value_and_grad_q(x, q, self.model)
        return float(f[0].item()), grad

    def _oracle_diag(self, x: torch.Tensor, q: torch.Tensor) -> Dict[str, Optional[float]]:
        if self.conservative_oracle is None or self.nominal_oracle is None:
            return {"g_conservative": None, "g_nominal_fov": None}
        with torch.no_grad():
            gc = oracle_visibility_g(x, q, self.conservative_oracle)
            gn = oracle_visibility_g(x, q, self.nominal_oracle)
        return {
            "g_conservative": float(gc[0].item()),
            "g_nominal_fov": float(gn[0].item()),
        }

    def _refine_root_bisection(
        self,
        x: torch.Tensor,
        qa: torch.Tensor,
        fa: float,
        qb: torch.Tensor,
        fb: float,
    ) -> Tuple[torch.Tensor, float, List[Dict[str, object]]]:
        if fa == 0.0:
            return qa.clone(), fa, []
        if fb == 0.0:
            return qb.clone(), fb, []
        if fa * fb > 0.0:
            raise ValueError("root refinement requires a sign-changing bracket")

        a, b = qa.clone(), qb.clone()
        f_a, f_b = float(fa), float(fb)
        best_q = a.clone() if abs(f_a) <= abs(f_b) else b.clone()
        best_f = f_a if abs(f_a) <= abs(f_b) else f_b
        history: List[Dict[str, object]] = []

        for iteration in range(1, self.root_refine_iters + 1):
            mid = 0.5 * (a + b)
            f_mid, _ = self._learned(x, mid)
            history.append({"iter": iteration, "f": f_mid})
            if abs(f_mid) < abs(best_f):
                best_q, best_f = mid.clone(), f_mid
            if abs(f_mid) <= self.root_tolerance_f:
                return mid.detach(), f_mid, history
            if f_a * f_mid <= 0.0:
                b, f_b = mid, f_mid
            else:
                a, f_a = mid, f_mid
        return best_q.detach(), float(best_f), history

    def _generate_waypoint(
        self,
        target: PointStamped,
        trajectory: JointTrajectory,
        sweep_time_s: float,
        trajectory_received: Optional[rospy.Time],
    ) -> Dict[str, object]:
        frame = str(target.header.frame_id).strip().lstrip("/")
        if frame not in ("", "base_link"):
            raise ValueError(f"target must be in base_link, got {target.header.frame_id!r}")
        x_np = self._target_array(target)
        deadline_from_start = max(0.0, sweep_time_s - self.safety_margin_s)
        q_nom_np = self._sample_trajectory(trajectory, deadline_from_start)
        if not _finite(q_nom_np):
            raise ValueError("non-finite nominal deadline configuration")

        # task_trajectory is stamped when the persistent nominal command is
        # created. Fall back to local receipt time only if an old producer gives
        # a zero stamp.
        if trajectory.header.stamp.to_sec() > 0.0:
            epoch_s = trajectory.header.stamp.to_sec()
            epoch_source = "trajectory_header"
        elif trajectory_received is not None:
            epoch_s = trajectory_received.to_sec()
            epoch_source = "trajectory_receive_time_fallback"
        else:
            epoch_s = rospy.Time.now().to_sec()
            epoch_source = "now_fallback"
        deadline_abs_s = epoch_s + deadline_from_start

        x = torch.tensor(x_np.reshape(1, 3), device=self.device, dtype=torch.float32)
        q0 = torch.tensor(q_nom_np.reshape(1, 7), device=self.device, dtype=torch.float32)
        q0, initial_clamped = self._clamp(q0)
        q = q0.clone()

        f_current, grad_current = self._learned(x, q)
        projection_history: List[Dict[str, object]] = [{
            "iter": 0, "f": f_current,
            "grad_norm": float(torch.linalg.vector_norm(grad_current[0]).item()),
            "q": q[0].detach().cpu().numpy().astype(float).tolist(),
        }]
        q_zero: Optional[torch.Tensor] = q.clone() if abs(f_current) <= self.projection_epsilon_f else None
        f_zero = f_current if q_zero is not None else math.nan
        root_history: List[Dict[str, object]] = []
        root_source = "initial_inside_tolerance" if q_zero is not None else "none"

        for iteration in range(1, self.projection_iters + 1):
            if q_zero is not None:
                break
            f_tensor, grad_tensor, _ = model_value_and_grad_q(x, q, self.model)
            q_next, diag = learned_projection_step(
                q=q, f=f_tensor, grad=grad_tensor,
                damping=self.projection_damping,
                max_step_norm=self.projection_max_step_norm,
                eps=self.math_eps)
            q_next, joint_clamped = self._clamp(q_next)
            f_next, grad_next = self._learned(x, q_next)
            projection_history.append({
                "iter": iteration,
                "f": f_next,
                "grad_norm": float(torch.linalg.vector_norm(grad_next[0]).item()),
                "raw_step_norm": float(diag["raw_step_norm"][0].item()),
                "applied_step_norm": float(diag["applied_step_norm"][0].item()),
                "algorithm_step_clipped": bool(diag["clipped"][0].item()),
                "joint_limit_clamped": joint_clamped,
                "q": q_next[0].detach().cpu().numpy().astype(float).tolist(),
            })

            # Preferred termination: explicitly capture the first learned sign
            # crossing and refine its root instead of letting Newton oscillate.
            if f_current * f_next <= 0.0 and f_current != f_next:
                q_zero, f_zero, root_history = self._refine_root_bisection(
                    x, q, f_current, q_next, f_next)
                root_source = "sign_crossing_bisection"
                break

            if abs(f_next) <= self.projection_epsilon_f:
                q_zero, f_zero = q_next.detach(), f_next
                root_source = "projection_tolerance"
                break

            q = q_next.detach()
            f_current = f_next

        if q_zero is None:
            raise RuntimeError(
                f"explicit projection failed after {self.projection_iters} iterations; last f={f_current:+.6f}")

        q_vis = q_zero.clone()
        ascent_history: List[Dict[str, object]] = []
        for step in range(1, self.ascent_steps + 1):
            _, grad_tensor, _ = model_value_and_grad_q(x, q_vis, self.model)
            q_next, diag = learned_ascent_step(
                q=q_vis, grad=grad_tensor,
                step_size=self.ascent_step_size,
                max_step_norm=self.ascent_max_step_norm,
                eps=self.math_eps)
            q_next, joint_clamped = self._clamp(q_next)
            q_vis = q_next.detach()
            f_vis, grad_vis = self._learned(x, q_vis)
            ascent_history.append({
                "step": step,
                "f": f_vis,
                "grad_norm": float(torch.linalg.vector_norm(grad_vis[0]).item()),
                "applied_step_norm": float(diag["applied_step_norm"][0].item()),
                "joint_limit_clamped": joint_clamped,
                "q": q_vis[0].detach().cpu().numpy().astype(float).tolist(),
            })

        f_initial, _ = self._learned(x, q0)
        f_final, _ = self._learned(x, q_vis)
        initial_diag = self._oracle_diag(x, q0)
        zero_diag = self._oracle_diag(x, q_zero)
        final_diag = self._oracle_diag(x, q_vis)

        return {
            "target_xyz": x_np.tolist(),
            "nominal_sweep_time_s": sweep_time_s,
            "safety_margin_s": self.safety_margin_s,
            "deadline_from_start_s": deadline_from_start,
            "deadline_absolute_ros_s": deadline_abs_s,
            "deadline_epoch_source": epoch_source,
            "q_deadline_nominal": q0[0].detach().cpu().numpy().astype(float).tolist(),
            "q_deadline_initial_clamped": initial_clamped,
            "initial_f": f_initial,
            "initial_oracle_diagnostic": initial_diag,
            "projection_root_source": root_source,
            "projection_zero_f": float(f_zero),
            "q_zero": q_zero[0].detach().cpu().numpy().astype(float).tolist(),
            "zero_oracle_diagnostic": zero_diag,
            "q_vis": q_vis[0].detach().cpu().numpy().astype(float).tolist(),
            "final_f": f_final,
            "final_oracle_diagnostic": final_diag,
            "distance_qzero_from_nominal": float(torch.linalg.vector_norm(q_zero - q0).item()),
            "distance_qvis_from_nominal": float(torch.linalg.vector_norm(q_vis - q0).item()),
            "projection_history": projection_history,
            "root_refinement_history": root_history,
            "ascent_history": ascent_history,
            "config": {
                "projection_iters": self.projection_iters,
                "projection_damping": self.projection_damping,
                "projection_epsilon_f": self.projection_epsilon_f,
                "projection_max_step_norm": self.projection_max_step_norm,
                "root_refine_iters": self.root_refine_iters,
                "root_tolerance_f": self.root_tolerance_f,
                "ascent_steps": self.ascent_steps,
                "ascent_step_size": self.ascent_step_size,
                "ascent_max_step_norm": self.ascent_max_step_norm,
            },
        }

    def _query_confidence(self, target: PointStamped):
        try:
            rospy.wait_for_service(self.confidence_query_service, timeout=self.service_wait_timeout)
            req = QueryConfidenceRequest(); req.points = [target.point]
            res = self.confidence_client(req)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logwarn_throttle(1.0, "[vbc_waypoint] confidence query unavailable: %s", exc)
            return None
        if len(res.confidence) != 1 or len(res.current_visibility) != 1 or len(res.inside_map) != 1:
            return None
        return float(res.confidence[0]), float(res.current_visibility[0]), bool(res.inside_map[0])

    def _maybe_generate(self) -> None:
        with self._lock:
            target = self._target
            target_xyz = None if self._target_xyz is None else self._target_xyz.copy()
            trajectory = self._trajectory
            trajectory_received = self._trajectory_received
            sweep = self._sweep_time_s
            old_key = self._generation_key

        if target is None or target_xyz is None or trajectory is None or sweep is None:
            return
        key = (
            tuple(np.round(target_xyz, 7).tolist()),
            round(float(sweep), 6),
            round(float(trajectory.header.stamp.to_sec()), 6),
        )
        if old_key == key:
            return

        try:
            result = self._generate_waypoint(target, trajectory, sweep, trajectory_received)
        except Exception as exc:
            with self._lock:
                self._generation_key = key
                self._generation_success = False
                self._summary = "generation_failed:" + str(exc).replace(" ", "_")
            rospy.logerr("[vbc_waypoint] generation failed: %s", exc)
            return

        q_zero = np.asarray(result["q_zero"], dtype=np.float64)
        q_vis = np.asarray(result["q_vis"], dtype=np.float64)
        deadline_abs = float(result["deadline_absolute_ros_s"])
        deadline_from_start = float(result["deadline_from_start_s"])

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_root / f"vbc_visibility_waypoint_{stamp}.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=True))

        with self._lock:
            self._generation_key = key
            self._q_zero = q_zero
            self._q_vis = q_vis
            self._deadline_abs_s = deadline_abs
            self._deadline_from_start_s = deadline_from_start
            self._generation_success = True
            self._summary = "ready"

        self.zero_pub.publish(_vector_msg(q_zero))
        self.waypoint_pub.publish(_vector_msg(q_vis))
        dmsg = Float64(); dmsg.data = deadline_abs; self.deadline_pub.publish(dmsg)

        rospy.logwarn("[vbc_waypoint] VISIBILITY WAYPOINT READY")
        rospy.logwarn("[vbc_waypoint] x*=%s sweep=%.6f deadline=%.6f s q_vis=%s",
                      _fmt(result["target_xyz"], 6), sweep, deadline_from_start, _fmt(q_vis, 5))
        rospy.logwarn("[vbc_waypoint] f: initial=%+.6f zero=%+.6f final=%+.6f root=%s",
                      result["initial_f"], result["projection_zero_f"], result["final_f"],
                      result["projection_root_source"])
        if self.enable_oracle_diagnostics:
            gd = result["final_oracle_diagnostic"]
            rospy.logwarn("[vbc_waypoint] DIAGNOSTIC ONLY final g_cons=%+.6f g_nominal=%+.6f",
                          gd["g_conservative"], gd["g_nominal_fov"])
        rospy.logwarn("[vbc_waypoint] saved generation trace: %s", path)

    def _publish_state(self) -> None:
        with self._lock:
            target = self._target
            generation_success = self._generation_success
            q_vis = None if self._q_vis is None else self._q_vis.copy()
            deadline_abs = self._deadline_abs_s
            deadline_from_start = self._deadline_from_start_s
            seen = self._seen_latched
            summary_reason = self._summary

        confidence = math.nan
        current_visibility = math.nan
        inside = False
        if target is not None:
            response = self._query_confidence(target)
            if response is not None:
                confidence, current_visibility, inside = response
                if inside and confidence >= self.seen_threshold and not seen:
                    with self._lock:
                        self._seen_latched = True
                    seen = True
                    rospy.logwarn(
                        "[vbc_waypoint] target CONFIRMED SEEN -> waypoint objective OFF: confidence=%.4f",
                        confidence)

        active = bool(generation_success and q_vis is not None and deadline_abs is not None and not seen)
        amsg = Bool(); amsg.data = active; self.active_pub.publish(amsg)
        if q_vis is not None:
            self.waypoint_pub.publish(_vector_msg(q_vis))
        if deadline_abs is not None:
            dmsg = Float64(); dmsg.data = float(deadline_abs); self.deadline_pub.publish(dmsg)

        now = rospy.Time.now().to_sec()
        remaining = math.nan if deadline_abs is None else deadline_abs - now
        msg = String()
        msg.data = (
            f"active={int(active)} seen={int(seen)} ready={int(generation_success)} "
            f"confidence={confidence:.4f} current_visibility={current_visibility:.4f} "
            f"inside_map={int(inside)} deadline_from_start={deadline_from_start if deadline_from_start is not None else math.nan:.6f} "
            f"deadline_remaining={remaining:.6f} reason={summary_reason}"
        )
        self.summary_pub.publish(msg)
        rospy.loginfo_throttle(0.5, "[vbc_waypoint] %s", msg.data)

    def _timer_callback(self, _event) -> None:
        self._maybe_generate()
        self._publish_state()


def main() -> None:
    rospy.init_node("vbc_deadline_waypoint")
    try:
        VbcDeadlineWaypointNode()
    except Exception as exc:
        rospy.logfatal("[vbc_waypoint] initialization failed: %s", exc)
        raise
    rospy.spin()


if __name__ == "__main__":
    main()
