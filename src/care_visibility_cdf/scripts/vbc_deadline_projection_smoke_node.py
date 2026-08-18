#!/usr/bin/env python3
"""One-shot explicit projection->ascent smoke test for a live VBC case.

This node is diagnostic-only. It waits for:
  * the frozen active-sensing target x*
  * the nominal task trajectory
  * the VBC selector's selected nominal sweep time

It defines the visibility deadline

    t_deadline = max(0, t_sweep - safety_margin)

samples q_nom(t_deadline), and then runs the *actual iterative* learned
zero-level projection used by the offline evaluator. Every projection iteration
updates q, re-evaluates f and df/dq, and then (after projection) local learned
ascent is run for a configurable number of steps.

For the current CAREPlanner diagnostic we intentionally evaluate multiple
per-update projection caps on exactly the same (x*, q_deadline). The default is
[0.25, 0.50], because previous experiments clearly used 0.25 as the projection
cap while 0.50 also appeared as a *total motion budget*; this smoke resolves the
per-step-cap question on the actual VBC failure case.

No trajectory or command topic is published.
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
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32
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
DEFAULT_URDF = (
    Path(rospkg.RosPack().get_path("arm_description")) / "urdf" / "Arm.urdf"
)
DEFAULT_OUTPUT_ROOT = (
    Path(rospkg.RosPack().get_path("egocentric_arm_planner")).resolve().parents[1]
    / "outputs"
    / "vbc_deadline_projection_smoke"
)

# Same 7-DoF limits used throughout the CAREPlanner VisCDF diagnostics.
DEFAULT_Q_MIN = [-3.14, -2.30, -3.14, -2.65, -3.14, -3.14, -1.20]
DEFAULT_Q_MAX = [3.14, 2.30, 3.14, 2.65, 3.14, 3.14, 1.20]


def _finite(values: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(values)))


def _vec(values: Sequence[float], precision: int = 5) -> str:
    return "[" + ", ".join(f"{float(v):+.{precision}f}" for v in values) + "]"


def _to_float_list(value, default: Sequence[float]) -> List[float]:
    if value is None:
        return [float(v) for v in default]
    if isinstance(value, str):
        return [float(v) for v in value.replace(",", " ").split() if v]
    return [float(v) for v in value]


class VbcDeadlineProjectionSmokeNode:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ran = False

        self.target_topic = str(rospy.get_param(
            "~target_topic", "/care_planner/active_sensing/target_point"))
        self.trajectory_topic = str(rospy.get_param(
            "~trajectory_topic", "/care_planner/task_trajectory"))
        self.sweep_time_topic = str(rospy.get_param(
            "~sweep_time_topic",
            "/care_planner/trajectory_risk/vbc_selected_sweep_time_s"))

        self.safety_margin_s = float(rospy.get_param("~safety_margin_s", 0.30))
        self.projection_iters = int(rospy.get_param("~projection_iters", 10))
        self.projection_damping = float(rospy.get_param("~projection_damping", 0.5))
        self.projection_epsilon_f = float(rospy.get_param("~projection_epsilon_f", 0.03))
        self.projection_max_step_norms = _to_float_list(
            rospy.get_param("~projection_max_step_norms", [0.25, 0.50]), [0.25, 0.50])
        self.ascent_steps = int(rospy.get_param("~ascent_steps", 10))
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

        self.nominal_hfov = float(rospy.get_param("~nominal_horizontal_fov_deg", 55.0))
        self.nominal_vfov = float(rospy.get_param("~nominal_vertical_fov_deg", 72.0))
        self.nominal_z_min = float(rospy.get_param("~nominal_z_min", 0.15))
        self.nominal_z_max = float(rospy.get_param("~nominal_z_max", 0.75))

        self._validate_params()
        self.output_root.mkdir(parents=True, exist_ok=True)

        self._target: Optional[PointStamped] = None
        self._trajectory: Optional[JointTrajectory] = None
        self._sweep_time_s: Optional[float] = None

        rospy.loginfo("[vbc_proj_smoke] loading frozen VisCDF model...")
        self._initialize_model_and_oracles()

        self.target_sub = rospy.Subscriber(
            self.target_topic, PointStamped, self._target_callback, queue_size=1)
        self.trajectory_sub = rospy.Subscriber(
            self.trajectory_topic, JointTrajectory, self._trajectory_callback, queue_size=1)
        self.sweep_sub = rospy.Subscriber(
            self.sweep_time_topic, Float32, self._sweep_callback, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(0.02), self._timer_callback)

        rospy.logwarn(
            "[vbc_proj_smoke] DIAGNOSTIC ONLY: waiting for frozen target + nominal task trajectory + selected sweep time."
        )
        rospy.loginfo(
            "[vbc_proj_smoke] caps=%s damping=%.3f eps_f=%.3f proj_iters=%d ascent_step=%.3f ascent_steps=%d",
            self.projection_max_step_norms,
            self.projection_damping,
            self.projection_epsilon_f,
            self.projection_iters,
            self.ascent_step_size,
            self.ascent_steps,
        )

    def _validate_params(self) -> None:
        if self.device_name not in ("cpu", "cuda"):
            raise ValueError("~device must be cpu or cuda")
        if self.device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("~device=cuda requested but CUDA is unavailable")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint_path}")
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")
        if self.safety_margin_s < 0.0:
            raise ValueError("~safety_margin_s must be non-negative")
        if self.projection_iters <= 0 or self.ascent_steps < 0:
            raise ValueError("iteration counts are invalid")
        if not 0.0 < self.projection_damping <= 1.0:
            raise ValueError("~projection_damping must be in (0,1]")
        if self.projection_epsilon_f <= 0.0:
            raise ValueError("~projection_epsilon_f must be positive")
        if not self.projection_max_step_norms or any(v <= 0.0 for v in self.projection_max_step_norms):
            raise ValueError("all projection max-step norms must be positive")
        if self.ascent_step_size <= 0.0 or self.ascent_max_step_norm <= 0.0:
            raise ValueError("ascent step/max must be positive")
        if self.math_eps <= 0.0:
            raise ValueError("~math_eps must be positive")
        if len(self.q_min_list) != 7 or len(self.q_max_list) != 7:
            raise ValueError("q_min/q_max must each contain 7 values")

    def _initialize_model_and_oracles(self) -> None:
        self.device = torch.device(self.device_name)
        checkpoint = torch_load_checkpoint(str(self.checkpoint_path), self.device)
        self.model, self.checkpoint_args = build_model_from_checkpoint(checkpoint, self.device)

        hfov = float(self.checkpoint_args.get("horizontal_fov_deg", 50.0))
        vfov = float(self.checkpoint_args.get("vertical_fov_deg", 66.0))
        z_min = float(self.checkpoint_args.get("z_min", 0.20))
        z_max = float(self.checkpoint_args.get("z_max", 0.70))
        delta = float(self.checkpoint_args.get("delta", 0.01))

        self.conservative_oracle = PinocchioFOVOracle(
            urdf_path=str(self.urdf_path),
            joint_names=list(DEFAULT_JOINT_NAMES),
            sensor_frames=list(DEFAULT_SENSOR_FRAMES),
            horizontal_fov_deg=hfov,
            vertical_fov_deg=vfov,
            z_min=z_min,
            z_max=z_max,
            delta=delta,
            base_frame="base_link",
        )
        self.nominal_oracle = PinocchioFOVOracle(
            urdf_path=str(self.urdf_path),
            joint_names=list(DEFAULT_JOINT_NAMES),
            sensor_frames=list(DEFAULT_SENSOR_FRAMES),
            horizontal_fov_deg=self.nominal_hfov,
            vertical_fov_deg=self.nominal_vfov,
            z_min=self.nominal_z_min,
            z_max=self.nominal_z_max,
            delta=0.0,
            base_frame="base_link",
        )

        self.q_min = torch.tensor(self.q_min_list, device=self.device, dtype=torch.float32)
        self.q_max = torch.tensor(self.q_max_list, device=self.device, dtype=torch.float32)

        # Warm up model and oracle paths.
        x = torch.tensor([[0.0, 0.0, 0.5]], device=self.device, dtype=torch.float32)
        q = torch.zeros((1, 7), device=self.device, dtype=torch.float32)
        model_value_and_grad_q(x, q, self.model)
        oracle_visibility_g(x, q, self.conservative_oracle)
        oracle_visibility_g(x, q, self.nominal_oracle)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        rospy.loginfo("[vbc_proj_smoke] frozen model ready: %s", self.checkpoint_path)

    def _target_callback(self, msg: PointStamped) -> None:
        if msg is None:
            return
        with self._lock:
            self._target = msg

    def _trajectory_callback(self, msg: JointTrajectory) -> None:
        if msg is None or not msg.points:
            return
        with self._lock:
            self._trajectory = msg

    def _sweep_callback(self, msg: Float32) -> None:
        if msg is None or not math.isfinite(float(msg.data)) or msg.data < 0.0:
            return
        with self._lock:
            self._sweep_time_s = float(msg.data)

    def _snapshot(self):
        with self._lock:
            return self._target, self._trajectory, self._sweep_time_s

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
        points = trajectory.points
        if not points:
            raise ValueError("empty task trajectory")

        def positions(point) -> np.ndarray:
            if len(point.positions) < len(trajectory.joint_names):
                raise ValueError("task trajectory point has incomplete positions")
            return np.asarray([point.positions[i] for i in mapping], dtype=np.float64)

        times = np.asarray([p.time_from_start.to_sec() for p in points], dtype=np.float64)
        if not _finite(times) or np.any(np.diff(times) < -1e-9):
            raise ValueError("invalid/non-monotonic task trajectory timestamps")
        if t <= times[0]:
            return positions(points[0])
        if t >= times[-1]:
            return positions(points[-1])
        hi = int(np.searchsorted(times, t, side="right"))
        lo = hi - 1
        t0, t1 = times[lo], times[hi]
        if t1 - t0 <= 1e-12:
            return positions(points[lo])
        alpha = float((t - t0) / (t1 - t0))
        return (1.0 - alpha) * positions(points[lo]) + alpha * positions(points[hi])

    def _clamp(self, q: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        if not self.clamp_q:
            return q, False
        q_new = torch.maximum(torch.minimum(q, self.q_max[None, :]), self.q_min[None, :])
        changed = bool(torch.any(torch.abs(q_new - q) > 1e-9).item())
        return q_new, changed

    def _evaluate(self, x: torch.Tensor, q: torch.Tensor) -> Tuple[float, np.ndarray, float, float]:
        learned, grad, _ = model_value_and_grad_q(x, q, self.model)
        g_cons = oracle_visibility_g(x, q, self.conservative_oracle)
        g_nom = oracle_visibility_g(x, q, self.nominal_oracle)
        return (
            float(learned[0].item()),
            grad[0].detach().cpu().numpy().astype(np.float64),
            float(g_cons[0].item()),
            float(g_nom[0].item()),
        )

    def _run_one_cap(
        self,
        x: torch.Tensor,
        q0: torch.Tensor,
        cap: float,
    ) -> Dict[str, object]:
        q = q0.clone()
        projection_history: List[Dict[str, object]] = []

        f0, grad0, g0, gn0 = self._evaluate(x, q)
        projection_history.append({
            "iter": 0,
            "f": f0,
            "g_conservative": g0,
            "g_nominal_fov": gn0,
            "grad_norm": float(np.linalg.norm(grad0)),
            "raw_step_norm": 0.0,
            "applied_step_norm": 0.0,
            "distance_from_q_deadline": 0.0,
            "joint_limit_clamped": False,
            "q": q[0].detach().cpu().numpy().astype(float).tolist(),
        })

        reached_learned_boundary = abs(f0) < self.projection_epsilon_f
        for iteration in range(1, self.projection_iters + 1):
            if reached_learned_boundary:
                break
            f_tensor, grad_tensor, _ = model_value_and_grad_q(x, q, self.model)
            q_new, diag = learned_projection_step(
                q=q,
                f=f_tensor,
                grad=grad_tensor,
                damping=self.projection_damping,
                max_step_norm=cap,
                eps=self.math_eps,
            )
            q_new, joint_clamped = self._clamp(q_new)
            q = q_new.detach()
            f, grad, g_cons, g_nom = self._evaluate(x, q)
            distance = float(torch.linalg.vector_norm(q - q0).item())
            projection_history.append({
                "iter": iteration,
                "f": f,
                "g_conservative": g_cons,
                "g_nominal_fov": g_nom,
                "grad_norm": float(np.linalg.norm(grad)),
                "raw_step_norm": float(diag["raw_step_norm"][0].item()),
                "applied_step_norm": float(diag["applied_step_norm"][0].item()),
                "algorithm_step_clipped": bool(diag["clipped"][0].item()),
                "distance_from_q_deadline": distance,
                "joint_limit_clamped": joint_clamped,
                "q": q[0].detach().cpu().numpy().astype(float).tolist(),
            })
            reached_learned_boundary = abs(f) < self.projection_epsilon_f

        q_proj = q.clone()
        proj_f, proj_grad, proj_g, proj_g_nom = self._evaluate(x, q_proj)

        ascent_history: List[Dict[str, object]] = [{
            "step": 0,
            "f": proj_f,
            "g_conservative": proj_g,
            "g_nominal_fov": proj_g_nom,
            "grad_norm": float(np.linalg.norm(proj_grad)),
            "distance_from_projection": 0.0,
            "distance_from_q_deadline": float(torch.linalg.vector_norm(q_proj - q0).item()),
            "q": q_proj[0].detach().cpu().numpy().astype(float).tolist(),
        }]

        first_cons_visible_step: Optional[int] = 0 if proj_g >= 0.0 else None
        first_nom_visible_step: Optional[int] = 0 if proj_g_nom >= 0.0 else None
        best_cons_g = proj_g
        best_nom_g = proj_g_nom
        q = q_proj.clone()

        for step in range(1, self.ascent_steps + 1):
            _, grad_tensor, _ = model_value_and_grad_q(x, q, self.model)
            q_new, diag = learned_ascent_step(
                q=q,
                grad=grad_tensor,
                step_size=self.ascent_step_size,
                max_step_norm=self.ascent_max_step_norm,
                eps=self.math_eps,
            )
            q_new, joint_clamped = self._clamp(q_new)
            q = q_new.detach()
            f, grad, g_cons, g_nom = self._evaluate(x, q)
            best_cons_g = max(best_cons_g, g_cons)
            best_nom_g = max(best_nom_g, g_nom)
            if first_cons_visible_step is None and g_cons >= 0.0:
                first_cons_visible_step = step
            if first_nom_visible_step is None and g_nom >= 0.0:
                first_nom_visible_step = step
            ascent_history.append({
                "step": step,
                "f": f,
                "g_conservative": g_cons,
                "g_nominal_fov": g_nom,
                "grad_norm": float(np.linalg.norm(grad)),
                "applied_step_norm": float(diag["applied_step_norm"][0].item()),
                "algorithm_step_clipped": bool(diag["clipped"][0].item()),
                "joint_limit_clamped": joint_clamped,
                "distance_from_projection": float(torch.linalg.vector_norm(q - q_proj).item()),
                "distance_from_q_deadline": float(torch.linalg.vector_norm(q - q0).item()),
                "q": q[0].detach().cpu().numpy().astype(float).tolist(),
            })

        return {
            "projection_max_step_norm": cap,
            "projection_reached_learned_boundary": reached_learned_boundary,
            "projection_iterations_applied": len(projection_history) - 1,
            "projection_endpoint_f": proj_f,
            "projection_endpoint_g_conservative": proj_g,
            "projection_endpoint_g_nominal_fov": proj_g_nom,
            "projection_endpoint_distance_from_q_deadline": float(
                torch.linalg.vector_norm(q_proj - q0).item()),
            "first_conservative_visible_ascent_step": first_cons_visible_step,
            "first_nominal_fov_visible_ascent_step": first_nom_visible_step,
            "best_conservative_g_during_ascent": best_cons_g,
            "best_nominal_fov_g_during_ascent": best_nom_g,
            "final_f": ascent_history[-1]["f"],
            "final_g_conservative": ascent_history[-1]["g_conservative"],
            "final_g_nominal_fov": ascent_history[-1]["g_nominal_fov"],
            "final_distance_from_q_deadline": ascent_history[-1]["distance_from_q_deadline"],
            "projection_history": projection_history,
            "ascent_history": ascent_history,
        }

    def _run(self, target: PointStamped, trajectory: JointTrajectory, sweep_time_s: float) -> None:
        target_frame = str(target.header.frame_id).strip().lstrip("/")
        if target_frame not in ("", "base_link"):
            raise ValueError(
                f"frozen target frame must be base_link for this smoke, got {target.header.frame_id!r}")

        x_np = np.asarray([target.point.x, target.point.y, target.point.z], dtype=np.float64)
        if not _finite(x_np):
            raise ValueError("non-finite frozen target")

        deadline_s = max(0.0, sweep_time_s - self.safety_margin_s)
        q_deadline_np = self._sample_trajectory(trajectory, deadline_s)
        if not _finite(q_deadline_np):
            raise ValueError("non-finite q_deadline")

        x = torch.tensor(x_np.reshape(1, 3), device=self.device, dtype=torch.float32)
        q0 = torch.tensor(q_deadline_np.reshape(1, 7), device=self.device, dtype=torch.float32)
        q0, q0_clamped = self._clamp(q0)

        init_f, init_grad, init_g, init_g_nom = self._evaluate(x, q0)

        rospy.logwarn("[vbc_proj_smoke] ================================================")
        rospy.logwarn("[vbc_proj_smoke] VBC DEADLINE PROJECTION SMOKE")
        rospy.logwarn("[vbc_proj_smoke] target x*      = %s", _vec(x_np, 6))
        rospy.logwarn("[vbc_proj_smoke] sweep_time     = %.6f s", sweep_time_s)
        rospy.logwarn("[vbc_proj_smoke] safety_margin  = %.6f s", self.safety_margin_s)
        rospy.logwarn("[vbc_proj_smoke] deadline_time  = %.6f s", deadline_s)
        rospy.logwarn("[vbc_proj_smoke] q_deadline_nom = %s", _vec(q0[0].detach().cpu().numpy(), 5))
        rospy.logwarn(
            "[vbc_proj_smoke] initial: f=%+.6f g_cons=%+.6f g_nominal=%+.6f |grad|=%.6f",
            init_f, init_g, init_g_nom, float(np.linalg.norm(init_grad)))

        result: Dict[str, object] = {
            "config": {
                "checkpoint": str(self.checkpoint_path),
                "urdf": str(self.urdf_path),
                "device": self.device_name,
                "safety_margin_s": self.safety_margin_s,
                "projection_iters": self.projection_iters,
                "projection_damping": self.projection_damping,
                "projection_epsilon_f": self.projection_epsilon_f,
                "projection_max_step_norms": self.projection_max_step_norms,
                "ascent_steps": self.ascent_steps,
                "ascent_step_size": self.ascent_step_size,
                "ascent_max_step_norm": self.ascent_max_step_norm,
                "clamp_q": self.clamp_q,
            },
            "target_xyz": x_np.tolist(),
            "nominal_sweep_time_s": sweep_time_s,
            "visibility_deadline_time_s": deadline_s,
            "q_deadline_nominal": q0[0].detach().cpu().numpy().astype(float).tolist(),
            "q_deadline_was_clamped": q0_clamped,
            "initial": {
                "f": init_f,
                "g_conservative": init_g,
                "g_nominal_fov": init_g_nom,
                "grad_norm": float(np.linalg.norm(init_grad)),
            },
            "caps": {},
        }

        for cap in self.projection_max_step_norms:
            cap_result = self._run_one_cap(x, q0, cap)
            result["caps"][f"{cap:.6g}"] = cap_result

            rospy.logwarn(
                "[vbc_proj_smoke] cap=%.3f projection: reached=%s iters=%d f=%+.6f g_cons=%+.6f g_nominal=%+.6f dq=%.4f",
                cap,
                cap_result["projection_reached_learned_boundary"],
                cap_result["projection_iterations_applied"],
                cap_result["projection_endpoint_f"],
                cap_result["projection_endpoint_g_conservative"],
                cap_result["projection_endpoint_g_nominal_fov"],
                cap_result["projection_endpoint_distance_from_q_deadline"],
            )
            rospy.logwarn(
                "[vbc_proj_smoke] cap=%.3f ascent: first g_cons>=0 step=%s first g_nominal>=0 step=%s final f=%+.6f final g_cons=%+.6f final g_nominal=%+.6f total_dq=%.4f",
                cap,
                str(cap_result["first_conservative_visible_ascent_step"]),
                str(cap_result["first_nominal_fov_visible_ascent_step"]),
                cap_result["final_f"],
                cap_result["final_g_conservative"],
                cap_result["final_g_nominal_fov"],
                cap_result["final_distance_from_q_deadline"],
            )

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_root / f"vbc_deadline_projection_smoke_{stamp}.json"
        output_path.write_text(json.dumps(result, indent=2, allow_nan=True))
        rospy.logwarn("[vbc_proj_smoke] saved: %s", output_path)
        rospy.logwarn("[vbc_proj_smoke] ================================================")

    def _timer_callback(self, _event) -> None:
        if self._ran:
            return
        target, trajectory, sweep_time_s = self._snapshot()
        if target is None or trajectory is None or sweep_time_s is None:
            return

        self._ran = True
        try:
            self._run(target, trajectory, sweep_time_s)
        except Exception as exc:
            self._ran = False
            rospy.logerr_throttle(1.0, "[vbc_proj_smoke] run failed: %s", exc)
            return

        # Keep the node alive briefly enough for terminal/log output to flush,
        # then request clean shutdown. No background work is needed after one run.
        rospy.signal_shutdown("one-shot VBC projection smoke complete")


def main() -> None:
    rospy.init_node("vbc_deadline_projection_smoke")
    VbcDeadlineProjectionSmokeNode()
    rospy.spin()


if __name__ == "__main__":
    main()
