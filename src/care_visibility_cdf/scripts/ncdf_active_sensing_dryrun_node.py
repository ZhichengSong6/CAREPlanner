#!/usr/bin/env python3
"""ROS1 Stage-I dry-run validation for CARE visibility NCDF guidance.

This node is deliberately NON-ACTUATING. It subscribes to the real CAREPlanner
active-sensing target, current 7-DoF JointState, and nominal command trajectory,
then applies the frozen direct normalized NCDF ascent used by the offline tests.

The learned field is the ONLY source used by the optimizer. After optimization,
the conservative analytic FOV oracle is evaluated at q_nominal and q_corrected
for diagnostics only:

    learned: f(x*, q_nominal) -> f(x*, q_corrected)
    oracle : g(x*, q_nominal) -> g(x*, q_corrected)

Oracle g never changes the step direction, line search, stopping condition, or
published correction. This keeps Stage I a genuine real-distribution validation
of the learned NCDF gradient.

Robot state/configuration is exactly 7-D:

    q = [joint1, joint2, joint3, joint4,
         wrist_joint1, wrist_joint2, wrist_joint3] in R^7.

No GCDF mobile-base state dimensions are used here.

The built-in Stage-I logger stores only unique real-frontier cases. A case is
considered new when either the frontier point changes by at least the configured
Cartesian threshold or q_nominal changes by at least the configured 7-D L2
threshold. Repeated 20-Hz evaluations of the same case are therefore not
counted multiple times.
"""

from __future__ import annotations

import copy
import csv
import math
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rospkg
import rospy
import torch
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32, String
from trajectory_msgs.msg import JointTrajectory


PACKAGE_DIR = Path(rospkg.RosPack().get_path("care_visibility_cdf")).resolve()
SCRIPT_DIR = PACKAGE_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_direct_vs_projection_ascent import (  # noqa: E402
    DEFAULT_SENSOR_FRAMES,
    PinocchioFOVOracle,
    build_model_from_checkpoint,
    torch_load_checkpoint,
)
from export_ncdf_l4casadi import build_casadi_functions  # noqa: E402
from test_ncdf_local_optimizer import (  # noqa: E402
    DEFAULT_JOINT_NAMES,
    DEFAULT_URDF,
    read_joint_limits,
)


DEFAULT_CHECKPOINT = (
    PACKAGE_DIR
    / "checkpoints"
    / "exp1_yiming_k500_fov_signed"
    / "final.pt"
)
DEFAULT_BUILD_DIR = PACKAGE_DIR / "generated" / "l4casadi_ncdf_ros_dryrun"


def _value_grad(value_grad_fn, x: np.ndarray, q: np.ndarray) -> Tuple[float, np.ndarray]:
    outputs = value_grad_fn(
        np.asarray(x, dtype=np.float64).reshape(1, 3),
        np.asarray(q, dtype=np.float64).reshape(1, 7),
    )
    value = float(np.asarray(outputs[0]).reshape(()))
    grad = np.asarray(outputs[1], dtype=np.float64).reshape(7)
    return value, grad


def _format_vec(values: Sequence[float], precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(v):+.{precision}f}" for v in values) + "]"


def _oracle_cohort(g0: float) -> str:
    if -0.03 < g0 < 0.0:
        return "near"
    if -0.10 < g0 <= -0.03:
        return "middle"
    if g0 <= -0.10:
        return "far"
    return "inside_or_boundary"


class NcdfActiveSensingDryRunNode:
    def __init__(self) -> None:
        self._lock = threading.Lock()

        # ROS topics / timing.
        self.target_topic = rospy.get_param(
            "~target_topic", "/care_planner/active_sensing/target_point"
        )
        self.joint_state_topic = rospy.get_param(
            "~joint_state_topic", "/care_arm/joint_states"
        )
        self.command_trajectory_topic = rospy.get_param(
            "~command_trajectory_topic", "/care_planner/command_trajectory_candidate"
        )
        self.base_frame = str(rospy.get_param("~base_frame", "base_link"))
        self.update_rate = float(rospy.get_param("~update_rate", 20.0))
        self.target_timeout = float(rospy.get_param("~target_timeout", 0.20))
        self.joint_state_timeout = float(rospy.get_param("~joint_state_timeout", 0.20))
        self.trajectory_timeout = float(rospy.get_param("~trajectory_timeout", 0.20))
        self.lookahead_time = float(rospy.get_param("~lookahead_time", 0.05))

        # Frozen direct optimizer parameters.
        self.iterations = int(rospy.get_param("~iterations", 5))
        self.iter_step = float(rospy.get_param("~iter_step", 0.01))
        self.step_max = float(rospy.get_param("~step_max", 0.05))
        self.max_backtracks = int(rospy.get_param("~max_backtracks", 4))
        self.grad_eps = float(rospy.get_param("~grad_eps", 1e-10))

        # Frozen conservative analytic FOV oracle. These are the same defaults
        # used by the offline NCDF-vs-oracle evaluation.
        self.horizontal_fov_deg = float(rospy.get_param("~oracle_horizontal_fov_deg", 50.0))
        self.vertical_fov_deg = float(rospy.get_param("~oracle_vertical_fov_deg", 66.0))
        self.oracle_z_min = float(rospy.get_param("~oracle_z_min", 0.20))
        self.oracle_z_max = float(rospy.get_param("~oracle_z_max", 0.70))
        self.oracle_delta = float(rospy.get_param("~oracle_delta", 0.01))

        self.device_name = str(rospy.get_param("~device", "cpu"))
        self.checkpoint_path = Path(
            rospy.get_param("~checkpoint", str(DEFAULT_CHECKPOINT))
        ).expanduser().resolve()
        self.urdf_path = Path(
            rospy.get_param("~urdf", str(DEFAULT_URDF))
        ).expanduser().resolve()
        self.build_dir = Path(
            rospy.get_param("~build_dir", str(DEFAULT_BUILD_DIR))
        ).expanduser().resolve()
        self.model_name = str(
            rospy.get_param("~model_name", "care_visibility_ncdf_ros_dryrun")
        )

        raw_joint_names = rospy.get_param("~joint_names", list(DEFAULT_JOINT_NAMES))
        self.joint_names = [str(name) for name in raw_joint_names]

        # Stage-I unique-case logger.
        self.enable_logger = bool(rospy.get_param("~enable_logger", True))
        self.unique_target_threshold_m = float(
            rospy.get_param("~unique_target_threshold_m", 0.01)
        )
        self.unique_q_threshold_rad = float(
            rospy.get_param("~unique_q_threshold_rad", 0.02)
        )
        run_stamp = time.strftime("%Y%m%d_%H%M%S")
        default_log_path = (
            PACKAGE_DIR / "outputs" / f"ncdf_stage1_real_frontier_{run_stamp}.csv"
        )
        self.log_path = Path(
            rospy.get_param("~log_path", str(default_log_path))
        ).expanduser().resolve()

        self._validate_params()

        # Cached ROS inputs and local receipt times.
        self._latest_target: Optional[PointStamped] = None
        self._target_received: Optional[rospy.Time] = None
        self._latest_joint_state: Optional[JointState] = None
        self._joint_state_received: Optional[rospy.Time] = None
        self._latest_trajectory: Optional[JointTrajectory] = None
        self._trajectory_received: Optional[rospy.Time] = None

        # Logger state.
        self._log_file = None
        self._csv_writer = None
        self._last_logged_x: Optional[np.ndarray] = None
        self._last_logged_q_nominal: Optional[np.ndarray] = None
        self._logged_results: List[Dict[str, float]] = []
        self._case_id = 0

        rospy.loginfo("[ncdf_dryrun] Loading frozen NCDF / L4CasADi interface...")
        self._initialize_model_and_oracle()
        self._initialize_logger()

        # Debug-only publishers. There is intentionally no publisher for the
        # planner command trajectory or any controller command topic.
        self.nominal_pub = rospy.Publisher(
            "~nominal_joint_state", JointState, queue_size=1
        )
        self.corrected_pub = rospy.Publisher(
            "~corrected_joint_state", JointState, queue_size=1
        )
        self.active_pub = rospy.Publisher("~active", Bool, queue_size=1)
        self.delta_f_pub = rospy.Publisher("~delta_f", Float32, queue_size=1)
        self.oracle_g_before_pub = rospy.Publisher(
            "~oracle_g_before", Float32, queue_size=1
        )
        self.oracle_g_after_pub = rospy.Publisher(
            "~oracle_g_after", Float32, queue_size=1
        )
        self.delta_g_pub = rospy.Publisher("~delta_g", Float32, queue_size=1)
        self.oracle_agrees_pub = rospy.Publisher(
            "~oracle_agrees", Bool, queue_size=1
        )
        self.optimizer_time_pub = rospy.Publisher(
            "~optimizer_time_ms", Float32, queue_size=1
        )
        self.oracle_time_pub = rospy.Publisher(
            "~oracle_time_ms", Float32, queue_size=1
        )
        self.summary_pub = rospy.Publisher("~summary", String, queue_size=1)

        self.target_sub = rospy.Subscriber(
            self.target_topic, PointStamped, self._target_callback, queue_size=1
        )
        self.joint_state_sub = rospy.Subscriber(
            self.joint_state_topic, JointState, self._joint_state_callback, queue_size=1
        )
        self.trajectory_sub = rospy.Subscriber(
            self.command_trajectory_topic,
            JointTrajectory,
            self._trajectory_callback,
            queue_size=1,
        )

        self.timer = rospy.Timer(
            rospy.Duration.from_sec(1.0 / self.update_rate), self._timer_callback
        )
        rospy.on_shutdown(self._on_shutdown)

        rospy.logwarn(
            "[ncdf_dryrun] DRY RUN ONLY: no command trajectory or controller command is published."
        )
        rospy.loginfo(
            "[ncdf_dryrun] 7-DoF state q=%s",
            ",".join(self.joint_names),
        )
        rospy.loginfo(
            "[ncdf_dryrun] target=%s joint_state=%s trajectory=%s",
            self.target_topic,
            self.joint_state_topic,
            self.command_trajectory_topic,
        )
        rospy.loginfo(
            "[ncdf_dryrun] lookahead=%.3fs target_timeout=%.3fs direct=%dx%.4frad trust=+/-%.4frad",
            self.lookahead_time,
            self.target_timeout,
            self.iterations,
            self.iter_step,
            self.step_max,
        )
        rospy.loginfo(
            "[ncdf_dryrun] oracle diagnostic only: hFOV=%.1fdeg vFOV=%.1fdeg z=[%.2f,%.2f] delta=%.3f",
            self.horizontal_fov_deg,
            self.vertical_fov_deg,
            self.oracle_z_min,
            self.oracle_z_max,
            self.oracle_delta,
        )
        if self.enable_logger:
            rospy.loginfo(
                "[stage1_logger] unique if dx>=%.3fm OR dq_nominal_l2>=%.3frad",
                self.unique_target_threshold_m,
                self.unique_q_threshold_rad,
            )
            rospy.loginfo("[stage1_logger] CSV: %s", self.log_path)

    def _validate_params(self) -> None:
        if self.update_rate <= 0.0:
            raise ValueError("~update_rate must be positive")
        if self.target_timeout <= 0.0:
            raise ValueError("~target_timeout must be positive")
        if self.joint_state_timeout <= 0.0 or self.trajectory_timeout <= 0.0:
            raise ValueError("input timeouts must be positive")
        if self.lookahead_time < 0.0:
            raise ValueError("~lookahead_time must be non-negative")
        if self.iterations <= 0:
            raise ValueError("~iterations must be positive")
        if self.iter_step <= 0.0 or self.step_max <= 0.0:
            raise ValueError("~iter_step and ~step_max must be positive")
        if len(self.joint_names) != 7 or len(set(self.joint_names)) != 7:
            raise ValueError("~joint_names must contain exactly 7 unique names")
        if self.oracle_z_min >= self.oracle_z_max:
            raise ValueError("oracle z_min must be smaller than z_max")
        if self.unique_target_threshold_m < 0.0 or self.unique_q_threshold_rad < 0.0:
            raise ValueError("unique-case thresholds must be non-negative")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"NCDF checkpoint not found: {self.checkpoint_path}")
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")
        if self.device_name not in ("cpu", "cuda"):
            raise ValueError("~device must be 'cpu' or 'cuda'")
        if self.device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("~device=cuda requested but torch CUDA is unavailable")

    def _initialize_model_and_oracle(self) -> None:
        self.device = torch.device(self.device_name)
        self.q_min, self.q_max = read_joint_limits(self.urdf_path, self.joint_names)

        checkpoint = torch_load_checkpoint(str(self.checkpoint_path), self.device)
        model, _ = build_model_from_checkpoint(checkpoint, self.device)
        _, _, _, self.value_grad_fn = build_casadi_functions(
            model=model,
            device=self.device_name,
            build_dir=self.build_dir,
            model_name=self.model_name,
        )

        # Warm up the L4CasADi external function before online timing starts.
        self.value_grad_fn(np.asarray([[0.0, 0.0, 0.5]]), np.zeros((1, 7)))

        self.oracle = PinocchioFOVOracle(
            urdf_path=str(self.urdf_path),
            joint_names=list(self.joint_names),
            sensor_frames=list(DEFAULT_SENSOR_FRAMES),
            horizontal_fov_deg=self.horizontal_fov_deg,
            vertical_fov_deg=self.vertical_fov_deg,
            z_min=self.oracle_z_min,
            z_max=self.oracle_z_max,
            delta=self.oracle_delta,
            base_frame=self.base_frame,
        )

        rospy.loginfo("[ncdf_dryrun] Frozen NCDF ready on %s.", self.device_name)
        rospy.loginfo("[ncdf_dryrun] Analytic FOV oracle ready for debug-only evaluation.")

    def _initialize_logger(self) -> None:
        if not self.enable_logger:
            rospy.loginfo("[stage1_logger] disabled")
            return

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("w", newline="", buffering=1)

        fields = [
            "case_id",
            "ros_time",
            "wall_time_unix",
            "cohort",
            "target_x",
            "target_y",
            "target_z",
            "target_age_s",
            "q_nominal_from_current_l2",
        ]
        fields += [f"q_current_{name}" for name in self.joint_names]
        fields += [f"q_nominal_{name}" for name in self.joint_names]
        fields += [f"q_corrected_{name}" for name in self.joint_names]
        fields += [f"delta_q_{name}" for name in self.joint_names]
        fields += [
            "f_before",
            "f_after",
            "delta_f",
            "g_before",
            "g_after",
            "delta_g",
            "oracle_agree",
            "crossed_oracle_boundary",
            "f_up_g_down",
            "delta_q_l2",
            "delta_q_inf",
            "accepted_steps",
            "requested_steps",
            "backtracks",
            "optimizer_time_ms",
            "oracle_time_ms",
        ]

        self._csv_writer = csv.DictWriter(self._log_file, fieldnames=fields)
        self._csv_writer.writeheader()
        self._log_file.flush()

    def _target_callback(self, msg: PointStamped) -> None:
        if msg is None:
            return
        with self._lock:
            self._latest_target = copy.deepcopy(msg)
            self._target_received = rospy.Time.now()

    def _joint_state_callback(self, msg: JointState) -> None:
        if msg is None:
            return
        with self._lock:
            self._latest_joint_state = copy.deepcopy(msg)
            self._joint_state_received = rospy.Time.now()

    def _trajectory_callback(self, msg: JointTrajectory) -> None:
        if msg is None:
            return
        with self._lock:
            self._latest_trajectory = copy.deepcopy(msg)
            self._trajectory_received = rospy.Time.now()

    @staticmethod
    def _age(now: rospy.Time, received: Optional[rospy.Time]) -> float:
        if received is None:
            return math.inf
        return max(0.0, (now - received).to_sec())

    def _publish_inactive(self, reason: str) -> None:
        self.active_pub.publish(Bool(data=False))
        rospy.logwarn_throttle(1.0, f"[ncdf_dryrun] inactive: {reason}")

    def _ordered_joint_positions(self, msg: JointState) -> Optional[np.ndarray]:
        if len(msg.name) != len(msg.position):
            return None
        index: Dict[str, int] = {name: i for i, name in enumerate(msg.name)}
        if any(name not in index for name in self.joint_names):
            return None
        q = np.asarray(
            [msg.position[index[name]] for name in self.joint_names],
            dtype=np.float64,
        )
        return q if np.all(np.isfinite(q)) else None

    def _trajectory_positions_in_order(
        self, msg: JointTrajectory
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not msg.points or not msg.joint_names:
            return None
        index: Dict[str, int] = {name: i for i, name in enumerate(msg.joint_names)}
        if any(name not in index for name in self.joint_names):
            return None

        times = []
        positions = []
        for point in msg.points:
            if len(point.positions) < len(msg.joint_names):
                return None
            q = [point.positions[index[name]] for name in self.joint_names]
            if not np.all(np.isfinite(q)):
                return None
            times.append(point.time_from_start.to_sec())
            positions.append(q)

        times_np = np.asarray(times, dtype=np.float64)
        positions_np = np.asarray(positions, dtype=np.float64)
        if np.any(np.diff(times_np) < -1e-9):
            return None
        return times_np, positions_np

    def _sample_nominal_q(self, msg: JointTrajectory) -> Optional[np.ndarray]:
        parsed = self._trajectory_positions_in_order(msg)
        if parsed is None:
            return None
        times, positions = parsed
        t = self.lookahead_time

        if len(times) == 1 or t <= times[0]:
            return positions[0].copy()
        if t >= times[-1]:
            return positions[-1].copy()

        upper = int(np.searchsorted(times, t, side="right"))
        lower = upper - 1
        t0, t1 = float(times[lower]), float(times[upper])
        if t1 <= t0 + 1e-12:
            return positions[upper].copy()
        ratio = (t - t0) / (t1 - t0)
        return (1.0 - ratio) * positions[lower] + ratio * positions[upper]

    def _direct_correct(self, x: np.ndarray, q_nominal: np.ndarray):
        # Stage-I diagnostic only: a local visibility-only configuration step.
        # This is NOT the final action-space controller.
        lb = np.maximum(self.q_min, q_nominal - self.step_max)
        ub = np.minimum(self.q_max, q_nominal + self.step_max)
        q = np.clip(q_nominal.copy(), lb, ub)

        f_initial, _ = _value_grad(self.value_grad_fn, x, q)
        accepted = 0
        backtracks = 0

        tic = time.perf_counter()
        for _ in range(self.iterations):
            f, grad = _value_grad(self.value_grad_fn, x, q)
            grad_norm = float(np.linalg.norm(grad))
            if not np.isfinite(grad_norm) or grad_norm <= self.grad_eps:
                break

            direction = grad / grad_norm
            step = self.iter_step
            accepted_this_iter = False

            for bt in range(self.max_backtracks + 1):
                candidate = np.clip(q + step * direction, lb, ub)
                if np.linalg.norm(candidate - q) <= 1e-12:
                    break
                f_candidate, _ = _value_grad(self.value_grad_fn, x, candidate)
                if np.isfinite(f_candidate) and f_candidate >= f - 1e-12:
                    q = candidate
                    accepted += 1
                    backtracks += bt
                    accepted_this_iter = True
                    break
                step *= 0.5

            if not accepted_this_iter:
                break

        elapsed_ms = 1000.0 * (time.perf_counter() - tic)
        f_final, _ = _value_grad(self.value_grad_fn, x, q)
        return q, f_initial, f_final, accepted, backtracks, elapsed_ms

    def _oracle_g(self, x: np.ndarray, q: np.ndarray) -> float:
        # Explicit 7-D configuration input. Oracle is evaluation-only.
        x_t = torch.as_tensor(
            x, dtype=torch.float32, device=self.device
        ).reshape(1, 3)
        q_t = torch.as_tensor(
            q, dtype=torch.float32, device=self.device
        ).reshape(1, 7)
        raw_margin, _ = self.oracle.signed_fov_margins(x_t, q_t)
        g = torch.max(raw_margin - self.oracle.delta, dim=-1).values
        return float(g.reshape(()).detach().cpu())

    def _is_unique_case(self, x: np.ndarray, q_nominal: np.ndarray) -> Tuple[bool, float, float]:
        if self._last_logged_x is None or self._last_logged_q_nominal is None:
            return True, math.inf, math.inf

        dx = float(np.linalg.norm(x - self._last_logged_x))
        dq = float(np.linalg.norm(q_nominal - self._last_logged_q_nominal))
        unique = (
            dx >= self.unique_target_threshold_m
            or dq >= self.unique_q_threshold_rad
        )
        return unique, dx, dq

    def _log_unique_case(
        self,
        *,
        now: rospy.Time,
        x: np.ndarray,
        target_age: float,
        q_current: np.ndarray,
        q_nominal: np.ndarray,
        q_corrected: np.ndarray,
        f0: float,
        f1: float,
        g0: float,
        g1: float,
        accepted: int,
        backtracks: int,
        optimizer_time_ms: float,
        oracle_time_ms: float,
    ) -> None:
        if not self.enable_logger or self._csv_writer is None:
            return

        is_unique, dx_from_last, dq_from_last = self._is_unique_case(x, q_nominal)
        if not is_unique:
            return

        self._case_id += 1
        delta_q = q_corrected - q_nominal
        delta_f = float(f1 - f0)
        delta_g = float(g1 - g0)
        oracle_agree = bool(delta_g >= -1e-8)
        crossed = bool(g0 < 0.0 and g1 >= 0.0)
        f_up_g_down = bool(delta_f > 1e-8 and delta_g < -1e-8)
        q_nominal_from_current_l2 = float(np.linalg.norm(q_nominal - q_current))

        row = {
            "case_id": self._case_id,
            "ros_time": float(now.to_sec()),
            "wall_time_unix": float(time.time()),
            "cohort": _oracle_cohort(g0),
            "target_x": float(x[0]),
            "target_y": float(x[1]),
            "target_z": float(x[2]),
            "target_age_s": float(target_age),
            "q_nominal_from_current_l2": q_nominal_from_current_l2,
            "f_before": float(f0),
            "f_after": float(f1),
            "delta_f": delta_f,
            "g_before": float(g0),
            "g_after": float(g1),
            "delta_g": delta_g,
            "oracle_agree": int(oracle_agree),
            "crossed_oracle_boundary": int(crossed),
            "f_up_g_down": int(f_up_g_down),
            "delta_q_l2": float(np.linalg.norm(delta_q)),
            "delta_q_inf": float(np.max(np.abs(delta_q))),
            "accepted_steps": int(accepted),
            "requested_steps": int(self.iterations),
            "backtracks": int(backtracks),
            "optimizer_time_ms": float(optimizer_time_ms),
            "oracle_time_ms": float(oracle_time_ms),
        }

        for i, name in enumerate(self.joint_names):
            row[f"q_current_{name}"] = float(q_current[i])
            row[f"q_nominal_{name}"] = float(q_nominal[i])
            row[f"q_corrected_{name}"] = float(q_corrected[i])
            row[f"delta_q_{name}"] = float(delta_q[i])

        self._csv_writer.writerow(row)
        self._log_file.flush()

        self._last_logged_x = x.copy()
        self._last_logged_q_nominal = q_nominal.copy()
        self._logged_results.append(
            {
                "delta_f": delta_f,
                "g_before": float(g0),
                "g_after": float(g1),
                "delta_g": delta_g,
                "oracle_agree": float(oracle_agree),
                "f_up_g_down": float(f_up_g_down),
            }
        )

        if math.isinf(dx_from_last):
            uniqueness_text = "first case"
        else:
            uniqueness_text = (
                f"dx={dx_from_last:.3f}m dq_nom_l2={dq_from_last:.3f}rad"
            )

        rospy.logwarn(
            "[stage1_logger] saved case %d (%s): cohort=%s g=%+.5f->%+.5f dg=%+.5f agree=%d f_up_g_down=%d",
            self._case_id,
            uniqueness_text,
            _oracle_cohort(g0),
            g0,
            g1,
            delta_g,
            int(oracle_agree),
            int(f_up_g_down),
        )
        self._print_running_logger_summary()

    def _print_running_logger_summary(self) -> None:
        if not self._logged_results:
            return
        dg = np.asarray([r["delta_g"] for r in self._logged_results], dtype=np.float64)
        improve = dg > 1e-8
        disagree = np.asarray(
            [r["f_up_g_down"] for r in self._logged_results], dtype=np.float64
        ) > 0.5
        rospy.logwarn(
            "[stage1_logger] N=%d oracle_improve=%.1f%% f_up_g_down=%.1f%% mean_dg=%+.5f median_dg=%+.5f",
            len(dg),
            100.0 * float(np.mean(improve)),
            100.0 * float(np.mean(disagree)),
            float(np.mean(dg)),
            float(np.median(dg)),
        )

    def _print_final_logger_summary(self) -> None:
        if not self.enable_logger:
            return
        if not self._logged_results:
            rospy.logwarn("[stage1_logger] no unique cases were recorded")
            return

        rospy.logwarn("[stage1_logger] ===== final Stage-I summary =====")
        self._print_running_logger_summary()

        for cohort in ("near", "middle", "far", "inside_or_boundary"):
            subset = [
                r
                for r in self._logged_results
                if _oracle_cohort(float(r["g_before"])) == cohort
            ]
            if not subset:
                continue
            dg = np.asarray([r["delta_g"] for r in subset], dtype=np.float64)
            disagree = np.asarray(
                [r["f_up_g_down"] for r in subset], dtype=np.float64
            ) > 0.5
            rospy.logwarn(
                "[stage1_logger] %-18s N=%d improve=%.1f%% disagree=%.1f%% mean_dg=%+.5f",
                cohort,
                len(subset),
                100.0 * float(np.mean(dg > 1e-8)),
                100.0 * float(np.mean(disagree)),
                float(np.mean(dg)),
            )
        rospy.logwarn("[stage1_logger] CSV saved to: %s", self.log_path)

    def _on_shutdown(self) -> None:
        try:
            self._print_final_logger_summary()
        finally:
            if self._log_file is not None:
                try:
                    self._log_file.flush()
                    self._log_file.close()
                except Exception:
                    pass
                self._log_file = None

    def _joint_state_msg(self, q: np.ndarray, stamp: rospy.Time) -> JointState:
        msg = JointState()
        msg.header.stamp = stamp
        msg.header.frame_id = self.base_frame
        msg.name = list(self.joint_names)
        msg.position = [float(v) for v in q]
        return msg

    def _timer_callback(self, _event) -> None:
        now = rospy.Time.now()
        with self._lock:
            target = copy.deepcopy(self._latest_target)
            target_received = self._target_received
            joint_state = copy.deepcopy(self._latest_joint_state)
            joint_received = self._joint_state_received
            trajectory = copy.deepcopy(self._latest_trajectory)
            trajectory_received = self._trajectory_received

        if target is None:
            self._publish_inactive("waiting for active-sensing target")
            return
        target_age = self._age(now, target_received)
        if target_age > self.target_timeout:
            self._publish_inactive(f"stale target age={target_age:.3f}s")
            return
        if target.header.frame_id and target.header.frame_id != self.base_frame:
            self._publish_inactive(
                f"target frame '{target.header.frame_id}' != expected '{self.base_frame}'"
            )
            return

        if joint_state is None:
            self._publish_inactive("waiting for JointState")
            return
        joint_age = self._age(now, joint_received)
        if joint_age > self.joint_state_timeout:
            self._publish_inactive(f"stale JointState age={joint_age:.3f}s")
            return

        if trajectory is None:
            self._publish_inactive("waiting for command trajectory")
            return
        trajectory_age = self._age(now, trajectory_received)
        if trajectory_age > self.trajectory_timeout:
            self._publish_inactive(f"stale trajectory age={trajectory_age:.3f}s")
            return

        q_current = self._ordered_joint_positions(joint_state)
        if q_current is None:
            self._publish_inactive("JointState lacks the configured 7 joint positions")
            return
        q_nominal = self._sample_nominal_q(trajectory)
        if q_nominal is None:
            self._publish_inactive("command trajectory cannot provide ordered 7-D nominal q")
            return

        x = np.asarray(
            [target.point.x, target.point.y, target.point.z], dtype=np.float64
        )
        if not np.all(np.isfinite(x)):
            self._publish_inactive("target contains non-finite coordinates")
            return

        try:
            q_corrected, f0, f1, accepted, backtracks, elapsed_ms = self._direct_correct(
                x, q_nominal
            )
        except Exception as exc:
            self._publish_inactive(f"optimizer exception: {exc}")
            rospy.logerr_throttle(1.0, "[ncdf_dryrun] optimizer exception: %s", exc)
            return

        # Stage-I oracle validation is deliberately post hoc. The oracle has no
        # influence on q_corrected above.
        try:
            oracle_tic = time.perf_counter()
            g0 = self._oracle_g(x, q_nominal)
            g1 = self._oracle_g(x, q_corrected)
            oracle_elapsed_ms = 1000.0 * (time.perf_counter() - oracle_tic)
        except Exception as exc:
            self._publish_inactive(f"oracle debug exception: {exc}")
            rospy.logerr_throttle(1.0, "[ncdf_dryrun] oracle exception: %s", exc)
            return

        delta_q = q_corrected - q_nominal
        delta_f = f1 - f0
        delta_g = g1 - g0
        step_l2 = float(np.linalg.norm(delta_q))
        step_inf = float(np.max(np.abs(delta_q)))
        nominal_from_current_l2 = float(np.linalg.norm(q_nominal - q_current))
        oracle_agrees = bool(delta_g >= -1e-8)
        crossed_oracle_boundary = bool(g0 < 0.0 and g1 >= 0.0)
        f_up_g_down = bool(delta_f > 1e-8 and delta_g < -1e-8)

        stamp = now
        self.nominal_pub.publish(self._joint_state_msg(q_nominal, stamp))
        self.corrected_pub.publish(self._joint_state_msg(q_corrected, stamp))
        self.active_pub.publish(Bool(data=True))
        self.delta_f_pub.publish(Float32(data=float(delta_f)))
        self.oracle_g_before_pub.publish(Float32(data=float(g0)))
        self.oracle_g_after_pub.publish(Float32(data=float(g1)))
        self.delta_g_pub.publish(Float32(data=float(delta_g)))
        self.oracle_agrees_pub.publish(Bool(data=oracle_agrees))
        self.optimizer_time_pub.publish(Float32(data=float(elapsed_ms)))
        self.oracle_time_pub.publish(Float32(data=float(oracle_elapsed_ms)))

        self._log_unique_case(
            now=now,
            x=x,
            target_age=target_age,
            q_current=q_current,
            q_nominal=q_nominal,
            q_corrected=q_corrected,
            f0=f0,
            f1=f1,
            g0=g0,
            g1=g1,
            accepted=accepted,
            backtracks=backtracks,
            optimizer_time_ms=elapsed_ms,
            oracle_time_ms=oracle_elapsed_ms,
        )

        summary = (
            f"x={_format_vec(x)} target_age={target_age:.3f}s "
            f"q_nominal_from_current_l2={nominal_from_current_l2:.4f} "
            f"f={f0:+.5f}->{f1:+.5f} df={delta_f:+.5f} "
            f"g={g0:+.5f}->{g1:+.5f} dg={delta_g:+.5f} "
            f"oracle_agree={int(oracle_agrees)} cross={int(crossed_oracle_boundary)} "
            f"f_up_g_down={int(f_up_g_down)} "
            f"dq_l2={step_l2:.4f} dq_inf={step_inf:.4f} "
            f"accepted={accepted}/{self.iterations} backtracks={backtracks} "
            f"opt_time={elapsed_ms:.3f}ms oracle_time={oracle_elapsed_ms:.3f}ms "
            f"dq={_format_vec(delta_q)}"
        )
        self.summary_pub.publish(String(data=summary))
        rospy.loginfo_throttle(0.5, "[ncdf_dryrun] %s", summary)


def main() -> None:
    rospy.init_node("ncdf_active_sensing_dryrun")
    try:
        NcdfActiveSensingDryRunNode()
    except Exception as exc:
        rospy.logfatal("[ncdf_dryrun] initialization failed: %s", exc)
        raise
    rospy.spin()


if __name__ == "__main__":
    main()
