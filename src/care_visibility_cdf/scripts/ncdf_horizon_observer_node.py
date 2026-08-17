#!/usr/bin/env python3
"""Phase B.1 non-actuating visibility NCDF horizon observer.

This ROS1 node observes the physically feasible trajectory predicted by the
Phase-A velocity QP-MPC and evaluates, for one active-sensing target x*, the
frozen visibility NCDF and the conservative analytic visibility oracle at every
configuration on the horizon.

For each q_k it computes

    f_theta(x*, q_k)
    grad_q f_theta(x*, q_k)
    g(x*, q_k)

in one batched PyTorch evaluation.  It never publishes a trajectory, velocity,
or any other actuation command.  Phase B.1 is therefore diagnostic-only: adding
this node must not change the robot motion produced by Phase A.

The canonical 7-DoF order is

    [joint1, joint2, joint3, joint4,
     wrist_joint1, wrist_joint2, wrist_joint3].

The incoming JointTrajectory is mapped by joint name rather than assumed to be
already ordered correctly.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import rospkg
import rospy
import torch
from geometry_msgs.msg import PointStamped
from std_msgs.msg import (
    Bool,
    Float32,
    Float64MultiArray,
    MultiArrayDimension,
    String,
)
from trajectory_msgs.msg import JointTrajectory


PACKAGE_DIR = Path(rospkg.RosPack().get_path("care_visibility_cdf")).resolve()
SCRIPT_DIR = PACKAGE_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_direct_vs_projection_ascent import (  # noqa: E402
    DEFAULT_SENSOR_FRAMES,
    PinocchioFOVOracle,
    build_model_from_checkpoint,
    model_value,
    model_value_and_grad_q,
    oracle_visibility_g,
    torch_load_checkpoint,
)


DEFAULT_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "wrist_joint1",
    "wrist_joint2",
    "wrist_joint3",
]
DEFAULT_CHECKPOINT = (
    PACKAGE_DIR
    / "checkpoints"
    / "exp1_yiming_k500_fov_signed"
    / "final.pt"
)
DEFAULT_URDF = (
    Path(rospkg.RosPack().get_path("arm_description"))
    / "urdf"
    / "Arm.urdf"
)


def _normalize_frame(frame_id: str) -> str:
    return str(frame_id).strip().lstrip("/")


def _finite_array(values: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(values)))


def _format_vec(values: Sequence[float], precision: int = 3) -> str:
    return "[" + ",".join(f"{float(v):.{precision}f}" for v in values) + "]"


def _vector_msg(values: np.ndarray) -> Float64MultiArray:
    msg = Float64MultiArray()
    msg.data = [float(v) for v in np.asarray(values).reshape(-1)]
    return msg


def _matrix_msg(values: np.ndarray, row_label: str, col_label: str) -> Float64MultiArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected a matrix, got shape {array.shape}.")

    rows, cols = array.shape
    msg = Float64MultiArray()
    row_dim = MultiArrayDimension()
    row_dim.label = row_label
    row_dim.size = rows
    row_dim.stride = rows * cols
    col_dim = MultiArrayDimension()
    col_dim.label = col_label
    col_dim.size = cols
    col_dim.stride = cols
    msg.layout.dim = [row_dim, col_dim]
    msg.data = [float(v) for v in array.reshape(-1)]
    return msg


class NcdfHorizonObserverNode:
    def __init__(self) -> None:
        self._lock = threading.Lock()

        self.target_topic = str(
            rospy.get_param(
                "~target_topic", "/care_planner/active_sensing/target_point"
            )
        )
        self.predicted_trajectory_topic = str(
            rospy.get_param(
                "~predicted_trajectory_topic",
                "/care_planner/mpc/predicted_trajectory",
            )
        )
        self.base_frame = str(rospy.get_param("~base_frame", "base_link"))
        self.update_rate = float(rospy.get_param("~update_rate", 20.0))
        self.target_timeout = float(rospy.get_param("~target_timeout", 0.25))
        self.trajectory_timeout = float(rospy.get_param("~trajectory_timeout", 0.25))
        self.include_current_state = bool(
            rospy.get_param("~include_current_state", False)
        )

        self.device_name = str(rospy.get_param("~device", "cpu"))
        self.checkpoint_path = Path(
            rospy.get_param("~checkpoint", str(DEFAULT_CHECKPOINT))
        ).expanduser().resolve()
        self.urdf_path = Path(
            rospy.get_param("~urdf", str(DEFAULT_URDF))
        ).expanduser().resolve()

        raw_joint_names = rospy.get_param("~joint_names", list(DEFAULT_JOINT_NAMES))
        self.joint_names = [str(name) for name in raw_joint_names]

        self.horizontal_fov_deg = float(
            rospy.get_param("~oracle_horizontal_fov_deg", 50.0)
        )
        self.vertical_fov_deg = float(
            rospy.get_param("~oracle_vertical_fov_deg", 66.0)
        )
        self.oracle_z_min = float(rospy.get_param("~oracle_z_min", 0.20))
        self.oracle_z_max = float(rospy.get_param("~oracle_z_max", 0.70))
        self.oracle_delta = float(rospy.get_param("~oracle_delta", 0.01))

        # Set to zero to disable.  When enabled, every N observer cycles a single
        # selected horizon configuration is checked by central finite difference.
        self.fd_check_every_n = int(rospy.get_param("~fd_check_every_n", 0))
        self.fd_eps = float(rospy.get_param("~fd_eps", 1e-4))
        self.fd_horizon_index = int(rospy.get_param("~fd_horizon_index", 0))

        self._validate_params()

        self._latest_target: Optional[PointStamped] = None
        self._target_received: Optional[rospy.Time] = None
        self._latest_trajectory: Optional[JointTrajectory] = None
        self._trajectory_received: Optional[rospy.Time] = None
        self._sequence = 0

        rospy.loginfo("[ncdf_horizon] Loading frozen visibility NCDF...")
        self._initialize_model_and_oracle()

        self.active_pub = rospy.Publisher("~active", Bool, queue_size=1)
        self.horizon_q_pub = rospy.Publisher(
            "~horizon_q", Float64MultiArray, queue_size=1
        )
        self.time_from_start_pub = rospy.Publisher(
            "~time_from_start", Float64MultiArray, queue_size=1
        )
        self.learned_values_pub = rospy.Publisher(
            "~learned_values", Float64MultiArray, queue_size=1
        )
        self.oracle_values_pub = rospy.Publisher(
            "~oracle_values", Float64MultiArray, queue_size=1
        )
        self.gradient_q_pub = rospy.Publisher(
            "~gradient_q", Float64MultiArray, queue_size=1
        )
        self.gradient_norms_pub = rospy.Publisher(
            "~gradient_norms", Float64MultiArray, queue_size=1
        )
        self.model_time_pub = rospy.Publisher(
            "~model_time_ms", Float32, queue_size=1
        )
        self.oracle_time_pub = rospy.Publisher(
            "~oracle_time_ms", Float32, queue_size=1
        )
        self.total_time_pub = rospy.Publisher(
            "~total_time_ms", Float32, queue_size=1
        )
        self.fd_max_abs_error_pub = rospy.Publisher(
            "~fd_max_abs_error", Float32, queue_size=1
        )
        self.fd_relative_l2_error_pub = rospy.Publisher(
            "~fd_relative_l2_error", Float32, queue_size=1
        )
        self.summary_pub = rospy.Publisher("~summary", String, queue_size=1)

        self.target_sub = rospy.Subscriber(
            self.target_topic, PointStamped, self._target_callback, queue_size=1
        )
        self.trajectory_sub = rospy.Subscriber(
            self.predicted_trajectory_topic,
            JointTrajectory,
            self._trajectory_callback,
            queue_size=1,
        )
        self.timer = rospy.Timer(
            rospy.Duration.from_sec(1.0 / self.update_rate), self._timer_callback
        )

        rospy.logwarn(
            "[ncdf_horizon] PHASE B.1 OBSERVER ONLY: this node publishes no trajectory or control command."
        )
        rospy.loginfo(
            "[ncdf_horizon] target=%s predicted_horizon=%s device=%s",
            self.target_topic,
            self.predicted_trajectory_topic,
            self.device_name,
        )
        rospy.loginfo(
            "[ncdf_horizon] joints=%s include_current=%s rate=%.1fHz",
            ",".join(self.joint_names),
            self.include_current_state,
            self.update_rate,
        )
        rospy.loginfo(
            "[ncdf_horizon] oracle diagnostic: hFOV=%.1f vFOV=%.1f z=[%.2f,%.2f] delta=%.3f",
            self.horizontal_fov_deg,
            self.vertical_fov_deg,
            self.oracle_z_min,
            self.oracle_z_max,
            self.oracle_delta,
        )
        if self.fd_check_every_n > 0:
            rospy.loginfo(
                "[ncdf_horizon] finite-difference check every %d cycles, eps=%.1e, horizon_index=%d",
                self.fd_check_every_n,
                self.fd_eps,
                self.fd_horizon_index,
            )

    def _validate_params(self) -> None:
        if self.update_rate <= 0.0:
            raise ValueError("~update_rate must be positive")
        if self.target_timeout <= 0.0 or self.trajectory_timeout <= 0.0:
            raise ValueError("input timeouts must be positive")
        if len(self.joint_names) != 7 or len(set(self.joint_names)) != 7:
            raise ValueError("~joint_names must contain exactly 7 unique names")
        if self.device_name not in ("cpu", "cuda"):
            raise ValueError("~device must be 'cpu' or 'cuda'")
        if self.device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("~device=cuda requested but torch CUDA is unavailable")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"NCDF checkpoint not found: {self.checkpoint_path}")
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")
        if self.oracle_z_min >= self.oracle_z_max:
            raise ValueError("oracle z_min must be smaller than z_max")
        if self.fd_check_every_n < 0:
            raise ValueError("~fd_check_every_n must be non-negative")
        if self.fd_eps <= 0.0:
            raise ValueError("~fd_eps must be positive")
        if self.fd_horizon_index < 0:
            raise ValueError("~fd_horizon_index must be non-negative")

    def _initialize_model_and_oracle(self) -> None:
        self.device = torch.device(self.device_name)
        checkpoint = torch_load_checkpoint(str(self.checkpoint_path), self.device)
        self.model, self.checkpoint_args = build_model_from_checkpoint(
            checkpoint, self.device
        )

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

        # Warm up both paths so first-use setup is excluded from online timing.
        x_warm = torch.tensor(
            [[0.0, 0.0, 0.5]], device=self.device, dtype=torch.float32
        )
        q_warm = torch.zeros((2, 7), device=self.device, dtype=torch.float32)
        model_value_and_grad_q(x_warm, q_warm, self.model)
        oracle_visibility_g(x_warm, q_warm, self.oracle)
        self._sync_device()
        rospy.loginfo(
            "[ncdf_horizon] Frozen model ready: %s", self.checkpoint_path
        )

    def _sync_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _target_callback(self, msg: PointStamped) -> None:
        if msg is None:
            return
        with self._lock:
            self._latest_target = msg
            self._target_received = rospy.Time.now()

    def _trajectory_callback(self, msg: JointTrajectory) -> None:
        if msg is None or not msg.points:
            return
        with self._lock:
            self._latest_trajectory = msg
            self._trajectory_received = rospy.Time.now()

    def _snapshot_inputs(
        self,
    ) -> Tuple[
        Optional[PointStamped],
        Optional[rospy.Time],
        Optional[JointTrajectory],
        Optional[rospy.Time],
    ]:
        with self._lock:
            return (
                self._latest_target,
                self._target_received,
                self._latest_trajectory,
                self._trajectory_received,
            )

    def _extract_horizon(
        self, trajectory: JointTrajectory
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(trajectory.joint_names) != len(set(trajectory.joint_names)):
            raise ValueError("predicted trajectory contains duplicate joint names")

        name_to_index = {
            name: index for index, name in enumerate(trajectory.joint_names)
        }
        missing = [name for name in self.joint_names if name not in name_to_index]
        if missing:
            raise ValueError(
                "predicted trajectory is missing joints: " + ",".join(missing)
            )
        mapping = [name_to_index[name] for name in self.joint_names]

        start = 0 if self.include_current_state else 1
        if len(trajectory.points) <= start:
            raise ValueError(
                f"predicted trajectory has {len(trajectory.points)} point(s); "
                f"need more than {start}"
            )

        q_rows = []
        times = []
        for point in trajectory.points[start:]:
            if len(point.positions) < len(trajectory.joint_names):
                raise ValueError("trajectory point has incomplete position vector")
            q_rows.append([float(point.positions[index]) for index in mapping])
            times.append(float(point.time_from_start.to_sec()))

        q_horizon = np.asarray(q_rows, dtype=np.float32)
        time_from_start = np.asarray(times, dtype=np.float64)
        if q_horizon.ndim != 2 or q_horizon.shape[1] != 7:
            raise ValueError(f"unexpected q_horizon shape {q_horizon.shape}")
        if not _finite_array(q_horizon) or not _finite_array(time_from_start):
            raise ValueError("predicted horizon contains non-finite values")
        return q_horizon, time_from_start

    def _finite_difference_check(
        self, x_tensor: torch.Tensor, q_tensor: torch.Tensor, grad_tensor: torch.Tensor
    ) -> Tuple[float, float]:
        index = min(self.fd_horizon_index, q_tensor.shape[0] - 1)
        q0 = q_tensor[index].detach()

        batch = q0.unsqueeze(0).repeat(14, 1)
        for joint in range(7):
            batch[2 * joint, joint] += self.fd_eps
            batch[2 * joint + 1, joint] -= self.fd_eps

        values = model_value(x_tensor, batch, self.model)
        fd = torch.empty((7,), device=self.device, dtype=q_tensor.dtype)
        for joint in range(7):
            fd[joint] = (
                values[2 * joint] - values[2 * joint + 1]
            ) / (2.0 * self.fd_eps)

        auto = grad_tensor[index].detach()
        diff = fd - auto
        max_abs = float(torch.max(torch.abs(diff)).item())
        relative_l2 = float(
            torch.linalg.vector_norm(diff).item()
            / max(torch.linalg.vector_norm(fd).item(), 1e-12)
        )
        return max_abs, relative_l2

    def _publish_inactive(self, reason: str) -> None:
        msg = Bool()
        msg.data = False
        self.active_pub.publish(msg)
        rospy.logwarn_throttle(1.0, "[ncdf_horizon] inactive: %s", reason)

    def _timer_callback(self, _event) -> None:
        target, target_received, trajectory, trajectory_received = self._snapshot_inputs()
        now = rospy.Time.now()

        if target is None or target_received is None:
            self._publish_inactive("waiting for active-sensing target")
            return
        if trajectory is None or trajectory_received is None:
            self._publish_inactive("waiting for MPC predicted trajectory")
            return
        if (now - target_received).to_sec() > self.target_timeout:
            self._publish_inactive("stale active-sensing target")
            return
        if (now - trajectory_received).to_sec() > self.trajectory_timeout:
            self._publish_inactive("stale MPC predicted trajectory")
            return

        target_frame = _normalize_frame(target.header.frame_id)
        base_frame = _normalize_frame(self.base_frame)
        if target_frame and target_frame != base_frame:
            self._publish_inactive(
                f"target frame '{target.header.frame_id}' != base frame '{self.base_frame}'"
            )
            return

        x = np.asarray(
            [target.point.x, target.point.y, target.point.z], dtype=np.float32
        )
        if not _finite_array(x):
            self._publish_inactive("target contains non-finite coordinates")
            return

        try:
            q_horizon, time_from_start = self._extract_horizon(trajectory)
        except ValueError as exc:
            self._publish_inactive(str(exc))
            return

        x_tensor = torch.as_tensor(
            x.reshape(1, 3), device=self.device, dtype=torch.float32
        )
        q_tensor = torch.as_tensor(
            q_horizon, device=self.device, dtype=torch.float32
        )

        total_tic = time.perf_counter()
        self._sync_device()
        model_tic = time.perf_counter()
        learned, gradient, _ = model_value_and_grad_q(
            x_tensor, q_tensor, self.model
        )
        self._sync_device()
        model_ms = (time.perf_counter() - model_tic) * 1000.0

        oracle_tic = time.perf_counter()
        oracle = oracle_visibility_g(x_tensor, q_tensor, self.oracle)
        self._sync_device()
        oracle_ms = (time.perf_counter() - oracle_tic) * 1000.0

        self._sequence += 1
        fd_abs = math.nan
        fd_rel = math.nan
        if (
            self.fd_check_every_n > 0
            and self._sequence % self.fd_check_every_n == 0
        ):
            self._sync_device()
            fd_abs, fd_rel = self._finite_difference_check(
                x_tensor, q_tensor, gradient
            )
            self._sync_device()

        total_ms = (time.perf_counter() - total_tic) * 1000.0

        learned_np = learned.detach().cpu().numpy().astype(np.float64)
        oracle_np = oracle.detach().cpu().numpy().astype(np.float64)
        gradient_np = gradient.detach().cpu().numpy().astype(np.float64)
        gradient_norms = np.linalg.norm(gradient_np, axis=1)

        if not (
            _finite_array(learned_np)
            and _finite_array(oracle_np)
            and _finite_array(gradient_np)
            and _finite_array(gradient_norms)
        ):
            self._publish_inactive("NCDF/oracle produced non-finite diagnostics")
            return

        active_msg = Bool()
        active_msg.data = True
        self.active_pub.publish(active_msg)
        self.horizon_q_pub.publish(_matrix_msg(q_horizon, "horizon_step", "joint"))
        self.time_from_start_pub.publish(_vector_msg(time_from_start))
        self.learned_values_pub.publish(_vector_msg(learned_np))
        self.oracle_values_pub.publish(_vector_msg(oracle_np))
        self.gradient_q_pub.publish(
            _matrix_msg(gradient_np, "horizon_step", "joint")
        )
        self.gradient_norms_pub.publish(_vector_msg(gradient_norms))

        model_time_msg = Float32()
        model_time_msg.data = float(model_ms)
        self.model_time_pub.publish(model_time_msg)
        oracle_time_msg = Float32()
        oracle_time_msg.data = float(oracle_ms)
        self.oracle_time_pub.publish(oracle_time_msg)
        total_time_msg = Float32()
        total_time_msg.data = float(total_ms)
        self.total_time_pub.publish(total_time_msg)

        if math.isfinite(fd_abs):
            fd_abs_msg = Float32()
            fd_abs_msg.data = float(fd_abs)
            self.fd_max_abs_error_pub.publish(fd_abs_msg)
            fd_rel_msg = Float32()
            fd_rel_msg.data = float(fd_rel)
            self.fd_relative_l2_error_pub.publish(fd_rel_msg)

        learned_visible_fraction = float(np.mean(learned_np >= 0.0))
        true_visible_fraction = float(np.mean(oracle_np >= 0.0))
        sign_agreement_fraction = float(
            np.mean((learned_np >= 0.0) == (oracle_np >= 0.0))
        )

        summary_text = (
            f"seq={self._sequence} steps={len(learned_np)} "
            f"target={_format_vec(x)} "
            f"f=[{learned_np.min():+.4f},{learned_np.max():+.4f}] "
            f"g=[{oracle_np.min():+.4f},{oracle_np.max():+.4f}] "
            f"learned_visible_frac={learned_visible_fraction:.3f} "
            f"true_visible_frac={true_visible_fraction:.3f} "
            f"sign_agree={sign_agreement_fraction:.3f} "
            f"grad_norm=[{gradient_norms.min():.4e},{gradient_norms.max():.4e}] "
            f"model={model_ms:.3f}ms oracle={oracle_ms:.3f}ms total={total_ms:.3f}ms"
        )
        if math.isfinite(fd_abs):
            summary_text += f" fd_abs={fd_abs:.3e} fd_rel={fd_rel:.3e}"

        summary = String()
        summary.data = summary_text
        self.summary_pub.publish(summary)
        rospy.loginfo_throttle(0.5, "[ncdf_horizon] %s", summary_text)


def main() -> None:
    rospy.init_node("ncdf_horizon_observer")
    try:
        NcdfHorizonObserverNode()
    except Exception as exc:
        rospy.logfatal("[ncdf_horizon] initialization failed: %s", exc)
        raise
    rospy.spin()


if __name__ == "__main__":
    main()
