#!/usr/bin/env python3
"""Stage-II/III 7-DoF NCDF action-QP core for CAREPlanner.

Default mode is dry-run. When ~execute:=true, this node becomes the sole
velocity-command publisher for the arm:
- no fresh active-sensing target -> nominal executor action;
- fresh target -> NCDF-QP corrected action;
- stale robot/trajectory input, invalid target frame, or active-QP failure -> zero.

Never run ~execute:=true together with TrajectoryExecutionManager because both
would publish to the same velocity-controller command topic.
"""

from __future__ import annotations

import copy
import math
import sys
import threading
import time
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import rospkg
import rospy
import torch
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32, Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory

PACKAGE_DIR = Path(rospkg.RosPack().get_path("care_visibility_cdf")).resolve()
SCRIPT_DIR = PACKAGE_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_direct_vs_projection_ascent import (
    DEFAULT_SENSOR_FRAMES,
    PinocchioFOVOracle,
    build_model_from_checkpoint,
    torch_load_checkpoint,
)
from export_ncdf_l4casadi import build_casadi_functions
from test_ncdf_local_optimizer import DEFAULT_JOINT_NAMES, DEFAULT_URDF, read_joint_limits

DEFAULT_CHECKPOINT = PACKAGE_DIR / "checkpoints" / "exp1_yiming_k500_fov_signed" / "final.pt"
DEFAULT_BUILD_DIR = PACKAGE_DIR / "generated" / "l4casadi_ncdf_action_qp_dryrun"
DEFAULT_VELOCITY_LIMITS = np.asarray([2.0] * 7, dtype=np.float64)
DEFAULT_ACCELERATION_LIMITS = np.asarray(
    [8.0, 8.0, 10.0, 10.0, 15.0, 15.0, 15.0], dtype=np.float64
)
DEFAULT_TRACKING_WEIGHTS = np.ones(7, dtype=np.float64)


def _value_grad(value_grad_fn, x: np.ndarray, q: np.ndarray) -> Tuple[float, np.ndarray]:
    outputs = value_grad_fn(
        np.asarray(x, dtype=np.float64).reshape(1, 3),
        np.asarray(q, dtype=np.float64).reshape(1, 7),
    )
    value = float(np.asarray(outputs[0]).reshape(()))
    grad = np.asarray(outputs[1], dtype=np.float64).reshape(7)
    return value, grad


def _format_vec(values: Sequence[float], precision: int = 6) -> str:
    return "[" + ",".join(f"{float(v):+.{precision}f}" for v in values) + "]"


def _as_7_vector(name: str, value, default: np.ndarray) -> np.ndarray:
    if value is None:
        arr = default.copy()
    elif np.isscalar(value):
        arr = np.full(7, float(value), dtype=np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (7,):
        raise ValueError(f"{name} must contain exactly 7 values")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


class NcdfActionQPDryRunNode:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cycle_id = 0

        self.target_topic = str(
            rospy.get_param("~target_topic", "/care_planner/active_sensing/target_point")
        )
        self.joint_state_topic = str(
            rospy.get_param("~joint_state_topic", "/care_arm/joint_states")
        )
        self.command_trajectory_topic = str(
            rospy.get_param(
                "~command_trajectory_topic",
                "/care_planner/command_trajectory_candidate",
            )
        )
        self.base_frame = str(rospy.get_param("~base_frame", "base_link"))
        self.update_rate = float(rospy.get_param("~update_rate", 20.0))
        self.target_timeout = float(rospy.get_param("~target_timeout", 0.20))
        self.joint_state_timeout = float(rospy.get_param("~joint_state_timeout", 0.20))
        self.trajectory_timeout = float(rospy.get_param("~trajectory_timeout", 0.20))

        self.control_dt = float(rospy.get_param("~control_dt", 0.05))
        self.lookahead_time = float(rospy.get_param("~lookahead_time", self.control_dt))
        self.position_feedback_gain = float(
            rospy.get_param("~position_feedback_gain", 2.5)
        )
        self.max_command_velocity = float(
            rospy.get_param("~max_command_velocity", 2.0)
        )
        self.visibility_velocity_gain = float(
            rospy.get_param("~visibility_velocity_gain", 0.20)
        )
        self.grad_eps = float(rospy.get_param("~grad_eps", 1e-10))

        self.execute = bool(rospy.get_param("~execute", False))
        self.velocity_command_topic = str(
            rospy.get_param(
                "~velocity_command_topic",
                "/care_arm/arm_group_velocity_controller/command",
            )
        )

        self.velocity_limits = _as_7_vector(
            "velocity_limits",
            rospy.get_param("~velocity_limits", None),
            DEFAULT_VELOCITY_LIMITS,
        )
        self.velocity_limits = np.minimum(
            self.velocity_limits,
            np.full(7, self.max_command_velocity, dtype=np.float64),
        )
        self.acceleration_limits = _as_7_vector(
            "acceleration_limits",
            rospy.get_param("~acceleration_limits", None),
            DEFAULT_ACCELERATION_LIMITS,
        )
        self.correction_acceleration_limits = _as_7_vector(
            "correction_acceleration_limits",
            rospy.get_param("~correction_acceleration_limits", None),
            DEFAULT_ACCELERATION_LIMITS,
        )
        self.tracking_weights = _as_7_vector(
            "tracking_weights",
            rospy.get_param("~tracking_weights", None),
            DEFAULT_TRACKING_WEIGHTS,
        )

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
            rospy.get_param("~model_name", "care_visibility_ncdf_action_qp_dryrun")
        )

        self.horizontal_fov_deg = float(
            rospy.get_param("~oracle_horizontal_fov_deg", 50.0)
        )
        self.vertical_fov_deg = float(
            rospy.get_param("~oracle_vertical_fov_deg", 66.0)
        )
        self.oracle_z_min = float(rospy.get_param("~oracle_z_min", 0.20))
        self.oracle_z_max = float(rospy.get_param("~oracle_z_max", 0.70))
        self.oracle_delta = float(rospy.get_param("~oracle_delta", 0.01))

        self.joint_names = [
            str(v) for v in rospy.get_param("~joint_names", list(DEFAULT_JOINT_NAMES))
        ]
        self._validate_params()

        self._latest_target = None
        self._target_received = None
        self._latest_joint_state = None
        self._joint_state_received = None
        self._latest_trajectory = None
        self._trajectory_received = None

        rospy.loginfo("[ncdf_action_qp] Loading frozen NCDF / L4CasADi interface...")
        self._initialize_model_and_oracle()

        self.active_pub = rospy.Publisher("~active", Bool, queue_size=1)
        self.gradient_pub = rospy.Publisher(
            "~gradient_q", Float64MultiArray, queue_size=1
        )
        self.nominal_action_pub = rospy.Publisher(
            "~nominal_action", Float64MultiArray, queue_size=1
        )
        self.corrected_action_pub = rospy.Publisher(
            "~corrected_action", Float64MultiArray, queue_size=1
        )
        self.delta_action_pub = rospy.Publisher(
            "~delta_action", Float64MultiArray, queue_size=1
        )
        self.predicted_nominal_state_pub = rospy.Publisher(
            "~predicted_nominal_joint_state", JointState, queue_size=1
        )
        self.predicted_corrected_state_pub = rospy.Publisher(
            "~predicted_corrected_joint_state", JointState, queue_size=1
        )
        self.visibility_rate_nominal_pub = rospy.Publisher(
            "~visibility_rate_nominal", Float32, queue_size=1
        )
        self.visibility_rate_corrected_pub = rospy.Publisher(
            "~visibility_rate_corrected", Float32, queue_size=1
        )
        self.delta_visibility_rate_pub = rospy.Publisher(
            "~delta_visibility_rate", Float32, queue_size=1
        )
        self.delta_f_pub = rospy.Publisher("~delta_f", Float32, queue_size=1)
        self.delta_g_pub = rospy.Publisher("~delta_g", Float32, queue_size=1)
        self.qp_time_pub = rospy.Publisher("~qp_time_ms", Float32, queue_size=1)
        self.ncdf_time_pub = rospy.Publisher("~ncdf_time_ms", Float32, queue_size=1)
        self.oracle_time_pub = rospy.Publisher(
            "~oracle_time_ms", Float32, queue_size=1
        )
        self.summary_pub = rospy.Publisher("~summary", String, queue_size=1)

        self.command_pub = None
        if self.execute:
            self.command_pub = rospy.Publisher(
                self.velocity_command_topic, Float64MultiArray, queue_size=1
            )

        rospy.Subscriber(
            self.target_topic, PointStamped, self._target_callback, queue_size=1
        )
        rospy.Subscriber(
            self.joint_state_topic, JointState, self._joint_state_callback, queue_size=1
        )
        rospy.Subscriber(
            self.command_trajectory_topic,
            JointTrajectory,
            self._trajectory_callback,
            queue_size=1,
        )
        self.timer = rospy.Timer(
            rospy.Duration.from_sec(1.0 / self.update_rate), self._timer_callback
        )
        rospy.on_shutdown(self._on_shutdown)

        rospy.loginfo(
            "[ncdf_action_qp] 7D q/u joints=%s", ",".join(self.joint_names)
        )
        rospy.loginfo(
            "[ncdf_action_qp] dt=%.3fs lookahead=%.3fs gain=%.3frad/s Kp=%.3f",
            self.control_dt,
            self.lookahead_time,
            self.visibility_velocity_gain,
            self.position_feedback_gain,
        )
        if self.execute:
            rospy.logwarn(
                "[ncdf_action_qp] EXECUTION ENABLED -> %s. "
                "TrajectoryExecutionManager MUST be disabled.",
                self.velocity_command_topic,
            )
        else:
            rospy.logwarn(
                "[ncdf_action_qp] DRY RUN: no robot command is published."
            )

    def _validate_params(self) -> None:
        if len(self.joint_names) != 7 or len(set(self.joint_names)) != 7:
            raise ValueError("joint_names must contain exactly 7 unique joints")
        if self.update_rate <= 0.0 or self.control_dt <= 0.0:
            raise ValueError("update_rate/control_dt must be positive")
        if self.lookahead_time < 0.0:
            raise ValueError("lookahead_time must be non-negative")
        if min(
            self.target_timeout, self.joint_state_timeout, self.trajectory_timeout
        ) <= 0.0:
            raise ValueError("input timeouts must be positive")
        if self.device_name not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if self.device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint_path}")
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")

    def _initialize_model_and_oracle(self) -> None:
        self.device = torch.device(self.device_name)
        self.q_min, self.q_max = read_joint_limits(self.urdf_path, self.joint_names)
        ckpt = torch_load_checkpoint(str(self.checkpoint_path), self.device)
        model, _ = build_model_from_checkpoint(ckpt, self.device)
        _, _, _, self.value_grad_fn = build_casadi_functions(
            model=model,
            device=self.device_name,
            build_dir=self.build_dir,
            model_name=self.model_name,
        )
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
        rospy.loginfo("[ncdf_action_qp] Frozen NCDF and debug oracle ready.")

    def _target_callback(self, msg: PointStamped) -> None:
        with self._lock:
            self._latest_target = copy.deepcopy(msg)
            self._target_received = rospy.Time.now()

    def _joint_state_callback(self, msg: JointState) -> None:
        with self._lock:
            self._latest_joint_state = copy.deepcopy(msg)
            self._joint_state_received = rospy.Time.now()

    def _trajectory_callback(self, msg: JointTrajectory) -> None:
        with self._lock:
            self._latest_trajectory = copy.deepcopy(msg)
            self._trajectory_received = rospy.Time.now()

    @staticmethod
    def _age(now: rospy.Time, stamp) -> float:
        return math.inf if stamp is None else max(0.0, (now - stamp).to_sec())

    def _publish_velocity(self, u: np.ndarray) -> None:
        if self.command_pub is not None:
            self.command_pub.publish(self._array_msg(u))

    def _publish_zero(self, reason: str) -> None:
        self.active_pub.publish(Bool(data=False))
        if self.execute:
            self._publish_velocity(np.zeros(7, dtype=np.float64))
        rospy.logwarn_throttle(1.0, "[ncdf_action_qp] zero/idle: %s", reason)

    def _ordered_measured_state(self, msg: JointState):
        index = {name: i for i, name in enumerate(msg.name)}
        if any(name not in index for name in self.joint_names):
            return None
        q = np.zeros(7, dtype=np.float64)
        dq = np.zeros(7, dtype=np.float64)
        for i, name in enumerate(self.joint_names):
            j = index[name]
            if j >= len(msg.position):
                return None
            q[i] = float(msg.position[j])
            if j < len(msg.velocity):
                dq[i] = float(msg.velocity[j])
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq)):
            return None
        return q, dq

    def _sample_trajectory(self, msg: JointTrajectory, t: float):
        if not msg.points or not msg.joint_names:
            return None
        index = {name: i for i, name in enumerate(msg.joint_names)}
        if any(name not in index for name in self.joint_names):
            return None

        def vec(point, field_name: str, allow_zero: bool):
            field = getattr(point, field_name)
            if not field:
                return np.zeros(7, dtype=np.float64) if allow_zero else None
            out = np.zeros(7, dtype=np.float64)
            for k, name in enumerate(self.joint_names):
                src = index[name]
                if src >= len(field):
                    if allow_zero:
                        out[k] = 0.0
                    else:
                        return None
                else:
                    out[k] = float(field[src])
            return out if np.all(np.isfinite(out)) else None

        times = np.asarray(
            [p.time_from_start.to_sec() for p in msg.points], dtype=np.float64
        )
        if np.any(np.diff(times) < -1e-12):
            return None

        if len(msg.points) == 1 or t <= times[0]:
            p = msg.points[0]
            vals = (
                vec(p, "positions", False),
                vec(p, "velocities", True),
                vec(p, "accelerations", True),
            )
            return None if any(v is None for v in vals) else vals

        if t >= times[-1]:
            p = msg.points[-1]
            vals = (
                vec(p, "positions", False),
                vec(p, "velocities", True),
                vec(p, "accelerations", True),
            )
            return None if any(v is None for v in vals) else vals

        upper = int(np.searchsorted(times, t, side="left"))
        lower = upper - 1
        h = float(times[upper] - times[lower])
        if h <= 1e-12:
            return None
        s = float((t - times[lower]) / h)

        p0, p1 = msg.points[lower], msg.points[upper]
        vals = [
            vec(p0, "positions", False),
            vec(p1, "positions", False),
            vec(p0, "velocities", True),
            vec(p1, "velocities", True),
            vec(p0, "accelerations", True),
            vec(p1, "accelerations", True),
        ]
        if any(v is None for v in vals):
            return None
        q0, q1, dq0, dq1, ddq0, ddq1 = vals
        return (
            (1.0 - s) * q0 + s * q1,
            (1.0 - s) * dq0 + s * dq1,
            (1.0 - s) * ddq0 + s * ddq1,
        )

    def _nominal_executor_action(
        self, q_current: np.ndarray, q_ref: np.ndarray, dq_ref: np.ndarray
    ):
        raw = dq_ref + self.position_feedback_gain * (q_ref - q_current)
        clipped = np.clip(raw, -self.velocity_limits, self.velocity_limits)
        return raw, clipped

    def _solve_action_qp(
        self,
        direction: np.ndarray,
        u_nominal: np.ndarray,
        q_current: np.ndarray,
        dq_current: np.ndarray,
    ):
        du_free = (
            self.visibility_velocity_gain * direction / self.tracking_weights
        )

        lower = -self.velocity_limits - u_nominal
        upper = self.velocity_limits - u_nominal

        physical_dv = self.acceleration_limits * self.control_dt
        lower = np.maximum(
            lower, dq_current - physical_dv - u_nominal
        )
        upper = np.minimum(
            upper, dq_current + physical_dv - u_nominal
        )

        lower = np.maximum(
            lower,
            (self.q_min - q_current) / self.control_dt - u_nominal,
        )
        upper = np.minimum(
            upper,
            (self.q_max - q_current) / self.control_dt - u_nominal,
        )

        correction_dv = self.correction_acceleration_limits * self.control_dt
        lower = np.maximum(lower, -correction_dv)
        upper = np.minimum(upper, correction_dv)

        if np.any(lower > upper + 1e-12):
            raise RuntimeError("empty action-QP feasible interval")

        du = np.clip(du_free, lower, upper)
        return u_nominal + du, du, lower, upper

    def _oracle_g(self, x: np.ndarray, q: np.ndarray) -> float:
        x_t = torch.as_tensor(
            x, dtype=torch.float32, device=self.device
        ).reshape(1, 3)
        q_t = torch.as_tensor(
            q, dtype=torch.float32, device=self.device
        ).reshape(1, 7)
        raw_margin, _ = self.oracle.signed_fov_margins(x_t, q_t)
        g = torch.max(raw_margin - self.oracle.delta, dim=-1).values
        return float(g.reshape(()).detach().cpu())

    def _joint_state(self, q: np.ndarray, stamp: rospy.Time) -> JointState:
        msg = JointState()
        msg.header.stamp = stamp
        msg.header.frame_id = self.base_frame
        msg.name = list(self.joint_names)
        msg.position = [float(v) for v in q]
        return msg

    @staticmethod
    def _array_msg(v: np.ndarray) -> Float64MultiArray:
        return Float64MultiArray(data=[float(x) for x in v])

    def _timer_callback(self, _event) -> None:
        now = rospy.Time.now()
        with self._lock:
            target = copy.deepcopy(self._latest_target)
            target_received = self._target_received
            js = copy.deepcopy(self._latest_joint_state)
            joint_received = self._joint_state_received
            traj = copy.deepcopy(self._latest_trajectory)
            trajectory_received = self._trajectory_received

        if js is None or self._age(now, joint_received) > self.joint_state_timeout:
            self._publish_zero("waiting for fresh JointState")
            return

        measured = self._ordered_measured_state(js)
        if measured is None:
            self._publish_zero("invalid 7-D measured state")
            return
        q_current, dq_current = measured

        if traj is None or self._age(now, trajectory_received) > self.trajectory_timeout:
            self._publish_zero("waiting for fresh command trajectory")
            return

        sampled = self._sample_trajectory(traj, self.lookahead_time)
        if sampled is None:
            self._publish_zero("cannot sample nominal trajectory")
            return
        q_ref, dq_ref, _ = sampled
        u_nom_raw, u_nom = self._nominal_executor_action(
            q_current, q_ref, dq_ref
        )

        target_fresh = (
            target is not None
            and self._age(now, target_received) <= self.target_timeout
        )
        if not target_fresh:
            self.active_pub.publish(Bool(data=False))
            if self.execute:
                self._publish_velocity(u_nom)
            rospy.loginfo_throttle(
                1.0,
                "[ncdf_action_qp] no fresh active target: nominal action%s",
                " EXECUTED" if self.execute else " (dry-run)",
            )
            return

        if target.header.frame_id and target.header.frame_id != self.base_frame:
            self._publish_zero(
                f"target frame {target.header.frame_id} != {self.base_frame}"
            )
            return

        x = np.asarray(
            [target.point.x, target.point.y, target.point.z],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(x)):
            self._publish_zero("non-finite active target")
            return

        q_nom_next = q_current + self.control_dt * u_nom

        try:
            tic = time.perf_counter()
            f_nom, grad = _value_grad(self.value_grad_fn, x, q_nom_next)
            grad_norm = float(np.linalg.norm(grad))
            ncdf_ms = 1000.0 * (time.perf_counter() - tic)
            if not np.isfinite(grad_norm) or grad_norm <= self.grad_eps:
                raise RuntimeError(f"NCDF gradient too small: {grad_norm}")
            direction = grad / grad_norm

            tic = time.perf_counter()
            u_corr, du, lo, hi = self._solve_action_qp(
                direction, u_nom, q_current, dq_current
            )
            qp_ms = 1000.0 * (time.perf_counter() - tic)

            q_corr_next = q_current + self.control_dt * u_corr
            f_corr, _ = _value_grad(self.value_grad_fn, x, q_corr_next)

            tic = time.perf_counter()
            g_nom = self._oracle_g(x, q_nom_next)
            g_corr = self._oracle_g(x, q_corr_next)
            oracle_ms = 1000.0 * (time.perf_counter() - tic)
        except Exception as exc:
            self._publish_zero(f"active QP failed: {exc}")
            rospy.logerr_throttle(
                1.0, "[ncdf_action_qp] active QP failed: %s", exc
            )
            return

        rate_nom = float(np.dot(grad, u_nom))
        rate_corr = float(np.dot(grad, u_corr))
        drate = float(np.dot(grad, du))
        df = float(f_corr - f_nom)
        dg = float(g_corr - g_nom)

        nom_acc = (u_nom - dq_current) / self.control_dt
        corr_acc = (u_corr - dq_current) / self.control_dt
        accel_ratio = float(
            np.max(np.abs(corr_acc) / self.acceleration_limits)
        )
        nom_accel_ratio = float(
            np.max(np.abs(nom_acc) / self.acceleration_limits)
        )
        active_bounds = int(
            np.count_nonzero(
                (np.abs(du - lo) <= 1e-6)
                | (np.abs(du - hi) <= 1e-6)
            )
        )
        accel_ok = bool(accel_ratio <= 1.0 + 1e-8)

        self.active_pub.publish(Bool(data=True))
        self.gradient_pub.publish(self._array_msg(grad))
        self.nominal_action_pub.publish(self._array_msg(u_nom))
        self.corrected_action_pub.publish(self._array_msg(u_corr))
        self.delta_action_pub.publish(self._array_msg(du))
        self.predicted_nominal_state_pub.publish(
            self._joint_state(q_nom_next, now)
        )
        self.predicted_corrected_state_pub.publish(
            self._joint_state(q_corr_next, now)
        )
        self.visibility_rate_nominal_pub.publish(Float32(data=rate_nom))
        self.visibility_rate_corrected_pub.publish(Float32(data=rate_corr))
        self.delta_visibility_rate_pub.publish(Float32(data=drate))
        self.delta_f_pub.publish(Float32(data=df))
        self.delta_g_pub.publish(Float32(data=dg))
        self.qp_time_pub.publish(Float32(data=qp_ms))
        self.ncdf_time_pub.publish(Float32(data=ncdf_ms))
        self.oracle_time_pub.publish(Float32(data=oracle_ms))

        if self.execute:
            if not accel_ok:
                self._publish_velocity(np.zeros(7, dtype=np.float64))
                rospy.logerr_throttle(
                    1.0,
                    "[ncdf_action_qp] refusing command: accel_ratio=%.3f",
                    accel_ratio,
                )
            else:
                self._publish_velocity(u_corr)

        self._cycle_id += 1
        summary = (
            f"seq={self._cycle_id} dt={self.control_dt:.8f} "
            f"x={_format_vec(x)} "
            f"q_cur={_format_vec(q_current)} "
            f"dq_meas={_format_vec(dq_current)} "
            f"q_nom_next={_format_vec(q_nom_next)} "
            f"q_corr_next={_format_vec(q_corr_next)} "
            f"grad={_format_vec(grad)} grad_norm={grad_norm:.8e} "
            f"u_nom={_format_vec(u_nom)} du={_format_vec(du)} "
            f"u_corr={_format_vec(u_corr)} "
            f"rate={rate_nom:+.8f}->{rate_corr:+.8f} "
            f"drate={drate:+.8f} "
            f"f_next={f_nom:+.8f}->{f_corr:+.8f} df={df:+.8f} "
            f"g_next={g_nom:+.8f}->{g_corr:+.8f} dg={dg:+.8f} "
            f"learned_up={int(df > 1e-8)} "
            f"oracle_up={int(dg > 1e-8)} "
            f"f_up_g_down={int(df > 1e-8 and dg < -1e-8)} "
            f"active_bounds={active_bounds} accel_ok={int(accel_ok)} "
            f"accel_ratio={accel_ratio:.6f} "
            f"nom_accel_ratio={nom_accel_ratio:.6f} "
            f"nom_vel_clip={int(np.max(np.abs(u_nom_raw-u_nom)) > 1e-8)} "
            f"qpert_l2={np.linalg.norm(q_corr_next-q_nom_next):.8f} "
            f"execute={int(self.execute)} "
            f"ncdf={ncdf_ms:.6f}ms qp={qp_ms:.6f}ms "
            f"oracle={oracle_ms:.6f}ms"
        )
        self.summary_pub.publish(String(data=summary))
        rospy.loginfo_throttle(
            0.5, "[ncdf_action_qp] %s", summary
        )

    def _on_shutdown(self) -> None:
        if self.execute and self.command_pub is not None:
            try:
                for _ in range(3):
                    self._publish_velocity(np.zeros(7, dtype=np.float64))
                    rospy.sleep(0.01)
            except Exception:
                pass


def main() -> None:
    rospy.init_node("ncdf_action_qp_dryrun")
    NcdfActionQPDryRunNode()
    rospy.spin()
