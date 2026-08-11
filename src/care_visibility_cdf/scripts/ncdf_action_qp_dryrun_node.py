#!/usr/bin/env python3
"""Stage-II 7-DoF action-space NCDF dry run for CAREPlanner.

This node is NON-ACTUATING. It mirrors the current short-step velocity backend,
constructs the nominal velocity command at the executor's 0.05-s lookahead,
and applies a one-step 7-D visibility action QP.

State / action
--------------
    q in R^7
    u = qdot in R^7

There are no mobile-base state dimensions.

Nominal action
--------------
The current TrajectoryExecutionManager computes

    u_nom = qdot_ref + Kp * (q_ref - q_measured)

and then clamps the command velocity. This node reproduces that calculation for
debug only.

Visibility QP
-------------
At q_ref, evaluate the frozen learned field and gradient

    grad = d f(x*, q_ref) / d q
    d    = grad / ||grad||.

The optimization variable is the ACTION CORRECTION du in R^7:

    min_du  0.5 * du^T W du - visibility_velocity_gain * d^T du

subject to elementwise bounds from

  1) total command velocity limits,
  2) one-control-cycle correction acceleration budget,
  3) next-step joint limits for q_ref + control_dt * du.

With diagonal W and box constraints this is a convex QP with an exact closed
form: compute the unconstrained minimizer and clip each joint to its feasible
interval. No numerical QP package is needed for this Stage-II one-step test.

The default visibility_velocity_gain is 0.2 rad/s. With control_dt=0.05 s and
a unit-norm direction, the unconstrained configuration perturbation has L2 norm
0.01 rad, matching ONE Stage-I Direct step (not the five-step 0.05-rad Stage-I
diagnostic budget).

For validation only, the node also evaluates learned f and analytic oracle g at

    q_nominal  = q_ref
    q_action   = q_ref + control_dt * du.

The oracle never participates in the QP.
"""

from __future__ import annotations

import copy
import math
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

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


def _format_vec(values: Sequence[float], precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(v):+.{precision}f}" for v in values) + "]"


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

        # Inputs and timing.
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

        # Match the current execution backend exactly by default.
        self.control_dt = float(rospy.get_param("~control_dt", 0.05))
        self.lookahead_time = float(rospy.get_param("~lookahead_time", self.control_dt))
        self.position_feedback_gain = float(
            rospy.get_param("~position_feedback_gain", 2.5)
        )
        self.max_command_velocity = float(
            rospy.get_param("~max_command_velocity", 2.0)
        )

        # Stage-II QP parameters.
        self.visibility_velocity_gain = float(
            rospy.get_param("~visibility_velocity_gain", 0.20)
        )
        self.grad_eps = float(rospy.get_param("~grad_eps", 1e-10))
        self.velocity_limits = _as_7_vector(
            "velocity_limits",
            rospy.get_param("~velocity_limits", None),
            DEFAULT_VELOCITY_LIMITS,
        )
        # The actual executor also globally clamps to max_command_velocity.
        self.velocity_limits = np.minimum(
            self.velocity_limits,
            np.full(7, self.max_command_velocity, dtype=np.float64),
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

        # Frozen model / oracle.
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

        raw_joint_names = rospy.get_param("~joint_names", list(DEFAULT_JOINT_NAMES))
        self.joint_names = [str(v) for v in raw_joint_names]
        self._validate_params()

        # Cached ROS data.
        self._latest_target: Optional[PointStamped] = None
        self._target_received: Optional[rospy.Time] = None
        self._latest_joint_state: Optional[JointState] = None
        self._joint_state_received: Optional[rospy.Time] = None
        self._latest_trajectory: Optional[JointTrajectory] = None
        self._trajectory_received: Optional[rospy.Time] = None

        rospy.loginfo("[ncdf_action_qp] Loading frozen NCDF / L4CasADi interface...")
        self._initialize_model_and_oracle()

        # Debug-only outputs.
        self.active_pub = rospy.Publisher("~active", Bool, queue_size=1)
        self.gradient_pub = rospy.Publisher("~gradient_q", Float64MultiArray, queue_size=1)
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
        self.oracle_time_pub = rospy.Publisher("~oracle_time_ms", Float32, queue_size=1)
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

        rospy.logwarn(
            "[ncdf_action_qp] STAGE-II DRY RUN ONLY: no robot command is published."
        )
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
        rospy.loginfo(
            "[ncdf_action_qp] velocity_limits=%s correction_accel=%s",
            _format_vec(self.velocity_limits),
            _format_vec(self.correction_acceleration_limits),
        )

    def _validate_params(self) -> None:
        if len(self.joint_names) != 7 or len(set(self.joint_names)) != 7:
            raise ValueError("joint_names must contain exactly 7 unique joints")
        if self.update_rate <= 0.0 or self.control_dt <= 0.0:
            raise ValueError("update_rate and control_dt must be positive")
        if self.lookahead_time < 0.0:
            raise ValueError("lookahead_time must be non-negative")
        if self.target_timeout <= 0.0 or self.joint_state_timeout <= 0.0:
            raise ValueError("target/joint_state timeouts must be positive")
        if self.trajectory_timeout <= 0.0:
            raise ValueError("trajectory_timeout must be positive")
        if self.position_feedback_gain < 0.0:
            raise ValueError("position_feedback_gain must be non-negative")
        if self.max_command_velocity <= 0.0:
            raise ValueError("max_command_velocity must be positive")
        if self.visibility_velocity_gain < 0.0:
            raise ValueError("visibility_velocity_gain must be non-negative")
        if np.any(self.velocity_limits <= 0.0):
            raise ValueError("velocity_limits must be positive")
        if np.any(self.correction_acceleration_limits <= 0.0):
            raise ValueError("correction_acceleration_limits must be positive")
        if np.any(self.tracking_weights <= 0.0):
            raise ValueError("tracking_weights must be positive")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint_path}")
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")
        if self.device_name not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if self.device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

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
    def _age(now: rospy.Time, stamp: Optional[rospy.Time]) -> float:
        if stamp is None:
            return math.inf
        return max(0.0, (now - stamp).to_sec())

    def _publish_inactive(self, reason: str) -> None:
        self.active_pub.publish(Bool(data=False))
        rospy.logwarn_throttle(1.0, "[ncdf_action_qp] inactive: %s", reason)

    def _ordered_measured_state(
        self, msg: JointState
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        index: Dict[str, int] = {name: i for i, name in enumerate(msg.name)}
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

    def _sample_trajectory(
        self, msg: JointTrajectory, t: float
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if not msg.points or not msg.joint_names:
            return None
        index: Dict[str, int] = {name: i for i, name in enumerate(msg.joint_names)}
        if any(name not in index for name in self.joint_names):
            return None

        def vec(point, field_name: str, allow_missing_zero: bool) -> Optional[np.ndarray]:
            field = getattr(point, field_name)
            if not field:
                return np.zeros(7, dtype=np.float64) if allow_missing_zero else None
            out = np.zeros(7, dtype=np.float64)
            for k, name in enumerate(self.joint_names):
                src = index[name]
                if src >= len(field):
                    if allow_missing_zero:
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
            q = vec(p, "positions", False)
            dq = vec(p, "velocities", True)
            ddq = vec(p, "accelerations", True)
            return None if q is None or dq is None or ddq is None else (q, dq, ddq)

        if t >= times[-1]:
            p = msg.points[-1]
            q = vec(p, "positions", False)
            dq = vec(p, "velocities", True)
            ddq = vec(p, "accelerations", True)
            return None if q is None or dq is None or ddq is None else (q, dq, ddq)

        upper = int(np.searchsorted(times, t, side="left"))
        lower = upper - 1
        t0, t1 = float(times[lower]), float(times[upper])
        h = t1 - t0
        if h <= 1e-12:
            return None
        s = (t - t0) / h

        p0, p1 = msg.points[lower], msg.points[upper]
        q0, q1 = vec(p0, "positions", False), vec(p1, "positions", False)
        dq0, dq1 = vec(p0, "velocities", True), vec(p1, "velocities", True)
        ddq0, ddq1 = vec(p0, "accelerations", True), vec(p1, "accelerations", True)
        if any(v is None for v in (q0, q1, dq0, dq1, ddq0, ddq1)):
            return None
        return (
            (1.0 - s) * q0 + s * q1,
            (1.0 - s) * dq0 + s * dq1,
            (1.0 - s) * ddq0 + s * ddq1,
        )

    def _nominal_executor_action(
        self, q_current: np.ndarray, q_ref: np.ndarray, dq_ref: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        raw = dq_ref + self.position_feedback_gain * (q_ref - q_current)
        clipped = np.clip(raw, -self.velocity_limits, self.velocity_limits)
        return raw, clipped

    def _solve_action_qp(
        self,
        direction: np.ndarray,
        u_nominal: np.ndarray,
        q_nominal: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return corrected action, du, lower_du, upper_du.

        Exact solution for diagonal quadratic objective + box constraints.
        """
        # Unconstrained minimizer of
        # 0.5 * du^T W du - gain * direction^T du.
        du_free = self.visibility_velocity_gain * direction / self.tracking_weights

        # Total action velocity limits.
        lower = -self.velocity_limits - u_nominal
        upper = self.velocity_limits - u_nominal

        # Conservative one-cycle budget for the additional visibility action.
        correction_dv = self.correction_acceleration_limits * self.control_dt
        lower = np.maximum(lower, -correction_dv)
        upper = np.minimum(upper, correction_dv)

        # Joint limits for the perturbation around the nominal next state.
        lower = np.maximum(lower, (self.q_min - q_nominal) / self.control_dt)
        upper = np.minimum(upper, (self.q_max - q_nominal) / self.control_dt)

        if np.any(lower > upper + 1e-12):
            raise RuntimeError(
                "empty action-QP feasible interval: "
                f"lower={lower.tolist()} upper={upper.tolist()}"
            )

        du = np.clip(du_free, lower, upper)
        u_corrected = u_nominal + du
        return u_corrected, du, lower, upper

    def _oracle_g(self, x: np.ndarray, q: np.ndarray) -> float:
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device).reshape(1, 3)
        q_t = torch.as_tensor(q, dtype=torch.float32, device=self.device).reshape(1, 7)
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
            joint_state = copy.deepcopy(self._latest_joint_state)
            joint_received = self._joint_state_received
            trajectory = copy.deepcopy(self._latest_trajectory)
            trajectory_received = self._trajectory_received

        if target is None:
            self._publish_inactive("waiting for active-sensing target")
            return
        if self._age(now, target_received) > self.target_timeout:
            self._publish_inactive("active-sensing target is stale")
            return
        if target.header.frame_id and target.header.frame_id != self.base_frame:
            self._publish_inactive(
                f"target frame {target.header.frame_id} != {self.base_frame}"
            )
            return

        if joint_state is None or self._age(now, joint_received) > self.joint_state_timeout:
            self._publish_inactive("waiting for fresh JointState")
            return
        measured = self._ordered_measured_state(joint_state)
        if measured is None:
            self._publish_inactive("invalid 7-D measured state")
            return
        q_current, dq_current = measured

        if trajectory is None or self._age(now, trajectory_received) > self.trajectory_timeout:
            self._publish_inactive("waiting for fresh command trajectory")
            return
        sampled = self._sample_trajectory(trajectory, self.lookahead_time)
        if sampled is None:
            self._publish_inactive("cannot sample nominal trajectory")
            return
        q_ref, dq_ref, ddq_ref = sampled

        x = np.asarray(
            [target.point.x, target.point.y, target.point.z], dtype=np.float64
        )
        if not np.all(np.isfinite(x)):
            self._publish_inactive("non-finite target")
            return

        u_nom_raw, u_nom = self._nominal_executor_action(q_current, q_ref, dq_ref)

        try:
            ncdf_tic = time.perf_counter()
            f0, grad = _value_grad(self.value_grad_fn, x, q_ref)
            grad_norm = float(np.linalg.norm(grad))
            ncdf_ms = 1000.0 * (time.perf_counter() - ncdf_tic)
            if not np.isfinite(grad_norm) or grad_norm <= self.grad_eps:
                self._publish_inactive(f"NCDF gradient norm too small ({grad_norm})")
                return
            direction = grad / grad_norm

            qp_tic = time.perf_counter()
            u_corr, du, du_lower, du_upper = self._solve_action_qp(
                direction, u_nom, q_ref
            )
            qp_ms = 1000.0 * (time.perf_counter() - qp_tic)

            q_action = np.clip(q_ref + self.control_dt * du, self.q_min, self.q_max)
            f1, _ = _value_grad(self.value_grad_fn, x, q_action)

            oracle_tic = time.perf_counter()
            g0 = self._oracle_g(x, q_ref)
            g1 = self._oracle_g(x, q_action)
            oracle_ms = 1000.0 * (time.perf_counter() - oracle_tic)
        except Exception as exc:
            self._publish_inactive(f"Stage-II computation failed: {exc}")
            rospy.logerr_throttle(1.0, "[ncdf_action_qp] computation failed: %s", exc)
            return

        visibility_rate_nom = float(np.dot(grad, u_nom))
        visibility_rate_corr = float(np.dot(grad, u_corr))
        delta_visibility_rate = float(np.dot(grad, du))
        delta_f = float(f1 - f0)
        delta_g = float(g1 - g0)

        tol = 1e-8
        lower_active = np.abs(du - du_lower) <= 1e-6
        upper_active = np.abs(du - du_upper) <= 1e-6
        num_active = int(np.count_nonzero(lower_active | upper_active))
        velocity_clipped_nominal = bool(
            np.max(np.abs(u_nom_raw - u_nom)) > tol
        )
        learned_improved = bool(delta_f > tol)
        oracle_improved = bool(delta_g > tol)
        f_up_g_down = bool(delta_f > tol and delta_g < -tol)

        self.active_pub.publish(Bool(data=True))
        self.gradient_pub.publish(self._array_msg(grad))
        self.nominal_action_pub.publish(self._array_msg(u_nom))
        self.corrected_action_pub.publish(self._array_msg(u_corr))
        self.delta_action_pub.publish(self._array_msg(du))
        self.predicted_nominal_state_pub.publish(self._joint_state(q_ref, now))
        self.predicted_corrected_state_pub.publish(self._joint_state(q_action, now))
        self.visibility_rate_nominal_pub.publish(Float32(data=visibility_rate_nom))
        self.visibility_rate_corrected_pub.publish(Float32(data=visibility_rate_corr))
        self.delta_visibility_rate_pub.publish(Float32(data=delta_visibility_rate))
        self.delta_f_pub.publish(Float32(data=delta_f))
        self.delta_g_pub.publish(Float32(data=delta_g))
        self.qp_time_pub.publish(Float32(data=float(qp_ms)))
        self.ncdf_time_pub.publish(Float32(data=float(ncdf_ms)))
        self.oracle_time_pub.publish(Float32(data=float(oracle_ms)))

        summary = (
            f"x={_format_vec(x)} "
            f"grad_norm={grad_norm:.4e} "
            f"u_nom={_format_vec(u_nom)} "
            f"du={_format_vec(du)} u_corr={_format_vec(u_corr)} "
            f"rate={visibility_rate_nom:+.5f}->{visibility_rate_corr:+.5f} "
            f"drate={delta_visibility_rate:+.5f} "
            f"f={f0:+.5f}->{f1:+.5f} df={delta_f:+.5f} "
            f"g={g0:+.5f}->{g1:+.5f} dg={delta_g:+.5f} "
            f"learned_up={int(learned_improved)} oracle_up={int(oracle_improved)} "
            f"f_up_g_down={int(f_up_g_down)} active_bounds={num_active} "
            f"nom_vel_clip={int(velocity_clipped_nominal)} "
            f"qpert_l2={np.linalg.norm(q_action-q_ref):.5f} "
            f"ncdf={ncdf_ms:.3f}ms qp={qp_ms:.4f}ms oracle={oracle_ms:.3f}ms"
        )
        self.summary_pub.publish(String(data=summary))
        rospy.loginfo_throttle(0.5, "[ncdf_action_qp] %s", summary)


def main() -> None:
    rospy.init_node("ncdf_action_qp_dryrun")
    try:
        NcdfActionQPDryRunNode()
    except Exception as exc:
        rospy.logfatal("[ncdf_action_qp] initialization failed: %s", exc)
        raise
    rospy.spin()


if __name__ == "__main__":
    main()
