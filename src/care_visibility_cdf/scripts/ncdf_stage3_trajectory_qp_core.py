#!/usr/bin/env python3
"""Stage-III short-horizon trajectory QP for CAREPlanner visibility guidance."""
from __future__ import annotations

import copy
import math
import sys
import threading
import time
from pathlib import Path

import casadi as ca
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

from evaluate_direct_vs_projection_ascent import (
    DEFAULT_SENSOR_FRAMES,
    PinocchioFOVOracle,
    build_model_from_checkpoint,
    torch_load_checkpoint,
)
from export_ncdf_l4casadi import build_casadi_functions
from test_ncdf_local_optimizer import DEFAULT_JOINT_NAMES, DEFAULT_URDF, read_joint_limits

DEFAULT_CHECKPOINT = (
    PACKAGE_DIR / "checkpoints" / "exp1_yiming_k500_fov_signed" / "final.pt"
)
DEFAULT_BUILD_DIR = PACKAGE_DIR / "generated" / "l4casadi_ncdf_stage3"
DEFAULT_VELOCITY_LIMITS = np.asarray([2.0] * 7, dtype=np.float64)
DEFAULT_ACCELERATION_LIMITS = np.asarray(
    [8.0, 8.0, 10.0, 10.0, 15.0, 15.0, 15.0], dtype=np.float64
)


def _value_grad(value_grad_fn, x, q):
    out = value_grad_fn(
        np.asarray(x, dtype=np.float64).reshape(1, 3),
        np.asarray(q, dtype=np.float64).reshape(1, 7),
    )
    value = float(np.asarray(out[0]).reshape(()))
    grad = np.asarray(out[1], dtype=np.float64).reshape(7)
    return value, grad


def _format_vec(values, precision=5):
    return "[" + ",".join(f"{float(v):+.{precision}f}" for v in values) + "]"


def _as_vec7(name, value, default):
    if value is None:
        arr = default.copy()
    elif np.isscalar(value):
        arr = np.full(7, float(value), dtype=np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (7,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain exactly 7 finite values")
    return arr


class NcdfStage3TrajectoryQPNode:
    """Filter nominal command trajectories through a short-horizon convex QP.

    The active-sensing target is intentionally allowed to change every planning
    cycle. Drift is controlled at trajectory level instead: each optimization
    starts from the newest nominal trajectory, starts with zero deviation, has
    hard configuration/action trust regions, and returns to zero deviation at
    the end of the Stage-III horizon.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._seq = 0

        self.joint_names = [
            str(v) for v in rospy.get_param("~joint_names", list(DEFAULT_JOINT_NAMES))
        ]
        self.input_topic = str(
            rospy.get_param(
                "~input_trajectory_topic",
                "/care_planner/command_trajectory_candidate",
            )
        )
        self.output_topic = str(
            rospy.get_param(
                "~output_trajectory_topic",
                "/care_planner/command_trajectory_stage3",
            )
        )
        self.target_topic = str(
            rospy.get_param(
                "~target_topic", "/care_planner/active_sensing/target_point"
            )
        )
        self.joint_state_topic = str(
            rospy.get_param("~joint_state_topic", "/care_arm/joint_states")
        )
        self.base_frame = str(rospy.get_param("~base_frame", "base_link"))

        self.target_timeout = float(rospy.get_param("~target_timeout", 0.20))
        self.joint_state_timeout = float(
            rospy.get_param("~joint_state_timeout", 0.20)
        )
        self.horizon_duration = float(rospy.get_param("~horizon_duration", 1.0))
        self.num_intervals = int(rospy.get_param("~num_intervals", 10))
        self.stage_dt = self.horizon_duration / self.num_intervals
        self.control_dt = float(rospy.get_param("~control_dt", 0.05))

        # Conservative first Stage-III weights.  With a unit visibility
        # direction these defaults typically produce only a few hundredths of a
        # radian of trajectory deviation, rather than Stage-II's repeated 0.01
        # rad one-step accumulation.
        self.q_weight = float(rospy.get_param("~q_deviation_weight", 20.0))
        self.u_weight = float(rospy.get_param("~u_deviation_weight", 1.0))
        self.smooth_weight = float(rospy.get_param("~u_smooth_weight", 5.0))
        self.visibility_weight = float(rospy.get_param("~visibility_weight", 1.0))

        self.q_trust = _as_vec7(
            "q_trust", rospy.get_param("~q_trust", None), np.full(7, 0.08)
        )
        self.du_trust = _as_vec7(
            "du_trust", rospy.get_param("~du_trust", None), np.full(7, 0.25)
        )
        self.velocity_limits = _as_vec7(
            "velocity_limits",
            rospy.get_param("~velocity_limits", None),
            DEFAULT_VELOCITY_LIMITS,
        )
        self.acceleration_limits = _as_vec7(
            "acceleration_limits",
            rospy.get_param("~acceleration_limits", None),
            DEFAULT_ACCELERATION_LIMITS,
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
            rospy.get_param("~model_name", "care_visibility_ncdf_stage3")
        )
        self.oracle_debug = bool(rospy.get_param("~oracle_debug", True))
        self.grad_eps = float(rospy.get_param("~grad_eps", 1e-10))

        self._validate_params()
        self.q_min, self.q_max = read_joint_limits(self.urdf_path, self.joint_names)
        self._initialize_model()
        self._build_qp_solver()

        self._latest_target = None
        self._target_received = None
        self._latest_joint_state = None
        self._joint_state_received = None

        self.output_pub = rospy.Publisher(
            self.output_topic, JointTrajectory, queue_size=1
        )
        self.active_pub = rospy.Publisher("~active", Bool, queue_size=1)
        self.summary_pub = rospy.Publisher("~summary", String, queue_size=1)
        self.solve_time_pub = rospy.Publisher(
            "~solve_time_ms", Float32, queue_size=1
        )

        rospy.Subscriber(
            self.target_topic, PointStamped, self._target_callback, queue_size=1
        )
        rospy.Subscriber(
            self.joint_state_topic, JointState, self._joint_state_callback, queue_size=1
        )
        rospy.Subscriber(
            self.input_topic, JointTrajectory, self._trajectory_callback, queue_size=1
        )

        rospy.logwarn(
            "[ncdf_stage3] TRAJECTORY FILTER ONLY: publishes corrected trajectory; "
            "does not publish controller commands."
        )
        rospy.loginfo("[ncdf_stage3] input=%s", self.input_topic)
        rospy.loginfo("[ncdf_stage3] output=%s", self.output_topic)
        rospy.loginfo(
            "[ncdf_stage3] horizon=%.2fs K=%d dt=%.3fs "
            "weights[q=%.2f,u=%.2f,smooth=%.2f,vis=%.2f]",
            self.horizon_duration,
            self.num_intervals,
            self.stage_dt,
            self.q_weight,
            self.u_weight,
            self.smooth_weight,
            self.visibility_weight,
        )

    def _validate_params(self):
        if len(self.joint_names) != 7 or len(set(self.joint_names)) != 7:
            raise ValueError("joint_names must contain exactly 7 unique joints")
        if self.num_intervals < 2 or self.horizon_duration <= 0.0:
            raise ValueError("invalid Stage-III horizon")
        if self.control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        if min(
            self.q_weight,
            self.u_weight,
            self.smooth_weight,
            self.visibility_weight,
        ) < 0.0:
            raise ValueError("QP weights must be non-negative")
        if (
            np.any(self.q_trust <= 0.0)
            or np.any(self.du_trust <= 0.0)
            or np.any(self.velocity_limits <= 0.0)
            or np.any(self.acceleration_limits <= 0.0)
        ):
            raise ValueError("Stage-III limits must be positive")
        if self.device_name not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if self.device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint_path}")
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")

    def _initialize_model(self):
        self.device = torch.device(self.device_name)
        ckpt = torch_load_checkpoint(str(self.checkpoint_path), self.device)
        model, _ = build_model_from_checkpoint(ckpt, self.device)
        _, _, _, self.value_grad_fn = build_casadi_functions(
            model=model,
            device=self.device_name,
            build_dir=self.build_dir,
            model_name=self.model_name,
        )
        self.value_grad_fn(np.asarray([[0.0, 0.0, 0.5]]), np.zeros((1, 7)))

        self.oracle = None
        if self.oracle_debug:
            self.oracle = PinocchioFOVOracle(
                urdf_path=str(self.urdf_path),
                joint_names=list(self.joint_names),
                sensor_frames=list(DEFAULT_SENSOR_FRAMES),
                horizontal_fov_deg=50.0,
                vertical_fov_deg=66.0,
                z_min=0.20,
                z_max=0.70,
                delta=0.01,
                base_frame=self.base_frame,
            )

    def _build_qp_solver(self):
        n = 7
        K = self.num_intervals
        dt = self.stage_dt

        delta_q = ca.MX.sym("delta_q", n, K + 1)
        delta_u = ca.MX.sym("delta_u", n, K)

        p_len = n * (K + 1) + n * K + n * (K + 1) + n
        p = ca.MX.sym("p", p_len)
        off = 0
        q_nom = ca.reshape(p[off : off + n * (K + 1)], n, K + 1)
        off += n * (K + 1)
        u_nom = ca.reshape(p[off : off + n * K], n, K)
        off += n * K
        grad_dir = ca.reshape(p[off : off + n * (K + 1)], n, K + 1)
        off += n * (K + 1)
        dq_current = p[off : off + n]

        objective = (
            self.q_weight * ca.sumsqr(delta_q)
            + self.u_weight * ca.sumsqr(delta_u)
        )
        if K > 1:
            objective += self.smooth_weight * ca.sumsqr(
                delta_u[:, 1:] - delta_u[:, :-1]
            )
        # Visibility acts only on interior knots. q0 and qK are exactly on the
        # nominal trajectory, so the maneuver starts and returns smoothly.
        for k in range(1, K):
            objective -= self.visibility_weight * ca.dot(
                grad_dir[:, k], delta_q[:, k]
            )

        constraints = []
        lbg = []
        ubg = []

        def add_constraint(expr, lower, upper):
            constraints.append(ca.vec(expr))
            count = int(expr.numel())
            if np.ndim(lower) == 0:
                lbg.extend([float(lower)] * count)
            else:
                lbg.extend(
                    np.asarray(lower, dtype=np.float64)
                    .reshape(-1, order="F")
                    .tolist()
                )
            if np.ndim(upper) == 0:
                ubg.extend([float(upper)] * count)
            else:
                ubg.extend(
                    np.asarray(upper, dtype=np.float64)
                    .reshape(-1, order="F")
                    .tolist()
                )

        # Start from and return to the current nominal command trajectory.
        add_constraint(delta_q[:, 0], 0.0, 0.0)
        for k in range(K):
            add_constraint(
                delta_q[:, k + 1] - delta_q[:, k] - dt * delta_u[:, k],
                0.0,
                0.0,
            )
        add_constraint(delta_q[:, K], 0.0, 0.0)

        # Bounded trajectory deviation and physical joint limits.
        add_constraint(
            delta_q,
            np.tile(-self.q_trust.reshape(7, 1), (1, K + 1)),
            np.tile(self.q_trust.reshape(7, 1), (1, K + 1)),
        )
        add_constraint(
            q_nom + delta_q,
            np.tile(self.q_min.reshape(7, 1), (1, K + 1)),
            np.tile(self.q_max.reshape(7, 1), (1, K + 1)),
        )

        # Bounded action deviation and total velocity limits.
        add_constraint(
            delta_u,
            np.tile(-self.du_trust.reshape(7, 1), (1, K)),
            np.tile(self.du_trust.reshape(7, 1), (1, K)),
        )
        add_constraint(
            u_nom + delta_u,
            np.tile(-self.velocity_limits.reshape(7, 1), (1, K)),
            np.tile(self.velocity_limits.reshape(7, 1), (1, K)),
        )

        # Total corrected-action acceleration limits. The first knot is tied to
        # measured qdot; later knots constrain corrected velocity differences.
        add_constraint(
            u_nom[:, 0] + delta_u[:, 0] - dq_current,
            -self.acceleration_limits * self.control_dt,
            self.acceleration_limits * self.control_dt,
        )
        for k in range(1, K):
            add_constraint(
                (u_nom[:, k] + delta_u[:, k])
                - (u_nom[:, k - 1] + delta_u[:, k - 1]),
                -self.acceleration_limits * dt,
                self.acceleration_limits * dt,
            )

        x = ca.vertcat(ca.vec(delta_q), ca.vec(delta_u))
        g = ca.vertcat(*constraints)
        self._solver = ca.qpsol(
            "stage3_qp",
            "qrqp",
            {"x": x, "p": p, "f": objective, "g": g},
            {
                "print_header": False,
                "print_iter": False,
                "print_info": False,
                "error_on_fail": False,
            },
        )
        self._lbg = np.asarray(lbg, dtype=np.float64)
        self._ubg = np.asarray(ubg, dtype=np.float64)
        self._n_delta_q = n * (K + 1)

    def _target_callback(self, msg):
        with self._lock:
            self._latest_target = copy.deepcopy(msg)
            self._target_received = rospy.Time.now()

    def _joint_state_callback(self, msg):
        with self._lock:
            self._latest_joint_state = copy.deepcopy(msg)
            self._joint_state_received = rospy.Time.now()

    @staticmethod
    def _age(stamp):
        if stamp is None:
            return math.inf
        return max(0.0, (rospy.Time.now() - stamp).to_sec())

    def _ordered_joint_state(self, msg):
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

    def _trajectory_mapping(self, traj):
        index = {name: i for i, name in enumerate(traj.joint_names)}
        if any(name not in index for name in self.joint_names):
            return None
        return [index[name] for name in self.joint_names]

    def _sample_trajectory(self, traj, mapping, t):
        times = np.asarray(
            [point.time_from_start.to_sec() for point in traj.points], dtype=np.float64
        )
        if len(times) == 0:
            return None

        def field(point, name, allow_zero):
            values = getattr(point, name)
            if not values:
                return np.zeros(7, dtype=np.float64) if allow_zero else None
            out = np.zeros(7, dtype=np.float64)
            for i, src in enumerate(mapping):
                if src >= len(values):
                    if allow_zero:
                        out[i] = 0.0
                    else:
                        return None
                else:
                    out[i] = float(values[src])
            return out

        if t <= times[0]:
            point = traj.points[0]
            return field(point, "positions", False), field(point, "velocities", True)
        if t >= times[-1]:
            point = traj.points[-1]
            return field(point, "positions", False), field(point, "velocities", True)

        upper = int(np.searchsorted(times, t, side="left"))
        lower = upper - 1
        h = float(times[upper] - times[lower])
        if h <= 1e-12:
            return None
        alpha = float((t - times[lower]) / h)
        q0 = field(traj.points[lower], "positions", False)
        q1 = field(traj.points[upper], "positions", False)
        u0 = field(traj.points[lower], "velocities", True)
        u1 = field(traj.points[upper], "velocities", True)
        if any(v is None for v in (q0, q1, u0, u1)):
            return None
        return (1.0 - alpha) * q0 + alpha * q1, (1.0 - alpha) * u0 + alpha * u1

    def _pack_parameters(self, q_nom, u_nom, grad_dir, dq_current):
        return np.concatenate(
            [
                q_nom.reshape(-1, order="F"),
                u_nom.reshape(-1, order="F"),
                grad_dir.reshape(-1, order="F"),
                dq_current,
            ]
        )

    def _apply_correction(self, traj, mapping, delta_q, delta_u):
        corrected = copy.deepcopy(traj)
        K = self.num_intervals
        dt = self.stage_dt
        for point in corrected.points:
            t = point.time_from_start.to_sec()
            if t <= 0.0 or t >= self.horizon_duration:
                dq_corr = np.zeros(7, dtype=np.float64)
                du_corr = np.zeros(7, dtype=np.float64)
            else:
                k = min(K - 1, int(math.floor(t / dt)))
                alpha = float((t - k * dt) / dt)
                dq_corr = (
                    (1.0 - alpha) * delta_q[:, k]
                    + alpha * delta_q[:, k + 1]
                )
                du_corr = delta_u[:, k]

            for i, src in enumerate(mapping):
                point.positions[src] = float(point.positions[src] + dq_corr[i])
            if point.velocities:
                for i, src in enumerate(mapping):
                    if src < len(point.velocities):
                        point.velocities[src] = float(
                            point.velocities[src] + du_corr[i]
                        )
        return corrected

    def _oracle_g(self, x, q):
        if self.oracle is None:
            return math.nan
        x_tensor = torch.as_tensor(
            x, dtype=torch.float32, device=self.device
        ).reshape(1, 3)
        q_tensor = torch.as_tensor(
            q, dtype=torch.float32, device=self.device
        ).reshape(1, 7)
        raw, _ = self.oracle.signed_fov_margins(x_tensor, q_tensor)
        return float(
            torch.max(raw - self.oracle.delta, dim=-1)
            .values.reshape(())
            .detach()
            .cpu()
        )

    def _trajectory_callback(self, traj):
        mapping = self._trajectory_mapping(traj)
        if mapping is None or not traj.points:
            rospy.logwarn_throttle(1.0, "[ncdf_stage3] invalid nominal trajectory")
            return

        with self._lock:
            target = copy.deepcopy(self._latest_target)
            target_received = self._target_received
            joint_state = copy.deepcopy(self._latest_joint_state)
            joint_received = self._joint_state_received

        target_fresh = (
            target is not None
            and self._age(target_received) <= self.target_timeout
            and (not target.header.frame_id or target.header.frame_id == self.base_frame)
        )
        if not target_fresh:
            self.output_pub.publish(traj)
            self.active_pub.publish(Bool(data=False))
            return

        if joint_state is None or self._age(joint_received) > self.joint_state_timeout:
            self.output_pub.publish(traj)
            self.active_pub.publish(Bool(data=False))
            rospy.logwarn_throttle(
                1.0, "[ncdf_stage3] stale JointState -> nominal pass-through"
            )
            return

        state = self._ordered_joint_state(joint_state)
        if state is None:
            self.output_pub.publish(traj)
            self.active_pub.publish(Bool(data=False))
            return
        _, dq_current = state

        if traj.points[-1].time_from_start.to_sec() < self.horizon_duration - 1e-6:
            self.output_pub.publish(traj)
            self.active_pub.publish(Bool(data=False))
            rospy.logwarn_throttle(
                1.0,
                "[ncdf_stage3] nominal trajectory shorter than Stage-III horizon",
            )
            return

        x = np.asarray(
            [target.point.x, target.point.y, target.point.z], dtype=np.float64
        )
        if not np.all(np.isfinite(x)):
            self.output_pub.publish(traj)
            self.active_pub.publish(Bool(data=False))
            return

        K = self.num_intervals
        q_nom = np.zeros((7, K + 1), dtype=np.float64)
        u_nom = np.zeros((7, K), dtype=np.float64)
        grad_dir = np.zeros((7, K + 1), dtype=np.float64)
        f_nom = []

        tic = time.perf_counter()
        try:
            for k in range(K + 1):
                sampled = self._sample_trajectory(
                    traj, mapping, k * self.stage_dt
                )
                if sampled is None:
                    raise RuntimeError("trajectory sampling failed")
                q_nom[:, k] = sampled[0]
                if k < K:
                    u_nom[:, k] = sampled[1]

                f_value, grad = _value_grad(
                    self.value_grad_fn, x, q_nom[:, k]
                )
                f_nom.append(f_value)
                grad_norm = float(np.linalg.norm(grad))
                if grad_norm > self.grad_eps:
                    grad_dir[:, k] = grad / grad_norm

            parameters = self._pack_parameters(
                q_nom, u_nom, grad_dir, dq_current
            )
            solution = self._solver(
                p=parameters, lbg=self._lbg, ubg=self._ubg
            )
            stats = self._solver.stats()
            if not bool(stats.get("success", True)):
                raise RuntimeError(
                    f"QP failed: {stats.get('return_status', 'unknown')}"
                )

            z = np.asarray(solution["x"], dtype=np.float64).reshape(-1)
            delta_q = z[: self._n_delta_q].reshape(
                (7, K + 1), order="F"
            )
            delta_u = z[self._n_delta_q :].reshape((7, K), order="F")

            corrected = self._apply_correction(
                traj, mapping, delta_q, delta_u
            )

            learned_delta = []
            oracle_delta = []
            for k in range(1, K):
                f_corrected, _ = _value_grad(
                    self.value_grad_fn, x, q_nom[:, k] + delta_q[:, k]
                )
                learned_delta.append(f_corrected - f_nom[k])
                if self.oracle is not None:
                    oracle_delta.append(
                        self._oracle_g(x, q_nom[:, k] + delta_q[:, k])
                        - self._oracle_g(x, q_nom[:, k])
                    )

            total_ms = 1000.0 * (time.perf_counter() - tic)
            self._seq += 1
            max_delta_q = float(np.max(np.abs(delta_q)))
            max_delta_u = float(np.max(np.abs(delta_u)))
            terminal_norm = float(np.linalg.norm(delta_q[:, -1]))

            self.output_pub.publish(corrected)
            self.active_pub.publish(Bool(data=True))
            self.solve_time_pub.publish(Float32(data=total_ms))

            dg_mean = (
                float(np.mean(oracle_delta)) if oracle_delta else math.nan
            )
            dg_improve = (
                float(np.mean(np.asarray(oracle_delta) > 0.0))
                if oracle_delta
                else math.nan
            )
            summary = (
                f"seq={self._seq} x={_format_vec(x)} "
                f"max_dq={max_delta_q:.5f} max_du={max_delta_u:.5f} "
                f"terminal_dq={terminal_norm:.3e} "
                f"df_mean={float(np.mean(learned_delta)):+.5f} "
                f"df_min={float(np.min(learned_delta)):+.5f} "
                f"dg_mean={dg_mean:+.5f} dg_improve={dg_improve:.3f} "
                f"total={total_ms:.2f}ms"
            )
            self.summary_pub.publish(String(data=summary))
            rospy.loginfo_throttle(0.5, "[ncdf_stage3] %s", summary)

        except Exception as exc:
            # Stage III is a filter. Any optimization problem falls back to the
            # untouched nominal trajectory; the executor remains downstream.
            self.output_pub.publish(traj)
            self.active_pub.publish(Bool(data=False))
            rospy.logerr_throttle(
                1.0,
                "[ncdf_stage3] QP failed -> nominal pass-through: %s",
                exc,
            )


def main():
    rospy.init_node("ncdf_stage3_trajectory_qp")
    NcdfStage3TrajectoryQPNode()
    rospy.spin()
