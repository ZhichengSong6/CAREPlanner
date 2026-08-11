#!/usr/bin/env python3
"""ROS1 dry-run interface for frozen CARE visibility NCDF guidance.

This node is deliberately NON-ACTUATING.  It subscribes to the real CAREPlanner
active-sensing target, current JointState, and nominal command trajectory, then
computes the frozen direct normalized projected-ascent correction used by the
offline L4CasADi tests.

It NEVER publishes to the planner command-trajectory topic and NEVER publishes
robot velocity/position commands.  Outputs are debug-only topics.

Runtime data flow
-----------------
/care_planner/active_sensing/target_point       geometry_msgs/PointStamped
/care_arm/joint_states                          sensor_msgs/JointState
/care_planner/command_trajectory_candidate      trajectory_msgs/JointTrajectory
                                  |
                                  v
                    q_nominal at lookahead time
                                  |
                                  v
                 frozen NCDF direct ascent (5 x 0.01 rad)
                                  |
                                  v
              debug nominal/corrected JointState + metrics

The active-sensing target is required to be fresh.  Freshness uses local ROS
receipt time rather than trusting the incoming header stamp, which keeps this
node robust to zero or simulation-time stamps.
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
from std_msgs.msg import Bool, Float32, String
from trajectory_msgs.msg import JointTrajectory


# Resolve the source package explicitly.  catkin_install_python places the
# runtime entry point in devel/lib, while the frozen model reconstruction and
# L4CasADi helpers intentionally remain source-tree tools.
PACKAGE_DIR = Path(rospkg.RosPack().get_path("care_visibility_cdf")).resolve()
SCRIPT_DIR = PACKAGE_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_direct_vs_projection_ascent import (  # noqa: E402
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


class NcdfActiveSensingDryRunNode:
    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Topics / timing.
        self.target_topic = rospy.get_param(
            "~target_topic", "/care_planner/active_sensing/target_point"
        )
        self.joint_state_topic = rospy.get_param(
            "~joint_state_topic", "/care_arm/joint_states"
        )
        self.command_trajectory_topic = rospy.get_param(
            "~command_trajectory_topic", "/care_planner/command_trajectory_candidate"
        )
        self.base_frame = rospy.get_param("~base_frame", "base_link")
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

        self._validate_params()

        # Cached ROS inputs and local receipt times.
        self._latest_target: Optional[PointStamped] = None
        self._target_received: Optional[rospy.Time] = None
        self._latest_joint_state: Optional[JointState] = None
        self._joint_state_received: Optional[rospy.Time] = None
        self._latest_trajectory: Optional[JointTrajectory] = None
        self._trajectory_received: Optional[rospy.Time] = None

        rospy.loginfo("[ncdf_dryrun] Loading frozen NCDF / L4CasADi interface...")
        self._initialize_model()

        # Debug-only publishers.  There is intentionally no publisher for the
        # planner command trajectory or any controller command topic.
        self.nominal_pub = rospy.Publisher(
            "~nominal_joint_state", JointState, queue_size=1
        )
        self.corrected_pub = rospy.Publisher(
            "~corrected_joint_state", JointState, queue_size=1
        )
        self.active_pub = rospy.Publisher("~active", Bool, queue_size=1)
        self.delta_f_pub = rospy.Publisher("~delta_f", Float32, queue_size=1)
        self.optimizer_time_pub = rospy.Publisher(
            "~optimizer_time_ms", Float32, queue_size=1
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

        rospy.logwarn(
            "[ncdf_dryrun] DRY RUN ONLY: this node does not publish commands and cannot actuate the robot."
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
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"NCDF checkpoint not found: {self.checkpoint_path}")
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")
        if self.device_name not in ("cpu", "cuda"):
            raise ValueError("~device must be 'cpu' or 'cuda'")
        if self.device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("~device=cuda requested but torch CUDA is unavailable")

    def _initialize_model(self) -> None:
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

        # First external-function call can include initialization overhead; do
        # it once before online timing starts.
        self.value_grad_fn(np.asarray([[0.0, 0.0, 0.5]]), np.zeros((1, 7)))
        rospy.loginfo("[ncdf_dryrun] Frozen NCDF ready on %s.", self.device_name)

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
        q = np.asarray([msg.position[index[name]] for name in self.joint_names], dtype=np.float64)
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
        # The trust region is intentionally centered on q_nominal: this node is
        # diagnosing a local correction to the task reference, not replacing
        # the task planner with visibility-only planning.
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
            self._publish_inactive("command trajectory cannot provide ordered nominal q")
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
        except Exception as exc:  # keep ROS process alive for diagnostic use
            self._publish_inactive(f"optimizer exception: {exc}")
            rospy.logerr_throttle(1.0, "[ncdf_dryrun] optimizer exception: %s", exc)
            return

        delta_q = q_corrected - q_nominal
        delta_f = f1 - f0
        step_l2 = float(np.linalg.norm(delta_q))
        step_inf = float(np.max(np.abs(delta_q)))
        nominal_from_current_l2 = float(np.linalg.norm(q_nominal - q_current))

        stamp = now
        self.nominal_pub.publish(self._joint_state_msg(q_nominal, stamp))
        self.corrected_pub.publish(self._joint_state_msg(q_corrected, stamp))
        self.active_pub.publish(Bool(data=True))
        self.delta_f_pub.publish(Float32(data=float(delta_f)))
        self.optimizer_time_pub.publish(Float32(data=float(elapsed_ms)))

        summary = (
            f"x={_format_vec(x)} target_age={target_age:.3f}s "
            f"q_nominal_from_current_l2={nominal_from_current_l2:.4f} "
            f"f={f0:+.5f}->{f1:+.5f} df={delta_f:+.5f} "
            f"dq_l2={step_l2:.4f} dq_inf={step_inf:.4f} "
            f"accepted={accepted}/{self.iterations} backtracks={backtracks} "
            f"time={elapsed_ms:.3f}ms dq={_format_vec(delta_q)}"
        )
        self.summary_pub.publish(String(data=summary))

        rospy.loginfo_throttle(
            0.5,
            "[ncdf_dryrun] %s",
            summary,
        )


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
