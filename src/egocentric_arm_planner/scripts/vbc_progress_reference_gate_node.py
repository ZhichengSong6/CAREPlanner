#!/usr/bin/env python3
"""Progress-anchored nominal reference gate for continuous C4.3 CAREPlanner.

The validated VBCExecutionReferenceGate is kept for frozen C4.2 behavior.  This
C4.3 specialization preserves its pre-execution/T0 semantics but changes how the
nominal master trajectory advances after release:

  wall-clock gate:      phase = now - T0
  progress-aware gate:  phase = monotone projection of measured q onto master

The projection is deliberately conservative:
* nominal progress never moves backward;
* progress advance per wall-clock second is capped;
* if the robot is far from the nominal path (e.g. visibility detour), progress
  is frozen instead of letting the nominal reference run away to the goal.

The resulting advancing suffix remains a *soft task reference* for the 1 s MPC;
it is not an execution trajectory and is never sent directly to the actuator.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rospkg
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, String


def _load_base_class():
    pkg = Path(rospkg.RosPack().get_path("egocentric_arm_planner"))
    path = pkg / "scripts" / "vbc_execution_reference_gate_node.py"
    spec = importlib.util.spec_from_file_location("_care_vbc_gate_base", str(path))
    if spec is None or spec.loader is None:
        raise ImportError("cannot load VBC gate base from {}".format(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.VBCExecutionReferenceGate


VBCExecutionReferenceGate = _load_base_class()


class ProgressAnchoredVBCExecutionReferenceGate(VBCExecutionReferenceGate):
    def __init__(self) -> None:
        # These fields must exist before base __init__ installs the timer, because
        # Python binds the overridden _timer_callback below.
        self._progress_phase_s = 0.0
        self._progress_projection_error_inf = math.nan
        self._progress_projection_candidate_s = 0.0
        self._progress_last_update_ros_s: Optional[float] = None
        self._progress_update_count = 0
        self._progress_frozen_count = 0
        self._latest_measured_q: Optional[List[float]] = None
        self._latest_measured_q_ros_s: Optional[float] = None

        super().__init__()

        self.progress_enabled = bool(rospy.get_param("~progress/enabled", True))
        self.joint_state_topic = str(rospy.get_param(
            "~progress/joint_states", "/care_arm/joint_states"))
        self.joint_state_timeout_s = float(rospy.get_param(
            "~progress/joint_state_timeout_s", 0.20))
        self.progress_search_ahead_s = float(rospy.get_param(
            "~progress/search_ahead_s", 0.35))
        self.progress_max_rate = float(rospy.get_param(
            "~progress/max_rate", 1.25))
        self.progress_max_step_s = float(rospy.get_param(
            "~progress/max_step_s", 0.08))
        self.progress_max_path_error_inf = float(rospy.get_param(
            "~progress/max_path_error_inf", 0.25))
        self.joint_names = list(rospy.get_param("~joint_names", []))
        if not self.joint_names:
            raise ValueError("~joint_names is required for progress projection")
        if self.joint_state_timeout_s <= 0.0:
            raise ValueError("progress joint-state timeout must be positive")
        if self.progress_search_ahead_s <= 0.0:
            raise ValueError("progress search-ahead must be positive")
        if self.progress_max_rate <= 0.0 or self.progress_max_step_s <= 0.0:
            raise ValueError("progress rate/step limits must be positive")
        if self.progress_max_path_error_inf <= 0.0:
            raise ValueError("progress max path error must be positive")

        self.progress_summary_topic = str(rospy.get_param(
            "~progress/summary_topic", "/care_planner/execution/nominal_progress_summary"))
        self.progress_summary_pub = rospy.Publisher(
            self.progress_summary_topic, String, queue_size=1, latch=True)
        rospy.Subscriber(
            self.joint_state_topic, JointState, self._joint_state_callback, queue_size=1)

        rospy.logwarn(
            "[vbc_progress_gate] ENABLED=%d joint_states=%s search_ahead=%.3fs "
            "max_rate=%.2fx max_step=%.3fs max_path_error_inf=%.3frad",
            int(self.progress_enabled), self.joint_state_topic,
            self.progress_search_ahead_s, self.progress_max_rate,
            self.progress_max_step_s, self.progress_max_path_error_inf)

    def _joint_state_callback(self, msg: JointState) -> None:
        if msg is None or not msg.name or not msg.position:
            return
        index: Dict[str, int] = {name: i for i, name in enumerate(msg.name)}
        q: List[float] = []
        for name in self.joint_names:
            idx = index.get(name)
            if idx is None or idx >= len(msg.position):
                return
            value = float(msg.position[idx])
            if not math.isfinite(value):
                return
            q.append(value)
        with self._lock:
            self._latest_measured_q = q
            self._latest_measured_q_ros_s = rospy.Time.now().to_sec()

    @staticmethod
    def _interp_positions(p0, p1, alpha: float, mapping: List[int]) -> Optional[List[float]]:
        if len(p0.positions) <= max(mapping) or len(p1.positions) <= max(mapping):
            return None
        return [
            (1.0 - alpha) * float(p0.positions[idx]) + alpha * float(p1.positions[idx])
            for idx in mapping
        ]

    @staticmethod
    def _inf_error(a: List[float], b: List[float]) -> float:
        return max(abs(float(x) - float(y)) for x, y in zip(a, b)) if a else math.inf

    def _project_phase_locked(self, q: List[float]) -> Tuple[float, float]:
        master = self._master_reference
        if master is None or len(master.points) < 2:
            return self._progress_phase_s, math.inf

        name_to_idx = {name: i for i, name in enumerate(master.joint_names)}
        try:
            mapping = [name_to_idx[name] for name in self.joint_names]
        except KeyError:
            return self._progress_phase_s, math.inf

        times = [float(p.time_from_start.to_sec()) for p in master.points]
        start = max(0.0, self._progress_phase_s)
        end = min(self._master_duration_s, start + self.progress_search_ahead_s)

        best_phase = start
        best_error_inf = math.inf
        best_sq = math.inf

        for i in range(len(master.points) - 1):
            t0, t1 = times[i], times[i + 1]
            if t1 < start - 1e-9 or t0 > end + 1e-9:
                continue
            p0 = master.points[i]
            p1 = master.points[i + 1]
            if len(p0.positions) <= max(mapping) or len(p1.positions) <= max(mapping):
                continue
            q0 = [float(p0.positions[idx]) for idx in mapping]
            q1 = [float(p1.positions[idx]) for idx in mapping]
            d = [q1[j] - q0[j] for j in range(len(q))]
            denom = sum(x * x for x in d)
            if denom <= 1e-12:
                alpha = 0.0
            else:
                alpha = sum((q[j] - q0[j]) * d[j] for j in range(len(q))) / denom
            alpha = max(0.0, min(1.0, alpha))
            phase = t0 + alpha * max(0.0, t1 - t0)
            if phase < start:
                phase = start
                if t1 > t0:
                    alpha = max(0.0, min(1.0, (phase - t0) / (t1 - t0)))
            if phase > end:
                phase = end
                if t1 > t0:
                    alpha = max(0.0, min(1.0, (phase - t0) / (t1 - t0)))
            q_proj = self._interp_positions(p0, p1, alpha, mapping)
            if q_proj is None:
                continue
            diff = [q[j] - q_proj[j] for j in range(len(q))]
            sq = sum(x * x for x in diff)
            err_inf = max(abs(x) for x in diff) if diff else math.inf
            if sq < best_sq:
                best_sq = sq
                best_phase = phase
                best_error_inf = err_inf

        return max(start, best_phase), best_error_inf

    def _update_progress_locked(self, now_s: float) -> None:
        if not self.progress_enabled:
            if self._execution_start_ros_s is not None:
                self._progress_phase_s = min(
                    self._master_duration_s,
                    max(0.0, now_s - self._execution_start_ros_s))
            return
        if self._master_reference is None or self._latest_measured_q is None:
            return
        if self._latest_measured_q_ros_s is None:
            return
        age = now_s - self._latest_measured_q_ros_s
        if age < 0.0 or age > self.joint_state_timeout_s:
            self._progress_frozen_count += 1
            return

        candidate, err_inf = self._project_phase_locked(self._latest_measured_q)
        self._progress_projection_candidate_s = candidate
        self._progress_projection_error_inf = err_inf

        if not math.isfinite(err_inf) or err_inf > self.progress_max_path_error_inf:
            self._progress_frozen_count += 1
            self._progress_last_update_ros_s = now_s
            return

        if self._progress_last_update_ros_s is None:
            dt_wall = 1.0 / self.publish_rate
        else:
            dt_wall = max(0.0, now_s - self._progress_last_update_ros_s)
        max_advance = min(
            self.progress_max_step_s,
            self.progress_max_rate * max(dt_wall, 1.0 / self.publish_rate))
        new_phase = min(candidate, self._progress_phase_s + max_advance)
        new_phase = max(self._progress_phase_s, new_phase)
        self._progress_phase_s = min(self._master_duration_s, new_phase)
        self._progress_last_update_ros_s = now_s
        self._progress_update_count += 1

    def _release_locked(self) -> None:
        super()._release_locked()
        self._progress_phase_s = 0.0
        self._progress_projection_candidate_s = 0.0
        self._progress_projection_error_inf = math.nan
        self._progress_last_update_ros_s = rospy.Time.now().to_sec()

    def _publish_progress_summary(self) -> None:
        with self._lock:
            now = rospy.Time.now().to_sec()
            wall_elapsed = (
                math.nan if self._execution_start_ros_s is None
                else max(0.0, now - self._execution_start_ros_s))
            lag = (
                math.nan if not math.isfinite(wall_elapsed)
                else max(0.0, wall_elapsed - self._progress_phase_s))
            msg = String()
            msg.data = (
                "enabled={} phase_s={:.6f} candidate_phase_s={:.6f} "
                "projection_error_inf={} wall_elapsed_s={} lag_s={} "
                "master_duration_s={:.6f} update_count={} frozen_count={}"
            ).format(
                int(self.progress_enabled),
                self._progress_phase_s,
                self._progress_projection_candidate_s,
                "nan" if not math.isfinite(self._progress_projection_error_inf)
                else "{:.6f}".format(self._progress_projection_error_inf),
                "nan" if not math.isfinite(wall_elapsed) else "{:.6f}".format(wall_elapsed),
                "nan" if not math.isfinite(lag) else "{:.6f}".format(lag),
                self._master_duration_s,
                self._progress_update_count,
                self._progress_frozen_count,
            )
        self.progress_summary_pub.publish(msg)

    def _trace_payload_locked(self):
        payload = super()._trace_payload_locked()
        payload.update({
            "nominal_progress_mode": "projection" if self.progress_enabled else "wall_clock",
            "nominal_progress_phase_s": self._progress_phase_s,
            "nominal_projection_candidate_s": self._progress_projection_candidate_s,
            "nominal_projection_error_inf": None if not math.isfinite(
                self._progress_projection_error_inf) else self._progress_projection_error_inf,
            "nominal_progress_update_count": self._progress_update_count,
            "nominal_progress_frozen_count": self._progress_frozen_count,
        })
        return payload

    def _timer_callback(self, _event) -> None:
        reference = None
        deadline_msg = None
        send_replan_ready = False

        with self._lock:
            if self._can_release_locked():
                self._release_locked()

            if (self._released and not self._waiting_replan
                    and self._master_reference is not None):
                now = rospy.Time.now().to_sec()
                self._update_progress_locked(now)
                reference = self._suffix_from_phase(
                    self._master_reference, self._progress_phase_s)
                self._publish_count += 1

                if self._synced_deadline_ros_s is not None:
                    deadline_msg = Float64()
                    deadline_msg.data = self._synced_deadline_ros_s
                    self._deadline_publish_count += 1

                if self._replan_ready_pending:
                    send_replan_ready = True
                    self._replan_ready_pending = False

                if self._publish_count % 10 == 0:
                    self._write_trace_locked()

        if reference is not None and reference.points:
            self.reference_pub.publish(reference)
        if deadline_msg is not None:
            self.deadline_pub.publish(deadline_msg)
        if send_replan_ready:
            msg = Bool(); msg.data = True; self.replan_ready_pub.publish(msg)
        self._publish_summary()
        self._publish_progress_summary()


def main() -> None:
    rospy.init_node("vbc_execution_reference_gate")
    ProgressAnchoredVBCExecutionReferenceGate()
    rospy.spin()


if __name__ == "__main__":
    main()
