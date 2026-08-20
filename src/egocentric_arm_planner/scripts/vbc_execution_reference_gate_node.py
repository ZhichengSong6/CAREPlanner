#!/usr/bin/env python3
"""Gate/re-time nominal references and coordinate post-recovery replanning.

Initial execution is unchanged from the validated Phase-B2 gate:
  nominal plan -> stable VBC decision -> q_vis ready -> release synchronized T0

If the waypoint MPC later reports visibility recovery complete, the gate stops
publishing the old advancing nominal reference. The controlled-trial node
republishes the original EE goal; RecedingHorizonPlanner emits a fresh
/task_trajectory only after that measured-state replan succeeds. The gate uses
that message as an unambiguous generation marker, accepts the *next* upstream
command trajectory as the new master, re-bases it at phase zero, and only then
emits visibility_replan_ready so the MPC may leave RECOVERY_HOLD.
"""

from __future__ import annotations

import copy
import json
import math
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import rospy
from std_msgs.msg import Bool, Float32, Float64, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


_TOKEN_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


def _tokens(text: str):
    return {k: v for k, v in _TOKEN_RE.findall(text or "")}


def _as_bool_token(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    if value in ("1", "true", "True"):
        return True
    if value in ("0", "false", "False"):
        return False
    return None


def _sanitize(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    return out or "trial"


class VBCExecutionReferenceGate:
    def __init__(self) -> None:
        self._lock = threading.Lock()

        self.input_reference_topic = str(rospy.get_param(
            "~input_reference_topic", "/care_planner/command_trajectory_candidate"))
        self.output_reference_topic = str(rospy.get_param(
            "~output_reference_topic", "/care_planner/command_trajectory_vbc_gated"))
        self.task_trajectory_topic = str(rospy.get_param(
            "~task_trajectory_topic", "/care_planner/task_trajectory"))
        self.vbc_summary_topic = str(rospy.get_param(
            "~vbc_summary_topic", "/care_planner/trajectory_risk/vbc_summary"))
        self.waypoint_summary_topic = str(rospy.get_param(
            "~waypoint_summary_topic",
            "/care_planner/active_sensing/visibility_waypoint_summary"))
        self.frozen_sweep_topic = str(rospy.get_param(
            "~frozen_sweep_time_topic",
            "/care_planner/active_sensing/frozen_sweep_time_s"))
        self.recovery_complete_topic = str(rospy.get_param(
            "~recovery_complete_topic",
            "/care_planner/execution/visibility_recovery_complete"))
        self.replan_ready_topic = str(rospy.get_param(
            "~replan_ready_topic",
            "/care_planner/execution/visibility_replan_ready"))

        self.execution_ready_topic = str(rospy.get_param(
            "~execution_ready_topic", "/care_planner/execution/ready"))
        self.execution_start_topic = str(rospy.get_param(
            "~execution_start_topic", "/care_planner/execution/start_time"))
        self.synced_deadline_topic = str(rospy.get_param(
            "~synced_deadline_topic",
            "/care_planner/active_sensing/visibility_waypoint_deadline_synced"))
        self.summary_topic = str(rospy.get_param(
            "~summary_topic", "/care_planner/execution/gate_summary"))

        self.publish_rate = float(rospy.get_param("~publish_rate", 20.0))
        self.safety_margin_s = float(rospy.get_param("~safety_margin_s", 0.30))
        self.decision_stability_count = int(rospy.get_param("~decision_stability_count", 2))
        self.trial_label = str(rospy.get_param("~trial_label", "vbc_waypoint"))
        self.output_root = Path(rospy.get_param(
            "~output_root", "outputs/phase_b2_vbc_waypoint")).expanduser().resolve()

        if self.publish_rate <= 0.0:
            raise ValueError("~publish_rate must be positive")
        if self.safety_margin_s < 0.0:
            raise ValueError("~safety_margin_s must be non-negative")
        if self.decision_stability_count <= 0:
            raise ValueError("~decision_stability_count must be positive")

        self.output_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trace_path = self.output_root / (
            f"execution_gate_{_sanitize(self.trial_label)}_{stamp}.json")

        self._master_reference: Optional[JointTrajectory] = None
        self._master_duration_s = -1.0
        self._master_received_ros_s: Optional[float] = None

        self._decision: Optional[bool] = None
        self._last_observed_decision: Optional[bool] = None
        self._decision_repeat_count = 0
        self._waypoint_ready = False
        self._frozen_sweep_time_s: Optional[float] = None

        self._released = False
        self._execution_start_ros_s: Optional[float] = None
        self._synced_deadline_ros_s: Optional[float] = None
        self._release_reason = "waiting_vbc_decision"
        self._publish_count = 0
        self._deadline_publish_count = 0

        self._waiting_replan = False
        self._replan_task_seen = False
        self._replan_ready_pending = False
        self._replan_count = 0

        self.reference_pub = rospy.Publisher(
            self.output_reference_topic, JointTrajectory, queue_size=1)
        self.ready_pub = rospy.Publisher(
            self.execution_ready_topic, Bool, queue_size=1, latch=True)
        self.start_pub = rospy.Publisher(
            self.execution_start_topic, Float64, queue_size=1, latch=True)
        self.deadline_pub = rospy.Publisher(
            self.synced_deadline_topic, Float64, queue_size=1, latch=True)
        self.summary_pub = rospy.Publisher(
            self.summary_topic, String, queue_size=1, latch=True)
        self.replan_ready_pub = rospy.Publisher(
            self.replan_ready_topic, Bool, queue_size=1, latch=False)

        rospy.Subscriber(
            self.input_reference_topic, JointTrajectory,
            self._reference_callback, queue_size=5)
        rospy.Subscriber(
            self.task_trajectory_topic, JointTrajectory,
            self._task_trajectory_callback, queue_size=5)
        rospy.Subscriber(
            self.vbc_summary_topic, String, self._vbc_summary_callback, queue_size=5)
        rospy.Subscriber(
            self.waypoint_summary_topic, String,
            self._waypoint_summary_callback, queue_size=5)
        rospy.Subscriber(
            self.frozen_sweep_topic, Float32,
            self._frozen_sweep_callback, queue_size=1)
        rospy.Subscriber(
            self.recovery_complete_topic, Bool,
            self._recovery_complete_callback, queue_size=1)

        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate), self._timer_callback)
        rospy.on_shutdown(self._on_shutdown)

        ready = Bool(); ready.data = False; self.ready_pub.publish(ready)
        self._publish_summary()
        rospy.logwarn(
            "[vbc_exec_gate] PRE-EXECUTION GATE ARMED: input=%s output=%s",
            self.input_reference_topic, self.output_reference_topic)
        rospy.logwarn(
            "[vbc_exec_gate] recovery protocol: complete=%s task_marker=%s replan_ready=%s",
            self.recovery_complete_topic, self.task_trajectory_topic,
            self.replan_ready_topic)

    @staticmethod
    def _duration(msg: JointTrajectory) -> float:
        if msg is None or not msg.points:
            return -1.0
        return float(msg.points[-1].time_from_start.to_sec())

    def _reference_callback(self, msg: JointTrajectory) -> None:
        if msg is None or not msg.points:
            return
        duration = self._duration(msg)
        if not math.isfinite(duration) or duration < 0.0:
            return

        with self._lock:
            if not self._released:
                if self._master_reference is None or duration > self._master_duration_s + 1e-9:
                    self._master_reference = copy.deepcopy(msg)
                    self._master_duration_s = duration
                    self._master_received_ros_s = rospy.Time.now().to_sec()
                    rospy.logwarn(
                        "[vbc_exec_gate] cached fuller nominal reference: duration=%.6f s points=%d",
                        duration, len(msg.points))
                return

            # During post-recovery hold, old persistent suffixes are ignored.
            # Only after the planner publishes a fresh task_trajectory generation
            # marker can an upstream command become the new master reference.
            if not (self._waiting_replan and self._replan_task_seen):
                return

            now = rospy.Time.now().to_sec()
            self._master_reference = copy.deepcopy(msg)
            self._master_duration_s = duration
            self._master_received_ros_s = now
            self._execution_start_ros_s = now
            self._synced_deadline_ros_s = None
            self._waiting_replan = False
            self._replan_task_seen = False
            self._replan_ready_pending = True
            self._replan_count += 1
            self._release_reason = "recovery_replan_installed"

            start = Float64(); start.data = now; self.start_pub.publish(start)
            rospy.logwarn(
                "[vbc_exec_gate] NEW POST-RECOVERY MASTER INSTALLED: duration=%.6f s points=%d replan_count=%d",
                duration, len(msg.points), self._replan_count)
            self._write_trace_locked()

    def _task_trajectory_callback(self, msg: JointTrajectory) -> None:
        if msg is None or not msg.points:
            return
        with self._lock:
            if not self._waiting_replan:
                return
            self._replan_task_seen = True
            self._release_reason = "recovery_replan_task_generated_waiting_command"
            rospy.logwarn(
                "[vbc_exec_gate] fresh post-recovery task trajectory observed; next upstream command will become the new master")

    def _recovery_complete_callback(self, msg: Bool) -> None:
        if msg is None or not bool(msg.data):
            return
        with self._lock:
            if not self._released or self._waiting_replan:
                return
            self._waiting_replan = True
            self._replan_task_seen = False
            self._replan_ready_pending = False
            self._synced_deadline_ros_s = None
            self._release_reason = "recovery_complete_waiting_task_replan"
            rospy.logwarn(
                "[vbc_exec_gate] RECOVERY COMPLETE: freezing old nominal progress and waiting for measured-state task replan")
            self._write_trace_locked()

    def _vbc_summary_callback(self, msg: String) -> None:
        if msg is None:
            return
        t = _tokens(msg.data)
        decision = _as_bool_token(t.get("has_violation"))
        if decision is None:
            return
        with self._lock:
            if self._decision is not None or self._released:
                return
            if self._last_observed_decision is None or decision != self._last_observed_decision:
                self._last_observed_decision = decision
                self._decision_repeat_count = 1
            else:
                self._decision_repeat_count += 1
            if self._decision_repeat_count >= self.decision_stability_count:
                self._decision = decision
                self._waypoint_ready = False
                self._release_reason = (
                    "vbc_violation_waiting_waypoint" if decision
                    else "vbc_safe_waiting_reference")
                rospy.logwarn(
                    "[vbc_exec_gate] latched pre-execution VBC decision: has_violation=%d after %d consistent evaluations",
                    int(decision), self._decision_repeat_count)

    def _waypoint_summary_callback(self, msg: String) -> None:
        if msg is None:
            return
        t = _tokens(msg.data)
        ready = _as_bool_token(t.get("ready"))
        if ready is None:
            return
        with self._lock:
            current_violation_context = (
                self._decision is True and self._frozen_sweep_time_s is not None)
            if ready and current_violation_context and not self._waypoint_ready:
                self._waypoint_ready = True
                rospy.logwarn(
                    "[vbc_exec_gate] explicit visibility waypoint READY for current frozen VBC target")

    def _frozen_sweep_callback(self, msg: Float32) -> None:
        if msg is None:
            return
        value = float(msg.data)
        if not math.isfinite(value) or value < 0.0:
            return
        with self._lock:
            if self._frozen_sweep_time_s is None:
                self._frozen_sweep_time_s = value
                rospy.logwarn(
                    "[vbc_exec_gate] frozen nominal sweep time received: %.6f s", value)

    def _can_release_locked(self) -> bool:
        if self._released or self._master_reference is None or self._decision is None:
            return False
        if not self._decision:
            return True
        return self._waypoint_ready and self._frozen_sweep_time_s is not None

    def _release_locked(self) -> None:
        now = rospy.Time.now().to_sec()
        self._execution_start_ros_s = now
        self._released = True

        if self._decision:
            deadline_from_start = max(
                0.0, float(self._frozen_sweep_time_s) - self.safety_margin_s)
            self._synced_deadline_ros_s = now + deadline_from_start
            self._release_reason = "vbc_violation_waypoint_ready"
        else:
            self._synced_deadline_ros_s = None
            self._release_reason = "vbc_safe_nominal_release"

        start = Float64(); start.data = now; self.start_pub.publish(start)
        ready = Bool(); ready.data = True; self.ready_pub.publish(ready)
        if self._synced_deadline_ros_s is not None:
            d = Float64(); d.data = self._synced_deadline_ros_s; self.deadline_pub.publish(d)
            self._deadline_publish_count += 1

        rospy.logwarn("[vbc_exec_gate] ================================================")
        rospy.logwarn("[vbc_exec_gate] EXECUTION RELEASED: reason=%s", self._release_reason)
        rospy.logwarn(
            "[vbc_exec_gate] T0=%.6f master_duration=%.6f s",
            now, self._master_duration_s)
        if self._synced_deadline_ros_s is not None:
            rospy.logwarn(
                "[vbc_exec_gate] synchronized VBC deadline: +%.6f s absolute=%.6f",
                self._synced_deadline_ros_s - now,
                self._synced_deadline_ros_s)
        rospy.logwarn("[vbc_exec_gate] ================================================")
        self._write_trace_locked()

    @staticmethod
    def _lerp_array(a, b, alpha: float):
        if len(a) == 0 and len(b) == 0:
            return []
        if len(a) != len(b):
            return list(a) if alpha < 0.5 else list(b)
        return [(1.0 - alpha) * float(a[i]) + alpha * float(b[i]) for i in range(len(a))]

    @classmethod
    def _interpolate_point(cls, p0: JointTrajectoryPoint,
                           p1: JointTrajectoryPoint,
                           alpha: float) -> JointTrajectoryPoint:
        out = JointTrajectoryPoint()
        out.positions = cls._lerp_array(p0.positions, p1.positions, alpha)
        out.velocities = cls._lerp_array(p0.velocities, p1.velocities, alpha)
        out.accelerations = cls._lerp_array(p0.accelerations, p1.accelerations, alpha)
        out.effort = cls._lerp_array(p0.effort, p1.effort, alpha)
        out.time_from_start = rospy.Duration(0.0)
        return out

    @classmethod
    def _suffix_from_phase(cls, master: JointTrajectory, phase_s: float) -> JointTrajectory:
        out = JointTrajectory()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = master.header.frame_id
        out.joint_names = list(master.joint_names)
        points = master.points
        if not points:
            return out

        times = [float(p.time_from_start.to_sec()) for p in points]
        phase = max(0.0, float(phase_s))

        if phase >= times[-1] - 1e-9:
            endpoint = copy.deepcopy(points[-1])
            endpoint.time_from_start = rospy.Duration(0.0)
            out.points = [endpoint]
            return out

        if phase <= times[0] + 1e-12:
            out.points = [copy.deepcopy(p) for p in points]
            base = times[0]
            for p in out.points:
                p.time_from_start = rospy.Duration(
                    max(0.0, p.time_from_start.to_sec() - base))
            return out

        hi = 1
        while hi < len(times) and times[hi] < phase:
            hi += 1
        lo = max(0, hi - 1)
        if hi >= len(times):
            endpoint = copy.deepcopy(points[-1])
            endpoint.time_from_start = rospy.Duration(0.0)
            out.points = [endpoint]
            return out

        denom = max(times[hi] - times[lo], 1e-12)
        alpha = min(1.0, max(0.0, (phase - times[lo]) / denom))
        first = cls._interpolate_point(points[lo], points[hi], alpha)
        out.points.append(first)
        for idx in range(hi, len(points)):
            p = copy.deepcopy(points[idx])
            p.time_from_start = rospy.Duration(max(0.0, times[idx] - phase))
            out.points.append(p)
        return out

    def _publish_summary(self) -> None:
        with self._lock:
            decision_text = (
                "unknown" if self._decision is None else
                ("violation" if self._decision else "safe"))
            now = rospy.Time.now().to_sec()
            elapsed = (
                math.nan if self._execution_start_ros_s is None
                else max(0.0, now - self._execution_start_ros_s))
            deadline_remaining = (
                math.nan if self._synced_deadline_ros_s is None
                else self._synced_deadline_ros_s - now)
            msg = String()
            msg.data = (
                f"released={int(self._released)} decision={decision_text} "
                f"waypoint_ready={int(self._waypoint_ready)} "
                f"has_master={int(self._master_reference is not None)} "
                f"master_duration={self._master_duration_s:.6f} "
                f"execution_elapsed={elapsed:.6f} "
                f"deadline_remaining={deadline_remaining:.6f} "
                f"waiting_replan={int(self._waiting_replan)} "
                f"replan_task_seen={int(self._replan_task_seen)} "
                f"replan_count={self._replan_count} "
                f"publish_count={self._publish_count} "
                f"deadline_publish_count={self._deadline_publish_count} "
                f"reason={self._release_reason}"
            )
        self.summary_pub.publish(msg)

    def _trace_payload_locked(self):
        return {
            "trial_label": self.trial_label,
            "released": self._released,
            "decision": None if self._decision is None else (
                "violation" if self._decision else "safe"),
            "decision_repeat_count": self._decision_repeat_count,
            "waypoint_ready": self._waypoint_ready,
            "frozen_sweep_time_s": self._frozen_sweep_time_s,
            "safety_margin_s": self.safety_margin_s,
            "deadline_from_execution_start_s": (
                None if self._decision is not True or self._frozen_sweep_time_s is None
                else max(0.0, self._frozen_sweep_time_s - self.safety_margin_s)),
            "execution_start_ros_s": self._execution_start_ros_s,
            "synced_deadline_ros_s": self._synced_deadline_ros_s,
            "master_duration_s": None if self._master_reference is None else self._master_duration_s,
            "master_points": 0 if self._master_reference is None else len(self._master_reference.points),
            "master_received_ros_s": self._master_received_ros_s,
            "reference_publish_count": self._publish_count,
            "deadline_publish_count": self._deadline_publish_count,
            "waiting_replan": self._waiting_replan,
            "replan_task_seen": self._replan_task_seen,
            "replan_count": self._replan_count,
            "release_reason": self._release_reason,
            "output_reference_topic": self.output_reference_topic,
            "trace_json": str(self.trace_path),
        }

    def _write_trace_locked(self) -> None:
        payload = self._trace_payload_locked()
        tmp = self.trace_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(self.trace_path)

    def _timer_callback(self, _event) -> None:
        reference = None
        deadline_msg = None
        send_replan_ready = False

        with self._lock:
            if self._can_release_locked():
                self._release_locked()

            # During RECOVERY_HOLD no stale nominal reference is published.
            if (self._released and not self._waiting_replan
                    and self._master_reference is not None):
                elapsed = max(
                    0.0, rospy.Time.now().to_sec() - self._execution_start_ros_s)
                reference = self._suffix_from_phase(self._master_reference, elapsed)
                self._publish_count += 1

                if self._synced_deadline_ros_s is not None:
                    deadline_msg = Float64()
                    deadline_msg.data = self._synced_deadline_ros_s
                    self._deadline_publish_count += 1

                if self._replan_ready_pending:
                    # Publish the new gated reference first; the ready pulse below
                    # tells MPC that RECOVERY_HOLD may now end.
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
            rospy.logwarn(
                "[vbc_exec_gate] POST-RECOVERY REPLAN RELEASED from phase zero; MPC may leave recovery hold")
        self._publish_summary()

    def _on_shutdown(self) -> None:
        try:
            with self._lock:
                self._write_trace_locked()
        except Exception:
            pass


def main() -> None:
    rospy.init_node("vbc_execution_reference_gate")
    VBCExecutionReferenceGate()
    rospy.spin()


if __name__ == "__main__":
    main()
