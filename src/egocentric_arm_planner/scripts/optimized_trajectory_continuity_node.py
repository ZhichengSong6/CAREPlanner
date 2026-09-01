#!/usr/bin/env python3
"""Serialize candidate VBC verification and publish only committed trajectories.

C4.4 verification protocol
--------------------------
The C++ VBC selector runs on its own periodic timer and may publish repeated
summaries for the same cached trajectory. Verification is synchronized to
selector cycles: a candidate dispatched after cycle k may only be decided by a
strictly later predicted cycle.

C5.8 executable safety/commit protocol
--------------------------------------
For REPAIR and PROBE_NORMAL, the node first converts the optimizer candidate
into the exact trajectory that could reach the actuator:

    short optimizer prefix + dynamically feasible braking tail + hold tail

When final GCDF is enabled, that exact executable view must first pass the
current-map learned-GCDF audit. The same q-trajectory is then passed to exact
VBC. Only a trajectory that passes BOTH gates is committed.

In the C5.4+ full-trajectory tracker architecture, the committed trajectory is
published exactly once; the tracker owns it until completion. ROS Header.seq is
treated as publisher-local diagnostics only. A committed header timestamp is
used as the stable execution token for PROBE completion handshakes.
"""

from __future__ import annotations

import copy
import math
import re
import threading

import rospy
from care_collision_cdf.msg import CollisionCDFConstraintBatch
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


_TOKEN_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


def _tokens(text):
    return {key: value for key, value in _TOKEN_RE.findall(text or "")}


def _as_bool(value):
    if value in ("1", "true", "True"):
        return True
    if value in ("0", "false", "False"):
        return False
    return None


class OptimizedTrajectoryContinuityNode:
    def __init__(self) -> None:
        self._lock = threading.RLock()

        self.input_topic = str(rospy.get_param(
            "~input_topic", "/care_planner/mpc/predicted_trajectory"))
        self.verification_topic = str(rospy.get_param(
            "~output_topic", "/care_planner/optimized_trajectory"))
        self.committed_topic = str(rospy.get_param(
            "~committed_output_topic", "/care_planner/committed_trajectory"))
        self.execution_audit_topic = str(rospy.get_param(
            "~execution_audit_output_topic",
            "/care_planner/execution/audit_trajectory"))
        self.global_summary_topic = str(rospy.get_param(
            "~global_summary_topic", "/care_planner/trajectory_risk/vbc_summary"))
        self.summary_topic = str(rospy.get_param(
            "~summary_topic", "/care_planner/optimized_trajectory_summary"))
        self.verification_event_topic = str(rospy.get_param(
            "~verification_event_topic", "/care_planner/verification_outcome"))
        self.final_gcdf_query_topic = str(rospy.get_param(
            "~final_gcdf_query_topic",
            "/care_planner/final_gcdf/query_trajectory"))
        self.final_gcdf_batch_topic = str(rospy.get_param(
            "~final_gcdf_batch_topic",
            "/care_planner/final_gcdf/constraint_batch"))
        self.final_gcdf_recovery_trajectory_topic = str(rospy.get_param(
            "~final_gcdf_recovery_trajectory_topic",
            "/care_planner/final_gcdf/recovery_trajectory"))
        self.final_gcdf_recovery_event_topic = str(rospy.get_param(
            "~final_gcdf_recovery_event_topic",
            "/care_planner/final_gcdf/recovery_visibility_event"))
        self.repair_active_topic = str(rospy.get_param(
            "~repair_active_topic",
            "/care_planner/execution/predicted_vbc_recovery_triggered"))
        self.probe_active_topic = str(rospy.get_param(
            "~probe_active_topic",
            "/care_planner/c4_4/probe_active"))
        self.joint_state_topic = str(rospy.get_param(
            "~joint_state_topic", "/care_arm/joint_states"))
        self.probe_start_rebase_enabled = bool(rospy.get_param(
            "~probe_start_rebase_enabled", True))
        self.probe_rebase_max_shift_inf = float(rospy.get_param(
            "~probe_rebase_max_shift_inf", 0.10))
        self.probe_start_joint_state_max_age_s = float(rospy.get_param(
            "~probe_start_joint_state_max_age_s", 0.05))
        self.probe_commit_start_mismatch_max = float(rospy.get_param(
            "~probe_commit_start_mismatch_max", 0.03))
        # C5.21: slow only PROBE task motion before prefix/brake construction.
        # Time scaling preserves q-path geometry and scales dq/ddq consistently.
        self.probe_velocity_cap = float(rospy.get_param(
            "~probe_velocity_cap", 0.50))
        self.probe_time_scale_max = float(rospy.get_param(
            "~probe_time_scale_max", 5.0))
        self.joint_position_margin = float(rospy.get_param(
            "~joint_position_margin", 0.01))
        self.joint_position_limits = dict(rospy.get_param(
            "~joint_position_limits", {}))

        self.rate = float(rospy.get_param("~rate", 20.0))
        self.continuation_enabled = bool(rospy.get_param(
            "~continuation_enabled", False))
        self.execution_audit_enabled = bool(rospy.get_param(
            "~execution_audit_enabled", False))
        self.execution_audit_post_hold_s = float(rospy.get_param(
            "~execution_audit_post_hold_s", 0.15))
        self.continuation_start_delay_s = float(rospy.get_param(
            "~continuation_start_delay_s", 0.065))
        self.continuation_timeout_s = float(rospy.get_param(
            "~continuation_timeout_s", 0.50))
        self.verification_timeout_s = float(rospy.get_param(
            "~verification_timeout_s", 0.25))
        self.final_gcdf_enabled = bool(rospy.get_param(
            "~final_gcdf_enabled", False))
        self.final_gcdf_timeout_s = float(rospy.get_param(
            "~final_gcdf_timeout_s", 0.50))
        self.final_gcdf_safety_margin = float(rospy.get_param(
            "~final_gcdf_safety_margin", 0.0))
        self.final_gcdf_stamp_tolerance_s = float(rospy.get_param(
            "~final_gcdf_stamp_tolerance_s", 1e-6))

        self.repair_prefix_verification_enabled = bool(rospy.get_param(
            "~repair_prefix_verification_enabled", False))
        self.repair_execution_prefix_s = float(rospy.get_param(
            "~repair_execution_prefix_s", 0.15))
        self.repair_brake_dt_s = float(rospy.get_param(
            "~repair_brake_dt_s", 0.05))
        self.repair_hold_s = float(rospy.get_param(
            "~repair_hold_s", 0.10))
        self.repair_brake_max_steps = int(rospy.get_param(
            "~repair_brake_max_steps", 20))
        self.repair_acceleration_limits = [float(v) for v in rospy.get_param(
            "~repair_acceleration_limits", [3.0, 3.0, 4.0, 4.0, 6.0, 6.0, 6.0])]

        if self.rate <= 0.0:
            raise ValueError("~rate must be positive")
        if self.continuation_enabled:
            if self.continuation_start_delay_s <= 0.0:
                raise ValueError("~continuation_start_delay_s must be positive")
            if self.continuation_timeout_s <= self.continuation_start_delay_s:
                raise ValueError("~continuation_timeout_s must exceed start delay")
        if self.verification_timeout_s <= 0.0:
            raise ValueError("~verification_timeout_s must be positive")
        if self.final_gcdf_enabled and self.final_gcdf_timeout_s <= 0.0:
            raise ValueError("~final_gcdf_timeout_s must be positive")
        if self.execution_audit_post_hold_s < 0.0:
            raise ValueError("~execution_audit_post_hold_s must be non-negative")
        if self.final_gcdf_stamp_tolerance_s <= 0.0:
            raise ValueError("~final_gcdf_stamp_tolerance_s must be positive")
        if self.repair_execution_prefix_s <= 0.0:
            raise ValueError("~repair_execution_prefix_s must be positive")
        if self.repair_brake_dt_s <= 0.0 or self.repair_hold_s < 0.0:
            raise ValueError("repair brake dt must be positive and hold must be non-negative")
        if self.repair_brake_max_steps < 1:
            raise ValueError("~repair_brake_max_steps must be >= 1")
        if any(v <= 0.0 for v in self.repair_acceleration_limits):
            raise ValueError("~repair_acceleration_limits must be positive")
        if self.probe_rebase_max_shift_inf <= 0.0:
            raise ValueError("~probe_rebase_max_shift_inf must be positive")
        if self.probe_start_joint_state_max_age_s <= 0.0:
            raise ValueError(
                "~probe_start_joint_state_max_age_s must be positive")
        if self.probe_commit_start_mismatch_max <= 0.0:
            raise ValueError(
                "~probe_commit_start_mismatch_max must be positive")
        if self.probe_velocity_cap <= 0.0:
            raise ValueError("~probe_velocity_cap must be positive")
        if self.probe_time_scale_max < 1.0:
            raise ValueError("~probe_time_scale_max must be >= 1")

        self._repair_active = False
        self._probe_active = False
        self._pending_raw = None
        self._pending_raw_received = None
        self._pending_repair = False
        self._pending_probe = False
        self._last_raw_received = None
        self._last_input_gap_s = math.nan
        self._max_input_gap_s = 0.0
        self._latest_measured_q_by_name = {}
        self._latest_joint_state_received = None

        self._selector_cycle_count = 0
        self._last_selector_cycle_time = None

        self._gcdf_outstanding = None
        self._gcdf_outstanding_sent = None
        self._gcdf_outstanding_seq = 0
        self._gcdf_outstanding_repair = False
        self._gcdf_outstanding_probe = False
        self._gcdf_outstanding_view = "none"
        self._gcdf_outstanding_raw_received = None

        self._outstanding = None
        self._outstanding_sent = None
        self._outstanding_seq = 0
        self._outstanding_dispatch_cycle = -1
        self._outstanding_repair = False
        self._outstanding_probe = False
        self._outstanding_view = "none"
        self._outstanding_raw_received = None
        self._next_verification_seq = 1

        self._committed_master = None
        self._committed_received = None

        self._raw_input_count = 0
        self._final_gcdf_query_count = 0
        self._final_gcdf_safe_count = 0
        self._final_gcdf_unsafe_count = 0
        self._final_gcdf_timeout_count = 0
        self._final_gcdf_stamp_miss_count = 0
        self._last_final_gcdf_min_d = math.nan
        self._final_gcdf_recovery_event_count = 0
        self._final_gcdf_recovery_event_drop_count = 0
        self._last_final_gcdf_recovery_seq = 0
        self._last_final_gcdf_recovery_timestep = -1
        self._last_final_gcdf_recovery_sweep_s = math.nan
        self._last_final_gcdf_recovery_point_count = 0
        self._verification_publish_count = 0
        self._verification_safe_count = 0
        self._verification_unsafe_count = 0
        self._verification_timeout_count = 0
        self._verification_outcome_count = 0
        self._commit_count = 0
        self._committed_publish_count = 0
        self._continuation_count = 0
        self._execution_audit_publish_count = 0
        self._pending_replace_count = 0
        self._repair_prefix_build_count = 0
        self._repair_prefix_safe_count = 0
        self._repair_prefix_unsafe_count = 0
        self._probe_prefix_build_count = 0
        self._probe_prefix_safe_count = 0
        self._probe_prefix_unsafe_count = 0
        self._last_repair_prefix_duration_s = math.nan
        self._last_repair_brake_duration_s = math.nan
        self._last_source = "waiting_selector_cycle"
        self._last_committed_age_s = math.nan
        self._last_verification_age_s = math.nan
        self._last_verification_seq = 0
        self._last_verification_result = "none"
        self._last_verification_view = "none"
        self._last_execution_stamp_ns = 0

        # C5.13 stage timing (wall-clock durations measured by ROS time in the
        # same process, reported in milliseconds).
        self._last_raw_to_safety_dispatch_ms = math.nan
        # C5.19 executable-construction diagnostics. These are observational
        # only and never modify the candidate trajectory.
        self._last_prefix_endpoint_max_abs_velocity = math.nan
        self._last_brake_displacement_inf = math.nan
        self._last_constructed_executable_duration_s = math.nan
        self._probe_rebase_count = 0
        self._probe_rebase_clamp_count = 0
        self._probe_start_continuity_reject_count = 0
        self._last_probe_rebase_shift_inf = math.nan
        self._last_probe_rebase_target_shift_inf = math.nan
        self._last_probe_rebase_residual_inf = math.nan
        self._last_probe_commit_start_mismatch_inf = math.nan
        self._probe_speed_scale_count = 0
        self._probe_speed_scale_clamp_count = 0
        self._last_probe_speed_scale = 1.0
        self._last_probe_speed_max_before = math.nan
        self._last_probe_speed_max_after = math.nan
        self._last_probe_effective_prefix_s = math.nan
        self._gcdf_outstanding_diag = None
        self._outstanding_diag = None
        self._last_verification_diag = None
        self._last_final_gcdf_roundtrip_ms = math.nan
        self._last_exact_vbc_roundtrip_ms = math.nan
        self._last_candidate_total_safety_pipeline_ms = math.nan

        self.final_gcdf_query_pub = rospy.Publisher(
            self.final_gcdf_query_topic, JointTrajectory, queue_size=1)
        self.final_gcdf_recovery_trajectory_pub = rospy.Publisher(
            self.final_gcdf_recovery_trajectory_topic,
            JointTrajectory, queue_size=1, latch=True)
        self.final_gcdf_recovery_event_pub = rospy.Publisher(
            self.final_gcdf_recovery_event_topic,
            Float64MultiArray, queue_size=1, latch=True)
        self.verification_pub = rospy.Publisher(
            self.verification_topic, JointTrajectory, queue_size=1)
        self.committed_pub = rospy.Publisher(
            self.committed_topic, JointTrajectory, queue_size=1)
        self.execution_audit_pub = rospy.Publisher(
            self.execution_audit_topic, JointTrajectory, queue_size=1)
        self.summary_pub = rospy.Publisher(
            self.summary_topic, String, queue_size=1, latch=True)
        self.verification_event_pub = rospy.Publisher(
            self.verification_event_topic, String, queue_size=20, latch=False)

        self.raw_sub = rospy.Subscriber(
            self.input_topic, JointTrajectory, self._raw_cb, queue_size=1)
        self.summary_sub = rospy.Subscriber(
            self.global_summary_topic, String, self._global_summary_cb, queue_size=1)
        self.final_gcdf_batch_sub = rospy.Subscriber(
            self.final_gcdf_batch_topic, CollisionCDFConstraintBatch,
            self._final_gcdf_batch_cb, queue_size=2)
        self.repair_sub = rospy.Subscriber(
            self.repair_active_topic, Bool, self._repair_active_cb, queue_size=1)
        self.probe_sub = rospy.Subscriber(
            self.probe_active_topic, Bool, self._probe_active_cb, queue_size=1)
        self.joint_state_sub = rospy.Subscriber(
            self.joint_state_topic, JointState, self._joint_state_cb, queue_size=1)
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.rate), self._timer_cb)

        self._publish_summary()
        rospy.logwarn(
            "[optimized_trajectory_continuity] SELECTOR-CYCLE VERIFY raw=%s "
            "final_gcdf=%s verify=%s committed=%s audit=%s continuation=%d "
            "audit_enabled=%d repair_prefix=%d prefix=%.3fs brake_dt=%.3fs "
            "hold=%.3fs timeout=%.3fs",
            self.input_topic, self.final_gcdf_query_topic,
            self.verification_topic, self.committed_topic,
            self.execution_audit_topic,
            int(self.continuation_enabled),
            int(self.execution_audit_enabled),
            int(self.repair_prefix_verification_enabled),
            self.repair_execution_prefix_s, self.repair_brake_dt_s,
            self.repair_hold_s, self.verification_timeout_s)

    def _joint_state_cb(self, msg):
        if msg is None or len(msg.name) != len(msg.position):
            return
        now = rospy.Time.now()
        measured = {}
        try:
            for name, position in zip(msg.name, msg.position):
                measured[str(name)] = float(position)
        except Exception:
            return
        with self._lock:
            self._latest_measured_q_by_name = measured
            self._latest_joint_state_received = now

    def _measured_mismatch_inf_locked(self, joint_names, positions):
        if (not joint_names or not positions or
                len(joint_names) != len(positions) or
                not self._latest_measured_q_by_name):
            return math.nan
        errors = []
        for name, position in zip(joint_names, positions):
            if name not in self._latest_measured_q_by_name:
                return math.nan
            errors.append(
                abs(float(position) -
                    float(self._latest_measured_q_by_name[name])))
        return max(errors) if errors else math.nan

    def _joint_state_age_ms_locked(self, now):
        if self._latest_joint_state_received is None:
            return math.nan
        return 1000.0 * max(
            0.0, (now - self._latest_joint_state_received).to_sec())

    def _measured_positions_locked(self, joint_names):
        if not joint_names or not self._latest_measured_q_by_name:
            return None
        measured = []
        for name in joint_names:
            if name not in self._latest_measured_q_by_name:
                return None
            measured.append(float(self._latest_measured_q_by_name[name]))
        return measured

    def _probe_rebase_candidate_locked(self, raw, now):
        """Translate PROBE q(t) so q(0) matches the latest measured q.

        A constant configuration translation preserves dq/ddq and the local
        trajectory shape.  The translated candidate is constructed BEFORE
        final GCDF/VBC, so the exact rebased executable is what gets audited.
        Translation is clipped by both a small max shift and any configured
        joint-position limits.  Any residual start mismatch is checked again
        immediately before commit.
        """
        out = copy.deepcopy(raw)
        if (not self.probe_start_rebase_enabled or raw is None or
                not raw.points or not raw.joint_names):
            return out, {
                "enabled": 0,
                "applied": 0,
                "clamped": 0,
                "target_shift_inf": math.nan,
                "applied_shift_inf": math.nan,
                "residual_inf": math.nan,
                "joint_state_age_ms":
                    self._joint_state_age_ms_locked(now),
            }

        measured = self._measured_positions_locked(raw.joint_names)
        joint_state_age_ms = self._joint_state_age_ms_locked(now)
        if (measured is None or not math.isfinite(joint_state_age_ms) or
                joint_state_age_ms >
                1000.0 * self.probe_start_joint_state_max_age_s):
            return out, {
                "enabled": 1,
                "applied": 0,
                "clamped": 0,
                "target_shift_inf": math.nan,
                "applied_shift_inf": 0.0,
                "residual_inf": self._measured_mismatch_inf_locked(
                    raw.joint_names, raw.points[0].positions),
                "joint_state_age_ms": joint_state_age_ms,
            }

        if len(raw.points[0].positions) != len(raw.joint_names):
            return out, {
                "enabled": 1,
                "applied": 0,
                "clamped": 0,
                "target_shift_inf": math.nan,
                "applied_shift_inf": 0.0,
                "residual_inf": math.nan,
                "joint_state_age_ms": joint_state_age_ms,
            }

        target_delta = [
            measured[j] - float(raw.points[0].positions[j])
            for j in range(len(raw.joint_names))
        ]
        applied_delta = []
        clamped = False
        for j, name in enumerate(raw.joint_names):
            delta = max(
                -self.probe_rebase_max_shift_inf,
                min(self.probe_rebase_max_shift_inf, target_delta[j]))
            if abs(delta - target_delta[j]) > 1e-12:
                clamped = True

            limit = self.joint_position_limits.get(name)
            if isinstance(limit, dict):
                try:
                    lower = float(limit["lower"]) + self.joint_position_margin
                    upper = float(limit["upper"]) - self.joint_position_margin
                    positions = [
                        float(p.positions[j]) for p in raw.points
                        if len(p.positions) == len(raw.joint_names)
                    ]
                    if len(positions) == len(raw.points):
                        delta_lo = max(lower - q for q in positions)
                        delta_hi = min(upper - q for q in positions)
                        clipped = max(delta_lo, min(delta_hi, delta))
                        if abs(clipped - delta) > 1e-12:
                            clamped = True
                        delta = clipped
                except Exception:
                    pass
            applied_delta.append(delta)

        for p in out.points:
            if len(p.positions) != len(raw.joint_names):
                continue
            p.positions = [
                float(p.positions[j]) + applied_delta[j]
                for j in range(len(raw.joint_names))
            ]

        applied_inf = max(abs(v) for v in applied_delta) if applied_delta else 0.0
        target_inf = max(abs(v) for v in target_delta) if target_delta else 0.0
        residual_inf = self._measured_mismatch_inf_locked(
            out.joint_names, out.points[0].positions)

        self._probe_rebase_count += 1
        if clamped:
            self._probe_rebase_clamp_count += 1
        self._last_probe_rebase_shift_inf = applied_inf
        self._last_probe_rebase_target_shift_inf = target_inf
        self._last_probe_rebase_residual_inf = residual_inf

        return out, {
            "enabled": 1,
            "applied": 1,
            "clamped": int(clamped),
            "target_shift_inf": target_inf,
            "applied_shift_inf": applied_inf,
            "residual_inf": residual_inf,
            "joint_state_age_ms": joint_state_age_ms,
        }

    def _repair_active_cb(self, msg):
        if msg is None:
            return
        with self._lock:
            self._repair_active = bool(msg.data)

    def _probe_active_cb(self, msg):
        if msg is None:
            return
        with self._lock:
            self._probe_active = bool(msg.data)

    @staticmethod
    def _duration(msg: JointTrajectory) -> float:
        if msg is None or not msg.points:
            return -1.0
        return float(msg.points[-1].time_from_start.to_sec())

    @staticmethod
    def _lerp_array(a, b, alpha):
        if len(a) == 0 and len(b) == 0:
            return []
        if len(a) != len(b):
            return list(a) if alpha < 0.5 else list(b)
        return [(1.0 - alpha) * float(a[i]) + alpha * float(b[i])
                for i in range(len(a))]

    @classmethod
    def _interpolate_point(cls, p0, p1, alpha):
        out = JointTrajectoryPoint()
        out.positions = cls._lerp_array(p0.positions, p1.positions, alpha)
        out.velocities = cls._lerp_array(p0.velocities, p1.velocities, alpha)
        out.accelerations = cls._lerp_array(p0.accelerations, p1.accelerations, alpha)
        out.effort = cls._lerp_array(p0.effort, p1.effort, alpha)
        out.time_from_start = rospy.Duration(0.0)
        return out

    @classmethod
    def _point_at_time(cls, msg, t_s):
        if msg is None or not msg.points:
            return None
        t = max(0.0, min(float(t_s), cls._duration(msg)))
        points = msg.points
        if len(points) == 1 or t <= points[0].time_from_start.to_sec():
            out = copy.deepcopy(points[0])
            out.time_from_start = rospy.Duration(t)
            return out
        if t >= points[-1].time_from_start.to_sec():
            out = copy.deepcopy(points[-1])
            out.time_from_start = rospy.Duration(t)
            return out
        hi = 1
        while hi < len(points) and points[hi].time_from_start.to_sec() < t:
            hi += 1
        lo = hi - 1
        t0 = points[lo].time_from_start.to_sec()
        t1 = points[hi].time_from_start.to_sec()
        alpha = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        out = cls._interpolate_point(points[lo], points[hi], alpha)
        out.time_from_start = rospy.Duration(t)
        return out

    @staticmethod
    def _position_shift_inf(a, b):
        if a is None or b is None or len(a) != len(b) or not a:
            return math.nan
        try:
            return max(abs(float(a[i]) - float(b[i])) for i in range(len(a)))
        except Exception:
            return math.nan

    @classmethod
    def _suffix_from_phase(cls, master, phase_s):
        if master is None or not master.points:
            return None
        duration = cls._duration(master)
        if not math.isfinite(duration) or duration < 0.0:
            return None
        phase = max(0.0, min(float(phase_s), duration))

        out = JointTrajectory()
        out.header = copy.deepcopy(master.header)
        out.header.stamp = rospy.Time.now()
        out.joint_names = list(master.joint_names)
        points = master.points

        start_point = cls._point_at_time(master, phase)
        if start_point is None:
            return None
        start_point.time_from_start = rospy.Duration(0.0)
        out.points.append(start_point)
        for point in points:
            t = point.time_from_start.to_sec()
            if t <= phase + 1e-9:
                continue
            p = copy.deepcopy(point)
            p.time_from_start = rospy.Duration(t - phase)
            out.points.append(p)

        if len(out.points) == 1:
            endpoint = copy.deepcopy(out.points[0])
            endpoint.time_from_start = rospy.Duration(0.05)
            if endpoint.velocities:
                endpoint.velocities = [0.0 for _ in endpoint.velocities]
            if endpoint.accelerations:
                endpoint.accelerations = [0.0 for _ in endpoint.accelerations]
            out.points.append(endpoint)
        return out

    @staticmethod
    def _trajectory_max_abs_velocity(candidate):
        if candidate is None or not candidate.points:
            return math.nan
        vmax = 0.0
        found = False
        n = len(candidate.joint_names)
        for p in candidate.points:
            if p.velocities and (n == 0 or len(p.velocities) == n):
                for v in p.velocities:
                    try:
                        vmax = max(vmax, abs(float(v)))
                        found = True
                    except Exception:
                        pass
        if found:
            return vmax

        # Fallback for trajectories without explicit velocity fields.
        for i in range(1, len(candidate.points)):
            p0 = candidate.points[i - 1]
            p1 = candidate.points[i]
            if (not p0.positions or not p1.positions or
                    len(p0.positions) != len(p1.positions)):
                continue
            dt = (p1.time_from_start - p0.time_from_start).to_sec()
            if dt <= 1e-9:
                continue
            for q0, q1 in zip(p0.positions, p1.positions):
                vmax = max(vmax, abs((float(q1) - float(q0)) / dt))
                found = True
        return vmax if found else math.nan

    @staticmethod
    def _time_scale_trajectory(candidate, scale):
        if candidate is None or not candidate.points:
            return None
        scale = max(1.0, float(scale))
        out = copy.deepcopy(candidate)
        inv = 1.0 / scale
        inv2 = inv * inv
        for p in out.points:
            p.time_from_start = rospy.Duration(
                p.time_from_start.to_sec() * scale)
            if p.velocities:
                p.velocities = [float(v) * inv for v in p.velocities]
            if p.accelerations:
                p.accelerations = [float(a) * inv2 for a in p.accelerations]
        out.header.stamp = rospy.Time.now()
        return out

    def _slow_probe_candidate(self, candidate):
        vmax_before = self._trajectory_max_abs_velocity(candidate)
        if not math.isfinite(vmax_before) or vmax_before <= 0.0:
            return copy.deepcopy(candidate), {
                "speed_max_before": vmax_before,
                "speed_max_after": vmax_before,
                "time_scale": 1.0,
                "time_scale_clamped": 0,
            }

        requested = max(1.0, vmax_before / self.probe_velocity_cap)
        scale = min(requested, self.probe_time_scale_max)
        clamped = requested > self.probe_time_scale_max + 1e-12
        out = self._time_scale_trajectory(candidate, scale)
        vmax_after = self._trajectory_max_abs_velocity(out)

        if scale > 1.0 + 1e-12:
            self._probe_speed_scale_count += 1
        if clamped:
            self._probe_speed_scale_clamp_count += 1
        self._last_probe_speed_scale = scale
        self._last_probe_speed_max_before = vmax_before
        self._last_probe_speed_max_after = vmax_after

        return out, {
            "speed_max_before": vmax_before,
            "speed_max_after": vmax_after,
            "time_scale": scale,
            "time_scale_clamped": int(clamped),
        }

    def _repair_prefix_with_braking_tail(
            self, candidate, prefix_duration_s=None):
        if candidate is None or not candidate.points:
            return None
        duration = self._duration(candidate)
        if duration <= 0.0:
            return None
        requested_prefix_s = (
            self.repair_execution_prefix_s
            if prefix_duration_s is None else float(prefix_duration_s))
        prefix_t = min(max(1e-6, requested_prefix_s), duration)
        endpoint = self._point_at_time(candidate, prefix_t)
        if endpoint is None or not endpoint.positions:
            return None

        n = len(candidate.joint_names)
        if n == 0:
            n = len(endpoint.positions)
        if len(endpoint.positions) != n:
            return None
        if len(self.repair_acceleration_limits) != n:
            rospy.logerr_throttle(
                1.0,
                "[optimized_trajectory_continuity] repair acceleration-limit size %d != dof %d",
                len(self.repair_acceleration_limits), n)
            return None

        out = JointTrajectory()
        out.header = copy.deepcopy(candidate.header)
        out.header.stamp = rospy.Time.now()
        out.joint_names = list(candidate.joint_names)

        for p in candidate.points:
            t = p.time_from_start.to_sec()
            if t < prefix_t - 1e-9:
                out.points.append(copy.deepcopy(p))
        endpoint = copy.deepcopy(endpoint)
        endpoint.time_from_start = rospy.Duration(prefix_t)
        out.points.append(endpoint)

        q = [float(v) for v in endpoint.positions]
        prefix_endpoint_q = list(q)
        if len(endpoint.velocities) == n:
            vel = [float(v) for v in endpoint.velocities]
        elif len(out.points) >= 2:
            p0 = out.points[-2]
            dt = prefix_t - p0.time_from_start.to_sec()
            if dt > 1e-9 and len(p0.positions) == n:
                vel = [(q[j] - float(p0.positions[j])) / dt for j in range(n)]
            else:
                vel = [0.0] * n
        else:
            vel = [0.0] * n

        self._last_prefix_endpoint_max_abs_velocity = (
            max(abs(v) for v in vel) if vel else 0.0)
        t = prefix_t
        brake_start = t
        for _ in range(self.repair_brake_max_steps):
            if max(abs(v) for v in vel) <= 1e-6:
                break
            dt = self.repair_brake_dt_s
            next_vel = []
            accel = []
            for j, v in enumerate(vel):
                max_dv = self.repair_acceleration_limits[j] * dt
                if v > max_dv:
                    vn = v - max_dv
                elif v < -max_dv:
                    vn = v + max_dv
                else:
                    vn = 0.0
                next_vel.append(vn)
                accel.append((vn - v) / dt)
            q = [q[j] + 0.5 * (vel[j] + next_vel[j]) * dt
                 for j in range(n)]
            t += dt
            p = JointTrajectoryPoint()
            p.positions = list(q)
            p.velocities = list(next_vel)
            p.accelerations = list(accel)
            p.time_from_start = rospy.Duration(t)
            out.points.append(p)
            vel = next_vel

        if max(abs(v) for v in vel) > 1e-5:
            rospy.logwarn_throttle(
                1.0,
                "[optimized_trajectory_continuity] braking tail hit max steps; max|v|=%.4f",
                max(abs(v) for v in vel))
            return None

        if self.repair_hold_s > 1e-9:
            t += self.repair_hold_s
            hold = JointTrajectoryPoint()
            hold.positions = list(q)
            hold.velocities = [0.0] * n
            hold.accelerations = [0.0] * n
            hold.time_from_start = rospy.Duration(t)
            out.points.append(hold)

        self._last_repair_prefix_duration_s = prefix_t
        self._last_repair_brake_duration_s = max(
            0.0, t - self.repair_hold_s - brake_start)
        self._last_brake_displacement_inf = self._position_shift_inf(
            q, prefix_endpoint_q)
        self._last_constructed_executable_duration_s = t
        return out

    def _raw_cb(self, msg):
        if msg is None or not msg.points:
            return
        now = rospy.Time.now()
        with self._lock:
            if self._last_raw_received is not None:
                gap = (now - self._last_raw_received).to_sec()
                if gap >= 0.0:
                    self._last_input_gap_s = gap
                    self._max_input_gap_s = max(self._max_input_gap_s, gap)
            if self._pending_raw is not None:
                self._pending_replace_count += 1
            self._pending_raw = copy.deepcopy(msg)
            self._pending_raw_received = now
            self._pending_repair = bool(self._repair_active)
            self._pending_probe = bool(self._probe_active)
            self._last_raw_received = now
            self._raw_input_count += 1
            self._last_source = "raw_candidate_buffered_waiting_selector_cycle"
            self._publish_summary_locked()

    def _dispatch_pending_locked(self, now):
        if (self._gcdf_outstanding is not None or
                self._outstanding is not None or
                self._pending_raw is None):
            return None, "none"

        raw = self._pending_raw
        raw_received = self._pending_raw_received
        pending_repair = bool(self._pending_repair or self._repair_active)
        pending_probe = bool(self._pending_probe or self._probe_active)
        self._pending_raw = None
        self._pending_raw_received = None
        self._pending_repair = False
        self._pending_probe = False

        age = max(0.0, (now - raw_received).to_sec())
        self._last_raw_to_safety_dispatch_ms = 1000.0 * age
        raw_start_positions = (
            list(raw.points[0].positions)
            if raw.points and raw.points[0].positions else [])
        raw_duration_s = self._duration(raw)
        rebase_diag = {
            "enabled": 0,
            "applied": 0,
            "clamped": 0,
            "target_shift_inf": math.nan,
            "applied_shift_inf": math.nan,
            "residual_inf": math.nan,
            "joint_state_age_ms": self._joint_state_age_ms_locked(now),
        }
        speed_diag = {
            "speed_max_before": math.nan,
            "speed_max_after": math.nan,
            "time_scale": 1.0,
            "time_scale_clamped": 0,
        }
        if pending_probe:
            # A candidate is NOT being executed while it waits for admission.
            # Do not advance it by wall-clock age. Rebase its q-origin to the
            # latest measured state, then slow its time parameterization before
            # building/auditing the exact executable.
            candidate, rebase_diag = self._probe_rebase_candidate_locked(
                raw, now)
            candidate, speed_diag = self._slow_probe_candidate(candidate)
            dispatch_suffix_phase_s = 0.0
        else:
            candidate = self._suffix_from_phase(raw, age)
            dispatch_suffix_phase_s = float(age)
        if candidate is None or not candidate.points:
            self._last_source = "discarded_expired_raw_candidate"
            return None, "none"

        dispatch_start_positions = (
            list(candidate.points[0].positions)
            if candidate.points[0].positions else [])
        diag = {
            "raw_candidate_age_s": float(age),
            "dispatch_suffix_phase_s": dispatch_suffix_phase_s,
            "raw_candidate_duration_s": float(raw_duration_s),
            "dispatch_suffix_start_shift_inf": (
                0.0 if pending_probe else self._position_shift_inf(
                    dispatch_start_positions, raw_start_positions)),
            "probe_rebase_enabled": int(rebase_diag.get("enabled", 0)),
            "probe_rebase_applied": int(rebase_diag.get("applied", 0)),
            "probe_rebase_clamped": int(rebase_diag.get("clamped", 0)),
            "probe_rebase_target_shift_inf": rebase_diag.get(
                "target_shift_inf", math.nan),
            "probe_rebase_shift_inf": rebase_diag.get(
                "applied_shift_inf", math.nan),
            "probe_rebase_residual_inf": rebase_diag.get(
                "residual_inf", math.nan),
            "probe_rebase_joint_state_age_ms": rebase_diag.get(
                "joint_state_age_ms", math.nan),
            "probe_speed_max_before": speed_diag.get(
                "speed_max_before", math.nan),
            "probe_speed_max_after": speed_diag.get(
                "speed_max_after", math.nan),
            "probe_time_scale": speed_diag.get("time_scale", 1.0),
            "probe_time_scale_clamped": int(
                speed_diag.get("time_scale_clamped", 0)),
            "probe_effective_prefix_s": (
                self.repair_execution_prefix_s *
                float(speed_diag.get("time_scale", 1.0))
                if pending_probe else self.repair_execution_prefix_s),
            "raw_start_measured_mismatch_inf_at_dispatch":
                self._measured_mismatch_inf_locked(
                    raw.joint_names, raw_start_positions),
            "dispatch_start_measured_mismatch_inf":
                self._measured_mismatch_inf_locked(
                    candidate.joint_names, dispatch_start_positions),
            "dispatch_joint_state_age_ms":
                self._joint_state_age_ms_locked(now),
            "raw_start_positions": raw_start_positions,
            "dispatch_start_positions": dispatch_start_positions,
            "prefix_endpoint_max_abs_velocity": math.nan,
            "brake_duration_s": math.nan,
            "brake_displacement_inf": math.nan,
            "constructed_executable_duration_s": float(self._duration(candidate)),
            "precommit_suffix_phase_s": math.nan,
            "precommit_suffix_start_shift_inf": math.nan,
            "total_start_shift_inf": math.nan,
            "commit_start_measured_mismatch_inf": math.nan,
            "commit_joint_state_age_ms": math.nan,
            "committed_duration_s": math.nan,
        }

        view = "full_horizon"
        prefix_mode = bool(pending_repair or pending_probe)
        if self.repair_prefix_verification_enabled and prefix_mode:
            effective_prefix_s = (
                self.repair_execution_prefix_s *
                float(speed_diag.get("time_scale", 1.0))
                if pending_probe else self.repair_execution_prefix_s)
            self._last_probe_effective_prefix_s = (
                effective_prefix_s if pending_probe
                else self._last_probe_effective_prefix_s)
            candidate = self._repair_prefix_with_braking_tail(
                candidate, prefix_duration_s=effective_prefix_s)
            if candidate is None or not candidate.points:
                self._last_source = "prefix_build_failed"
                return None, "none"
            diag["prefix_endpoint_max_abs_velocity"] = (
                self._last_prefix_endpoint_max_abs_velocity)
            diag["brake_duration_s"] = self._last_repair_brake_duration_s
            diag["brake_displacement_inf"] = self._last_brake_displacement_inf
            diag["constructed_executable_duration_s"] = (
                self._last_constructed_executable_duration_s)
            if pending_repair:
                view = "repair_prefix_brake_hold"
                self._repair_prefix_build_count += 1
            else:
                view = "probe_prefix_brake_hold"
                self._probe_prefix_build_count += 1

        seq = self._next_verification_seq
        self._next_verification_seq += 1
        candidate.header.seq = seq
        candidate.header.stamp = now

        if self.final_gcdf_enabled:
            # C5.7/C5.8: audit the exact executable view against learned GCDF
            # first. Only this same q-trajectory may proceed to exact VBC.
            self._gcdf_outstanding = copy.deepcopy(candidate)
            self._gcdf_outstanding_sent = now
            self._gcdf_outstanding_seq = seq
            self._gcdf_outstanding_repair = bool(pending_repair)
            self._gcdf_outstanding_probe = bool(pending_probe)
            self._gcdf_outstanding_view = view
            self._gcdf_outstanding_raw_received = raw_received
            self._gcdf_outstanding_diag = copy.deepcopy(diag)
            self._final_gcdf_query_count += 1
            self._last_source = "candidate_sent_to_final_gcdf"
            self._last_verification_view = view
            self._publish_summary_locked()
            return candidate, "gcdf"

        # Legacy/non-local path: preserve the original exact-VBC-only commit
        # protocol when no final learned-GCDF transport is configured.
        self._outstanding = copy.deepcopy(candidate)
        self._outstanding_sent = now
        self._outstanding_seq = seq
        self._outstanding_dispatch_cycle = self._selector_cycle_count
        self._outstanding_repair = bool(pending_repair)
        self._outstanding_probe = bool(pending_probe)
        self._outstanding_view = view
        self._outstanding_raw_received = raw_received
        self._outstanding_diag = copy.deepcopy(diag)
        self._verification_publish_count += 1
        self._last_source = "candidate_sent_directly_to_exact_vbc"
        self._last_verification_view = view
        self._publish_summary_locked()
        return candidate, "vbc"

    def _publish_committed(self, msg, source, age_s):
        if msg is None or not msg.points:
            return
        self.committed_pub.publish(msg)
        if self.execution_audit_enabled:
            self.execution_audit_pub.publish(copy.deepcopy(msg))
        with self._lock:
            self._committed_publish_count += 1
            if source == "committed_continuation":
                self._continuation_count += 1
            if self.execution_audit_enabled and source != "committed_continuation":
                self._execution_audit_publish_count += 1
            self._last_source = source
            self._last_committed_age_s = float(age_s)
            self._publish_summary_locked()

    @staticmethod
    def _diag_value(diag, key):
        if not diag:
            return "nan"
        try:
            value = float(diag.get(key, math.nan))
            return "nan" if not math.isfinite(value) else "{:.6f}".format(value)
        except Exception:
            return "nan"

    def _make_verification_event(
            self, seq, result, committed, age_s, view, safety_gate="vbc",
            execution_stamp_ns=0, diag=None):
        msg = String()
        msg.data = (
            "seq={} result={} committed={} verification_age_s={:.6f} "
            "verification_view={} safety_gate={} execution_stamp_ns={} "
            "outcome_count={} safe_count={} unsafe_count={} timeout_count={} "
            "commit_count={} "
            "raw_candidate_age_s={} dispatch_suffix_phase_s={} "
            "dispatch_suffix_start_shift_inf={} "
            "probe_rebase_enabled={} probe_rebase_applied={} "
            "probe_rebase_clamped={} "
            "probe_rebase_target_shift_inf={} probe_rebase_shift_inf={} "
            "probe_rebase_residual_inf={} "
            "probe_rebase_joint_state_age_ms={} "
            "probe_speed_max_before={} probe_speed_max_after={} "
            "probe_time_scale={} probe_time_scale_clamped={} "
            "probe_effective_prefix_s={} "
            "precommit_suffix_phase_s={} precommit_suffix_start_shift_inf={} "
            "total_start_shift_inf={} "
            "raw_start_measured_mismatch_inf_at_dispatch={} "
            "dispatch_start_measured_mismatch_inf={} "
            "dispatch_joint_state_age_ms={} "
            "commit_start_measured_mismatch_inf={} "
            "commit_joint_state_age_ms={} "
            "raw_candidate_duration_s={} "
            "constructed_executable_duration_s={} committed_duration_s={} "
            "prefix_endpoint_max_abs_velocity={} brake_duration_s={} "
            "brake_displacement_inf={}"
        ).format(
            int(seq), result, int(bool(committed)), float(age_s), view,
            safety_gate, int(execution_stamp_ns),
            self._verification_outcome_count,
            self._verification_safe_count, self._verification_unsafe_count,
            self._verification_timeout_count, self._commit_count,
            self._diag_value(diag, "raw_candidate_age_s"),
            self._diag_value(diag, "dispatch_suffix_phase_s"),
            self._diag_value(diag, "dispatch_suffix_start_shift_inf"),
            self._diag_value(diag, "probe_rebase_enabled"),
            self._diag_value(diag, "probe_rebase_applied"),
            self._diag_value(diag, "probe_rebase_clamped"),
            self._diag_value(diag, "probe_rebase_target_shift_inf"),
            self._diag_value(diag, "probe_rebase_shift_inf"),
            self._diag_value(diag, "probe_rebase_residual_inf"),
            self._diag_value(diag, "probe_rebase_joint_state_age_ms"),
            self._diag_value(diag, "probe_speed_max_before"),
            self._diag_value(diag, "probe_speed_max_after"),
            self._diag_value(diag, "probe_time_scale"),
            self._diag_value(diag, "probe_time_scale_clamped"),
            self._diag_value(diag, "probe_effective_prefix_s"),
            self._diag_value(diag, "precommit_suffix_phase_s"),
            self._diag_value(diag, "precommit_suffix_start_shift_inf"),
            self._diag_value(diag, "total_start_shift_inf"),
            self._diag_value(
                diag, "raw_start_measured_mismatch_inf_at_dispatch"),
            self._diag_value(
                diag, "dispatch_start_measured_mismatch_inf"),
            self._diag_value(diag, "dispatch_joint_state_age_ms"),
            self._diag_value(
                diag, "commit_start_measured_mismatch_inf"),
            self._diag_value(diag, "commit_joint_state_age_ms"),
            self._diag_value(diag, "raw_candidate_duration_s"),
            self._diag_value(diag, "constructed_executable_duration_s"),
            self._diag_value(diag, "committed_duration_s"),
            self._diag_value(diag, "prefix_endpoint_max_abs_velocity"),
            self._diag_value(diag, "brake_duration_s"),
            self._diag_value(diag, "brake_displacement_inf"))
        return msg

    def _build_final_gcdf_recovery_evidence(
            self, candidate, seq, batch, distances):
        """Extract the earliest real low-confidence blocker from final GCDF.

        The final-GCDF batch already contains the low-confidence voxel point
        paired with each executable body anchor.  When the exact executable
        prefix+brake+hold is rejected, use the earliest violated timestep as
        active-sensing evidence instead of guessing from the nominal task VBC.
        """
        unsafe = []
        for i, d in enumerate(distances):
            if (not math.isfinite(d) or
                    d >= self.final_gcdf_safety_margin or
                    i >= len(batch.original_timestep)):
                continue
            k = int(batch.original_timestep[i])
            unsafe.append((k, i))
        if not unsafe:
            return None, None

        earliest = min(k for k, _ in unsafe)
        if earliest < 0 or earliest >= len(candidate.points):
            return None, None

        sweep_s = float(
            candidate.points[earliest].time_from_start.to_sec())
        if not math.isfinite(sweep_s) or sweep_s < 0.0:
            return None, None

        points = []
        seen = set()
        for k, i in unsafe:
            if k != earliest:
                continue
            j = 3 * i
            if j + 2 >= len(batch.point_flat):
                continue
            p = (
                float(batch.point_flat[j]),
                float(batch.point_flat[j + 1]),
                float(batch.point_flat[j + 2]),
            )
            if not all(math.isfinite(v) for v in p):
                continue
            # Exact float identity is enough here because all points originate
            # from the same confidence-map voxel centers.
            if p in seen:
                continue
            seen.add(p)
            points.append(p)

        if not points:
            return None, None

        # Correlate the rejected executable and its blocker event by the
        # exact executable header.stamp.  Header.seq is retained only as a
        # human-readable diagnostic; ROS topic ordering / stale seq values must
        # not be the transaction identity.
        traj = copy.deepcopy(candidate)
        traj.header.seq = int(seq)
        stamp = traj.header.stamp

        event = Float64MultiArray()
        event.data = [
            float(seq),
            float(stamp.secs),
            float(stamp.nsecs),
            float(sweep_s),
            float(earliest),
            float(len(points)),
        ]
        for p in points:
            event.data.extend([float(p[0]), float(p[1]), float(p[2])])
        return traj, event

    def _final_gcdf_batch_cb(self, msg):
        if msg is None:
            return

        now = rospy.Time.now()
        verification_to_publish = None
        next_gcdf_to_publish = None
        event_to_publish = None
        gcdf_recovery_trajectory_to_publish = None
        gcdf_recovery_event_to_publish = None

        with self._lock:
            if (self._gcdf_outstanding is None or
                    self._gcdf_outstanding_sent is None):
                return

            expected_stamp = self._gcdf_outstanding.header.stamp
            dt = abs((msg.header.stamp - expected_stamp).to_sec())
            if dt > self.final_gcdf_stamp_tolerance_s:
                self._final_gcdf_stamp_miss_count += 1
                self._publish_summary_locked()
                return

            candidate = copy.deepcopy(self._gcdf_outstanding)
            seq = int(self._gcdf_outstanding_seq)
            was_repair = bool(self._gcdf_outstanding_repair)
            was_probe = bool(self._gcdf_outstanding_probe)
            view = str(self._gcdf_outstanding_view)
            raw_received = self._gcdf_outstanding_raw_received
            diag = copy.deepcopy(self._gcdf_outstanding_diag)
            age = max(0.0, (now - self._gcdf_outstanding_sent).to_sec())
            self._last_final_gcdf_roundtrip_ms = 1000.0 * age

            self._gcdf_outstanding = None
            self._gcdf_outstanding_sent = None
            self._gcdf_outstanding_seq = 0
            self._gcdf_outstanding_repair = False
            self._gcdf_outstanding_probe = False
            self._gcdf_outstanding_view = "none"
            self._gcdf_outstanding_raw_received = None
            self._gcdf_outstanding_diag = None

            distances = [float(d) for d in msg.distance]
            finite_distances = [d for d in distances if math.isfinite(d)]
            malformed = (
                len(finite_distances) != len(distances) or
                len(distances) != int(msg.num_pairs))
            min_d = min(finite_distances) if finite_distances else math.inf
            self._last_final_gcdf_min_d = min_d

            gcdf_safe = (
                not malformed and
                (int(msg.num_pairs) == 0 or
                 min_d >= self.final_gcdf_safety_margin))

            if gcdf_safe:
                self._final_gcdf_safe_count += 1
                # GCDF certifies the q-trajectory geometry; refresh only the
                # epoch before exact VBC so its temporal audit starts now.
                candidate.header.stamp = now
                self._outstanding = candidate
                self._outstanding_sent = now
                self._outstanding_seq = seq
                self._outstanding_dispatch_cycle = self._selector_cycle_count
                self._outstanding_repair = was_repair
                self._outstanding_probe = was_probe
                self._outstanding_view = view
                self._outstanding_raw_received = raw_received
                self._outstanding_diag = copy.deepcopy(diag)
                self._verification_publish_count += 1
                self._last_source = "final_gcdf_safe_sent_to_exact_vbc"
                verification_to_publish = copy.deepcopy(candidate)
            else:
                self._final_gcdf_unsafe_count += 1
                self._verification_outcome_count += 1
                self._verification_unsafe_count += 1
                if was_repair and view == "repair_prefix_brake_hold":
                    self._repair_prefix_unsafe_count += 1
                if was_probe and view == "probe_prefix_brake_hold":
                    self._probe_prefix_unsafe_count += 1
                self._last_verification_seq = seq
                self._last_verification_result = "unsafe"
                self._last_verification_age_s = age
                self._last_verification_view = view
                if raw_received is not None:
                    self._last_candidate_total_safety_pipeline_ms = (
                        1000.0 * max(0.0, (now - raw_received).to_sec()))
                self._last_source = "candidate_rejected_final_gcdf_unsafe"

                (
                    gcdf_recovery_trajectory_to_publish,
                    gcdf_recovery_event_to_publish,
                ) = self._build_final_gcdf_recovery_evidence(
                    candidate, seq, msg, distances)
                if gcdf_recovery_event_to_publish is not None:
                    self._final_gcdf_recovery_event_count += 1
                    self._last_final_gcdf_recovery_seq = seq
                    data = list(gcdf_recovery_event_to_publish.data)
                    self._last_final_gcdf_recovery_sweep_s = float(data[3])
                    self._last_final_gcdf_recovery_timestep = int(round(data[4]))
                    self._last_final_gcdf_recovery_point_count = int(round(data[5]))
                else:
                    self._final_gcdf_recovery_event_drop_count += 1

                event_to_publish = self._make_verification_event(
                    seq, "unsafe", False, age, view, safety_gate="gcdf",
                    diag=diag)
                dispatched, route = self._dispatch_pending_locked(now)
                if route == "gcdf":
                    next_gcdf_to_publish = dispatched
                elif route == "vbc":
                    verification_to_publish = dispatched

            self._publish_summary_locked()

        if gcdf_recovery_trajectory_to_publish is not None:
            self.final_gcdf_recovery_trajectory_pub.publish(
                gcdf_recovery_trajectory_to_publish)
        if gcdf_recovery_event_to_publish is not None:
            self.final_gcdf_recovery_event_pub.publish(
                gcdf_recovery_event_to_publish)
        if verification_to_publish is not None:
            self.verification_pub.publish(verification_to_publish)
        if event_to_publish is not None:
            self.verification_event_pub.publish(event_to_publish)
        if next_gcdf_to_publish is not None:
            self.final_gcdf_query_pub.publish(next_gcdf_to_publish)

    def _global_summary_cb(self, msg):
        if msg is None:
            return
        fields = _tokens(msg.data)
        source = fields.get("trajectory_source", "")
        if source not in ("bootstrap", "predicted"):
            return
        violation = _as_bool(fields.get("has_violation"))
        if violation is None:
            return

        now = rospy.Time.now()
        committed_to_publish = None
        event_to_publish = None
        verification_to_publish = None
        gcdf_to_publish = None
        with self._lock:
            self._selector_cycle_count += 1
            self._last_selector_cycle_time = now

            if self._outstanding is None or self._outstanding_sent is None:
                dispatched, route = self._dispatch_pending_locked(now)
                if route == "gcdf":
                    gcdf_to_publish = dispatched
                elif route == "vbc":
                    verification_to_publish = dispatched
                if (dispatched is None and
                        self._gcdf_outstanding is None and
                        self._outstanding is None):
                    self._last_source = "selector_cycle_complete_no_pending_candidate"
                self._publish_summary_locked()
            else:
                if source != "predicted":
                    self._publish_summary_locked()
                elif self._selector_cycle_count <= self._outstanding_dispatch_cycle:
                    self._publish_summary_locked()
                else:
                    verification_age = max(
                        0.0, (now - self._outstanding_sent).to_sec())
                    self._last_exact_vbc_roundtrip_ms = 1000.0 * verification_age
                    raw_received = self._outstanding_raw_received
                    diag = copy.deepcopy(self._outstanding_diag)
                    if raw_received is not None:
                        self._last_candidate_total_safety_pipeline_ms = (
                            1000.0 * max(0.0, (now - raw_received).to_sec()))
                    candidate = copy.deepcopy(self._outstanding)
                    seq = int(self._outstanding_seq)
                    was_repair = bool(self._outstanding_repair)
                    was_probe = bool(self._outstanding_probe)
                    view = str(self._outstanding_view)
                    self._outstanding = None
                    self._outstanding_sent = None
                    self._outstanding_seq = 0
                    self._outstanding_dispatch_cycle = -1
                    self._outstanding_repair = False
                    self._outstanding_probe = False
                    self._outstanding_view = "none"
                    self._outstanding_raw_received = None
                    self._outstanding_diag = None
                    self._last_verification_age_s = verification_age
                    self._last_verification_seq = seq
                    self._last_verification_view = view
                    self._verification_outcome_count += 1

                    if violation:
                        self._verification_unsafe_count += 1
                        if was_repair and view == "repair_prefix_brake_hold":
                            self._repair_prefix_unsafe_count += 1
                        if was_probe and view == "probe_prefix_brake_hold":
                            self._probe_prefix_unsafe_count += 1
                        self._last_verification_result = "unsafe"
                        self._last_source = "candidate_rejected_vbc_unsafe"
                        event_to_publish = self._make_verification_event(
                            seq, "unsafe", False, verification_age, view,
                            diag=diag)
                    else:
                        if was_probe:
                            # The exact candidate that passed GCDF + VBC is the
                            # one we may execute.  Verification wall time is
                            # NOT execution time, so do not suffix it again.
                            committed = copy.deepcopy(candidate)
                        else:
                            committed = self._suffix_from_phase(
                                candidate, verification_age)
                        if committed is not None and committed.points:
                            if diag is None:
                                diag = {}
                            candidate_start_positions = (
                                list(candidate.points[0].positions)
                                if candidate.points[0].positions else [])
                            committed_start_positions = (
                                list(committed.points[0].positions)
                                if committed.points[0].positions else [])
                            raw_start_positions = list(
                                diag.get("raw_start_positions", []))
                            diag["precommit_suffix_phase_s"] = (
                                0.0 if was_probe else float(verification_age))
                            diag["precommit_suffix_start_shift_inf"] = (
                                0.0 if was_probe else self._position_shift_inf(
                                    committed_start_positions,
                                    candidate_start_positions))
                            diag["total_start_shift_inf"] = (
                                self._position_shift_inf(
                                    committed_start_positions,
                                    raw_start_positions))
                            commit_start_mismatch = (
                                self._measured_mismatch_inf_locked(
                                    committed.joint_names,
                                    committed_start_positions))
                            diag["commit_start_measured_mismatch_inf"] = (
                                commit_start_mismatch)
                            diag["commit_joint_state_age_ms"] = (
                                self._joint_state_age_ms_locked(now))
                            diag["committed_duration_s"] = float(
                                self._duration(committed))
                            self._last_probe_commit_start_mismatch_inf = (
                                commit_start_mismatch
                                if was_probe else
                                self._last_probe_commit_start_mismatch_inf)
                            self._last_verification_diag = copy.deepcopy(diag)

                            if (was_probe and
                                    (not math.isfinite(commit_start_mismatch) or
                                     commit_start_mismatch >
                                     self.probe_commit_start_mismatch_max)):
                                # Fail closed: the certified geometry no longer
                                # starts close enough to the robot.  Do not
                                # mutate it after certification; ask for a fresh
                                # measured-state plan instead.
                                self._probe_start_continuity_reject_count += 1
                                self._verification_timeout_count += 1
                                self._last_verification_result = "timeout"
                                self._last_source = (
                                    "probe_start_continuity_rejected_replan")
                                event_to_publish = self._make_verification_event(
                                    seq, "timeout", False, verification_age,
                                    view, safety_gate="start_continuity",
                                    diag=diag)
                                committed = None
                            else:
                                self._verification_safe_count += 1
                            if committed is not None:
                                if was_repair and view == "repair_prefix_brake_hold":
                                    self._repair_prefix_safe_count += 1
                                if was_probe and view == "probe_prefix_brake_hold":
                                    self._probe_prefix_safe_count += 1
                                self._last_verification_result = "safe"
                                committed.header.seq = seq
                                committed.header.stamp = now
                                execution_stamp_ns = int(
                                    committed.header.stamp.to_nsec())
                                self._last_execution_stamp_ns = execution_stamp_ns
                                self._committed_master = copy.deepcopy(committed)
                                self._committed_received = now
                                self._commit_count += 1
                                self._last_committed_age_s = 0.0
                                self._last_source = (
                                    "candidate_verified_safe_committed")
                                committed_to_publish = copy.deepcopy(committed)
                                event_to_publish = self._make_verification_event(
                                    seq, "safe", True, verification_age, view,
                                    execution_stamp_ns=execution_stamp_ns,
                                    diag=diag)
                        else:
                            self._verification_timeout_count += 1
                            self._last_verification_result = "expired_safe"
                            self._last_source = "candidate_safe_but_expired_before_commit"
                            event_to_publish = self._make_verification_event(
                                seq, "timeout", False, verification_age, view,
                                diag=diag)

                    dispatched, route = self._dispatch_pending_locked(now)
                    if route == "gcdf":
                        gcdf_to_publish = dispatched
                    elif route == "vbc":
                        verification_to_publish = dispatched
                    self._publish_summary_locked()

        if committed_to_publish is not None:
            self._publish_committed(
                committed_to_publish, "candidate_verified_safe_committed", 0.0)
        if event_to_publish is not None:
            self.verification_event_pub.publish(event_to_publish)
        if verification_to_publish is not None:
            self.verification_pub.publish(verification_to_publish)
        if gcdf_to_publish is not None:
            self.final_gcdf_query_pub.publish(gcdf_to_publish)

    def _timer_cb(self, _event):
        now = rospy.Time.now()
        continuation_to_publish = None
        execution_audit_to_publish = None
        execution_audit_age = math.nan
        timeout_event = None
        gcdf_timeout_event = None
        next_gcdf_to_publish = None
        next_verification_to_publish = None
        continuation_age = math.nan

        with self._lock:
            if (self._gcdf_outstanding is not None and
                    self._gcdf_outstanding_sent is not None):
                gcdf_age = (now - self._gcdf_outstanding_sent).to_sec()
                if gcdf_age > self.final_gcdf_timeout_s:
                    seq = int(self._gcdf_outstanding_seq)
                    view = str(self._gcdf_outstanding_view)
                    raw_received = self._gcdf_outstanding_raw_received
                    if raw_received is not None:
                        self._last_candidate_total_safety_pipeline_ms = (
                            1000.0 * max(0.0, (now - raw_received).to_sec()))
                    self._last_final_gcdf_roundtrip_ms = 1000.0 * gcdf_age
                    self._gcdf_outstanding = None
                    self._gcdf_outstanding_sent = None
                    self._gcdf_outstanding_seq = 0
                    self._gcdf_outstanding_repair = False
                    self._gcdf_outstanding_probe = False
                    self._gcdf_outstanding_view = "none"
                    self._gcdf_outstanding_raw_received = None
                    self._final_gcdf_timeout_count += 1
                    self._verification_timeout_count += 1
                    self._verification_outcome_count += 1
                    self._last_verification_seq = seq
                    self._last_verification_result = "timeout"
                    self._last_source = "final_gcdf_timeout_candidate_rejected"
                    self._last_verification_age_s = gcdf_age
                    self._last_verification_view = view
                    gcdf_timeout_event = self._make_verification_event(
                        seq, "timeout", False, gcdf_age, view,
                        safety_gate="gcdf")
                    dispatched, route = self._dispatch_pending_locked(now)
                    if route == "gcdf":
                        next_gcdf_to_publish = dispatched
                    elif route == "vbc":
                        next_verification_to_publish = dispatched

            if self._outstanding is not None and self._outstanding_sent is not None:
                age = (now - self._outstanding_sent).to_sec()
                if age > self.verification_timeout_s:
                    seq = int(self._outstanding_seq)
                    view = str(self._outstanding_view)
                    raw_received = self._outstanding_raw_received
                    self._last_exact_vbc_roundtrip_ms = 1000.0 * age
                    if raw_received is not None:
                        self._last_candidate_total_safety_pipeline_ms = (
                            1000.0 * max(0.0, (now - raw_received).to_sec()))
                    self._outstanding = None
                    self._outstanding_sent = None
                    self._outstanding_seq = 0
                    self._outstanding_dispatch_cycle = -1
                    self._outstanding_repair = False
                    self._outstanding_probe = False
                    self._outstanding_view = "none"
                    self._outstanding_raw_received = None
                    self._verification_timeout_count += 1
                    self._verification_outcome_count += 1
                    self._last_verification_seq = seq
                    self._last_verification_result = "timeout"
                    self._last_source = "verification_timeout_candidate_rejected"
                    self._last_verification_age_s = age
                    self._last_verification_view = view
                    timeout_event = self._make_verification_event(
                        seq, "timeout", False, age, view)

            if (self.execution_audit_enabled and
                    self._committed_master is not None and
                    self._committed_received is not None):
                execution_audit_age = (
                    now - self._committed_received).to_sec()
                master_duration = self._duration(self._committed_master)
                if (0.0 <= execution_audit_age <=
                        master_duration + self.execution_audit_post_hold_s):
                    execution_audit_to_publish = self._suffix_from_phase(
                        self._committed_master,
                        min(execution_audit_age, master_duration))

            if (self.continuation_enabled and
                    self._committed_master is not None and
                    self._committed_received is not None):
                continuation_age = (now - self._committed_received).to_sec()
                if (continuation_age >= self.continuation_start_delay_s and
                        continuation_age <= self.continuation_timeout_s):
                    continuation_to_publish = self._suffix_from_phase(
                        self._committed_master, continuation_age)
                elif continuation_age > self.continuation_timeout_s:
                    self._last_source = "committed_plan_stale_stop_publish"
                    self._last_committed_age_s = continuation_age
            self._publish_summary_locked()

        if execution_audit_to_publish is not None:
            self.execution_audit_pub.publish(execution_audit_to_publish)
            with self._lock:
                self._execution_audit_publish_count += 1
                self._publish_summary_locked()
        if continuation_to_publish is not None:
            self._publish_committed(
                continuation_to_publish, "committed_continuation", continuation_age)
        if timeout_event is not None:
            self.verification_event_pub.publish(timeout_event)
        if gcdf_timeout_event is not None:
            self.verification_event_pub.publish(gcdf_timeout_event)
        if next_gcdf_to_publish is not None:
            self.final_gcdf_query_pub.publish(next_gcdf_to_publish)
        if next_verification_to_publish is not None:
            self.verification_pub.publish(next_verification_to_publish)

    def _publish_summary_locked(self):
        msg = String()
        msg.data = " ".join([
            "source={}".format(self._last_source),
            "raw_input_count={}".format(self._raw_input_count),
            "pending_candidate={}".format(int(self._pending_raw is not None)),
            "pending_replace_count={}".format(self._pending_replace_count),
            "selector_cycle_count={}".format(self._selector_cycle_count),
            "repair_active={}".format(int(self._repair_active)),
            "probe_active={}".format(int(self._probe_active)),
            "repair_prefix_enabled={}".format(int(self.repair_prefix_verification_enabled)),
            "final_gcdf_enabled={}".format(int(self.final_gcdf_enabled)),
            "final_gcdf_outstanding={}".format(int(self._gcdf_outstanding is not None)),
            "final_gcdf_outstanding_seq={}".format(int(self._gcdf_outstanding_seq)),
            "final_gcdf_query_count={}".format(self._final_gcdf_query_count),
            "final_gcdf_safe_count={}".format(self._final_gcdf_safe_count),
            "final_gcdf_unsafe_count={}".format(self._final_gcdf_unsafe_count),
            "final_gcdf_timeout_count={}".format(self._final_gcdf_timeout_count),
            "final_gcdf_stamp_miss_count={}".format(self._final_gcdf_stamp_miss_count),
            "final_gcdf_recovery_event_count={}".format(
                self._final_gcdf_recovery_event_count),
            "final_gcdf_recovery_event_drop_count={}".format(
                self._final_gcdf_recovery_event_drop_count),
            "last_final_gcdf_recovery_seq={}".format(
                self._last_final_gcdf_recovery_seq),
            "last_final_gcdf_recovery_timestep={}".format(
                self._last_final_gcdf_recovery_timestep),
            "last_final_gcdf_recovery_sweep_s={}".format(
                "nan" if not math.isfinite(self._last_final_gcdf_recovery_sweep_s)
                else "{:.6f}".format(self._last_final_gcdf_recovery_sweep_s)),
            "last_final_gcdf_recovery_point_count={}".format(
                self._last_final_gcdf_recovery_point_count),
            "final_gcdf_min_d={}".format(
                "nan" if not math.isfinite(self._last_final_gcdf_min_d)
                else "{:.6f}".format(self._last_final_gcdf_min_d)),
            "verification_outstanding={}".format(int(self._outstanding is not None)),
            "outstanding_seq={}".format(int(self._outstanding_seq)),
            "outstanding_view={}".format(self._outstanding_view),
            "verification_publish_count={}".format(self._verification_publish_count),
            "verification_outcome_count={}".format(self._verification_outcome_count),
            "verification_safe_count={}".format(self._verification_safe_count),
            "verification_unsafe_count={}".format(self._verification_unsafe_count),
            "verification_timeout_count={}".format(self._verification_timeout_count),
            "last_verification_seq={}".format(self._last_verification_seq),
            "last_verification_result={}".format(self._last_verification_result),
            "last_verification_view={}".format(self._last_verification_view),
            "last_execution_stamp_ns={}".format(self._last_execution_stamp_ns),
            "repair_prefix_build_count={}".format(self._repair_prefix_build_count),
            "repair_prefix_safe_count={}".format(self._repair_prefix_safe_count),
            "repair_prefix_unsafe_count={}".format(self._repair_prefix_unsafe_count),
            "probe_prefix_build_count={}".format(self._probe_prefix_build_count),
            "probe_prefix_safe_count={}".format(self._probe_prefix_safe_count),
            "probe_prefix_unsafe_count={}".format(self._probe_prefix_unsafe_count),
            "probe_rebase_count={}".format(self._probe_rebase_count),
            "probe_rebase_clamp_count={}".format(
                self._probe_rebase_clamp_count),
            "probe_start_continuity_reject_count={}".format(
                self._probe_start_continuity_reject_count),
            "probe_rebase_shift_inf={}".format(
                "nan" if not math.isfinite(self._last_probe_rebase_shift_inf)
                else "{:.6f}".format(self._last_probe_rebase_shift_inf)),
            "probe_rebase_target_shift_inf={}".format(
                "nan" if not math.isfinite(
                    self._last_probe_rebase_target_shift_inf)
                else "{:.6f}".format(
                    self._last_probe_rebase_target_shift_inf)),
            "probe_rebase_residual_inf={}".format(
                "nan" if not math.isfinite(
                    self._last_probe_rebase_residual_inf)
                else "{:.6f}".format(self._last_probe_rebase_residual_inf)),
            "probe_commit_start_mismatch_inf={}".format(
                "nan" if not math.isfinite(
                    self._last_probe_commit_start_mismatch_inf)
                else "{:.6f}".format(
                    self._last_probe_commit_start_mismatch_inf)),
            "probe_speed_scale_count={}".format(
                self._probe_speed_scale_count),
            "probe_speed_scale_clamp_count={}".format(
                self._probe_speed_scale_clamp_count),
            "probe_speed_time_scale={:.6f}".format(
                self._last_probe_speed_scale),
            "probe_speed_max_before={}".format(
                "nan" if not math.isfinite(
                    self._last_probe_speed_max_before)
                else "{:.6f}".format(
                    self._last_probe_speed_max_before)),
            "probe_speed_max_after={}".format(
                "nan" if not math.isfinite(
                    self._last_probe_speed_max_after)
                else "{:.6f}".format(
                    self._last_probe_speed_max_after)),
            "probe_effective_prefix_s={}".format(
                "nan" if not math.isfinite(
                    self._last_probe_effective_prefix_s)
                else "{:.6f}".format(
                    self._last_probe_effective_prefix_s)),
            "repair_prefix_duration_s={}".format(
                "nan" if not math.isfinite(self._last_repair_prefix_duration_s)
                else "{:.6f}".format(self._last_repair_prefix_duration_s)),
            "repair_brake_duration_s={}".format(
                "nan" if not math.isfinite(self._last_repair_brake_duration_s)
                else "{:.6f}".format(self._last_repair_brake_duration_s)),
            "commit_count={}".format(self._commit_count),
            "continuation_enabled={}".format(int(self.continuation_enabled)),
            "execution_audit_enabled={}".format(int(self.execution_audit_enabled)),
            "execution_audit_publish_count={}".format(
                self._execution_audit_publish_count),
            "has_committed_plan={}".format(int(self._committed_master is not None)),
            "committed_publish_count={}".format(self._committed_publish_count),
            "continuation_count={}".format(self._continuation_count),
            "raw_to_safety_dispatch_ms={}".format(
                "nan" if not math.isfinite(self._last_raw_to_safety_dispatch_ms)
                else "{:.3f}".format(self._last_raw_to_safety_dispatch_ms)),
            "final_gcdf_roundtrip_ms={}".format(
                "nan" if not math.isfinite(self._last_final_gcdf_roundtrip_ms)
                else "{:.3f}".format(self._last_final_gcdf_roundtrip_ms)),
            "exact_vbc_roundtrip_ms={}".format(
                "nan" if not math.isfinite(self._last_exact_vbc_roundtrip_ms)
                else "{:.3f}".format(self._last_exact_vbc_roundtrip_ms)),
            "candidate_total_safety_pipeline_ms={}".format(
                "nan" if not math.isfinite(
                    self._last_candidate_total_safety_pipeline_ms)
                else "{:.3f}".format(
                    self._last_candidate_total_safety_pipeline_ms)),
            "verification_age_s={}".format(
                "nan" if not math.isfinite(self._last_verification_age_s)
                else "{:.6f}".format(self._last_verification_age_s)),
            "committed_age_s={}".format(
                "nan" if not math.isfinite(self._last_committed_age_s)
                else "{:.6f}".format(self._last_committed_age_s)),
            "last_input_gap_s={}".format(
                "nan" if not math.isfinite(self._last_input_gap_s)
                else "{:.6f}".format(self._last_input_gap_s)),
            "max_input_gap_s={:.6f}".format(self._max_input_gap_s),
        ])
        self.summary_pub.publish(msg)

    def _publish_summary(self):
        with self._lock:
            self._publish_summary_locked()


def main():
    rospy.init_node("optimized_trajectory_continuity")
    OptimizedTrajectoryContinuityNode()
    rospy.spin()


if __name__ == "__main__":
    main()
