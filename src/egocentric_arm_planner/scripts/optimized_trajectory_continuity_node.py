#!/usr/bin/env python3
"""Serialize candidate VBC verification and publish only committed trajectories.

C4.4 verification protocol
--------------------------
The C++ VBC selector may publish periodic refreshes for a cached trajectory and,
in C5.34, may also evaluate a freshly-arrived predicted trajectory immediately.
Verification is synchronized by exact trajectory header.stamp transaction
identity; selector-cycle ordering remains a secondary stale-cycle guard.

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
import json
import math
import re
import threading
import time
import uuid
from collections import OrderedDict, deque
from pathlib import Path
import sys

import rospy
from care_collision_cdf.msg import CollisionCDFConstraintBatch, CollisionCDFRejectionFeedback
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import PoseStamped
from egocentric_arm_planner.msg import CandidateTransfer, RepairPrefixSelection
from care_velocity_controller.msg import CommandStatus
# catkin devel relay modules execute in a private context and do not export
# helpers to importers. __file__ here names this actual source (or install) file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gcdf_rejection_feedback import rejection_points
from occupied_clearance import occupied_violations
from candidate_admission import payload_sha256, WallWorker, TaskContextReceiver
from task_recovery_events import (TaskEventSource, TaskEventReceiver, VERIFICATION_TOPIC,
                                 CONTROL_TOPIC, RECOVERY_TOPIC, canonical, validate_regime_control,
                                 validate_recovery_snapshot, regime_identity)
from initial_shell_recovery import context as shell_context, commit_check as shell_commit_check
from recovery_motion import ramp_step, interpolate_motion, has_nodal_motion
from handoff_schedule import terminal_hold_start, state_variation_ready, predict_held_motion
from repair_prefix_contract import parse as parse_prefix_contract, make as make_prefix_contract, construction as prefix_construction
from geometric_restore import permits_unknown_conflicts
from execution_motion_limits import effective_limits, violation as motion_limit_violation
from execution_preparation import ExecutionPreparationLedger


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
        self._replacement_fence = None
        self._replacement_closed_id = 0
        self._replacement_raw_floor_ns = 0
        self.input_topic = str(rospy.get_param(
            "~input_topic", "/care_planner/mpc/predicted_trajectory"))
        self.admission_enabled = bool(rospy.get_param("~candidate_admission_enabled", False))
        self.prefix_binding_enabled = bool(rospy.get_param("~repair_prefix_binding_enabled", False))
        self._prefix_revision = 0
        self._selected_prefix = None
        self._prefix_context = None
        self._admission_boot_id = uuid.uuid4().hex
        self._admission_session = None
        self._admission_high_sequence = 0
        self._admission_entries = OrderedDict()
        self._admission_events = deque(maxlen=128)
        self._admission_queries = deque()
        self._admission_summary_pending = None
        self._pending_transfer_key = None
        self._admission_goal_identity = None
        self.task_context_enabled = bool(rospy.get_param("~enable_observation_context", False))
        if self.task_context_enabled and not self.admission_enabled:
            raise ValueError("Task context requires candidate admission")
        self.task_context = TaskContextReceiver()
        self._task_commit_bindings = OrderedDict()
        self._task_commit_regimes = {}
        self._task_verification_events = OrderedDict()
        self._task_event_source = TaskEventSource()
        self.require_bound_regime = bool(rospy.get_param('~require_bound_regime', False))
        if self.require_bound_regime and not self.task_context_enabled:
            raise ValueError('Bound regimes require task contexts')
        self.require_execution_preparation = bool(rospy.get_param('~require_execution_preparation', self.require_bound_regime))
        if self.require_bound_regime and not self.require_execution_preparation:
            raise ValueError('Task-bound automatic recovery requires qualified execution preparation')
        if self.require_execution_preparation and not self.require_bound_regime:
            raise ValueError('Execution preparation requires bound task and regime identities')
        self._execution_preparations = ExecutionPreparationLedger()
        self._execution_prepared_to_publish = None
        self._execution_prepared_count = 0
        self._task_control_receiver = TaskEventReceiver()
        self._task_control_dispatch = threading.local()
        self._task_control_until_ns = 0
        self._task_control_event = None
        self._task_regime_identity = ''
        self._admission_receive_count = 0
        self._admission_duplicate_count = 0
        self._admission_expiry_count = 0
        self._admission_audit_stamp_ns = 0
        # Rejection identities belong to the planner that produced the raw
        # candidate.  With the C5.9 single-flight gate enabled, ``input_topic``
        # is the gate output while the local planner still owns the raw input
        # topic.  Keep the feedback channel explicit so the gate cannot hide
        # confirmed GCDF witnesses from the planner's repair state.
        self.gcdf_rejection_feedback_topic = str(rospy.get_param(
            "~gcdf_rejection_feedback_topic",
            self.input_topic + "/gcdf_rejection_feedback"))
        self.gcdf_rejection_feedback_pub = rospy.Publisher(
            self.gcdf_rejection_feedback_topic, CollisionCDFRejectionFeedback,
            queue_size=4)
        self.verification_topic = str(rospy.get_param(
            "~output_topic", "/care_planner/optimized_trajectory"))
        self.initial_shell_recovery_enabled = bool(rospy.get_param("~initial_shell_recovery_enabled", False))
        self.initial_shell_execution_margin_m = float(rospy.get_param("~initial_shell_execution_margin_m", 0.))
        self.shell_context_pub = rospy.Publisher(self.verification_topic + '/initial_shell_context', String, queue_size=4)
        self.shell_quality_pub = rospy.Publisher(self.verification_topic + '/initial_shell_quality', String, queue_size=4)
        self.committed_topic = str(rospy.get_param(
            "~committed_output_topic", "/care_planner/committed_trajectory"))
        self.execution_audit_topic = str(rospy.get_param(
            "~execution_audit_output_topic",
            "/care_planner/execution/audit_trajectory"))
        self.execution_audit_identity_topic = str(rospy.get_param(
            "~execution_audit_identity_topic",
            "/care_planner/execution/audit_identity"))
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
        self.blocker_stack_summary_topic = str(rospy.get_param(
            "~blocker_stack_summary_topic",
            "/care_planner/active_sensing/blocker_stack_summary"))
        self.probe_start_rebase_enabled = bool(rospy.get_param(
            "~probe_start_rebase_enabled", True))
        self.probe_rebase_max_shift_inf = float(rospy.get_param(
            "~probe_rebase_max_shift_inf", 0.10))
        self.probe_start_joint_state_max_age_s = float(rospy.get_param(
            "~probe_start_joint_state_max_age_s", 0.05))
        self.probe_commit_start_mismatch_max = float(rospy.get_param(
            "~probe_commit_start_mismatch_max", 0.03))
        self.normal_commit_start_mismatch_max = float(rospy.get_param(
            "~normal_commit_start_mismatch_max", 0.03))
        self.repair_commit_start_mismatch_max = float(rospy.get_param(
            "~repair_commit_start_mismatch_max", 0.03))
        self.smooth_handoff_rebase_enabled = bool(rospy.get_param(
            "~smooth_handoff_rebase_enabled", False))
        self.smooth_handoff_rebase_max_shift_inf = float(rospy.get_param(
            "~smooth_handoff_rebase_max_shift_inf", 0.10))
        self.smooth_handoff_joint_state_max_age_s = float(rospy.get_param(
            "~smooth_handoff_joint_state_max_age_s", 0.05))
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
        self.final_occupied_resolution_m = float(rospy.get_param(
            "~final_occupied_resolution_m", 0.05))
        self.final_occupied_hard_margin_m = float(rospy.get_param(
            "~final_occupied_hard_margin_m", 0.0))
        if (not math.isfinite(self.final_occupied_resolution_m) or
                self.final_occupied_resolution_m <= 0 or
                not math.isfinite(self.final_occupied_hard_margin_m)):
            raise ValueError('Invalid final occupied geometry contract')
        self.final_gcdf_stamp_tolerance_s = float(rospy.get_param(
            "~final_gcdf_stamp_tolerance_s", 1e-6))

        self.repair_prefix_verification_enabled = bool(rospy.get_param(
            "~repair_prefix_verification_enabled", False))
        self.repair_projection_consistent_dispatch_enabled = bool(rospy.get_param(
            "~repair_projection_consistent_dispatch_enabled", False))
        self.repair_execution_prefix_s = float(rospy.get_param(
            "~repair_execution_prefix_s", 0.15))
        self.probe_execution_prefix_s = float(rospy.get_param(
            "~probe_execution_prefix_s", self.repair_execution_prefix_s))
        self.repair_brake_dt_s = float(rospy.get_param(
            "~repair_brake_dt_s", 0.05))
        self.repair_hold_s = float(rospy.get_param(
            "~repair_hold_s", 0.10))
        self.repair_brake_max_steps = int(rospy.get_param(
            "~repair_brake_max_steps", 20))
        self.repair_acceleration_limits = [float(v) for v in rospy.get_param(
            "~repair_acceleration_limits", [3.0, 3.0, 4.0, 4.0, 6.0, 6.0, 6.0])]

        # Cycle-aware visibility frontier recovery. This never bypasses either
        # safety gate: it only shortens the exact executable REPAIR prefix that
        # will still pass final GCDF and exact VBC before commit.
        self.cycle_recovery_enabled = bool(rospy.get_param(
            "~cycle_recovery_enabled", False))
        raw_cycle_prefixes = rospy.get_param(
            "~cycle_recovery_prefixes_s", [0.15, 0.10, 0.05])
        if not isinstance(raw_cycle_prefixes, (list, tuple)):
            raise ValueError("~cycle_recovery_prefixes_s must be a list")
        fallback_prefixes = [float(v) for v in raw_cycle_prefixes]
        ladder = [float(self.repair_execution_prefix_s)] + [
            v for v in fallback_prefixes
            if v <= self.repair_execution_prefix_s + 1e-12]
        self.cycle_recovery_prefixes_s = []
        for value in sorted(ladder, reverse=True):
            if value <= 0.0 or not math.isfinite(value):
                raise ValueError(
                    "~cycle_recovery_prefixes_s must contain positive finite values")
            if not self.cycle_recovery_prefixes_s or abs(
                    value - self.cycle_recovery_prefixes_s[-1]) > 1e-9:
                self.cycle_recovery_prefixes_s.append(value)

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
        if self.normal_commit_start_mismatch_max <= 0.0:
            raise ValueError(
                "~normal_commit_start_mismatch_max must be positive")
        if self.repair_commit_start_mismatch_max <= 0.0:
            raise ValueError(
                "~repair_commit_start_mismatch_max must be positive")
        if self.smooth_handoff_rebase_max_shift_inf <= 0.0:
            raise ValueError(
                "~smooth_handoff_rebase_max_shift_inf must be positive")
        if self.smooth_handoff_joint_state_max_age_s <= 0.0:
            raise ValueError(
                "~smooth_handoff_joint_state_max_age_s must be positive")
        if self.probe_execution_prefix_s <= 0.0:
            raise ValueError("~probe_execution_prefix_s must be positive")
        if self.probe_velocity_cap <= 0.0:
            raise ValueError("~probe_velocity_cap must be positive")
        if self.probe_time_scale_max < 1.0:
            raise ValueError("~probe_time_scale_max must be >= 1")
        if not self.cycle_recovery_prefixes_s:
            raise ValueError("cycle recovery prefix ladder is empty")

        self._repair_active = False
        if self.initial_shell_recovery_enabled and (not self.final_gcdf_enabled or
                not self.repair_prefix_verification_enabled or
                not math.isfinite(self.initial_shell_execution_margin_m) or self.initial_shell_execution_margin_m<0.):
            raise ValueError('Initial-shell recovery requires executable GCDF and prefix/brake certification')
        self._probe_active = False
        self._pending_raw = None
        self._pending_raw_received = None
        self._pending_repair = False
        self._pending_probe = False
        self._last_raw_received = None
        self._preflight_wait_key = None
        self._preflight_wait_owner = None
        self._preflight_owner_token = None
        self._preflight_wait_count = self._preflight_resume_count = 0
        self._preflight_reason = 'none'
        self._preflight_measured_age_ns = self._preflight_actuator_age_ns = 0
        self._handoff_previous_measured = None
        self._handoff_tracker_fields = {}
        self._handoff_completed_token = 0
        self._handoff_certificate = None
        self._handoff_wait_count = self._handoff_resume_count = 0
        self._handoff_wait_key = None
        self._last_input_gap_s = math.nan
        self._max_input_gap_s = 0.0
        self._latest_measured_q_by_name = {}
        self._latest_measured_velocity_by_name = {}
        self._latest_measured_stamp_ns = 0
        self.repair_motion_consistency_enabled = bool(rospy.get_param('~repair_motion_consistency_enabled', True))
        self.handoff_prediction_enabled = bool(rospy.get_param('~repair_handoff_prediction_enabled', True))
        self._latest_actuator_stamp_ns = 0
        self._latest_actuator_velocity = {}
        self._controller_session = 0
        self._retired_controller_sessions = set()
        self._controller_joint_names = list(rospy.get_param('~controller_joint_names',
            ['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']))
        if len(self._controller_joint_names) != 7 or len(set(self._controller_joint_names)) != 7:
            raise ValueError('The controller must report seven uniquely ordered joints')
        self.tracking_feedback_gain = float(rospy.get_param('~execution/position_feedback_gain',
            rospy.get_param('/trajectory_execution_manager_node/execution/position_feedback_gain',1.)))
        self.recovery_position_limits = dict(rospy.get_param('~mpc/joint_position_limits',
            rospy.get_param('/trajectory_execution_manager_node/mpc/joint_position_limits',self.joint_position_limits)))
        self.recovery_velocity_limits = dict(rospy.get_param('~mpc/joint_velocity_limits',
            rospy.get_param('/trajectory_execution_manager_node/mpc/joint_velocity_limits',{})))
        self.enforce_execution_motion_limits = bool(rospy.get_param('~execution/enforce_motion_limits', False))
        self.execution_velocity_cap = float(rospy.get_param('~execution/max_command_velocity', 2.))
        if self.enforce_execution_motion_limits:
            effective_limits(list(rospy.get_param('~joint_names', [])),
                             self.recovery_velocity_limits, self.execution_velocity_cap)
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
        self._last_final_gcdf_blocker_class = "none"
        self._last_final_gcdf_unsafe_unknown_count = 0
        self._last_final_gcdf_unsafe_occupied_count = 0
        self._verification_publish_count = 0
        self._verification_safe_count = 0
        self._verification_unsafe_count = 0
        self._verification_timeout_count = 0
        self._verification_protocol_invalid_count = 0
        self._last_vbc_protocol_reason = 'none'
        self._exact_vbc_stamp_miss_count = 0
        self._verification_outcome_count = 0
        self._commit_count = 0
        self._committed_publish_count = 0
        self._continuation_count = 0
        self._execution_audit_publish_count = 0
        self._pending_replace_count = 0
        # C5.36 safety admission diagnostics.
        self._raw_immediate_dispatch_count = 0
        self._raw_busy_buffer_count = 0
        self._post_cert_pending_dispatch_count = 0
        self._selector_fallback_dispatch_count = 0
        self._repair_prefix_build_count = 0
        self._repair_prefix_safe_count = 0
        self._repair_prefix_unsafe_count = 0
        self._probe_prefix_build_count = 0
        self._probe_prefix_safe_count = 0
        self._probe_prefix_unsafe_count = 0
        self._last_repair_prefix_duration_s = math.nan
        self._last_repair_brake_duration_s = math.nan
        self._last_source = "waiting_raw_candidate"
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

        self._cycle_recovery_active = False
        self._cycle_recovery_signature = "none"
        self._cycle_recovery_stack = "none"
        self._cycle_recovery_earliest_layer_ids = "none"
        self._cycle_recovery_cycle_count = 0
        self._cycle_recovery_prefix_index = 0
        self._cycle_recovery_activation_count = 0
        self._cycle_recovery_clear_count = 0
        self._cycle_recovery_backoff_count = 0
        self._cycle_recovery_safe_commit_count = 0
        self._cycle_recovery_floor_unsafe_count = 0
        self._cycle_recovery_last_reason = "startup"
        self._last_cycle_recovery_prefix_s = self.repair_execution_prefix_s

        self._gcdf_outstanding_diag = None
        self._outstanding_diag = None
        self._last_verification_diag = None
        self._last_final_gcdf_roundtrip_ms = math.nan
        self._last_exact_vbc_roundtrip_ms = math.nan
        self._last_candidate_total_safety_pipeline_ms = math.nan

        if self.prefix_binding_enabled:
            if not (self.admission_enabled and self.initial_shell_recovery_enabled and
                    self.repair_projection_consistent_dispatch_enabled):
                raise ValueError("Prefix binding requires admission, shell context and projection-consistent dispatch")
            self.prefix_selection_pub = rospy.Publisher('/care_planner/repair_prefix_selection',
                RepairPrefixSelection, queue_size=1, latch=True)
            self._sync_repair_prefix_locked()
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
        self.task_stop_pub = (rospy.Publisher("/care_planner/execution/vbc_stop_request", String, queue_size=16)
                              if self.task_context_enabled else None)
        self.committed_pub = rospy.Publisher(
            self.committed_topic, JointTrajectory, queue_size=1)
        self.execution_audit_pub = rospy.Publisher(
            self.execution_audit_topic, JointTrajectory, queue_size=1)
        self.execution_audit_identity_pub = rospy.Publisher(
            self.execution_audit_identity_topic, String, queue_size=20)
        self.summary_pub = rospy.Publisher(
            self.summary_topic, String, queue_size=1, latch=True)
        self.verification_event_pub = rospy.Publisher(
            self.verification_event_topic, String, queue_size=20, latch=False)
        self.task_verification_pub = (rospy.Publisher(VERIFICATION_TOPIC, String, queue_size=32)
                                      if self.task_context_enabled else None)
        self.execution_prepared_pub = (rospy.Publisher('/care_planner/execution/prepared_trajectory',
            CandidateTransfer, queue_size=1) if self.require_execution_preparation else None)
        self.execution_prepared_status_pub = (rospy.Publisher('/care_planner/execution/preparation_status',
            String, queue_size=1, latch=True) if self.require_execution_preparation else None)
        if self.require_bound_regime:
            rospy.Subscriber(CONTROL_TOPIC, String, self._task_control_cb, queue_size=8)
            rospy.Subscriber(RECOVERY_TOPIC, String, self._task_recovery_cb, queue_size=8)
        self.replacement_ack_pub = rospy.Publisher(
            '/care_planner/local_planner/candidate_replacement_commit_ack', String, queue_size=8)
        self.replacement_grant_sub = rospy.Subscriber(
            '/care_planner/local_planner/candidate_replacement_grant', String,
            self._replacement_grant_cb, queue_size=8)
        self.replacement_finish_sub = rospy.Subscriber(
            '/care_planner/local_planner/candidate_replacement_request', String,
            self._replacement_finish_cb, queue_size=8)

        verification_tcp_nodelay = bool(rospy.get_param("~tcp_nodelay", False))
        if self.admission_enabled:
            self.admission_status_pub = rospy.Publisher(
                self.input_topic + "/admission_status", String, queue_size=20)
            self.raw_sub = rospy.Subscriber(
                self.input_topic + "/transfer", CandidateTransfer, self._transfer_cb,
                queue_size=10, tcp_nodelay=verification_tcp_nodelay)
            if self.task_context_enabled:
                rospy.Subscriber("/care_planner/active_sensing/observation_context", String,
                                 self._task_context_cb, queue_size=32)
            else:
                rospy.Subscriber("/care_planner/ee_target_pose", PoseStamped,
                                 self._admission_goal_cb, queue_size=1)
            rospy.Subscriber('/care_planner/active_sensing/visibility_acquisition_summary',
                             String, self._preflight_owner_cb, queue_size=1)
        else:
            self.raw_sub = rospy.Subscriber(
                self.input_topic, JointTrajectory, self._raw_cb, queue_size=1,
                tcp_nodelay=verification_tcp_nodelay)
        self.summary_sub = rospy.Subscriber(
            self.global_summary_topic, String, self._global_summary_cb, queue_size=1,
            tcp_nodelay=verification_tcp_nodelay)
        if self.initial_shell_recovery_enabled:
            self.verification_protocol_sub = rospy.Subscriber(
                self.verification_topic + '/protocol_status', String,
                self._verification_protocol_cb, queue_size=8,
                tcp_nodelay=verification_tcp_nodelay)
        self.final_gcdf_batch_sub = rospy.Subscriber(
            self.final_gcdf_batch_topic, CollisionCDFConstraintBatch,
            self._final_gcdf_batch_cb, queue_size=2, tcp_nodelay=verification_tcp_nodelay)
        self.repair_sub = rospy.Subscriber(
            self.repair_active_topic, Bool, self._repair_active_cb, queue_size=1)
        self.probe_sub = rospy.Subscriber(
            self.probe_active_topic, Bool, self._probe_active_cb, queue_size=1)
        self.joint_state_sub = rospy.Subscriber(
            self.joint_state_topic, JointState, self._joint_state_cb, queue_size=1)
        self.actuator_state_sub = rospy.Subscriber(
            rospy.get_param('~actuator_state_topic', '/care_arm/arm_group_velocity_controller/command_stamped'),
            JointState, self._actuator_state_cb, queue_size=1, tcp_nodelay=True)
        if self.require_bound_regime:
            rospy.Subscriber(rospy.get_param('~controller_state_topic',
                '/care_arm/arm_group_velocity_controller/state'), CommandStatus,
                self._controller_state_cb, queue_size=8, tcp_nodelay=True)
        self._execution_tracker_state = None
        self._execution_tracker_received = None
        self.execution_tracker_sub = rospy.Subscriber(
            '/care_planner/execution/tracker_summary', String,
            self._execution_tracker_cb, queue_size=1)
        self.blocker_stack_sub = rospy.Subscriber(
            self.blocker_stack_summary_topic, String,
            self._blocker_stack_summary_cb, queue_size=1)
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.rate), self._timer_cb)

        self._publish_summary()
        if self.admission_enabled:
            self.admission_wall_worker = WallWorker(self._admission_tick,
                lambda exc: rospy.logerr("[candidate_admission receiver] %s", exc))
            rospy.on_shutdown(self.admission_wall_worker.stop.set)
        rospy.logwarn(
            "[optimized_trajectory_continuity] SELECTOR-CYCLE VERIFY raw=%s "
            "final_gcdf=%s verify=%s committed=%s audit=%s continuation=%d "
            "audit_enabled=%d repair_prefix=%d prefix=%.3fs brake_dt=%.3fs "
            "hold=%.3fs cycle_recovery=%d ladder=%s timeout=%.3fs",
            self.input_topic, self.final_gcdf_query_topic,
            self.verification_topic, self.committed_topic,
            self.execution_audit_topic,
            int(self.continuation_enabled),
            int(self.execution_audit_enabled),
            int(self.repair_prefix_verification_enabled),
            self.repair_execution_prefix_s, self.repair_brake_dt_s,
            self.repair_hold_s, int(self.cycle_recovery_enabled),
            ":".join("{:.3f}".format(v) for v in self.cycle_recovery_prefixes_s),
            self.verification_timeout_s)

    def _admission_status_locked(self, entry, stage, tail=""):
        if stage in ("gcdf", "vbc"):
            entry["audit_started"] = True
        entry["stage"] = stage
        entry["status"] = String(data=(
            "session={} sequence={} receiver_boot_id={} payload_sha256={} "
            "raw_candidate_stamp_ns={} admission_stage={} {}".format(
                entry["session"], entry["sequence"], self._admission_boot_id,
                entry["digest"], entry["raw_stamp"], stage, tail)))
        self._admission_events.append(copy.deepcopy(entry["status"]))

    def _admission_audit_stamp_locked(self, now):
        # Old replies cannot decide a different candidate after cancellation,
        # even when ROS time pauses or repeats a timestamp.
        self._admission_audit_stamp_ns = max(now.to_nsec(), self._admission_audit_stamp_ns + 1)
        return rospy.Time(0, self._admission_audit_stamp_ns)

    def _admission_cancel_locked(self, key, stage="cancelled", reason=None):
        entry = self._admission_entries[key]
        if entry["stage"] in ("terminal", "cancelled", "expired", "rejected"):
            self._admission_events.append(copy.deepcopy(entry["status"]))
            return
        if self._pending_transfer_key == key:
            self._pending_raw = self._pending_raw_received = None
            self._pending_transfer_key = None
        if self._preflight_wait_key == key:
            self._preflight_wait_key = self._preflight_wait_owner = None
        if self._handoff_wait_key == key:
            rospy.loginfo('[HANDOFF_SCHEDULE] event=closed raw=%s stage=%s', entry['raw_stamp'], stage)
            self._handoff_wait_key = None
        for prefix in ("_gcdf_outstanding", "_outstanding"):
            diag = getattr(self, prefix + "_diag") or {}
            if diag.get("transfer_key") == key:
                setattr(self, prefix, None)
                setattr(self, prefix + "_sent", None)
                setattr(self, prefix + "_diag", None)
        view = entry["view"]
        reason = reason or ("deadline_expired" if stage == "expired" else
                            "context_changed" if stage == "cancelled" else "admission_rejected")
        result = "timeout" if stage == "expired" else stage
        self._admission_status_locked(entry, stage,
            "result={} committed=0 verification_view={} execution_stamp_ns=0 reason={} audit_started={}".format(
                result, view, reason, int(entry.get("audit_started", False))))

    def _admission_expire_locked(self):
        if not self.admission_enabled:
            return
        wall = time.monotonic()
        regime_expired = getattr(self, 'require_bound_regime', False) and not self._task_regime_current_locked()
        if self.require_execution_preparation:
            try:
                self._execution_preparations.snapshot(rospy.Time.now().to_nsec(),
                    (self.task_context.snapshot(rospy.Time.now().to_nsec()) or {}).get('candidate_binding',''),
                    self._task_regime_identity if not regime_expired else '')
            except ValueError:
                self._execution_preparations.cancel('preparation_clock_fault')
        if (self.task_context_enabled and self.task_context.snapshot(rospy.Time.now().to_nsec()) is None) or regime_expired:
            self._task_stop_committed_locked()
        for key, entry in list(self._admission_entries.items()):
            if (entry["stage"] not in ("terminal", "cancelled", "expired", "rejected") and
                    (regime_expired or wall >= entry["deadline"] or (self.task_context_enabled and not self.task_context.accepts(
                        entry.get("task_context", ""), rospy.Time.now().to_nsec())))):
                self._admission_expiry_count += 1
                self._admission_cancel_locked(key, "expired")

    def _admission_invalidate_locked(self):
        if self.admission_enabled:
            for key in list(self._admission_entries):
                if self._admission_entries[key]["stage"] not in (
                        "terminal", "cancelled", "expired", "rejected"):
                    self._admission_cancel_locked(key)

    def _task_stop_committed_locked(self):
        if self._committed_master is not None:
            token = self._committed_master.header.stamp.to_nsec()
            if token and self.task_stop_pub is not None:
                self.task_stop_pub.publish(String(data='execution_stamp_ns='+str(token)))
        self._committed_master = self._committed_received = None

    def _task_context_cb(self, msg):
        try:
            with self._lock:
                before = self.task_context.snapshot(rospy.Time.now().to_nsec())
                after = self.task_context.update(json.loads(msg.data), rospy.Time.now().to_nsec())
                if (before or {}).get('candidate_binding') != (after or {}).get('candidate_binding') or after is None:
                    self._execution_preparations.cancel('task_context_changed')
                    self._execution_prepared_to_publish = None
                    self._admission_invalidate_locked()
                    self._task_stop_committed_locked()
            self._flush_admission()
        except (ValueError, KeyError, TypeError) as exc:
            rospy.logwarn('[TaskContext continuity] %s', exc)

    def _admission_goal_cb(self, msg):
        p, q = msg.pose.position, msg.pose.orientation
        identity = (msg.header.frame_id, p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        with self._lock:
            if identity != self._admission_goal_identity:
                self._admission_goal_identity = identity
                self._admission_invalidate_locked()
        self._flush_admission()

    def _preflight_owner_cb(self, msg):
        if getattr(self, 'require_bound_regime', False) and not getattr(self._task_control_dispatch, 'active', False):
            return
        token = _tokens(msg.data).get('active_q_vis_token')
        if token != 'none' and (not token or not token.startswith('care_obs_v1_')):
            return
        with self._lock:
            self._preflight_owner_token = token
            if (self._preflight_wait_key is not None and
                    self._preflight_wait_owner is not None and
                    token != self._preflight_wait_owner):
                self._admission_cancel_locked(self._preflight_wait_key,
                                              reason='preflight_owner_changed')
        self._flush_admission()

    def _resume_preflight(self):
        if not self.admission_enabled:
            return
        with self._lock:
            if self._preflight_wait_key is None:
                return
            candidate, route = self._dispatch_pending_locked(rospy.Time.now())
            if candidate is not None:
                self._admission_queries.append((route, candidate))
        self._flush_admission()

    def _defer_preflight_locked(self, raw, received, key, repair, probe, reason):
        entry = self._admission_entries.get(key)
        if (not self.admission_enabled or entry is None or
                entry['stage'] in ('terminal', 'cancelled', 'expired', 'rejected') or
                time.monotonic() >= entry['deadline']):
            return False
        self._pending_raw, self._pending_raw_received = raw, received
        self._pending_transfer_key = key
        self._pending_repair, self._pending_probe = repair, probe
        if self._preflight_wait_key != key:
            self._preflight_wait_count += 1
            self._preflight_wait_owner = (entry.get('geometric_restore') or {}).get(
                'observation_token', self._preflight_owner_token)
        self._preflight_wait_key = key
        self._preflight_reason = reason
        self._last_source = 'candidate_waiting_preflight_state'
        return True

    def _transfer_cb(self, msg):
        if msg.action == "ping":
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", msg.session):
                return
            # Readiness response requires acquiring the actual admission mutex.
            with self._lock:
                self._admission_events.append(String(data=(
                    "action=pong ready_nonce={} receiver_boot_id={}".format(
                        msg.session, self._admission_boot_id))))
            self._flush_admission()
            return
        if (msg.action not in ("offer", "deliver", "cancel") or
                not re.fullmatch(r"[a-f0-9]{32}", msg.session) or msg.sequence <= 0 or
                msg.receiver_boot_id != self._admission_boot_id or
                not re.fullmatch(r"[a-f0-9]{64}", msg.payload_sha256)):
            return
        raw = msg.trajectory
        if not raw.points or raw.header.stamp.to_nsec() <= 0:
            return
        key = (msg.session, int(msg.sequence))
        with self._lock:
            self._admission_expire_locked()
            if self._admission_session is None:
                self._admission_session = msg.session
            if self._admission_session != msg.session:
                return  # No hot restart can replay into this live commit owner.
            entry = self._admission_entries.get(key)
            if entry is None:
                # Evicted terminal identities remain closed by this watermark.
                if msg.sequence <= self._admission_high_sequence:
                    return
                self._admission_high_sequence = int(msg.sequence)
                entry = dict(session=msg.session, sequence=int(msg.sequence),
                             digest=msg.payload_sha256, task_context=getattr(msg,"task_context",""),
                             task_regime=getattr(msg, 'task_regime', ''),
                             raw_stamp=raw.header.stamp.to_nsec(), stage="new",
                             deadline=time.monotonic(),
                             view=("probe_prefix_brake_hold" if self._probe_active else
                                   "repair_prefix_brake_hold" if self._repair_active else "full_horizon"))
                # At most 64 receipts; retire an old live identity under the
                # commit mutex if unsolicited traffic fills the bounded ledger.
                if len(self._admission_entries) >= 64:
                    for old_key, old in list(self._admission_entries.items()):
                        if old["stage"] in ("terminal", "cancelled", "expired", "rejected"):
                            del self._admission_entries[old_key]
                            break
                    else:
                        old_key = next(iter(self._admission_entries))
                        self._admission_cancel_locked(old_key)
                        del self._admission_entries[old_key]
                self._admission_entries[key] = entry
                valid_offer = (msg.action == "offer" and
                    math.isfinite(msg.remaining_budget_s) and msg.remaining_budget_s > 0 and
                    payload_sha256(raw, msg.planned_prefix_context, getattr(msg,"task_context",""), getattr(msg,'task_regime','')) == msg.payload_sha256 and
                    (not self.require_bound_regime or (self._task_regime_current_locked() and
                                                       entry['task_regime'] == self._task_regime_identity)) and
                    self._offer_prefix_valid_locked(entry, msg.planned_prefix_context) and
                    (not self.task_context_enabled or self.task_context.accepts(
                        entry["task_context"], rospy.Time.now().to_nsec(), raw.header.frame_id)) and
                    not self._replacement_active_locked(rospy.Time.now()) and
                    raw.header.stamp.to_nsec() > self._replacement_raw_floor_ns)
                if msg.action == "cancel":
                    self._admission_cancel_locked(key)
                elif not valid_offer:
                    self._admission_cancel_locked(key, "rejected", "invalid_offer_or_context")
                else:
                    entry["deadline"] = time.monotonic() + min(
                        msg.remaining_budget_s,
                        self.final_gcdf_timeout_s + self.verification_timeout_s)
                    self._admission_status_locked(entry, "offered")
            elif (entry["digest"] != msg.payload_sha256 or
                  entry["raw_stamp"] != raw.header.stamp.to_nsec() or
                  payload_sha256(raw, msg.planned_prefix_context, getattr(msg,"task_context",""), getattr(msg,'task_regime','')) != entry["digest"]):
                self._admission_cancel_locked(key, "rejected", "payload_identity_mismatch")
            elif msg.action == "cancel":
                self._admission_cancel_locked(key)
            elif entry["stage"] == "offered" and msg.action == "deliver":
                self._admission_receive_count += 1
                self._admission_status_locked(entry, "received")
                self._raw_cb(raw, _transfer_key=key)
                if entry["stage"] == "received" and self._pending_transfer_key != key:
                    self._admission_cancel_locked(key, "rejected", "dispatch_unavailable")
            else:
                self._admission_duplicate_count += 1
                self._admission_events.append(copy.deepcopy(entry["status"]))
        self._flush_admission()

    def _flush_admission(self):
        self._flush_execution_preparations()
        if not self.admission_enabled:
            return
        with self._lock:
            events = list(self._admission_events)
            self._admission_events.clear()
            queries = list(self._admission_queries)
            self._admission_queries.clear()
            summary = self._admission_summary_pending
            self._admission_summary_pending = None
        for msg in events:
            self.admission_status_pub.publish(msg)
        for route, msg in queries:
            (self.final_gcdf_query_pub if route == "gcdf" else self.verification_pub).publish(msg)
        if summary is not None:
            self.summary_pub.publish(summary)

    def _admission_tick(self):
        if not self._lock.acquire(blocking=False):
            return
        try:
            self._admission_expire_locked()
            self._sync_repair_prefix_locked()
        finally:
            self._lock.release()
        self._resume_preflight()
        self._flush_admission()

    def _blocker_stack_summary_cb(self, msg):
        """Track a live recursive blocker cycle without changing stack semantics."""
        if msg is None:
            return
        fields = _tokens(msg.data)
        reason = fields.get("switch_reason", "")
        stack = fields.get("stack", "none")
        earliest = fields.get("earliest_layer_ids", "none")
        target = fields.get("current_target_id", "-1")
        try:
            cycle_count = int(fields.get("cycle_block_count", "0"))
        except Exception:
            cycle_count = 0

        def parse_ids(text):
            out = []
            for token in str(text).split(":"):
                try:
                    out.append(int(token))
                except Exception:
                    pass
            return out

        stack_ids = parse_ids(stack)
        earliest_ids = parse_ids(earliest)
        try:
            target_id = int(target)
        except Exception:
            target_id = -1

        # Reconstruct the actual dependency-cycle predicate instead of trusting
        # a sticky diagnostic reason string: while targeting the stack top, the
        # earliest VBC layer requires an already-stacked ancestor first.
        ancestor_blocks_target = bool(
            len(stack_ids) >= 2 and
            target_id in stack_ids and
            target_id not in earliest_ids and
            any(oid in stack_ids and oid != target_id for oid in earliest_ids))
        active = bool(
            self.cycle_recovery_enabled and
            cycle_count > 0 and
            ancestor_blocks_target)
        signature = "{}|{}".format(stack, target)

        with self._lock:
            previously_active = self._cycle_recovery_active
            previous_signature = self._cycle_recovery_signature

            self._cycle_recovery_cycle_count = max(
                self._cycle_recovery_cycle_count, cycle_count)
            self._cycle_recovery_stack = stack
            self._cycle_recovery_earliest_layer_ids = earliest

            if active:
                if (not previously_active or
                        signature != previous_signature):
                    self._cycle_recovery_prefix_index = 0
                    self._cycle_recovery_activation_count += 1
                    self._cycle_recovery_last_reason = "cycle_detected"
                self._cycle_recovery_active = True
                self._cycle_recovery_signature = signature
            else:
                if previously_active:
                    self._cycle_recovery_clear_count += 1
                    self._cycle_recovery_last_reason = "cycle_cleared"
                self._cycle_recovery_active = False
                self._cycle_recovery_signature = "none"
                self._cycle_recovery_prefix_index = 0

            self._sync_repair_prefix_locked()
            self._publish_summary_locked()

    def _cycle_recovery_prefix_locked(self):
        if not (self.cycle_recovery_enabled and self._cycle_recovery_active):
            return float(self.repair_execution_prefix_s), False
        idx = min(
            max(0, int(self._cycle_recovery_prefix_index)),
            len(self.cycle_recovery_prefixes_s) - 1)
        return float(self.cycle_recovery_prefixes_s[idx]), True

    def _prefix_current_locked(self, contract):
        return bool(contract is not None and prefix_construction(contract) == self._prefix_context)

    def _offer_prefix_valid_locked(self, entry, raw_context):
        if not getattr(self, 'prefix_binding_enabled', False):
            return not raw_context
        if entry['view'] != 'repair_prefix_brake_hold':
            return not raw_context
        try:
            contract = parse_prefix_contract(raw_context)
        except (ValueError, TypeError):
            return False
        if not self._prefix_current_locked(contract):
            return False
        restore=contract.get('geometric_restore')
        if restore is not None:
            if (not self.initial_shell_recovery_enabled or not self._repair_active or self._probe_active or
                    int(restore['raw_candidate_stamp_ns']) != entry['raw_stamp']):
                return False
            entry['geometric_restore']=restore
        entry['planned_prefix'] = prefix_construction(contract)
        return True

    def _sync_repair_prefix_locked(self):
        if not getattr(self, 'prefix_binding_enabled', False):
            return
        prefix, _ = self._cycle_recovery_prefix_locked()
        if prefix != self._selected_prefix:
            self._selected_prefix = prefix
            self._prefix_revision += 1
            self._prefix_context = parse_prefix_contract(make_prefix_contract(
                self._admission_boot_id, self._prefix_revision, prefix,
                self.repair_acceleration_limits, self.repair_brake_dt_s, self.repair_brake_max_steps))
            # Cancel only uncommitted work. An executing certified brake/hold
            # remains the original fallback and is never retroactively revoked.
            for key, entry in list(self._admission_entries.items()):
                if entry.get('planned_prefix') and entry['stage'] not in (
                        'terminal', 'cancelled', 'expired', 'rejected'):
                    self._admission_cancel_locked(key)
        msg = RepairPrefixSelection(session=self._admission_boot_id,
            revision=self._prefix_revision, prefix_s=prefix,
            acceleration=list(self.repair_acceleration_limits),
            brake_dt_s=self.repair_brake_dt_s, brake_steps=self.repair_brake_max_steps)
        msg.header.stamp = rospy.Time.now()
        self.prefix_selection_pub.publish(msg)

    def _cycle_recovery_backoff_locked(self, used_prefix_s):
        if not (self.cycle_recovery_enabled and self._cycle_recovery_active):
            return
        # Resolve the attempted prefix against the configured discrete ladder.
        idx = min(
            range(len(self.cycle_recovery_prefixes_s)),
            key=lambda i: abs(
                self.cycle_recovery_prefixes_s[i] - float(used_prefix_s)))
        if idx + 1 < len(self.cycle_recovery_prefixes_s):
            self._cycle_recovery_prefix_index = idx + 1
            self._cycle_recovery_backoff_count += 1
            self._cycle_recovery_last_reason = "exact_vbc_unsafe_backoff"
            self._last_cycle_recovery_prefix_s = float(
                self.cycle_recovery_prefixes_s[idx + 1])
            rospy.logwarn(
                "[optimized_trajectory_continuity] CYCLE FRONTIER backoff "
                "stack=%s earliest=%s prefix %.3f -> %.3f s",
                self._cycle_recovery_stack,
                self._cycle_recovery_earliest_layer_ids,
                float(used_prefix_s),
                self._last_cycle_recovery_prefix_s)
        else:
            self._cycle_recovery_floor_unsafe_count += 1
            self._cycle_recovery_last_reason = "exact_vbc_unsafe_at_floor"
            self._last_cycle_recovery_prefix_s = float(
                self.cycle_recovery_prefixes_s[-1])

    def _execution_tracker_cb(self, msg):
        pairs = _TOKEN_RE.findall(msg.data)
        values = dict(pairs)
        if len(values) != len(pairs):
            return
        try:
            state = (int(values['execution_stamp_ns']), float(values['phase_s']))
            if not math.isfinite(state[1]) or state[1] < 0:
                return
        except (KeyError, ValueError):
            return
        with self._lock:
            self._execution_tracker_state = state
            self._execution_tracker_received = rospy.Time.now()
            self._handoff_tracker_fields = values
            master = self._committed_master
            if (master is not None and state[0] == master.header.stamp.to_nsec() and
                    state[1] >= self._duration(master)-1e-9 and
                    values.get('complete') == '1' and
                    values.get('execution_aborted') == '0' and
                    values.get('source') == 'trajectory_complete_hold'):
                self._handoff_completed_token = state[0]
        self._resume_preflight()

    def _actuator_state_cb(self, msg):
        if getattr(self, 'require_bound_regime', False):
            return
        if len(msg.name)!=len(msg.velocity) or len(set(msg.name))!=len(msg.name):
            return
        if not all(math.isfinite(v) for v in msg.velocity):
            return
        with self._lock:
            self._latest_actuator_stamp_ns=msg.header.stamp.to_nsec()
            self._latest_actuator_velocity=dict(zip(msg.name,msg.velocity))
        self._resume_preflight()

    def _controller_state_cb(self, msg):
        now = rospy.Time.now().to_nsec()
        stamp = msg.header.stamp.to_nsec()
        if (not msg.execution_lease_required or not msg.controller_session or msg.clock_fault or
                not 0 < stamp <= now or now-stamp > 50000000 or
                len(msg.applied_velocity) != 7 or len(msg.velocity_limits) != 7 or
                any(not math.isfinite(v) for v in msg.applied_velocity) or
                any(not math.isfinite(v) or v <= 0 for v in msg.velocity_limits) or
                any(abs(v) > cap+1e-12 for v,cap in zip(msg.applied_velocity,msg.velocity_limits))):
            return
        with self._lock:
            if stamp <= self._latest_actuator_stamp_ns or msg.controller_session in self._retired_controller_sessions:
                return
            if self._controller_session and self._controller_session != msg.controller_session:
                if len(self._retired_controller_sessions) >= 16:
                    self._admission_invalidate_locked()
                    self._task_stop_committed_locked()
                    return
                self._retired_controller_sessions.add(self._controller_session)
                self._admission_invalidate_locked()
                self._task_stop_committed_locked()
            self._controller_session = msg.controller_session
            self._latest_actuator_stamp_ns = stamp
            self._latest_actuator_velocity = dict(zip(self._controller_joint_names,msg.applied_velocity))
        self._resume_preflight()

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
            self._handoff_previous_measured = (
                self._latest_measured_stamp_ns,
                dict(self._latest_measured_q_by_name),
                dict(self._latest_measured_velocity_by_name))
            self._latest_measured_q_by_name = measured
            self._latest_measured_velocity_by_name = (
                dict(zip(msg.name, msg.velocity)) if len(msg.velocity)==len(msg.name) and len(set(msg.name))==len(msg.name) else {})
            self._latest_measured_stamp_ns = msg.header.stamp.to_nsec()
            self._latest_joint_state_received = now
        self._resume_preflight()

    def _publish_verification(self, candidate):
        if getattr(self, 'initial_shell_recovery_enabled', False):
            with self._lock:
                if self._outstanding is None or self._outstanding.header.stamp != candidate.header.stamp:
                    return
                eligible = bool(self.final_gcdf_enabled and self._outstanding_repair and
                    not self._outstanding_probe and self._outstanding_view=='repair_prefix_brake_hold' and
                    ((self._outstanding_diag or {}).get('final_gcdf_passed', False) or
                     (self._outstanding_diag or {}).get('geometry_override_pending', False)))
                diag = self._outstanding_diag or {}
                key = diag.get('transfer_key')
                session, sequence = key if key is not None else (self._admission_boot_id, self._outstanding_seq)
                data = shell_context(candidate, candidate.header.stamp.to_nsec()+500000000,
                    eligible, self.repair_acceleration_limits, self.repair_brake_dt_s, self.repair_brake_max_steps,
                    request_session=session, request_sequence=sequence,
                    planned_prefix=diag.get('planned_prefix'),
                    geometric_restore=diag.get('geometric_restore'))
                self.shell_context_pub.publish(String(data=data))
                target=diag.get('shell_required_residual_m',0.)
                if target>0. and getattr(self,'shell_quality_pub',None) is not None:
                    self.shell_quality_pub.publish(String(data=json.dumps(dict(schema=1,
                        trajectory_stamp_ns=str(candidate.header.stamp.to_nsec()),
                        request_session=session,request_sequence=sequence,required_residual_m=target,
                        reply_allowance_s=diag.get('shell_reply_allowance_s',.003)),
                        separators=(',',':'),allow_nan=False)))
        self.verification_pub.publish(candidate)

    def _verification_protocol_cb(self, msg):
        """Close an invalid audit without manufacturing a geometric rejection."""
        if msg is None or not self.initial_shell_recovery_enabled:
            return
        fields = _tokens(msg.data)
        try:
            stamp = int(fields['trajectory_stamp_ns'])
        except (KeyError, TypeError, ValueError):
            return
        state = fields.get('state')
        reason = fields.get('reason', 'invalid_context')
        if state not in ('waiting', 'invalid', 'validated'):
            return
        event = None
        dispatched = None
        route = 'none'
        with self._lock:
            if self.admission_enabled:
                self._admission_expire_locked()
            if (self._outstanding is not None and self._outstanding_sent is not None and
                    stamp == self._outstanding.header.stamp.to_nsec()):
                self._last_vbc_protocol_reason = reason
                if state == 'validated':
                    self._last_vbc_protocol_reason = 'none'
                    if self._outstanding_diag:
                        self._outstanding_diag.pop('vbc_protocol_reason', None)
                elif state == 'waiting':
                    # Keep the original deadline while the other topic arrives.
                    self._outstanding_diag = self._outstanding_diag or {}
                    self._outstanding_diag['vbc_protocol_reason'] = reason
                else:
                    now = rospy.Time.now()
                    age = max(0., (now-self._outstanding_sent).to_sec())
                    seq, view = self._outstanding_seq, self._outstanding_view
                    diag = copy.deepcopy(self._outstanding_diag) or {}
                    diag['audited_trajectory_stamp_ns'] = stamp
                    diag['vbc_protocol_reason'] = reason
                    if reason == 'geometric_restore_path_counterexample':
                        evidence = fields.get('path_counterexample_evidence', 'none')
                        if len(evidence) <= 2048:
                            diag['path_counterexample_evidence'] = evidence
                    if reason == 'geometric_restore_proof_resource_exhausted':
                        evidence = fields.get('proof_resource_evidence', 'none')
                        if len(evidence) <= 2048:
                            diag['proof_resource_evidence'] = evidence
                    self._outstanding = self._outstanding_sent = None
                    self._outstanding_seq = 0
                    self._outstanding_dispatch_cycle = -1
                    self._outstanding_repair = self._outstanding_probe = False
                    self._outstanding_view = 'none'
                    self._outstanding_raw_received = self._outstanding_diag = None
                    self._verification_protocol_invalid_count += 1
                    self._verification_outcome_count += 1
                    self._last_verification_result = 'protocol_invalid'
                    self._last_verification_seq = seq
                    self._last_verification_age_s = age
                    self._last_verification_view = view
                    self._last_source = 'candidate_rejected_vbc_protocol_invalid'
                    event = self._make_verification_event(seq, 'protocol_invalid', False,
                        age, view, safety_gate='vbc_protocol', diag=diag)
                    dispatched, route = self._dispatch_pending_locked(now)
                self._publish_summary_locked()
        if event is not None:
            self._publish_verification_event(event)
        if dispatched is not None:
            if route == 'gcdf':
                self.final_gcdf_query_pub.publish(dispatched)
            elif route == 'vbc':
                self._publish_verification(dispatched)
        if self.admission_enabled:
            self._flush_admission()

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

    def _measured_rebase_candidate_locked(
            self, raw, now, enabled, max_shift_inf, max_age_s):
        """Translate q(t) by a constant offset so q(0) matches measured q.

        The translation preserves dq/ddq and trajectory shape. The returned
        trajectory is always audited by final GCDF + exact VBC before commit.
        """
        out = copy.deepcopy(raw)
        base_diag = {
            "enabled": int(bool(enabled)),
            "applied": 0,
            "clamped": 0,
            "target_shift_inf": math.nan,
            "applied_shift_inf": 0.0,
            "residual_inf": math.nan,
            "joint_state_age_ms": self._joint_state_age_ms_locked(now),
        }
        if (not enabled or raw is None or
                not raw.points or not raw.joint_names):
            return out, base_diag

        measured = self._measured_positions_locked(raw.joint_names)
        joint_state_age_ms = self._joint_state_age_ms_locked(now)
        if (measured is None or not math.isfinite(joint_state_age_ms) or
                joint_state_age_ms > 1000.0 * max_age_s):
            base_diag["residual_inf"] = self._measured_mismatch_inf_locked(
                raw.joint_names, raw.points[0].positions)
            return out, base_diag

        if len(raw.points[0].positions) != len(raw.joint_names):
            return out, base_diag

        target_delta = [
            measured[j] - float(raw.points[0].positions[j])
            for j in range(len(raw.joint_names))
        ]
        applied_delta = []
        clamped = False
        for j, name in enumerate(raw.joint_names):
            delta = max(
                -max_shift_inf,
                min(max_shift_inf, target_delta[j]))
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

        return out, {
            "enabled": 1,
            "applied": 1,
            "clamped": int(clamped),
            "target_shift_inf": target_inf,
            "applied_shift_inf": applied_inf,
            "residual_inf": residual_inf,
            "joint_state_age_ms": joint_state_age_ms,
        }

    def _probe_rebase_candidate_locked(self, raw, now):
        out, diag = self._measured_rebase_candidate_locked(
            raw, now,
            self.probe_start_rebase_enabled,
            self.probe_rebase_max_shift_inf,
            self.probe_start_joint_state_max_age_s)
        if diag.get("applied", 0):
            self._probe_rebase_count += 1
            if diag.get("clamped", 0):
                self._probe_rebase_clamp_count += 1
            self._last_probe_rebase_shift_inf = diag.get(
                "applied_shift_inf", math.nan)
            self._last_probe_rebase_target_shift_inf = diag.get(
                "target_shift_inf", math.nan)
            self._last_probe_rebase_residual_inf = diag.get(
                "residual_inf", math.nan)
        return out, diag

    def _smooth_handoff_rebase_candidate_locked(self, raw, now):
        return self._measured_rebase_candidate_locked(
            raw, now,
            self.smooth_handoff_rebase_enabled,
            self.smooth_handoff_rebase_max_shift_inf,
            self.smooth_handoff_joint_state_max_age_s)

    def _repair_active_cb(self, msg):
        if getattr(self, 'require_bound_regime', False) and not getattr(self._task_control_dispatch, 'active', False):
            return
        if msg is None:
            return
        with self._lock:
            if self._repair_active != bool(msg.data):
                self._admission_invalidate_locked()
            self._repair_active = bool(msg.data)
        self._flush_admission()

    def _probe_active_cb(self, msg):
        if getattr(self, 'require_bound_regime', False) and not getattr(self._task_control_dispatch, 'active', False):
            return
        if msg is None:
            return
        with self._lock:
            if self._probe_active != bool(msg.data):
                self._admission_invalidate_locked()
            self._probe_active = bool(msg.data)
        self._flush_admission()

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
        motion=interpolate_motion(p0.positions,p1.positions,p0.velocities,p1.velocities,
            (p1.time_from_start-p0.time_from_start).to_sec(),alpha)
        if motion is not None:
            out.positions,out.velocities,out.accelerations=motion
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
            self, candidate, prefix_duration_s=None, consistent_motion=None,
            normal_velocity_reference=False, motion_start=None):
        def fail(reason):
            self._last_prefix_build_reason=reason
            return None
        self._last_prefix_build_reason="none"
        if candidate is None or not candidate.points:
            return fail("empty_candidate")
        duration = self._duration(candidate)
        if duration <= 0.0:
            return fail("invalid_duration")
        requested_prefix_s = (
            self.repair_execution_prefix_s
            if prefix_duration_s is None else float(prefix_duration_s))
        prefix_t = min(max(1e-6, requested_prefix_s), duration)
        endpoint = self._point_at_time(candidate, prefix_t)
        if endpoint is None or not endpoint.positions:
            return fail("missing_endpoint")

        n = len(candidate.joint_names)
        if n == 0:
            n = len(endpoint.positions)
        if len(endpoint.positions) != n:
            return fail("joint_dimensions")
        if len(self.repair_acceleration_limits) != n:
            rospy.logerr_throttle(
                1.0,
                "[optimized_trajectory_continuity] repair acceleration-limit size %d != dof %d",
                len(self.repair_acceleration_limits), n)
            return fail("brake_or_acceleration_contract")

        out = JointTrajectory()
        out.header = copy.deepcopy(candidate.header)
        out.header.stamp = rospy.Time.now()
        out.joint_names = list(candidate.joint_names)

        consistent = getattr(self,'repair_motion_consistency_enabled',False) if consistent_motion is None else consistent_motion
        nodal_motion = (normal_velocity_reference or consistent) and has_nodal_motion(
            candidate.points, self.repair_acceleration_limits)
        self._last_normal_nodal_motion = int(normal_velocity_reference and nodal_motion)
        self._last_repair_nodal_motion = int(not normal_velocity_reference and nodal_motion)
        if consistent:
            now_ns=rospy.Time.now().to_nsec()
            try:
                if not 0<=now_ns-self._latest_measured_stamp_ns<=50000000:
                    return fail("live_state_stale")
                if not 0<=now_ns-self._latest_actuator_stamp_ns<=50000000:
                    return fail("actuator_state_stale")
                q=[float(self._latest_measured_q_by_name[name]) for name in out.joint_names]
                vel=[float(self._latest_measured_velocity_by_name[name]) for name in out.joint_names]
                applied=[float(self._latest_actuator_velocity[name]) for name in out.joint_names]
                if not all(math.isfinite(x) for x in q+vel+applied):return fail("nonfinite_state")
                if any(abs(v-u)>a*.05 for v,u,a in zip(vel,applied,self.repair_acceleration_limits)):
                    return fail("actuator_velocity_mismatch")
                if motion_start is not None:
                    if (motion_start['source_stamp_ns']!=self._latest_measured_stamp_ns or
                            len(motion_start['q'])!=n or len(motion_start['velocity'])!=n or
                            not all(math.isfinite(x) for x in motion_start['q']+motion_start['velocity']) or
                            max(abs(x-y) for x,y in zip(q,motion_start['q']))>.03):
                        return fail("invalid_handoff_start_proposal")
                    # A forecast is a proposal anchor, never a measured state.
                    # The real measured dictionaries remain intact for both
                    # geometry audits and the live initial/actuator check.
                    q=list(motion_start['q']);vel=list(motion_start['velocity'])
                start=JointTrajectoryPoint(positions=q,velocities=vel,accelerations=[0.]*n,
                    time_from_start=rospy.Duration(0.))
                out.points=[start]
                t=0.
                # Use original knot boundaries; build a single reference ramp.
                ends=sorted(set([p.time_from_start.to_sec() for p in candidate.points
                    if 0.<p.time_from_start.to_sec()<prefix_t]+[prefix_t]))
                for end in ends:
                    while t<end-1e-9:
                        dt=min(self.repair_brake_dt_s,end-t)
                        next_time=rospy.Duration.from_sec(t+dt)
                        dt=next_time.to_sec()-t
                        if dt<=0:return fail("invalid_knot_time")
                        p0=self._point_at_time(candidate,t)
                        p1=self._point_at_time(candidate,t+dt)
                        # A NORMAL ramp QP supplies endpoint velocities. Using
                        # its secant as the next velocity integrates a second
                        # ramp and moves the goal. Legacy proposals still get
                        # the complete original physical ramp/brake construction.
                        desired=(list(p1.velocities) if nodal_motion else
                            [(b-a)/dt for a,b in zip(p0.positions,p1.positions)])
                        next_vel=ramp_step(vel,desired,self.repair_acceleration_limits,dt)
                        q=[x+.5*dt*(v+w) for x,v,w in zip(q,vel,next_vel)]
                        t=next_time.to_sec()
                        out.points.append(JointTrajectoryPoint(positions=q,velocities=next_vel,
                            accelerations=[(w-v)/dt for v,w in zip(vel,next_vel)],time_from_start=next_time))
                        vel=next_vel
            except (KeyError,ValueError,TypeError,OverflowError):
                return fail("motion_construction_invalid")
            endpoint=out.points[-1]

        else:
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
            next_time=rospy.Duration.from_sec(t+dt)
            dt=next_time.to_sec()-t
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
            t = next_time.to_sec()
            p = JointTrajectoryPoint()
            p.positions = list(q)
            p.velocities = list(next_vel)
            p.accelerations = list(accel)
            p.time_from_start = next_time
            out.points.append(p)
            vel = next_vel

        if max(abs(v) for v in vel) > 1e-5:
            rospy.logwarn_throttle(
                1.0,
                "[optimized_trajectory_continuity] braking tail hit max steps; max|v|=%.4f",
                max(abs(v) for v in vel))
            return fail("brake_or_acceleration_contract")

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
        if consistent:
            for j,name in enumerate(out.joint_names):
                velocity_limit=getattr(self,'recovery_velocity_limits',{}).get(name)
                bounds=getattr(self,'recovery_position_limits',{}).get(name)
                if velocity_limit is not None and any(abs(p.velocities[j])>float(velocity_limit)+1e-9 for p in out.points):
                    return fail("motion_limits_or_missing_state")
                if bounds is not None:
                    low=float(bounds['lower'])+self.joint_position_margin
                    high=float(bounds['upper'])-self.joint_position_margin
                    extrema=[p.positions[j] for p in out.points]
                    for a,b in zip(out.points,out.points[1:]):
                        dt=(b.time_from_start-a.time_from_start).to_sec()
                        dv=b.velocities[j]-a.velocities[j]
                        if abs(dv)>1e-12:
                            tau=-a.velocities[j]*dt/dv
                            if 0.<tau<dt:extrema.append(a.positions[j]+a.velocities[j]*tau+.5*dv/dt*tau*tau)
                    if min(extrema)<low or max(extrema)>high:return fail("continuous_joint_limit")
        return out

    def _replacement_active_locked(self, now):
        fence = self._replacement_fence
        if fence is not None and (now.to_nsec() >= fence['deadline_ns'] or
                                  time.monotonic() >= fence['wall_deadline']):
            self._replacement_close_locked(now)
        return self._replacement_fence is not None

    def _replacement_close_locked(self, now):
        if self._replacement_fence is not None:
            self._replacement_closed_id = max(self._replacement_closed_id,
                                              self._replacement_fence['request_id'])
        self._replacement_fence = None
        # Late deliveries must not become eligible after release.
        self._replacement_raw_floor_ns = max(self._replacement_raw_floor_ns, now.to_nsec())

    def _replacement_grant_cb(self, msg):
        try:
            words = msg.data.split()
            fields = dict(word.split('=', 1) for word in words)
            if len(fields) != len(words) or fields['granted'] != '1':
                return
            fence = {k: int(fields[k]) for k in
                     ('request_id', 'mode_epoch', 'query_stamp_ns', 'deadline_ns')}
            if min(fence.values()) <= 0:
                return
        except (KeyError, ValueError):
            return
        with self._lock:
            now = rospy.Time.now()
            self._replacement_active_locked(now)
            if (fence['request_id'] <= self._replacement_closed_id or
                    now.to_nsec() >= fence['deadline_ns']):
                return
            current = self._replacement_fence
            if (current is not None and fence['request_id'] > current['request_id'] and
                    fence['mode_epoch'] >= current['mode_epoch']):
                # A newer planner-authorized reservation supersedes the old
                # one even if its finish was delayed/lost. Old finishes cannot
                # release the new identity.
                self._replacement_close_locked(now)
                current = None
            if current is not None:
                # Duplicates cannot extend a deadline or change identity.
                if any(current[k] != fence[k] for k in fence):
                    return
            else:
                fence['wall_deadline'] = time.monotonic() + 3.
                self._admission_invalidate_locked()
                self._replacement_fence = fence
                self._replacement_raw_floor_ns = max(self._replacement_raw_floor_ns, now.to_nsec())
                cancelled = dict(pending_raw=(self._pending_raw.header.stamp.to_nsec()
                    if self._pending_raw is not None else 0),
                    gcdf_raw=(self._gcdf_outstanding_diag or {}).get('raw_candidate_stamp_ns', 0)
                    if self._gcdf_outstanding is not None else 0,
                    vbc_raw=(self._outstanding_diag or {}).get('raw_candidate_stamp_ns', 0)
                    if self._outstanding is not None else 0)
                self._pending_raw = self._pending_raw_received = None
                self._gcdf_outstanding = self._gcdf_outstanding_sent = None
                self._outstanding = self._outstanding_sent = None
                rospy.logwarn('[replacement_commit_fence] acquired request=%s cancelled=%s',
                              fence['request_id'], cancelled)
            # Same mutex as actual committed publication: any earlier commit
            # must be observed by the tracker before generation can start.
            self.replacement_ack_pub.publish(String(data=' '.join(
                '{}={}'.format(k, fence[k]) for k in
                ('request_id', 'mode_epoch', 'query_stamp_ns', 'deadline_ns')) +
                ' confirmed=1 last_execution_stamp_ns={}'.format(self._last_execution_stamp_ns)))
        self._flush_admission()

    def _replacement_finish_cb(self, msg):
        try:
            words = msg.data.split()
            fields = dict(word.split('=', 1) for word in words)
            if len(fields) != len(words) or fields.get('action') != 'finish':
                return
            rid = int(fields['request_id'])
            if rid <= 0:
                return
        except (KeyError, ValueError):
            return
        with self._lock:
            # Different ROS topics can reorder finish and grant.
            self._replacement_closed_id = max(self._replacement_closed_id, rid)
            if self._replacement_fence is not None and self._replacement_fence['request_id'] == rid:
                self._replacement_close_locked(rospy.Time.now())

    def _raw_cb(self, msg, _transfer_key=None):
        if msg is None or not msg.points:
            return

        now = rospy.Time.now()
        gcdf_to_publish = None
        verification_to_publish = None

        with self._lock:
            if self.admission_enabled and _transfer_key is None:
                return  # The raw diagnostic topic cannot bypass admission.
            if (self._replacement_active_locked(now) or
                    msg.header.stamp.to_nsec() <= self._replacement_raw_floor_ns):
                return
            if self._last_raw_received is not None:
                gap = (now - self._last_raw_received).to_sec()
                if gap >= 0.0:
                    self._last_input_gap_s = gap
                    self._max_input_gap_s = max(self._max_input_gap_s, gap)

            if self._pending_raw is not None:
                self._pending_replace_count += 1
                if self._pending_transfer_key is not None:
                    self._admission_cancel_locked(self._pending_transfer_key)

            self._pending_raw = copy.deepcopy(msg)
            self._pending_transfer_key = _transfer_key
            self._pending_raw_received = now
            self._pending_repair = bool(self._repair_active)
            self._pending_probe = bool(self._probe_active)
            self._last_raw_received = now
            self._raw_input_count += 1

            # C5.36: safety admission is event-driven. A new raw candidate does
            # not need a background VBC selector cycle before it may enter the
            # single-flight final-GCDF -> exact-VBC certification pipeline.
            # If certification is busy, preserve the existing latest-only
            # pending buffer; the newest candidate is dispatched immediately
            # when the current certification terminates.
            dispatched, route = self._dispatch_pending_locked(now)
            if dispatched is not None:
                self._raw_immediate_dispatch_count += 1
                self._last_source = "raw_candidate_immediate_safety_dispatch"
                if route == "gcdf":
                    gcdf_to_publish = dispatched
                elif route == "vbc":
                    verification_to_publish = dispatched
            else:
                self._raw_busy_buffer_count += 1
                self._last_source = "raw_candidate_buffered_safety_busy"

            self._publish_summary_locked()
            if self.admission_enabled:
                if gcdf_to_publish is not None:
                    self._admission_queries.append(("gcdf", gcdf_to_publish))
                if verification_to_publish is not None:
                    self._admission_queries.append(("vbc", verification_to_publish))
                return  # Caller flushes after releasing the outer admission lock.

        # Never publish while holding the commit-gate mutex: final GCDF and
        # event-driven exact VBC can answer quickly.
        if gcdf_to_publish is not None:
            self.final_gcdf_query_pub.publish(gcdf_to_publish)
        if verification_to_publish is not None:
            self._publish_verification(verification_to_publish)

    @staticmethod
    def _handoff_scope(diag):
        restore = diag.get('geometric_restore') or {}
        prefix = diag.get('planned_prefix') or {}
        return (restore.get('mode_epoch'), restore.get('observation_token'),
                prefix.get('session'), prefix.get('revision'), prefix.get('prefix_s'))

    def _handoff_wait_reason_locked(self, raw, entry, now):
        """Let the current certified stop finish before building a restore.

        Pending raw is a shape proposal, never an executable certificate. This
        uses the original first-offer lease and does not start or refund an audit.
        """
        master = self._committed_master
        if master is None:
            return 'none'
        hold_start = terminal_hold_start(master.points)
        if hold_start is None:
            return 'none'
        state, received = self._execution_tracker_state, self._execution_tracker_received
        fields = self._handoff_tracker_fields
        if state is None or received is None or not 0 <= (now-received).to_sec() <= .05:
            return 'handoff_tracker_unavailable'
        token = master.header.stamp.to_nsec()
        paused_sources = ('external_vbc_hold', 'external_gcdf_hard_hold',
                          'primitive_tracking_envelope_hold', 'tracking_error_hold',
                          'zero_velocity_hold', 'initial_zero_hold')
        if state[0] in (0, token) and fields.get('source') in paused_sources:
            # A safety-held tracker freezes phase; its old braking tail cannot
            # finish. Rebuild from fresh stopped state instead of waiting for
            # that impossible event. This does not clear any execution hold:
            # final GCDF, full VBC and the actual commit checks still decide.
            return self._handoff_state_wait_reason_locked(raw, entry)
        active = (state[0] == token and fields.get('source') in
                  ('active_trajectory', 'trajectory_complete_hold') and
                  fields.get('execution_aborted') == '0')
        # This reports no active motion; it need not prove completion of the
        # last published curve. The rebuilt curve starts at measured q/v and
        # still needs full certification, including its actuator-state bound.
        held = (state[0] == 0 and
                     fields.get('source') == 'last_reference_hold' and
                     fields.get('execution_aborted') == '0')
        if not active and not held:
            # An aborted reference has no certified tail left to wait for. A
            # fresh, stopped state may still seed a new fully audited curve.
            if fields.get('execution_aborted') == '1':
                return self._handoff_state_wait_reason_locked(raw, entry)
            return 'handoff_tracker_identity'
        if active and state[1] < hold_start-1e-9:
            return 'handoff_certified_brake_pending'
        return self._handoff_state_wait_reason_locked(raw, entry)

    def _handoff_state_wait_reason_locked(self, raw, entry):
        try:
            entry.pop('handoff_prediction', None)
            names = raw.joint_names
            q = [self._latest_measured_q_by_name[n] for n in names]
            v = [self._latest_measured_velocity_by_name[n] for n in names]
            u = [self._latest_actuator_velocity[n] for n in names]
            certificate = self._handoff_certificate
            if certificate is not None and certificate[0] == self._handoff_scope(entry):
                old = self._handoff_previous_measured
                previous = None if old is None else (
                    old[0], [old[1][n] for n in names], [old[2][n] for n in names])
                ready = state_variation_ready(previous, (self._latest_measured_stamp_ns, q, v),
                    u, certificate[1], self.repair_acceleration_limits, self.tracking_feedback_gain)
                prediction = None
                fields = self._handoff_tracker_fields
                master = self._committed_master
                hold = None if master is None else terminal_hold_start(master.points)
                state = self._execution_tracker_state
                stationary_reference = (fields.get('source') == 'last_reference_hold' or
                    (fields.get('source') in ('active_trajectory','trajectory_complete_hold') and
                     master is not None and state is not None and state[0] == master.header.stamp.to_nsec() and
                     hold is not None and state[1] >= hold-1e-9))
                if getattr(self, 'handoff_prediction_enabled', False) and stationary_reference:
                    prediction = predict_held_motion(previous, (self._latest_measured_stamp_ns,q,v),
                        u, certificate[1], self.repair_acceleration_limits, self.tracking_feedback_gain)
                    if prediction is not None:
                        entry['handoff_prediction'] = prediction
                # The original100ms low-speed window may release a new start
                # only when measured/applied samples support the held-state
                # forecast. A small speed alone cannot cover a thin reserve.
                low_speed = all(math.isfinite(x) and abs(x) <= .01 for x in v+u)
                settled = entry.get('handoff_settle_start')
                stamp = self._latest_measured_stamp_ns
                if (not low_speed or not all(math.isfinite(x) for x in q)):
                    entry.pop('handoff_settle_start', None)
                elif (settled is None or stamp < settled[0] or
                      max(abs(a-b) for a,b in zip(q, settled[1])) > .001):
                    entry['handoff_settle_start'] = (stamp, list(q))
                elif not ready and prediction is not None and stamp-settled[0] >= 100000000:
                    ready = True
                    rospy.loginfo('[HANDOFF_SCHEDULE] event=settled_rebuild raw=%s old_reserve_m=%.12g',
                                  raw.header.stamp.to_nsec(), certificate[1]['certified_residual_m'])
            else:
                ready = all(math.isfinite(x) and abs(x) <= .01 for x in v+u)
            return 'none' if ready else 'handoff_state_changing'
        except (KeyError, TypeError, ValueError):
            return 'handoff_state_unavailable'

    def _dispatch_pending_locked(self, now):
        self._admission_expire_locked()
        if self._replacement_active_locked(now):
            return None, "none"
        if (self._gcdf_outstanding is not None or
                self._outstanding is not None or
                self._pending_raw is None):
            return None, "none"

        raw = self._pending_raw
        transfer_key = self._pending_transfer_key
        raw_received = self._pending_raw_received
        pending_repair = bool(self._pending_repair or self._repair_active)
        pending_probe = bool(self._pending_probe or self._probe_active)
        planned_prefix = None
        if getattr(self, '_handoff_tracker_fields', {}).get('execution_state') == 'STOPPING':
            if transfer_key in self._admission_entries:
                self._defer_preflight_locked(raw, raw_received, transfer_key,
                                            pending_repair, pending_probe, 'tracker_stopping')
            return None, 'none'
        if getattr(self, 'prefix_binding_enabled', False) and pending_repair and not pending_probe:
            entry = self._admission_entries.get(transfer_key, {})
            planned_prefix = entry.get('planned_prefix')
            if not self._prefix_current_locked(planned_prefix):
                if transfer_key in self._admission_entries:
                    self._admission_cancel_locked(transfer_key, 'rejected')
                self._pending_raw = self._pending_raw_received = None
                return None, 'none'
        # Keep the admitted request while temporary state evidence is missing.
        # No audit starts, and neither first-offer deadline nor recovery ledger
        # changes. State callbacks/tick resume this exact pending identity.
        entry = self._admission_entries.get(transfer_key)
        needs_live = self.repair_motion_consistency_enabled
        if self.admission_enabled and entry is not None and needs_live:
            now_ns = rospy.Time.now().to_nsec()
            measured_age = now_ns - self._latest_measured_stamp_ns
            actuator_age = now_ns - self._latest_actuator_stamp_ns
            self._preflight_measured_age_ns = measured_age
            self._preflight_actuator_age_ns = actuator_age
            reason = ('live_state_unavailable' if not self._latest_measured_stamp_ns or
                      not 0 <= measured_age <= 50000000 else
                      'actuator_state_unavailable' if not self._latest_actuator_stamp_ns or
                      not 0 <= actuator_age <= 50000000 else 'none')
            if reason != 'none':
                self._defer_preflight_locked(raw, raw_received, transfer_key,
                                             pending_repair, pending_probe, reason)
                return None, 'none'
            if (self.initial_shell_recovery_enabled and pending_repair and not pending_probe and
                    entry.get('geometric_restore') is not None):
                reason = self._handoff_wait_reason_locked(raw, entry, rospy.Time.now())
                if reason != 'none':
                    if self._handoff_wait_key != transfer_key:
                        self._handoff_wait_key = transfer_key
                        self._handoff_wait_count += 1
                        rospy.loginfo('[HANDOFF_SCHEDULE] event=wait raw=%s reason=%s deadline_wall=%.9f',
                            raw.header.stamp.to_nsec(), reason, entry['deadline'])
                    self._defer_preflight_locked(raw, raw_received, transfer_key,
                                                 pending_repair, pending_probe, reason)
                    return None, 'none'
                if self._handoff_wait_key == transfer_key:
                    self._handoff_resume_count += 1
                    self._handoff_wait_key = None
                    rospy.loginfo('[HANDOFF_SCHEDULE] event=resume raw=%s deadline_wall=%.9f',
                        raw.header.stamp.to_nsec(), entry['deadline'])
            if self._preflight_wait_key == transfer_key:
                self._preflight_resume_count += 1
                self._preflight_wait_key = self._preflight_wait_owner = None
                self._preflight_reason = 'fresh_state_resumed'
            now = rospy.Time.now()
        self._pending_transfer_key = None
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
        handoff_rebase_diag = {
            "enabled": int(self.smooth_handoff_rebase_enabled),
            "applied": 0,
            "clamped": 0,
            "target_shift_inf": math.nan,
            "applied_shift_inf": 0.0,
            "residual_inf": math.nan,
            "joint_state_age_ms": self._joint_state_age_ms_locked(now),
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
        elif pending_repair:
            if self.repair_projection_consistent_dispatch_enabled:
                # This newly proposed trajectory has not executed while waiting
                # for admission. Keep its planned prefix; only rebase q-origin
                # to the fresh measured state before both unchanged audits.
                candidate = copy.deepcopy(raw)
                dispatch_suffix_phase_s = 0.0
            else:
                candidate = self._suffix_from_phase(raw, age)
                dispatch_suffix_phase_s = float(age)
            candidate, handoff_rebase_diag = (
                self._smooth_handoff_rebase_candidate_locked(candidate, now))
        else:
            # Verification latency is not execution time. Preserve the full
            # horizon, then align only its q-origin to the latest measured state
            # BEFORE safety certification.
            candidate = copy.deepcopy(raw)
            dispatch_suffix_phase_s = 0.0
            candidate, handoff_rebase_diag = (
                self._smooth_handoff_rebase_candidate_locked(candidate, now))
        if candidate is None or not candidate.points:
            self._last_source = "discarded_expired_raw_candidate"
            return None, "none"

        dispatch_start_positions = (
            list(candidate.points[0].positions)
            if candidate.points[0].positions else [])
        diag = {
            "raw_candidate_age_s": float(age),
            "raw_candidate_stamp_ns": int(raw.header.stamp.to_nsec()),
            "dispatch_suffix_phase_s": dispatch_suffix_phase_s,
            "raw_candidate_duration_s": float(raw_duration_s),
            "dispatch_suffix_start_shift_inf": (
                self._position_shift_inf(
                    dispatch_start_positions, raw_start_positions)
                if pending_repair else 0.0),
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
            "handoff_rebase_enabled": int(
                handoff_rebase_diag.get("enabled", 0)),
            "handoff_rebase_applied": int(
                handoff_rebase_diag.get("applied", 0)),
            "handoff_rebase_clamped": int(
                handoff_rebase_diag.get("clamped", 0)),
            "handoff_rebase_target_shift_inf": handoff_rebase_diag.get(
                "target_shift_inf", math.nan),
            "handoff_rebase_shift_inf": handoff_rebase_diag.get(
                "applied_shift_inf", math.nan),
            "handoff_rebase_residual_inf": handoff_rebase_diag.get(
                "residual_inf", math.nan),
            "handoff_rebase_joint_state_age_ms": handoff_rebase_diag.get(
                "joint_state_age_ms", math.nan),
            "probe_speed_max_before": speed_diag.get(
                "speed_max_before", math.nan),
            "probe_speed_max_after": speed_diag.get(
                "speed_max_after", math.nan),
            "probe_time_scale": speed_diag.get("time_scale", 1.0),
            "probe_time_scale_clamped": int(
                speed_diag.get("time_scale_clamped", 0)),
            "probe_effective_prefix_s": (
                self.probe_execution_prefix_s *
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
            "motion_anchor_stamp_ns": self._latest_measured_stamp_ns,
            "handoff_start_prediction": 0,
            "handoff_prediction_delay_s": 0.,
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

        if transfer_key is not None:
            diag["transfer_key"] = transfer_key
            restore=self._admission_entries.get(transfer_key,{}).get('geometric_restore')
            if restore is not None:
                diag['geometric_restore']=copy.deepcopy(restore)
        view = "full_horizon"
        prefix_mode = bool(pending_repair or pending_probe)
        if self.repair_prefix_verification_enabled and prefix_mode:
            cycle_prefix_s, cycle_prefix_active = (
                self._cycle_recovery_prefix_locked()
                if pending_repair and not pending_probe
                else (self.repair_execution_prefix_s, False))
            if planned_prefix is not None:
                cycle_prefix_s = planned_prefix['prefix_s']
                diag['planned_prefix'] = planned_prefix
            effective_prefix_s = (
                self.probe_execution_prefix_s *
                float(speed_diag.get("time_scale", 1.0))
                if pending_probe else cycle_prefix_s)
            diag["cycle_recovery_active"] = int(cycle_prefix_active)
            diag["cycle_recovery_prefix_index"] = int(
                self._cycle_recovery_prefix_index if cycle_prefix_active else 0)
            diag["cycle_recovery_prefix_s"] = float(effective_prefix_s)
            diag["cycle_recovery_cycle_count"] = int(
                self._cycle_recovery_cycle_count)
            self._last_probe_effective_prefix_s = (
                effective_prefix_s if pending_probe
                else self._last_probe_effective_prefix_s)
            start_proposal = (entry or {}).get('handoff_prediction') if pending_repair and not pending_probe else None
            if start_proposal is not None:
                diag['handoff_start_prediction'] = 1
                diag['handoff_prediction_delay_s'] = start_proposal['latency_s']
                rospy.loginfo('[HANDOFF_START_PROPOSAL] raw=%s source_stamp_ns=%s delay_s=%.9g estimated_error_m=%.9g',
                    raw.header.stamp.to_nsec(),start_proposal['source_stamp_ns'],start_proposal['latency_s'],
                    start_proposal['estimated_error_m'])
            candidate = self._repair_prefix_with_braking_tail(
                candidate, prefix_duration_s=effective_prefix_s,
                consistent_motion=bool(self.repair_motion_consistency_enabled),
                motion_start=start_proposal)
            if candidate is None or not candidate.points:
                self._last_source = "prefix_build_failed"
                reason = getattr(self, '_last_prefix_build_reason', 'unknown')
                if reason in ('live_state_stale', 'actuator_state_stale') and self._defer_preflight_locked(
                        raw, raw_received, transfer_key, pending_repair, pending_probe, reason):
                    return None, 'none'
                if transfer_key in self._admission_entries:
                    self._admission_cancel_locked(transfer_key,'rejected',
                        'prefix_build_failed_' + reason)
                return None, "none"
            diag["prefix_endpoint_max_abs_velocity"] = (
                self._last_prefix_endpoint_max_abs_velocity)
            diag["brake_duration_s"] = self._last_repair_brake_duration_s
            diag["brake_displacement_inf"] = self._last_brake_displacement_inf
            diag["constructed_executable_duration_s"] = (
                self._last_constructed_executable_duration_s)
            if pending_repair:
                diag["repair_nodal_motion"] = self._last_repair_nodal_motion
                view = "repair_prefix_brake_hold"
                self._repair_prefix_build_count += 1
                self._last_cycle_recovery_prefix_s = float(effective_prefix_s)
            else:
                view = "probe_prefix_brake_hold"
                self._probe_prefix_build_count += 1

        elif not prefix_mode and self.repair_motion_consistency_enabled:
            # NORMAL also starts at the measured actuator state and ends in a
            # physically feasible stop. A raw QP's last zero-velocity marker
            # cannot stop a moving actuator; certify the complete ramp and tail.
            candidate = self._repair_prefix_with_braking_tail(
                candidate, prefix_duration_s=self._duration(candidate),
                consistent_motion=True, normal_velocity_reference=True)
            if candidate is None or not candidate.points:
                self._last_source = "normal_motion_build_failed"
                if self._last_prefix_build_reason in ('live_state_stale', 'actuator_state_stale') and self._defer_preflight_locked(
                        raw, raw_received, transfer_key, pending_repair, pending_probe, self._last_prefix_build_reason):
                    return None, 'none'
                if transfer_key in self._admission_entries:
                    self._admission_cancel_locked(transfer_key, 'rejected',
                        'normal_motion_build_failed_' + self._last_prefix_build_reason)
                return None, "none"
            diag["normal_motion_consistent"] = 1
            diag["normal_nodal_motion"] = self._last_normal_nodal_motion
            diag["prefix_endpoint_max_abs_velocity"] = self._last_prefix_endpoint_max_abs_velocity
            diag["brake_duration_s"] = self._last_repair_brake_duration_s
            diag["brake_displacement_inf"] = self._last_brake_displacement_inf
            diag["constructed_executable_duration_s"] = self._last_constructed_executable_duration_s

        if getattr(self, 'enforce_execution_motion_limits', False):
            limits = effective_limits(candidate.joint_names, self.recovery_velocity_limits,
                                      self.execution_velocity_cap)
            failure = motion_limit_violation(candidate, limits, self.repair_acceleration_limits)
            if failure:
                self._last_source = 'execution_' + failure
                if transfer_key in self._admission_entries:
                    self._admission_cancel_locked(transfer_key, 'rejected', self._last_source)
                return None, 'none'

        seq = self._next_verification_seq
        self._next_verification_seq += 1
        candidate.header.seq = seq
        candidate.header.stamp = self._admission_audit_stamp_locked(now) if self.admission_enabled else now

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
            if transfer_key is not None:
                self._admission_status_locked(self._admission_entries[transfer_key], "gcdf")
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
        if transfer_key is not None:
            self._admission_status_locked(self._admission_entries[transfer_key], "vbc")
        self._verification_publish_count += 1
        self._last_source = "candidate_sent_directly_to_exact_vbc"
        self._last_verification_view = view
        self._publish_summary_locked()
        return candidate, "vbc"

    def _publish_execution_audit(self, msg, execution_stamp_ns):
        # A suffix has a fresh time origin. Carry its original committed token
        # separately; Header.seq is overwritten by ROS publishers.
        self.execution_audit_identity_pub.publish(String(data=(
            "audit_trajectory_stamp_ns={} execution_stamp_ns={}".format(
                msg.header.stamp.to_nsec(), int(execution_stamp_ns)))))
        self.execution_audit_pub.publish(msg)

    def _publish_committed(self, msg, source, age_s):
        if getattr(self, 'require_execution_preparation', False):
            return  # Geometry-verified requests await qualified issuance/application.
        if msg is None or not msg.points:
            return
        with self._lock:
            if (getattr(self, 'require_bound_regime', False) and
                    (not self._task_regime_current_locked() or
                     self._task_commit_regimes.get(msg.header.stamp.to_nsec()) != self._task_regime_identity)):
                return
            if (self._replacement_active_locked(rospy.Time.now()) or
                    msg.header.stamp.to_nsec() <= self._replacement_raw_floor_ns):
                return
            if self.task_context_enabled and not self.task_context.accepts(
                    self._task_commit_bindings.get(msg.header.stamp.to_nsec(), ""),
                    rospy.Time.now().to_nsec(), msg.header.frame_id):
                return
            if self.task_context_enabled:
                from stopping_trajectory_prefix import trajectory_sha256
                context = self.task_context.snapshot(rospy.Time.now().to_nsec())
                event = self._task_event_source.event('execution_binding', context,
                    dict(execution_stamp_ns=msg.header.stamp.to_nsec(),
                         trajectory_sha256=trajectory_sha256(msg)), rospy.Time.now().to_nsec())
                self.task_verification_pub.publish(String(data=canonical(event)))
            self.committed_pub.publish(msg)
            if self.execution_audit_enabled:
                self._publish_execution_audit(
                    copy.deepcopy(msg), msg.header.stamp.to_nsec())
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
            "handoff_rebase_enabled={} handoff_rebase_applied={} "
            "handoff_rebase_clamped={} "
            "handoff_rebase_target_shift_inf={} handoff_rebase_shift_inf={} "
            "handoff_rebase_residual_inf={} "
            "handoff_rebase_joint_state_age_ms={} "
            "probe_speed_max_before={} probe_speed_max_after={} "
            "probe_time_scale={} probe_time_scale_clamped={} "
            "probe_effective_prefix_s={} "
            "cycle_recovery_active={} cycle_recovery_prefix_index={} "
            "cycle_recovery_prefix_s={} cycle_recovery_cycle_count={} "
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
            self._diag_value(diag, "handoff_rebase_enabled"),
            self._diag_value(diag, "handoff_rebase_applied"),
            self._diag_value(diag, "handoff_rebase_clamped"),
            self._diag_value(diag, "handoff_rebase_target_shift_inf"),
            self._diag_value(diag, "handoff_rebase_shift_inf"),
            self._diag_value(diag, "handoff_rebase_residual_inf"),
            self._diag_value(diag, "handoff_rebase_joint_state_age_ms"),
            self._diag_value(diag, "probe_speed_max_before"),
            self._diag_value(diag, "probe_speed_max_after"),
            self._diag_value(diag, "probe_time_scale"),
            self._diag_value(diag, "probe_time_scale_clamped"),
            self._diag_value(diag, "probe_effective_prefix_s"),
            self._diag_value(diag, "cycle_recovery_active"),
            self._diag_value(diag, "cycle_recovery_prefix_index"),
            self._diag_value(diag, "cycle_recovery_prefix_s"),
            self._diag_value(diag, "cycle_recovery_cycle_count"),
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
        msg.data += " raw_candidate_stamp_ns={}".format(
            int((diag or {}).get("raw_candidate_stamp_ns", 0)))
        msg.data += " audited_trajectory_stamp_ns={} vbc_evidence={}".format(
            int((diag or {}).get("audited_trajectory_stamp_ns", 0)),
            (diag or {}).get("vbc_evidence", "none"))
        msg.data += " shell_recovery_evidence={} shell_recovery_onset_rejected={}".format(
            (diag or {}).get('shell_recovery_evidence', 'none'),
            (diag or {}).get('shell_recovery_onset_rejected', 0))
        msg.data += " shell_recovery_onset_detail={}".format(json.dumps(
            (diag or {}).get('shell_recovery_onset_detail', {'reason': 'not_checked'}),
            separators=(',', ':'), allow_nan=False))
        msg.data += " failure_kind={}".format((diag or {}).get('failure_kind', 'none'))
        msg.data += " motion_anchor_stamp_ns={} handoff_start_prediction={} handoff_prediction_delay_s={}".format(
            int((diag or {}).get('motion_anchor_stamp_ns',0)),
            int((diag or {}).get('handoff_start_prediction',0)),
            float((diag or {}).get('handoff_prediction_delay_s',0.)))
        msg.data += " normal_motion_consistent={} normal_nodal_motion={} repair_nodal_motion={}".format(
            (diag or {}).get('normal_motion_consistent', 0),
            (diag or {}).get('normal_nodal_motion', 0),
            (diag or {}).get('repair_nodal_motion', 0))
        msg.data += " rejection_snapshot=disabled"
        msg.data += " prepared_execution_stamp_ns={}".format(int((diag or {}).get("prepared_execution_stamp_ns", 0)))
        msg.data += " vbc_protocol_reason={}".format((diag or {}).get('vbc_protocol_reason', 'none'))
        msg.data += " path_counterexample_evidence={}".format(
            (diag or {}).get('path_counterexample_evidence', 'none'))
        msg.data += " proof_resource_evidence={}".format(
            (diag or {}).get('proof_resource_evidence', 'none'))
        msg.data += " handoff_residual_evidence={}".format(
            (diag or {}).get('handoff_residual_evidence','none'))
        prefix = (diag or {}).get('planned_prefix') or {}
        msg.data += " planned_prefix_session={} planned_prefix_revision={} planned_prefix_s={}".format(
            prefix.get('session', 'none'), prefix.get('revision', 0), prefix.get('prefix_s', 0.))
        msg.data += " occupied_volume_min_m={} occupied_geometry_unsafe_count={} occupied_geometry_malformed={}".format(
            (diag or {}).get('occupied_volume_min_m', math.nan),
            (diag or {}).get('occupied_geometry_unsafe_count', 0),
            (diag or {}).get('occupied_geometry_malformed', 0))
        key = (diag or {}).get("transfer_key")
        if key is not None and key in self._admission_entries:
            entry = self._admission_entries[key]
            msg.data += " transfer_session={} transfer_sequence={}".format(*key)
            self._admission_status_locked(entry, "terminal", msg.data)
        if getattr(self, 'task_context_enabled', False):
            entry = self._admission_entries.get(key, {})
            now = rospy.Time.now().to_nsec()
            context = self.task_context.snapshot(now)
            if context and context['candidate_binding'] == entry.get('task_context'):
                event = self._task_event_source.event('candidate_outcome', context,
                    dict(text=msg.data, task_regime=entry.get('task_regime', '')), now)
                self._task_verification_events[msg.data] = event
                while len(self._task_verification_events) > 64:
                    self._task_verification_events.popitem(last=False)
        return msg

    def _publish_verification_event(self, msg):
        self._flush_execution_preparations()
        self.verification_event_pub.publish(msg)
        if not getattr(self, 'task_context_enabled', False):
            return
        with self._lock:
            event = self._task_verification_events.pop(msg.data, None)
            now = rospy.Time.now().to_nsec()
            if (event is not None and event['valid_until_ns'] >= now and
                    self.task_context.accepts(event['task_context'], now)):
                self.task_verification_pub.publish(String(data=canonical(event)))

    def _task_control_cb(self, msg):
        try:
            with self._lock:
                now = rospy.Time.now().to_nsec()
                event = self._task_control_receiver.parse(msg.data,
                    self.task_context.snapshot(now), now, 'regime_control')
                payload = validate_regime_control(event['payload'])
                identity = regime_identity(event)
                changed = identity != self._task_regime_identity
                self._task_control_receiver.remember(event)
                self._task_control_until_ns = event['valid_until_ns']
                self._task_control_event = event
                self._task_regime_identity = identity
                self._task_control_dispatch.active = True
                try:
                    self._repair_active_cb(Bool(data=payload['ready'] and payload['mode'] == 'REPAIR'))
                    self._probe_active_cb(Bool(data=payload['ready'] and payload['mode'] == 'PROBE_NORMAL'))
                finally:
                    self._task_control_dispatch.active = False
                if changed:
                    self._admission_invalidate_locked()
                    self._execution_preparations.cancel('regime_changed')
                    self._execution_prepared_to_publish = None
                if not payload['ready']:
                    self._task_stop_committed_locked()
            self._flush_admission()
        except (ValueError, TypeError, KeyError, OverflowError):
            return

    def _task_regime_current_locked(self):
        event = self._task_control_event
        now = rospy.Time.now().to_nsec()
        return bool(event and event['stamp_ns'] <= now <= event['valid_until_ns'] and
                    event['payload']['ready'] and self.task_context.accepts(event['task_context'], now))

    def _task_recovery_cb(self, msg):
        try:
            with self._lock:
                now = rospy.Time.now().to_nsec()
                event = self._task_control_receiver.parse(msg.data,
                    self.task_context.snapshot(now), now, 'recovery_snapshot')
                payload = validate_recovery_snapshot(event['payload'], event['task_context'])
                self._task_control_receiver.remember(event)
                token = (payload['target']['observation_token'] if payload['target'] is not None else 'none')
                self._task_control_dispatch.active = True
                try:
                    self._preflight_owner_cb(String(data='active_q_vis_token='+token))
                finally:
                    self._task_control_dispatch.active = False
        except (ValueError, TypeError, KeyError, OverflowError):
            return

    def _flush_execution_preparations(self):
        if not getattr(self, 'require_execution_preparation', False):
            return
        with self._lock:
            try:
                now = rospy.Time.now().to_nsec()
                context = self.task_context.snapshot(now)
                message = self._execution_preparations.snapshot(now,
                    (context or {}).get('candidate_binding',''),
                    self._task_regime_identity if self._task_regime_current_locked() else '')
                queued = self._execution_prepared_to_publish
                self._execution_prepared_to_publish = None
                if message is not None and queued is not None and message.payload_sha256 == queued.payload_sha256:
                    self.execution_prepared_pub.publish(message)
                self.execution_prepared_status_pub.publish(String(data=canonical(self._execution_preparations.status())))
            except ValueError:
                self._execution_prepared_to_publish = None
                self._execution_preparations.cancel('preparation_clock_fault')

    def _classify_final_gcdf_unsafe(self, batch, distances, geometry_unsafe=()):
        """Return semantic blocker class for violated final-GCDF rows."""
        source_types = list(getattr(batch, "source_type", []))
        source_unknown = int(getattr(batch, "SOURCE_UNKNOWN", 0))
        source_occupied = int(getattr(batch, "SOURCE_OCCUPIED", 1))
        unknown = 0
        occupied = 0
        for i, d in enumerate(distances):
            if i not in geometry_unsafe and (not math.isfinite(d) or
                    d >= self.final_gcdf_safety_margin):
                continue
            source_type = (
                int(source_types[i])
                if i < len(source_types)
                else source_unknown)
            if source_type == source_occupied:
                occupied += 1
            else:
                unknown += 1

        if unknown > 0 and occupied > 0:
            blocker_class = "mixed"
        elif occupied > 0:
            blocker_class = "occupied"
        elif unknown > 0:
            blocker_class = "unknown"
        else:
            blocker_class = "none"
        return blocker_class, unknown, occupied

    def _build_final_gcdf_recovery_evidence(
            self, candidate, seq, batch, distances):
        """Extract earliest UNKNOWN blocker evidence from final GCDF.

        Phase E4 batches may contain both UNKNOWN and OCCUPIED hard constraints.
        Only UNKNOWN rows are valid active-sensing evidence.  OCCUPIED rows are
        physical obstacles and must never become q_vis obligations.  When the exact executable
        prefix+brake+hold is rejected, use the earliest violated timestep as
        active-sensing evidence instead of guessing from the nominal task VBC.
        """
        unsafe = []
        source_types = list(getattr(batch, "source_type", []))
        source_unknown = int(getattr(batch, "SOURCE_UNKNOWN", 0))
        for i, d in enumerate(distances):
            if (not math.isfinite(d) or
                    d >= self.final_gcdf_safety_margin or
                    i >= len(batch.original_timestep)):
                continue
            source_type = (
                int(source_types[i])
                if i < len(source_types)
                else source_unknown)
            if source_type != source_unknown:
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
        rejection_feedback_to_publish = None

        with self._lock:
            self._admission_expire_locked()
            if (self._gcdf_outstanding is None or
                    self._gcdf_outstanding_sent is None):
                return

            expected_stamp = self._gcdf_outstanding.header.stamp
            dt = abs((msg.header.stamp - expected_stamp).to_sec())
            if (dt > self.final_gcdf_stamp_tolerance_s or
                    (self.admission_enabled and msg.header.stamp.to_nsec() != expected_stamp.to_nsec())):
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
            geometry_unsafe = []
            geometry_min = math.inf
            try:
                geometry_unsafe, geometry_min = occupied_violations(
                    msg, self.final_occupied_resolution_m,
                    self.final_occupied_hard_margin_m)
            except (ValueError, TypeError, AttributeError, OverflowError):
                malformed = True
            if diag is None:
                diag = {}
            diag['occupied_volume_min_m'] = geometry_min
            diag['occupied_geometry_unsafe_count'] = len(geometry_unsafe)
            diag['occupied_geometry_malformed'] = int(malformed)

            gcdf_safe = (
                not malformed and not geometry_unsafe and
                (int(msg.num_pairs) == 0 or
                 min_d >= self.final_gcdf_safety_margin))
            restore=diag.get('geometric_restore')
            geometry_request=False
            if restore is not None and was_repair and not was_probe and not malformed and not geometry_unsafe:
                try:
                    geometry_request=permits_unknown_conflicts(restore,msg,distances,self.final_gcdf_safety_margin)
                except (ValueError,TypeError,AttributeError):
                    geometry_request=False
            if gcdf_safe or geometry_request:
                diag['final_gcdf_passed'] = bool(gcdf_safe)
                diag['geometry_override_pending'] = bool(geometry_request)
                self._final_gcdf_safe_count += 1
                # GCDF certifies the q-trajectory geometry; refresh only the
                # epoch before exact VBC so its temporal audit starts now.
                candidate.header.stamp = self._admission_audit_stamp_locked(now) if self.admission_enabled else now
                self._outstanding = candidate
                self._outstanding_sent = now
                self._outstanding_seq = seq
                self._outstanding_dispatch_cycle = self._selector_cycle_count
                self._outstanding_repair = was_repair
                self._outstanding_probe = was_probe
                self._outstanding_view = view
                self._outstanding_raw_received = raw_received
                self._outstanding_diag = copy.deepcopy(diag)
                key = (diag or {}).get("transfer_key")
                if key is not None:
                    self._admission_status_locked(self._admission_entries[key], "vbc")
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
                    gcdf_blocker_class,
                    gcdf_unknown_count,
                    gcdf_occupied_count,
                ) = self._classify_final_gcdf_unsafe(msg, distances, geometry_unsafe)
                self._last_final_gcdf_blocker_class = gcdf_blocker_class
                self._last_final_gcdf_unsafe_unknown_count = gcdf_unknown_count
                self._last_final_gcdf_unsafe_occupied_count = gcdf_occupied_count

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

                gcdf_safety_gate = (
                    "gcdf_{}".format(gcdf_blocker_class)
                    if gcdf_blocker_class in ("unknown", "occupied", "mixed")
                    else "gcdf")
                if diag is None:
                    diag = {}
                # This is the GCDF-audited executable, not the future VBC epoch.
                diag['audited_trajectory_stamp_ns'] = candidate.header.stamp.to_nsec()
                if was_repair and not was_probe:
                    try:
                        points, overflow = rejection_points(msg, candidate, self.final_gcdf_safety_margin)
                        raw_stamp = int(diag.get('raw_candidate_stamp_ns', 0))
                        if raw_stamp > 0 and (points or overflow):
                            rejection_feedback_to_publish = CollisionCDFRejectionFeedback(
                                header=copy.deepcopy(candidate.header),
                                raw_candidate_stamp=rospy.Time(raw_stamp // 1000000000, raw_stamp % 1000000000),
                                point_flat=points, overflow=overflow)
                    except (ValueError, OverflowError) as exc:
                        rospy.logwarn('GCDF rejection feedback unavailable: %s', exc)
                event_to_publish = self._make_verification_event(
                    seq, "unsafe", False, age, view,
                    safety_gate=gcdf_safety_gate, diag=diag)
                dispatched, route = self._dispatch_pending_locked(now)
                if dispatched is not None:
                    self._post_cert_pending_dispatch_count += 1
                if route == "gcdf":
                    next_gcdf_to_publish = dispatched
                elif route == "vbc":
                    verification_to_publish = dispatched

            self._publish_summary_locked()

        if rejection_feedback_to_publish is not None:
            self.gcdf_rejection_feedback_pub.publish(rejection_feedback_to_publish)
        if gcdf_recovery_trajectory_to_publish is not None:
            self.final_gcdf_recovery_trajectory_pub.publish(
                gcdf_recovery_trajectory_to_publish)
        if gcdf_recovery_event_to_publish is not None:
            self.final_gcdf_recovery_event_pub.publish(
                gcdf_recovery_event_to_publish)
        if verification_to_publish is not None:
            self._publish_verification(verification_to_publish)
        if event_to_publish is not None:
            self._publish_verification_event(event_to_publish)
        if next_gcdf_to_publish is not None:
            self.final_gcdf_query_pub.publish(next_gcdf_to_publish)
        self._flush_admission()

    def _shell_commit_valid_locked(self, evidence, stamp, now, diagnostics):
        try:
            value=json.loads(evidence)
            if not isinstance(value,dict):
                diagnostics['reason']='malformed_evidence_or_state'
                return False
            restore=(self._outstanding_diag or {}).get('geometric_restore')
            if restore is not None and (value.get('schema')!=2 or value.get('contract')!='geometric_restore' or
                                        value.get('restoration_points')!=restore['points']):
                diagnostics['reason']='geometric_restore_certificate_mismatch'
                return False
            if value.get('schema')==2:
                first=self._outstanding.points[0]
                if (value['initial_q']!=list(first.positions) or
                        value['initial_velocity']!=list(first.velocities) or
                        abs(value['motion_horizon_s']-self._outstanding.points[-1].time_from_start.to_sec())>1e-12):
                    diagnostics['reason']='geometric_certificate_motion_mismatch'
                    return False
        except (ValueError,KeyError,TypeError,IndexError):
            diagnostics['reason']='malformed_evidence_or_state'
            return False
        actuator=None
        if 0<=now.to_nsec()-getattr(self,'_latest_actuator_stamp_ns',0)<=50000000:
            actuator=getattr(self,'_latest_actuator_velocity',None)
        gain=getattr(self,'tracking_feedback_gain',1.)
        accepted=shell_commit_check(evidence,stamp,now.to_nsec(),self._latest_measured_stamp_ns,
            list(self._outstanding.joint_names),self._latest_measured_q_by_name,
            self._latest_measured_velocity_by_name,self.repair_acceleration_limits,
            self.repair_brake_dt_s,self.repair_brake_max_steps,self.initial_shell_execution_margin_m,
            diagnostics=diagnostics,feedback_gain=gain,actuator_velocity=actuator)
        restore=(self._outstanding_diag or {}).get('geometric_restore')
        if restore is not None and value.get('schema') == 2 and 'certified_residual_m' in diagnostics:
            scheduling = copy.deepcopy(value)
            # Local scheduling metadata only; never serialize it into proof
            # evidence or alter the independent admission formula.
            anchor = (self._outstanding_diag or {}).get('motion_anchor_stamp_ns', stamp)
            scheduling['handoff_latency_s'] = max(0., (now.to_nsec()-anchor)*1e-9)
            self._handoff_certificate = (self._handoff_scope(self._outstanding_diag), scheduling)
        if diagnostics.get('failure_kind')=='geometric_certificate_state_changed' and restore is not None:
            try:
                e=json.loads(evidence)
                record=dict(schema=1,kind='handoff_residual_insufficient',trajectory_stamp_ns=str(stamp),
                    raw_candidate_stamp_ns=str((self._outstanding_diag or {}).get('raw_candidate_stamp_ns',0)),
                    mode_epoch=restore['mode_epoch'],observation_token=restore['observation_token'],
                    stage=e['residual_stage'],floor_m=e['residual_floor_m'],point=e['residual_point'],
                    ordinary_unknown=e['residual_ordinary_unknown'],interval_index=e['residual_interval_index'],
                    required_residual_m=diagnostics['motion_bound_m'],certified_residual_m=e['certified_residual_m'],
                    requires_reconstruction=diagnostics['requires_reconstruction'],
                    initial_state_changed=diagnostics['initial_state_changed'])
                diagnostics['handoff_residual_evidence']=record
            except (KeyError,ValueError,TypeError):pass
        return accepted

    def _global_summary_cb(self, msg):
        if msg is None:
            return
        fields = _tokens(msg.data)
        source = fields.get("trajectory_source", "")
        if source not in ("bootstrap", "predicted"):
            return
        try:
            trajectory_stamp_ns = int(fields.get("trajectory_stamp_ns", "0"))
        except Exception:
            trajectory_stamp_ns = 0
        violation = _as_bool(fields.get("has_violation"))
        if violation is None:
            return

        # The verifier echoes explicit business identity after exact content
        # binding. A reply from another admission session cannot decide this
        # candidate, even if its stamp was copied or a publisher restarted.
        if self.initial_shell_recovery_enabled and source == 'predicted':
            with self._lock:
                identity_mismatch = False
                if (self._outstanding is not None and
                        trajectory_stamp_ns == self._outstanding.header.stamp.to_nsec()):
                    key = (self._outstanding_diag or {}).get('transfer_key')
                    expected = key if key is not None else (self._admission_boot_id, self._outstanding_seq)
                    try:
                        actual = (fields['context_request_session'], int(fields['context_request_sequence']))
                    except (KeyError, TypeError, ValueError):
                        actual = None
                    identity_mismatch = actual != expected
                    contract = (self._outstanding_diag or {}).get('planned_prefix')
                    if contract is not None:
                        try:
                            echoed = (fields['context_prefix_session'], int(fields['context_prefix_revision']),
                                      float(fields['context_prefix_s']))
                        except (KeyError, TypeError, ValueError):
                            echoed = None
                        identity_mismatch = identity_mismatch or not self._prefix_current_locked(contract) or (
                            echoed != (contract['session'], contract['revision'], contract['prefix_s']))
            if identity_mismatch:
                self._verification_protocol_cb(String(data='trajectory_stamp_ns={} state=invalid '
                    'reason=summary_business_identity_mismatch'.format(trajectory_stamp_ns)))
                return

        now = rospy.Time.now()
        refresh=None
        if source=='predicted' and not violation and self.initial_shell_recovery_enabled:
            with self._lock:
                detail={}
                diag=self._outstanding_diag or {}
                if (self._outstanding is not None and self._outstanding.header.stamp.to_nsec()==trajectory_stamp_ns and
                        self._outstanding_repair and not self._outstanding_probe and
                        (diag.get('final_gcdf_passed',False) or diag.get('geometry_override_pending',False)) and
                        not diag.get('shell_refresh_requested',False) and
                        0<=(now-self._outstanding_sent).to_sec()<self.verification_timeout_s):
                    shell_valid=self._shell_commit_valid_locked(
                        fields.get('shell_recovery_evidence','none'),trajectory_stamp_ns,now,detail)
                    rospy.loginfo('[SHELL_HANDOFF] %s',json.dumps(dict(
                        audit_stamp_ns=str(trajectory_stamp_ns),receive_ros_ns=str(now.to_nsec()),
                        raw_stamp_ns=str(diag.get('raw_candidate_stamp_ns','none')),
                        accepted=bool(shell_valid),diagnostics=detail),separators=(',',':')))
                    if (detail.get('failure_kind') in ('certificate_state_stale', 'map_snapshot_stale',
                                                     'geometric_certificate_state_changed') and
                            not detail.get('requires_reconstruction',False) and
                            detail.get('live_state_age_ms', 1e9) <= 50. and
                            detail.get('live_state_age_ms', -1.) >= 0. and
                            json.loads(fields.get('shell_recovery_evidence', '{}')).get('schema') == 2):
                        self._outstanding_diag['shell_refresh_requested']=True
                        self._outstanding_diag['shell_required_residual_m']=detail.get('motion_bound_m',0.)
                        try:
                            ended=int(json.loads(fields['shell_recovery_evidence'])['certificate_end_ros_ns'])
                            self._outstanding_diag['shell_reply_allowance_s']=min(.01,max(.001,(now.to_nsec()-ended)/1e9+.001))
                        except (KeyError,ValueError,TypeError):pass
                        refresh=copy.deepcopy(self._outstanding)
            if refresh is not None:
                # Preserve identity, timeout, candidate and remaining native FK
                # ledger. Publish after releasing the admission mutex.
                self._publish_verification(refresh)
                return
        committed_to_publish = None
        event_to_publish = None
        verification_to_publish = None
        gcdf_to_publish = None
        with self._lock:
            if getattr(self, "admission_enabled", False):
                self._admission_expire_locked()
            self._selector_cycle_count += 1
            self._last_selector_cycle_time = now

            if self._outstanding is None or self._outstanding_sent is None:
                # C5.36: this is now fallback-only. Normal admission happens
                # directly in _raw_cb(), and completion paths dispatch pending
                # latest-only work immediately.
                dispatched, route = self._dispatch_pending_locked(now)
                if dispatched is not None:
                    self._selector_fallback_dispatch_count += 1
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
                else:
                    expected_stamp_ns = int(
                        self._outstanding.header.stamp.to_nsec())
                    if (trajectory_stamp_ns <= 0 or
                            trajectory_stamp_ns != expected_stamp_ns):
                        # C5.34 transaction identity: periodic VBC refreshes may
                        # still publish summaries for an older cached predicted
                        # trajectory. They must never decide a newer outstanding
                        # executable candidate.
                        self._exact_vbc_stamp_miss_count += 1
                        self._last_source = (
                            "exact_vbc_stamp_mismatch_ignored")
                        self._publish_summary_locked()
                        return
                    if (self._selector_cycle_count <=
                            self._outstanding_dispatch_cycle):
                        self._publish_summary_locked()
                        return
                    contract = (self._outstanding_diag or {}).get('planned_prefix')
                    if contract is not None and not self._prefix_current_locked(contract):
                        key = (self._outstanding_diag or {}).get('transfer_key')
                        if key in self._admission_entries:
                            self._admission_cancel_locked(key)
                        return

                    verification_age = max(
                        0.0, (now - self._outstanding_sent).to_sec())
                    self._last_exact_vbc_roundtrip_ms = 1000.0 * verification_age
                    raw_received = self._outstanding_raw_received
                    diag = copy.deepcopy(self._outstanding_diag)
                    if diag is None:
                        diag = {}
                    # Carry the exact accepted summary, not the latest periodic
                    # refresh for this trajectory or a nearby raw candidate.
                    diag["audited_trajectory_stamp_ns"] = expected_stamp_ns
                    diag["vbc_evidence"] = fields.get("vbc_evidence", "none")
                    diag['shell_recovery_evidence'] = fields.get('shell_recovery_evidence', 'none')
                    if getattr(self, 'initial_shell_recovery_enabled', False) and not violation:
                        try:
                            count = int(fields['shell_recovery_count'])
                            if count<0 or count>32:
                                raise ValueError('Invalid recovery count')
                            if count:
                                allowed = bool(self._outstanding_repair and not self._outstanding_probe and
                                    self._outstanding_view=='repair_prefix_brake_hold' and
                                    (diag.get('final_gcdf_passed', False) or diag.get('geometry_override_pending', False)))
                                onset_detail = dict(reason='ineligible_recovery_view')
                                violation = not (allowed and self._shell_commit_valid_locked(
                                    diag['shell_recovery_evidence'],expected_stamp_ns,now,onset_detail))
                                diag['shell_recovery_onset_detail'] = onset_detail
                                if 'handoff_residual_evidence' in onset_detail:
                                    diag['handoff_residual_evidence']=json.dumps(onset_detail['handoff_residual_evidence'],
                                        separators=(',',':'),allow_nan=False)
                                diag['shell_recovery_onset_rejected'] = int(violation)
                                if violation:
                                    diag['failure_kind'] = onset_detail.get('failure_kind', onset_detail['reason'])
                            elif diag['shell_recovery_evidence'] != 'none':
                                violation = True
                            if diag.get('geometric_restore') is not None and count == 0:
                                violation = True
                                diag['failure_kind']='missing_geometric_restore_certificate'
                        except (KeyError, ValueError, TypeError, OverflowError):
                            violation = True
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
                        state_failure = diag.get('failure_kind') in (
                            'certificate_state_stale', 'live_state_stale', 'state_order_invalid',
                            'map_snapshot_stale', 'geometric_certificate_state_changed', 'missing_actuator_state')
                        protocol_failure = diag.get('failure_kind') in (
                            'geometric_certificate_motion_mismatch', 'geometric_restore_certificate_mismatch',
                            'missing_geometric_restore_certificate', 'malformed_evidence_or_state',
                            'identity_or_shape', 'oversized_evidence', 'joint_dimensions',
                            'brake_contract', 'invalid_geometric_certificate', 'invalid_measured_stop_contract', 'point_identity',
                            'nonfinite_clearance', 'nonfinite_or_missing_state', 'nonfinite_floor', 'invalid_floor')
                        if protocol_failure:
                            self._verification_protocol_invalid_count += 1
                        elif state_failure:
                            self._verification_timeout_count += 1
                        else:
                            self._verification_unsafe_count += 1
                        if not state_failure and not protocol_failure and was_repair and view == "repair_prefix_brake_hold":
                            self._repair_prefix_unsafe_count += 1
                            if self._cycle_recovery_active:
                                attempted_prefix = (
                                    float(diag.get(
                                        "cycle_recovery_prefix_s",
                                        self.repair_execution_prefix_s))
                                    if diag is not None
                                    else self.repair_execution_prefix_s)
                                self._cycle_recovery_backoff_locked(
                                    attempted_prefix)
                        if not state_failure and not protocol_failure and was_probe and view == "probe_prefix_brake_hold":
                            self._probe_prefix_unsafe_count += 1
                        self._last_verification_result = "protocol_invalid" if protocol_failure else "timeout" if state_failure else "unsafe"
                        self._last_source = "candidate_rejected_state_evidence" if state_failure else "candidate_rejected_vbc_unsafe"
                        event_to_publish = self._make_verification_event(
                            seq, self._last_verification_result, False, verification_age, view,
                            diag=diag, safety_gate="certificate_protocol" if protocol_failure else "state_evidence" if state_failure else "vbc")
                        self._sync_repair_prefix_locked()
                    else:
                        was_normal = (not was_probe and not was_repair)
                        # Execute the exact candidate that passed GCDF + VBC.
                        # Verification wall time is not robot execution time;
                        # suffixing a REPAIR candidate here moves its start
                        # away from the measured joint state and repeatedly
                        # trips the start-continuity gate.
                        committed = copy.deepcopy(candidate)
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
                            diag["precommit_suffix_phase_s"] = 0.0
                            diag["precommit_suffix_start_shift_inf"] = (
                                self._position_shift_inf(
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

                            start_gate_limit = None
                            start_gate_source = None
                            if was_probe:
                                start_gate_limit = (
                                    self.probe_commit_start_mismatch_max)
                                start_gate_source = (
                                    "probe_start_continuity_rejected_replan")
                            elif was_repair:
                                start_gate_limit = (
                                    self.repair_commit_start_mismatch_max)
                                start_gate_source = (
                                    "repair_start_continuity_rejected_replan")
                            elif was_normal:
                                start_gate_limit = (
                                    self.normal_commit_start_mismatch_max)
                                start_gate_source = (
                                    "normal_start_continuity_rejected_replan")

                            tracker_stopping = getattr(self, '_handoff_tracker_fields', {}).get('execution_state') == 'STOPPING'
                            if tracker_stopping:
                                start_gate_source = 'tracker_stopping_before_commit'
                            if tracker_stopping or (start_gate_limit is not None and
                                    (not math.isfinite(commit_start_mismatch) or
                                     commit_start_mismatch >
                                     start_gate_limit)):
                                # Fail closed: never mutate a certified
                                # trajectory after safety audit. Ask the planner
                                # for a fresh measured-state candidate instead.
                                if was_probe:
                                    self._probe_start_continuity_reject_count += 1
                                self._verification_timeout_count += 1
                                self._last_verification_result = "timeout"
                                self._last_source = start_gate_source
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
                                    if self._cycle_recovery_active:
                                        self._cycle_recovery_safe_commit_count += 1
                                        self._cycle_recovery_last_reason = (
                                            "certified_frontier_commit")
                                if was_probe and view == "probe_prefix_brake_hold":
                                    self._probe_prefix_safe_count += 1
                                self._last_verification_result = "safe"
                                committed.header.seq = seq
                                committed.header.stamp = (rospy.Time(0, max(now.to_nsec(), self._last_execution_stamp_ns + 1))
                                    if getattr(self, "admission_enabled", False) else now)
                                execution_stamp_ns = int(
                                    committed.header.stamp.to_nsec())
                                self._last_execution_stamp_ns = execution_stamp_ns
                                if self.require_execution_preparation:
                                    key = (diag or {}).get('transfer_key')
                                    entry = self._admission_entries.get(key, {})
                                    prepare_now = rospy.Time.now().to_nsec()
                                    context = self.task_context.snapshot(prepare_now)
                                    if context is None or not self._task_regime_current_locked():
                                        self._last_source = 'execution_preparation_context_expired'
                                        event_to_publish = self._make_verification_event(
                                            seq, 'timeout', False, verification_age, view,
                                            safety_gate='execution_preparation', diag=diag)
                                    else:
                                        prepared = self._execution_preparations.prepare(committed,
                                            context['candidate_binding'],entry.get('task_regime',''),key,
                                            entry['digest'],seq,prepare_now,
                                            min(prepare_now+250000000,context['valid_until_ns'],
                                                self._task_control_until_ns))
                                        self._execution_prepared_to_publish = prepared
                                        self._execution_prepared_count += 1
                                        self._last_source = 'geometry_verified_waiting_stopping_qualification'
                                        self._last_verification_result = 'safe_awaiting_execution'
                                        diag = dict(diag or {}, prepared_execution_stamp_ns=execution_stamp_ns)
                                        event_to_publish = self._make_verification_event(
                                            seq, 'safe', False, verification_age, view,
                                            safety_gate='execution_preparation', diag=diag)
                                else:
                                    if self.task_context_enabled:
                                        key = (diag or {}).get("transfer_key")
                                        binding = self._admission_entries.get(key, {}).get("task_context", "")
                                        self._task_commit_bindings[execution_stamp_ns] = binding
                                        self._task_commit_regimes[execution_stamp_ns] = self._admission_entries.get(key, {}).get('task_regime', '')
                                        while len(self._task_commit_bindings) > 64:
                                            retired, _ = self._task_commit_bindings.popitem(last=False)
                                            self._task_commit_regimes.pop(retired, None)
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
                    if dispatched is not None:
                        self._post_cert_pending_dispatch_count += 1
                    if route == "gcdf":
                        gcdf_to_publish = dispatched
                    elif route == "vbc":
                        verification_to_publish = dispatched
                    self._publish_summary_locked()

            if committed_to_publish is not None:
                self._publish_committed(
                    committed_to_publish, "candidate_verified_safe_committed", 0.0)
        if event_to_publish is not None:
            self._publish_verification_event(event_to_publish)
        if verification_to_publish is not None:
            self._publish_verification(verification_to_publish)
        if gcdf_to_publish is not None:
            self.final_gcdf_query_pub.publish(gcdf_to_publish)
        if getattr(self, "admission_enabled", False):
            self._flush_admission()

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
            self._admission_expire_locked()
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
                        safety_gate="gcdf", diag=self._gcdf_outstanding_diag)
                    dispatched, route = self._dispatch_pending_locked(now)
                    if dispatched is not None:
                        self._post_cert_pending_dispatch_count += 1
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
                        seq, "timeout", False, age, view, diag=self._outstanding_diag)
                    dispatched, route = self._dispatch_pending_locked(now)
                    if dispatched is not None:
                        self._post_cert_pending_dispatch_count += 1
                    if route == "gcdf":
                        next_gcdf_to_publish = dispatched
                    elif route == "vbc":
                        next_verification_to_publish = dispatched

            if (self.execution_audit_enabled and
                    self._committed_master is not None and
                    self._committed_received is not None):
                execution_audit_age = (
                    now - self._committed_received).to_sec()
                master_duration = self._duration(self._committed_master)
                # Tracker may pause/discard an execution on a safety hold.
                # Wall time since commit is not evidence of executed progress.
                # A missing/stale/mismatched tracker token retains the complete
                # path rather than discarding unexecuted geometry by elapsed time.
                phase = 0.0
                state = self._execution_tracker_state
                received = self._execution_tracker_received
                if (state is not None and received is not None and
                        state[0] == self._committed_master.header.stamp.to_nsec() and
                        0.0 <= (now - received).to_sec() <= 0.15):
                    phase = min(state[1], master_duration)
                execution_audit_to_publish = self._suffix_from_phase(
                    self._committed_master, phase)
                execution_audit_token = (
                    self._committed_master.header.stamp.to_nsec())

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
            self._publish_execution_audit(
                execution_audit_to_publish, execution_audit_token)
            with self._lock:
                self._execution_audit_publish_count += 1
                self._publish_summary_locked()
        if continuation_to_publish is not None:
            self._publish_committed(
                continuation_to_publish, "committed_continuation", continuation_age)
        if timeout_event is not None:
            self._publish_verification_event(timeout_event)
        if gcdf_timeout_event is not None:
            self._publish_verification_event(gcdf_timeout_event)
        if next_gcdf_to_publish is not None:
            self.final_gcdf_query_pub.publish(next_gcdf_to_publish)
        if next_verification_to_publish is not None:
            self._publish_verification(next_verification_to_publish)
        self._flush_admission()

    def _publish_summary_locked(self):
        msg = String()
        msg.data = " ".join([
            "source={}".format(self._last_source),
            "raw_input_count={}".format(self._raw_input_count),
            "pending_candidate={}".format(int(self._pending_raw is not None)),
            "preflight_waiting={}".format(int(self._preflight_wait_key is not None)),
            "preflight_wait_count={}".format(self._preflight_wait_count),
            "preflight_resume_count={}".format(self._preflight_resume_count),
            "preflight_reason={}".format(self._preflight_reason),
            "handoff_wait_count={}".format(self._handoff_wait_count),
            "handoff_resume_count={}".format(self._handoff_resume_count),
            "preflight_measured_age_ns={}".format(self._preflight_measured_age_ns),
            "preflight_actuator_age_ns={}".format(self._preflight_actuator_age_ns),
            "pending_replace_count={}".format(self._pending_replace_count),
            "raw_immediate_dispatch_count={}".format(
                self._raw_immediate_dispatch_count),
            "raw_busy_buffer_count={}".format(
                self._raw_busy_buffer_count),
            "post_cert_pending_dispatch_count={}".format(
                self._post_cert_pending_dispatch_count),
            "selector_fallback_dispatch_count={}".format(
                self._selector_fallback_dispatch_count),
            "selector_cycle_count={}".format(self._selector_cycle_count),
            "repair_active={}".format(int(self._repair_active)),
            "probe_active={}".format(int(self._probe_active)),
            "repair_prefix_enabled={}".format(int(self.repair_prefix_verification_enabled)),
            "cycle_recovery_enabled={}".format(int(self.cycle_recovery_enabled)),
            "cycle_recovery_active={}".format(int(self._cycle_recovery_active)),
            "cycle_recovery_signature={}".format(self._cycle_recovery_signature),
            "cycle_recovery_stack={}".format(self._cycle_recovery_stack),
            "cycle_recovery_earliest_layer_ids={}".format(
                self._cycle_recovery_earliest_layer_ids),
            "cycle_recovery_cycle_count={}".format(
                self._cycle_recovery_cycle_count),
            "cycle_recovery_prefix_index={}".format(
                self._cycle_recovery_prefix_index),
            "cycle_recovery_prefix_s={:.6f}".format(
                self._last_cycle_recovery_prefix_s),
            "cycle_recovery_activation_count={}".format(
                self._cycle_recovery_activation_count),
            "cycle_recovery_clear_count={}".format(
                self._cycle_recovery_clear_count),
            "cycle_recovery_backoff_count={}".format(
                self._cycle_recovery_backoff_count),
            "cycle_recovery_safe_commit_count={}".format(
                self._cycle_recovery_safe_commit_count),
            "cycle_recovery_floor_unsafe_count={}".format(
                self._cycle_recovery_floor_unsafe_count),
            "cycle_recovery_last_reason={}".format(
                self._cycle_recovery_last_reason),
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
            "final_gcdf_blocker_class={}".format(
                self._last_final_gcdf_blocker_class),
            "final_gcdf_unsafe_unknown_count={}".format(
                self._last_final_gcdf_unsafe_unknown_count),
            "final_gcdf_unsafe_occupied_count={}".format(
                self._last_final_gcdf_unsafe_occupied_count),
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
            "verification_protocol_invalid_count={}".format(self._verification_protocol_invalid_count),
            "last_vbc_protocol_reason={}".format(self._last_vbc_protocol_reason),
            "exact_vbc_stamp_miss_count={}".format(
                self._exact_vbc_stamp_miss_count),
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
        if self.admission_enabled:
            msg.data += " admission_enabled=1 admission_receive_count={} admission_duplicate_count={} admission_expiry_count={} receiver_boot_id={}".format(
                self._admission_receive_count, self._admission_duplicate_count,
                self._admission_expiry_count, self._admission_boot_id)
            self._admission_summary_pending = msg
        else:
            self.summary_pub.publish(msg)

    def _publish_summary(self):
        with self._lock:
            self._publish_summary_locked()
        self._flush_admission()


def main():
    rospy.init_node("optimized_trajectory_continuity")
    OptimizedTrajectoryContinuityNode()
    rospy.spin()


if __name__ == "__main__":
    main()
