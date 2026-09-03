#!/usr/bin/env python3
"""Verified CAREPlanner candidate/execution regime manager.

C4.4 baseline semantics:
    NORMAL -> REPAIR -> PROBE_NORMAL -> NORMAL
and a safe committed REPAIR candidate immediately enters PROBE_NORMAL.

C4.7 optional visibility-acquisition gate:
    a safe REPAIR candidate is only permission to execute the active-sensing
    detour.  It does NOT prove that the visibility objective has been acquired.
    While the gate is enabled, safe repair candidates may keep committing and the
    planner stays in REPAIR.  PROBE_NORMAL is entered only after the real
    confidence/visibility layer reports acquisition_complete=True after first
    publishing False in the current REPAIR episode.

Candidate certification and committed-execution safety remain separate. A C5.8
candidate may be rejected by the final executable GCDF gate or by exact VBC.
PROBE success is NOT counted at commit time. It is counted only after the
tracker confirms that the certified task-prefix duration for the exact committed
header.stamp execution token has actually elapsed. The certified brake+hold tail
remains the fail-safe while the next probe is planned/certified. ROS Header.seq
is diagnostic only.
"""

import math
import re
import threading

import rospy
from std_msgs.msg import Bool, Float64, String


_TOKEN_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


def _tokens(text):
    return {k: v for k, v in _TOKEN_RE.findall(text or "")}


def _as_bool(v):
    if v in ("1", "true", "True"):
        return True
    if v in ("0", "false", "False"):
        return False
    return None


def _as_int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _as_float(v, default=math.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class C44VerifiedRegimeManager:
    NORMAL = "NORMAL"
    REPAIR = "REPAIR"
    PROBE_NORMAL = "PROBE_NORMAL"

    def __init__(self):
        self._lock = threading.RLock()

        self.rate = float(rospy.get_param("~rate", 20.0))
        self.candidate_unsafe_required = int(rospy.get_param(
            "~candidate_unsafe_required", 2))
        self.execution_unsafe_required = int(rospy.get_param(
            "~execution_unsafe_required", 2))
        legacy_probe_required = int(rospy.get_param(
            "~probe_safe_commits_required", 3))
        self.probe_completed_prefixes_required = int(rospy.get_param(
            "~probe_completed_prefixes_required", legacy_probe_required))
        self.input_timeout = float(rospy.get_param("~input_timeout", 0.25))
        self.clear_pulse_s = float(rospy.get_param("~clear_pulse_s", 0.12))
        self.probe_ignore_s = float(rospy.get_param("~probe_ignore_s", 0.16))

        self.repair_completion_gate_enabled = bool(rospy.get_param(
            "~repair_completion_gate_enabled", False))
        self.repair_completion_topic = str(rospy.get_param(
            "~repair_completion_topic",
            "/care_planner/active_sensing/visibility_acquisition_complete"))

        if self.rate <= 0.0:
            raise ValueError("~rate must be positive")
        if self.candidate_unsafe_required < 1 or self.execution_unsafe_required < 1:
            raise ValueError("unsafe streak requirements must be >= 1")
        if self.probe_completed_prefixes_required < 1:
            raise ValueError("probe_completed_prefixes_required must be >= 1")
        if min(self.input_timeout,
               self.clear_pulse_s, self.probe_ignore_s) <= 0.0:
            raise ValueError("timeouts/pulses must be positive")

        self.candidate_outcome_topic = str(rospy.get_param(
            "~candidate_outcome_topic", "/care_planner/verification_outcome"))
        self.execution_summary_topic = str(rospy.get_param(
            "~execution_summary_topic", "/care_planner/execution_vbc/summary"))
        self.tracker_summary_topic = str(rospy.get_param(
            "~tracker_summary_topic", "/care_planner/execution/tracker_summary"))
        self.probe_single_flight_summary_topic = str(rospy.get_param(
            "~probe_single_flight_summary_topic",
            "/care_planner/execution/probe_single_flight_summary"))
        self.replan_request_topic = str(rospy.get_param(
            "~replan_request_topic", "/care_planner/local_planner/replan_request"))
        self.committed_trajectory_topic = str(rospy.get_param(
            "~committed_trajectory_topic", "/care_planner/committed_trajectory"))
        self.execution_ready_topic = str(rospy.get_param(
            "~execution_ready_topic", "/care_planner/execution/ready"))
        self.physical_deadline_topic = str(rospy.get_param(
            "~physical_deadline_topic",
            "/care_planner/active_sensing/visibility_waypoint_deadline_rolling"))
        self.effective_deadline_topic = str(rospy.get_param(
            "~effective_deadline_topic",
            "/care_planner/active_sensing/visibility_waypoint_deadline_effective"))
        self.trigger_topic = str(rospy.get_param(
            "~trigger_topic",
            "/care_planner/execution/predicted_vbc_recovery_triggered"))
        self.clear_topic = str(rospy.get_param(
            "~clear_topic", "/care_planner/execution/predicted_vbc_recovery_clear"))
        self.verification_hold_topic = str(rospy.get_param(
            "~verification_hold_topic",
            "/care_planner/execution/predicted_vbc_verification_hold"))
        self.probe_active_topic = str(rospy.get_param(
            "~probe_active_topic",
            "/care_planner/c4_4/probe_active"))
        self.summary_topic = str(rospy.get_param(
            "~summary_topic", "/care_planner/c4_4/regime_summary"))
        self.task_infeasible_topic = str(rospy.get_param(
            "~task_infeasible_topic",
            "/care_planner/local_planner/task_infeasible"))
        self.task_uncertified_topic = str(rospy.get_param(
            "~task_uncertified_topic",
            "/care_planner/local_planner/task_uncertified"))
        self.visibility_waypoint_active_topic = str(rospy.get_param(
            "~visibility_waypoint_active_topic",
            "/care_planner/active_sensing/visibility_waypoint_active"))
        self.candidate_vbc_summary_topic = str(rospy.get_param(
            "~candidate_vbc_summary_topic",
            "/care_planner/candidate_vbc/summary"))
        self.force_vbc_bootstrap_topic = str(rospy.get_param(
            "~force_vbc_bootstrap_topic",
            "/care_planner/trajectory_risk/force_bootstrap"))
        self.normal_task_replan_topic = str(rospy.get_param(
            "~normal_task_replan_topic",
            "/care_planner/local_planner/normal_task_replan_request"))

        self.state = self.NORMAL
        self.execution_ready = False
        self.execution_ready_time = None

        self.candidate_outcome_time = None
        self.last_candidate_seq = 0
        self.last_candidate_result = "none"
        self.last_candidate_safety_gate = "none"
        self.last_candidate_verification_view = "none"
        self.candidate_outcome_pending_gate_replay = False
        self.candidate_gate_replay_count = 0
        self.candidate_last_unsafe = False
        self.candidate_unsafe_streak = 0
        self.candidate_unsafe_count = 0
        self.candidate_safe_count = 0
        self.candidate_timeout_count = 0
        self.candidate_event_count = 0
        # C5.22: NORMAL VBC-unsafe hysteresis must keep producing fresh
        # candidates; otherwise streak=1 can deadlock forever with no commit.
        self.normal_unsafe_retry_count = 0
        self.normal_unsafe_confirmed_repair_count = 0

        self.execution_summary_time = None
        self.execution_last_unsafe = False
        self.execution_unsafe_streak = 0
        self.execution_event_latched = False
        self.execution_safety_event_count = 0
        self.execution_safe_count = 0

        self.tracker_summary_time = None
        self.tracker_active = False
        self.tracker_complete = False
        self.tracker_execution_stamp_ns = 0
        self.tracker_source = "none"

        # C5.17 decision-level PROBE single-flight. Candidate admission is
        # serialized by probe_single_flight_gate_node; mirror its phase here so
        # later planner failures cannot preempt an already certified/in-flight
        # PROBE decision.
        self.probe_single_flight_phase = "IDLE"
        self.probe_single_flight_execution_stamp_ns = 0
        self.probe_single_flight_reason = "startup"
        self.probe_single_flight_summary_time = None
        self.probe_failure_suppressed_busy_count = 0
        self.probe_infeasible_suppressed_busy_count = 0
        self.probe_uncertified_suppressed_busy_count = 0

        self.last_committed_trajectory_time = None
        self.last_commit_count = 0
        self.last_commit_event_time = None
        self.probe_completed_prefix_streak = 0
        self.probe_completed_execution_count = 0
        self.pending_probe_candidate_seq = 0
        self.pending_probe_execution_stamp_ns = 0
        self.pending_probe_effective_prefix_s = math.nan
        self.last_tracker_complete_seq = 0
        self.last_tracker_complete_stamp_ns = 0
        self.repair_safe_commit_count = 0

        self.repair_entry_count = 0
        self.probe_entry_count = 0
        self.normal_entry_count = 0
        self.normal_task_replan_count = 0
        self.probe_failure_count = 0
        self.candidate_repair_entry_count = 0
        self.gcdf_repair_entry_count = 0
        self.execution_repair_entry_count = 0
        self.task_infeasible_repair_entry_count = 0
        self.task_infeasible_pending = False
        self.task_uncertified_repair_entry_count = 0
        self.task_uncertified_pending = False

        # C4.7: a stale latched True from a previous acquisition episode must not
        # immediately clear a newly-entered REPAIR.  Each episode arms only after
        # observing completion=False, then accepts a later False->True.
        self.repair_completion = False
        self.repair_completion_armed = False
        self.repair_completion_time = None
        self.repair_completion_event_count = 0

        # C5.11: solver failure and active-sensing demand are separate facts.
        # A failed PROBE task QP may request blocker rediscovery, but REPAIR is
        # entered only after a persistent visibility waypoint/obligation exists.
        self.visibility_waypoint_active = False
        self.blocker_rediscovery_pending = False
        self.blocker_rediscovery_origin = "none"
        self.blocker_rediscovery_force_bootstrap = False
        self.blocker_rediscovery_count = 0
        self.blocker_rediscovery_vbc_unsafe_count = 0
        self.blocker_rediscovery_vbc_safe_count = 0

        self.clear_until = None
        self.probe_ignore_until = None
        self.last_transition_reason = "startup"
        self.physical_deadline_abs = None

        self.trigger_pub = rospy.Publisher(
            self.trigger_topic, Bool, queue_size=1, latch=True)
        self.clear_pub = rospy.Publisher(
            self.clear_topic, Bool, queue_size=1, latch=True)
        self.hold_pub = rospy.Publisher(
            self.verification_hold_topic, Bool, queue_size=1, latch=True)
        self.probe_active_pub = rospy.Publisher(
            self.probe_active_topic, Bool, queue_size=1, latch=True)
        self.deadline_pub = rospy.Publisher(
            self.effective_deadline_topic, Float64, queue_size=1)
        self.summary_pub = rospy.Publisher(
            self.summary_topic, String, queue_size=1, latch=True)
        self.replan_request_pub = rospy.Publisher(
            self.replan_request_topic, Bool, queue_size=10, latch=False)
        self.force_vbc_bootstrap_pub = rospy.Publisher(
            self.force_vbc_bootstrap_topic, Bool, queue_size=1, latch=True)
        self.normal_task_replan_pub = rospy.Publisher(
            self.normal_task_replan_topic, Bool, queue_size=10, latch=False)

        rospy.Subscriber(
            self.candidate_outcome_topic, String, self._candidate_outcome_cb,
            queue_size=20)
        rospy.Subscriber(
            self.execution_summary_topic, String, self._execution_summary_cb,
            queue_size=1)
        rospy.Subscriber(
            self.tracker_summary_topic, String, self._tracker_summary_cb,
            queue_size=20)
        rospy.Subscriber(
            self.probe_single_flight_summary_topic, String,
            self._probe_single_flight_summary_cb, queue_size=20)
        rospy.Subscriber(
            self.committed_trajectory_topic, rospy.AnyMsg,
            self._committed_trajectory_cb, queue_size=1)
        rospy.Subscriber(
            self.execution_ready_topic, Bool, self._execution_ready_cb,
            queue_size=1)
        rospy.Subscriber(
            self.physical_deadline_topic, Float64, self._deadline_cb,
            queue_size=1)
        rospy.Subscriber(
            self.task_infeasible_topic, Bool, self._task_infeasible_cb,
            queue_size=10)
        rospy.Subscriber(
            self.task_uncertified_topic, Bool, self._task_uncertified_cb,
            queue_size=10)
        rospy.Subscriber(
            self.visibility_waypoint_active_topic, Bool,
            self._visibility_waypoint_active_cb, queue_size=1)
        rospy.Subscriber(
            self.candidate_vbc_summary_topic, String,
            self._candidate_vbc_summary_cb, queue_size=20)
        if self.repair_completion_gate_enabled:
            rospy.Subscriber(
                self.repair_completion_topic, Bool,
                self._repair_completion_cb, queue_size=1)

        self.trigger_pub.publish(Bool(data=False))
        self.probe_active_pub.publish(Bool(data=False))
        self.clear_pub.publish(Bool(data=False))
        self.hold_pub.publish(Bool(data=False))
        self.force_vbc_bootstrap_pub.publish(Bool(data=False))
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate), self._timer_cb)
        self._publish_summary()

        rospy.logwarn(
            "[c4_regime] NORMAL->REPAIR->PROBE_NORMAL->NORMAL; candidate=%s "
            "execution=%s probe_completed_prefixes=%d completion_gate=%d completion_topic=%s",
            self.candidate_outcome_topic, self.execution_summary_topic,
            self.probe_completed_prefixes_required,
            int(self.repair_completion_gate_enabled),
            self.repair_completion_topic)

    def _transition_locked(self, new_state, reason, now):
        if new_state == self.state:
            self.last_transition_reason = reason
            return
        old = self.state
        self.state = new_state
        self.last_transition_reason = reason
        # Blocker rediscovery is a PROBE-only fail-closed substate. Any real
        # regime transition terminates that substate.
        self.blocker_rediscovery_pending = False
        self.blocker_rediscovery_origin = "none"
        if self.blocker_rediscovery_force_bootstrap:
            self.force_vbc_bootstrap_pub.publish(Bool(data=False))
        self.blocker_rediscovery_force_bootstrap = False

        # Publish planner modes immediately on transition so subscribers cannot
        # race against stale REPAIR/PROBE state before the next 20 Hz timer tick.
        if hasattr(self, "trigger_pub"):
            self.trigger_pub.publish(Bool(
                data=bool(self.execution_ready and new_state == self.REPAIR)))
        if hasattr(self, "probe_active_pub"):
            self.probe_active_pub.publish(Bool(
                data=bool(self.execution_ready and new_state == self.PROBE_NORMAL)))

        if new_state == self.REPAIR:
            self.repair_entry_count += 1
            self.probe_completed_prefix_streak = 0
            self.pending_probe_candidate_seq = 0
            self.pending_probe_execution_stamp_ns = 0
            self.clear_until = None
            self.probe_ignore_until = None
            self.repair_completion = False
            self.repair_completion_armed = False
            if reason.startswith("execution_"):
                self.execution_repair_entry_count += 1
            elif reason.startswith("final_gcdf_"):
                self.gcdf_repair_entry_count += 1
            elif reason.startswith("task_"):
                # Task-QP infeasibility has its own counter and is neither a
                # candidate-VBC failure nor an execution-VBC failure.
                pass
            else:
                self.candidate_repair_entry_count += 1
        elif new_state == self.PROBE_NORMAL:
            self.probe_entry_count += 1
            # The REPAIR episode is complete only after its obligations are
            # gone. Locally clear the waypoint-active latch so a stale final
            # q_vis publication from that episode cannot authorize a future
            # PROBE->REPAIR transition.
            self.visibility_waypoint_active = False
            self.probe_completed_prefix_streak = 0
            self.pending_probe_candidate_seq = 0
            self.pending_probe_execution_stamp_ns = 0
            self.clear_until = now + rospy.Duration(self.clear_pulse_s)
            self.probe_ignore_until = now + rospy.Duration(self.probe_ignore_s)
        elif new_state == self.NORMAL:
            self.normal_entry_count += 1
            self.probe_completed_prefix_streak = 0
            self.pending_probe_candidate_seq = 0
            self.pending_probe_execution_stamp_ns = 0
            self.clear_until = None
            self.probe_ignore_until = None

            # C5.33: after active-sensing detours, the original one-shot
            # /task_trajectory time axis is stale. Regenerate the same EE task
            # from the current measured state before NORMAL planning resumes.
            # This refreshes only the task reference; it does not restore the
            # legacy RECOVERY_HOLD / replan-ready controller handshake.
            if old == self.PROBE_NORMAL and self.execution_ready:
                self.normal_task_replan_count += 1
                self.normal_task_replan_pub.publish(Bool(data=True))

        rospy.logwarn(
            "[c4_regime] %s -> %s reason=%s repair_entries=%d probe_entries=%d",
            old, new_state, reason, self.repair_entry_count, self.probe_entry_count)

    def _execution_ready_cb(self, msg):
        if msg is None:
            return
        with self._lock:
            was_ready = bool(self.execution_ready)
            new_ready = bool(msg.data)
            self.execution_ready = new_ready
            self.execution_ready_time = rospy.Time.now()

            if not new_ready:
                # Repeated startup/not-ready heartbeats must NOT erase a safety
                # outcome that arrived while the gate was closed. Only a true
                # ready->not-ready falling edge starts a new execution epoch.
                if was_ready:
                    self.state = self.NORMAL
                    self.candidate_unsafe_streak = 0
                    self.candidate_outcome_pending_gate_replay = False
                    self.execution_unsafe_streak = 0
                    self.execution_event_latched = False
                    self.probe_completed_prefix_streak = 0
                    self.pending_probe_candidate_seq = 0
                    self.pending_probe_execution_stamp_ns = 0
                    self.pending_probe_effective_prefix_s = math.nan
                    self.repair_completion = False
                    self.repair_completion_armed = False
                    self.blocker_rediscovery_pending = False
                    self.blocker_rediscovery_origin = "none"
                    self.blocker_rediscovery_force_bootstrap = False
                    self.force_vbc_bootstrap_pub.publish(Bool(data=False))
                    self.clear_until = None
                    self.probe_ignore_until = None
                self.last_transition_reason = "execution_not_ready"
                return

            # Ignore ready=True heartbeats. Replay pending decisions only on the
            # actual not-ready -> ready edge.
            if was_ready:
                return

            if self.task_infeasible_pending and self.state == self.NORMAL:
                self.task_infeasible_pending = False
                self.task_uncertified_pending = False
                self.task_infeasible_repair_entry_count += 1
                self._transition_locked(
                    self.REPAIR, "task_planner_infeasible_after_gate_release",
                    self.execution_ready_time)
            elif self.task_uncertified_pending and self.state == self.NORMAL:
                self.task_uncertified_pending = False
                self.task_infeasible_pending = False
                self.task_uncertified_repair_entry_count += 1
                self._transition_locked(
                    self.REPAIR, "task_planner_uncertified_after_gate_release",
                    self.execution_ready_time)
            elif (self.candidate_outcome_pending_gate_replay and
                  self.state == self.NORMAL):
                self.candidate_outcome_pending_gate_replay = False
                self.candidate_gate_replay_count += 1
                self._handle_normal_candidate_result_locked(
                    self.last_candidate_result,
                    self.last_candidate_safety_gate,
                    self.execution_ready_time,
                    replay=True)

    def _begin_blocker_rediscovery_locked(
            self, origin, force_bootstrap=True):
        """Fail closed in PROBE until a real visibility obligation exists.

        Numerical task-QP failures need same-SCP-trajectory VBC rediscovery.
        Exact-VBC rejection already carries a VBC active set, while final-GCDF
        rejection now exports the exact low-confidence voxels from the rejected
        executable trajectory.  Those latter two paths must not replace their
        stronger evidence with a guessed task-bootstrap blocker.
        """
        self.blocker_rediscovery_pending = True
        self.blocker_rediscovery_origin = str(origin)
        self.blocker_rediscovery_force_bootstrap = bool(force_bootstrap)
        self.blocker_rediscovery_count += 1
        self.pending_probe_candidate_seq = 0
        self.pending_probe_execution_stamp_ns = 0
        self.last_transition_reason = (
            "{}_wait_visibility_obligation".format(origin))
        self.force_vbc_bootstrap_pub.publish(
            Bool(data=bool(force_bootstrap)))

    def _visibility_waypoint_active_cb(self, msg):
        if msg is None:
            return
        value = bool(msg.data)
        now = rospy.Time.now()
        with self._lock:
            self.visibility_waypoint_active = value
            if (not value or not self.blocker_rediscovery_pending or
                    not self.execution_ready or
                    self.state != self.PROBE_NORMAL):
                return

            origin = self.blocker_rediscovery_origin
            force_bootstrap = self.blocker_rediscovery_force_bootstrap
            self.blocker_rediscovery_pending = False
            self.blocker_rediscovery_origin = "none"
            self.blocker_rediscovery_force_bootstrap = False
            if force_bootstrap:
                self.force_vbc_bootstrap_pub.publish(Bool(data=False))
            if "uncertified" in origin:
                self.task_uncertified_repair_entry_count += 1
            elif "infeasible" in origin:
                self.task_infeasible_repair_entry_count += 1
            self._transition_locked(
                self.REPAIR,
                "{}_visibility_obligation_ready".format(origin),
                now)

    def _candidate_vbc_summary_cb(self, msg):
        if msg is None:
            return
        f = _tokens(msg.data)
        # C5.30: solver-failure rediscovery is valid only on the latest
        # Sparse-SCP query trajectory: the same trajectory that generated the
        # local-GCDF batch. Task-bootstrap verdicts are not comparable here.
        if f.get("trajectory_source") != "scp_rediscovery":
            return
        has_violation = _as_bool(f.get("has_violation"))
        if has_violation is None:
            return

        request_replan = False
        clear_bootstrap = False
        with self._lock:
            if (not self.blocker_rediscovery_pending or
                    not self.blocker_rediscovery_force_bootstrap or
                    self.state != self.PROBE_NORMAL):
                return

            if has_violation:
                self.blocker_rediscovery_vbc_unsafe_count += 1
                self.last_transition_reason = (
                    "probe_blocker_confirmed_wait_visibility_waypoint")
                return

            # A fresh SAME-SCP-trajectory VBC SAFE verdict means this solver
            # failure is not attributable to a low-confidence swept volume on
            # the exact GCDF linearization. Stay in PROBE and replan.
            self.blocker_rediscovery_vbc_safe_count += 1
            self.blocker_rediscovery_pending = False
            self.blocker_rediscovery_origin = "none"
            self.blocker_rediscovery_force_bootstrap = False
            self.last_transition_reason = (
                "probe_no_visibility_blocker_after_solver_failure_replan")
            request_replan = True
            clear_bootstrap = True

        if clear_bootstrap:
            self.force_vbc_bootstrap_pub.publish(Bool(data=False))
        if request_replan:
            self.replan_request_pub.publish(Bool(data=True))

    def _probe_single_flight_busy_locked(self):
        return self.probe_single_flight_phase in ("VERIFYING", "EXECUTING")

    def _probe_single_flight_summary_cb(self, msg):
        if msg is None:
            return
        f = _tokens(msg.data)
        phase = f.get("phase", "IDLE")
        if phase not in ("IDLE", "VERIFYING", "EXECUTING"):
            return
        execution_stamp_ns = _as_int(f.get("execution_stamp_ns"), 0)
        reason = f.get("reason", "none")
        with self._lock:
            self.probe_single_flight_phase = phase
            self.probe_single_flight_execution_stamp_ns = execution_stamp_ns
            self.probe_single_flight_reason = reason
            self.probe_single_flight_summary_time = rospy.Time.now()

    def _task_infeasible_cb(self, msg):
        if msg is None or not bool(msg.data):
            return
        now = rospy.Time.now()
        with self._lock:
            # NORMAL infeasibility starts the first sensing episode. A
            # PROBE_NORMAL infeasibility means the just-expanded confidence map
            # is still insufficient for the task, so it starts the next sensing
            # episode instead of leaving the state machine stuck in PROBE_NORMAL.
            if self.state not in (self.NORMAL, self.PROBE_NORMAL):
                return

            if not self.execution_ready:
                if self.state == self.NORMAL:
                    self.task_infeasible_pending = True
                    self.last_transition_reason = "task_planner_infeasible_pending"
                return

            self.task_infeasible_pending = False
            self.task_uncertified_pending = False

            if self.state == self.PROBE_NORMAL:
                if self._probe_single_flight_busy_locked():
                    # A prior PROBE decision already owns the downstream
                    # verification/execution flight. A later planner attempt
                    # may fail, but that stale failure must not cancel or
                    # reinterpret the certified flight currently in progress.
                    self.probe_failure_suppressed_busy_count += 1
                    self.probe_infeasible_suppressed_busy_count += 1
                    self.last_transition_reason = (
                        "task_probe_infeasible_suppressed_single_flight_{}".format(
                            self.probe_single_flight_phase.lower()))
                    self._publish_summary()
                    return
                self.probe_failure_count += 1
                if (self.repair_completion_gate_enabled and
                        not self.visibility_waypoint_active):
                    self._begin_blocker_rediscovery_locked(
                        "task_probe_infeasible")
                else:
                    self.task_infeasible_repair_entry_count += 1
                    self._transition_locked(
                        self.REPAIR, "task_probe_infeasible", now)
            else:
                self.task_infeasible_repair_entry_count += 1
                self._transition_locked(
                    self.REPAIR, "task_planner_infeasible", now)

    def _task_uncertified_cb(self, msg):
        if msg is None or not bool(msg.data):
            return
        now = rospy.Time.now()
        with self._lock:
            # "Uncertified" is deliberately distinct from mathematical
            # infeasibility. It means PIQP exhausted its iteration budget and
            # therefore did not provide a hard-GCDF-certified task candidate.
            # C5.11 no longer equates this numerical failure with a sensing
            # request: PROBE fails closed and asks VBC to rediscover a blocker.
            # Only a real visibility obligation is allowed to enter REPAIR.
            if self.state not in (self.NORMAL, self.PROBE_NORMAL):
                return

            if not self.execution_ready:
                if self.state == self.NORMAL:
                    self.task_uncertified_pending = True
                    self.last_transition_reason = "task_planner_uncertified_pending"
                return

            self.task_uncertified_pending = False
            self.task_infeasible_pending = False

            if self.state == self.PROBE_NORMAL:
                if self._probe_single_flight_busy_locked():
                    self.probe_failure_suppressed_busy_count += 1
                    self.probe_uncertified_suppressed_busy_count += 1
                    self.last_transition_reason = (
                        "task_probe_uncertified_suppressed_single_flight_{}".format(
                            self.probe_single_flight_phase.lower()))
                    self._publish_summary()
                    return
                self.probe_failure_count += 1
                if (self.repair_completion_gate_enabled and
                        not self.visibility_waypoint_active):
                    self._begin_blocker_rediscovery_locked(
                        "task_probe_uncertified")
                else:
                    self.task_uncertified_repair_entry_count += 1
                    self._transition_locked(
                        self.REPAIR, "task_probe_uncertified", now)
            else:
                self.task_uncertified_repair_entry_count += 1
                self._transition_locked(
                    self.REPAIR, "task_planner_uncertified", now)

    def _deadline_cb(self, msg):
        if msg is None:
            return
        value = float(msg.data)
        if not math.isfinite(value) or value <= 0.0:
            return
        with self._lock:
            self.physical_deadline_abs = value
        self.deadline_pub.publish(Float64(data=value))

    def _committed_trajectory_cb(self, _msg):
        with self._lock:
            self.last_committed_trajectory_time = rospy.Time.now()

    def _repair_completion_cb(self, msg):
        if msg is None or not self.repair_completion_gate_enabled:
            return
        value = bool(msg.data)
        now = rospy.Time.now()
        with self._lock:
            self.repair_completion = value
            self.repair_completion_time = now
            if self.state != self.REPAIR or not self.execution_ready:
                return
            if not value:
                self.repair_completion_armed = True
                return
            if value and self.repair_completion_armed:
                self.repair_completion_event_count += 1
                self._transition_locked(
                    self.PROBE_NORMAL, "actual_visibility_acquisition_complete", now)

    def _handle_normal_candidate_result_locked(
            self, result, safety_gate, now, replay=False):
        """Apply NORMAL candidate safety semantics from live or replayed input.

        C5.22: ordinary VBC unsafe uses hysteresis, but every rejected candidate
        before the threshold must explicitly request a fresh NORMAL plan.
        Otherwise no candidate is committed and there may be no later event to
        advance candidate_unsafe_streak from 1 to the configured threshold.
        Final-GCDF unsafe remains fail-closed and transitions immediately.
        """
        source = "after_gate_release" if replay else "live"
        if result == "unsafe" and safety_gate == "gcdf":
            self._transition_locked(
                self.REPAIR,
                "final_gcdf_normal_unsafe_{}".format(source),
                now)
            return

        if result == "unsafe":
            if self.candidate_unsafe_streak >= self.candidate_unsafe_required:
                self.normal_unsafe_confirmed_repair_count += 1
                self._transition_locked(
                    self.REPAIR,
                    "candidate_unique_unsafe_confirmed_{}".format(source),
                    now)
                return

            # Hysteresis not yet confirmed: stay NORMAL, but the rejected
            # candidate produced no commit/tracker-complete trigger. Request a
            # fresh plan from the latest measured state so a subsequent safe
            # result can clear the streak or a second unsafe can confirm REPAIR.
            self.normal_unsafe_retry_count += 1
            self.last_transition_reason = (
                "candidate_unsafe_retry_{}/{}_{}".format(
                    self.candidate_unsafe_streak,
                    self.candidate_unsafe_required,
                    source))
            self.replan_request_pub.publish(Bool(data=True))
            return

        if result == "timeout":
            # No candidate was certified or committed. Stay in NORMAL and
            # explicitly request a fresh plan; never reuse stale execution.
            self.last_transition_reason = (
                "{}_normal_timeout_replan_{}".format(
                    safety_gate, source))
            self.replan_request_pub.publish(Bool(data=True))

    def _candidate_outcome_cb(self, msg):
        if msg is None:
            return
        f = _tokens(msg.data)
        seq = _as_int(f.get("seq"), 0)
        result = f.get("result", "")
        committed = _as_bool(f.get("committed"))
        verification_view = f.get("verification_view", "none")
        safety_gate = f.get("safety_gate", "vbc")
        execution_stamp_ns = _as_int(f.get("execution_stamp_ns"), 0)
        probe_effective_prefix_s = _as_float(
            f.get("probe_effective_prefix_s"), math.nan)
        if seq <= 0 or result not in ("safe", "unsafe", "timeout"):
            return

        now = rospy.Time.now()
        with self._lock:
            if seq <= self.last_candidate_seq:
                return
            self.last_candidate_seq = seq
            self.last_candidate_result = result
            self.last_candidate_safety_gate = safety_gate
            self.last_candidate_verification_view = verification_view
            self.candidate_outcome_time = now
            self.candidate_event_count += 1

            if result == "unsafe":
                self.candidate_last_unsafe = True
                self.candidate_unsafe_count += 1
                self.candidate_unsafe_streak += 1
            elif result == "safe":
                self.candidate_last_unsafe = False
                self.candidate_safe_count += 1
                self.candidate_unsafe_streak = 0
            else:
                self.candidate_timeout_count += 1
                self.candidate_unsafe_streak = 0

            if not self.execution_ready:
                # Startup race protection: preserve the safety fact and replay
                # it when execution_ready rises.  The old code recorded the
                # unsafe result then returned forever if no later candidate
                # event arrived.
                self.candidate_outcome_pending_gate_replay = True
                self.last_transition_reason = (
                    "candidate_outcome_pending_gate_replay")
                return

            self.candidate_outcome_pending_gate_replay = False
            if self.state == self.NORMAL:
                self._handle_normal_candidate_result_locked(
                    result, safety_gate, now, replay=False)
                return

            if self.state == self.REPAIR:
                if result == "unsafe":
                    # Final executable GCDF/VBC rejection means no trajectory
                    # was committed, so there will be no tracker-complete event
                    # to drive another REPAIR plan. Stay in REPAIR and explicitly
                    # request a fresh local plan.
                    self.last_transition_reason = "repair_candidate_unsafe_replan"
                    self.replan_request_pub.publish(Bool(data=True))
                    return
                if result == "timeout":
                    self.last_transition_reason = "repair_candidate_timeout_replan"
                    self.replan_request_pub.publish(Bool(data=True))
                    return
                if result == "safe" and committed is True:
                    self.last_commit_count += 1
                    self.repair_safe_commit_count += 1
                    self.last_commit_event_time = now
                    if self.repair_completion_gate_enabled:
                        # C4.7: executing a VBC-safe detour is progress toward
                        # seeing. Stay in REPAIR until real confidence confirms it.
                        self.last_transition_reason = (
                            "safe_repair_commit_continue_visibility_acquisition")
                    else:
                        self._transition_locked(
                            self.PROBE_NORMAL, "safe_repair_commit", now)
                return

            if self.state == self.PROBE_NORMAL:
                # A stale REPAIR verification may arrive just after the regime
                # transition. Only a candidate explicitly built as a PROBE
                # executable prefix may arm the PROBE execution handshake.
                if (result == "safe" and committed is True and
                        verification_view == "probe_prefix_brake_hold"):
                    if execution_stamp_ns <= 0:
                        self.pending_probe_candidate_seq = 0
                        self.pending_probe_execution_stamp_ns = 0
                        self.pending_probe_effective_prefix_s = math.nan
                        self.last_transition_reason = (
                            "probe_commit_missing_execution_token_replan")
                        self.replan_request_pub.publish(Bool(data=True))
                        return
                    self.last_commit_count += 1
                    self.last_commit_event_time = now
                    self.pending_probe_candidate_seq = seq
                    self.pending_probe_execution_stamp_ns = execution_stamp_ns
                    self.pending_probe_effective_prefix_s = (
                        probe_effective_prefix_s
                        if (math.isfinite(probe_effective_prefix_s) and
                            probe_effective_prefix_s > 0.0)
                        else math.nan)
                    self.last_transition_reason = (
                        "safe_probe_commit_wait_prefix_stamp_{}".format(
                            execution_stamp_ns))
                    return

                if result == "unsafe":
                    # A current PROBE prefix has an unambiguous view tag and
                    # must never be hidden by the short post-transition ignore
                    # window. The window only suppresses stale non-PROBE events.
                    if (verification_view != "probe_prefix_brake_hold" and
                            self.probe_ignore_until is not None and
                            now < self.probe_ignore_until):
                        return
                    self.probe_failure_count += 1
                    self.pending_probe_candidate_seq = 0
                    self.pending_probe_execution_stamp_ns = 0
                    self.pending_probe_effective_prefix_s = math.nan

                    origin = (
                        "final_gcdf_probe_unsafe"
                        if safety_gate == "gcdf"
                        else "candidate_probe_unique_unsafe")

                    # C5.12 invariant:
                    #   PROBE -> REPAIR only after q_vis/obligation is active.
                    # Exact VBC already emitted an active set. Final GCDF now
                    # emits its own earliest rejected low-confidence voxels.
                    if (self.repair_completion_gate_enabled and
                            not self.visibility_waypoint_active):
                        self._begin_blocker_rediscovery_locked(
                            origin, force_bootstrap=False)
                    else:
                        self._transition_locked(
                            self.REPAIR, origin, now)
                    return

                if (result == "timeout" and
                        verification_view == "probe_prefix_brake_hold"):
                    # Timeout is not evidence that active sensing is required;
                    # stay in PROBE, fail closed, and request a fresh candidate.
                    self.pending_probe_candidate_seq = 0
                    self.pending_probe_execution_stamp_ns = 0
                    self.pending_probe_effective_prefix_s = math.nan
                    self.last_transition_reason = "probe_candidate_timeout_replan"
                    self.replan_request_pub.publish(Bool(data=True))
                    return

    def _tracker_summary_cb(self, msg):
        if msg is None:
            return
        f = _tokens(msg.data)
        active = _as_bool(f.get("active"))
        complete = _as_bool(f.get("complete"))
        seq = _as_int(f.get("seq"), 0)
        execution_stamp_ns = _as_int(f.get("execution_stamp_ns"), 0)
        source = f.get("source", "")
        phase_s = _as_float(f.get("phase_s"), math.nan)

        now = rospy.Time.now()
        request_next_probe = False
        with self._lock:
            # Tracker liveness/ownership is the execution freshness signal in
            # the single-publish architecture. A committed topic message is no
            # longer expected to refresh while a trajectory is running.
            self.tracker_summary_time = now
            self.tracker_active = bool(active) if active is not None else False
            self.tracker_complete = bool(complete) if complete is not None else False
            self.tracker_execution_stamp_ns = execution_stamp_ns
            self.tracker_source = source

            # Preserve full-completion diagnostics independently of PROBE
            # progress counting.
            full_complete = (
                complete is True and execution_stamp_ns > 0 and
                source == "trajectory_complete_hold")
            if full_complete:
                self.last_tracker_complete_seq = seq
                if execution_stamp_ns != self.last_tracker_complete_stamp_ns:
                    self.last_tracker_complete_stamp_ns = execution_stamp_ns

            if self.state != self.PROBE_NORMAL:
                return
            if self.pending_probe_execution_stamp_ns <= 0:
                return
            if execution_stamp_ns != self.pending_probe_execution_stamp_ns:
                # Stable timestamp matching rejects stale status from an older
                # committed trajectory.
                return

            prefix_complete = (
                math.isfinite(self.pending_probe_effective_prefix_s) and
                self.pending_probe_effective_prefix_s > 0.0 and
                math.isfinite(phase_s) and
                phase_s + 1e-6 >= self.pending_probe_effective_prefix_s)

            # Metadata-missing fallback preserves the old fail-closed behavior:
            # only a full certified trajectory completion can count.
            if not prefix_complete and not full_complete:
                return

            completed_stamp = self.pending_probe_execution_stamp_ns
            self.pending_probe_candidate_seq = 0
            self.pending_probe_execution_stamp_ns = 0
            self.pending_probe_effective_prefix_s = math.nan
            self.probe_completed_prefix_streak += 1
            # Compatibility counter: this now means an actually executed,
            # certified PROBE prefix (full completion remains separately logged
            # through last_tracker_complete_*).
            self.probe_completed_execution_count += 1

            if self.probe_completed_prefix_streak >= self.probe_completed_prefixes_required:
                self._transition_locked(
                    self.NORMAL, "probe_normal_completed_prefixes", now)
            else:
                self.last_transition_reason = (
                    "probe_certified_prefix_complete_stamp_{}".format(
                        completed_stamp))
                # Start the next measured-state PROBE now. The old committed
                # trajectory remains safe through its certified brake+hold tail
                # until the replacement itself passes final GCDF + exact VBC.
                request_next_probe = True

        if request_next_probe:
            self.replan_request_pub.publish(Bool(data=True))

    def _execution_summary_cb(self, msg):
        if msg is None:
            return
        f = _tokens(msg.data)
        if f.get("trajectory_source") != "predicted":
            return
        unsafe = _as_bool(f.get("has_violation"))
        if unsafe is None:
            return
        now = rospy.Time.now()
        with self._lock:
            if self.last_committed_trajectory_time is None:
                return
            self.execution_summary_time = now
            self.execution_last_unsafe = bool(unsafe)
            if unsafe:
                self.execution_unsafe_streak += 1
            else:
                self.execution_safe_count += 1
                self.execution_unsafe_streak = 0
                self.execution_event_latched = False

            if (self.execution_ready and unsafe and
                    self.execution_unsafe_streak >= self.execution_unsafe_required and
                    not self.execution_event_latched):
                self.execution_event_latched = True
                self.execution_safety_event_count += 1
                if self.state != self.REPAIR:
                    self._transition_locked(
                        self.REPAIR, "execution_committed_unsafe", now)

    @staticmethod
    def _fresh(stamp, now, timeout):
        if stamp is None:
            return False
        age = (now - stamp).to_sec()
        return 0.0 <= age <= timeout

    def _timer_cb(self, _event):
        now = rospy.Time.now()
        with self._lock:
            trigger = self.execution_ready and self.state == self.REPAIR
            probe_active = self.execution_ready and self.state == self.PROBE_NORMAL
            clear = (
                self.execution_ready and self.state == self.PROBE_NORMAL and
                self.clear_until is not None and now <= self.clear_until)

            exec_summary_fresh = self._fresh(
                self.execution_summary_time, now, self.input_timeout)
            tracker_summary_fresh = self._fresh(
                self.tracker_summary_time, now, self.input_timeout)
            startup_grace = (
                self.execution_ready_time is not None and
                (now - self.execution_ready_time).to_sec() < self.input_timeout)
            execution_in_progress = bool(
                self.tracker_active and self.tracker_execution_stamp_ns > 0)
            hold = bool(
                self.execution_ready and execution_in_progress and
                not startup_grace and
                (not exec_summary_fresh or not tracker_summary_fresh))
            deadline = self.physical_deadline_abs

        self.trigger_pub.publish(Bool(data=trigger))
        self.probe_active_pub.publish(Bool(data=probe_active))
        self.clear_pub.publish(Bool(data=clear))
        self.hold_pub.publish(Bool(data=hold))
        if deadline is not None:
            self.deadline_pub.publish(Float64(data=deadline))
        self._publish_summary()

    def _publish_summary(self):
        with self._lock:
            now = rospy.Time.now()

            def age(stamp):
                if stamp is None:
                    return math.nan
                return max(0.0, (now - stamp).to_sec())

            msg = String()
            msg.data = " ".join([
                "state={}".format(self.state),
                "reason={}".format(self.last_transition_reason),
                "execution_ready={}".format(int(self.execution_ready)),
                "last_candidate_seq={}".format(self.last_candidate_seq),
                "last_candidate_result={}".format(self.last_candidate_result),
                "last_candidate_safety_gate={}".format(
                    self.last_candidate_safety_gate),
                "last_candidate_verification_view={}".format(
                    self.last_candidate_verification_view),
                "candidate_outcome_pending_gate_replay={}".format(
                    int(self.candidate_outcome_pending_gate_replay)),
                "candidate_gate_replay_count={}".format(
                    self.candidate_gate_replay_count),
                "candidate_event_count={}".format(self.candidate_event_count),
                "candidate_last_unsafe={}".format(int(self.candidate_last_unsafe)),
                "candidate_unsafe_streak={}".format(self.candidate_unsafe_streak),
                "candidate_unsafe_count={}".format(self.candidate_unsafe_count),
                "candidate_safe_count={}".format(self.candidate_safe_count),
                "candidate_timeout_count={}".format(self.candidate_timeout_count),
                "normal_unsafe_retry_count={}".format(
                    self.normal_unsafe_retry_count),
                "normal_unsafe_confirmed_repair_count={}".format(
                    self.normal_unsafe_confirmed_repair_count),
                "execution_last_unsafe={}".format(int(self.execution_last_unsafe)),
                "execution_unsafe_streak={}".format(self.execution_unsafe_streak),
                "execution_event_latched={}".format(int(self.execution_event_latched)),
                "execution_safety_event_count={}".format(self.execution_safety_event_count),
                "tracker_active={}".format(int(self.tracker_active)),
                "tracker_complete={}".format(int(self.tracker_complete)),
                "tracker_execution_stamp_ns={}".format(
                    self.tracker_execution_stamp_ns),
                "tracker_source={}".format(self.tracker_source),
                "repair_entry_count={}".format(self.repair_entry_count),
                "candidate_repair_entry_count={}".format(self.candidate_repair_entry_count),
                "gcdf_repair_entry_count={}".format(self.gcdf_repair_entry_count),
                "execution_repair_entry_count={}".format(self.execution_repair_entry_count),
                "task_infeasible_repair_entry_count={}".format(
                    self.task_infeasible_repair_entry_count),
                "task_infeasible_pending={}".format(
                    int(self.task_infeasible_pending)),
                "task_uncertified_repair_entry_count={}".format(
                    self.task_uncertified_repair_entry_count),
                "task_uncertified_pending={}".format(
                    int(self.task_uncertified_pending)),
                "probe_entry_count={}".format(self.probe_entry_count),
                "probe_failure_count={}".format(self.probe_failure_count),
                "probe_single_flight_phase={}".format(
                    self.probe_single_flight_phase),
                "probe_single_flight_execution_stamp_ns={}".format(
                    self.probe_single_flight_execution_stamp_ns),
                "probe_single_flight_reason={}".format(
                    self.probe_single_flight_reason),
                "probe_failure_suppressed_busy_count={}".format(
                    self.probe_failure_suppressed_busy_count),
                "probe_infeasible_suppressed_busy_count={}".format(
                    self.probe_infeasible_suppressed_busy_count),
                "probe_uncertified_suppressed_busy_count={}".format(
                    self.probe_uncertified_suppressed_busy_count),
                "probe_completed_prefix_streak={}".format(
                    self.probe_completed_prefix_streak),
                # Compatibility alias for older result parsers.
                "probe_safe_commit_streak={}".format(
                    self.probe_completed_prefix_streak),
                "probe_completed_prefixes_required={}".format(
                    self.probe_completed_prefixes_required),
                "probe_completed_execution_count={}".format(
                    self.probe_completed_execution_count),
                "pending_probe_candidate_seq={}".format(
                    self.pending_probe_candidate_seq),
                "pending_probe_execution_stamp_ns={}".format(
                    self.pending_probe_execution_stamp_ns),
                "pending_probe_effective_prefix_s={}".format(
                    "nan" if not math.isfinite(
                        self.pending_probe_effective_prefix_s)
                    else "{:.6f}".format(
                        self.pending_probe_effective_prefix_s)),
                "last_tracker_complete_seq={}".format(
                    self.last_tracker_complete_seq),
                "last_tracker_complete_stamp_ns={}".format(
                    self.last_tracker_complete_stamp_ns),
                "normal_entry_count={}".format(self.normal_entry_count),
                "normal_task_replan_count={}".format(
                    self.normal_task_replan_count),
                "commit_count={}".format(self.last_commit_count),
                "repair_safe_commit_count={}".format(self.repair_safe_commit_count),
                "repair_completion_gate_enabled={}".format(
                    int(self.repair_completion_gate_enabled)),
                "repair_completion={}".format(int(self.repair_completion)),
                "repair_completion_armed={}".format(
                    int(self.repair_completion_armed)),
                "repair_completion_event_count={}".format(
                    self.repair_completion_event_count),
                "visibility_waypoint_active={}".format(
                    int(self.visibility_waypoint_active)),
                "blocker_rediscovery_pending={}".format(
                    int(self.blocker_rediscovery_pending)),
                "blocker_rediscovery_origin={}".format(
                    self.blocker_rediscovery_origin),
                "blocker_rediscovery_force_bootstrap={}".format(
                    int(self.blocker_rediscovery_force_bootstrap)),
                "blocker_rediscovery_count={}".format(
                    self.blocker_rediscovery_count),
                "blocker_rediscovery_vbc_unsafe_count={}".format(
                    self.blocker_rediscovery_vbc_unsafe_count),
                "blocker_rediscovery_vbc_safe_count={}".format(
                    self.blocker_rediscovery_vbc_safe_count),
                "candidate_outcome_age_s={:.6f}".format(
                    age(self.candidate_outcome_time)),
                "execution_summary_age_s={:.6f}".format(
                    age(self.execution_summary_time)),
                "tracker_summary_age_s={:.6f}".format(
                    age(self.tracker_summary_time)),
                "committed_trajectory_age_s={:.6f}".format(
                    age(self.last_committed_trajectory_time)),
                "repair_completion_age_s={:.6f}".format(
                    age(self.repair_completion_time)),
            ])
        self.summary_pub.publish(msg)


def main():
    rospy.init_node("c4_4_verified_regime_manager")
    C44VerifiedRegimeManager()
    rospy.spin()


if __name__ == "__main__":
    main()
