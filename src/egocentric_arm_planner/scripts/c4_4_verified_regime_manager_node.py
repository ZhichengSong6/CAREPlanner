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

Candidate safety and committed-execution safety remain separate. Candidate-side
transitions consume unique verification outcome events; execution VBC continuously
audits only committed trajectories.
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
        self.probe_safe_commits_required = int(rospy.get_param(
            "~probe_safe_commits_required", 3))
        self.input_timeout = float(rospy.get_param("~input_timeout", 0.25))
        self.committed_trajectory_timeout = float(rospy.get_param(
            "~committed_trajectory_timeout", 0.30))
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
        if self.probe_safe_commits_required < 1:
            raise ValueError("probe_safe_commits_required must be >= 1")
        if min(self.input_timeout, self.committed_trajectory_timeout,
               self.clear_pulse_s, self.probe_ignore_s) <= 0.0:
            raise ValueError("timeouts/pulses must be positive")

        self.candidate_outcome_topic = str(rospy.get_param(
            "~candidate_outcome_topic", "/care_planner/verification_outcome"))
        self.execution_summary_topic = str(rospy.get_param(
            "~execution_summary_topic", "/care_planner/execution_vbc/summary"))
        self.tracker_summary_topic = str(rospy.get_param(
            "~tracker_summary_topic", "/care_planner/execution/tracker_summary"))
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

        self.state = self.NORMAL
        self.execution_ready = False
        self.execution_ready_time = None

        self.candidate_outcome_time = None
        self.last_candidate_seq = 0
        self.last_candidate_result = "none"
        self.candidate_last_unsafe = False
        self.candidate_unsafe_streak = 0
        self.candidate_unsafe_count = 0
        self.candidate_safe_count = 0
        self.candidate_timeout_count = 0
        self.candidate_event_count = 0

        self.execution_summary_time = None
        self.execution_last_unsafe = False
        self.execution_unsafe_streak = 0
        self.execution_event_latched = False
        self.execution_safety_event_count = 0
        self.execution_safe_count = 0

        self.last_committed_trajectory_time = None
        self.last_commit_count = 0
        self.last_commit_event_time = None
        self.probe_safe_commit_streak = 0
        self.probe_completed_execution_count = 0
        self.pending_probe_execution_seq = 0
        self.last_tracker_complete_seq = 0
        self.repair_safe_commit_count = 0

        self.repair_entry_count = 0
        self.probe_entry_count = 0
        self.normal_entry_count = 0
        self.probe_failure_count = 0
        self.candidate_repair_entry_count = 0
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
        if self.repair_completion_gate_enabled:
            rospy.Subscriber(
                self.repair_completion_topic, Bool,
                self._repair_completion_cb, queue_size=1)

        self.trigger_pub.publish(Bool(data=False))
        self.probe_active_pub.publish(Bool(data=False))
        self.clear_pub.publish(Bool(data=False))
        self.hold_pub.publish(Bool(data=False))
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate), self._timer_cb)
        self._publish_summary()

        rospy.logwarn(
            "[c4_regime] NORMAL->REPAIR->PROBE_NORMAL->NORMAL; candidate=%s "
            "execution=%s probe_safe_commits=%d completion_gate=%d completion_topic=%s",
            self.candidate_outcome_topic, self.execution_summary_topic,
            self.probe_safe_commits_required,
            int(self.repair_completion_gate_enabled),
            self.repair_completion_topic)

    def _transition_locked(self, new_state, reason, now):
        if new_state == self.state:
            self.last_transition_reason = reason
            return
        old = self.state
        self.state = new_state
        self.last_transition_reason = reason

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
            self.probe_safe_commit_streak = 0
            self.pending_probe_execution_seq = 0
            self.clear_until = None
            self.probe_ignore_until = None
            self.repair_completion = False
            self.repair_completion_armed = False
            if reason.startswith("execution_"):
                self.execution_repair_entry_count += 1
            elif reason.startswith("task_"):
                # Task-QP infeasibility has its own counter and is neither a
                # candidate-VBC failure nor an execution-VBC failure.
                pass
            else:
                self.candidate_repair_entry_count += 1
        elif new_state == self.PROBE_NORMAL:
            self.probe_entry_count += 1
            self.probe_safe_commit_streak = 0
            self.pending_probe_execution_seq = 0
            self.clear_until = now + rospy.Duration(self.clear_pulse_s)
            self.probe_ignore_until = now + rospy.Duration(self.probe_ignore_s)
        elif new_state == self.NORMAL:
            self.normal_entry_count += 1
            self.probe_safe_commit_streak = 0
            self.pending_probe_execution_seq = 0
            self.clear_until = None
            self.probe_ignore_until = None

        rospy.logwarn(
            "[c4_regime] %s -> %s reason=%s repair_entries=%d probe_entries=%d",
            old, new_state, reason, self.repair_entry_count, self.probe_entry_count)

    def _execution_ready_cb(self, msg):
        if msg is None:
            return
        with self._lock:
            self.execution_ready = bool(msg.data)
            self.execution_ready_time = rospy.Time.now()
            if not self.execution_ready:
                self.state = self.NORMAL
                self.candidate_unsafe_streak = 0
                self.execution_unsafe_streak = 0
                self.execution_event_latched = False
                self.probe_safe_commit_streak = 0
                self.pending_probe_execution_seq = 0
                self.repair_completion = False
                self.repair_completion_armed = False
                self.clear_until = None
                self.probe_ignore_until = None
                self.last_transition_reason = "execution_not_ready"
            elif self.task_infeasible_pending and self.state == self.NORMAL:
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
            self.task_infeasible_repair_entry_count += 1

            if self.state == self.PROBE_NORMAL:
                self.probe_failure_count += 1
                self._transition_locked(
                    self.REPAIR, "task_probe_infeasible", now)
            else:
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
            # Safety takes priority over task progress, so NORMAL/PROBE both
            # fall back to REPAIR while preserving the numerical diagnosis.
            if self.state not in (self.NORMAL, self.PROBE_NORMAL):
                return

            if not self.execution_ready:
                if self.state == self.NORMAL:
                    self.task_uncertified_pending = True
                    self.last_transition_reason = "task_planner_uncertified_pending"
                return

            self.task_uncertified_pending = False
            self.task_infeasible_pending = False
            self.task_uncertified_repair_entry_count += 1

            if self.state == self.PROBE_NORMAL:
                self.probe_failure_count += 1
                self._transition_locked(
                    self.REPAIR, "task_probe_uncertified", now)
            else:
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

    def _candidate_outcome_cb(self, msg):
        if msg is None:
            return
        f = _tokens(msg.data)
        seq = _as_int(f.get("seq"), 0)
        result = f.get("result", "")
        committed = _as_bool(f.get("committed"))
        verification_view = f.get("verification_view", "none")
        if seq <= 0 or result not in ("safe", "unsafe", "timeout"):
            return

        now = rospy.Time.now()
        with self._lock:
            if seq <= self.last_candidate_seq:
                return
            self.last_candidate_seq = seq
            self.last_candidate_result = result
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
                return

            if self.state == self.NORMAL:
                if (result == "unsafe" and
                        self.candidate_unsafe_streak >= self.candidate_unsafe_required):
                    self._transition_locked(
                        self.REPAIR, "candidate_unique_unsafe_confirmed", now)
                return

            if self.state == self.REPAIR:
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
                    self.last_commit_count += 1
                    self.last_commit_event_time = now
                    self.pending_probe_execution_seq = seq
                    self.last_transition_reason = (
                        "safe_probe_commit_wait_execution_seq_{}".format(seq))
                    return

                if self.probe_ignore_until is not None and now < self.probe_ignore_until:
                    return
                if result == "unsafe":
                    self.probe_failure_count += 1
                    self.pending_probe_execution_seq = 0
                    self._transition_locked(
                        self.REPAIR, "candidate_probe_unique_unsafe", now)
                    return

    def _tracker_summary_cb(self, msg):
        if msg is None:
            return
        f = _tokens(msg.data)
        complete = _as_bool(f.get("complete"))
        seq = _as_int(f.get("seq"), 0)
        if complete is not True or seq <= 0:
            return

        now = rospy.Time.now()
        request_next_probe = False
        with self._lock:
            if seq <= self.last_tracker_complete_seq:
                return
            self.last_tracker_complete_seq = seq

            if self.state != self.PROBE_NORMAL:
                return
            if self.pending_probe_execution_seq <= 0:
                return
            if seq != self.pending_probe_execution_seq:
                # Exact sequence matching rejects stale completion edges from
                # the previous REPAIR episode or an older replaced trajectory.
                return

            self.pending_probe_execution_seq = 0
            self.probe_safe_commit_streak += 1
            self.probe_completed_execution_count += 1

            if self.probe_safe_commit_streak >= self.probe_safe_commits_required:
                self._transition_locked(
                    self.NORMAL, "probe_normal_completed_prefixes", now)
            else:
                self.last_transition_reason = (
                    "probe_prefix_execution_complete_seq_{}".format(seq))
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
            committed_fresh = self._fresh(
                self.last_committed_trajectory_time, now,
                self.committed_trajectory_timeout)
            startup_grace = (
                self.execution_ready_time is not None and
                (now - self.execution_ready_time).to_sec() < self.input_timeout)
            has_committed = self.last_committed_trajectory_time is not None
            hold = bool(
                self.execution_ready and has_committed and not startup_grace and
                (not exec_summary_fresh or not committed_fresh))
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
                "candidate_event_count={}".format(self.candidate_event_count),
                "candidate_last_unsafe={}".format(int(self.candidate_last_unsafe)),
                "candidate_unsafe_streak={}".format(self.candidate_unsafe_streak),
                "candidate_unsafe_count={}".format(self.candidate_unsafe_count),
                "candidate_safe_count={}".format(self.candidate_safe_count),
                "candidate_timeout_count={}".format(self.candidate_timeout_count),
                "execution_last_unsafe={}".format(int(self.execution_last_unsafe)),
                "execution_unsafe_streak={}".format(self.execution_unsafe_streak),
                "execution_event_latched={}".format(int(self.execution_event_latched)),
                "execution_safety_event_count={}".format(self.execution_safety_event_count),
                "repair_entry_count={}".format(self.repair_entry_count),
                "candidate_repair_entry_count={}".format(self.candidate_repair_entry_count),
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
                "probe_safe_commit_streak={}".format(self.probe_safe_commit_streak),
                "probe_completed_execution_count={}".format(
                    self.probe_completed_execution_count),
                "pending_probe_execution_seq={}".format(
                    self.pending_probe_execution_seq),
                "last_tracker_complete_seq={}".format(
                    self.last_tracker_complete_seq),
                "normal_entry_count={}".format(self.normal_entry_count),
                "commit_count={}".format(self.last_commit_count),
                "repair_safe_commit_count={}".format(self.repair_safe_commit_count),
                "repair_completion_gate_enabled={}".format(
                    int(self.repair_completion_gate_enabled)),
                "repair_completion={}".format(int(self.repair_completion)),
                "repair_completion_armed={}".format(
                    int(self.repair_completion_armed)),
                "repair_completion_event_count={}".format(
                    self.repair_completion_event_count),
                "candidate_outcome_age_s={:.6f}".format(
                    age(self.candidate_outcome_time)),
                "execution_summary_age_s={:.6f}".format(
                    age(self.execution_summary_time)),
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
