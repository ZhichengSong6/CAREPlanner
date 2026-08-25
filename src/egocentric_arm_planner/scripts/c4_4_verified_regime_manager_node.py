#!/usr/bin/env python3
"""C4.4 verified planner regime manager.

Separates planner-candidate safety from committed-execution safety and owns the
NORMAL -> REPAIR -> PROBE_NORMAL -> NORMAL state machine.

Candidate VBC:
  * unsafe candidate: reject is handled by the verify/commit node; this manager
    only changes planner regime.
  * a new safe commit while REPAIR proves a repaired plan exists and releases
    the MPC into PROBE_NORMAL.
  * PROBE_NORMAL requires several consecutive *safe commits* before declaring
    NORMAL. Any unsafe probe candidate returns to REPAIR.

Execution VBC:
  * independently audits the trajectory actually published to the low-level
    tracker.
  * a confirmed unsafe committed trajectory is one execution-safety episode and
    immediately requests REPAIR from any state.

The legacy MPC still consumes Bool recovery_trigger/recovery_clear inputs. A
short clear pulse is used only as a compatibility transition from its RECOVERY
mode to normal-objective probing; it is no longer interpreted as "the problem is
gone". Unsafe candidate rejection and execution safety are deliberately kept as
separate counters/diagnostics.
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

        if self.rate <= 0.0:
            raise ValueError("~rate must be positive")
        if self.candidate_unsafe_required < 1 or self.execution_unsafe_required < 1:
            raise ValueError("unsafe streak requirements must be >= 1")
        if self.probe_safe_commits_required < 1:
            raise ValueError("probe_safe_commits_required must be >= 1")
        if min(self.input_timeout, self.committed_trajectory_timeout,
               self.clear_pulse_s, self.probe_ignore_s) <= 0.0:
            raise ValueError("timeouts/pulses must be positive")

        self.candidate_summary_topic = str(rospy.get_param(
            "~candidate_summary_topic", "/care_planner/candidate_vbc/summary"))
        self.execution_summary_topic = str(rospy.get_param(
            "~execution_summary_topic", "/care_planner/execution_vbc/summary"))
        self.commit_summary_topic = str(rospy.get_param(
            "~commit_summary_topic", "/care_planner/optimized_trajectory_summary"))
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
            "~trigger_topic", "/care_planner/execution/predicted_vbc_recovery_triggered"))
        self.clear_topic = str(rospy.get_param(
            "~clear_topic", "/care_planner/execution/predicted_vbc_recovery_clear"))
        self.verification_hold_topic = str(rospy.get_param(
            "~verification_hold_topic",
            "/care_planner/execution/predicted_vbc_verification_hold"))
        self.summary_topic = str(rospy.get_param(
            "~summary_topic", "/care_planner/c4_4/regime_summary"))

        self.state = self.NORMAL
        self.execution_ready = False
        self.execution_ready_time = None

        self.candidate_summary_time = None
        self.candidate_last_unsafe = False
        self.candidate_unsafe_streak = 0
        self.candidate_unsafe_count = 0
        self.candidate_safe_count = 0

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

        self.repair_entry_count = 0
        self.probe_entry_count = 0
        self.normal_entry_count = 0
        self.probe_failure_count = 0
        self.candidate_repair_entry_count = 0
        self.execution_repair_entry_count = 0

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
        self.deadline_pub = rospy.Publisher(
            self.effective_deadline_topic, Float64, queue_size=1)
        self.summary_pub = rospy.Publisher(
            self.summary_topic, String, queue_size=1, latch=True)

        rospy.Subscriber(
            self.candidate_summary_topic, String, self._candidate_summary_cb,
            queue_size=1)
        rospy.Subscriber(
            self.execution_summary_topic, String, self._execution_summary_cb,
            queue_size=1)
        rospy.Subscriber(
            self.commit_summary_topic, String, self._commit_summary_cb,
            queue_size=1)
        rospy.Subscriber(
            self.committed_trajectory_topic, rospy.AnyMsg,
            self._committed_trajectory_cb, queue_size=1)
        rospy.Subscriber(
            self.execution_ready_topic, Bool, self._execution_ready_cb,
            queue_size=1)
        rospy.Subscriber(
            self.physical_deadline_topic, Float64, self._deadline_cb,
            queue_size=1)

        self.trigger_pub.publish(Bool(data=False))
        self.clear_pub.publish(Bool(data=False))
        self.hold_pub.publish(Bool(data=False))
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate), self._timer_cb)
        self._publish_summary()

        rospy.logwarn(
            "[c4_4_regime] NORMAL->REPAIR->PROBE_NORMAL->NORMAL; "
            "candidate=%s execution=%s probe_safe_commits=%d",
            self.candidate_summary_topic, self.execution_summary_topic,
            self.probe_safe_commits_required)

    def _transition_locked(self, new_state, reason, now):
        if new_state == self.state:
            self.last_transition_reason = reason
            return
        old = self.state
        self.state = new_state
        self.last_transition_reason = reason

        if new_state == self.REPAIR:
            self.repair_entry_count += 1
            self.probe_safe_commit_streak = 0
            self.clear_until = None
            self.probe_ignore_until = None
            if reason.startswith("execution_"):
                self.execution_repair_entry_count += 1
            else:
                self.candidate_repair_entry_count += 1
        elif new_state == self.PROBE_NORMAL:
            self.probe_entry_count += 1
            self.probe_safe_commit_streak = 0
            self.clear_until = now + rospy.Duration(self.clear_pulse_s)
            self.probe_ignore_until = now + rospy.Duration(self.probe_ignore_s)
        elif new_state == self.NORMAL:
            self.normal_entry_count += 1
            self.probe_safe_commit_streak = 0
            self.clear_until = None
            self.probe_ignore_until = None

        rospy.logwarn(
            "[c4_4_regime] %s -> %s reason=%s repair_entries=%d probe_entries=%d",
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
                self.clear_until = None
                self.probe_ignore_until = None
                self.last_transition_reason = "execution_not_ready"

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

    def _candidate_summary_cb(self, msg):
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
            self.candidate_summary_time = now
            self.candidate_last_unsafe = bool(unsafe)
            if unsafe:
                self.candidate_unsafe_count += 1
                self.candidate_unsafe_streak += 1
            else:
                self.candidate_safe_count += 1
                self.candidate_unsafe_streak = 0

            if not self.execution_ready:
                return

            if self.state == self.NORMAL:
                if unsafe and self.candidate_unsafe_streak >= self.candidate_unsafe_required:
                    self._transition_locked(
                        self.REPAIR, "candidate_unsafe_confirmed", now)
            elif self.state == self.PROBE_NORMAL:
                if self.probe_ignore_until is not None and now < self.probe_ignore_until:
                    return
                if unsafe and self.candidate_unsafe_streak >= self.candidate_unsafe_required:
                    self.probe_failure_count += 1
                    self._transition_locked(
                        self.REPAIR, "candidate_probe_unsafe", now)
            # REPAIR exit is intentionally commit-driven, not summary-driven.

    def _execution_summary_cb(self, msg):
        if msg is None:
            return
        f = _tokens(msg.data)
        unsafe = _as_bool(f.get("has_violation"))
        if unsafe is None:
            return
        now = rospy.Time.now()
        with self._lock:
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

    def _commit_summary_cb(self, msg):
        if msg is None:
            return
        f = _tokens(msg.data)
        commit_count = _as_int(f.get("commit_count"), self.last_commit_count)
        source = f.get("source", "")
        now = rospy.Time.now()
        with self._lock:
            if commit_count <= self.last_commit_count:
                return
            delta = commit_count - self.last_commit_count
            self.last_commit_count = commit_count
            self.last_commit_event_time = now

            if source != "candidate_verified_safe_committed":
                return

            if self.state == self.REPAIR:
                self._transition_locked(
                    self.PROBE_NORMAL, "safe_repair_commit", now)
            elif self.state == self.PROBE_NORMAL:
                if self.probe_ignore_until is not None and now < self.probe_ignore_until:
                    return
                self.probe_safe_commit_streak += delta
                if self.probe_safe_commit_streak >= self.probe_safe_commits_required:
                    self._transition_locked(
                        self.NORMAL, "probe_normal_safe_commits", now)

    def _fresh_locked(self, stamp, now, timeout):
        if stamp is None:
            return False
        age = (now - stamp).to_sec()
        return 0.0 <= age <= timeout

    def _timer_cb(self, _event):
        now = rospy.Time.now()
        with self._lock:
            trigger = self.execution_ready and self.state == self.REPAIR
            clear = (
                self.execution_ready and self.state == self.PROBE_NORMAL and
                self.clear_until is not None and now <= self.clear_until)

            # Candidate-verifier staleness does not invalidate an already-safe
            # committed execution. Only execution-audit / committed-stream
            # staleness requests the MPC verification hold.
            exec_summary_fresh = self._fresh_locked(
                self.execution_summary_time, now, self.input_timeout)
            committed_fresh = self._fresh_locked(
                self.last_committed_trajectory_time, now,
                self.committed_trajectory_timeout)
            startup_grace = (
                self.execution_ready_time is not None and
                (now - self.execution_ready_time).to_sec() < self.input_timeout)
            hold = bool(
                self.execution_ready and not startup_grace and
                (not exec_summary_fresh or not committed_fresh))
            deadline = self.physical_deadline_abs

        self.trigger_pub.publish(Bool(data=trigger))
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
                "candidate_last_unsafe={}".format(int(self.candidate_last_unsafe)),
                "candidate_unsafe_streak={}".format(self.candidate_unsafe_streak),
                "candidate_unsafe_count={}".format(self.candidate_unsafe_count),
                "candidate_safe_count={}".format(self.candidate_safe_count),
                "execution_last_unsafe={}".format(int(self.execution_last_unsafe)),
                "execution_unsafe_streak={}".format(self.execution_unsafe_streak),
                "execution_event_latched={}".format(int(self.execution_event_latched)),
                "execution_safety_event_count={}".format(self.execution_safety_event_count),
                "repair_entry_count={}".format(self.repair_entry_count),
                "candidate_repair_entry_count={}".format(self.candidate_repair_entry_count),
                "execution_repair_entry_count={}".format(self.execution_repair_entry_count),
                "probe_entry_count={}".format(self.probe_entry_count),
                "probe_failure_count={}".format(self.probe_failure_count),
                "probe_safe_commit_streak={}".format(self.probe_safe_commit_streak),
                "normal_entry_count={}".format(self.normal_entry_count),
                "commit_count={}".format(self.last_commit_count),
                "candidate_summary_age_s={:.6f}".format(age(self.candidate_summary_time)),
                "execution_summary_age_s={:.6f}".format(age(self.execution_summary_time)),
                "committed_trajectory_age_s={:.6f}".format(age(self.last_committed_trajectory_time)),
            ])
        self.summary_pub.publish(msg)


def main():
    rospy.init_node("c4_4_verified_regime_manager")
    C44VerifiedRegimeManager()
    rospy.spin()


if __name__ == "__main__":
    main()
