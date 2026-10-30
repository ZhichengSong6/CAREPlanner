#!/usr/bin/env python3
"""Single-flight gate for PROBE_NORMAL candidate execution.

C5.9 execution ownership rule
-----------------------------
PROBE_NORMAL is a receding executable-prefix mode. At most one PROBE prefix may
be in verification/execution at a time:

    raw PROBE candidate -> VERIFYING -> committed -> EXECUTING
        -> certified task prefix complete -> next flight admitted

The committed brake+hold tail is still part of the exact GCDF/VBC-certified
trajectory. It remains the fail-safe execution while the next fresh probe is
planned/certified. No new probe is admitted before the current certified task
prefix has actually executed.

When candidate admission is enabled, all modes use an acknowledged, bounded
transfer to the downstream certifier with one latest pending candidate. A mode
change cancels uncommitted work under the receiver's commit mutex. REPAIR can
then submit fresh work without waiting for the old PROBE task prefix to finish.
"""

import copy
import math
import re
import threading
import time
import sys
from pathlib import Path

import rospy
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import PoseStamped
from egocentric_arm_planner.msg import CandidateTransfer
sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate_admission import CandidateSender, WallWorker


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


def _as_float(v, default=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class ProbeSingleFlightGate:
    IDLE = "IDLE"
    VERIFYING = "VERIFYING"
    EXECUTING = "EXECUTING"

    def __init__(self):
        self._lock = threading.RLock()

        self.input_topic = str(rospy.get_param(
            "~input_topic", "/care_planner/local_planner/candidate_trajectory"))
        self.output_topic = str(rospy.get_param(
            "~output_topic",
            "/care_planner/local_planner/candidate_trajectory_execution_gated"))
        self.probe_active_topic = str(rospy.get_param(
            "~probe_active_topic", "/care_planner/c4_4/probe_active"))
        self.verification_outcome_topic = str(rospy.get_param(
            "~verification_outcome_topic", "/care_planner/verification_outcome"))
        self.tracker_summary_topic = str(rospy.get_param(
            "~tracker_summary_topic", "/care_planner/execution/tracker_summary"))
        self.regime_summary_topic = str(rospy.get_param(
            "~regime_summary_topic", "/care_planner/c4_4/regime_summary"))
        self.replan_request_topic = str(rospy.get_param(
            "~replan_request_topic", "/care_planner/local_planner/replan_request"))
        self.lookahead_plan_lead_s = float(rospy.get_param(
            "~lookahead_plan_lead_s", 0.18))
        self.summary_topic = str(rospy.get_param(
            "~summary_topic", "/care_planner/execution/probe_single_flight_summary"))
        self.admission_enabled = bool(rospy.get_param("~candidate_admission_enabled", False))
        self.transfer_topic = self.output_topic + "/transfer"
        self.admission_status_topic = self.output_topic + "/admission_status"
        self.admission_ping_topic = self.output_topic + "/admission_ping"
        self.sender = CandidateSender() if self.admission_enabled else None
        self._summary_pending = None
        self._verifying_raw_stamp_ns = 0
        self._verifying_transfer_sequence = 0
        self._repair_active = False
        self._goal_identity = None

        self.probe_active = False
        self.phase = self.IDLE
        self.execution_stamp_ns = 0
        self.effective_prefix_s = float("nan")
        self.buffered_candidate = None
        self.lookahead_requested = False
        self.probe_completed_prefix_streak = 0
        self.probe_completed_prefixes_required = 3

        self.input_count = 0
        self.forward_count = 0
        self.drop_busy_count = 0
        self.verify_release_count = 0
        self.execution_complete_count = 0
        self.prefix_release_count = 0
        self.lookahead_request_count = 0
        self.buffer_count = 0
        self.buffer_replace_count = 0
        self.buffer_forward_count = 0
        self.mode_reset_count = 0
        self.abort_release_count = 0
        self.last_aborted_execution_stamp_ns = 0
        self.last_reason = "startup"

        self.output_pub = rospy.Publisher(
            self.output_topic, JointTrajectory, queue_size=1)
        self.summary_pub = rospy.Publisher(
            self.summary_topic, String, queue_size=1, latch=True)
        self.replan_request_pub = rospy.Publisher(
            self.replan_request_topic, Bool, queue_size=10, latch=False)
        if self.admission_enabled:
            self.transfer_pub = rospy.Publisher(self.transfer_topic, CandidateTransfer, queue_size=10)
            self.admission_terminal_pub = rospy.Publisher(
                self.input_topic + "/admission_terminal", CandidateTransfer, queue_size=8)
            self.admission_ready_pub = rospy.Publisher(self.admission_ping_topic + "/ready", String, queue_size=4)
            rospy.Subscriber(self.admission_status_topic, String, self._admission_status_cb, queue_size=20)
            rospy.Subscriber(self.admission_ping_topic, String, self._admission_ping_cb, queue_size=4)
            rospy.Subscriber("/care_planner/execution/predicted_vbc_recovery_triggered", Bool,
                             self._repair_active_cb, queue_size=1)
            rospy.Subscriber("/care_planner/ee_target_pose", PoseStamped, self._goal_cb, queue_size=1)
            rospy.Subscriber("/care_planner/local_planner/candidate_replacement_grant", String,
                             self._replacement_grant_cb, queue_size=8)

        self.prefix_binding_enabled = bool(rospy.get_param("~repair_prefix_binding_enabled", False))
        if self.prefix_binding_enabled:
            if not self.admission_enabled:
                raise ValueError("Prefix binding requires candidate admission")
            rospy.Subscriber(self.input_topic + "/planned", CandidateTransfer,
                             self._planned_candidate_cb, queue_size=1)
        rospy.Subscriber(
            self.input_topic, JointTrajectory, self._candidate_cb, queue_size=1)
        rospy.Subscriber(
            self.probe_active_topic, Bool, self._probe_active_cb, queue_size=1)
        rospy.Subscriber(
            self.verification_outcome_topic, String,
            self._verification_outcome_cb, queue_size=20)
        rospy.Subscriber(
            self.tracker_summary_topic, String,
            self._tracker_summary_cb, queue_size=20)
        rospy.Subscriber(
            self.regime_summary_topic, String,
            self._regime_summary_cb, queue_size=20)

        self._publish_summary()
        if self.admission_enabled:
            self.wall_worker = WallWorker(self._admission_tick,
                lambda exc: rospy.logerr("[candidate_admission sender] %s", exc))
            rospy.on_shutdown(self.wall_worker.stop.set)
        rospy.logwarn(
            "[probe_single_flight_gate] input=%s output=%s",
            self.input_topic, self.output_topic)

    def _reset_locked(self, reason):
        if self.phase != self.IDLE or self.execution_stamp_ns != 0:
            self.mode_reset_count += 1
        self.phase = self.IDLE
        self.execution_stamp_ns = 0
        self.effective_prefix_s = float("nan")
        self.buffered_candidate = None
        self.lookahead_requested = False
        self._verifying_raw_stamp_ns = 0
        self._verifying_transfer_sequence = 0
        self.last_reason = reason

    def _probe_active_cb(self, msg):
        if msg is None:
            return
        value = bool(msg.data)
        actions = []
        with self._lock:
            if self.probe_active == value:
                return
            self.probe_active = value
            if self.sender is not None:
                actions = self.sender.invalidate(time.monotonic())
            if not value:
                # Leaving PROBE cancels serialization immediately.  A REPAIR
                # candidate is allowed to preempt the old probe execution.
                self._reset_locked("leave_probe_reset")
            else:
                self._reset_locked("enter_probe_idle")
            self._publish_summary_locked()
        self._emit_admission_actions(actions)

    def _repair_active_cb(self, msg):
        with self._lock:
            if self._repair_active == bool(msg.data):
                return
            self._repair_active = bool(msg.data)
            actions = self.sender.invalidate(time.monotonic())
        self._emit_admission_actions(actions)

    def _goal_cb(self, msg):
        p, q = msg.pose.position, msg.pose.orientation
        identity = (msg.header.frame_id, p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        with self._lock:
            if identity == self._goal_identity:
                return
            self._goal_identity = identity
            actions = self.sender.invalidate(time.monotonic())
        self._emit_admission_actions(actions)

    def _replacement_grant_cb(self, msg):
        if _tokens(msg.data).get("granted") != "1":
            return
        with self._lock:
            actions = self.sender.invalidate(time.monotonic())
        self._emit_admission_actions(actions)

    def _admission_ping_cb(self, msg):
        nonce = str(msg.data)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", nonce):
            return
        self.transfer_pub.publish(self.sender.ping(nonce))

    def _emit_admission_actions(self, actions):
        # No publisher call holds the gate's state mutex.
        for action, msg in actions:
            if action == "transfer":
                if msg.action == "offer":
                    with self._lock:
                        if self.probe_active and msg.trajectory.header.stamp.to_nsec() == self._verifying_raw_stamp_ns:
                            self._verifying_transfer_sequence = int(msg.sequence)
                self.transfer_pub.publish(msg)
                if msg.action == "deliver":
                    self.output_pub.publish(copy.deepcopy(msg.trajectory))  # raw diagnostics
            elif action == "replan":
                self.replan_request_pub.publish(Bool(data=True))
            elif action == "preaudit_terminal":
                self.admission_terminal_pub.publish(msg)
        self._flush_summary()

    def _admission_status_cb(self, msg):
        fields = _tokens(msg.data)
        if any(fields[k] != v for k,v in _TOKEN_RE.findall(msg.data or "")):
            return  # Conflicting repeated fields cannot authenticate a receipt.
        with self._lock:
            if fields.get("action") == "pong":
                boot = fields.get("receiver_boot_id", "")
                if not re.fullmatch(r"[a-f0-9]{32}", boot):
                    return
                actions = self.sender.pong(boot, time.monotonic())
                terminal = False
            else:
                actions, terminal = self.sender.status(fields, time.monotonic())
            # Bind queued work before interpreting the old cancellation outcome.
            for action, transfer in actions:
                if (action == "transfer" and transfer.action == "offer" and self.probe_active and
                        transfer.trajectory.header.stamp.to_nsec() == self._verifying_raw_stamp_ns):
                    self._verifying_transfer_sequence = int(transfer.sequence)
            self._publish_summary_locked()
        if terminal and fields.get("result"):
            self._verification_outcome_cb(msg)
        self._emit_admission_actions(actions)
        if fields.get("action") == "pong" and not self.sender.fault:
            self.admission_ready_pub.publish(String(data="ready_nonce={} receiver_boot_id={}".format(
                fields.get("ready_nonce", "none"), boot)))

    def _admission_tick(self):
        with self._lock:
            actions = self.sender.tick(time.monotonic())
            self._publish_summary_locked()
        self._emit_admission_actions(actions)

    def _planned_candidate_cb(self, msg):
        if msg.action == "planned":
            self._candidate_cb(msg.trajectory, msg.planned_prefix_context)

    def _forward_candidate(self, msg, prefix_context=""):
        with self._lock:
            if self.probe_active:
                self._verifying_raw_stamp_ns = msg.header.stamp.to_nsec()
            actions = self.sender.forward(msg, time.monotonic(), prefix_context) if self.sender else []
        if self.sender is not None:
            self._emit_admission_actions(actions)
        else:
            self.output_pub.publish(msg)

    def _candidate_cb(self, msg, prefix_context=None):
        if getattr(self, 'prefix_binding_enabled', False) and prefix_context is None:
            return  # Raw diagnostic topic cannot bypass the bound transport.
        prefix_context = prefix_context or ""
        if msg is None or not msg.points:
            return

        publish = False
        with self._lock:
            self.input_count += 1
            if not self.probe_active:
                self.forward_count += 1
                self.last_reason = "non_probe_passthrough"
                publish = True
            elif self.phase == self.IDLE:
                self.phase = self.VERIFYING
                self.execution_stamp_ns = 0
                self.forward_count += 1
                self.last_reason = "probe_candidate_forwarded_verify"
                publish = True
            elif (self.phase == self.EXECUTING and
                  (self.lookahead_requested or
                   self.probe_completed_prefix_streak + 1 <
                   self.probe_completed_prefixes_required)):
                # C5.40: planner work may happen before the certified prefix
                # ends, but the raw candidate is NOT admitted to safety or
                # execution yet. Keep only the freshest one-slot look-ahead.
                # The streak check covers the short interval before the regime
                # summary reaches this node after a new prefix is committed.
                if self.buffered_candidate is None:
                    self.buffer_count += 1
                else:
                    self.buffer_replace_count += 1
                self.buffered_candidate = copy.deepcopy(msg)
                self.last_reason = "probe_candidate_buffered_lookahead"
            else:
                self.drop_busy_count += 1
                self.last_reason = "probe_candidate_dropped_{}".format(
                    self.phase.lower())
            self._publish_summary_locked()

        if publish:
            self._forward_candidate(msg, prefix_context)

    def _verification_outcome_cb(self, msg):
        if msg is None:
            return
        f = _tokens(msg.data)
        view = f.get("verification_view", "")
        if view != "probe_prefix_brake_hold":
            return

        result = f.get("result", "")
        committed = _as_bool(f.get("committed"))
        execution_stamp_ns = _as_int(f.get("execution_stamp_ns"), 0)
        effective_prefix_s = _as_float(
            f.get("probe_effective_prefix_s"), float("nan"))

        request_lookahead = False
        with self._lock:
            if not self.probe_active or self.phase != self.VERIFYING:
                return
            if (self.admission_enabled and
                    _as_int(f.get("raw_candidate_stamp_ns")) != self._verifying_raw_stamp_ns):
                return
            if (self.admission_enabled and
                    (f.get("transfer_session", f.get("session")) != self.sender.session or
                     _as_int(f.get("transfer_sequence", f.get("sequence"))) != self._verifying_transfer_sequence)):
                return

            if result == "safe" and committed is True and execution_stamp_ns > 0:
                self.phase = self.EXECUTING
                self.execution_stamp_ns = execution_stamp_ns
                self.effective_prefix_s = effective_prefix_s
                self.buffered_candidate = None
                # Schedule preplanning from measured execution phase. A raw
                # candidate built at prefix entry can have an obsolete start
                # by handoff, especially after executable time scaling.
                another_probe_needed = (
                    self.probe_completed_prefix_streak + 1 <
                    self.probe_completed_prefixes_required)
                self.lookahead_requested = False
                self.last_reason = ("probe_verified_wait_lookahead_phase"
                    if another_probe_needed else "probe_verified_wait_prefix")
            elif result in ("unsafe", "timeout") or committed is False:
                # No executable trajectory exists, so a fresh PROBE candidate
                # may be considered immediately when the planner replans.
                self.phase = self.IDLE
                self.execution_stamp_ns = 0
                self.verify_release_count += 1
                self.last_reason = "probe_verification_released_{}".format(result)
            self._publish_summary_locked()

        if request_lookahead:
            self.replan_request_pub.publish(Bool(data=True))

    def _tracker_summary_cb(self, msg):
        if msg is None:
            return
        f = _tokens(msg.data)
        complete = _as_bool(f.get("complete"))
        aborted = _as_bool(f.get("execution_aborted"))
        source = f.get("source", "")
        execution_stamp_ns = _as_int(f.get("execution_stamp_ns"), 0)
        phase_s = _as_float(f.get("phase_s"), float("nan"))

        candidate_to_publish = None
        request_lookahead = False

        with self._lock:
            if (not self.probe_active or self.phase != self.EXECUTING or
                    self.execution_stamp_ns <= 0):
                return
            if execution_stamp_ns != self.execution_stamp_ns:
                return

            # A tracking-envelope stop is an explicit cancellation of this
            # execution token. It is not a short prefix completion, even when
            # the reported phase happens to be at/after the prefix boundary.
            # Release the single-flight owner immediately so the manager can
            # submit a fresh measured-state candidate.
            if aborted is True:
                self.phase = self.IDLE
                self.execution_stamp_ns = 0
                self.effective_prefix_s = float("nan")
                self.buffered_candidate = None
                self.lookahead_requested = False
                self.abort_release_count += 1
                self.last_aborted_execution_stamp_ns = execution_stamp_ns
                self.last_reason = (
                    "probe_execution_aborted_token_{}".format(
                        execution_stamp_ns))
                self._publish_summary_locked()
                return

            prefix_complete = (
                math.isfinite(self.effective_prefix_s) and
                self.effective_prefix_s > 0.0 and
                math.isfinite(phase_s) and
                phase_s + 1e-6 >= self.effective_prefix_s)
            full_complete = (
                complete is True and source == "trajectory_complete_hold")

            # Only preplan when another PROBE will still be required after this
            # certified prefix. The final probe therefore cannot leak a stale
            # fourth probe across PROBE->NORMAL.
            another_probe_needed = (
                self.probe_completed_prefix_streak + 1 <
                self.probe_completed_prefixes_required)
            lookahead_due = (
                another_probe_needed and
                math.isfinite(self.effective_prefix_s) and
                math.isfinite(phase_s) and
                phase_s + 1e-6 >= max(
                    0.0, self.effective_prefix_s -
                    min(max(0.0, self.lookahead_plan_lead_s),
                        .5*self.effective_prefix_s)))

            if lookahead_due and not self.lookahead_requested:
                self.lookahead_requested = True
                self.lookahead_request_count += 1
                request_lookahead = True
                self.last_reason = "probe_lookahead_replan_requested"
                self._publish_summary_locked()

            if not prefix_complete and not full_complete:
                # Planning may happen now, but current certified execution
                # ownership remains unchanged.
                pass
            else:
                self.prefix_release_count += 1
                self.execution_stamp_ns = 0
                self.effective_prefix_s = float("nan")

                if self.buffered_candidate is not None and another_probe_needed:
                    # Admit the preplanned RAW candidate only now. Downstream
                    # will rebase it to current measured q, rebuild the exact
                    # prefix+brake+hold executable, then run final GCDF + exact
                    # VBC before any commit. No safety authority is bypassed.
                    candidate_to_publish = self.buffered_candidate
                    self.buffered_candidate = None
                    self.phase = self.VERIFYING
                    self.lookahead_requested = False
                    self.forward_count += 1
                    self.buffer_forward_count += 1
                    self.last_reason = (
                        "probe_prefix_complete_forward_buffered_verify")
                else:
                    # A stale regime summary may have allowed a one-slot
                    # look-ahead to be buffered just before the final prefix.
                    # Never leak that fourth candidate across PROBE->NORMAL.
                    self.buffered_candidate = None
                    self.phase = self.IDLE
                    self.lookahead_requested = False
                    if full_complete:
                        self.execution_complete_count += 1
                        self.last_reason = (
                            "probe_execution_complete_fallback_release")
                    else:
                        self.last_reason = "probe_prefix_complete_release"
                self._publish_summary_locked()

        if request_lookahead:
            self.replan_request_pub.publish(Bool(data=True))
        if candidate_to_publish is not None:
            self._forward_candidate(candidate_to_publish)

    def _regime_summary_cb(self, msg):
        if msg is None:
            return
        f = _tokens(msg.data)
        streak = _as_int(f.get("probe_completed_prefix_streak"), -1)
        required = _as_int(f.get("probe_completed_prefixes_required"), -1)
        if streak < 0 or required < 1:
            return
        with self._lock:
            self.probe_completed_prefix_streak = streak
            self.probe_completed_prefixes_required = required

    def _publish_summary_locked(self):
        msg = String()
        msg.data = " ".join([
            "probe_active={}".format(int(self.probe_active)),
            "phase={}".format(self.phase),
            "execution_stamp_ns={}".format(self.execution_stamp_ns),
            "input_count={}".format(self.input_count),
            "forward_count={}".format(self.forward_count),
            "drop_busy_count={}".format(self.drop_busy_count),
            "verify_release_count={}".format(self.verify_release_count),
            "execution_complete_count={}".format(self.execution_complete_count),
            "prefix_release_count={}".format(self.prefix_release_count),
            "lookahead_request_count={}".format(self.lookahead_request_count),
            "buffered_candidate={}".format(
                int(self.buffered_candidate is not None)),
            "buffer_count={}".format(self.buffer_count),
            "buffer_replace_count={}".format(self.buffer_replace_count),
            "buffer_forward_count={}".format(self.buffer_forward_count),
            "probe_completed_prefix_streak={}".format(
                self.probe_completed_prefix_streak),
            "probe_completed_prefixes_required={}".format(
                self.probe_completed_prefixes_required),
            "effective_prefix_s={}".format(
                "nan" if not math.isfinite(self.effective_prefix_s)
                else "{:.6f}".format(self.effective_prefix_s)),
            "mode_reset_count={}".format(self.mode_reset_count),
            "abort_release_count={}".format(self.abort_release_count),
            "last_aborted_execution_stamp_ns={}".format(
                self.last_aborted_execution_stamp_ns),
            "reason={}".format(self.last_reason),
        ])
        if self.sender is not None:
            a = self.sender.active or {}
            msg.data += " admission_enabled=1 admission_stage={} transfer_sequence={} receipt_count={} resend_count={} admission_timeout_count={} transport_replan_count={} transport_fault={} transfer_connections={}".format(
                self.sender.last_stage, a.get("sequence", 0), self.sender.receipt_count,
                self.sender.resend_count, self.sender.timeout_count, self.sender.replan_count,
                int(self.sender.fault), self.transfer_pub.get_num_connections())
            self._summary_pending = msg
        else:
            self.summary_pub.publish(msg)

    def _flush_summary(self):
        with self._lock:
            msg = self._summary_pending
            self._summary_pending = None
        if msg is not None:
            self.summary_pub.publish(msg)

    def _publish_summary(self):
        with self._lock:
            self._publish_summary_locked()
        self._flush_summary()


def main():
    rospy.init_node("probe_single_flight_gate")
    ProbeSingleFlightGate()
    rospy.spin()


if __name__ == "__main__":
    main()
