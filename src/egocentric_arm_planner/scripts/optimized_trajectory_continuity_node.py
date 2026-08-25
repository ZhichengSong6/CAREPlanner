#!/usr/bin/env python3
"""Serialize candidate VBC verification and publish only committed trajectories.

C4.4 verification protocol
--------------------------
The C++ VBC selector runs on its own periodic timer and may publish repeated
summaries for the same cached trajectory. A millisecond delay is therefore not
a reliable way to associate a selector verdict with a newly published candidate.

This node uses an explicit selector-cycle barrier instead:

  selector cycle k finishes and publishes a summary
      -> a new candidate may be dispatched immediately after that cycle
      -> only selector cycle k+1 (or later) may decide that candidate

At most one candidate is outstanding. Each dispatched candidate also receives a
monotone ``header.seq`` for diagnostics. Every completed unique verification
publishes exactly one event on ``verification_event_topic``. The C4.4 regime
manager consumes those events rather than repeated selector summaries.

Unsafe candidates are never published to the low-level controller. While a new
safe candidate is being generated / verified, the remaining suffix of the last
committed trajectory is re-timed and republished for bounded plan memory.
"""

from __future__ import annotations

import copy
import math
import re
import threading

import rospy
from std_msgs.msg import String
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
        self.global_summary_topic = str(rospy.get_param(
            "~global_summary_topic", "/care_planner/trajectory_risk/vbc_summary"))
        self.summary_topic = str(rospy.get_param(
            "~summary_topic", "/care_planner/optimized_trajectory_summary"))
        self.verification_event_topic = str(rospy.get_param(
            "~verification_event_topic", "/care_planner/verification_outcome"))

        self.rate = float(rospy.get_param("~rate", 20.0))
        self.continuation_start_delay_s = float(rospy.get_param(
            "~continuation_start_delay_s", 0.065))
        self.continuation_timeout_s = float(rospy.get_param(
            "~continuation_timeout_s", 0.50))
        self.verification_timeout_s = float(rospy.get_param(
            "~verification_timeout_s", 0.25))

        if self.rate <= 0.0:
            raise ValueError("~rate must be positive")
        if self.continuation_start_delay_s <= 0.0:
            raise ValueError("~continuation_start_delay_s must be positive")
        if self.continuation_timeout_s <= self.continuation_start_delay_s:
            raise ValueError("~continuation_timeout_s must exceed start delay")
        if self.verification_timeout_s <= 0.0:
            raise ValueError("~verification_timeout_s must be positive")

        self._pending_raw = None
        self._pending_raw_received = None
        self._last_raw_received = None
        self._last_input_gap_s = math.nan
        self._max_input_gap_s = 0.0

        self._selector_cycle_count = 0
        self._selector_barrier_ready = False
        self._last_selector_cycle_time = None

        self._outstanding = None
        self._outstanding_sent = None
        self._outstanding_seq = 0
        self._outstanding_dispatch_cycle = -1
        self._next_verification_seq = 1

        self._committed_master = None
        self._committed_received = None

        self._raw_input_count = 0
        self._verification_publish_count = 0
        self._verification_safe_count = 0
        self._verification_unsafe_count = 0
        self._verification_timeout_count = 0
        self._verification_outcome_count = 0
        self._commit_count = 0
        self._committed_publish_count = 0
        self._continuation_count = 0
        self._pending_replace_count = 0
        self._last_source = "waiting_raw_candidate"
        self._last_committed_age_s = math.nan
        self._last_verification_age_s = math.nan
        self._last_verification_seq = 0
        self._last_verification_result = "none"

        self.verification_pub = rospy.Publisher(
            self.verification_topic, JointTrajectory, queue_size=1)
        self.committed_pub = rospy.Publisher(
            self.committed_topic, JointTrajectory, queue_size=1)
        self.summary_pub = rospy.Publisher(
            self.summary_topic, String, queue_size=1, latch=True)
        self.verification_event_pub = rospy.Publisher(
            self.verification_event_topic, String, queue_size=20, latch=False)

        self.raw_sub = rospy.Subscriber(
            self.input_topic, JointTrajectory, self._raw_cb, queue_size=1)
        self.summary_sub = rospy.Subscriber(
            self.global_summary_topic, String, self._global_summary_cb, queue_size=1)
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.rate), self._timer_cb)

        self._publish_summary()
        rospy.logwarn(
            "[optimized_trajectory_continuity] CYCLE-CORRELATED VERIFY raw=%s "
            "verify=%s committed=%s selector_summary=%s event=%s rate=%.1fHz "
            "verify_timeout=%.3fs continuation_timeout=%.3fs",
            self.input_topic, self.verification_topic, self.committed_topic,
            self.global_summary_topic, self.verification_event_topic,
            self.rate, self.verification_timeout_s, self.continuation_timeout_s)

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

        if len(points) == 1 or phase <= points[0].time_from_start.to_sec():
            start_point = copy.deepcopy(points[0])
        elif phase >= points[-1].time_from_start.to_sec():
            start_point = copy.deepcopy(points[-1])
        else:
            hi = 1
            while hi < len(points) and points[hi].time_from_start.to_sec() < phase:
                hi += 1
            lo = hi - 1
            t0 = points[lo].time_from_start.to_sec()
            t1 = points[hi].time_from_start.to_sec()
            alpha = 0.0 if t1 <= t0 else (phase - t0) / (t1 - t0)
            start_point = cls._interpolate_point(points[lo], points[hi], alpha)

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
            self._last_raw_received = now
            self._raw_input_count += 1
            self._last_source = "raw_candidate_buffered"
            self._publish_summary_locked()

    def _dispatch_pending_locked(self, now):
        if self._outstanding is not None or self._pending_raw is None:
            return None
        if not self._selector_barrier_ready:
            return None

        raw = self._pending_raw
        raw_received = self._pending_raw_received
        self._pending_raw = None
        self._pending_raw_received = None
        age = max(0.0, (now - raw_received).to_sec())
        candidate = self._suffix_from_phase(raw, age)
        if candidate is None or not candidate.points:
            self._last_source = "discarded_expired_raw_candidate"
            return None

        seq = self._next_verification_seq
        self._next_verification_seq += 1
        candidate.header.seq = seq
        candidate.header.stamp = now
        self._outstanding = copy.deepcopy(candidate)
        self._outstanding_sent = now
        self._outstanding_seq = seq
        self._outstanding_dispatch_cycle = self._selector_cycle_count
        self._selector_barrier_ready = False
        self._verification_publish_count += 1
        self._last_source = "candidate_sent_for_verification"
        self._publish_summary_locked()
        return candidate

    def _publish_committed(self, msg, source, age_s):
        if msg is None or not msg.points:
            return
        self.committed_pub.publish(msg)
        with self._lock:
            self._committed_publish_count += 1
            if source == "committed_continuation":
                self._continuation_count += 1
            self._last_source = source
            self._last_committed_age_s = float(age_s)
            self._publish_summary_locked()

    def _make_verification_event(self, seq, result, committed, age_s):
        msg = String()
        msg.data = (
            "seq={} result={} committed={} verification_age_s={:.6f} "
            "outcome_count={} safe_count={} unsafe_count={} timeout_count={} "
            "commit_count={}"
        ).format(
            int(seq), result, int(bool(committed)), float(age_s),
            self._verification_outcome_count, self._verification_safe_count,
            self._verification_unsafe_count, self._verification_timeout_count,
            self._commit_count)
        return msg

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
        with self._lock:
            # Bootstrap cycles are allowed to establish the very first barrier.
            # Once a candidate is outstanding, however, only a later *predicted*
            # cycle may decide it.
            self._selector_cycle_count += 1
            self._last_selector_cycle_time = now

            if self._outstanding is None or self._outstanding_sent is None:
                self._selector_barrier_ready = True
                self._last_source = "selector_cycle_barrier_ready"
                self._publish_summary_locked()
                return

            if source != "predicted":
                return
            if self._selector_cycle_count <= self._outstanding_dispatch_cycle:
                return

            verification_age = max(0.0, (now - self._outstanding_sent).to_sec())
            candidate = copy.deepcopy(self._outstanding)
            seq = int(self._outstanding_seq)
            self._outstanding = None
            self._outstanding_sent = None
            self._outstanding_seq = 0
            self._outstanding_dispatch_cycle = -1
            self._selector_barrier_ready = True
            self._last_verification_age_s = verification_age
            self._last_verification_seq = seq
            self._verification_outcome_count += 1

            if violation:
                self._verification_unsafe_count += 1
                self._last_verification_result = "unsafe"
                self._last_source = "candidate_rejected_vbc_unsafe"
                event_to_publish = self._make_verification_event(
                    seq, "unsafe", False, verification_age)
            else:
                committed = self._suffix_from_phase(candidate, verification_age)
                if committed is not None and committed.points:
                    self._verification_safe_count += 1
                    self._last_verification_result = "safe"
                    committed.header.seq = seq
                    committed.header.stamp = now
                    self._committed_master = copy.deepcopy(committed)
                    self._committed_received = now
                    self._commit_count += 1
                    self._last_committed_age_s = 0.0
                    self._last_source = "candidate_verified_safe_committed"
                    committed_to_publish = copy.deepcopy(committed)
                    event_to_publish = self._make_verification_event(
                        seq, "safe", True, verification_age)
                else:
                    self._verification_timeout_count += 1
                    self._last_verification_result = "expired_safe"
                    self._last_source = "candidate_safe_but_expired_before_commit"
                    event_to_publish = self._make_verification_event(
                        seq, "timeout", False, verification_age)
            self._publish_summary_locked()

        if committed_to_publish is not None:
            self._publish_committed(
                committed_to_publish, "candidate_verified_safe_committed", 0.0)
        if event_to_publish is not None:
            self.verification_event_pub.publish(event_to_publish)

    def _timer_cb(self, _event):
        now = rospy.Time.now()
        verification_to_publish = None
        continuation_to_publish = None
        timeout_event = None
        continuation_age = math.nan

        with self._lock:
            if self._outstanding is not None and self._outstanding_sent is not None:
                age = (now - self._outstanding_sent).to_sec()
                if age > self.verification_timeout_s:
                    seq = int(self._outstanding_seq)
                    self._outstanding = None
                    self._outstanding_sent = None
                    self._outstanding_seq = 0
                    self._outstanding_dispatch_cycle = -1
                    self._selector_barrier_ready = False
                    self._verification_timeout_count += 1
                    self._verification_outcome_count += 1
                    self._last_verification_seq = seq
                    self._last_verification_result = "timeout"
                    self._last_source = "verification_timeout_candidate_rejected"
                    self._last_verification_age_s = age
                    timeout_event = self._make_verification_event(
                        seq, "timeout", False, age)

            verification_to_publish = self._dispatch_pending_locked(now)

            if self._committed_master is not None and self._committed_received is not None:
                continuation_age = (now - self._committed_received).to_sec()
                if (continuation_age >= self.continuation_start_delay_s and
                        continuation_age <= self.continuation_timeout_s):
                    continuation_to_publish = self._suffix_from_phase(
                        self._committed_master, continuation_age)
                elif continuation_age > self.continuation_timeout_s:
                    self._last_source = "committed_plan_stale_stop_publish"
                    self._last_committed_age_s = continuation_age
            self._publish_summary_locked()

        if verification_to_publish is not None:
            self.verification_pub.publish(verification_to_publish)
        if continuation_to_publish is not None:
            self._publish_committed(
                continuation_to_publish, "committed_continuation", continuation_age)
        if timeout_event is not None:
            self.verification_event_pub.publish(timeout_event)

    def _publish_summary_locked(self):
        msg = String()
        msg.data = " ".join([
            "source={}".format(self._last_source),
            "raw_input_count={}".format(self._raw_input_count),
            "pending_candidate={}".format(int(self._pending_raw is not None)),
            "pending_replace_count={}".format(self._pending_replace_count),
            "selector_cycle_count={}".format(self._selector_cycle_count),
            "selector_barrier_ready={}".format(int(self._selector_barrier_ready)),
            "verification_outstanding={}".format(int(self._outstanding is not None)),
            "outstanding_seq={}".format(int(self._outstanding_seq)),
            "verification_publish_count={}".format(self._verification_publish_count),
            "verification_outcome_count={}".format(self._verification_outcome_count),
            "verification_safe_count={}".format(self._verification_safe_count),
            "verification_unsafe_count={}".format(self._verification_unsafe_count),
            "verification_timeout_count={}".format(self._verification_timeout_count),
            "last_verification_seq={}".format(self._last_verification_seq),
            "last_verification_result={}".format(self._last_verification_result),
            "commit_count={}".format(self._commit_count),
            "has_committed_plan={}".format(int(self._committed_master is not None)),
            "committed_publish_count={}".format(self._committed_publish_count),
            "continuation_count={}".format(self._continuation_count),
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
