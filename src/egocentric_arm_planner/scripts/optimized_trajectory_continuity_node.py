#!/usr/bin/env python3
"""Serialize candidate VBC verification and publish only committed trajectories.

CAREPlanner's MPC produces raw receding-horizon candidates.  A raw candidate is
NOT an execution reference until the global VBC selector has evaluated it.
This node therefore separates three streams:

  raw MPC candidate
      -> one-at-a-time verification candidate
      -> committed trajectory (only if global VBC safe)

At most one verification candidate is outstanding.  This makes the existing
selector summary unambiguous without changing the C++ selector protocol: every
fresh ``trajectory_source=predicted`` summary received while a candidate is
outstanding belongs to that candidate.  After a result is consumed, one selector
period is deliberately left empty before the next candidate is dispatched so a
late repeated summary cannot be mistaken for the next candidate.

Unsafe candidates are never published to the low-level controller.  While a new
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


_TOKEN_RE = re.compile(r"([A-Za-z0-9_]+)=([^\\s]+)")


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

        # Backward-compatible launch naming: output_topic is now explicitly the
        # serialized candidate stream consumed by the VBC selector.
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

        self.rate = float(rospy.get_param("~rate", 20.0))
        self.continuation_start_delay_s = float(rospy.get_param(
            "~continuation_start_delay_s", 0.065))
        self.continuation_timeout_s = float(rospy.get_param(
            "~continuation_timeout_s", 0.50))
        self.verification_timeout_s = float(rospy.get_param(
            "~verification_timeout_s", 0.25))
        self.verification_rearm_delay_s = float(rospy.get_param(
            "~verification_rearm_delay_s", 0.055))
        self.verification_min_response_delay_s = float(rospy.get_param(
            "~verification_min_response_delay_s", 0.002))

        if self.rate <= 0.0:
            raise ValueError("~rate must be positive")
        if self.continuation_start_delay_s <= 0.0:
            raise ValueError("~continuation_start_delay_s must be positive")
        if self.continuation_timeout_s <= self.continuation_start_delay_s:
            raise ValueError("~continuation_timeout_s must exceed start delay")
        if self.verification_timeout_s <= 0.0:
            raise ValueError("~verification_timeout_s must be positive")
        if self.verification_rearm_delay_s < 0.0:
            raise ValueError("~verification_rearm_delay_s must be non-negative")
        if self.verification_min_response_delay_s < 0.0:
            raise ValueError("~verification_min_response_delay_s must be non-negative")

        # Latest raw candidate replaces any older candidate that has not yet
        # been dispatched for verification.
        self._pending_raw = None
        self._pending_raw_received = None
        self._last_raw_received = None
        self._last_input_gap_s = math.nan
        self._max_input_gap_s = 0.0

        # Exactly one candidate can be visible to the selector at once.
        self._outstanding = None
        self._outstanding_sent = None
        self._next_verification_not_before = rospy.Time(0)

        # Last VBC-safe execution plan and its commit epoch.
        self._committed_master = None
        self._committed_received = None

        self._raw_input_count = 0
        self._verification_publish_count = 0
        self._verification_safe_count = 0
        self._verification_unsafe_count = 0
        self._verification_timeout_count = 0
        self._commit_count = 0
        self._committed_publish_count = 0
        self._continuation_count = 0
        self._pending_replace_count = 0
        self._last_source = "waiting_raw_candidate"
        self._last_committed_age_s = math.nan
        self._last_verification_age_s = math.nan

        self.verification_pub = rospy.Publisher(
            self.verification_topic, JointTrajectory, queue_size=1)
        self.committed_pub = rospy.Publisher(
            self.committed_topic, JointTrajectory, queue_size=1)
        self.summary_pub = rospy.Publisher(
            self.summary_topic, String, queue_size=1, latch=True)

        self.raw_sub = rospy.Subscriber(
            self.input_topic, JointTrajectory, self._raw_cb, queue_size=1)
        self.summary_sub = rospy.Subscriber(
            self.global_summary_topic, String, self._global_summary_cb, queue_size=1)
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.rate), self._timer_cb)

        self._publish_summary()
        rospy.logwarn(
            "[optimized_trajectory_continuity] VERIFY-BEFORE-COMMIT raw=%s "
            "verify=%s committed=%s selector_summary=%s rate=%.1fHz "
            "verify_timeout=%.3fs rearm_delay=%.3fs continuation_timeout=%.3fs",
            self.input_topic, self.verification_topic, self.committed_topic,
            self.global_summary_topic, self.rate, self.verification_timeout_s,
            self.verification_rearm_delay_s, self.continuation_timeout_s)

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
        return [
            (1.0 - alpha) * float(a[i]) + alpha * float(b[i])
            for i in range(len(a))
        ]

    @classmethod
    def _interpolate_point(cls, p0, p1, alpha):
        out = JointTrajectoryPoint()
        out.positions = cls._lerp_array(p0.positions, p1.positions, alpha)
        out.velocities = cls._lerp_array(p0.velocities, p1.velocities, alpha)
        out.accelerations = cls._lerp_array(
            p0.accelerations, p1.accelerations, alpha)
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
        if now < self._next_verification_not_before:
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

        candidate.header.stamp = now
        self._outstanding = copy.deepcopy(candidate)
        self._outstanding_sent = now
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

    def _global_summary_cb(self, msg):
        if msg is None:
            return
        fields = _tokens(msg.data)
        if fields.get("trajectory_source") != "predicted":
            return
        violation = _as_bool(fields.get("has_violation"))
        if violation is None:
            return

        now = rospy.Time.now()
        committed_to_publish = None
        verification_age = math.nan
        with self._lock:
            if self._outstanding is None or self._outstanding_sent is None:
                return
            verification_age = (now - self._outstanding_sent).to_sec()
            if verification_age < self.verification_min_response_delay_s:
                return

            candidate = copy.deepcopy(self._outstanding)
            self._outstanding = None
            self._outstanding_sent = None
            self._next_verification_not_before = (
                now + rospy.Duration(self.verification_rearm_delay_s))
            self._last_verification_age_s = verification_age

            if violation:
                self._verification_unsafe_count += 1
                self._last_source = "candidate_rejected_vbc_unsafe"
            else:
                self._verification_safe_count += 1
                # The selector evaluated the outstanding candidate from its send
                # epoch.  Shift by the audit latency before making it executable.
                committed = self._suffix_from_phase(candidate, verification_age)
                if committed is not None and committed.points:
                    committed.header.stamp = now
                    self._committed_master = copy.deepcopy(committed)
                    self._committed_received = now
                    self._commit_count += 1
                    self._last_committed_age_s = 0.0
                    self._last_source = "candidate_verified_safe_committed"
                    committed_to_publish = copy.deepcopy(committed)
            self._publish_summary_locked()

        if committed_to_publish is not None:
            self._publish_committed(
                committed_to_publish, "candidate_verified_safe_committed", 0.0)

    def _timer_cb(self, _event):
        now = rospy.Time.now()
        verification_to_publish = None
        continuation_to_publish = None
        continuation_age = math.nan

        with self._lock:
            if self._outstanding is not None and self._outstanding_sent is not None:
                age = (now - self._outstanding_sent).to_sec()
                if age > self.verification_timeout_s:
                    self._outstanding = None
                    self._outstanding_sent = None
                    self._verification_timeout_count += 1
                    self._next_verification_not_before = (
                        now + rospy.Duration(self.verification_rearm_delay_s))
                    self._last_source = "verification_timeout_candidate_rejected"
                    self._last_verification_age_s = age

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
                continuation_to_publish,
                "committed_continuation",
                continuation_age)

    def _publish_summary_locked(self):
        pending = int(self._pending_raw is not None)
        outstanding = int(self._outstanding is not None)
        committed = int(self._committed_master is not None)
        msg = String()
        msg.data = " ".join([
            "source={}".format(self._last_source),
            "raw_input_count={}".format(self._raw_input_count),
            "pending_candidate={}".format(pending),
            "pending_replace_count={}".format(self._pending_replace_count),
            "verification_outstanding={}".format(outstanding),
            "verification_publish_count={}".format(self._verification_publish_count),
            "verification_safe_count={}".format(self._verification_safe_count),
            "verification_unsafe_count={}".format(self._verification_unsafe_count),
            "verification_timeout_count={}".format(self._verification_timeout_count),
            "commit_count={}".format(self._commit_count),
            "has_committed_plan={}".format(committed),
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
