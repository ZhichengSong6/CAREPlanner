#!/usr/bin/env python3
import math
import threading

import rospy
from std_msgs.msg import Bool, String
from care_collision_cdf.msg import CollisionCDFConstraintBatch


class ExecutionGCDFSafetyMonitor:
    def __init__(self):
        self.batch_topic = str(rospy.get_param(
            "~batch_topic", "/care_planner/execution_gcdf/constraint_batch"))
        self.summary_topic = str(rospy.get_param(
            "~summary_topic", "/care_planner/execution_gcdf/safety_summary"))
        self.replan_topic = str(rospy.get_param(
            "~replan_topic", "/care_planner/local_planner/replan_request"))
        self.hard_hold_topic = str(rospy.get_param(
            "~hard_hold_topic", "/care_planner/execution_gcdf/hard_hold"))

        self.warning_margin = float(rospy.get_param("~warning_margin", 0.05))
        self.hard_margin = float(rospy.get_param("~hard_margin", 0.0))
        self.stale_timeout_s = float(rospy.get_param("~stale_timeout_s", 0.35))
        self.startup_grace_s = float(rospy.get_param("~startup_grace_s", 2.0))
        self.replan_min_interval_s = float(
            rospy.get_param("~replan_min_interval_s", 0.20))
        self.fail_closed_on_stale = bool(
            rospy.get_param("~fail_closed_on_stale", True))
        self.clear_consecutive_required = int(
            rospy.get_param("~clear_consecutive_required", 2))

        if self.warning_margin <= self.hard_margin:
            raise RuntimeError("warning_margin must be > hard_margin")
        if self.stale_timeout_s <= 0.0 or self.startup_grace_s < 0.0:
            raise RuntimeError("invalid E5 timeout")
        if self.clear_consecutive_required < 1:
            raise RuntimeError("clear_consecutive_required must be >=1")

        self._lock = threading.Lock()
        self.start_time = rospy.Time.now()
        self.last_batch_received = rospy.Time(0)
        self.last_replan = rospy.Time(0)

        self.hard_hold = False
        self.stale_hold = False
        self.clear_consecutive = 0
        self.batch_count = 0
        self.warning_event_count = 0
        self.hard_event_count = 0
        self.stale_event_count = 0
        self.replan_count = 0

        self.last_d_min = math.nan
        self.last_unknown_min = math.nan
        self.last_occupied_min = math.nan
        self.last_pair_count = 0
        self.last_unknown_count = 0
        self.last_occupied_count = 0
        self.last_warning_count = 0
        self.last_hard_count = 0
        self.last_source = "none"

        self.summary_pub = rospy.Publisher(
            self.summary_topic, String, queue_size=10)
        self.replan_pub = rospy.Publisher(
            self.replan_topic, Bool, queue_size=10)
        self.hold_pub = rospy.Publisher(
            self.hard_hold_topic, Bool, queue_size=10, latch=True)
        self.hold_pub.publish(Bool(data=False))

        rospy.Subscriber(
            self.batch_topic, CollisionCDFConstraintBatch,
            self._batch_cb, queue_size=2)
        self.timer = rospy.Timer(rospy.Duration(0.05), self._timer_cb)

        rospy.logwarn(
            "[Phase E5 execution GCDF] warning=%.3f hard=%.3f stale=%.3fs fail_closed_stale=%d",
            self.warning_margin, self.hard_margin, self.stale_timeout_s,
            int(self.fail_closed_on_stale))

    @staticmethod
    def _finite_min(values):
        vals = [float(v) for v in values if math.isfinite(float(v))]
        return min(vals) if vals else math.nan

    def _maybe_replan_locked(self, now, reason):
        if (not self.last_replan.is_zero() and
                (now - self.last_replan).to_sec() < self.replan_min_interval_s):
            return
        self.last_replan = now
        self.replan_count += 1
        self.replan_pub.publish(Bool(data=True))
        rospy.logwarn_throttle(
            0.5, "[Phase E5 execution GCDF] replan reason=%s d_min=%.4f",
            reason, self.last_d_min)

    def _publish_hold_locked(self):
        self.hold_pub.publish(Bool(data=bool(self.hard_hold or self.stale_hold)))

    def _batch_cb(self, msg):
        if msg is None:
            return
        now = rospy.Time.now()
        distances = list(msg.distance)
        n = min(int(msg.num_pairs), len(distances))
        sources = list(msg.source_type)
        source_unknown = int(getattr(msg, "SOURCE_UNKNOWN", 0))
        source_occupied = int(getattr(msg, "SOURCE_OCCUPIED", 1))

        unknown_d = []
        occupied_d = []
        warning_count = 0
        hard_count = 0
        all_d = []

        for i in range(n):
            d = float(distances[i])
            if not math.isfinite(d):
                continue
            all_d.append(d)
            source = int(sources[i]) if i < len(sources) else source_unknown
            if source == source_occupied:
                occupied_d.append(d)
            else:
                unknown_d.append(d)
            if d < self.warning_margin:
                warning_count += 1
            if d < self.hard_margin:
                hard_count += 1

        d_min = self._finite_min(all_d)
        unknown_min = self._finite_min(unknown_d)
        occupied_min = self._finite_min(occupied_d)

        if math.isfinite(d_min):
            if math.isfinite(unknown_min) and abs(d_min - unknown_min) < 1e-12:
                source_name = "unknown"
            elif math.isfinite(occupied_min) and abs(d_min - occupied_min) < 1e-12:
                source_name = "occupied"
            else:
                source_name = "mixed"
        else:
            source_name = "none"

        with self._lock:
            self.last_batch_received = now
            self.batch_count += 1
            self.last_d_min = d_min
            self.last_unknown_min = unknown_min
            self.last_occupied_min = occupied_min
            self.last_pair_count = n
            self.last_unknown_count = len(unknown_d)
            self.last_occupied_count = len(occupied_d)
            self.last_warning_count = warning_count
            self.last_hard_count = hard_count
            self.last_source = source_name

            # A fresh batch clears stale transport hold.
            self.stale_hold = False

            hard_now = math.isfinite(d_min) and d_min < self.hard_margin
            warning_now = math.isfinite(d_min) and d_min < self.warning_margin

            if hard_now:
                self.clear_consecutive = 0
                if not self.hard_hold:
                    self.hard_event_count += 1
                self.hard_hold = True
                self._maybe_replan_locked(now, "hard")
            else:
                if self.hard_hold:
                    # Require repeated fresh non-violating batches before release.
                    self.clear_consecutive += 1
                    if self.clear_consecutive >= self.clear_consecutive_required:
                        self.hard_hold = False
                        self.clear_consecutive = 0
                else:
                    self.clear_consecutive = 0

                if warning_now:
                    self.warning_event_count += 1
                    self._maybe_replan_locked(now, "warning")

            self._publish_hold_locked()
            self._publish_summary_locked(now)

    def _timer_cb(self, _event):
        now = rospy.Time.now()
        with self._lock:
            since_start = (now - self.start_time).to_sec()
            if since_start < self.startup_grace_s:
                self._publish_summary_locked(now)
                return

            stale = (self.last_batch_received.is_zero() or
                     (now - self.last_batch_received).to_sec() >
                     self.stale_timeout_s)
            if stale and self.fail_closed_on_stale:
                if not self.stale_hold:
                    self.stale_event_count += 1
                self.stale_hold = True
                self._maybe_replan_locked(now, "stale")
                self._publish_hold_locked()
            self._publish_summary_locked(now)

    def _publish_summary_locked(self, now):
        age = (math.nan if self.last_batch_received.is_zero()
               else max(0.0, (now - self.last_batch_received).to_sec()))
        state = ("HARD_HOLD" if self.hard_hold else
                 "STALE_HOLD" if self.stale_hold else
                 "WARN" if (math.isfinite(self.last_d_min) and
                            self.last_d_min < self.warning_margin) else
                 "SAFE")
        data = (
            "phase=E5 state={} batch_count={} d_min={} unknown_min={} "
            "occupied_min={} pair_count={} unknown_pairs={} occupied_pairs={} "
            "warning_pairs={} hard_pairs={} min_source={} "
            "warning_event_count={} hard_event_count={} stale_event_count={} "
            "replan_count={} hard_hold={} stale_hold={} batch_age_s={}"
        ).format(
            state, self.batch_count, self.last_d_min,
            self.last_unknown_min, self.last_occupied_min,
            self.last_pair_count, self.last_unknown_count,
            self.last_occupied_count, self.last_warning_count,
            self.last_hard_count, self.last_source,
            self.warning_event_count, self.hard_event_count,
            self.stale_event_count, self.replan_count,
            int(self.hard_hold), int(self.stale_hold), age)
        self.summary_pub.publish(String(data=data))


if __name__ == "__main__":
    rospy.init_node("execution_gcdf_safety_monitor")
    ExecutionGCDFSafetyMonitor()
    rospy.spin()
