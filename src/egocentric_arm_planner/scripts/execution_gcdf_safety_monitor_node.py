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

        # Phase E5 margins are metric workspace clearances in meters.
        self.warning_margin = float(rospy.get_param("~warning_margin", 0.05))
        self.hard_margin = float(rospy.get_param("~hard_margin", 0.0))
        self.voxel_resolution_m = float(
            rospy.get_param("~voxel_resolution_m", 0.05))
        self.voxel_half_diagonal_m = (
            0.5 * math.sqrt(3.0) * self.voxel_resolution_m)
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
        if self.voxel_resolution_m <= 0.0:
            raise RuntimeError("voxel_resolution_m must be positive")
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
        self.warning_active = False
        self.clear_consecutive = 0
        self.batch_count = 0
        self.warning_event_count = 0
        self.hard_event_count = 0
        self.stale_event_count = 0
        self.replan_count = 0

        self.last_d_min = math.nan
        self.last_unknown_min = math.nan
        self.last_occupied_min = math.nan
        self.last_learned_d_min = math.nan
        self.last_raw_center_clearance_min = math.nan
        self.last_pair_count = 0
        self.last_unknown_count = 0
        self.last_occupied_count = 0
        self.last_warning_count = 0
        self.last_hard_count = 0
        self.last_source = "none"
        self.last_min_occupied_pair_index = -1
        self.last_min_occupied_timestep = -1
        self.last_min_occupied_point = [math.nan, math.nan, math.nan]
        self.last_min_occupied_raw_center_clearance = math.nan
        self.last_min_occupied_volume_clearance = math.nan
        self.last_min_occupied_learned_d = math.nan

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
            "[Phase E5 execution GCDF] workspace warning=%.3fm hard=%.3fm voxel=%.3fm halfdiag=%.4fm stale=%.3fs fail_closed_stale=%d",
            self.warning_margin, self.hard_margin,
            self.voxel_resolution_m, self.voxel_half_diagonal_m,
            self.stale_timeout_s, int(self.fail_closed_on_stale))

    @staticmethod
    def _finite_min(values):
        vals = [float(v) for v in values if math.isfinite(float(v))]
        return min(vals) if vals else math.nan

    def _maybe_replan_locked(self, now, reason):
        if (self.last_replan != rospy.Time(0) and
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
        learned_distances = list(msg.distance)
        raw_clearances = list(msg.approx_body_clearance_m)
        n = min(
            int(msg.num_pairs),
            len(learned_distances),
            len(raw_clearances))
        sources = list(msg.source_type)
        source_unknown = int(getattr(msg, "SOURCE_UNKNOWN", 0))
        source_occupied = int(getattr(msg, "SOURCE_OCCUPIED", 1))

        unknown_d = []
        occupied_d = []
        warning_count = 0
        hard_count = 0
        all_d = []
        learned_valid = []
        raw_center_clearance_valid = []
        min_occupied_pair = None
        hard_occupied_records = []

        for i in range(n):
            learned_d = float(learned_distances[i])
            raw_clearance = float(raw_clearances[i])
            if math.isfinite(learned_d):
                learned_valid.append(learned_d)
            if not math.isfinite(raw_clearance):
                continue

            # Convert sphere-to-voxel-CENTER clearance into a conservative
            # sphere-to-voxel-VOLUME clearance. Anchor radius already includes
            # the Phase-E body-model inflation.
            d = raw_clearance - self.voxel_half_diagonal_m
            raw_center_clearance_valid.append(raw_clearance)

            source = int(sources[i]) if i < len(sources) else source_unknown

            # Phase E5 execution safety is physical collision safety only.
            # UNKNOWN is an epistemic/planning constraint, not collision
            # evidence. Ignore it here even if an upstream legacy batch
            # accidentally contains UNKNOWN rows.
            if source != source_occupied:
                unknown_d.append(d)
                continue

            occupied_d.append(d)
            all_d.append(d)

            j = 3 * i
            point = [math.nan, math.nan, math.nan]
            if j + 2 < len(msg.point_flat):
                point = [
                    float(msg.point_flat[j]),
                    float(msg.point_flat[j + 1]),
                    float(msg.point_flat[j + 2]),
                ]
            timestep = (
                int(msg.original_timestep[i])
                if i < len(msg.original_timestep) else -1)

            record = {
                "pair_index": i,
                "timestep": timestep,
                "point": point,
                "learned_d": learned_d,
                "raw_center_clearance": raw_clearance,
                "volume_clearance": d,
            }
            if (min_occupied_pair is None or
                    d < min_occupied_pair["volume_clearance"]):
                min_occupied_pair = record

            if d < self.warning_margin:
                warning_count += 1
            if d < self.hard_margin:
                hard_count += 1
                hard_occupied_records.append(record)

        d_min = self._finite_min(all_d)
        unknown_min = self._finite_min(unknown_d)
        occupied_min = self._finite_min(occupied_d)
        learned_d_min = self._finite_min(learned_valid)
        raw_center_clearance_min = self._finite_min(raw_center_clearance_valid)

        # d_min is intentionally computed from OCCUPIED rows only.
        # unknown_min is diagnostic and never participates in the execution
        # collision gate.
        source_name = "occupied" if math.isfinite(d_min) else "none"

        with self._lock:
            self.last_batch_received = now
            self.batch_count += 1
            self.last_d_min = d_min
            self.last_unknown_min = unknown_min
            self.last_occupied_min = occupied_min
            self.last_learned_d_min = learned_d_min
            self.last_raw_center_clearance_min = raw_center_clearance_min
            self.last_pair_count = n
            self.last_unknown_count = len(unknown_d)
            self.last_occupied_count = len(occupied_d)
            self.last_warning_count = warning_count
            self.last_hard_count = hard_count
            self.last_source = source_name
            if min_occupied_pair is not None:
                self.last_min_occupied_pair_index = int(
                    min_occupied_pair["pair_index"])
                self.last_min_occupied_timestep = int(
                    min_occupied_pair["timestep"])
                self.last_min_occupied_point = list(
                    min_occupied_pair["point"])
                self.last_min_occupied_raw_center_clearance = float(
                    min_occupied_pair["raw_center_clearance"])
                self.last_min_occupied_volume_clearance = float(
                    min_occupied_pair["volume_clearance"])
                self.last_min_occupied_learned_d = float(
                    min_occupied_pair["learned_d"])
            else:
                self.last_min_occupied_pair_index = -1
                self.last_min_occupied_timestep = -1
                self.last_min_occupied_point = [math.nan, math.nan, math.nan]
                self.last_min_occupied_raw_center_clearance = math.nan
                self.last_min_occupied_volume_clearance = math.nan
                self.last_min_occupied_learned_d = math.nan

            # A fresh batch clears stale transport hold.
            self.stale_hold = False

            hard_now = math.isfinite(d_min) and d_min < self.hard_margin
            warning_now = math.isfinite(d_min) and d_min < self.warning_margin

            if hard_now:
                self.warning_active = False
                self.clear_consecutive = 0
                if not self.hard_hold:
                    self.hard_event_count += 1
                    if min_occupied_pair is not None:
                        hard_voxels = []
                        seen = set()
                        for rec in hard_occupied_records:
                            p = rec["point"]
                            key = tuple(p)
                            if key in seen:
                                continue
                            seen.add(key)
                            hard_voxels.append(
                                "[{:.3f},{:.3f},{:.3f}]".format(
                                    p[0], p[1], p[2]))
                        p = min_occupied_pair["point"]
                        ground_band = (
                            math.isfinite(p[2]) and
                            p[2] <= 0.5 * self.voxel_resolution_m + 1e-9)
                        rospy.logerr(
                            "[EXECUTION_GCDF_OCCUPIED_BLOCKER] "
                            "voxel=[%.6f,%.6f,%.6f] "
                            "pair_index=%d timestep=%d "
                            "raw_center_clearance=%.6f "
                            "voxel_volume_clearance=%.6f learned_d=%.6f "
                            "ground_band_candidate=%d "
                            "hard_occupied_voxel_count=%d "
                            "all_hard_occupied_voxels=%s",
                            p[0], p[1], p[2],
                            int(min_occupied_pair["pair_index"]),
                            int(min_occupied_pair["timestep"]),
                            float(min_occupied_pair["raw_center_clearance"]),
                            float(min_occupied_pair["volume_clearance"]),
                            float(min_occupied_pair["learned_d"]),
                            int(ground_band),
                            len(hard_voxels),
                            ";".join(hard_voxels))
                self.hard_hold = True
                self._maybe_replan_locked(now, "hard")
            else:
                if self.hard_hold:
                    # Fail closed: an empty/no-pair batch cannot prove that a
                    # robot already in hard hold is safe (it could be inside a
                    # surface-only obstacle shell). Release only after repeated
                    # finite clearances beyond the warning margin.
                    if (math.isfinite(d_min) and
                            d_min >= self.warning_margin):
                        self.clear_consecutive += 1
                        if self.clear_consecutive >= self.clear_consecutive_required:
                            self.hard_hold = False
                            self.clear_consecutive = 0
                    else:
                        self.clear_consecutive = 0
                else:
                    self.clear_consecutive = 0

                if warning_now:
                    if not self.warning_active:
                        self.warning_event_count += 1
                    self.warning_active = True
                    self._maybe_replan_locked(now, "warning")
                else:
                    self.warning_active = False

            self._publish_hold_locked()
            self._publish_summary_locked(now)

    def _timer_cb(self, _event):
        now = rospy.Time.now()
        with self._lock:
            since_start = (now - self.start_time).to_sec()
            if since_start < self.startup_grace_s:
                self._publish_summary_locked(now)
                return

            stale = (self.last_batch_received == rospy.Time(0) or
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
        age = (math.nan if self.last_batch_received == rospy.Time(0)
               else max(0.0, (now - self.last_batch_received).to_sec()))
        state = ("HARD_HOLD" if self.hard_hold else
                 "STALE_HOLD" if self.stale_hold else
                 "WARN" if (math.isfinite(self.last_d_min) and
                            self.last_d_min < self.warning_margin) else
                 "SAFE")
        data = (
            "phase=E5 state={} batch_count={} d_min={} unknown_min={} "
            "occupied_min={} learned_d_min={} raw_center_clearance_min={} "
            "voxel_half_diagonal_m={} pair_count={} unknown_pairs={} occupied_pairs={} "
            "warning_pairs={} hard_pairs={} min_source={} "
            "min_occupied_pair_index={} min_occupied_timestep={} "
            "min_occupied_point={} "
            "min_occupied_raw_center_clearance={} "
            "min_occupied_volume_clearance={} "
            "min_occupied_learned_d={} "
            "warning_event_count={} hard_event_count={} stale_event_count={} "
            "replan_count={} hard_hold={} stale_hold={} batch_age_s={}"
        ).format(
            state, self.batch_count, self.last_d_min,
            self.last_unknown_min, self.last_occupied_min,
            self.last_learned_d_min, self.last_raw_center_clearance_min,
            self.voxel_half_diagonal_m,
            self.last_pair_count, self.last_unknown_count,
            self.last_occupied_count, self.last_warning_count,
            self.last_hard_count, self.last_source,
            self.last_min_occupied_pair_index,
            self.last_min_occupied_timestep,
            (
                "{:.6f},{:.6f},{:.6f}".format(
                    self.last_min_occupied_point[0],
                    self.last_min_occupied_point[1],
                    self.last_min_occupied_point[2])
                if all(math.isfinite(v) for v in self.last_min_occupied_point)
                else "nan,nan,nan"
            ),
            self.last_min_occupied_raw_center_clearance,
            self.last_min_occupied_volume_clearance,
            self.last_min_occupied_learned_d,
            self.warning_event_count, self.hard_event_count,
            self.stale_event_count, self.replan_count,
            int(self.hard_hold), int(self.stale_hold), age)
        self.summary_pub.publish(String(data=data))


if __name__ == "__main__":
    rospy.init_node("execution_gcdf_safety_monitor")
    ExecutionGCDFSafetyMonitor()
    rospy.spin()
