#!/usr/bin/env python3
"""C4.9 blocker-aware recursive visibility acquisition.

Builds on C4.7/C4.8 semantics:
  * obligations clear only after actual confidence confirms they were seen;
  * REPAIR may abandon nominal timing;
  * downstream exact VBC remains the only commit authority;
  * C4.8 may verify only executable prefix + brake + hold.

C4.9 adds blocker-aware target scheduling.  The current visibility obligation is
kept active until seen, except when the earliest VBC temporal layer of the current
repair motion exposes another urgent spatial region.  That region is pushed as a
nested blocker.  Once actual confidence clears it, the previous obligation is
resumed.  Spatial proximity alone never causes preemption.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import rospy
import torch
from std_msgs.msg import Bool, Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory

from vbc_visibility_acquisition_impl import VisibilityAcquisitionWaypointNode


class BlockerAwareVisibilityAcquisitionWaypointNode(VisibilityAcquisitionWaypointNode):
    def __init__(self) -> None:
        self._c49_ready = False

        # C5.26: candidate VBC active-set geometry and sweep time are consumed
        # as one coherent message. Legacy split active-set/sweep topics remain
        # subscribed for diagnostics, but they do not own obligation lifetime.
        self._coherent_bundle_enabled = True
        self._coherent_bundle_lock = threading.Lock()
        self._coherent_bundle_seq = 0
        self._coherent_bundle_sweep_s = math.nan
        self._coherent_bundle_points = np.zeros((0, 3), dtype=np.float64)
        self._coherent_bundle_pending_nonempty = False
        self._coherent_bundle_received_count = 0
        self._coherent_bundle_processed_count = 0
        self._coherent_bundle_drop_count = 0
        self._coherent_bundle_last_reason = "startup"

        # C5.42: separate steering-only L0+L1 transport. L0 still owns
        # persistent obligation materialization through the coherent bundle
        # above. L1 is never inserted into self._obligations here.
        self._spatiotemporal_lock = threading.Lock()
        self._spatiotemporal_seq = 0
        self._spatiotemporal_layers: List[Dict[str, object]] = []
        self._spatiotemporal_received_count = 0
        self._spatiotemporal_drop_count = 0
        self._spatiotemporal_last_reason = "startup"
        self._lookahead_last_region_count = 0
        self._lookahead_last_point_count = 0

        self._repair_stack: List[int] = []
        self._pending_blocker_id = None
        self._pending_blocker_count = 0
        self._stack_push_count = 0
        self._stack_pop_count = 0
        self._stack_cycle_block_count = 0
        self._last_active_layer_ids: List[int] = []
        self._last_active_layer_sweep_s = math.nan
        self._last_switch_reason = "startup"
        self._process_attempt_count = 0
        self._process_success_count = 0
        self._last_process_reason = "startup"

        # C5.41 motion-efficient progressive shared visibility steering.
        # This cache affects only the learned REPAIR steering target. Exact
        # final GCDF + VBC remain downstream commit authorities.
        self._progressive_shared_lock = threading.RLock()
        self._progressive_shared_cache_key = None
        self._progressive_shared_cache = None
        self._progressive_shared_attempt_count = 0
        self._progressive_shared_success_count = 0
        self._progressive_shared_fallback_count = 0
        self._progressive_shared_last_mode = "startup"
        self._progressive_shared_last_considered_ids: List[int] = []
        self._progressive_shared_last_kept_ids: List[int] = []
        self._progressive_shared_last_dropped_ids: List[int] = []
        self._progressive_shared_last_slacks: Dict[int, float] = {}

        # C5.12 direct final-GCDF recovery evidence.  A rejected executable
        # trajectory can expose low-confidence voxels that the task-bootstrap
        # VBC does not report (notably braking-tail sweep).  Keep this evidence
        # on a separate persistent channel and convert it into ordinary
        # visibility obligations using the same VisCDF projector.
        self._gcdf_recovery_lock = threading.Lock()
        self._gcdf_recovery_trajectory = None
        self._gcdf_recovery_trajectory_received = None
        self._gcdf_recovery_trajectory_cache = OrderedDict()
        self._gcdf_recovery_cache_capacity = 8
        self._pending_gcdf_recovery_event = None
        self._processed_gcdf_recovery_seq = 0
        self._gcdf_recovery_event_count = 0
        self._gcdf_recovery_generated_count = 0
        self._gcdf_recovery_match_count = 0
        self._gcdf_recovery_drop_count = 0
        self._last_gcdf_recovery_reason = "startup"
        self._last_gcdf_recovery_event_stamp = "none"
        self._last_gcdf_recovery_trajectory_stamp = "none"
        self._gcdf_recovery_cache_hit_count = 0
        self._gcdf_recovery_cache_miss_count = 0
        super().__init__()

        self.blocker_push_max_sweep_s = float(rospy.get_param(
            "~blocker_push_max_sweep_s", 0.30))
        self.blocker_confirmations = int(rospy.get_param(
            "~blocker_confirmations", 2))

        # C5.41: first try to satisfy several important spatial visibility
        # regions with one shared learned q_vis. If the shared max-min solve
        # cannot make the set jointly visible, progressively remove the
        # lowest-priority region with the largest learned visibility deficit.
        # The current blocker is mandatory and is never dropped.
        self.progressive_shared_repair_enabled = bool(rospy.get_param(
            "~progressive_shared_repair_enabled", True))
        self.progressive_shared_max_regions = int(rospy.get_param(
            "~progressive_shared_max_regions", 4))
        self.progressive_shared_accept_f_min = float(rospy.get_param(
            "~progressive_shared_accept_f_min", 0.0))
        self.one_layer_lookahead_enabled = bool(rospy.get_param(
            "~one_layer_lookahead_enabled", True))

        self.blocker_stack_summary_topic = str(rospy.get_param(
            "~blocker_stack_summary_topic",
            "/care_planner/active_sensing/blocker_stack_summary"))
        self.active_set_bundle_topic = str(rospy.get_param(
            "~active_set_bundle_topic",
            "/care_planner/trajectory_risk/vbc_active_set_bundle"))
        self.spatiotemporal_bundle_topic = str(rospy.get_param(
            "~spatiotemporal_bundle_topic",
            "/care_planner/trajectory_risk/vbc_spatiotemporal_bundle"))
        self.gcdf_recovery_trajectory_topic = str(rospy.get_param(
            "~gcdf_recovery_trajectory_topic",
            "/care_planner/final_gcdf/recovery_trajectory"))
        self.gcdf_recovery_event_topic = str(rospy.get_param(
            "~gcdf_recovery_event_topic",
            "/care_planner/final_gcdf/recovery_visibility_event"))
        if self.blocker_push_max_sweep_s <= 0.0:
            raise ValueError("~blocker_push_max_sweep_s must be positive")
        if self.blocker_confirmations < 1:
            raise ValueError("~blocker_confirmations must be >= 1")
        if self.progressive_shared_max_regions < 1:
            raise ValueError("~progressive_shared_max_regions must be >= 1")
        if not math.isfinite(self.progressive_shared_accept_f_min):
            raise ValueError("~progressive_shared_accept_f_min must be finite")

        self.blocker_stack_summary_pub = rospy.Publisher(
            self.blocker_stack_summary_topic, String, queue_size=1, latch=True)
        self.active_set_bundle_sub = rospy.Subscriber(
            self.active_set_bundle_topic, Float64MultiArray,
            self._active_set_bundle_cb, queue_size=1)
        self.spatiotemporal_bundle_sub = rospy.Subscriber(
            self.spatiotemporal_bundle_topic, Float64MultiArray,
            self._spatiotemporal_bundle_cb, queue_size=1)
        self.gcdf_recovery_trajectory_sub = rospy.Subscriber(
            self.gcdf_recovery_trajectory_topic, JointTrajectory,
            self._gcdf_recovery_trajectory_cb, queue_size=1)
        self.gcdf_recovery_event_sub = rospy.Subscriber(
            self.gcdf_recovery_event_topic, Float64MultiArray,
            self._gcdf_recovery_event_cb, queue_size=1)
        self._c49_ready = True
        self._prune_or_initialize_stack()
        self._publish_schedule()
        self._publish_blocker_stack_summary()
        rospy.logwarn(
            "[vbc_blocker_stack] C5.42 ENABLED max_blocker_sweep=%.3fs "
            "confirmations=%d coherent_bundle=%s one_layer_lookahead=%d "
            "spatiotemporal_bundle=%s",
            self.blocker_push_max_sweep_s, self.blocker_confirmations,
            self.active_set_bundle_topic, int(self.one_layer_lookahead_enabled),
            self.spatiotemporal_bundle_topic)

    def _active_set_callback(self, msg: Float64MultiArray) -> None:
        """Ignore the legacy split active-set stream in C5.26 blocker mode.

        The legacy topic is intentionally kept alive for diagnostics and older
        modes. Letting it mutate _raw_active_set/_sweep_time_s would reintroduce
        the exact cross-topic race that C5.26 removes.
        """
        if getattr(self, "_coherent_bundle_enabled", False):
            return
        super()._active_set_callback(msg)

    def _active_set_bundle_cb(self, msg: Float64MultiArray) -> None:
        if msg is None:
            return
        values = np.asarray(list(msg.data), dtype=np.float64)
        if values.size < 3:
            self._coherent_bundle_drop_count += 1
            self._coherent_bundle_last_reason = "malformed_short_bundle"
            return

        seq_f, sweep, count_f = values[:3]
        if (not math.isfinite(float(seq_f)) or
                not math.isfinite(float(count_f))):
            self._coherent_bundle_drop_count += 1
            self._coherent_bundle_last_reason = "nonfinite_header"
            return
        seq = int(round(float(seq_f)))
        count = int(round(float(count_f)))
        if (seq <= 0 or count < 0 or
                abs(float(count_f) - count) > 1e-6 or
                values.size != 3 + 3 * count):
            self._coherent_bundle_drop_count += 1
            self._coherent_bundle_last_reason = "malformed_shape"
            return
        if count > 0 and (not math.isfinite(float(sweep)) or float(sweep) < 0.0):
            self._coherent_bundle_drop_count += 1
            self._coherent_bundle_last_reason = "invalid_nonempty_sweep"
            return

        points = (
            values[3:].reshape(count, 3)
            if count else np.zeros((0, 3), dtype=np.float64))
        if count and not np.all(np.isfinite(points)):
            self._coherent_bundle_drop_count += 1
            self._coherent_bundle_last_reason = "nonfinite_points"
            return

        by_key = {}
        for point in points:
            by_key[self._cell_key(point)] = point.copy()
        ordered_keys = tuple(sorted(by_key.keys()))
        canonical = (
            np.asarray([by_key[k] for k in ordered_keys], dtype=np.float64)
            if ordered_keys else np.zeros((0, 3), dtype=np.float64))

        with self._coherent_bundle_lock:
            if seq <= self._coherent_bundle_seq:
                self._coherent_bundle_last_reason = "stale_bundle"
                return
            was_pending = self._coherent_bundle_pending_nonempty
            self._coherent_bundle_seq = seq
            self._coherent_bundle_sweep_s = (
                float(sweep) if canonical.shape[0] else math.nan)
            self._coherent_bundle_points = canonical
            self._coherent_bundle_pending_nonempty = canonical.shape[0] > 0
            self._coherent_bundle_received_count += 1
            self._coherent_bundle_last_reason = (
                "pending_nonempty" if canonical.shape[0] else "empty_bundle")

        # Keep legacy summary serials meaningful, but derive them from the
        # coherent generation ID rather than from split-topic callback count.
        with self._obligation_lock:
            self._raw_active_set = canonical.copy()
            self._raw_active_set_serial = seq
            no_obligations = len(self._obligations) == 0

        if canonical.shape[0] > 0:
            # A fresh confirmed blocker must keep acquisition incomplete until
            # this exact bundle has materialized into an obligation.
            if self._acquisition_complete and no_obligations and not was_pending:
                self._visibility_episode_reopen_count += 1
            self._acquisition_started = True
            self._acquisition_complete = False
            self.acquisition_complete_pub.publish(Bool(data=False))

    def _spatiotemporal_bundle_cb(self, msg: Float64MultiArray) -> None:
        """Parse C5.42 L0+L1 steering look-ahead without owning obligation life."""
        if msg is None:
            return
        values = np.asarray(list(msg.data), dtype=np.float64)
        if values.size < 2:
            self._spatiotemporal_drop_count += 1
            self._spatiotemporal_last_reason = "malformed_short"
            return
        try:
            seq = int(round(float(values[0])))
            layer_count = int(round(float(values[1])))
        except Exception:
            self._spatiotemporal_drop_count += 1
            self._spatiotemporal_last_reason = "malformed_header"
            return
        if (seq < 0 or layer_count < 0 or layer_count > 2 or
                abs(float(values[0]) - seq) > 1e-6 or
                abs(float(values[1]) - layer_count) > 1e-6):
            self._spatiotemporal_drop_count += 1
            self._spatiotemporal_last_reason = "invalid_header"
            return

        cursor = 2
        layers = []
        try:
            for expected_index in range(layer_count):
                if cursor + 6 > values.size:
                    raise ValueError("short_layer_header")
                layer_index = int(round(float(values[cursor + 0])))
                min_sweep = float(values[cursor + 1])
                max_sweep = float(values[cursor + 2])
                start_k = int(round(float(values[cursor + 3])))
                end_k = int(round(float(values[cursor + 4])))
                point_count = int(round(float(values[cursor + 5])))
                cursor += 6
                if (layer_index != expected_index or point_count < 0 or
                        not math.isfinite(min_sweep) or
                        not math.isfinite(max_sweep) or
                        min_sweep < 0.0 or max_sweep + 1e-12 < min_sweep):
                    raise ValueError("invalid_layer_header")
                need = 3 * point_count
                if cursor + need > values.size:
                    raise ValueError("short_layer_points")
                points = values[cursor:cursor + need].reshape(point_count, 3)
                cursor += need
                if point_count and not np.all(np.isfinite(points)):
                    raise ValueError("nonfinite_layer_points")

                by_key = {}
                for point in points:
                    by_key[self._cell_key(point)] = point.copy()
                ordered_keys = tuple(sorted(by_key.keys()))
                canonical = (
                    np.asarray([by_key[k] for k in ordered_keys], dtype=np.float64)
                    if ordered_keys else np.zeros((0, 3), dtype=np.float64))
                layers.append({
                    "layer_index": int(layer_index),
                    "min_sweep_s": float(min_sweep),
                    "max_sweep_s": float(max_sweep),
                    "start_original_timestep": int(start_k),
                    "end_original_timestep": int(end_k),
                    "points": canonical,
                    "keys": ordered_keys,
                })
            if cursor != values.size:
                raise ValueError("trailing_values")
        except Exception as exc:
            self._spatiotemporal_drop_count += 1
            self._spatiotemporal_last_reason = "malformed_{}".format(str(exc))
            return

        def geometry_signature(layer_list):
            return tuple(
                (int(layer.get("layer_index", -1)),
                 tuple(layer.get("keys", ())))
                for layer in layer_list)

        with self._spatiotemporal_lock:
            if seq < self._spatiotemporal_seq:
                self._spatiotemporal_last_reason = "stale"
                return
            changed_geometry = (
                geometry_signature(self._spatiotemporal_layers) !=
                geometry_signature(layers))
            self._spatiotemporal_seq = seq
            self._spatiotemporal_layers = layers
            self._spatiotemporal_received_count += 1
            self._spatiotemporal_last_reason = (
                "l0_l1_ready" if len(layers) > 1 else
                ("l0_only" if layers else "empty"))

        # Preserve C5.41 target hysteresis across periodic VBC refreshes that
        # carry the same L0/L1 geometry. Re-solve only when the actual spatial
        # look-ahead set changes, not merely because generation seq increments.
        if changed_geometry:
            with self._progressive_shared_lock:
                self._progressive_shared_cache_key = None
                self._progressive_shared_cache = None

    @staticmethod
    def _stamp_key(stamp):
        if stamp is None:
            return None
        return (int(stamp.secs), int(stamp.nsecs))

    @staticmethod
    def _stamp_text(key) -> str:
        if key is None:
            return "none"
        return "{}.{:09d}".format(int(key[0]), int(key[1]))

    def _gcdf_recovery_trajectory_cb(self, msg) -> None:
        if msg is None or not msg.points:
            return
        key = self._stamp_key(msg.header.stamp)
        if key is None or (key[0] == 0 and key[1] == 0):
            self._gcdf_recovery_drop_count += 1
            self._last_gcdf_recovery_reason = "recovery_trajectory_zero_stamp"
            return
        received = rospy.Time.now()
        with self._gcdf_recovery_lock:
            self._gcdf_recovery_trajectory = msg
            self._gcdf_recovery_trajectory_received = received
            self._last_gcdf_recovery_trajectory_stamp = self._stamp_text(key)
            self._gcdf_recovery_trajectory_cache[key] = (msg, received)
            self._gcdf_recovery_trajectory_cache.move_to_end(key)
            while (len(self._gcdf_recovery_trajectory_cache) >
                   self._gcdf_recovery_cache_capacity):
                self._gcdf_recovery_trajectory_cache.popitem(last=False)

    def _gcdf_recovery_event_cb(self, msg) -> None:
        if msg is None:
            return
        values = list(msg.data)
        if len(values) < 6:
            self._gcdf_recovery_drop_count += 1
            self._last_gcdf_recovery_reason = "malformed_short_event"
            return

        try:
            seq = int(round(float(values[0])))
            stamp_secs = int(round(float(values[1])))
            stamp_nsecs = int(round(float(values[2])))
            sweep_s = float(values[3])
            timestep = int(round(float(values[4])))
            count = int(round(float(values[5])))
        except Exception:
            self._gcdf_recovery_drop_count += 1
            self._last_gcdf_recovery_reason = "malformed_header"
            return

        stamp_key = (stamp_secs, stamp_nsecs)
        if (seq <= 0 or stamp_secs < 0 or
                stamp_nsecs < 0 or stamp_nsecs >= 1000000000 or
                (stamp_secs == 0 and stamp_nsecs == 0) or
                not math.isfinite(sweep_s) or sweep_s < 0.0 or
                timestep < 0 or count <= 0 or len(values) != 6 + 3 * count):
            self._gcdf_recovery_drop_count += 1
            self._last_gcdf_recovery_reason = "malformed_shape"
            return

        points = np.asarray(values[6:], dtype=np.float64).reshape(count, 3)
        if not np.all(np.isfinite(points)):
            self._gcdf_recovery_drop_count += 1
            self._last_gcdf_recovery_reason = "nonfinite_points"
            return

        with self._gcdf_recovery_lock:
            if seq <= self._processed_gcdf_recovery_seq:
                self._last_gcdf_recovery_reason = "stale_processed_event"
                return
            self._pending_gcdf_recovery_event = {
                "seq": seq,
                "stamp_key": stamp_key,
                "sweep_s": sweep_s,
                "timestep": timestep,
                "points": points.copy(),
            }
            self._gcdf_recovery_event_count += 1
            self._last_gcdf_recovery_event_stamp = self._stamp_text(stamp_key)
            self._last_gcdf_recovery_reason = "event_pending"

        # A fresh hard-GCDF blocker defines a new acquisition episode even
        # before q_vis generation finishes.  This makes the current latched
        # completion state truthful while the regime manager remains fail-closed.
        with self._obligation_lock:
            no_obligations = len(self._obligations) == 0
        if self._acquisition_complete and no_obligations:
            self._acquisition_started = True
            self._acquisition_complete = False
            self._visibility_episode_reopen_count += 1
            self.acquisition_complete_pub.publish(Bool(data=False))
            rospy.logwarn(
                "[vbc_blocker_stack] FINAL-GCDF blocker reopens acquisition "
                "episode seq=%d count=%d sweep=%.3fs",
                seq, count, sweep_s)

    def _process_gcdf_recovery_event(self) -> None:
        with self._gcdf_recovery_lock:
            event = (
                None if self._pending_gcdf_recovery_event is None
                else dict(self._pending_gcdf_recovery_event))

        if event is None:
            return
        seq = int(event["seq"])
        stamp_key = tuple(event["stamp_key"])
        if seq <= self._processed_gcdf_recovery_seq:
            with self._gcdf_recovery_lock:
                self._pending_gcdf_recovery_event = None
            self._last_gcdf_recovery_reason = "already_processed"
            return

        with self._gcdf_recovery_lock:
            cached = self._gcdf_recovery_trajectory_cache.get(stamp_key)
        if cached is None:
            self._gcdf_recovery_cache_miss_count += 1
            self._last_gcdf_recovery_reason = "waiting_matching_recovery_trajectory"
            return
        trajectory, trajectory_received = cached
        self._gcdf_recovery_cache_hit_count += 1

        if self._latest_measured_q is None:
            self._last_gcdf_recovery_reason = "waiting_measured_q"
            return

        points = np.asarray(event["points"], dtype=np.float64).reshape(-1, 3)
        sweep_s = float(event["sweep_s"])
        regions = self._cluster_regions(points)
        active_ids: List[int] = []
        new_ids: List[int] = []
        all_regions_handled = True

        for region in regions:
            with self._obligation_lock:
                matched = self._match_existing(region)
                if matched is not None:
                    matched["last_seen_ros_s"] = rospy.Time.now().to_sec()
                    matched["points"] = np.asarray(
                        region["points"], dtype=np.float64).copy()
                    matched["keys"] = tuple(region["keys"])
                    matched["centroid"] = np.asarray(
                        region["centroid"], dtype=np.float64).copy()
                    active_ids.append(int(matched["id"]))
                    self._schedule_matched_obligations += 1
                    self._gcdf_recovery_match_count += 1
                    continue
                if len(self._obligations) >= self.max_obligations:
                    all_regions_handled = False
                    self._last_gcdf_recovery_reason = "max_obligations"
                    continue

            try:
                new_ob = self._generate_new_obligation(
                    region,
                    trajectory,
                    sweep_s,
                    trajectory_received,
                    "final_gcdf_rejected")
            except Exception as exc:
                self._schedule_generation_failures += 1
                all_regions_handled = False
                self._last_gcdf_recovery_reason = "generation_failed"
                rospy.logerr(
                    "[vbc_blocker_stack] final-GCDF obligation generation "
                    "failed seq=%d: %s", seq, exc)
                continue

            with self._obligation_lock:
                matched = self._match_existing(region)
                if matched is None and len(self._obligations) < self.max_obligations:
                    self._obligations.append(new_ob)
                    self._schedule_new_obligations += 1
                    oid = int(new_ob["id"])
                    active_ids.append(oid)
                    new_ids.append(oid)
                    self._gcdf_recovery_generated_count += 1
                    rospy.logwarn(
                        "[vbc_blocker_stack] FINAL-GCDF ADD obligation=%d "
                        "seq=%d points=%d sweep=%.3fs",
                        oid, seq, len(region["points"]), sweep_s)
                elif matched is not None:
                    active_ids.append(int(matched["id"]))
                    self._gcdf_recovery_match_count += 1

        if not all_regions_handled:
            return

        with self._gcdf_recovery_lock:
            self._processed_gcdf_recovery_seq = max(
                self._processed_gcdf_recovery_seq, seq)
            self._gcdf_recovery_trajectory_cache.pop(stamp_key, None)
            if (self._pending_gcdf_recovery_event is not None and
                    int(self._pending_gcdf_recovery_event["seq"]) == seq):
                self._pending_gcdf_recovery_event = None
        self._last_gcdf_recovery_reason = "handled"

        self._prune_or_initialize_stack()
        self._consider_active_layer(active_ids, new_ids, sweep_s)
        self._publish_schedule()
        self._publish_blocker_stack_summary()

    def _prune_or_initialize_stack(self) -> None:
        with self._obligation_lock:
            existing = {int(ob["id"]) for ob in self._obligations}
            before = list(self._repair_stack)
            self._repair_stack = [oid for oid in self._repair_stack if oid in existing]
            self._stack_pop_count += max(0, len(before) - len(self._repair_stack))
            if not self._repair_stack and existing:
                root = min(existing)
                self._repair_stack.append(root)
                self._last_switch_reason = "initialize_oldest_obligation"

    def _lookahead_region_items(self):
        """Return transient L1 spatial regions for steering only.

        Negative IDs are local synthetic identities. They are never appended to
        self._obligations and therefore cannot block acquisition completion.
        """
        if not self.one_layer_lookahead_enabled:
            self._lookahead_last_region_count = 0
            self._lookahead_last_point_count = 0
            return {}
        with self._coherent_bundle_lock:
            l0_seq = int(self._coherent_bundle_seq)
        with self._spatiotemporal_lock:
            seq = int(self._spatiotemporal_seq)
            layers = [dict(layer) for layer in self._spatiotemporal_layers]
        if seq != l0_seq or len(layers) < 2:
            self._lookahead_last_region_count = 0
            self._lookahead_last_point_count = 0
            return {}

        l1 = layers[1]
        points = np.asarray(l1.get("points", []), dtype=np.float64).reshape(-1, 3)
        if points.shape[0] == 0:
            self._lookahead_last_region_count = 0
            self._lookahead_last_point_count = 0
            return {}

        regions = self._cluster_regions(points)
        items = {}
        for index, region in enumerate(regions):
            oid = -(1001 + index)
            items[oid] = {
                "id": oid,
                "points": np.asarray(region["points"], dtype=np.float64).copy(),
                "keys": tuple(region["keys"]),
                "centroid": np.asarray(region["centroid"], dtype=np.float64).copy(),
                "q_vis": np.asarray([], dtype=np.float64),
                "q_zero": np.asarray([], dtype=np.float64),
                "final_f_min": math.nan,
                "discovered_sweep_time_s": float(l1["min_sweep_s"]),
                "temporal_priority": 3,
                "steering_only_lookahead": True,
                "lookahead_layer_index": 1,
            }
        self._lookahead_last_region_count = len(items)
        self._lookahead_last_point_count = int(points.shape[0])
        return items

    def _progressive_priority_order(self, by_id, active_id):
        """Hierarchical spatiotemporal priority for progressive dropout.

        0 current blocker (mandatory)
        1 other regions in current earliest temporal layer
        2 other persistent visibility debt
        3 next temporal layer L1 (opportunistic)
        """
        q0 = None if self._latest_measured_q is None else np.asarray(
            self._latest_measured_q, dtype=np.float64)
        urgent = set(int(v) for v in self._last_active_layer_ids)
        active_centroid = np.asarray(
            by_id[active_id].get("centroid", [math.nan] * 3),
            dtype=np.float64).reshape(-1)

        for oid, item in by_id.items():
            if oid == active_id:
                item["temporal_priority"] = 0
            elif bool(item.get("steering_only_lookahead", False)):
                item["temporal_priority"] = 3
            elif oid in urgent:
                item["temporal_priority"] = 1
            else:
                item["temporal_priority"] = 2

        def distance(oid):
            qv = np.asarray(by_id[oid].get("q_vis", []), dtype=np.float64)
            if (q0 is not None and qv.shape == (7,) and
                    np.all(np.isfinite(qv))):
                return float(np.linalg.norm(qv - q0))
            centroid = np.asarray(
                by_id[oid].get("centroid", [math.nan] * 3),
                dtype=np.float64).reshape(-1)
            if (active_centroid.shape == (3,) and centroid.shape == (3,) and
                    np.all(np.isfinite(active_centroid)) and
                    np.all(np.isfinite(centroid))):
                return float(np.linalg.norm(centroid - active_centroid))
            return math.inf

        others = [oid for oid in by_id if oid != active_id]
        others.sort(
            key=lambda oid: (
                int(by_id[oid].get("temporal_priority", 9)),
                distance(oid),
                oid))
        return [active_id] + others

    def _region_learned_slack(self, ob, q_vis):
        points = np.asarray(ob.get("points", []), dtype=np.float64).reshape(-1, 3)
        if points.shape[0] == 0 or not np.all(np.isfinite(points)):
            return math.inf, -math.inf
        x = torch.tensor(points, device=self.device, dtype=torch.float32)
        q = torch.tensor(
            np.asarray(q_vis, dtype=np.float64).reshape(1, 7),
            device=self.device, dtype=torch.float32)
        values = self._per_point_values(x, q)
        f_min = float(np.min(values)) if values.size else -math.inf
        slack = max(0.0, self.progressive_shared_accept_f_min - f_min)
        return float(slack), f_min

    def _compute_progressive_shared_target(
            self, by_id, active_id, priority_order):
        if (not self.progressive_shared_repair_enabled or
                self.progressive_shared_max_regions <= 1 or
                len(priority_order) <= 1):
            return None

        considered = priority_order[:self.progressive_shared_max_regions]
        # Cache by target identity + exact spatial cells. Do not continuously
        # re-solve as q moves; target persistence is intentional hysteresis.
        geometry_key = tuple(
            (oid, tuple(by_id[oid].get("keys", ())))
            for oid in considered)
        cache_key = (
            int(active_id),
            tuple(considered),
            geometry_key,
            tuple(sorted(int(v) for v in self._last_active_layer_ids)))
        with self._progressive_shared_lock:
            if self._progressive_shared_cache_key == cache_key:
                return self._progressive_shared_cache

            self._progressive_shared_attempt_count += 1
            self._progressive_shared_last_considered_ids = list(considered)
            self._progressive_shared_last_kept_ids = [active_id]
            self._progressive_shared_last_dropped_ids = []
            self._progressive_shared_last_slacks = {}

            measured = (
                None if self._latest_measured_q is None
                else np.asarray(self._latest_measured_q, dtype=np.float64).copy())
            if measured is None or measured.shape != (7,) or not np.all(
                    np.isfinite(measured)):
                self._progressive_shared_last_mode = "no_measured_q"
                # Transient startup state: do not cache the miss, so the same
                # obligation set is retried as soon as measured q arrives.
                self._progressive_shared_cache_key = None
                self._progressive_shared_cache = None
                return None

            with self._lock:
                trajectory, trajectory_received, _ = (
                    self._preferred_trajectory_locked())
            if trajectory is None:
                self._progressive_shared_last_mode = "no_trajectory"
                # Transient startup/refresh state: retry later.
                self._progressive_shared_cache_key = None
                self._progressive_shared_cache = None
                return None

            active = list(considered)
            dropped = []
            last_slacks = {}

            while active:
                points = np.vstack([
                    np.asarray(by_id[oid]["points"], dtype=np.float64).reshape(-1, 3)
                    for oid in active])
                sweep_candidates = [
                    float(by_id[oid].get("discovered_sweep_time_s", math.nan))
                    for oid in active]
                sweep_candidates = [
                    v for v in sweep_candidates if math.isfinite(v) and v >= 0.0]
                sweep_s = (
                    min(sweep_candidates)
                    if sweep_candidates
                    else max(self.safety_margin_s, 0.30))

                self._seed_override = measured.copy()
                try:
                    result = self._generate_active_set_waypoint(
                        points, trajectory, sweep_s, trajectory_received)
                except Exception as exc:
                    rospy.logwarn(
                        "[vbc_blocker_stack] progressive shared q_vis solve "
                        "failed ids=%s: %s",
                        ":".join(str(v) for v in active), exc)
                    result = None
                finally:
                    self._seed_override = None

                if result is None:
                    slacks = {oid: math.inf for oid in active}
                else:
                    qv = np.asarray(result["q_vis"], dtype=np.float64).reshape(7)
                    slacks = {}
                    for oid in active:
                        slack, _ = self._region_learned_slack(by_id[oid], qv)
                        slacks[oid] = slack

                    if all(
                            math.isfinite(slacks[oid]) and
                            slacks[oid] <= 1e-9
                            for oid in active):
                        cache = {
                            "q_vis": qv.copy(),
                            "q_zero": np.asarray(
                                result["q_zero"], dtype=np.float64).reshape(7).copy(),
                            "final_f_min": float(result["final_f_min"]),
                            "kept_ids": list(active),
                            "dropped_ids": list(dropped),
                            "slacks": dict(slacks),
                            "mode": (
                                "progressive_shared_all"
                                if len(active) == len(considered)
                                else "progressive_shared_reduced"),
                        }
                        self._progressive_shared_success_count += 1
                        self._progressive_shared_last_mode = cache["mode"]
                        self._progressive_shared_last_kept_ids = list(active)
                        self._progressive_shared_last_dropped_ids = list(dropped)
                        self._progressive_shared_last_slacks = dict(slacks)
                        self._progressive_shared_cache_key = cache_key
                        self._progressive_shared_cache = cache
                        rospy.logwarn(
                            "[vbc_blocker_stack] C5.42 shared q_vis %s "
                            "considered=%s kept=%s dropped=%s min_f=%+.4f",
                            cache["mode"],
                            ":".join(str(v) for v in considered),
                            ":".join(str(v) for v in active),
                            ":".join(str(v) for v in dropped) or "none",
                            cache["final_f_min"])
                        return cache

                last_slacks = dict(slacks)
                if len(active) <= 1:
                    break

                # Current blocker is mandatory. Among all optional regions,
                # discard a lower-importance tier first; within that tier,
                # discard the one demanding the largest learned slack.
                optional = [oid for oid in active if oid != active_id]
                if not optional:
                    break
                lowest_importance = max(
                    int(by_id[oid].get("temporal_priority", 9))
                    for oid in optional)
                pool = [
                    oid for oid in optional
                    if int(by_id[oid].get("temporal_priority", 9)) ==
                    lowest_importance]
                drop_id = max(
                    pool,
                    key=lambda oid: (
                        slacks.get(oid, math.inf),
                        priority_order.index(oid)))
                active.remove(drop_id)
                dropped.append(drop_id)

            # Fail closed on steering quality: retain the already-generated
            # individual current-blocker q_vis rather than publishing a poor
            # shared best-effort pose.
            self._progressive_shared_fallback_count += 1
            self._progressive_shared_last_mode = "individual_current_fallback"
            self._progressive_shared_last_kept_ids = [active_id]
            self._progressive_shared_last_dropped_ids = list(dropped)
            self._progressive_shared_last_slacks = dict(last_slacks)
            self._progressive_shared_cache_key = cache_key
            self._progressive_shared_cache = None
            return None

    def _ordered_obligations(self):
        with self._obligation_lock:
            copied = [dict(ob) for ob in self._obligations]
            stack = list(self._repair_stack)
        if not copied:
            return []
        persistent_by_id = {int(ob["id"]): ob for ob in copied}
        valid_stack = [oid for oid in stack if oid in persistent_by_id]
        active_id = (
            valid_stack[-1] if valid_stack else min(persistent_by_id))

        steering_by_id = {
            oid: dict(ob) for oid, ob in persistent_by_id.items()}
        steering_by_id.update(self._lookahead_region_items())

        priority_order = self._progressive_priority_order(
            steering_by_id, active_id)
        shared = self._compute_progressive_shared_target(
            steering_by_id, active_id, priority_order)

        first = dict(persistent_by_id[active_id])
        if shared is not None:
            first["q_vis"] = np.asarray(
                shared["q_vis"], dtype=np.float64).copy()
            first["q_zero"] = np.asarray(
                shared["q_zero"], dtype=np.float64).copy()
            first["final_f_min"] = float(shared["final_f_min"])
            first["shared_solution_mode"] = str(shared["mode"])

        # Never publish steering-only L1 pseudo-regions as actual obligations.
        persistent_order = [
            oid for oid in priority_order if oid in persistent_by_id and
            oid != active_id]
        return [first] + [persistent_by_id[oid] for oid in persistent_order]

    def _select_nearest_qvis_id(self, ids: List[int]):
        with self._obligation_lock:
            by_id = {int(ob["id"]): dict(ob) for ob in self._obligations}
        ids = [oid for oid in ids if oid in by_id]
        if not ids:
            return None
        q0 = None if self._latest_measured_q is None else np.asarray(
            self._latest_measured_q, dtype=np.float64)
        def key(oid):
            if q0 is None:
                return (math.inf, oid)
            qv = np.asarray(by_id[oid]["q_vis"], dtype=np.float64)
            return (float(np.linalg.norm(qv - q0)), oid)
        return min(ids, key=key)

    def _consider_active_layer(self, active_ids: List[int], new_ids: List[int], sweep_s: float) -> None:
        self._prune_or_initialize_stack()
        active_ids = sorted(set(int(v) for v in active_ids))
        new_ids = sorted(set(int(v) for v in new_ids))
        self._last_active_layer_ids = active_ids
        self._last_active_layer_sweep_s = float(sweep_s)

        with self._obligation_lock:
            current = self._repair_stack[-1] if self._repair_stack else None
            stack = list(self._repair_stack)

        if current is None or not active_ids:
            self._pending_blocker_id = None
            self._pending_blocker_count = 0
            return
        if not math.isfinite(sweep_s) or sweep_s > self.blocker_push_max_sweep_s:
            self._pending_blocker_id = None
            self._pending_blocker_count = 0
            return

        # Prefer genuinely new regions in the earliest urgent layer.  Otherwise
        # preempt only when the current target is absent from that earliest layer.
        candidates = [oid for oid in new_ids if oid != current]
        if not candidates and current not in active_ids:
            candidates = [oid for oid in active_ids if oid != current]
        if not candidates:
            self._pending_blocker_id = None
            self._pending_blocker_count = 0
            return

        blocker = self._select_nearest_qvis_id(candidates)
        if blocker is None:
            return
        if blocker in stack:
            # Do not create O1->O4->O1 cycles from rejected hypothetical plans.
            self._stack_cycle_block_count += 1
            self._last_switch_reason = "existing_stack_cycle_not_pushed"
            return

        if self._pending_blocker_id == blocker:
            self._pending_blocker_count += 1
        else:
            self._pending_blocker_id = blocker
            self._pending_blocker_count = 1
        if self._pending_blocker_count < self.blocker_confirmations:
            return

        with self._obligation_lock:
            existing = {int(ob["id"]) for ob in self._obligations}
            if blocker in existing and blocker not in self._repair_stack:
                self._repair_stack.append(blocker)
                self._stack_push_count += 1
                self._last_switch_reason = "urgent_earliest_layer_blocker"
                rospy.logwarn(
                    "[vbc_blocker_stack] PUSH blocker=%d sweep=%.3fs stack=%s",
                    blocker, sweep_s,
                    ":".join(str(v) for v in self._repair_stack))
        self._pending_blocker_id = None
        self._pending_blocker_count = 0

    def _process_new_active_set(self) -> None:
        if self._c49_ready:
            self._process_gcdf_recovery_event()
        if not self._c49_ready:
            self._last_process_reason = "not_ready"
            return
        if self._coherent_bundle_enabled:
            with self._coherent_bundle_lock:
                serial = self._coherent_bundle_seq
                raw = self._coherent_bundle_points.copy()
                sweep = self._coherent_bundle_sweep_s
            with self._obligation_lock:
                processed = self._processed_active_set_serial
        else:
            with self._obligation_lock:
                serial = self._raw_active_set_serial
                processed = self._processed_active_set_serial
                raw = self._raw_active_set.copy()
            with self._lock:
                sweep = self._sweep_time_s

        if serial == processed:
            self._last_process_reason = "duplicate_serial"
            return

        self._process_attempt_count += 1
        if raw.shape[0] == 0:
            with self._obligation_lock:
                self._processed_active_set_serial = max(
                    self._processed_active_set_serial, serial)
            if self._coherent_bundle_enabled:
                with self._coherent_bundle_lock:
                    if self._coherent_bundle_seq == serial:
                        self._coherent_bundle_pending_nonempty = False
                        self._coherent_bundle_processed_count += 1
                        self._coherent_bundle_last_reason = "processed_empty"
            self._last_process_reason = "empty_active_set"
            return

        with self._lock:
            trajectory, trajectory_received, trajectory_source = (
                self._preferred_trajectory_locked())
        if not math.isfinite(float(sweep)) or float(sweep) < 0.0:
            # This should be unreachable for a validated coherent non-empty
            # bundle; keep fail-closed behavior for legacy mode/malformed state.
            self._last_process_reason = "waiting_sweep"
            return
        if trajectory is None:
            self._last_process_reason = "waiting_trajectory"
            return
        if self._latest_measured_q is None:
            self._last_process_reason = "waiting_measured_q"
            return

        regions = self._cluster_regions(raw)
        active_ids: List[int] = []
        new_ids: List[int] = []
        all_regions_handled = True
        for region in regions:
            with self._obligation_lock:
                matched = self._match_existing(region)
                if matched is not None:
                    matched["last_seen_ros_s"] = rospy.Time.now().to_sec()
                    matched["points"] = np.asarray(region["points"], dtype=np.float64).copy()
                    matched["keys"] = tuple(region["keys"])
                    matched["centroid"] = np.asarray(region["centroid"], dtype=np.float64).copy()
                    oid = int(matched["id"])
                    active_ids.append(oid)
                    self._schedule_matched_obligations += 1
                    continue
                if len(self._obligations) >= self.max_obligations:
                    rospy.logerr_throttle(
                        1.0, "[vbc_blocker_stack] max_obligations=%d reached",
                        self.max_obligations)
                    continue
            try:
                new_ob = self._generate_new_obligation(
                    region, trajectory, float(sweep), trajectory_received,
                    trajectory_source)
            except Exception as exc:
                self._schedule_generation_failures += 1
                all_regions_handled = False
                rospy.logerr(
                    "[vbc_blocker_stack] obligation generation failed; retrying: %s", exc)
                continue
            with self._obligation_lock:
                matched = self._match_existing(region)
                if matched is None and len(self._obligations) < self.max_obligations:
                    self._obligations.append(new_ob)
                    self._schedule_new_obligations += 1
                    oid = int(new_ob["id"])
                    active_ids.append(oid)
                    new_ids.append(oid)
                elif matched is not None:
                    active_ids.append(int(matched["id"]))

        if all_regions_handled:
            with self._obligation_lock:
                self._processed_active_set_serial = max(
                    self._processed_active_set_serial, serial)
            if self._coherent_bundle_enabled:
                with self._coherent_bundle_lock:
                    if self._coherent_bundle_seq == serial:
                        self._coherent_bundle_pending_nonempty = False
                        self._coherent_bundle_processed_count += 1
                        self._coherent_bundle_last_reason = "materialized"
            self._process_success_count += 1
            self._last_process_reason = "handled"
        else:
            self._last_process_reason = "generation_failed"
        self._prune_or_initialize_stack()
        self._consider_active_layer(active_ids, new_ids, float(sweep))
        self._publish_schedule()
        self._publish_blocker_stack_summary()

    def _update_actual_visibility_completion(self) -> None:
        # C5.26 fail-closed pending semantics: a confirmed non-empty coherent
        # bundle that has not yet materialized into an obligation is itself an
        # outstanding acquisition responsibility. Empty obligation storage must
        # not be misread as "everything has been seen".
        with self._coherent_bundle_lock:
            pending_nonempty = self._coherent_bundle_pending_nonempty
        with self._obligation_lock:
            no_obligations = len(self._obligations) == 0
        if self._coherent_bundle_enabled and pending_nonempty and no_obligations:
            self._acquisition_started = True
            self._acquisition_complete = False
            self.acquisition_complete_pub.publish(Bool(data=False))
            self._publish_acquisition_summary([])
            self._prune_or_initialize_stack()
            self._publish_schedule()
            self._publish_blocker_stack_summary()
            return

        super()._update_actual_visibility_completion()
        self._prune_or_initialize_stack()
        self._publish_schedule()
        self._publish_blocker_stack_summary()

    def _publish_schedule(self) -> None:
        super()._publish_schedule()
        if getattr(self, "_c49_ready", False):
            self._publish_blocker_stack_summary()

    def _publish_blocker_stack_summary(self) -> None:
        if not getattr(self, "_c49_ready", False) or not hasattr(
                self, "blocker_stack_summary_pub"):
            return
        with self._obligation_lock:
            existing = sorted(int(ob["id"]) for ob in self._obligations)
            stack = [oid for oid in self._repair_stack if oid in set(existing)]
            qvis_times = [
                float(ob.get("q_vis_generation_ms", math.nan))
                for ob in self._obligations
                if math.isfinite(float(ob.get("q_vis_generation_ms", math.nan)))
            ]
        msg = String()
        msg.data = (
            "policy=blocker_aware_recursive"
            f" current_target_id={(stack[-1] if stack else -1)}"
            f" stack={':'.join(str(v) for v in stack) or 'none'}"
            f" pending_ids={':'.join(str(v) for v in existing) or 'none'}"
            f" earliest_layer_ids={':'.join(str(v) for v in self._last_active_layer_ids) or 'none'}"
            f" earliest_layer_sweep_s={self._last_active_layer_sweep_s:.6f}"
            f" pending_blocker_id={self._pending_blocker_id if self._pending_blocker_id is not None else -1}"
            f" pending_blocker_count={self._pending_blocker_count}"
            f" push_count={self._stack_push_count}"
            f" pop_count={self._stack_pop_count}"
            f" cycle_block_count={self._stack_cycle_block_count}"
            f" process_attempt_count={self._process_attempt_count}"
            f" process_success_count={self._process_success_count}"
            f" raw_active_set_serial={self._raw_active_set_serial}"
            f" processed_active_set_serial={self._processed_active_set_serial}"
            f" process_reason={self._last_process_reason}"
            f" gcdf_recovery_event_count={self._gcdf_recovery_event_count}"
            f" gcdf_recovery_generated_count={self._gcdf_recovery_generated_count}"
            f" gcdf_recovery_match_count={self._gcdf_recovery_match_count}"
            f" gcdf_recovery_drop_count={self._gcdf_recovery_drop_count}"
            f" gcdf_recovery_processed_seq={self._processed_gcdf_recovery_seq}"
            f" gcdf_recovery_reason={self._last_gcdf_recovery_reason}"
            f" gcdf_recovery_event_stamp={self._last_gcdf_recovery_event_stamp}"
            f" gcdf_recovery_trajectory_stamp={self._last_gcdf_recovery_trajectory_stamp}"
            f" gcdf_recovery_cache_size={len(self._gcdf_recovery_trajectory_cache)}"
            f" gcdf_recovery_cache_hit_count={self._gcdf_recovery_cache_hit_count}"
            f" gcdf_recovery_cache_miss_count={self._gcdf_recovery_cache_miss_count}"
            f" q_vis_generation_last_ms="
            f"{(qvis_times[-1] if qvis_times else math.nan):.3f}"
            f" q_vis_generation_max_ms="
            f"{(max(qvis_times) if qvis_times else math.nan):.3f}"
            f" coherent_bundle_enabled={int(self._coherent_bundle_enabled)}"
            f" coherent_bundle_seq={self._coherent_bundle_seq}"
            f" coherent_bundle_pending={int(self._coherent_bundle_pending_nonempty)}"
            f" coherent_bundle_received_count={self._coherent_bundle_received_count}"
            f" coherent_bundle_processed_count={self._coherent_bundle_processed_count}"
            f" coherent_bundle_drop_count={self._coherent_bundle_drop_count}"
            f" coherent_bundle_reason={self._coherent_bundle_last_reason}"
            f" switch_reason={self._last_switch_reason}"
            f" one_layer_lookahead_enabled={int(self.one_layer_lookahead_enabled)}"
            f" spatiotemporal_seq={self._spatiotemporal_seq}"
            f" spatiotemporal_layer_count={len(self._spatiotemporal_layers)}"
            f" spatiotemporal_received_count={self._spatiotemporal_received_count}"
            f" spatiotemporal_drop_count={self._spatiotemporal_drop_count}"
            f" spatiotemporal_reason={self._spatiotemporal_last_reason}"
            f" lookahead_l1_region_count={self._lookahead_last_region_count}"
            f" lookahead_l1_point_count={self._lookahead_last_point_count}"
            f" progressive_shared_enabled={int(self.progressive_shared_repair_enabled)}"
            f" progressive_shared_max_regions={self.progressive_shared_max_regions}"
            f" progressive_shared_accept_f_min={self.progressive_shared_accept_f_min:.6f}"
            f" progressive_shared_attempt_count={self._progressive_shared_attempt_count}"
            f" progressive_shared_success_count={self._progressive_shared_success_count}"
            f" progressive_shared_fallback_count={self._progressive_shared_fallback_count}"
            f" progressive_shared_mode={self._progressive_shared_last_mode}"
            f" progressive_shared_considered_ids="
            f"{':'.join(str(v) for v in self._progressive_shared_last_considered_ids) or 'none'}"
            f" progressive_shared_kept_ids="
            f"{':'.join(str(v) for v in self._progressive_shared_last_kept_ids) or 'none'}"
            f" progressive_shared_dropped_ids="
            f"{':'.join(str(v) for v in self._progressive_shared_last_dropped_ids) or 'none'}"
            f" progressive_shared_slacks="
            f"{';'.join(str(k)+':' + ('inf' if not math.isfinite(v) else f'{v:.4f}') for k,v in sorted(self._progressive_shared_last_slacks.items())) or 'none'}"
        )
        self.blocker_stack_summary_pub.publish(msg)
