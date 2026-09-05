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
import time
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import rospy
import torch
from std_msgs.msg import Bool, Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory

from vbc_visibility_acquisition_impl import VisibilityAcquisitionWaypointNode
from evaluate_direct_vs_projection_ascent import model_value_and_grad_q


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

        # Unified safe visibility-frontier steering. This runs in every
        # multi-obligation REPAIR episode at low weight and is boosted when
        # recursive VBC dependencies form a live cycle. It only proposes a
        # local learned target; downstream final GCDF + exact VBC still decide
        # whether any motion may execute.
        self._frontier_lock = threading.RLock()
        self._frontier_cache_key = None
        self._frontier_cache = None
        self._frontier_compute_count = 0
        self._frontier_publish_count = 0
        self._frontier_error_count = 0
        self._frontier_last_mode = "startup"
        self._frontier_last_considered_ids: List[int] = []
        self._frontier_last_f_values: Dict[int, float] = {}
        self._frontier_last_softmin_weights: Dict[int, float] = {}
        self._frontier_last_direction_norm_inf = math.nan
        self._frontier_last_target_shift_inf = math.nan
        self._frontier_last_weight_scale = 0.0
        self._frontier_last_qvis_weight_scale = 1.0
        self._frontier_last_cycle_active = False

        # Adaptive-resolution obligations. Coarse regions remain the default.
        # A verified recursive dependency cycle may request one refinement of
        # the conflicting spatial obligations. Refined zones are remembered so
        # later active-set updates cannot silently merge the children back into
        # the old coarse cluster.
        self._adaptive_refinement_lock = threading.RLock()
        self._adaptive_refinement_families = {}
        self._pending_refinement_ids: List[int] = []
        self._adaptive_refinement_trigger_count = 0
        self._adaptive_refinement_success_count = 0
        self._adaptive_refinement_failure_count = 0
        self._adaptive_refinement_skip_count = 0
        self._adaptive_refinement_parent_count = 0
        self._adaptive_refinement_child_count = 0
        self._adaptive_refinement_last_parent_ids: List[int] = []
        self._adaptive_refinement_last_child_ids: List[int] = []
        self._adaptive_refinement_last_reason = "startup"
        self._adaptive_refinement_last_cross_f_ab = math.nan
        self._adaptive_refinement_last_cross_f_ba = math.nan
        self._adaptive_refinement_family_route_count = 0
        self._adaptive_refinement_absorb_count = 0
        self._adaptive_refinement_absorbed_point_count = 0
        self._adaptive_refinement_qvis_reuse_count = 0
        self._adaptive_refinement_qvis_regen_count = 0
        self._adaptive_refinement_qvis_regen_failure_count = 0
        self._adaptive_refinement_last_family_id = -1
        self._adaptive_refinement_last_child_id = -1
        self._adaptive_refinement_last_absorb_f_min = math.nan

        # Obligation identity coherence. Spatial proximity alone is insufficient:
        # a matched obligation may keep its old q_vis only while that q_vis still
        # makes the newly observed region learned-visible. These fields exist
        # before parent initialization because ROS callbacks may start early.
        self._obligation_match_qvis_min_f = 0.0
        self._qvis_match_check_count = 0
        self._qvis_match_accept_count = 0
        self._qvis_match_reject_count = 0
        self._qvis_match_error_count = 0
        self._qvis_match_last_obligation_id = -1
        self._qvis_match_last_f_min = math.nan
        self._qvis_match_last_reason = "startup"

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
            "~progressive_shared_max_regions", 3))
        self.progressive_shared_accept_f_min = float(rospy.get_param(
            "~progressive_shared_accept_f_min", 0.0))
        self._obligation_match_qvis_min_f = float(rospy.get_param(
            "~obligation_match_qvis_min_f", 0.0))

        self.frontier_steering_enabled = bool(rospy.get_param(
            "~frontier_steering_enabled", True))
        self.frontier_max_regions = int(rospy.get_param(
            "~frontier_max_regions", 3))
        self.frontier_softmin_temperature = float(rospy.get_param(
            "~frontier_softmin_temperature", 0.05))
        self.frontier_step_inf = float(rospy.get_param(
            "~frontier_step_inf", 0.05))
        self.frontier_recompute_q_inf = float(rospy.get_param(
            "~frontier_recompute_q_inf", 0.01))
        self.frontier_base_weight_scale = float(rospy.get_param(
            "~frontier_base_weight_scale", 0.10))
        self.frontier_cycle_weight_scale = float(rospy.get_param(
            "~frontier_cycle_weight_scale", 1.00))
        self.frontier_base_qvis_weight_scale = float(rospy.get_param(
            "~frontier_base_qvis_weight_scale", 1.00))
        self.frontier_cycle_qvis_weight_scale = float(rospy.get_param(
            "~frontier_cycle_qvis_weight_scale", 0.10))
        self.frontier_gradient_eps = float(rospy.get_param(
            "~frontier_gradient_eps", 1e-6))
        self.frontier_target_topic = str(rospy.get_param(
            "~frontier_target_topic",
            "/care_planner/active_sensing/visibility_frontier_target"))
        self.frontier_summary_topic = str(rospy.get_param(
            "~frontier_summary_topic",
            "/care_planner/active_sensing/visibility_frontier_summary"))

        self.adaptive_refinement_enabled = bool(rospy.get_param(
            "~adaptive_refinement_enabled", False))
        self.adaptive_refinement_max_depth = int(rospy.get_param(
            "~adaptive_refinement_max_depth", 1))
        self.adaptive_refinement_target_diameter_m = float(rospy.get_param(
            "~adaptive_refinement_target_diameter_m", 0.055))
        self.adaptive_refinement_max_children = int(rospy.get_param(
            "~adaptive_refinement_max_children_per_parent", 4))
        self.adaptive_refinement_min_points = int(rospy.get_param(
            "~adaptive_refinement_min_points", 2))
        self.adaptive_refinement_cross_visibility_threshold = float(
            rospy.get_param(
                "~adaptive_refinement_cross_visibility_threshold", 0.0))
        self.adaptive_refinement_family_margin_m = float(rospy.get_param(
            "~adaptive_refinement_family_margin_m", 0.075))

        self.blocker_stack_summary_topic = str(rospy.get_param(
            "~blocker_stack_summary_topic",
            "/care_planner/active_sensing/blocker_stack_summary"))
        self.active_set_bundle_topic = str(rospy.get_param(
            "~active_set_bundle_topic",
            "/care_planner/trajectory_risk/vbc_active_set_bundle"))
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
        if not math.isfinite(self._obligation_match_qvis_min_f):
            raise ValueError("~obligation_match_qvis_min_f must be finite")
        if self.frontier_max_regions < 2:
            raise ValueError("~frontier_max_regions must be >= 2")
        if (not math.isfinite(self.frontier_softmin_temperature) or
                self.frontier_softmin_temperature <= 0.0):
            raise ValueError(
                "~frontier_softmin_temperature must be positive finite")
        if (not math.isfinite(self.frontier_step_inf) or
                self.frontier_step_inf <= 0.0):
            raise ValueError("~frontier_step_inf must be positive finite")
        if (not math.isfinite(self.frontier_recompute_q_inf) or
                self.frontier_recompute_q_inf <= 0.0):
            raise ValueError(
                "~frontier_recompute_q_inf must be positive finite")
        for name, value in (
                ("frontier_base_weight_scale", self.frontier_base_weight_scale),
                ("frontier_cycle_weight_scale", self.frontier_cycle_weight_scale),
                ("frontier_base_qvis_weight_scale",
                 self.frontier_base_qvis_weight_scale),
                ("frontier_cycle_qvis_weight_scale",
                 self.frontier_cycle_qvis_weight_scale)):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("~{} must be nonnegative finite".format(name))
        if (not math.isfinite(self.frontier_gradient_eps) or
                self.frontier_gradient_eps <= 0.0):
            raise ValueError("~frontier_gradient_eps must be positive finite")
        if self.adaptive_refinement_max_depth < 1:
            raise ValueError("~adaptive_refinement_max_depth must be >= 1")
        if (not math.isfinite(self.adaptive_refinement_target_diameter_m) or
                self.adaptive_refinement_target_diameter_m <= 0.0):
            raise ValueError(
                "~adaptive_refinement_target_diameter_m must be positive finite")
        if self.adaptive_refinement_max_children < 2:
            raise ValueError(
                "~adaptive_refinement_max_children_per_parent must be >= 2")
        if self.adaptive_refinement_min_points < 2:
            raise ValueError("~adaptive_refinement_min_points must be >= 2")
        if not math.isfinite(
                self.adaptive_refinement_cross_visibility_threshold):
            raise ValueError(
                "~adaptive_refinement_cross_visibility_threshold must be finite")
        if (not math.isfinite(self.adaptive_refinement_family_margin_m) or
                self.adaptive_refinement_family_margin_m < 0.0):
            raise ValueError(
                "~adaptive_refinement_family_margin_m must be nonnegative finite")

        self.blocker_stack_summary_pub = rospy.Publisher(
            self.blocker_stack_summary_topic, String, queue_size=1, latch=True)
        self.frontier_target_pub = rospy.Publisher(
            self.frontier_target_topic,
            Float64MultiArray, queue_size=1, latch=True)
        self.frontier_summary_pub = rospy.Publisher(
            self.frontier_summary_topic,
            String, queue_size=1, latch=True)
        self.active_set_bundle_sub = rospy.Subscriber(
            self.active_set_bundle_topic, Float64MultiArray,
            self._active_set_bundle_cb, queue_size=1)
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
            "[vbc_blocker_stack] C4.9/C5.26 ENABLED max_blocker_sweep=%.3fs "
            "confirmations=%d coherent_bundle=%s",
            self.blocker_push_max_sweep_s, self.blocker_confirmations,
            self.active_set_bundle_topic)

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
            routed = self._absorb_refined_partition_region(
                region, trajectory, float(sweep_s),
                trajectory_received, "final_gcdf_rejected")
            if routed is not None:
                if int(routed) < 0:
                    all_regions_handled = False
                    self._last_gcdf_recovery_reason = (
                        "refined_family_absorb_retry")
                else:
                    active_ids.append(int(routed))
                    self._schedule_matched_obligations += 1
                    self._gcdf_recovery_match_count += 1
                continue

            with self._obligation_lock:
                matched = self._match_existing(region)
                if matched is not None:
                    self._update_matched_geometry_diagnostics(
                        matched, region, "final_gcdf_recovery")
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
        self._process_pending_refinement(
            trajectory, trajectory_received,
            "final_gcdf_rejected", float(sweep_s))
        self._publish_schedule()
        self._publish_blocker_stack_summary()

    def _split_region_for_refinement(
            self, region: Dict[str, object], force_split: bool = True):
        """Deterministically bisect one spatial region at finer resolution.

        Children define persistent spatial partitions. Their partition anchors
        are frozen at refinement time; later blocker voxels are routed to the
        nearest live child in the same family rather than spawning new
        obligations merely because their voxel key was not present originally.
        """
        pts = np.asarray(
            region.get("points", []), dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] < self.adaptive_refinement_min_points:
            return [dict(region)]

        clusters = [list(range(pts.shape[0]))]
        must_split = bool(force_split)

        while len(clusters) < self.adaptive_refinement_max_children:
            candidates = []
            for ci, idx in enumerate(clusters):
                if len(idx) < 2:
                    continue
                diam = self._diameter(pts[idx])
                needs = (
                    diam > self.adaptive_refinement_target_diameter_m + 1e-9)
                if needs or (must_split and len(clusters) == 1):
                    candidates.append((diam, len(idx), ci))
            if not candidates:
                break

            _, _, ci = max(candidates)
            idx = clusters[ci]
            local = pts[idx]
            ranges = np.ptp(local, axis=0)
            axis = int(np.argmax(ranges))
            ordered = sorted(
                idx,
                key=lambda ii: (
                    float(pts[ii, axis]),
                    float(pts[ii, (axis + 1) % 3]),
                    float(pts[ii, (axis + 2) % 3])))
            mid = len(ordered) // 2
            left = ordered[:mid]
            right = ordered[mid:]
            if not left or not right:
                break
            clusters[ci] = left
            clusters.insert(ci + 1, right)
            must_split = False

        out = []
        family_id = int(region.get("refinement_family_id", -1))
        for idx in clusters:
            child_pts = pts[idx].copy()
            keys = tuple(sorted({self._cell_key(p) for p in child_pts}))
            anchor = np.mean(child_pts, axis=0)
            child = {
                "points": child_pts,
                "keys": keys,
                "centroid": anchor.copy(),
                "refinement_depth": int(region.get("refinement_depth", 1)),
                "parent_obligation_id": int(
                    region.get("parent_obligation_id", -1)),
                "root_obligation_id": int(
                    region.get("root_obligation_id", -1)),
                "refinement_reason": str(
                    region.get(
                        "refinement_reason",
                        "adaptive_dependency_conflict")),
                "refinement_family_id": family_id,
                "refinement_partition_anchor": anchor.copy(),
            }
            out.append(child)
        return out

    def _refinement_family_snapshot(self):
        with self._adaptive_refinement_lock:
            family_records = {
                int(fid): {
                    "family_id": int(fid),
                    "root_centroid": np.asarray(
                        fam["root_centroid"], dtype=np.float64).copy(),
                    "root_xyz_min": np.asarray(
                        fam["root_xyz_min"], dtype=np.float64).copy(),
                    "root_xyz_max": np.asarray(
                        fam["root_xyz_max"], dtype=np.float64).copy(),
                    "child_ids": list(fam.get("child_ids", [])),
                }
                for fid, fam in self._adaptive_refinement_families.items()
            }
        if not family_records:
            return []

        with self._obligation_lock:
            live = {
                int(ob["id"]): dict(ob)
                for ob in self._obligations
                if int(ob.get("refinement_depth", 0)) > 0
            }

        out = []
        for fid, fam in sorted(family_records.items()):
            children = []
            for oid in fam["child_ids"]:
                ob = live.get(int(oid))
                if ob is None:
                    continue
                anchor = np.asarray(
                    ob.get(
                        "refinement_partition_anchor",
                        ob.get("centroid", [math.nan] * 3)),
                    dtype=np.float64).reshape(3)
                if not np.all(np.isfinite(anchor)):
                    continue
                children.append({
                    "id": int(oid),
                    "anchor": anchor.copy(),
                    "refinement_depth": int(
                        ob.get("refinement_depth", 1)),
                    "parent_obligation_id": int(
                        ob.get("parent_obligation_id", -1)),
                    "root_obligation_id": int(
                        ob.get("root_obligation_id", -1)),
                })
            if children:
                fam["children"] = children
                out.append(fam)
        return out

    def _cluster_regions(self, points: np.ndarray):
        """Coarse everywhere, persistent fine partitions only where refined.

        A point is routed into an existing refined family when it lies inside
        the original parent bounding box expanded by a small physical margin.
        Within that family, nearest frozen child anchor determines ownership.
        This lets nearby newly exposed blocker voxels enlarge an existing child
        instead of creating O11/O12-style obligation proliferation.
        """
        coarse = super()._cluster_regions(points)
        if not getattr(self, "adaptive_refinement_enabled", False):
            return coarse

        families = self._refinement_family_snapshot()
        if not families:
            return coarse

        out = []
        margin = float(self.adaptive_refinement_family_margin_m)
        for region in coarse:
            pts = np.asarray(
                region.get("points", []),
                dtype=np.float64).reshape(-1, 3)
            if pts.shape[0] == 0:
                continue

            grouped = {}
            unassigned = []
            for point in pts:
                candidates = []
                for fam in families:
                    lo = fam["root_xyz_min"] - margin
                    hi = fam["root_xyz_max"] + margin
                    if np.all(point >= lo) and np.all(point <= hi):
                        root_dist = float(np.linalg.norm(
                            point - fam["root_centroid"]))
                        candidates.append((root_dist, fam))
                if not candidates:
                    unassigned.append(point.copy())
                    continue

                _, fam = min(candidates, key=lambda item: item[0])
                child = min(
                    fam["children"],
                    key=lambda item: float(np.linalg.norm(
                        point - item["anchor"])))
                key = (int(fam["family_id"]), int(child["id"]))
                grouped.setdefault(
                    key, {"family": fam, "child": child, "points": []})
                grouped[key]["points"].append(point.copy())

            if unassigned:
                rp = np.asarray(unassigned, dtype=np.float64).reshape(-1, 3)
                out.append({
                    "points": rp,
                    "keys": tuple(sorted({
                        self._cell_key(p) for p in rp})),
                    "centroid": np.mean(rp, axis=0),
                })

            for (_fid, _cid), item in sorted(grouped.items()):
                rp = np.asarray(
                    item["points"], dtype=np.float64).reshape(-1, 3)
                child = item["child"]
                out.append({
                    "points": rp,
                    "keys": tuple(sorted({
                        self._cell_key(p) for p in rp})),
                    "centroid": np.mean(rp, axis=0),
                    "preferred_child_id": int(child["id"]),
                    "refinement_family_id": int(
                        item["family"]["family_id"]),
                    "refinement_depth": int(
                        child["refinement_depth"]),
                    "parent_obligation_id": int(
                        child["parent_obligation_id"]),
                    "root_obligation_id": int(
                        child["root_obligation_id"]),
                    "refinement_partition_anchor": np.asarray(
                        child["anchor"], dtype=np.float64).copy(),
                    "refinement_reason": "persistent_family_partition",
                })
                self._adaptive_refinement_family_route_count += int(
                    rp.shape[0])
        return out

    def _cross_visibility_f_min(self, target_ob, q_vis) -> float:
        points = np.asarray(
            target_ob.get("points", []),
            dtype=np.float64).reshape(-1, 3)
        qv = np.asarray(q_vis, dtype=np.float64).reshape(-1)
        if (points.shape[0] == 0 or qv.shape != (7,) or
                not np.all(np.isfinite(points)) or
                not np.all(np.isfinite(qv))):
            return -math.inf
        x = torch.tensor(
            points, device=self.device, dtype=torch.float32)
        q = torch.tensor(
            qv.reshape(1, 7), device=self.device, dtype=torch.float32)
        values = self._per_point_values(x, q)
        return float(np.min(values)) if values.size else -math.inf

    def _merge_refined_child_region(self, child, region):
        by_key = {}
        old_points = np.asarray(
            child.get("points", []), dtype=np.float64).reshape(-1, 3)
        new_points = np.asarray(
            region.get("points", []), dtype=np.float64).reshape(-1, 3)
        for p in old_points:
            by_key[self._cell_key(p)] = p.copy()
        before_keys = set(by_key.keys())
        for p in new_points:
            by_key[self._cell_key(p)] = p.copy()
        keys = tuple(sorted(by_key.keys()))
        points = np.asarray(
            [by_key[k] for k in keys], dtype=np.float64).reshape(-1, 3)
        added = len(set(keys) - before_keys)
        return {
            "points": points,
            "keys": keys,
            "centroid": np.mean(points, axis=0),
            "preferred_child_id": int(child["id"]),
            "refinement_family_id": int(
                child.get("refinement_family_id", -1)),
            "refinement_depth": int(child.get("refinement_depth", 1)),
            "parent_obligation_id": int(
                child.get("parent_obligation_id", -1)),
            "root_obligation_id": int(
                child.get("root_obligation_id", -1)),
            "refinement_partition_anchor": np.asarray(
                child.get(
                    "refinement_partition_anchor",
                    child.get("centroid", [math.nan] * 3)),
                dtype=np.float64).reshape(3).copy(),
            "refinement_reason": "persistent_family_absorb",
            "_new_point_count": int(added),
        }

    def _regenerate_refined_child_qvis(
            self, child_id: int, merged_region,
            trajectory, sweep_time_s: float, trajectory_received,
            trajectory_source: str) -> bool:
        measured = (
            None if self._latest_measured_q is None
            else np.asarray(
                self._latest_measured_q, dtype=np.float64).copy())
        if (measured is None or measured.shape != (7,) or
                not np.all(np.isfinite(measured))):
            return False

        self._seed_override = measured
        started = time.perf_counter()
        try:
            result = self._generate_active_set_waypoint(
                merged_region["points"], trajectory,
                float(sweep_time_s), trajectory_received)
        except Exception as exc:
            self._adaptive_refinement_qvis_regen_failure_count += 1
            self._adaptive_refinement_last_reason = (
                "family_child_qvis_regeneration_failed")
            rospy.logerr(
                "[vbc_blocker_stack] refined child q_vis regeneration "
                "failed child=%d: %s", int(child_id), exc)
            return False
        finally:
            generation_ms = 1000.0 * (
                time.perf_counter() - started)
            self._seed_override = None

        q_vis = np.asarray(
            result["q_vis"], dtype=np.float64).reshape(7)
        q_zero = np.asarray(
            result["q_zero"], dtype=np.float64).reshape(7)
        points = np.asarray(
            merged_region["points"], dtype=np.float64).reshape(-1, 3)
        centroid = np.asarray(
            merged_region["centroid"], dtype=np.float64).reshape(3)
        keys = tuple(merged_region["keys"])
        xyz_min = np.min(points, axis=0)
        xyz_max = np.max(points, axis=0)
        deadline_abs = float(result["deadline_absolute_ros_s"])
        now_s = rospy.Time.now().to_sec()
        min_hit = self._rest_min_hit_time(measured, q_vis)
        deadline_remaining = deadline_abs - now_s

        with self._obligation_lock:
            live = next(
                (ob for ob in self._obligations
                 if int(ob["id"]) == int(child_id)),
                None)
            if live is None:
                return False

            live["points"] = points.copy()
            live["keys"] = keys
            live["centroid"] = centroid.copy()
            live["q_vis_source_points"] = points.copy()
            live["q_vis_source_keys"] = keys
            live["q_vis_source_centroid"] = centroid.copy()
            live["q_vis_source_xyz_min"] = xyz_min.copy()
            live["q_vis_source_xyz_max"] = xyz_max.copy()
            live["geometry_changed_since_qvis"] = False
            live["last_match_centroid_shift_m"] = 0.0
            live["max_centroid_shift_from_qvis_m"] = 0.0
            live["last_geometry_match_source"] = (
                "persistent_family_qvis_regenerated")
            live["last_seen_ros_s"] = now_s
            live["q_vis"] = q_vis.copy()
            live["q_zero"] = q_zero.copy()
            live["deadline_abs_s"] = deadline_abs
            live["discovered_sweep_time_s"] = float(sweep_time_s)
            live["trajectory_source"] = str(trajectory_source)
            live["min_hit_time_from_measured_rest_s"] = float(min_hit)
            live["deadline_remaining_at_discovery_s"] = float(
                deadline_remaining)
            live["reachable_before_discovered_deadline_lower_bound"] = bool(
                min_hit <= max(0.0, deadline_remaining) + 1e-9)
            live["final_f_min"] = float(result["final_f_min"])
            live["shared_solution_mode"] = str(
                result["shared_solution_mode"])
            live["q_vis_generation_ms"] = float(generation_ms)
            live["q_vis_regeneration_count"] = int(
                live.get("q_vis_regeneration_count", 0)) + 1

        with self._progressive_shared_lock:
            self._progressive_shared_cache_key = None
            self._progressive_shared_cache = None
        with self._frontier_lock:
            self._frontier_cache_key = None
            self._frontier_cache = None

        self._adaptive_refinement_qvis_regen_count += 1
        self._adaptive_refinement_last_reason = (
            "family_child_absorbed_qvis_regenerated")
        rospy.logwarn(
            "[vbc_blocker_stack] ADAPTIVE FAMILY regenerate child=%d "
            "points=%d min_f=%+.4f qvis_ms=%.3f",
            int(child_id), int(points.shape[0]),
            float(result["final_f_min"]), float(generation_ms))
        return True

    def _absorb_refined_partition_region(
            self, region, trajectory, sweep_time_s: float,
            trajectory_received, trajectory_source: str):
        """Absorb routed blocker voxels into an existing refined child.

        Returns child id on success, -1 when the routed child must be retried,
        and None when the region is not owned by a persistent family.
        """
        child_id = int(region.get("preferred_child_id", -1))
        family_id = int(region.get("refinement_family_id", -1))
        if child_id < 0 or family_id < 0:
            return None

        with self._obligation_lock:
            child = next(
                (dict(ob) for ob in self._obligations
                 if int(ob["id"]) == child_id),
                None)
        if child is None:
            return None
        if int(child.get("refinement_family_id", -1)) != family_id:
            return None

        merged = self._merge_refined_child_region(child, region)
        try:
            f_min = self._cross_visibility_f_min(
                merged, child.get("q_vis", []))
        except Exception as exc:
            self._adaptive_refinement_qvis_regen_failure_count += 1
            self._adaptive_refinement_last_reason = (
                "family_child_absorb_visibility_eval_failed")
            rospy.logerr_throttle(
                1.0,
                "[vbc_blocker_stack] refined child compatibility "
                "evaluation failed child=%d: %s", child_id, exc)
            return -1

        self._adaptive_refinement_last_family_id = family_id
        self._adaptive_refinement_last_child_id = child_id
        self._adaptive_refinement_last_absorb_f_min = float(f_min)
        added = int(merged.get("_new_point_count", 0))

        if (math.isfinite(f_min) and
                f_min + 1e-9 >= self._obligation_match_qvis_min_f):
            with self._obligation_lock:
                live = next(
                    (ob for ob in self._obligations
                     if int(ob["id"]) == child_id),
                    None)
                if live is None:
                    return None
                self._update_matched_geometry_diagnostics(
                    live, merged, "persistent_family_absorb")
            self._adaptive_refinement_qvis_reuse_count += 1
            self._adaptive_refinement_absorb_count += 1
            self._adaptive_refinement_absorbed_point_count += added
            self._adaptive_refinement_last_reason = (
                "family_child_absorbed_qvis_reused")
            return child_id

        if not self._regenerate_refined_child_qvis(
                child_id, merged, trajectory, float(sweep_time_s),
                trajectory_received, trajectory_source):
            return -1

        self._adaptive_refinement_absorb_count += 1
        self._adaptive_refinement_absorbed_point_count += added
        return child_id

    def _request_cycle_refinement(self, current_id: int, blocker_id: int) -> bool:
        """Request refinement only for a learned-incompatible dependency pair."""
        if not self.adaptive_refinement_enabled:
            return False
        with self._obligation_lock:
            by_id = {
                int(ob["id"]): dict(ob) for ob in self._obligations}
        if current_id not in by_id or blocker_id not in by_id:
            self._adaptive_refinement_skip_count += 1
            self._adaptive_refinement_last_reason = "missing_cycle_obligation"
            return False

        a = by_id[int(current_id)]
        b = by_id[int(blocker_id)]
        try:
            f_b_at_a = self._cross_visibility_f_min(b, a.get("q_vis", []))
            f_a_at_b = self._cross_visibility_f_min(a, b.get("q_vis", []))
        except Exception as exc:
            self._adaptive_refinement_failure_count += 1
            self._adaptive_refinement_last_reason = "cross_visibility_error"
            rospy.logerr_throttle(
                1.0,
                "[vbc_blocker_stack] adaptive refinement cross-visibility "
                "evaluation failed: %s", exc)
            return False

        self._adaptive_refinement_last_cross_f_ab = float(f_b_at_a)
        self._adaptive_refinement_last_cross_f_ba = float(f_a_at_b)
        threshold = self.adaptive_refinement_cross_visibility_threshold
        incompatible = bool(
            math.isfinite(f_b_at_a) and math.isfinite(f_a_at_b) and
            f_b_at_a < threshold and f_a_at_b < threshold)
        if not incompatible:
            self._adaptive_refinement_skip_count += 1
            self._adaptive_refinement_last_reason = (
                "cycle_pair_cross_visible_no_refine")
            return False

        refinable = []
        for oid in (int(current_id), int(blocker_id)):
            ob = by_id[oid]
            depth = int(ob.get("refinement_depth", 0))
            point_count = int(np.asarray(
                ob.get("points", []),
                dtype=np.float64).reshape(-1, 3).shape[0])
            if (depth < self.adaptive_refinement_max_depth and
                    point_count >= self.adaptive_refinement_min_points):
                refinable.append(oid)

        if not refinable:
            self._adaptive_refinement_skip_count += 1
            self._adaptive_refinement_last_reason = (
                "cycle_pair_at_refinement_floor")
            return False

        with self._adaptive_refinement_lock:
            if self._pending_refinement_ids:
                return True
            self._pending_refinement_ids = sorted(set(refinable))
            self._adaptive_refinement_trigger_count += 1
            self._adaptive_refinement_last_parent_ids = list(
                self._pending_refinement_ids)
            self._adaptive_refinement_last_reason = (
                "cycle_pair_refinement_requested")
        rospy.logwarn(
            "[vbc_blocker_stack] ADAPTIVE REFINE request parents=%s "
            "cross_f=(%+.4f,%+.4f) threshold=%+.4f",
            ":".join(str(v) for v in refinable),
            f_b_at_a, f_a_at_b, threshold)
        return True

    def _process_pending_refinement(
            self, trajectory, trajectory_received,
            trajectory_source: str, sweep_s: float) -> bool:
        with self._adaptive_refinement_lock:
            parent_ids = list(self._pending_refinement_ids)
        if not parent_ids:
            return False

        with self._obligation_lock:
            by_id = {
                int(ob["id"]): dict(ob) for ob in self._obligations}
        parents = [by_id[oid] for oid in parent_ids if oid in by_id]
        if len(parents) != len(parent_ids):
            with self._adaptive_refinement_lock:
                self._pending_refinement_ids = []
            self._adaptive_refinement_skip_count += 1
            self._adaptive_refinement_last_reason = (
                "parent_disappeared_before_refinement")
            return False

        child_regions = []
        family_records = {}
        for parent in parents:
            parent_id = int(parent["id"])
            depth = int(parent.get("refinement_depth", 0))
            points = np.asarray(
                parent.get("points", []),
                dtype=np.float64).reshape(-1, 3)
            parent_centroid = np.asarray(
                parent.get("centroid", np.mean(points, axis=0)),
                dtype=np.float64).reshape(3)
            family_id = parent_id
            family_records[family_id] = {
                "family_id": family_id,
                "root_centroid": parent_centroid.copy(),
                "root_xyz_min": np.min(points, axis=0).copy(),
                "root_xyz_max": np.max(points, axis=0).copy(),
                "root_keys": tuple(parent.get("keys", ())),
                "child_ids": [],
            }
            region = {
                "points": points.copy(),
                "keys": tuple(parent.get("keys", ())),
                "centroid": parent_centroid.copy(),
                "refinement_depth": depth + 1,
                "parent_obligation_id": parent_id,
                "root_obligation_id": int(
                    parent.get(
                        "root_obligation_id",
                        parent_id)
                    if int(parent.get("root_obligation_id", -1)) >= 0
                    else parent_id),
                "refinement_reason": "adaptive_dependency_conflict",
                "refinement_family_id": family_id,
            }
            split = self._split_region_for_refinement(
                region, force_split=True)
            if len(split) <= 1:
                continue
            child_regions.extend(split)

        if not child_regions:
            with self._adaptive_refinement_lock:
                self._pending_refinement_ids = []
            self._adaptive_refinement_skip_count += 1
            self._adaptive_refinement_last_reason = "no_splittable_parent"
            return False

        with self._obligation_lock:
            surviving_count = sum(
                1 for ob in self._obligations
                if int(ob["id"]) not in set(parent_ids))
        if surviving_count + len(child_regions) > self.max_obligations:
            with self._adaptive_refinement_lock:
                self._pending_refinement_ids = []
            self._adaptive_refinement_skip_count += 1
            self._adaptive_refinement_last_reason = (
                "refinement_capacity_exceeded")
            rospy.logwarn(
                "[vbc_blocker_stack] adaptive refinement skipped: "
                "survivors=%d children=%d max_obligations=%d",
                surviving_count, len(child_regions), self.max_obligations)
            return False

        children = []
        try:
            for region in child_regions:
                child = self._generate_new_obligation(
                    region, trajectory, float(sweep_s),
                    trajectory_received, trajectory_source)
                children.append(child)
        except Exception as exc:
            self._adaptive_refinement_failure_count += 1
            self._adaptive_refinement_last_reason = (
                "child_qvis_generation_failed")
            with self._adaptive_refinement_lock:
                self._pending_refinement_ids = []
            rospy.logerr(
                "[vbc_blocker_stack] adaptive refinement child generation "
                "failed; keeping coarse parents: %s", exc)
            return False

        with self._obligation_lock:
            existing_ids = {int(ob["id"]) for ob in self._obligations}
            if not all(oid in existing_ids for oid in parent_ids):
                self._adaptive_refinement_skip_count += 1
                self._adaptive_refinement_last_reason = (
                    "parent_cleared_during_child_generation")
                with self._adaptive_refinement_lock:
                    self._pending_refinement_ids = []
                return False
            self._obligations = [
                ob for ob in self._obligations
                if int(ob["id"]) not in set(parent_ids)]
            self._obligations.extend(children)

        for child in children:
            fid = int(child.get("refinement_family_id", -1))
            if fid in family_records:
                family_records[fid]["child_ids"].append(int(child["id"]))

        with self._adaptive_refinement_lock:
            for fid, record in family_records.items():
                if record["child_ids"]:
                    self._adaptive_refinement_families[int(fid)] = record
            self._pending_refinement_ids = []
            self._adaptive_refinement_success_count += 1
            self._adaptive_refinement_parent_count += len(parent_ids)
            self._adaptive_refinement_child_count += len(children)
            self._adaptive_refinement_last_child_ids = [
                int(ob["id"]) for ob in children]
            self._adaptive_refinement_last_reason = (
                "coarse_parents_replaced_by_children")

        # The dependency graph referred to the removed coarse nodes. Rebuild it
        # from exact VBC evidence instead of inheriting a stale parent cycle.
        child_ids = [int(ob["id"]) for ob in children]
        new_root = self._select_nearest_qvis_id(child_ids)
        with self._obligation_lock:
            self._repair_stack = [new_root] if new_root is not None else []
        self._pending_blocker_id = None
        self._pending_blocker_count = 0
        self._last_active_layer_ids = []
        self._last_active_layer_sweep_s = math.nan
        self._last_switch_reason = "adaptive_refinement_reset_stack"

        with self._progressive_shared_lock:
            self._progressive_shared_cache_key = None
            self._progressive_shared_cache = None
        with self._frontier_lock:
            self._frontier_cache_key = None
            self._frontier_cache = None

        rospy.logwarn(
            "[vbc_blocker_stack] ADAPTIVE REFINE success parents=%s "
            "children=%s new_root=%s",
            ":".join(str(v) for v in parent_ids),
            ":".join(str(v) for v in child_ids),
            str(new_root))
        return True

    def _match_candidate_compatible(
            self, ob: Dict[str, object], region: Dict[str, object]) -> bool:
        """Require the stored q_vis to remain valid for refreshed geometry.

        A spatial match is accepted only when the old q_vis still satisfies the
        learned visibility field for every point in the new region. If it does
        not, _match_existing returns no match and the normal obligation creation
        path generates a new q_vis tied to the new geometry instead of silently
        overwriting points/centroid under a stale q_vis.
        """
        self._qvis_match_check_count += 1
        oid = int(ob.get("id", -1))
        self._qvis_match_last_obligation_id = oid

        if int(ob.get("refinement_depth", 0)) > 0:
            ob_keys = set(ob.get("keys", ()))
            region_keys = set(region.get("keys", ()))
            if region_keys and not region_keys.issubset(ob_keys):
                self._qvis_match_reject_count += 1
                self._qvis_match_last_f_min = math.nan
                self._qvis_match_last_reason = (
                    "refined_child_rejects_coarse_geometry")
                return False

        q_vis = np.asarray(ob.get("q_vis", []), dtype=np.float64).reshape(-1)
        points = np.asarray(
            region.get("points", []), dtype=np.float64).reshape(-1, 3)
        if (q_vis.shape != (7,) or not np.all(np.isfinite(q_vis)) or
                points.shape[0] == 0 or not np.all(np.isfinite(points))):
            self._qvis_match_reject_count += 1
            self._qvis_match_last_f_min = math.nan
            self._qvis_match_last_reason = "invalid_qvis_or_region"
            return False

        try:
            x = torch.tensor(points, device=self.device, dtype=torch.float32)
            q = torch.tensor(
                q_vis.reshape(1, 7), device=self.device, dtype=torch.float32)
            values = self._per_point_values(x, q)
            f_min = float(np.min(values)) if values.size else -math.inf
        except Exception as exc:
            self._qvis_match_error_count += 1
            self._qvis_match_reject_count += 1
            self._qvis_match_last_f_min = math.nan
            self._qvis_match_last_reason = "learned_visibility_eval_error"
            rospy.logerr_throttle(
                1.0,
                "[vbc_blocker_stack] q_vis compatibility evaluation failed "
                "obligation=%d; refusing stale-identity reuse: %s",
                oid, exc)
            return False

        self._qvis_match_last_f_min = f_min
        compatible = bool(
            math.isfinite(f_min) and
            f_min + 1e-9 >= self._obligation_match_qvis_min_f)
        if compatible:
            self._qvis_match_accept_count += 1
            self._qvis_match_last_reason = "qvis_compatible"
            return True

        self._qvis_match_reject_count += 1
        self._qvis_match_last_reason = "qvis_incompatible_new_obligation"
        rospy.logwarn_throttle(
            0.5,
            "[vbc_blocker_stack] REJECT stale obligation identity id=%d "
            "old_q_vis new_region_f_min=%+.5f required=%+.5f; "
            "new geometry will receive its own q_vis",
            oid, f_min, self._obligation_match_qvis_min_f)
        return False

    def _update_matched_geometry_diagnostics(
            self, matched, region, source: str) -> None:
        """Refresh live matched-region geometry while preserving q_vis provenance.

        This intentionally keeps the existing planner semantics: matched
        obligations still inherit the newest region points/keys/centroid and
        q_vis is NOT regenerated. The additional state only measures whether
        the live geometry has drifted away from the geometry that originally
        generated q_vis.
        """
        old_centroid = np.asarray(
            matched.get("centroid", [math.nan, math.nan, math.nan]),
            dtype=np.float64).reshape(3)
        new_centroid = np.asarray(
            region["centroid"], dtype=np.float64).reshape(3)
        old_keys = tuple(matched.get("keys", ()))
        new_keys = tuple(region["keys"])
        qvis_source_centroid = np.asarray(
            matched.get("q_vis_source_centroid", old_centroid),
            dtype=np.float64).reshape(3)

        last_shift = (
            float(np.linalg.norm(new_centroid - old_centroid))
            if np.all(np.isfinite(old_centroid)) and
               np.all(np.isfinite(new_centroid))
            else math.nan)
        source_shift = (
            float(np.linalg.norm(new_centroid - qvis_source_centroid))
            if np.all(np.isfinite(qvis_source_centroid)) and
               np.all(np.isfinite(new_centroid))
            else math.nan)

        matched["geometry_match_update_count"] = int(
            matched.get("geometry_match_update_count", 0)) + 1
        changed = bool(old_keys != new_keys or (
            math.isfinite(last_shift) and last_shift > 1e-12))
        if changed:
            matched["geometry_match_change_count"] = int(
                matched.get("geometry_match_change_count", 0)) + 1
        matched["geometry_changed_since_qvis"] = bool(
            matched.get("geometry_changed_since_qvis", False) or
            old_keys != new_keys or
            (math.isfinite(source_shift) and source_shift > 1e-12))
        matched["last_match_centroid_shift_m"] = float(last_shift)
        if math.isfinite(source_shift):
            matched["max_centroid_shift_from_qvis_m"] = max(
                float(matched.get("max_centroid_shift_from_qvis_m", 0.0)),
                source_shift)
        matched["last_geometry_match_source"] = str(source)

        matched["last_seen_ros_s"] = rospy.Time.now().to_sec()
        matched["points"] = np.asarray(
            region["points"], dtype=np.float64).copy()
        matched["keys"] = new_keys
        matched["centroid"] = new_centroid.copy()

        if changed:
            rospy.logwarn_throttle(
                0.5,
                "[vbc_blocker_stack] MATCH geometry drift obligation=%d "
                "source=%s update=%d last_shift=%.4fm source_shift=%.4fm "
                "old_points=%d new_points=%d",
                int(matched["id"]), str(source),
                int(matched["geometry_match_update_count"]),
                float(last_shift), float(source_shift),
                len(old_keys), len(new_keys))

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

    def _progressive_priority_order(self, by_id, active_id):
        """Current blocker first; then urgent-layer regions; then the rest.

        Within the same importance tier prefer a q_vis closer to the measured
        configuration. This is only a steering preference; it cannot make an
        unsafe candidate executable.
        """
        q0 = None if self._latest_measured_q is None else np.asarray(
            self._latest_measured_q, dtype=np.float64)
        urgent = set(int(v) for v in self._last_active_layer_ids)

        def distance(oid):
            if q0 is None:
                return math.inf
            qv = np.asarray(by_id[oid].get("q_vis", []), dtype=np.float64)
            if qv.shape != (7,) or not np.all(np.isfinite(qv)):
                return math.inf
            return float(np.linalg.norm(qv - q0))

        others = [oid for oid in by_id if oid != active_id]
        others.sort(
            key=lambda oid: (
                0 if oid in urgent else 1,
                distance(oid),
                oid))
        return [active_id] + others

    def _dependency_cycle_active(
            self, stack: List[int], earliest_ids: List[int]) -> bool:
        if len(stack) < 2:
            return False
        current = int(stack[-1])
        earliest = set(int(v) for v in earliest_ids)
        if current in earliest:
            return False
        return any(int(oid) in earliest for oid in stack[:-1])

    def _compute_visibility_frontier(self):
        if not self.frontier_steering_enabled:
            return None

        with self._obligation_lock:
            copied = [dict(ob) for ob in self._obligations]
            stack = [
                int(oid) for oid in self._repair_stack
                if any(int(ob["id"]) == int(oid) for ob in self._obligations)
            ]
        if len(copied) < 2:
            return None

        by_id = {int(ob["id"]): ob for ob in copied}
        if not by_id:
            return None
        active_id = stack[-1] if stack else min(by_id)
        priority = self._progressive_priority_order(by_id, active_id)
        considered = [
            oid for oid in priority
            if oid in by_id][:self.frontier_max_regions]
        if len(considered) < 2:
            return None

        measured = (
            None if self._latest_measured_q is None
            else np.asarray(
                self._latest_measured_q, dtype=np.float64).copy())
        if (measured is None or measured.shape != (7,) or
                not np.all(np.isfinite(measured))):
            self._frontier_last_mode = "no_measured_q"
            return None

        cycle_active = self._dependency_cycle_active(
            stack, self._last_active_layer_ids)
        if cycle_active and self.adaptive_refinement_enabled:
            self._frontier_last_mode = (
                "cycle_deferred_to_adaptive_refinement")
            self._frontier_last_cycle_active = True
            self._frontier_last_weight_scale = 0.0
            self._frontier_last_qvis_weight_scale = 1.0
            return None
        quantized_q = tuple(
            int(round(float(v) / self.frontier_recompute_q_inf))
            for v in measured)
        geometry_key = tuple(
            (int(oid), tuple(by_id[oid].get("keys", ())))
            for oid in considered)
        cache_key = (
            tuple(considered),
            geometry_key,
            quantized_q,
            bool(cycle_active))

        with self._frontier_lock:
            if self._frontier_cache_key == cache_key:
                return self._frontier_cache

        q = torch.tensor(
            measured.reshape(1, 7),
            device=self.device, dtype=torch.float32)
        f_values = []
        gradients = []
        valid_ids = []
        try:
            for oid in considered:
                points = np.asarray(
                    by_id[oid].get("points", []),
                    dtype=np.float64).reshape(-1, 3)
                if (points.shape[0] == 0 or
                        not np.all(np.isfinite(points))):
                    continue
                x = torch.tensor(
                    points, device=self.device, dtype=torch.float32)
                f_tensor, grad_tensor, _ = model_value_and_grad_q(
                    x, q, self.model)
                f_val = float(f_tensor[0].detach().cpu().item())
                grad = grad_tensor[0].detach().cpu().numpy().astype(
                    np.float64)
                if (not math.isfinite(f_val) or
                        grad.shape != (7,) or
                        not np.all(np.isfinite(grad))):
                    continue
                f_values.append(f_val)
                gradients.append(grad)
                valid_ids.append(int(oid))
        except Exception as exc:
            self._frontier_error_count += 1
            self._frontier_last_mode = "gradient_eval_error"
            rospy.logerr_throttle(
                1.0,
                "[vbc_blocker_stack] visibility frontier gradient "
                "evaluation failed: %s", exc)
            return None

        if len(valid_ids) < 2:
            self._frontier_last_mode = "insufficient_valid_regions"
            return None

        f_arr = np.asarray(f_values, dtype=np.float64)
        grad_arr = np.asarray(gradients, dtype=np.float64)
        logits = -f_arr / self.frontier_softmin_temperature
        logits -= float(np.max(logits))
        weights = np.exp(logits)
        denom = float(np.sum(weights))
        if not math.isfinite(denom) or denom <= 0.0:
            self._frontier_last_mode = "invalid_softmin_weights"
            return None
        weights /= denom

        direction = np.sum(weights[:, None] * grad_arr, axis=0)
        direction_norm_inf = float(np.max(np.abs(direction)))
        if (not math.isfinite(direction_norm_inf) or
                direction_norm_inf <= self.frontier_gradient_eps):
            self._frontier_last_mode = "degenerate_softmin_gradient"
            return None

        step = (
            self.frontier_step_inf *
            direction / direction_norm_inf)
        target = measured + step
        target_tensor = torch.tensor(
            target.reshape(1, 7),
            device=self.device, dtype=torch.float32)
        target_tensor, _ = self._clamp(target_tensor)
        target = target_tensor[0].detach().cpu().numpy().astype(
            np.float64)
        target_shift_inf = float(
            np.max(np.abs(target - measured)))

        frontier_weight_scale = (
            self.frontier_cycle_weight_scale
            if cycle_active else self.frontier_base_weight_scale)
        qvis_weight_scale = (
            self.frontier_cycle_qvis_weight_scale
            if cycle_active else self.frontier_base_qvis_weight_scale)
        mode = (
            "cycle_boosted_softmin_frontier"
            if cycle_active else "base_softmin_frontier")
        result = {
            "active": True,
            "target": target,
            "frontier_weight_scale": float(frontier_weight_scale),
            "qvis_weight_scale": float(qvis_weight_scale),
            "cycle_active": bool(cycle_active),
            "mode": mode,
            "considered_ids": list(valid_ids),
            "f_values": {
                int(oid): float(fv)
                for oid, fv in zip(valid_ids, f_arr)},
            "softmin_weights": {
                int(oid): float(w)
                for oid, w in zip(valid_ids, weights)},
            "direction_norm_inf": direction_norm_inf,
            "target_shift_inf": target_shift_inf,
        }
        with self._frontier_lock:
            self._frontier_cache_key = cache_key
            self._frontier_cache = result
            self._frontier_compute_count += 1
            self._frontier_last_mode = mode
            self._frontier_last_considered_ids = list(valid_ids)
            self._frontier_last_f_values = dict(result["f_values"])
            self._frontier_last_softmin_weights = dict(
                result["softmin_weights"])
            self._frontier_last_direction_norm_inf = direction_norm_inf
            self._frontier_last_target_shift_inf = target_shift_inf
            self._frontier_last_weight_scale = float(
                frontier_weight_scale)
            self._frontier_last_qvis_weight_scale = float(
                qvis_weight_scale)
            self._frontier_last_cycle_active = bool(cycle_active)
        return result

    def _publish_visibility_frontier(self) -> None:
        if not hasattr(self, "frontier_target_pub"):
            return
        result = self._compute_visibility_frontier()
        msg = Float64MultiArray()
        if result is None:
            msg.data = [0.0, 0.0, 1.0] + [0.0] * 7
        else:
            msg.data = [
                1.0,
                float(result["frontier_weight_scale"]),
                float(result["qvis_weight_scale"]),
                *[float(v) for v in np.asarray(
                    result["target"], dtype=np.float64).reshape(7)],
            ]
            self._frontier_publish_count += 1
        self.frontier_target_pub.publish(msg)

        summary = String()
        summary.data = (
            "policy=softmin_visibility_frontier"
            f" enabled={int(self.frontier_steering_enabled)}"
            f" active={int(result is not None)}"
            f" mode={self._frontier_last_mode}"
            f" cycle_active={int(self._frontier_last_cycle_active)}"
            f" considered_ids="
            f"{':'.join(str(v) for v in self._frontier_last_considered_ids) or 'none'}"
            f" f_values="
            f"{';'.join(str(k)+':' + f'{v:.5f}' for k,v in sorted(self._frontier_last_f_values.items())) or 'none'}"
            f" softmin_weights="
            f"{';'.join(str(k)+':' + f'{v:.4f}' for k,v in sorted(self._frontier_last_softmin_weights.items())) or 'none'}"
            f" direction_norm_inf="
            f"{self._frontier_last_direction_norm_inf:.6f}"
            f" target_shift_inf="
            f"{self._frontier_last_target_shift_inf:.6f}"
            f" frontier_weight_scale="
            f"{self._frontier_last_weight_scale:.6f}"
            f" qvis_weight_scale="
            f"{self._frontier_last_qvis_weight_scale:.6f}"
            f" compute_count={self._frontier_compute_count}"
            f" publish_count={self._frontier_publish_count}"
            f" error_count={self._frontier_error_count}"
        )
        self.frontier_summary_pub.publish(summary)

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
            urgent = set(int(v) for v in self._last_active_layer_ids)
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
                            "[vbc_blocker_stack] C5.41 shared q_vis %s "
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
                    0 if oid in urgent else 1 for oid in optional)
                pool = [
                    oid for oid in optional
                    if (0 if oid in urgent else 1) == lowest_importance]
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
        by_id = {int(ob["id"]): ob for ob in copied}
        valid_stack = [oid for oid in stack if oid in by_id]
        active_id = valid_stack[-1] if valid_stack else min(by_id)

        priority_order = self._progressive_priority_order(by_id, active_id)
        shared = self._compute_progressive_shared_target(
            by_id, active_id, priority_order)

        first = dict(by_id[active_id])
        if shared is not None:
            first["q_vis"] = np.asarray(
                shared["q_vis"], dtype=np.float64).copy()
            first["q_zero"] = np.asarray(
                shared["q_zero"], dtype=np.float64).copy()
            first["final_f_min"] = float(shared["final_f_min"])
            first["shared_solution_mode"] = str(shared["mode"])

        rest = [by_id[oid] for oid in priority_order if oid != active_id]
        return [first] + rest

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
        # ROS trajectory times are floating-point values. A nominal 0.30 s
        # layer may arrive as 0.30000000000000004; treating that as strictly
        # beyond a 0.30 s blocker horizon silently prevents the second
        # confirmation and defeats recursive blocker preemption. Admit the
        # configured boundary with a tiny numerical tolerance.
        sweep_tol_s = 1e-9
        if (not math.isfinite(sweep_s) or
                sweep_s > self.blocker_push_max_sweep_s + sweep_tol_s):
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
            # First ask whether this is also a learned visibility conflict. If
            # so, refine the coarse spatial representation instead of repeatedly
            # negotiating between incompatible whole-region q_vis targets.
            self._stack_cycle_block_count += 1
            if self._request_cycle_refinement(int(current), int(blocker)):
                self._last_switch_reason = "adaptive_refinement_pending"
            else:
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
            routed = self._absorb_refined_partition_region(
                region, trajectory, float(sweep),
                trajectory_received, trajectory_source)
            if routed is not None:
                if int(routed) < 0:
                    all_regions_handled = False
                else:
                    active_ids.append(int(routed))
                    self._schedule_matched_obligations += 1
                continue

            with self._obligation_lock:
                matched = self._match_existing(region)
                if matched is not None:
                    self._update_matched_geometry_diagnostics(
                        matched, region, "candidate_vbc_active_set")
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
        self._process_pending_refinement(
            trajectory, trajectory_received,
            trajectory_source, float(sweep))
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
            self._publish_visibility_frontier()
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
            f" obligation_match_qvis_min_f={self._obligation_match_qvis_min_f:.6f}"
            f" qvis_match_check_count={self._qvis_match_check_count}"
            f" qvis_match_accept_count={self._qvis_match_accept_count}"
            f" qvis_match_reject_count={self._qvis_match_reject_count}"
            f" qvis_match_error_count={self._qvis_match_error_count}"
            f" qvis_match_last_obligation_id={self._qvis_match_last_obligation_id}"
            f" qvis_match_last_f_min={self._qvis_match_last_f_min:.6f}"
            f" qvis_match_last_reason={self._qvis_match_last_reason}"
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
            f" frontier_enabled={int(self.frontier_steering_enabled)}"
            f" frontier_mode={self._frontier_last_mode}"
            f" frontier_cycle_active={int(self._frontier_last_cycle_active)}"
            f" frontier_considered_ids="
            f"{':'.join(str(v) for v in self._frontier_last_considered_ids) or 'none'}"
            f" frontier_direction_norm_inf="
            f"{self._frontier_last_direction_norm_inf:.6f}"
            f" frontier_target_shift_inf="
            f"{self._frontier_last_target_shift_inf:.6f}"
            f" frontier_weight_scale="
            f"{self._frontier_last_weight_scale:.6f}"
            f" frontier_qvis_weight_scale="
            f"{self._frontier_last_qvis_weight_scale:.6f}"
            f" frontier_compute_count={self._frontier_compute_count}"
            f" frontier_publish_count={self._frontier_publish_count}"
            f" frontier_error_count={self._frontier_error_count}"
            f" adaptive_refinement_enabled="
            f"{int(self.adaptive_refinement_enabled)}"
            f" adaptive_refinement_target_diameter_m="
            f"{self.adaptive_refinement_target_diameter_m:.6f}"
            f" adaptive_refinement_max_depth="
            f"{self.adaptive_refinement_max_depth}"
            f" adaptive_refinement_trigger_count="
            f"{self._adaptive_refinement_trigger_count}"
            f" adaptive_refinement_success_count="
            f"{self._adaptive_refinement_success_count}"
            f" adaptive_refinement_failure_count="
            f"{self._adaptive_refinement_failure_count}"
            f" adaptive_refinement_skip_count="
            f"{self._adaptive_refinement_skip_count}"
            f" adaptive_refinement_parent_count="
            f"{self._adaptive_refinement_parent_count}"
            f" adaptive_refinement_child_count="
            f"{self._adaptive_refinement_child_count}"
            f" adaptive_refinement_last_parents="
            f"{':'.join(str(v) for v in self._adaptive_refinement_last_parent_ids) or 'none'}"
            f" adaptive_refinement_last_children="
            f"{':'.join(str(v) for v in self._adaptive_refinement_last_child_ids) or 'none'}"
            f" adaptive_refinement_cross_f_ab="
            f"{self._adaptive_refinement_last_cross_f_ab:.6f}"
            f" adaptive_refinement_cross_f_ba="
            f"{self._adaptive_refinement_last_cross_f_ba:.6f}"
            f" adaptive_refinement_refined_zone_count="
            f"{len(self._adaptive_refinement_families)}"
            f" adaptive_refinement_family_route_count="
            f"{self._adaptive_refinement_family_route_count}"
            f" adaptive_refinement_absorb_count="
            f"{self._adaptive_refinement_absorb_count}"
            f" adaptive_refinement_absorbed_point_count="
            f"{self._adaptive_refinement_absorbed_point_count}"
            f" adaptive_refinement_qvis_reuse_count="
            f"{self._adaptive_refinement_qvis_reuse_count}"
            f" adaptive_refinement_qvis_regen_count="
            f"{self._adaptive_refinement_qvis_regen_count}"
            f" adaptive_refinement_qvis_regen_failure_count="
            f"{self._adaptive_refinement_qvis_regen_failure_count}"
            f" adaptive_refinement_last_family_id="
            f"{self._adaptive_refinement_last_family_id}"
            f" adaptive_refinement_last_child_id="
            f"{self._adaptive_refinement_last_child_id}"
            f" adaptive_refinement_last_absorb_f_min="
            f"{self._adaptive_refinement_last_absorb_f_min:.6f}"
            f" adaptive_refinement_reason="
            f"{self._adaptive_refinement_last_reason}"
        )
        self.blocker_stack_summary_pub.publish(msg)
