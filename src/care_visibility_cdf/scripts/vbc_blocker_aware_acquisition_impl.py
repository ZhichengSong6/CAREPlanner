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

Non-urgent regions discovered by later trajectory evaluations remain queued.  The
progressive shared q_vis solver is allowed to combine regions only after a new
region has been confirmed as a path-associated earliest-layer blocker, or after
an explicit GCDF rejection has identified it on a refused candidate; unrelated
queued regions must not rewrite the current q_vis while it is being executed.
"""

from __future__ import annotations

import json
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
from observation_identity import observation_region_token, observation_token, vbc_bundle_identity
from bounded_visibility_recovery import VisibilityRecoveryBudget, region_key, certified_distinct_q
from candidate_replacement_runtime import attach_candidate_replacement


class BlockerAwareVisibilityAcquisitionWaypointNode(VisibilityAcquisitionWaypointNode):
    def __init__(self) -> None:
        self._c49_ready = False

        # C5.26: candidate VBC active-set geometry and sweep time are consumed
        # as one coherent message. Legacy split active-set/sweep topics remain
        # subscribed for diagnostics, but they do not own obligation lifetime.
        self._coherent_bundle_enabled = True
        self._coherent_bundle_lock = threading.Lock()
        self._coherent_bundle_seq = 0
        # Candidate and execution-audit selectors each start their publisher
        # sequence at one.  Keep a local monotonically increasing generation
        # for the shared processing queue, and track publisher sequence per
        # source so an execution bundle cannot be discarded as "stale" merely
        # because the candidate selector has already published many bundles.
        self._coherent_bundle_generation = 0
        self._coherent_bundle_last_candidate_seq = 0
        self._coherent_bundle_last_execution_seq = 0
        self._coherent_bundle_source = "none"
        self._coherent_bundle_source_seq = 0
        self._coherent_bundle_urgent_recovery = False
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
        # Target persistence: ordinary new obligations are queued.  Only a
        # confirmed blocker from the current earliest trajectory layer may join
        # the active target's shared solve and change q_vis.
        self._path_co_plan_ids: List[int] = []
        self._path_associated_ids: List[int] = []
        self._path_association_reason = "startup"
        self._path_associated_count = 0
        self._path_queued_count = 0
        self._process_attempt_count = 0
        self._process_success_count = 0
        self._last_process_reason = "startup"

        # C5.44: an active VBC obligation has two different kinds of state.
        # Its q_vis is the steering target and must remain fixed while the
        # same obligation is active.  Its reported voxels are safety evidence
        # and must accumulate; a later VBC bundle is allowed to add points but
        # must not make an earlier point disappear from the repair CDF query.
        # The lock is intentionally local to blocker-aware acquisition.  A
        # different obligation id still changes the active target normally.
        self._active_qvis_target_lock_enabled = True
        self._safety_union_merge_count = 0

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
        self._frontier_last_observation_token = "none"
        # A rejected candidate leaves measured q unchanged.  Keep a bounded,
        # target-scoped count so the producer can widen one later frontier
        # step instead of proposing the identical target forever.
        self._frontier_feedback_token = "none"
        self._frontier_feedback_seq = -1
        self._frontier_feedback_failures = 0
        self._frontier_last_escalation_active = False
        self._frontier_last_escalation_failures = 0

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

        # C5.12/C5.43 direct GCDF recovery evidence. Final executable GCDF and
        # local hard-GCDF may expose low-confidence voxels that the narrower VBC
        # swept-body selector does not report. Both publish the same
        # trajectory+exact-voxel transaction format and are converted directly
        # into ordinary visibility obligations using the same VisCDF projector.
        self._gcdf_recovery_lock = threading.Lock()
        self._gcdf_recovery_trajectory = None
        self._gcdf_recovery_trajectory_received = None
        self._gcdf_recovery_trajectory_cache = OrderedDict()
        self._gcdf_recovery_cache_capacity = 8
        self._pending_gcdf_recovery_event = None
        self._processed_gcdf_recovery_seq = 0
        # Header stamp is the immutable transaction identity. Sequence numbers
        # are publisher-local diagnostics, so they cannot be used to order
        # recovery events now that local and final GCDF share this channel.
        self._processed_gcdf_recovery_stamps = OrderedDict()
        self._processed_gcdf_recovery_stamp_capacity = 64
        self._gcdf_recovery_event_count = 0
        self._gcdf_recovery_generated_count = 0
        self._gcdf_recovery_match_count = 0
        self._gcdf_recovery_drop_count = 0
        self._last_gcdf_recovery_reason = "startup"
        self._last_gcdf_recovery_event_stamp = "none"
        self._last_gcdf_recovery_trajectory_stamp = "none"
        self._gcdf_recovery_cache_hit_count = 0
        self._gcdf_recovery_cache_miss_count = 0
        self._final_recovery_lock = threading.Lock()
        self._dependency_lock = threading.Lock()
        self._dependency_pending = None
        self._dependency_processing = False
        self._dependency_latest = (-1, -1, -1)
        self._dependency_attempts = {}
        self._dependency_pins = {}
        self._dependency_edges = {}  # child id -> parent id; only observation clears it
        self._dependency_reason = 'startup'
        self._dependency_pushes = 0
        self._dependency_resumes = 0
        self._final_recovery_pending = None
        self._final_recovery_active_request = None
        self._final_recovery_budget = VisibilityRecoveryBudget()
        # Published q_vis may come from a shared-solve snapshot rather than
        # the stored individual obligation. Keep a bounded identity history so
        # a temporarily preempted target can still be joined to its live
        # region without accepting an unboundedly old request.
        self._final_recovery_target_history = OrderedDict()
        self._final_recovery_target_history_capacity = 64
        # ``_final_recovery_latest_seq`` orders final-hold requests only.  A
        # planner summary for a newer plan may carry ``final_vbc_hold=0``
        # while the hold request is still waiting for the next active-set
        # callback.  That feedback must not cancel the request before it can
        # be consumed.
        self._final_recovery_latest_seq = -1
        self._final_recovery_latest_feedback_seq = -1
        self._final_recovery_feedback_count = 0
        self._final_recovery_hold_feedback_count = 0
        self._final_recovery_nonhold_feedback_count = 0
        self._final_recovery_preserved_pending_count = 0
        self._final_recovery_process_count = 0
        self._final_recovery_process_success_count = 0
        self._final_recovery_live_obligation_fallback_count = 0
        # A repeated scalar solve can converge to the same q_vis even though
        # the active obligation is still unsafe.  Permit one deterministic
        # alternate seed on that rare final-VBC hold; this is steering only and
        # remains behind the unchanged final GCDF + exact VBC gates.
        self._final_recovery_alternative_seed_attempt_count = 0
        self._final_recovery_alternative_seed_success_count = 0
        self._final_recovery_alternative_seed_rad = 0.05
        self._final_recovery_enabled = False
        self._final_recovery_last_reason = 'disabled'
        super().__init__()
        self._final_recovery_enabled = bool(rospy.get_param('~final_vbc_recovery_enabled', False))
        # Independent liveness trigger from certified, tracker-observed tiny
        # executions. It does not enable the legacy final-VBC-failure trigger.
        self._safe_stall_recovery_enabled = bool(rospy.get_param('~safe_stall_recovery_enabled', True))
        self._final_recovery_alternative_seed_rad = float(rospy.get_param(
            '~final_vbc_recovery_alternative_seed_rad', 0.05))
        if (not math.isfinite(self._final_recovery_alternative_seed_rad) or
                self._final_recovery_alternative_seed_rad <= 0.01 or
                self._final_recovery_alternative_seed_rad > 0.20):
            raise ValueError(
                '~final_vbc_recovery_alternative_seed_rad must be in (0.01, 0.20]')
        self._final_recovery_last_reason = ('waiting_final_vbc_hold' if self._final_recovery_enabled else 'disabled')

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
        self.progressive_shared_path_association_only = bool(rospy.get_param(
            "~progressive_shared_path_association_only", True))
        self._active_qvis_target_lock_enabled = bool(rospy.get_param(
            "~active_qvis_target_lock_enabled", True))
        self._obligation_match_qvis_min_f = float(rospy.get_param(
            "~obligation_match_qvis_min_f", 0.0))

        self.frontier_steering_enabled = bool(rospy.get_param(
            "~frontier_steering_enabled", True))
        # Phase-E strict mode: use q_vis only as a long-range direction.
        # The local planner is pulled toward one short measured-q -> q_vis step,
        # which still must pass final GCDF + exact VBC before execution.
        self.vbc_gated_frontier_step_enabled = bool(rospy.get_param(
            "~vbc_gated_frontier_step_enabled", False))
        self.frontier_max_regions = int(rospy.get_param(
            "~frontier_max_regions", 3))
        self.frontier_softmin_temperature = float(rospy.get_param(
            "~frontier_softmin_temperature", 0.05))
        self.frontier_step_inf = float(rospy.get_param(
            "~frontier_step_inf", 0.05))
        self.frontier_vbc_escalation_enabled = bool(rospy.get_param(
            "~frontier_vbc_escalation_enabled", False))
        self.frontier_vbc_escalation_after = int(rospy.get_param(
            "~frontier_vbc_escalation_after", 3))
        self.frontier_escalated_step_inf = float(rospy.get_param(
            "~frontier_escalated_step_inf",
            max(self.frontier_step_inf, 0.10)))
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
        # The execution audit is a separate, committed-trajectory safety
        # stream.  When configured, its UNKNOWN witness is a certified
        # recovery blocker and is handled immediately, while ordinary
        # candidate VBC discoveries retain the bounded queue/confirmation
        # policy above.
        self.execution_active_set_bundle_topic = str(rospy.get_param(
            "~execution_active_set_bundle_topic", "")).strip()
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
        if self.frontier_vbc_escalation_after < 1:
            raise ValueError("~frontier_vbc_escalation_after must be >= 1")
        if (not math.isfinite(self.frontier_escalated_step_inf) or
                self.frontier_escalated_step_inf < self.frontier_step_inf or
                self.frontier_escalated_step_inf > 0.20):
            raise ValueError(
                "~frontier_escalated_step_inf must be finite, >= frontier_step_inf, and <= 0.20")
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
        self.execution_active_set_bundle_sub = None
        if (self.execution_active_set_bundle_topic and
                self.execution_active_set_bundle_topic !=
                self.active_set_bundle_topic):
            self.execution_active_set_bundle_sub = rospy.Subscriber(
                self.execution_active_set_bundle_topic, Float64MultiArray,
                self._execution_active_set_bundle_cb, queue_size=1)
        self.gcdf_recovery_trajectory_sub = rospy.Subscriber(
            self.gcdf_recovery_trajectory_topic, JointTrajectory,
            self._gcdf_recovery_trajectory_cb, queue_size=1)
        self.gcdf_recovery_event_sub = rospy.Subscriber(
            self.gcdf_recovery_event_topic, Float64MultiArray,
            self._gcdf_recovery_event_cb, queue_size=1)
        if self._final_recovery_enabled or self._safe_stall_recovery_enabled:
            self.final_recovery_sub = rospy.Subscriber(str(rospy.get_param(
                '~final_vbc_recovery_feedback_topic', '/care_planner/local_planner/summary')),
                String, self._final_recovery_feedback_cb, queue_size=8)
        if self._safe_stall_recovery_enabled:
            self.observation_dependency_sub = rospy.Subscriber(
                '/care_planner/local_planner/observation_dependency', String,
                self._observation_dependency_cb, queue_size=8)
        self._candidate_replacement = attach_candidate_replacement(self)
        self._c49_ready = True
        self._prune_or_initialize_stack()
        self._publish_schedule()
        self._publish_blocker_stack_summary()
        rospy.logwarn(
            "[vbc_blocker_stack] C4.9/C5.26 ENABLED max_blocker_sweep=%.3fs "
            "confirmations=%d shared_path_only=%s coherent_bundle=%s "
            "execution_bundle=%s",
            self.blocker_push_max_sweep_s, self.blocker_confirmations,
            self.progressive_shared_path_association_only,
            self.active_set_bundle_topic,
            self.execution_active_set_bundle_topic or "disabled")

    def _active_set_callback(self, msg: Float64MultiArray) -> None:
        """Ignore the legacy split active-set stream in C5.26 blocker mode.

        The legacy topic is intentionally kept alive for diagnostics and older
        modes. Letting it mutate _raw_active_set/_sweep_time_s would reintroduce
        the exact cross-topic race that C5.26 removes.
        """
        if getattr(self, "_coherent_bundle_enabled", False):
            return
        super()._active_set_callback(msg)

    def _execution_active_set_bundle_cb(self, msg: Float64MultiArray) -> None:
        """Route committed-trajectory VBC evidence through urgent recovery."""
        self._active_set_bundle_cb(msg, urgent_recovery=True)

    def _active_set_bundle_cb(
            self, msg: Float64MultiArray, urgent_recovery: bool = False) -> None:
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

        source = "execution" if urgent_recovery else "candidate"
        with self._coherent_bundle_lock:
            last_source_seq = (
                self._coherent_bundle_last_execution_seq
                if source == "execution"
                else self._coherent_bundle_last_candidate_seq)
            if seq <= last_source_seq:
                self._coherent_bundle_last_reason = "stale_bundle"
                return
            was_pending = self._coherent_bundle_pending_nonempty
            self._coherent_bundle_generation += 1
            generation = self._coherent_bundle_generation
            self._coherent_bundle_seq = generation
            if source == "execution":
                self._coherent_bundle_last_execution_seq = seq
            else:
                self._coherent_bundle_last_candidate_seq = seq
            self._coherent_bundle_source = source
            self._coherent_bundle_source_seq = seq
            self._coherent_bundle_urgent_recovery = bool(urgent_recovery)
            self._trace_coherent_bundle_identity = vbc_bundle_identity(msg, seq)
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
            # Internal serials are local generations because candidate and
            # execution publishers have independent sequence domains.
            self._raw_active_set_serial = generation
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
            if stamp_key in self._processed_gcdf_recovery_stamps:
                self._last_gcdf_recovery_reason = "stale_processed_event_stamp"
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
                "[vbc_blocker_stack] GCDF blocker reopens acquisition "
                "episode seq=%d stamp=%s count=%d sweep=%.3fs",
                seq, self._stamp_text(stamp_key), count, sweep_s)

    def _process_gcdf_recovery_event(self) -> None:
        with self._gcdf_recovery_lock:
            event = (
                None if self._pending_gcdf_recovery_event is None
                else dict(self._pending_gcdf_recovery_event))

        if event is None:
            return
        seq = int(event["seq"])
        stamp_key = tuple(event["stamp_key"])
        with self._gcdf_recovery_lock:
            already_processed = (
                stamp_key in self._processed_gcdf_recovery_stamps)
        if already_processed:
            with self._gcdf_recovery_lock:
                self._pending_gcdf_recovery_event = None
            self._last_gcdf_recovery_reason = "already_processed_stamp"
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
                trajectory_received, "gcdf_rejected")
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
                        matched, region, "gcdf_recovery")
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
                    "gcdf_rejected")
            except Exception as exc:
                self._schedule_generation_failures += 1
                all_regions_handled = False
                self._last_gcdf_recovery_reason = "generation_failed"
                rospy.logerr(
                    "[vbc_blocker_stack] GCDF obligation generation "
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
                        "[vbc_blocker_stack] GCDF ADD obligation=%d "
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
            self._processed_gcdf_recovery_stamps[stamp_key] = seq
            self._processed_gcdf_recovery_stamps.move_to_end(stamp_key)
            while (len(self._processed_gcdf_recovery_stamps) >
                   self._processed_gcdf_recovery_stamp_capacity):
                self._processed_gcdf_recovery_stamps.popitem(last=False)
            self._gcdf_recovery_trajectory_cache.pop(stamp_key, None)
            if (self._pending_gcdf_recovery_event is not None and
                    int(self._pending_gcdf_recovery_event["seq"]) == seq):
                self._pending_gcdf_recovery_event = None
        self._last_gcdf_recovery_reason = "handled"

        self._prune_or_initialize_stack()
        # A final/local GCDF rejection is already a certified failure of the
        # candidate trajectory.  The resulting UNKNOWN region is therefore a
        # path blocker, even when its sweep time is just beyond the ordinary
        # earliest-layer window or when this is the first observation of it.
        # Promote it immediately; ordinary VBC active-set discoveries below
        # retain the bounded-window + confirmation policy.
        self._consider_active_layer(
            active_ids, new_ids, sweep_s, urgent_gcdf_recovery=True)
        self._process_pending_refinement(
            trajectory, trajectory_received,
            "gcdf_rejected", float(sweep_s))
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

        # Legacy cycle refinement replaces obligations and resets the whole
        # stack. It must not silently discard an in-flight sensing dependency
        # or make its disappearance look like actual observation completion.
        with self._obligation_lock:
            dependency_active = bool(getattr(self, '_dependency_edges', {}))
        if dependency_active:
            with self._adaptive_refinement_lock:
                self._pending_refinement_ids = []
            self._adaptive_refinement_skip_count += 1
            self._adaptive_refinement_last_reason = 'observation_dependency_in_flight'
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
            self._path_co_plan_ids = []
        self._pending_blocker_id = None
        self._pending_blocker_count = 0
        self._last_active_layer_ids = []
        self._last_active_layer_sweep_s = math.nan
        self._last_switch_reason = "adaptive_refinement_reset_stack"
        self._path_associated_ids = []
        self._path_association_reason = "adaptive_refinement_reset_stack"

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
            hybrid_used = bool(ob.get("per_sensor_hybrid_used", False))
            sensor_id = int(ob.get("per_sensor_selected_sensor_id", -1))
            if (hybrid_used and sensor_id >= 0 and
                    getattr(self, "_per_sensor_runtime", None) is not None):
                # Preserve the branch identity that certified this q_vis.
                # Reusing the scalar union score here would incorrectly reject
                # a true per-sensor candidate merely because the scalar model
                # underestimates that mode.
                f_min = float(
                    self._per_sensor_runtime.branch_score_numpy(
                        points, q_vis, sensor_id))
            else:
                x = torch.tensor(
                    points, device=self.device, dtype=torch.float32)
                q = torch.tensor(
                    q_vis.reshape(1, 7),
                    device=self.device, dtype=torch.float32)
                values = self._per_point_values(x, q)
                f_min = (
                    float(np.min(values)) if values.size else -math.inf)
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
            self._qvis_match_last_reason = (
                "per_sensor_qvis_compatible"
                if bool(ob.get("per_sensor_hybrid_used", False))
                else "qvis_compatible")
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

        A matched obligation keeps the original q_vis, while its safety
        geometry is accumulated as a monotone union of all matched VBC/CDF
        points. This prevents a later earliest-layer refresh from removing a
        previously reported unsafe voxel from the repair query.
        """
        old_points = np.asarray(
            matched.get("points", []), dtype=np.float64).reshape(-1, 3)
        old_centroid = np.asarray(
            matched.get("centroid", [math.nan, math.nan, math.nan]),
            dtype=np.float64).reshape(3)
        region_points = np.asarray(
            region.get("points", []), dtype=np.float64).reshape(-1, 3)
        if region_points.shape[0] == 0 or not np.all(np.isfinite(region_points)):
            return

        # Keep a monotone safety union for the lifetime of this obligation.
        # The VBC selector may report only the earliest layer on one callback
        # and a larger layer on the next callback. Replacing ``points`` would
        # remove a previously reported unsafe voxel from the repair query.
        by_key = {}
        for point in old_points:
            if np.all(np.isfinite(point)):
                by_key[self._cell_key(point)] = point.copy()
        before_count = len(by_key)
        for point in region_points:
            by_key[self._cell_key(point)] = point.copy()
        merged_keys = tuple(sorted(by_key.keys()))
        merged_points = np.asarray(
            [by_key[key] for key in merged_keys], dtype=np.float64).reshape(-1, 3)
        new_centroid = np.mean(merged_points, axis=0)
        added_count = len(merged_keys) - before_count
        if added_count > 0:
            self._safety_union_merge_count += int(added_count)
        old_keys = tuple(matched.get("keys", ()))
        new_keys = merged_keys
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
        matched["points"] = merged_points.copy()
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
            if not existing and hasattr(self, '_dependency_attempts'):
                # A genuinely completed acquisition episode may start anew;
                # movement, token changes and temporary preemption may not.
                self._dependency_attempts.clear()
            for child, parent in list(getattr(self, '_dependency_edges', {}).items()):
                if child not in existing or parent not in existing:
                    del self._dependency_edges[child]
                    if child not in existing and parent in existing:
                        self._dependency_resumes += 1
                        self._dependency_reason = 'child_observed_resume_parent'
            for oid in list(getattr(self, '_dependency_pins', {})):
                if oid not in existing:
                    del self._dependency_pins[oid]
            before = list(self._repair_stack)
            before_current = before[-1] if before else None
            self._repair_stack = [oid for oid in self._repair_stack if oid in existing]
            self._stack_pop_count += max(0, len(before) - len(self._repair_stack))
            if not self._repair_stack and existing:
                root = min(existing)
                self._repair_stack.append(root)
                self._last_switch_reason = "initialize_oldest_obligation"
            current = self._repair_stack[-1] if self._repair_stack else None
            self._path_co_plan_ids = [
                int(oid) for oid in self._path_co_plan_ids
                if int(oid) in existing
            ]
            # A completed target advances to the queued target.  That is an
            # intentional target transition, so the previous co-plan set must
            # not carry unrelated regions into the new episode.
            if before_current != current:
                self._path_co_plan_ids = []
                self._path_associated_ids = []
                self._path_association_reason = "active_target_changed"

    def _shared_path_candidate_ids(self, by_id, active_id):
        """Return regions allowed to influence the currently locked q_vis.

        By default, only the active target and a confirmed path-associated
        blocker may enter the progressive shared solve.  Newly discovered
        regions outside that set remain pending and cannot rewrite q_vis.
        ``progressive_shared_path_association_only`` is a diagnostic escape
        hatch for legacy A/B comparisons; the safety gates are unchanged.
        """
        if not self.progressive_shared_path_association_only:
            return set(int(oid) for oid in by_id)
        with self._obligation_lock:
            co_plan = set(int(oid) for oid in self._path_co_plan_ids)
        co_plan.add(int(active_id))
        return {oid for oid in co_plan if oid in by_id}

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
        by_id = {int(ob["id"]): ob for ob in copied}
        if not by_id:
            return None
        active_id = stack[-1] if stack else min(by_id)
        allowed_ids = self._shared_path_candidate_ids(by_id, active_id)
        copied = [ob for ob in copied if int(ob["id"]) in allowed_ids]
        by_id = {int(ob["id"]): ob for ob in copied}
        # The bounded VBC-gated mode is deliberately defined for one active
        # obligation as well.  The old multi-region soft-min mode still needs
        # two regions, but returning here would make the single-obligation
        # recovery path silently fall back to a full q_vis jump.  CASE001's
        # final UNKNOWN recovery commonly has exactly one active region.
        if not by_id:
            return None

        measured = (
            None if self._latest_measured_q is None
            else np.asarray(
                self._latest_measured_q, dtype=np.float64).copy())
        if (measured is None or measured.shape != (7,) or
                not np.all(np.isfinite(measured))):
            self._frontier_last_mode = "no_measured_q"
            return None

        if self.vbc_gated_frontier_step_enabled:
            with self._frontier_lock:
                rejection_failures = int(self._frontier_feedback_failures)
            escalation_active = bool(
                self.frontier_vbc_escalation_enabled and
                rejection_failures >= self.frontier_vbc_escalation_after)
            active = by_id.get(int(active_id))
            q_vis = np.asarray(
                [] if active is None else active.get("q_vis", []),
                dtype=np.float64).reshape(-1)
            qvis_token = (
                observation_token(active) if active is not None else "none")
            joint_mask = np.asarray(
                [1.0] * 7 if active is None else
                active.get("q_vis_joint_mask", [1.0] * 7),
                dtype=np.float64).reshape(7)
            # The waypoint publication is the executable q_vis snapshot. A
            # shared solve can refresh it before the obligation dictionary is
            # replaced; use that same target for the frontier direction and
            # expose its token in the diagnostic summary.
            with self._schedule_publish_lock:
                published = getattr(self, "_trace_published_target", None)
                if (published is not None and active is not None and
                        published.get("region_token") == observation_region_token(active)):
                    published_q = np.asarray(
                        published.get("q_vis", []), dtype=np.float64).reshape(-1)
                    if published_q.shape == (7,) and np.all(np.isfinite(published_q)):
                        q_vis = published_q
                        joint_mask = np.asarray(
                            published.get("q_vis_joint_mask", joint_mask),
                            dtype=np.float64).reshape(7)
                        qvis_token = str(
                            published.get("observation_token", qvis_token))
            if (q_vis.shape != (7,) or not np.all(np.isfinite(q_vis))):
                self._frontier_last_mode = "invalid_active_qvis"
                return None
            if (not np.all(np.isfinite(joint_mask)) or
                    np.any(joint_mask < 0.0) or np.any(joint_mask > 1.0)):
                joint_mask = np.ones(7, dtype=np.float64)

            delta = joint_mask * (q_vis - measured)
            delta_inf = float(np.max(np.abs(delta)))
            if not math.isfinite(delta_inf):
                self._frontier_last_mode = "invalid_active_qvis_delta"
                return None

            if delta_inf <= self.frontier_gradient_eps:
                target = q_vis.copy()
            else:
                step_limit = (self.frontier_escalated_step_inf
                              if escalation_active else self.frontier_step_inf)
                step_inf = min(step_limit, delta_inf)
                target = measured + step_inf * delta / delta_inf
                target_tensor = torch.tensor(
                    target.reshape(1, 7),
                    device=self.device, dtype=torch.float32)
                target_tensor, _ = self._clamp(target_tensor)
                target = target_tensor[0].detach().cpu().numpy().astype(
                    np.float64)

            target_shift_inf = float(
                np.max(np.abs(target - measured)))
            cycle_active = self._dependency_cycle_active(
                stack, self._last_active_layer_ids)
            result = {
                "active": True,
                "target": target,
                "q_vis_joint_mask": joint_mask.copy(),
                "frontier_weight_scale": 1.0,
                "qvis_weight_scale": 0.0,
                "cycle_active": bool(cycle_active),
                "mode": "vbc_gated_single_obligation_frontier_step",
                "considered_ids": [int(active_id)],
                "f_values": {},
                "softmin_weights": {},
                "direction_norm_inf": delta_inf,
                "target_shift_inf": target_shift_inf,
                "observation_token": qvis_token,
                "vbc_rejection_failures": rejection_failures,
                "escalation_active": escalation_active,
            }
            with self._frontier_lock:
                self._frontier_cache_key = None
                self._frontier_cache = result
                self._frontier_compute_count += 1
                self._frontier_last_mode = result["mode"]
                self._frontier_last_considered_ids = [int(active_id)]
                self._frontier_last_f_values = {}
                self._frontier_last_softmin_weights = {}
                self._frontier_last_direction_norm_inf = delta_inf
                self._frontier_last_target_shift_inf = target_shift_inf
                self._frontier_last_weight_scale = 1.0
                self._frontier_last_qvis_weight_scale = 0.0
                self._frontier_last_cycle_active = bool(cycle_active)
                self._frontier_last_observation_token = qvis_token
                self._frontier_last_escalation_active = escalation_active
                self._frontier_last_escalation_failures = rejection_failures
            return result

        # The legacy soft-min frontier is a multi-region objective and cannot
        # be formed from a single region.  Keep that behavior unchanged after
        # allowing the bounded single-obligation branch above.
        if len(copied) < 2:
            return None

        priority = self._progressive_priority_order(by_id, active_id)
        considered = [
            oid for oid in priority
            if oid in by_id][:self.frontier_max_regions]
        if len(considered) < 2:
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
        with self._frontier_lock:
            self._frontier_last_observation_token = (
                "none" if result is None else
                str(result.get("observation_token", "none")))
        msg = Float64MultiArray()
        if result is None:
            msg.data = [0.0, 0.0, 1.0] + [0.0] * 7 + [1.0] * 7
        else:
            joint_mask = np.asarray(
                result.get("q_vis_joint_mask", [1.0] * 7),
                dtype=np.float64).reshape(7)
            msg.data = [
                1.0,
                float(result["frontier_weight_scale"]),
                float(result["qvis_weight_scale"]),
                *[float(v) for v in np.asarray(
                    result["target"], dtype=np.float64).reshape(7)],
                *[float(v) for v in joint_mask],
            ]
            self._frontier_publish_count += 1
        self.frontier_target_pub.publish(msg)

        summary = String()
        summary.data = (
            "policy=visibility_frontier"
            f" enabled={int(self.frontier_steering_enabled)}"
            f" vbc_gated_step={int(self.vbc_gated_frontier_step_enabled)}"
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
            f" vbc_rejection_failures={self._frontier_last_escalation_failures}"
            f" escalation_active={int(self._frontier_last_escalation_active)}"
            f" observation_token={self._frontier_last_observation_token}"
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
        if self._active_qvis_target_lock_enabled:
            # Geometry belongs to the safety union.  It must not invalidate
            # the steering cache while the active obligation id and the
            # allowed path members remain unchanged. A new active id or a new
            # path-associated id still changes this key and gets a fresh q_vis.
            geometry_key = tuple((oid,) for oid in considered)
        else:
            geometry_key = tuple(
                (oid, tuple(by_id[oid].get("keys", ())))
                for oid in considered)
        layer_key = (
            () if self._active_qvis_target_lock_enabled else
            tuple(sorted(int(v) for v in self._last_active_layer_ids)))
        cache_key = (
            int(active_id),
            tuple(considered),
            geometry_key,
            layer_key)
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
                            "target_member_regions": [observation_region_token(by_id[oid]) for oid in active],
                            "target_solver_points": points.tolist(),
                            "q_vis": qv.copy(),
                            "q_zero": np.asarray(
                                result["q_zero"], dtype=np.float64).reshape(7).copy(),
                            "final_f_min": float(result["final_f_min"]),
                            "q_vis_joint_mask": np.asarray(
                                result.get("q_vis_joint_mask", [1.0] * 7),
                                dtype=np.float64).reshape(7).copy(),
                            "per_sensor_hybrid_used": bool(
                                result.get("per_sensor_hybrid_used", False)),
                            "per_sensor_selected_sensor_id": int(
                                result.get("per_sensor_selected_sensor_id", -1)),
                            "per_sensor_selected_sensor_frame": str(
                                result.get("per_sensor_selected_sensor_frame", "none")),
                            "per_sensor_selected_rank": int(
                                result.get("per_sensor_selected_rank", -1)),
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
        allowed_ids = self._shared_path_candidate_ids(by_id, active_id)
        shared_priority_order = [
            oid for oid in priority_order if oid in allowed_ids]
        # Retain a certified bounded refresh for this exact region. Otherwise
        # the shared-target cache could immediately overwrite it with the pose
        # whose final audit exhausted the budget. Other obligations stay live.
        recovering = (by_id[active_id].get('final_recovery_individual_region') ==
                      region_key(by_id[active_id]['points']))
        pin = getattr(self, '_dependency_pins', {}).get(active_id)
        pinned = pin is not None and (pin['region'] == region_key(by_id[active_id]['points']) or
            (pin.get('preserve_on_growth', False) and
             set(pin['region']).issubset(set(region_key(by_id[active_id]['points'])))))
        # High-witness experiments test the selected EE sensor candidate itself.
        # Shared generation must not silently replace it or bypass its ledger.
        high_ee_target = bool(
            getattr(self, 'per_sensor_high_witness_priority_enabled', False) and
            len(by_id[active_id]['points']) and np.all(
                np.asarray(by_id[active_id]['points'])[:, 2] >=
                getattr(self, 'per_sensor_high_witness_z_min', .85)))
        recovering = recovering or pinned or high_ee_target
        shared = None if recovering else self._compute_progressive_shared_target(
            by_id, active_id, shared_priority_order)
        if (shared is None and not recovering and
                self.progressive_shared_path_association_only and
                self.progressive_shared_repair_enabled):
            # Make the target lock explicit in the runtime summary.  This is
            # the normal case for queued, non-path-associated obligations.
            self._progressive_shared_last_mode = "current_target_locked"
            self._progressive_shared_last_considered_ids = [int(active_id)]
            self._progressive_shared_last_kept_ids = [int(active_id)]
            self._progressive_shared_last_dropped_ids = [
                int(oid) for oid in priority_order if oid != active_id]
            self._progressive_shared_last_slacks = {}

        first = dict(by_id[active_id])
        if pinned:
            first['q_vis'] = pin['q_vis'].copy()
            first['shared_solution_mode'] = 'observation_dependency_target_locked'
        if shared is not None:
            first["q_vis"] = np.asarray(
                shared["q_vis"], dtype=np.float64).copy()
            first["q_zero"] = np.asarray(
                shared["q_zero"], dtype=np.float64).copy()
            first["final_f_min"] = float(shared["final_f_min"])
            first["q_vis_joint_mask"] = np.asarray(
                shared["q_vis_joint_mask"], dtype=np.float64).copy()
            first["per_sensor_hybrid_used"] = bool(
                shared["per_sensor_hybrid_used"])
            first["per_sensor_selected_sensor_id"] = int(
                shared["per_sensor_selected_sensor_id"])
            first["per_sensor_selected_sensor_frame"] = str(
                shared["per_sensor_selected_sensor_frame"])
            first["per_sensor_selected_rank"] = int(
                shared["per_sensor_selected_rank"])
            first["shared_solution_mode"] = str(shared["mode"])
            first["target_member_regions"] = list(shared["target_member_regions"])
            first["target_solver_points"] = list(shared["target_solver_points"])

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

    def _consider_active_layer(
            self, active_ids: List[int], new_ids: List[int], sweep_s: float,
            urgent_gcdf_recovery: bool = False) -> None:
        self._prune_or_initialize_stack()
        active_ids = sorted(set(int(v) for v in active_ids))
        new_ids = sorted(set(int(v) for v in new_ids))
        self._last_active_layer_ids = active_ids
        self._last_active_layer_sweep_s = float(sweep_s)

        with self._obligation_lock:
            current = self._repair_stack[-1] if self._repair_stack else None
            stack = list(self._repair_stack)

        self._path_associated_ids = []
        if current is None or not active_ids:
            self._pending_blocker_id = None
            self._pending_blocker_count = 0
            self._path_association_reason = "no_active_target_or_layer"
            return

        # A QP-failed q_vis owns the sensor-replacement experiment until its
        # replacement target has received a hard-QP result. Keep newly found
        # path blockers live and queued, but do not let them change the stack
        # top (and invalidate the authenticated replacement identity) mid-flight.
        replacement = getattr(self, '_candidate_replacement', None)
        locked_owner = (replacement.locked_qp_owner_id()
                        if replacement is not None else None)
        if locked_owner is not None:
            candidates = [
                oid for oid in new_ids
                if oid != locked_owner and oid in active_ids]
            if (not candidates and self._pending_blocker_id is not None and
                    self._pending_blocker_id in active_ids and
                    self._pending_blocker_id != locked_owner):
                candidates = [int(self._pending_blocker_id)]
            if not candidates and locked_owner not in active_ids:
                candidates = [oid for oid in active_ids if oid != locked_owner]
            blocker = self._select_nearest_qvis_id(candidates)
            self._path_associated_ids = list(candidates)
            with self._obligation_lock:
                self._path_co_plan_ids = [int(locked_owner)]
            if blocker is not None:
                if self._pending_blocker_id == blocker:
                    self._pending_blocker_count += 1
                else:
                    self._pending_blocker_id = int(blocker)
                    self._pending_blocker_count = 1
            self._path_association_reason = "qp_replacement_owner_locked"
            self._last_switch_reason = "qp_replacement_owner_locked"
            return
        # ROS trajectory times are floating-point values. A nominal 0.30 s
        # layer may arrive as 0.30000000000000004; treating that as strictly
        # beyond a 0.30 s blocker horizon silently prevents the second
        # confirmation and defeats recursive blocker preemption. Admit the
        # configured boundary with a tiny numerical tolerance.
        sweep_tol_s = 1e-9
        if (not urgent_gcdf_recovery and
                (not math.isfinite(sweep_s) or
                 sweep_s > self.blocker_push_max_sweep_s + sweep_tol_s)):
            self._pending_blocker_id = None
            self._pending_blocker_count = 0
            self._path_association_reason = "queued_non_path_layer"
            self._path_queued_count += len(new_ids)
            return

        # A new region is path-associated only when it is part of the current
        # earliest predicted sweep.  This is the temporal/spatial association
        # available at this layer: unrelated later regions are queued and must
        # not rewrite the active q_vis.  An absent current target is a separate
        # safety invalidation path, not an ordinary queue insertion.
        candidates = [
            oid for oid in new_ids
            if oid != current and oid in active_ids
        ]
        self._path_associated_ids = list(candidates)
        self._path_associated_count += len(candidates)
        self._path_association_reason = (
            "path_associated_earliest_layer"
            if candidates else "no_new_path_associated_obligation")

        # Matching the same region on the next coherent bundle is still part
        # of the same path event. Keep the confirmation alive even though the
        # region is no longer listed in ``new_ids`` after the first insertion.
        if (not candidates and self._pending_blocker_id is not None and
                self._pending_blocker_id in active_ids and
                self._pending_blocker_id != current):
            candidates = [int(self._pending_blocker_id)]
            self._path_associated_ids = list(candidates)
            self._path_association_reason = (
                "path_associated_confirmation_refresh")

        # If the active target disappeared from the earliest layer, allow the
        # existing blocker mechanism to fail safe. This is not used for normal
        # new-obligation queuing.
        if not candidates and current not in active_ids:
            candidates = [oid for oid in active_ids if oid != current]
            self._path_associated_ids = list(candidates)
            self._path_association_reason = "active_target_missing_urgent_layer"
        if any(oid in getattr(self, '_dependency_edges', {}) for oid in stack):
            # An ancestor seen again in a VBC bundle is not a new emergency.
            # Keep the unfinished child active instead of O1->A->O1 ping-pong.
            candidates = [oid for oid in candidates if oid not in stack[:-1]]
            if not candidates:
                self._path_association_reason = 'observation_dependency_cycle_hold'
        if not candidates:
            self._pending_blocker_id = None
            self._pending_blocker_count = 0
            return

        blocker = self._select_nearest_qvis_id(candidates)
        if blocker is None:
            return
        if blocker in stack:
            # A confirmed path blocker can already be below the current target
            # in the recursive stack (for example O1->O4->O3 while the route
            # to O3 keeps sweeping O4).  Treating that as a permanent cycle
            # hold leaves the planner retrying the blocked target forever.
            # If the same two regions have already formed a path pair, however,
            # reversing the stack on every bundle creates a two-node ping-pong
            # (O4->O3->O4...).  Keep the current top target and retain the pair
            # for the progressive shared q_vis solve; a real target transition
            # is still allowed after one member is seen and pruned.
            with self._obligation_lock:
                previous_pair = set(int(oid) for oid in self._path_co_plan_ids)
            if previous_pair == {int(current), int(blocker)}:
                self._stack_cycle_block_count += 1
                with self._obligation_lock:
                    self._path_co_plan_ids = [int(current), int(blocker)]
                self._pending_blocker_id = None
                self._pending_blocker_count = 0
                self._last_switch_reason = (
                    "path_associated_reciprocal_pair_held")
                self._path_association_reason = (
                    "path_associated_reciprocal_pair_held")
                rospy.logwarn_throttle(
                    0.5,
                    "[vbc_blocker_stack] HOLD reciprocal path pair "
                    "current=%d blocker=%d stack=%s",
                    int(current), int(blocker),
                    ":".join(str(v) for v in self._repair_stack))
                return
            # Move the existing blocker to the top while retaining the old
            # target below it; once the blocker is actually seen, normal stack
            # pruning exposes the saved target again.  This branch is reached
            # only for an earliest-layer/path-associated candidate, so ordinary
            # queued obligations still cannot preempt the active target.
            self._stack_cycle_block_count += 1
            with self._obligation_lock:
                self._repair_stack = [
                    int(oid) for oid in self._repair_stack
                    if int(oid) != int(blocker)]
                self._repair_stack.append(int(blocker))
                self._path_co_plan_ids = [int(current), int(blocker)]
            self._pending_blocker_id = None
            self._pending_blocker_count = 0
            self._last_switch_reason = "path_associated_blocker_reordered"
            self._path_association_reason = (
                "path_associated_blocker_reordered")
            rospy.logwarn(
                "[vbc_blocker_stack] REORDER path blocker=%d current=%d stack=%s",
                int(blocker), int(current),
                ":".join(str(v) for v in self._repair_stack))
            return

        # A GCDF rejection is tied to the exact candidate that was refused,
        # so a second temporal confirmation would only replay the same blocked
        # target.  Keep confirmation for ordinary VBC active-set updates, but
        # promote this certified recovery blocker immediately.
        if urgent_gcdf_recovery:
            with self._obligation_lock:
                existing = {int(ob["id"]) for ob in self._obligations}
                if blocker in existing and blocker not in self._repair_stack:
                    previous_current = current
                    self._repair_stack.append(blocker)
                    self._path_co_plan_ids = [
                        int(oid) for oid in (previous_current, blocker)
                        if oid is not None
                    ]
                    self._stack_push_count += 1
                    self._last_switch_reason = (
                        "gcdf_recovery_blocker_pushed")
                    self._path_association_reason = (
                        "gcdf_recovery_blocker_pushed")
                    rospy.logwarn(
                        "[vbc_blocker_stack] PUSH urgent GCDF blocker=%d "
                        "sweep=%.3fs stack=%s",
                        blocker, sweep_s,
                        ":".join(str(v) for v in self._repair_stack))
            self._pending_blocker_id = None
            self._pending_blocker_count = 0
            return

        if self._pending_blocker_id == blocker:
            self._pending_blocker_count += 1
        else:
            self._pending_blocker_id = blocker
            self._pending_blocker_count = 1
        if self._pending_blocker_count < self.blocker_confirmations:
            self._path_association_reason = "path_associated_pending_confirmation"
            return

        with self._obligation_lock:
            existing = {int(ob["id"]) for ob in self._obligations}
            if blocker in existing and blocker not in self._repair_stack:
                previous_current = current
                self._repair_stack.append(blocker)
                self._path_co_plan_ids = [
                    int(oid) for oid in (previous_current, blocker)
                    if oid is not None
                ]
                self._stack_push_count += 1
                self._last_switch_reason = "path_associated_blocker_pushed"
                self._path_association_reason = "path_associated_blocker_pushed"
                rospy.logwarn(
                    "[vbc_blocker_stack] PUSH blocker=%d sweep=%.3fs stack=%s",
                    blocker, sweep_s,
                    ":".join(str(v) for v in self._repair_stack))
        self._pending_blocker_id = None
        self._pending_blocker_count = 0

    def _observation_dependency_cb(self, msg):
        """Queue a fresh local UNKNOWN blocking hypothesis, never execution authority."""
        try:
            def unique(items):
                result = {}
                for key, value in items:
                    if key in result:
                        raise ValueError('duplicate field')
                    result[key] = value
                return result
            event = json.loads(msg.data, object_pairs_hook=unique)
            identity = tuple(int(event[k]) for k in ('mode_epoch', 'plan_seq', 'query_stamp_ns'))
            age = rospy.Time.now().to_sec() - float(event['query_ros_s'])
            points = np.asarray(event['points'], dtype=np.float64)
            if (event['version'] != 1 or min(identity) < 0 or identity[2] == 0 or
                    not 0. <= age <= .5 or points.ndim != 2 or points.shape[1] != 3 or
                    not 1 <= len(points) <= 32 or not np.all(np.isfinite(points)) or
                    not event['observation_token'].startswith('care_obs_v1_')):
                return
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
            return
        with self._dependency_lock:
            if identity <= self._dependency_latest:
                return
            self._dependency_latest = identity
            self._dependency_pending = event

    def _process_observation_dependency(self):
        if not hasattr(self, '_dependency_lock'):
            return
        with self._dependency_lock:
            if getattr(self, '_dependency_processing', False):
                return
            event, self._dependency_pending = self._dependency_pending, None
            if event is None:
                return
            self._dependency_processing = True
        try:
            self._process_observation_dependency_event(event)
        finally:
            with self._dependency_lock:
                self._dependency_processing = False

    def _process_observation_dependency_event(self, event):
        if event is None:
            return
        self._dependency_reason = 'stale_dependency'
        age = rospy.Time.now().to_sec() - float(event['query_ros_s'])
        if not 0. <= age <= 1.:
            return
        with self._schedule_publish_lock, self._obligation_lock:
            published = getattr(self, '_trace_published_target', None)
            if not published or published['observation_token'] != event['observation_token']:
                return
            parent = next((ob for ob in self._obligations
                if self._repair_stack and int(ob['id']) == self._repair_stack[-1] and
                observation_region_token(ob) == published['region_token']), None)
            if parent is None:
                return
            parent = dict(parent)
            parent_q = np.asarray(published['q_vis'], dtype=np.float64).copy()
            if parent_q.shape != (7,) or not np.all(np.isfinite(parent_q)):
                return
            snapshot = [dict(ob) for ob in self._obligations]
            ancestors = set(self._repair_stack)
        # Exact point ownership, not proximity. Choose one child at a time.
        child, point, budget_key = None, None, None
        for p in event['points']:
            point_key = region_key([p])[0]
            owners = [ob for ob in snapshot if point_key in region_key(ob['points'])]
            if any(int(ob['id']) in ancestors for ob in owners):
                self._dependency_reason = 'self_or_ancestor_dependency_hold'
                continue
            key = (region_key(parent['points']), point_key)
            if self._dependency_attempts.get(key, 0) >= 2:
                self._dependency_reason = 'dependency_budget_exhausted'
                continue
            child = min(owners, key=lambda ob: int(ob['id'])) if owners else None
            point, budget_key = p, key
            break
        if point is None:
            replacement = getattr(self, '_candidate_replacement', None)
            if replacement is not None and self._dependency_reason == 'self_or_ancestor_dependency_hold':
                replacement.offer(event)
            return
        if budget_key not in self._dependency_attempts and len(self._dependency_attempts) >= 64:
            self._dependency_reason = 'dependency_capacity_hold'
            return
        if child is None:
            with self._lock:
                trajectory, received, _ = self._preferred_trajectory_locked()
            if trajectory is None or self._latest_measured_q is None:
                self._dependency_reason = 'dependency_waiting_state'
                return
            if len(snapshot) >= self.max_obligations:
                self._dependency_reason = 'dependency_obligation_capacity_hold'
                return
        self._dependency_attempts[budget_key] = self._dependency_attempts.get(budget_key, 0) + 1
        new_child = child is None
        if new_child:
            try:
                region = self._cluster_regions(np.asarray([point], dtype=np.float64))[0]
                with self._progressive_shared_lock:
                    child = self._generate_new_obligation(
                        region, trajectory, .1, received, 'unknown_constraint_dependency')
            except Exception as exc:
                self._dependency_reason = 'dependency_generation_failed'
                rospy.logwarn('observation dependency generation failed: %s', exc)
                return
        child_q = np.asarray(child['q_vis'], dtype=np.float64)
        if child_q.shape != (7,) or not np.all(np.isfinite(child_q)):
            self._dependency_reason = 'dependency_invalid_target'
            return
        # Preserve the publication transaction, but release the non-reentrant
        # obligation lock before publication re-enters _ordered_obligations().
        with self._schedule_publish_lock:
            with self._obligation_lock:
                published = getattr(self, '_trace_published_target', None)
                age = rospy.Time.now().to_sec() - float(event['query_ros_s'])
                live = {int(ob['id']): ob for ob in self._obligations}
                pid, cid = int(parent['id']), int(child['id'])
                if (not 0. <= age <= 2. or not published or
                        published['observation_token'] != event['observation_token'] or
                        not self._repair_stack or self._repair_stack[-1] != pid or pid not in live or
                        region_key(live[pid]['points']) != region_key(parent['points'])):
                    self._dependency_reason = 'dependency_changed_during_generation'
                    return
                if new_child:
                    # A concurrently discovered owner wins; do not duplicate it.
                    owner = next((ob for ob in self._obligations
                        if region_key([point])[0] in region_key(ob['points'])), None)
                    if owner is not None:
                        child, cid = owner, int(owner['id'])
                    elif len(live) < self.max_obligations:
                        self._obligations.append(child)
                        self._schedule_new_obligations += 1
                    else:
                        self._dependency_reason = 'dependency_obligation_capacity_hold'
                        return
                elif cid not in live or region_key(live[cid]['points']) != region_key(child['points']):
                    self._dependency_reason = 'dependency_child_changed'
                    return
                if cid in self._repair_stack:
                    self._dependency_reason = 'dependency_cycle_hold'
                    return
                child_q = np.asarray(child['q_vis'], dtype=np.float64)
                if child_q.shape != (7,) or not np.all(np.isfinite(child_q)):
                    self._dependency_reason = 'dependency_invalid_target'
                    return
                self._dependency_pins[pid] = dict(region=region_key(parent['points']), q_vis=parent_q)
                self._dependency_pins[cid] = dict(region=region_key(child['points']), q_vis=child_q.copy())
                self._dependency_edges[cid] = pid
                self._repair_stack.append(cid)
                self._stack_push_count += 1
                self._dependency_pushes += 1
                self._dependency_reason = self._last_switch_reason = 'unknown_constraint_dependency_push'
                self._path_co_plan_ids = []
                self._pending_blocker_id = None
                self._pending_blocker_count = 0
                self._progressive_shared_cache_key = None
                self._progressive_shared_cache = None
                self._trace_observation('observation_dependency_push', parent_id=pid, child_id=cid,
                    query_stamp_ns=event['query_stamp_ns'], points=[point],
                    observation_token=event['observation_token'])
            self._publish_schedule()
        self._publish_blocker_stack_summary()

    def _final_recovery_feedback_cb(self, msg):
        """Queue only; generation stays off the ROS feedback thread."""
        fields = {}
        for word in msg.data.split():
            if '=' not in word:
                continue
            key, value = word.split('=', 1)
            if key in fields:
                return
            fields[key] = value
        try:
            seq = int(fields['plan_seq'])
        except (KeyError, ValueError):
            return
        with self._final_recovery_lock:
            self._final_recovery_feedback_count += 1
            if seq < self._final_recovery_latest_feedback_seq:
                return
            self._final_recovery_latest_feedback_seq = seq
            is_repair = fields.get('repair') == '1'
            is_probe = fields.get('probe') == '1'
            token = fields.get('observation_token', '')
            # The local planner reports the cumulative final-VBC rejection
            # count before entering its no-progress hold. Feed it back to the
            # frontier producer, scoped by the immutable q_vis token. A new
            # q_vis therefore starts a fresh bounded escalation ladder.
            if is_repair and not is_probe and token.startswith('care_obs_v1_'):
                try:
                    failures = int(fields.get('final_vbc_failures', ''))
                except (TypeError, ValueError):
                    failures = None
                if failures is not None and failures >= 0:
                    with self._frontier_lock:
                        if token != self._frontier_feedback_token:
                            self._frontier_feedback_token = token
                            self._frontier_feedback_seq = -1
                            self._frontier_feedback_failures = 0
                        if seq >= self._frontier_feedback_seq:
                            self._frontier_feedback_seq = seq
                            self._frontier_feedback_failures = max(
                                self._frontier_feedback_failures, failures)
                        # The strict single-obligation path normally bypasses
                        # the general frontier cache, but invalidating here
                        # keeps the feedback contract correct for soft-min A/B
                        # runs too.
                        self._frontier_cache_key = None
            safe_stall = (getattr(self, '_safe_stall_recovery_enabled', False) and
                          fields.get('observation_dependency_policy') != '1' and
                          fields.get('safe_stall_reselect') == '1')
            final_hold = (self._final_recovery_enabled and
                          fields.get('final_vbc_hold') == '1')
            if (not is_repair or is_probe or not (safe_stall or final_hold)):
                # A newer non-hold summary is not a cancellation.  It is
                # common for the planner to publish the same plan_seq first
                # with final_vbc_hold=0 and then publish the terminal hold;
                # it can also publish a newer plan before the acquisition
                # node's next active-set callback runs.  Keep the newest
                # pending hold until _process_final_recovery authenticates
                # its target and measured seed.
                self._final_recovery_nonhold_feedback_count += 1
                if self._final_recovery_pending is not None:
                    self._final_recovery_preserved_pending_count += 1
                return
            if not token.startswith('care_obs_v1_'):
                return
            self._final_recovery_hold_feedback_count += 1
            if seq < self._final_recovery_latest_seq:
                return
            self._final_recovery_latest_seq = seq
            self._final_recovery_pending = (seq, token)
            self._final_recovery_active_request = (seq, token)

    def _traced_waypoint(self, ob, q):
        """Record a bounded publication identity for target-aware recovery."""
        msg = super()._traced_waypoint(ob, q)
        target = getattr(self, '_trace_published_target', None)
        if target is not None:
            token = str(target.get('observation_token', ''))
            if token:
                with self._schedule_publish_lock:
                    self._final_recovery_target_history[token] = dict(
                        observation_token=token,
                        region_token=str(target['region_token']),
                        q_vis=np.asarray(target['q_vis'], dtype=np.float64).copy())
                    self._final_recovery_target_history.move_to_end(token)
                    while (len(self._final_recovery_target_history) >
                           self._final_recovery_target_history_capacity):
                        self._final_recovery_target_history.popitem(last=False)
        return msg

    def _final_recovery_alternative_seed(self, measured):
        """Return one small, deterministic seed perturbation for scalar recovery.

        The first recovery solve starts exactly at the measured state.  If the
        frozen learned field returns the same q_vis, a second solve from a
        nearby alternating joint-space seed can take a different local branch.
        This is deliberately bounded and is still only a steering proposal;
        final GCDF and exact VBC decide whether it may execute.
        """
        q = np.asarray(measured, dtype=np.float64).reshape(7).copy()
        signs = np.asarray([1., -1., 1., -1., 1., -1., 1.], dtype=np.float64)
        delta = float(self._final_recovery_alternative_seed_rad) * signs
        candidate = q + delta
        q_min = np.asarray(getattr(self, 'q_min_list', []), dtype=np.float64).reshape(-1)
        q_max = np.asarray(getattr(self, 'q_max_list', []), dtype=np.float64).reshape(-1)
        if q_min.shape == (7,) and q_max.shape == (7,):
            candidate = np.clip(candidate, q_min, q_max)
        # If the first signed direction is blocked by joint limits, try its
        # opposite once.  The result remains deterministic and within limits.
        if float(np.max(np.abs(candidate - q))) <= 1e-9:
            candidate = q - delta
            if q_min.shape == (7,) and q_max.shape == (7,):
                candidate = np.clip(candidate, q_min, q_max)
        return candidate

    def _process_final_recovery(self):
        if not (self._final_recovery_enabled or
                getattr(self, '_safe_stall_recovery_enabled', False)):
            return
        with self._final_recovery_lock:
            request = self._final_recovery_pending
            self._final_recovery_pending = None
        if request is None:
            return
        self._final_recovery_process_count += 1
        with self._schedule_publish_lock:
            published = getattr(self, '_trace_published_target', None)
            published = None if published is None else dict(published)
            published_matches = bool(
                published is not None and
                published.get('observation_token') == request[1])
            historical = self._final_recovery_target_history.get(request[1])
            historical = None if historical is None else dict(historical)
        with self._obligation_lock:
            # A blocker may temporarily become the published target after the
            # planner emitted a final-VBC hold for the previous target. The
            # hold is still actionable when that exact observation token is a
            # live obligation; requiring it to remain the first published
            # target turns ordinary blocker preemption into a false stale
            # discard. Exact token matching prevents an old q_vis or changed
            # geometry from borrowing the recovery budget.
            live = next((ob for ob in self._obligations
                         if observation_token(ob) == request[1]), None)
            if live is None and published_matches:
                live = next((ob for ob in self._obligations
                             if observation_region_token(ob) == published['region_token']), None)
            if live is None and historical is not None:
                live = next((ob for ob in self._obligations
                             if observation_region_token(ob) == historical['region_token']), None)
            if live is None:
                self._final_recovery_last_reason = 'stale_target'
                return
            old = dict(live)
        if not published_matches:
            self._final_recovery_live_obligation_fallback_count += 1
            self._final_recovery_last_reason = 'live_obligation_target_fallback'
        previous_q_vis = (
            published['q_vis'] if published_matches else
            historical['q_vis'] if historical is not None else old.get('q_vis', []))
        previous_q_vis = np.asarray(previous_q_vis, dtype=np.float64).reshape(-1)
        if previous_q_vis.shape != (7,) or not np.all(np.isfinite(previous_q_vis)):
            self._final_recovery_last_reason = 'stale_target'
            return
        measured = self._latest_measured_q
        if measured is None:
            return
        measured = np.asarray(measured, dtype=np.float64).copy()
        with self._lock:
            trajectory, received, _ = self._preferred_trajectory_locked()
        if trajectory is None or received is None:
            return
        if not self._final_recovery_budget.reserve(old['points'], measured):
            self._final_recovery_last_reason = 'attempt_budget_hold'
            return
        start = time.perf_counter()
        alternative_seed_attempted = False
        try:
            with self._progressive_shared_lock:
                def generate(seed):
                    self._seed_override = np.asarray(seed, dtype=np.float64).copy()
                    try:
                        return self._generate_active_set_waypoint(
                            old['points'], trajectory,
                            float(old['discovered_sweep_time_s']), received)
                    finally:
                        self._seed_override = None

                result = generate(measured)
            if not certified_distinct_q(result, previous_q_vis):
                alternative = self._final_recovery_alternative_seed(measured)
                if float(np.max(np.abs(alternative - measured))) > 1e-9:
                    alternative_seed_attempted = True
                    self._final_recovery_alternative_seed_attempt_count += 1
                    with self._progressive_shared_lock:
                        result = generate(alternative)
                    if certified_distinct_q(result, previous_q_vis):
                        self._final_recovery_alternative_seed_success_count += 1
                if not certified_distinct_q(result, previous_q_vis):
                    self._final_recovery_last_reason = 'no_certified_distinct_target_hold'
                    return
            # Validate the entire update before touching a live obligation.
            updates = {}
            for key in ['final_f_min', 'shared_solution_mode']:
                if key not in result:
                    raise ValueError('recovery result missing ' + key)
                updates[key] = result[key]
            # Per-sensor provenance is optional in the production scalar path.
            # Preserve it when present, while allowing the same bounded
            # recovery policy to refresh a scalar q_vis result.
            sensor_keys = [
                'per_sensor_hybrid_used', 'per_sensor_selected_sensor_id',
                'per_sensor_selected_sensor_frame', 'per_sensor_selected_rank']
            if 'per_sensor_hybrid' in result:
                for key in sensor_keys:
                    if key not in result:
                        raise ValueError('recovery result missing ' + key)
                    updates[key] = result[key]
                updates['per_sensor_hybrid'] = result['per_sensor_hybrid']
            updates['q_vis'] = np.asarray(result['q_vis'], dtype=np.float64).reshape(7).copy()
            updates['q_vis_joint_mask'] = np.asarray(
                result.get('q_vis_joint_mask', [1.0] * 7),
                dtype=np.float64).reshape(7).copy()
            updates['q_zero'] = np.asarray(result['q_zero'], dtype=np.float64).reshape(7).copy()
            if not np.all(np.isfinite(updates['q_zero'])):
                raise ValueError('nonfinite regenerated q_zero')
            updates['final_recovery_individual_region'] = region_key(old['points'])
            updates['q_vis_generation_ms'] = 1000.*(time.perf_counter()-start)
            with self._schedule_publish_lock, self._obligation_lock, self._final_recovery_lock:
                current = getattr(self, '_trace_published_target', None)
                live = next((ob for ob in self._obligations if ob['id'] == old['id']), None)
                current_matches = bool(
                    current is not None and
                    current.get('observation_token') == request[1])
                live_matches = bool(
                    live is not None and
                    (observation_token(live) == request[1] or
                     (published_matches and current_matches and
                      observation_region_token(live) == published['region_token']) or
                     (historical is not None and
                      observation_region_token(live) == historical['region_token'])))
                if (live is None or not live_matches or
                        (published_matches and not current_matches) or
                        self._final_recovery_latest_seq != request[0] or
                        self._final_recovery_active_request != request or
                        self._latest_measured_q is None or
                        not np.all(np.isfinite(self._latest_measured_q)) or
                        np.max(np.abs(np.asarray(self._latest_measured_q)-measured)) > .01):
                    self._final_recovery_last_reason = 'stale_generation_discarded'
                    return
                live.update(updates)
            self._final_recovery_last_reason = 'certified_steering_refresh_pending_hard_audit'
            with self._final_recovery_lock:
                if self._final_recovery_active_request == request:
                    self._final_recovery_active_request = None
                self._final_recovery_process_success_count += 1
            self._publish_schedule()
        except Exception as exc:
            self._final_recovery_last_reason = 'generation_error_hold'
            rospy.logwarn('bounded final VBC recovery failed; hold retained: %s', exc)
        finally:
            self._trace_observation('final_vbc_recovery', observation_token=request[1],
                reason=self._final_recovery_last_reason, measured_seed=measured.tolist(),
                alternative_seed_attempted=bool(alternative_seed_attempted),
                compute_ms=1000.*(time.perf_counter()-start))

    def _process_new_active_set(self) -> None:
        if self._c49_ready:
            replacement = getattr(self, '_candidate_replacement', None)
            if replacement is not None:
                replacement.process()
            if hasattr(self, '_dependency_lock'):
                self._process_observation_dependency()
            self._process_final_recovery()
            self._process_gcdf_recovery_event()
        if not self._c49_ready:
            self._last_process_reason = "not_ready"
            return
        if self._coherent_bundle_enabled:
            with self._coherent_bundle_lock:
                serial = self._coherent_bundle_seq
                raw = self._coherent_bundle_points.copy()
                sweep = self._coherent_bundle_sweep_s
                trace_identity = getattr(self, '_trace_coherent_bundle_identity', None)
                urgent_recovery = bool(self._coherent_bundle_urgent_recovery)
                bundle_source = self._coherent_bundle_source
                bundle_source_seq = self._coherent_bundle_source_seq
            with self._obligation_lock:
                processed = self._processed_active_set_serial
        else:
            trace_identity = None
            urgent_recovery = False
            bundle_source = "legacy"
            with self._obligation_lock:
                serial = self._raw_active_set_serial
                processed = self._processed_active_set_serial
                raw = self._raw_active_set.copy()
            with self._lock:
                sweep = self._sweep_time_s
            bundle_source_seq = serial

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
        trace_routes = []
        trace_stack_before = list(self._repair_stack)
        active_ids: List[int] = []
        new_ids: List[int] = []
        all_regions_handled = True
        for region in regions:
            routed = self._absorb_refined_partition_region(
                region, trajectory, float(sweep),
                trajectory_received, trajectory_source)
            if routed is not None:
                trace_routes.append(dict(points=region['points'].tolist(), obligation_id=int(routed), route='refined_partition'))
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
                    trace_routes.append(dict(points=region['points'].tolist(), obligation_id=oid, route='matched_existing'))
                    active_ids.append(oid)
                    self._schedule_matched_obligations += 1
                    continue
                if len(self._obligations) >= self.max_obligations:
                    trace_routes.append(dict(points=region['points'].tolist(), obligation_id=None, route='capacity_exhausted'))
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
                trace_routes.append(dict(points=region['points'].tolist(), obligation_id=None, route='generation_failed'))
                rospy.logerr(
                    "[vbc_blocker_stack] obligation generation failed; retrying: %s", exc)
                continue
            with self._obligation_lock:
                matched = self._match_existing(region)
                if matched is None and len(self._obligations) < self.max_obligations:
                    self._obligations.append(new_ob)
                    self._schedule_new_obligations += 1
                    oid = int(new_ob["id"])
                    trace_routes.append(dict(points=region['points'].tolist(), obligation_id=oid, route='generated'))
                    active_ids.append(oid)
                    new_ids.append(oid)
                elif matched is not None:
                    active_ids.append(int(matched["id"]))
                    trace_routes.append(dict(points=region['points'].tolist(), obligation_id=int(matched['id']), route='matched_after_generation'))
                else:
                    trace_routes.append(dict(points=region['points'].tolist(), obligation_id=None, route='capacity_after_generation'))

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
        self._consider_active_layer(
            active_ids, new_ids, float(sweep),
            urgent_gcdf_recovery=urgent_recovery)
        self._process_pending_refinement(
            trajectory, trajectory_received,
            trajectory_source, float(sweep))
        self._publish_schedule()
        self._publish_blocker_stack_summary()

        # Captured bundle identity survives concurrent arrival of a newer
        # bundle while projection runs. Do not read the latest ID here.
        try:
            with self._obligation_lock:
                identities = {int(ob['id']): observation_region_token(ob) for ob in self._obligations}
                stored_points = {int(ob['id']): np.asarray(ob['points']).tolist()
                                 for ob in self._obligations if int(ob['id']) in active_ids}
            with self._schedule_publish_lock:
                published = getattr(self, '_trace_published_target', None)
                published_token = published['observation_token'] if published else None
            self._trace_observation('blocker_route', vbc_identity=trace_identity,
                bundle_seq=serial, bundle_source=bundle_source,
                bundle_source_seq=bundle_source_seq,
                urgent_recovery=urgent_recovery,
                points=raw.tolist(), routes=trace_routes,
                active_ids=active_ids, new_ids=new_ids, all_regions_handled=all_regions_handled,
                stack_before=trace_stack_before, stack_after=list(self._repair_stack),
                path_associated_ids=list(getattr(self, '_path_associated_ids', [])),
                path_co_plan_ids=list(getattr(self, '_path_co_plan_ids', [])),
                path_association_reason=getattr(
                    self, '_path_association_reason', 'legacy_fixture'),
                region_tokens=identities, stored_obligation_points=stored_points,
                published_observation_token=published_token)
        except Exception as exc:
            rospy.logwarn_throttle(5.0, '[observation_trace] blocker routing trace failed: %s', exc)

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
            path_co_plan = [
                oid for oid in self._path_co_plan_ids if oid in set(existing)]
            qvis_times = [
                float(ob.get("q_vis_generation_ms", math.nan))
                for ob in self._obligations
                if math.isfinite(float(ob.get("q_vis_generation_ms", math.nan)))
            ]
        msg = String()
        msg.data = (
            "policy=blocker_aware_recursive"
            f" dependency_reason={getattr(self, '_dependency_reason', 'disabled')}"
            f" dependency_pushes={getattr(self, '_dependency_pushes', 0)}"
            f" dependency_resumes={getattr(self, '_dependency_resumes', 0)}"
            f" final_vbc_recovery_enabled={int(self._final_recovery_enabled)}"
            f" safe_stall_recovery_enabled={int(getattr(self, '_safe_stall_recovery_enabled', False))}"
            f" final_vbc_recovery_reason={self._final_recovery_last_reason}"
            f" final_vbc_recovery_latest_hold_seq={self._final_recovery_latest_seq}"
            f" final_vbc_recovery_latest_feedback_seq={self._final_recovery_latest_feedback_seq}"
            f" final_vbc_recovery_feedback_count={self._final_recovery_feedback_count}"
            f" final_vbc_recovery_hold_feedback_count={self._final_recovery_hold_feedback_count}"
            f" final_vbc_recovery_nonhold_feedback_count={self._final_recovery_nonhold_feedback_count}"
            f" final_vbc_recovery_preserved_pending_count={self._final_recovery_preserved_pending_count}"
            f" final_vbc_recovery_process_count={self._final_recovery_process_count}"
            f" final_vbc_recovery_process_success_count={self._final_recovery_process_success_count}"
            f" final_vbc_recovery_live_obligation_fallback_count="
            f"{self._final_recovery_live_obligation_fallback_count}"
            f" final_vbc_recovery_alternative_seed_attempt_count="
            f"{self._final_recovery_alternative_seed_attempt_count}"
            f" final_vbc_recovery_alternative_seed_success_count="
            f"{self._final_recovery_alternative_seed_success_count}"
            f" final_vbc_recovery_alternative_seed_rad="
            f"{self._final_recovery_alternative_seed_rad:.6f}"
            f" current_target_id={(stack[-1] if stack else -1)}"
            f" stack={':'.join(str(v) for v in stack) or 'none'}"
            f" pending_ids={':'.join(str(v) for v in existing) or 'none'}"
            f" earliest_layer_ids={':'.join(str(v) for v in self._last_active_layer_ids) or 'none'}"
            f" earliest_layer_sweep_s={self._last_active_layer_sweep_s:.6f}"
            f" pending_blocker_id={self._pending_blocker_id if self._pending_blocker_id is not None else -1}"
            f" pending_blocker_count={self._pending_blocker_count}"
            f" path_co_plan_ids={':'.join(str(v) for v in path_co_plan) or 'none'}"
            f" path_associated_ids={':'.join(str(v) for v in self._path_associated_ids) or 'none'}"
            f" path_association_reason={self._path_association_reason}"
            f" path_associated_count={self._path_associated_count}"
            f" path_queued_count={self._path_queued_count}"
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
            f" coherent_bundle_source={self._coherent_bundle_source}"
            f" coherent_bundle_source_seq={self._coherent_bundle_source_seq}"
            f" coherent_bundle_urgent_recovery={int(self._coherent_bundle_urgent_recovery)}"
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
            f" progressive_shared_path_association_only={int(self.progressive_shared_path_association_only)}"
            f" active_qvis_target_lock_enabled={int(self._active_qvis_target_lock_enabled)}"
            f" safety_union_merge_count={self._safety_union_merge_count}"
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
