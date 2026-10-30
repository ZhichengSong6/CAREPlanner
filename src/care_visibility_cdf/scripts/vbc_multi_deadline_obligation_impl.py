#!/usr/bin/env python3
"""C4.6 accumulated multi-deadline visibility obligations.

This is a steering/optimization-side module. Exact VBC remains the only safety
truth and the only authority allowed to make a candidate committable.

Why this exists
---------------
C4.5 selected one spatial region at a time across *rejected* candidates.  A
candidate that repaired R1 but exposed R2 was rejected, so the robot never moved
to the R1-repaired state.  Switching the next optimization to R2 therefore lost
the R1 repair and produced R1<->R2 ping-pong.

C4.6 instead accumulates obligations across rejected candidates:

    O_r = (persistent spatial region, q_vis^(r), absolute VBC deadline_r)

A newly observed region is appended; an old region is not removed merely because
it disappears from one rejected hypothetical candidate. Legacy obligations
clear after a fresh predicted SAFE verdict. Optional stopping-prefix requests
remain until later real observations resolve them or their path is retired.
The downstream MPC receives all obligations at once and adds one
quadratic waypoint term per deadline.

Reachability-aware seed
-----------------------
The learned projection for a new obligation is seeded from the current measured
joint configuration, not q_nom(deadline).  Abandoning nominal tracking removes a
soft objective, but it cannot remove hard velocity/acceleration limits.  Seeding
from measured q therefore asks the learned field for a nearby visibility solution
in the configuration region the robot actually occupies.

The exact deadline is intentionally *not* relaxed when it looks dynamically hard:
VBC decides whether a new candidate delayed/avoided the sweep enough to become
safe.  This module logs a simple rest-to-q_vis lower bound only as a diagnostic.
"""

from __future__ import annotations

import json
import math
import os
import re
import copy
import threading
import time
import uuid
from observation_identity import observation_token, observation_region_token
from task_recovery_events import (RECOVERY_TOPIC, CONTROL_TOPIC, RecoverySnapshotSource,
    TaskEventReceiver, validate_regime_control, canonical)
from recovery_policy import build_policy
from stopping_observation_demands import ObservationDemandQueue, validate_report, exact_region_owner
from stopping_trajectory_prefix import trajectory_sha256, validate_trajectory_report, apply_absolute_observation_deadline
from stopping_observation_context import (ObservationContext, ContextObservationDemandQueue,
    context_key, context_path, validate_context_report)
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray, MultiArrayDimension, String


def q_vis_joint_mask(source) -> np.ndarray:
    """Return a validated objective mask; legacy/scalar targets constrain all joints."""
    try:
        mask = np.asarray(
            source.get("q_vis_joint_mask", [1.0] * 7),
            dtype=np.float64).reshape(7)
    except (AttributeError, TypeError, ValueError):
        return np.ones(7, dtype=np.float64)
    if (not np.all(np.isfinite(mask)) or
            np.any(mask < 0.0) or np.any(mask > 1.0)):
        return np.ones(7, dtype=np.float64)
    return mask

from evaluate_direct_vs_projection_ascent import DEFAULT_JOINT_NAMES
from vbc_deadline_waypoint_node import _fmt, _vector_msg
from vbc_deadline_waypoint_rolling_impl import RollingVbcDeadlineWaypointNode


_TOKEN = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


class AccumulatedMultiDeadlineWaypointNode(RollingVbcDeadlineWaypointNode):
    """Accumulate spatial visibility obligations until exact VBC is globally safe."""

    def __init__(self) -> None:
        self._stopping_prefix_lock = threading.RLock()
        self._stopping_prefix_queue = ObservationDemandQueue()
        self._stopping_prefix_pending = None
        self._stopping_prefix_path = None
        self._stopping_prefix_binding = None
        from task_observation_binding import TaskContextReceiver
        self._require_task_binding = bool(rospy.get_param("~require_task_binding", False))
        self._stopping_task_receiver = TaskContextReceiver()
        self._stopping_context = (self._stopping_task_receiver.context if getattr(self, "_require_task_binding", False)
                                  else ObservationContext())
        self._stopping_task_candidate = None
        self._task_recovery_source = RecoverySnapshotSource()
        self._task_policy_control = None
        self._task_policy_events = TaskEventReceiver()
        self._task_selected_obligations = None
        self._task_recovery_pub = (rospy.Publisher(RECOVERY_TOPIC, String, queue_size=8, latch=True)
                                   if self._require_task_binding else None)
        self._stopping_context_queue = ContextObservationDemandQueue()
        self._stopping_context_worker = threading.Lock()
        self._stopping_context_key = None
        if not hasattr(self, '_schedule_publish_lock'):
            self._schedule_publish_lock = threading.RLock()
        if self._require_task_binding:
            self._task_policy_sub = rospy.Subscriber(CONTROL_TOPIC, String,
                self._task_policy_control_cb, queue_size=8)
        # These fields must exist before the parent creates ROS subscribers.
        self._obligation_lock = threading.Lock()
        self._obligations: List[Dict[str, object]] = []
        self._next_obligation_id = 1
        self._raw_active_set = np.zeros((0, 3), dtype=np.float64)
        self._raw_active_set_serial = 0
        self._processed_active_set_serial = 0
        self._latest_measured_q: Optional[np.ndarray] = None
        self._seed_override: Optional[np.ndarray] = None
        self._schedule_clear_count = 0
        self._schedule_generation_failures = 0
        self._schedule_new_obligations = 0
        self._schedule_matched_obligations = 0
        self._last_safe_verdict_time = math.nan

        # Diagnostic-only q_vis override.  Disabled by default and intentionally
        # driven by environment variables so existing launch commands/runners do
        # not change semantics.  The Case-026 A/B wrapper enables it explicitly.
        self._diag_qvis_override_enabled = str(
            os.environ.get("CARE_DIAG_QVIS_OVERRIDE_ENABLED", "false")
        ).strip().lower() in ("1", "true", "yes", "on")
        self._diag_qvis_override_obligation_id = int(
            os.environ.get("CARE_DIAG_QVIS_OVERRIDE_OBLIGATION_ID", "7"))
        self._diag_qvis_override_target = np.asarray([
            float(x) for x in os.environ.get(
                "CARE_DIAG_QVIS_OVERRIDE_TARGET",
                "0.1,0.05,0.15").split(",")
        ], dtype=np.float64).reshape(3)
        self._diag_qvis_override_target_tol_m = float(
            os.environ.get("CARE_DIAG_QVIS_OVERRIDE_TARGET_TOL_M", "0.03"))
        self._diag_qvis_override_q = np.asarray([
            float(x) for x in os.environ.get(
                "CARE_DIAG_QVIS_OVERRIDE_Q",
                "-0.8144537806510925,0.5605988502502441,1.0171856880187988,"
                "-2.180335760116577,0.12358028441667557,-1.7687498331069946,"
                "1.1331826448440552").split(",")
        ], dtype=np.float64).reshape(7)
        self._diag_qvis_override_label = str(os.environ.get(
            "CARE_DIAG_QVIS_OVERRIDE_LABEL", "case026_robust_s7"))
        self._diag_qvis_override_applied_count = 0

        super().__init__()

        self.region_max_diameter_m = float(rospy.get_param(
            "~obligation_region_max_diameter_m", 0.12))
        self.region_match_distance_m = float(rospy.get_param(
            "~obligation_region_match_distance_m", 0.12))
        self.max_obligations = int(rospy.get_param("~max_obligations", 8))
        self.schedule_topic = str(rospy.get_param(
            "~schedule_topic",
            "/care_planner/active_sensing/visibility_waypoint_schedule"))
        self.schedule_summary_topic = str(rospy.get_param(
            "~schedule_summary_topic",
            "/care_planner/active_sensing/visibility_waypoint_schedule_summary"))
        self.candidate_vbc_summary_topic = str(rospy.get_param(
            "~candidate_vbc_summary_topic",
            "/care_planner/candidate_vbc/summary"))
        self.joint_state_topic = str(rospy.get_param(
            "~joint_state_topic", "/care_arm/joint_states"))
        self.acceleration_limits = np.asarray(rospy.get_param(
            "~joint_acceleration_limits", [3.0, 3.0, 4.0, 4.0, 6.0, 6.0, 6.0]),
            dtype=np.float64)
        self.velocity_limits = np.asarray(rospy.get_param(
            "~joint_velocity_limits", [2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5]),
            dtype=np.float64)

        if self.region_max_diameter_m <= 0.0 or self.region_match_distance_m <= 0.0:
            raise ValueError("obligation region distances must be positive")
        if self.max_obligations < 1:
            raise ValueError("~max_obligations must be >= 1")
        if self.acceleration_limits.shape != (7,) or np.any(self.acceleration_limits <= 0.0):
            raise ValueError("~joint_acceleration_limits must be seven positive values")
        if self.velocity_limits.shape != (7,) or np.any(self.velocity_limits <= 0.0):
            raise ValueError("~joint_velocity_limits must be seven positive values")

        self.schedule_pub = rospy.Publisher(
            self.schedule_topic, Float64MultiArray, queue_size=1, latch=True)
        self.schedule_summary_pub = rospy.Publisher(
            self.schedule_summary_topic, String, queue_size=1, latch=True)
        self.joint_state_sub_c46 = rospy.Subscriber(
            self.joint_state_topic, JointState,
            self._c46_joint_state_callback, queue_size=1)
        self.candidate_vbc_sub_c46 = rospy.Subscriber(
            self.candidate_vbc_summary_topic, String,
            self._candidate_vbc_summary_callback, queue_size=20)
        # Requests steer existing waypoint generation. They confer no execution
        # authority; the candidate and complete stopping gates still apply.
        self.stopping_prefix_sub = None
        self._stopping_context_enabled = bool(rospy.get_param('~enable_observation_context', False))
        if bool(rospy.get_param('~enable_stopping_observation_demands', False)):
            self.stopping_prefix_sub = rospy.Subscriber(
                '~stopping_observation_demands', String,
                self._stopping_prefix_callback, queue_size=1)
            if self._stopping_context_enabled:
                self.stopping_context_sub = rospy.Subscriber(
                    str(rospy.get_param('~observation_context_topic',
                        '/care_planner/active_sensing/observation_context')),
                    String, self._stopping_context_callback, queue_size=32)
                if getattr(self, "_require_task_binding", False):
                    from egocentric_arm_planner.msg import CandidateTransfer
                    rospy.Subscriber(str(rospy.get_param("~task_candidate_topic",
                        "/care_planner/local_planner/candidate_trajectory/planned")), CandidateTransfer,
                        self._stopping_task_candidate_callback, queue_size=1)

        rospy.logwarn(
            "[vbc_multi_deadline] C4.6 ENABLED: accumulated obligations; "
            "region_diam=%.3fm match=%.3fm max=%d schedule=%s seed=measured_q",
            self.region_max_diameter_m, self.region_match_distance_m,
            self.max_obligations, self.schedule_topic)

    # ------------------------------------------------------------------
    # Input canonicalization / clustering
    # ------------------------------------------------------------------
    def _c46_joint_state_callback(self, msg: JointState) -> None:
        if msg is None:
            return
        index = {name: i for i, name in enumerate(msg.name)}
        if any(name not in index for name in DEFAULT_JOINT_NAMES):
            return
        try:
            q = np.asarray(
                [msg.position[index[name]] for name in DEFAULT_JOINT_NAMES],
                dtype=np.float64)
        except Exception:
            return
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            return
        self._latest_measured_q = q
        self._trace_measured = (q.copy(), msg.header.stamp.to_sec())
        replacement = getattr(self, '_candidate_replacement', None)
        if replacement is not None:
            replacement.seed_gate.observe_measured(q, msg.header.stamp.to_sec())

    def _trace_observation(self, stage, **fields):
        from observation_research_trace import record_observation
        record_observation(self, stage, fields)

    def _active_set_callback(self, msg: Float64MultiArray) -> None:
        if msg is None:
            return
        values = np.asarray(list(msg.data), dtype=np.float64)
        if values.size == 0:
            points = np.zeros((0, 3), dtype=np.float64)
        elif values.size % 3 != 0:
            rospy.logwarn_throttle(
                1.0, "[vbc_multi_deadline] malformed active-set length=%d",
                int(values.size))
            return
        else:
            points = values.reshape(-1, 3)
            if not np.all(np.isfinite(points)):
                return

        # Preserve the Rolling base cache for diagnostics/backward-compatible
        # target streams, but do not let it own obligation lifetime.
        RollingVbcDeadlineWaypointNode._active_set_callback(self, msg)

        by_key = {}
        for p in points:
            by_key[self._cell_key(p)] = p.copy()
        keys = tuple(sorted(by_key.keys()))
        canonical = (
            np.asarray([by_key[k] for k in keys], dtype=np.float64)
            if keys else np.zeros((0, 3), dtype=np.float64))
        with self._obligation_lock:
            self._raw_active_set = canonical
            self._raw_active_set_serial += 1

    @staticmethod
    def _diameter(points: np.ndarray) -> float:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] <= 1:
            return 0.0
        diff = pts[:, None, :] - pts[None, :, :]
        return float(np.max(np.linalg.norm(diff, axis=-1)))

    def _cluster_regions(self, points: np.ndarray) -> List[Dict[str, object]]:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        clusters = [[i] for i in range(pts.shape[0])]
        while True:
            best = None
            best_d = math.inf
            centroids = [np.mean(pts[idx], axis=0) for idx in clusters]
            for a in range(len(clusters)):
                for b in range(a + 1, len(clusters)):
                    merged = clusters[a] + clusters[b]
                    if self._diameter(pts[merged]) > self.region_max_diameter_m + 1e-9:
                        continue
                    d = float(np.linalg.norm(centroids[a] - centroids[b]))
                    if d < best_d:
                        best_d = d
                        best = (a, b)
            if best is None:
                break
            a, b = best
            clusters[a].extend(clusters[b])
            del clusters[b]

        out = []
        for idx in clusters:
            rp = pts[idx].copy()
            keys = tuple(sorted({self._cell_key(p) for p in rp}))
            out.append({
                "points": rp,
                "keys": keys,
                "centroid": np.mean(rp, axis=0),
            })
        return out

    @staticmethod
    def _jaccard(a, b) -> float:
        sa, sb = set(a or ()), set(b or ())
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        return float(len(sa & sb)) / float(len(sa | sb))

    def _match_candidate_compatible(
            self, ob: Dict[str, object], region: Dict[str, object]) -> bool:
        """Semantic hook layered on top of spatial obligation matching.

        The accumulated C4.6 mode keeps the historical spatial-only behavior.
        Blocker-aware acquisition overrides this hook so a live region may reuse
        an existing obligation only when that obligation's stored q_vis is still
        a valid learned visibility solution for the new geometry.
        """
        return True

    def _match_existing(self, region: Dict[str, object]):
        best = None
        best_score = None
        for ob in self._obligations:
            j = self._jaccard(ob["keys"], region["keys"])
            d = float(np.linalg.norm(ob["centroid"] - region["centroid"]))
            if j > 0.0:
                score = (2, j, -d)
            elif d <= self.region_match_distance_m:
                score = (1, -d, 0.0)
            else:
                continue
            if not self._match_candidate_compatible(ob, region):
                continue
            if best_score is None or score > best_score:
                best_score = score
                best = ob
        return best

    # ------------------------------------------------------------------
    # Measured-state learned projection
    # ------------------------------------------------------------------
    def _sample_trajectory(self, trajectory, t):
        if self._seed_override is not None:
            return np.asarray(self._seed_override, dtype=np.float64).copy()
        return super()._sample_trajectory(trajectory, t)

    def _rest_min_hit_time(self, q0: np.ndarray, q1: np.ndarray) -> float:
        """Lower bound from rest with |a|/|v| bounds; no terminal-stop requirement."""
        dq = np.abs(np.asarray(q1) - np.asarray(q0))
        times = []
        for d, a, v in zip(dq, self.acceleration_limits, self.velocity_limits):
            d_switch = (v * v) / (2.0 * a)
            if d <= d_switch:
                t = math.sqrt(max(0.0, 2.0 * d / a))
            else:
                t = v / a + (d - d_switch) / v
            times.append(t)
        return float(max(times) if times else 0.0)

    def _generate_new_obligation(
            self, region, trajectory, sweep_time_s, trajectory_received,
            trajectory_source):
        measured = (
            None if self._latest_measured_q is None
            else self._latest_measured_q.copy())
        if measured is None:
            raise RuntimeError("measured_joint_state_not_ready")

        self._seed_override = measured
        generation_id = uuid.uuid4().hex
        t_generate_start = time.perf_counter()
        self._trace_observation("generation_started", generation_event_id=generation_id,
            obligation_id=int(self._next_obligation_id), measured_seed_q=measured.tolist(),
            points=np.asarray(region["points"]).tolist(),
            source_trajectory_stamp_ns=int(trajectory.header.stamp.to_nsec()),
            trajectory_source=str(trajectory_source))
        try:
            result = RollingVbcDeadlineWaypointNode._generate_active_set_waypoint(
                self, region["points"], trajectory, sweep_time_s,
                trajectory_received)
            result = apply_absolute_observation_deadline(result, region, trajectory_source)
        except Exception as exc:
            self._trace_observation("generation_failed", generation_event_id=generation_id,
                error=str(exc), compute_ms=1000.0*(time.perf_counter()-t_generate_start))
            raise
        finally:
            q_vis_generation_ms = 1000.0 * (
                time.perf_counter() - t_generate_start)
            self._seed_override = None

        original_q_vis = np.asarray(result["q_vis"], dtype=np.float64).copy()
        q_vis = original_q_vis.copy()

        # Narrow diagnostic hook for the Case-026 sensor-branch A/B.  Require
        # both the expected obligation id and a centroid match so this cannot
        # silently alter an unrelated obligation if numbering changes.
        override_applied = False
        override_target_error_m = math.nan
        if self._diag_qvis_override_enabled:
            current_oid = int(self._next_obligation_id)
            region_centroid = np.asarray(
                region["centroid"], dtype=np.float64).reshape(3)
            override_target_error_m = float(np.linalg.norm(
                region_centroid - self._diag_qvis_override_target))
            if (
                current_oid == self._diag_qvis_override_obligation_id
                and override_target_error_m
                    <= self._diag_qvis_override_target_tol_m
            ):
                q_vis = self._diag_qvis_override_q.copy()
                result["q_vis"] = q_vis.tolist()
                result["diagnostic_qvis_override_applied"] = True
                result["diagnostic_qvis_override_label"] = (
                    self._diag_qvis_override_label)
                result["diagnostic_qvis_override_original_q_vis"] = (
                    original_q_vis.tolist())
                result["diagnostic_qvis_override_q_vis"] = q_vis.tolist()
                result["diagnostic_qvis_override_target_error_m"] = (
                    override_target_error_m)
                result["shared_solution_mode"] = (
                    str(result.get("shared_solution_mode", "unknown"))
                    + "+diagnostic_qvis_override")
                self._diag_qvis_override_applied_count += 1
                override_applied = True
                rospy.logwarn(
                    "[vbc_multi_deadline][DIAG QVIS OVERRIDE] "
                    "APPLIED obligation=%d label=%s target_err=%.4fm "
                    "original=%s override=%s",
                    current_oid, self._diag_qvis_override_label,
                    override_target_error_m,
                    _fmt(original_q_vis, 5), _fmt(q_vis, 5))
            elif current_oid == self._diag_qvis_override_obligation_id:
                rospy.logwarn(
                    "[vbc_multi_deadline][DIAG QVIS OVERRIDE] "
                    "NOT applied obligation=%d target_err=%.4fm > tol=%.4fm",
                    current_oid, override_target_error_m,
                    self._diag_qvis_override_target_tol_m)

        deadline_abs = float(result["deadline_absolute_ros_s"])
        now_s = rospy.Time.now().to_sec()
        min_hit = self._rest_min_hit_time(measured, q_vis)
        deadline_remaining = deadline_abs - now_s
        source_points = np.asarray(
            region["points"], dtype=np.float64).copy()
        source_centroid = np.asarray(
            region["centroid"], dtype=np.float64).copy()
        source_keys = tuple(region["keys"])
        source_xyz_min = (
            np.min(source_points, axis=0)
            if source_points.shape[0]
            else np.asarray([math.nan, math.nan, math.nan], dtype=np.float64))
        source_xyz_max = (
            np.max(source_points, axis=0)
            if source_points.shape[0]
            else np.asarray([math.nan, math.nan, math.nan], dtype=np.float64))

        refinement_depth = int(region.get("refinement_depth", 0))
        parent_obligation_id = int(region.get("parent_obligation_id", -1))
        root_obligation_id = int(region.get(
            "root_obligation_id",
            parent_obligation_id if parent_obligation_id >= 0 else -1))
        refinement_reason = str(region.get(
            "refinement_reason", "coarse_discovery"))
        refinement_family_id = int(region.get(
            "refinement_family_id", -1))
        refinement_partition_anchor = np.asarray(
            region.get("refinement_partition_anchor", source_centroid),
            dtype=np.float64).reshape(3)

        ob = {
            "generation_event_id": generation_id,
            "id": int(self._next_obligation_id),
            "points": source_points.copy(),
            "keys": source_keys,
            "centroid": source_centroid.copy(),
            # Diagnostic-only immutable snapshot of the spatial region used to
            # generate q_vis. Later blocker matching is allowed to refresh the
            # live region geometry, but this snapshot lets us detect whether
            # q_vis has become stale relative to that refreshed geometry.
            "q_vis_source_points": source_points.copy(),
            "q_vis_source_keys": source_keys,
            "q_vis_source_centroid": source_centroid.copy(),
            "q_vis_source_xyz_min": source_xyz_min.copy(),
            "q_vis_source_xyz_max": source_xyz_max.copy(),
            "geometry_match_update_count": 0,
            "geometry_match_change_count": 0,
            "geometry_changed_since_qvis": False,
            "last_match_centroid_shift_m": 0.0,
            "max_centroid_shift_from_qvis_m": 0.0,
            "last_geometry_match_source": "discovery",
            "refinement_depth": refinement_depth,
            "parent_obligation_id": parent_obligation_id,
            "root_obligation_id": root_obligation_id,
            "refinement_reason": refinement_reason,
            "refinement_family_id": refinement_family_id,
            "refinement_partition_anchor":
                refinement_partition_anchor.copy(),
            "q_vis": q_vis,
            "q_vis_joint_mask": (
                np.ones(7, dtype=np.float64) if override_applied else
                q_vis_joint_mask(result)),
            "diagnostic_qvis_override_applied": bool(override_applied),
            "diagnostic_qvis_override_label": (
                self._diag_qvis_override_label if override_applied else "none"),
            "diagnostic_qvis_override_original_q_vis": original_q_vis.copy(),
            "diagnostic_qvis_override_target_error_m": float(
                override_target_error_m),
            "q_zero": np.asarray(result["q_zero"], dtype=np.float64),
            "deadline_abs_s": deadline_abs,
            "discovered_sweep_time_s": float(sweep_time_s),
            "discovered_at_ros_s": now_s,
            "last_seen_ros_s": now_s,
            "trajectory_source": str(trajectory_source),
            "min_hit_time_from_measured_rest_s": min_hit,
            "deadline_remaining_at_discovery_s": deadline_remaining,
            "reachable_before_discovered_deadline_lower_bound": bool(
                min_hit <= max(0.0, deadline_remaining) + 1e-9),
            "final_f_min": float(result["final_f_min"]),
            "shared_solution_mode": str(result["shared_solution_mode"]),
            "per_sensor_hybrid_used": bool(
                result.get("per_sensor_hybrid_used", False)),
            "per_sensor_selected_sensor_id": int(
                result.get("per_sensor_selected_sensor_id", -1)),
            "per_sensor_selected_sensor_frame": str(
                result.get("per_sensor_selected_sensor_frame", "none")),
            "per_sensor_selected_rank": int(
                result.get("per_sensor_selected_rank", -1)),
            "per_sensor_rejected_sensor_ids": list(
                result.get("per_sensor_hybrid", {}).get(
                    "rejected_sensor_ids", [])),
            "per_sensor_branch_compute_ms": float(
                result.get("per_sensor_hybrid", {}).get(
                    "compute_ms", math.nan)),
            "q_vis_generation_ms": float(q_vis_generation_ms),
        }
        self._next_obligation_id += 1
        self._trace_observation("generated", generation_event_id=generation_id,
            observation_token=observation_token(ob), obligation_id=int(ob["id"]),
            q_vis=q_vis.tolist(), points=source_points.tolist(),
            compute_ms=float(q_vis_generation_ms),
            per_sensor_hybrid=result.get("per_sensor_hybrid", {"enabled": False}),
            scalar_final_f=float(result["final_f_min"]),
            union_final_f=float(result["final_f_min"]),
            visibility_model=getattr(self, "visibility_model_metadata", None),
            oracle=result.get("final_oracle_diagnostic"))

        # Save a compact per-obligation trace alongside existing projector traces.
        trace = dict(result)
        trace.update({
            "generation_event_id": generation_id,
            "observation_token": observation_token(ob),
            "c4_6_obligation_id": ob["id"],
            "c4_6_seed_source": "measured_joint_state",
            "c4_6_measured_seed_q": measured.tolist(),
            "c4_6_min_hit_time_from_measured_rest_s": min_hit,
            "c4_6_deadline_remaining_at_discovery_s": deadline_remaining,
            "c4_6_reachable_before_discovered_deadline_lower_bound": ob[
                "reachable_before_discovered_deadline_lower_bound"],
            "c4_6_q_vis_generation_ms": float(q_vis_generation_ms),
            "c4_6_per_sensor_hybrid_used": bool(
                ob["per_sensor_hybrid_used"]),
            "c4_6_per_sensor_selected_sensor_id": int(
                ob["per_sensor_selected_sensor_id"]),
            "c4_6_per_sensor_selected_sensor_frame": str(
                ob["per_sensor_selected_sensor_frame"]),
            "c4_6_per_sensor_selected_rank": int(
                ob["per_sensor_selected_rank"]),
            "c4_6_per_sensor_rejected_sensor_ids": list(
                ob["per_sensor_rejected_sensor_ids"]),
            "c4_6_per_sensor_branch_compute_ms": float(
                ob["per_sensor_branch_compute_ms"]),
            "c4_6_diagnostic_qvis_override_enabled": bool(
                self._diag_qvis_override_enabled),
            "c4_6_diagnostic_qvis_override_applied": bool(override_applied),
            "c4_6_diagnostic_qvis_override_label": (
                self._diag_qvis_override_label if override_applied else "none"),
            "c4_6_diagnostic_qvis_override_original_q_vis":
                original_q_vis.tolist(),
            "c4_6_diagnostic_qvis_override_final_q_vis": q_vis.tolist(),
            "c4_6_diagnostic_qvis_override_target_error_m": float(
                override_target_error_m),
            "c4_6_q_vis_source_centroid": source_centroid.tolist(),
            "c4_6_q_vis_source_xyz_min": source_xyz_min.tolist(),
            "c4_6_q_vis_source_xyz_max": source_xyz_max.tolist(),
            "c4_6_q_vis_source_point_count": int(source_points.shape[0]),
            "c4_6_refinement_depth": refinement_depth,
            "c4_6_parent_obligation_id": parent_obligation_id,
            "c4_6_root_obligation_id": root_obligation_id,
            "c4_6_refinement_reason": refinement_reason,
            "c4_6_refinement_family_id": refinement_family_id,
            "c4_6_refinement_partition_anchor":
                refinement_partition_anchor.tolist(),
        })
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = Path(self.output_root) / (
            f"c46_obligation_{ob['id']:03d}_{stamp}.json")
        path.write_text(json.dumps(trace, indent=2, allow_nan=True))

        rospy.logwarn(
            "[vbc_multi_deadline] ADD obligation=%d points=%d depth=%d parent=%d "
            "deadline_rem=%.3fs Tmin_rest=%.3fs reachable_lb=%d "
            "min_f=%+.4f qvis_ms=%.3f hybrid=%d sensor=S%d rank=%d "
            "q_vis=%s",
            ob["id"], len(ob["points"]), refinement_depth,
            parent_obligation_id, deadline_remaining, min_hit,
            int(ob["reachable_before_discovered_deadline_lower_bound"]),
            ob["final_f_min"], q_vis_generation_ms,
            int(ob["per_sensor_hybrid_used"]),
            int(ob["per_sensor_selected_sensor_id"]),
            int(ob["per_sensor_selected_rank"]), _fmt(q_vis, 4))
        return ob

    # ------------------------------------------------------------------
    # Obligation lifecycle
    # ------------------------------------------------------------------
    def _candidate_vbc_summary_callback(self, msg: String) -> None:
        if msg is None:
            return
        fields = dict(_TOKEN.findall(str(msg.data)))
        if fields.get("trajectory_source") != "predicted":
            return
        if fields.get("has_violation") != "0":
            return
        with self._obligation_lock:
            before = len(self._obligations)
            self._obligations = [ob for ob in self._obligations if ob.get('stopping_prefix_request')]
            had = before-len(self._obligations)
            self._schedule_clear_count += 1
            self._last_safe_verdict_time = rospy.Time.now().to_sec()
        if had:
            rospy.logwarn(
                "[vbc_multi_deadline] EXACT PREDICTED VBC SAFE -> clear %d accumulated obligations",
                had)
        self._publish_schedule()

    def _process_new_active_set(self) -> None:
        with self._obligation_lock:
            serial = self._raw_active_set_serial
            if serial == self._processed_active_set_serial:
                return
            raw = self._raw_active_set.copy()
            self._processed_active_set_serial = serial

        if raw.shape[0] == 0:
            # Empty set from one hypothetical candidate does NOT clear obligations.
            # Only an exact predicted SAFE verdict may clear them.
            return

        with self._lock:
            sweep = self._sweep_time_s
            trajectory, trajectory_received, trajectory_source = (
                self._preferred_trajectory_locked())
        if sweep is None or trajectory is None:
            return

        regions = self._cluster_regions(raw)
        for region in regions:
            with self._obligation_lock:
                matched = self._match_existing(region)
                if matched is not None:
                    matched["last_seen_ros_s"] = rospy.Time.now().to_sec()
                    matched["points"] = np.asarray(region["points"], dtype=np.float64).copy()
                    matched["keys"] = tuple(region["keys"])
                    matched["centroid"] = np.asarray(region["centroid"], dtype=np.float64).copy()
                    self._schedule_matched_obligations += 1
                    continue
                if len(self._obligations) >= self.max_obligations:
                    rospy.logerr_throttle(
                        1.0,
                        "[vbc_multi_deadline] max_obligations=%d reached; refusing new region",
                        self.max_obligations)
                    continue

            try:
                new_ob = self._generate_new_obligation(
                    region, trajectory, float(sweep), trajectory_received,
                    trajectory_source)
            except Exception as exc:
                self._schedule_generation_failures += 1
                rospy.logerr(
                    "[vbc_multi_deadline] obligation generation failed: %s", exc)
                continue

            with self._obligation_lock:
                # Re-check after expensive generation in case another callback
                # added a matching region meanwhile.
                matched = self._match_existing(region)
                if matched is None and len(self._obligations) < self.max_obligations:
                    self._obligations.append(new_ob)
                    self._schedule_new_obligations += 1

        self._publish_schedule()

    def _ordered_obligations(self):
        with self._obligation_lock:
            return sorted(
                [dict(ob) for ob in self._obligations],
                key=lambda ob: (float(ob["deadline_abs_s"]), int(ob["id"])))

    def _publish_schedule(self) -> None:
        # Use the same publication lock as acquisition subclasses and context
        # cancellation; an old snapshot cannot publish after cancellation's clear.
        with self._schedule_publish_lock:
            if getattr(self, '_stopping_context_enabled', False):
                self._expire_stopping_context()
            self._publish_schedule_snapshot()
            if getattr(self, '_require_task_binding', False):
                self._publish_task_recovery_snapshot()

    def _task_policy_control_cb(self, msg):
        try:
            with self._schedule_publish_lock, self._stopping_prefix_lock:
                now = rospy.Time.now().to_nsec()
                context = self._stopping_context.snapshot(now)
                event = self._task_policy_events.parse(msg.data, context, now, 'regime_control')
                validate_regime_control(event['payload'])
                old = self._task_policy_control
                if old is not None and old['task_context'] == event['task_context']:
                    a,b = old['payload'],event['payload']
                    if (old['source_session'] == event['source_session'] and
                            (b['mode_revision'] < a['mode_revision'] or
                             (b['mode_revision'] == a['mode_revision'] and
                              (b['mode'] != a['mode'] or b.get('recovery_target_sha256','') !=
                               a.get('recovery_target_sha256',''))))):
                        raise ValueError('mutated_recovery_control')
                self._task_policy_events.remember(event)
                self._task_policy_control = event
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            self._task_policy_control_rejected=str(exc)
            return

    @staticmethod
    def _task_target(first):
        return dict(observation_token=observation_token(first),
            region_token=observation_region_token(first), obligation_id=int(first['id']),
            q_vis=np.asarray(first['q_vis']).tolist(), joint_mask=q_vis_joint_mask(first).tolist(),
            sensor_id=int(first.get('per_sensor_selected_sensor_id', -1)),
            points=np.asarray(first['points']).tolist(),
            deadline_ns=round(float(first['deadline_abs_s'])*1e9))

    def _task_current_obligations(self, obligations):
        with self._stopping_prefix_lock:
            context=self._stopping_context.snapshot(rospy.Time.now().to_nsec())
            if context is None:return []
            key=list(context_key(context))
            return [ob for ob in obligations if
                    ob.get('stopping_prefix_request',{}).get('context_key')==key]

    def _publish_task_recovery_snapshot(self, obligations=None, frontier=None):
        """The exact current-task demand ledger owns completion, not SAFE/QP."""
        if not getattr(self, '_require_task_binding', False):
            return
        with self._schedule_publish_lock, self._stopping_prefix_lock:
            now = rospy.Time.now().to_nsec()
            context = self._stopping_context.snapshot(now)
            if context is None:
                self._task_recovery_source.reset()
                return
            current_key = list(context_key(context))
            frozen = obligations is not None
            obligations = [ob for ob in (obligations if frozen else self._ordered_obligations())
                           if ob.get('stopping_prefix_request', {}).get('context_key') == current_key]
            pending_count = len(self._stopping_context_queue.pending)
            target = None
            if obligations and pending_count:
                first = obligations[0]
                target = self._task_target(first)
                if frozen:
                    with self._obligation_lock:
                        current = next((ob for ob in self._obligations if int(ob['id']) == int(first['id'])), None)
                        expected = first.get('_task_storage_target', target)
                        if (current is None or canonical(self._task_target(current)) != canonical(expected) or
                                ('_task_storage_stack' in first and first['_task_storage_stack'] != self._repair_stack) or
                                current.get('stopping_prefix_request', {}).get('context_key') != current_key):
                            return  # Computation cannot publish a replaced selection.
            # An obligation still being retired cannot yield a completion.
            count = max(pending_count, len(obligations))
            diagnostic={}
            policy = build_policy(target, frontier, context,
                                  getattr(self, '_task_policy_control', None), now, diagnostic)
            self._task_policy_last_reason=diagnostic['reason']
            if frontier is not None and hasattr(rospy,'loginfo_throttle'):
                rospy.loginfo_throttle(1.,'[recovery_policy] state=%s control_error=%s',
                    diagnostic['reason'],getattr(self,'_task_policy_control_rejected','none'))
            event = self._task_recovery_source.snapshot(context, count, target, now, policy=policy)
            self._task_recovery_pub.publish(String(data=canonical(event)))

    def _publish_schedule_snapshot(self) -> None:
        obligations = self._ordered_obligations()
        msg = Float64MultiArray()
        data = []
        for ob in obligations:
            q = np.asarray(ob["q_vis"], dtype=np.float64).reshape(7)
            mask = q_vis_joint_mask(ob)
            data.extend([
                float(ob["id"]), float(ob["deadline_abs_s"]),
                *[float(v) for v in q],
                *[float(v) for v in mask],
            ])
        msg.data = data
        msg.layout.dim = [MultiArrayDimension(
            label="care_visibility_schedule_v2_qmask",
            size=len(obligations), stride=16)]
        self.schedule_pub.publish(msg)

        # Backward-compatible single waypoint = earliest accumulated obligation.
        if obligations:
            first = obligations[0]
            q = np.asarray(first["q_vis"], dtype=np.float64)
            with self._lock:
                self._q_vis = q.copy()
                self._q_zero = np.asarray(first["q_zero"], dtype=np.float64).copy()
                self._deadline_abs_s = float(first["deadline_abs_s"])
                self._deadline_from_start_s = max(
                    0.0, self._deadline_abs_s - rospy.Time.now().to_sec())
                self._generation_success = True
                self._shared_min_f = float(first["final_f_min"])
                self._shared_solution_mode = "c46_accumulated_multi_deadline"
                self._summary = "c46_schedule_ready"
            self.waypoint_pub.publish(_vector_msg(q))
            self.zero_pub.publish(_vector_msg(np.asarray(first["q_zero"])))
            d = Float64(); d.data = float(first["deadline_abs_s"])
            self.deadline_pub.publish(d)
        else:
            with self._lock:
                self._generation_success = False
                self._summary = "c46_no_obligations"

        now = rospy.Time.now().to_sec()
        unreachable = sum(
            not bool(ob["reachable_before_discovered_deadline_lower_bound"])
            for ob in obligations)
        qvis_times = [
            float(ob.get("q_vis_generation_ms", math.nan))
            for ob in obligations
            if math.isfinite(float(ob.get("q_vis_generation_ms", math.nan)))
        ]
        hybrid_used_count = sum(
            bool(ob.get("per_sensor_hybrid_used", False))
            for ob in obligations)
        earliest_sensor = (
            int(obligations[0].get("per_sensor_selected_sensor_id", -1))
            if obligations else -1)
        s = String()
        s.data = (
            "steering_policy=accumulated_multi_deadline"
            f" obligation_count={len(obligations)}"
            f" obligation_ids={':'.join(str(int(ob['id'])) for ob in obligations) or 'none'}"
            f" unreachable_at_discovery_count={unreachable}"
            f" earliest_deadline_remaining_s="
            f"{(float(obligations[0]['deadline_abs_s']) - now) if obligations else math.nan:.6f}"
            f" new_obligation_count={self._schedule_new_obligations}"
            f" matched_obligation_count={self._schedule_matched_obligations}"
            f" clear_count={self._schedule_clear_count}"
            f" generation_failure_count={self._schedule_generation_failures}"
            f" q_vis_generation_last_ms="
            f"{(qvis_times[-1] if qvis_times else math.nan):.3f}"
            f" q_vis_generation_max_ms="
            f"{(max(qvis_times) if qvis_times else math.nan):.3f}"
            f" per_sensor_hybrid_used_count={hybrid_used_count}"
            f" earliest_per_sensor_id={earliest_sensor}"
        )
        self.schedule_summary_pub.publish(s)

    def _stopping_prefix_callback(self, msg):
        try:
            if len(msg.data) > 1000000:
                return
            report = json.loads(msg.data)
            if (getattr(self, '_stopping_context_enabled', False) and
                    report.get('kind') == 'observation_context_invalidated'):
                with self._schedule_publish_lock, self._stopping_prefix_lock:
                    current = self._stopping_context.snapshot(rospy.Time.now().to_nsec())
                    if current and context_key(report['observation_context']) == context_key(current):
                        self._stopping_context.invalidate(report['reason'])
                        self._clear_stopping_context()
                self._publish_schedule()
                return
            validate_report(report, rospy.Time.now().to_sec())
            with self._stopping_prefix_lock:
                self._stopping_prefix_pending = report
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
            return

    def _process_stopping_prefix_demands(self):
        if getattr(self, '_stopping_context_enabled', False):
            self._process_context_observation_demands()
            return
        if not hasattr(self, '_stopping_prefix_lock'):
            return
        with self._stopping_prefix_lock:
            report, self._stopping_prefix_pending = self._stopping_prefix_pending, None
        if report is None and getattr(self, '_stopping_prefix_binding', None) is None:
            return
        with self._lock:
            trajectory, received, source = self._preferred_trajectory_locked()
        try:
            binding = None if trajectory is None else (trajectory.header.stamp.to_nsec(), trajectory_sha256(trajectory))
        except (ValueError, TypeError, AttributeError):
            binding = None
        # Retire immediately on replacement/staleness, even without a new map
        # report. A shared stamp cannot transfer another path's obligations.
        old_binding = getattr(self, '_stopping_prefix_binding', None)
        if old_binding is not None and old_binding != binding:
            with self._stopping_prefix_lock:
                self._stopping_prefix_queue.retire(self._stopping_prefix_path)
                self._stopping_prefix_path = None
                self._stopping_prefix_binding = None
            with self._obligation_lock:
                kept = []
                for ob in self._obligations:
                    if ob.get('stopping_prefix_request'):
                        if ob.get('stopping_prefix_owned'):
                            continue
                        ob.pop('stopping_prefix_request', None)
                    kept.append(ob)
                self._obligations = kept
            self._publish_schedule()
        if report is None or binding is None:
            return
        try:
            with self._stopping_prefix_lock:
                validate_trajectory_report(report, trajectory, rospy.Time.now().to_sec())
                previous_path = self._stopping_prefix_path
                if previous_path is not None and previous_path != report['path_id']:
                    self._stopping_prefix_queue.retire(previous_path)
                demands = self._stopping_prefix_queue.update(report, rospy.Time.now().to_sec(), trajectory.header.stamp.to_nsec())
                self._stopping_prefix_path = report['path_id']
                self._stopping_prefix_binding = binding
        except (ValueError, TypeError, KeyError):
            return
        wanted = {tuple(d['point']) for d in demands}
        with self._obligation_lock:
            kept = []
            for ob in self._obligations:
                request = ob.get('stopping_prefix_request')
                if request and (request['path_id'] != report['path_id'] or
                        not any(tuple(p) in wanted for p in request['points'])):
                    if ob.get('stopping_prefix_owned'):
                        continue
                    ob.pop('stopping_prefix_request', None)
                kept.append(ob)
            self._obligations = kept
        actionable = [d for d in demands if d['action'] != 'blocked_occupied']
        if not actionable:
            self._publish_schedule()
            return
        for region in self._cluster_regions(np.asarray([d['point'] for d in actionable])):
            members = [d for d in actionable if any(np.array_equal(d['point'], p) for p in region['points'])]
            deadline = min(d['observation_deadline_s'] for d in members)
            request = dict(path_id=report['path_id'], points=region['points'].tolist(),
                           construction_sha256=report['construction_sha256'], map_identity=report['map_identity'])
            with self._obligation_lock:
                matched = exact_region_owner(self._obligations,region['points'])
                if matched is not None:
                    matched['deadline_abs_s'] = min(matched['deadline_abs_s'], deadline)
                    previous_request=matched.get('stopping_prefix_request')
                    if previous_request and previous_request['path_id']==report['path_id']:
                        request['points']=[list(p) for p in sorted({tuple(p) for p in previous_request['points']+request['points']})]
                    matched['stopping_prefix_request'] = request
                    continue
                if len(self._obligations) >= self.max_obligations:
                    break
            try:
                window=report.get('trajectory_construction',{}).get('start_window_ns')
                region=dict(region,stopping_observation_deadline_abs_s=deadline,
                    stopping_observation_epoch_s=(window[0] if window else report['snapshot_ns'])*1e-9)
                ob = self._generate_new_obligation(region, trajectory,
                    max(.001, deadline-rospy.Time.now().to_sec()), received, 'stopping_prefix_lookahead')
                with self._lock:
                    current_trajectory, _, _ = self._preferred_trajectory_locked()
                if current_trajectory is None:
                    continue
                validate_trajectory_report(report, current_trajectory, rospy.Time.now().to_sec())
            except Exception as exc:
                rospy.logwarn('stopping-prefix observation generation retained for retry: %s', exc)
                continue
            ob['deadline_abs_s'] = min(ob['deadline_abs_s'], deadline)
            ob['stopping_prefix_request'] = request
            ob['stopping_prefix_owned'] = True
            with self._obligation_lock:
                if exact_region_owner(self._obligations,region['points']) is None and len(self._obligations) < self.max_obligations:
                    self._obligations.append(ob)
        self._publish_schedule()

    def _clear_stopping_context(self):
        """Caller holds publication and prefix locks, in that order."""
        self._stopping_context_queue.reset()
        self._stopping_prefix_pending = None
        self._stopping_context_key = None
        if getattr(self, '_require_task_binding', False):
            self._task_recovery_source.reset()
        with self._obligation_lock:
            kept = []
            for ob in self._obligations:
                if ob.get('stopping_prefix_request'):
                    if ob.get('stopping_prefix_owned'):
                        continue
                    ob.pop('stopping_prefix_request', None)
                kept.append(ob)
            self._obligations = kept

    def _expire_stopping_context(self):
        with self._stopping_prefix_lock:
            current = self._stopping_context.snapshot(rospy.Time.now().to_nsec())
            if self._stopping_context_key is not None and (
                    current is None or context_key(current) != self._stopping_context_key):
                self._clear_stopping_context()

    def _stopping_task_candidate_callback(self, msg):
        with self._stopping_prefix_lock:
            if msg.action != 'planned':
                return
            try:
                self._stopping_task_receiver.offer_candidate(msg.trajectory,msg.task_context,rospy.Time.now().to_nsec())
            except (ValueError, KeyError, TypeError) as exc:
                rospy.logwarn('task observation seed rejected: %s', exc)
                return

    def _stopping_context_callback(self, msg):
        try:
            if len(msg.data) > 32768:
                return
            event = json.loads(msg.data)
            with self._schedule_publish_lock, self._stopping_prefix_lock:
                before = self._stopping_context.snapshot(rospy.Time.now().to_nsec())
                after = (self._stopping_task_receiver.update(event, rospy.Time.now().to_nsec())
                         if getattr(self, "_require_task_binding", False) else
                         self._stopping_context.update(event, rospy.Time.now().to_nsec()))
                changed = before is None or after is None or context_key(before) != context_key(after)
                if changed:
                    self._clear_stopping_context()
            if changed:
                self._publish_schedule()
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
            return

    def _process_context_observation_demands(self):
        if not self._stopping_context_worker.acquire(blocking=False):
            return
        try:
            with self._schedule_publish_lock, self._stopping_prefix_lock:
                self._expire_stopping_context()
                context = self._stopping_context.snapshot(rospy.Time.now().to_nsec())
                report, self._stopping_prefix_pending = self._stopping_prefix_pending, None
                if context is None or report is None:
                    return
                try:
                    demands = self._stopping_context_queue.update(report, context, rospy.Time.now().to_sec())
                except ValueError as exc:
                    if str(exc) in ('observation_context_map_changed', 'task_observation_map_scope_changed'):
                        self._stopping_context.invalidate(str(exc))
                        self._clear_stopping_context()
                    raise
                self._stopping_context_key = context_key(context)
                wanted = {tuple(d['point']) for d in demands}
                with self._obligation_lock:
                    kept = []
                    for ob in self._obligations:
                        request = ob.get('stopping_prefix_request')
                        if request and not any(tuple(p) in wanted for p in request['points']):
                            if ob.get('stopping_prefix_owned'):
                                continue
                            ob.pop('stopping_prefix_request', None)
                        kept.append(ob)
                    self._obligations = kept
            if getattr(self, "_require_task_binding", False):
                with self._stopping_prefix_lock:
                    candidate = self._stopping_task_receiver.candidate(rospy.Time.now().to_nsec())
                    if candidate is None:
                        return
                    trajectory, received = candidate[0], rospy.Time(0,candidate[2])
            else:
                with self._lock:
                    trajectory, received, _ = self._preferred_trajectory_locked()
            if trajectory is None:
                return
            actionable = [d for d in demands if d['action'] != 'blocked_occupied']
            for region in self._cluster_regions(np.asarray([d['point'] for d in actionable])) if actionable else []:
                members = [d for d in actionable if any(np.array_equal(d['point'], p) for p in region['points'])]
                deadline = min(d['observation_deadline_s'] for d in members)
                request = dict(path_id=context_path(context), points=region['points'].tolist(),
                    context_key=list(context_key(context)), map_identity=report['map_identity'],
                    execution_eligible=False)
                with self._obligation_lock:
                    matched = exact_region_owner(self._obligations, region['points'])
                    full = len(self._obligations) >= self.max_obligations
                if matched is None and not full:
                    region = dict(region, stopping_observation_deadline_abs_s=deadline,
                        stopping_observation_epoch_s=context['started_ns']*1e-9,
                        stopping_observation_epoch_source='observation_context_started')
                    ob = self._generate_new_obligation(region, trajectory,
                        max(.001, deadline-rospy.Time.now().to_sec()), received, 'stopping_prefix_lookahead')
                elif matched is None:
                    break
                with self._schedule_publish_lock, self._stopping_prefix_lock:
                    current = self._stopping_context.snapshot(rospy.Time.now().to_nsec())
                    if current is None or context_key(current) != context_key(context):
                        return
                    # A newer report can complete work while generation runs.
                    pending = self._stopping_context_queue.pending
                    if not any(d['grid_index'] in pending for d in members):
                        continue
                    with self._obligation_lock:
                        existing = exact_region_owner(self._obligations, region['points'])
                        if existing is None:
                            if len(self._obligations) >= self.max_obligations:
                                continue
                            existing = ob
                            existing['stopping_prefix_owned'] = True
                            self._obligations.append(existing)
                        existing['deadline_abs_s'] = min(existing['deadline_abs_s'], deadline)
                        previous = existing.get('stopping_prefix_request')
                        if previous and previous.get('context_key') == request['context_key']:
                            request['points'] = [list(p) for p in sorted({tuple(p) for p in previous['points']+request['points']})]
                        existing['stopping_prefix_request'] = request
        except (ValueError, TypeError, KeyError) as exc:
            rospy.logwarn('context observation demand rejected: %s', exc)
        except Exception as exc:
            rospy.logwarn('context observation generation retained for retry: %s', exc)
        finally:
            self._stopping_context_worker.release()
            self._publish_schedule()

    # Parent timer calls these virtual methods.
    def _maybe_generate(self) -> None:
        self._process_stopping_prefix_demands()
        self._process_new_active_set()
        self._publish_schedule()

    def _publish_state(self) -> None:
        obligations = self._ordered_obligations()
        active = len(obligations) > 0
        a = Bool(); a.data = active; self.active_pub.publish(a)
        if obligations:
            first = obligations[0]
            self.waypoint_pub.publish(_vector_msg(np.asarray(first["q_vis"])))
            d = Float64(); d.data = float(first["deadline_abs_s"])
            self.deadline_pub.publish(d)

        now = rospy.Time.now().to_sec()
        msg = String()
        msg.data = (
            f"active={int(active)} seen=0 ready={int(active)}"
            " confidence=nan current_visibility=nan inside_map=1"
            f" deadline_from_start={math.nan:.6f}"
            f" deadline_remaining="
            f"{(float(obligations[0]['deadline_abs_s']) - now) if obligations else math.nan:.6f}"
            " reason=c46_accumulated_obligations"
            " steering_policy=accumulated_multi_deadline"
            f" obligation_count={len(obligations)}"
            f" active_set_size={sum(len(ob['points']) for ob in obligations)}"
            " shared_solution_mode=c46_accumulated_multi_deadline"
        )
        self.summary_pub.publish(msg)
        rospy.loginfo_throttle(0.5, "[vbc_multi_deadline] %s", msg.data)
