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
it disappears from one rejected hypothetical candidate.  The whole schedule is
cleared only after the candidate VBC verifier reports a fresh predicted SAFE
verdict.  The downstream MPC receives all obligations at once and adds one
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
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray, String

from evaluate_direct_vs_projection_ascent import DEFAULT_JOINT_NAMES
from vbc_deadline_waypoint_node import _fmt, _vector_msg
from vbc_deadline_waypoint_rolling_impl import RollingVbcDeadlineWaypointNode


_TOKEN = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


class AccumulatedMultiDeadlineWaypointNode(RollingVbcDeadlineWaypointNode):
    """Accumulate spatial visibility obligations until exact VBC is globally safe."""

    def __init__(self) -> None:
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
        t_generate_start = time.perf_counter()
        try:
            result = RollingVbcDeadlineWaypointNode._generate_active_set_waypoint(
                self, region["points"], trajectory, sweep_time_s,
                trajectory_received)
        finally:
            q_vis_generation_ms = 1000.0 * (
                time.perf_counter() - t_generate_start)
            self._seed_override = None

        q_vis = np.asarray(result["q_vis"], dtype=np.float64)
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

        ob = {
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
            "q_vis": q_vis,
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
            "q_vis_generation_ms": float(q_vis_generation_ms),
        }
        self._next_obligation_id += 1

        # Save a compact per-obligation trace alongside existing projector traces.
        trace = dict(result)
        trace.update({
            "c4_6_obligation_id": ob["id"],
            "c4_6_seed_source": "measured_joint_state",
            "c4_6_measured_seed_q": measured.tolist(),
            "c4_6_min_hit_time_from_measured_rest_s": min_hit,
            "c4_6_deadline_remaining_at_discovery_s": deadline_remaining,
            "c4_6_reachable_before_discovered_deadline_lower_bound": ob[
                "reachable_before_discovered_deadline_lower_bound"],
            "c4_6_q_vis_generation_ms": float(q_vis_generation_ms),
            "c4_6_q_vis_source_centroid": source_centroid.tolist(),
            "c4_6_q_vis_source_xyz_min": source_xyz_min.tolist(),
            "c4_6_q_vis_source_xyz_max": source_xyz_max.tolist(),
            "c4_6_q_vis_source_point_count": int(source_points.shape[0]),
        })
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = Path(self.output_root) / (
            f"c46_obligation_{ob['id']:03d}_{stamp}.json")
        path.write_text(json.dumps(trace, indent=2, allow_nan=True))

        rospy.logwarn(
            "[vbc_multi_deadline] ADD obligation=%d points=%d deadline_rem=%.3fs "
            "Tmin_rest=%.3fs reachable_lb=%d min_f=%+.4f qvis_ms=%.3f q_vis=%s",
            ob["id"], len(ob["points"]), deadline_remaining, min_hit,
            int(ob["reachable_before_discovered_deadline_lower_bound"]),
            ob["final_f_min"], q_vis_generation_ms, _fmt(q_vis, 4))
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
            had = len(self._obligations)
            self._obligations = []
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
        obligations = self._ordered_obligations()
        msg = Float64MultiArray()
        data = []
        for ob in obligations:
            q = np.asarray(ob["q_vis"], dtype=np.float64).reshape(7)
            data.extend([
                float(ob["id"]), float(ob["deadline_abs_s"]),
                *[float(v) for v in q],
            ])
        msg.data = data
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
        )
        self.schedule_summary_pub.publish(s)

    # Parent timer calls these virtual methods.
    def _maybe_generate(self) -> None:
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
