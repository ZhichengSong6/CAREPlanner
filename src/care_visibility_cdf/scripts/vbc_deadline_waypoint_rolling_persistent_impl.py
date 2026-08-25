#!/usr/bin/env python3
"""Persistent multi-region extension for the C4.3 rolling waypoint generator.

The upstream selector still publishes the current earliest bounded temporal layer.
This class adds short-horizon steering memory at the *spatial-region* level:

* incoming active-set points are clustered spatially with a bounded physical
  diameter;
* previously active regions are matched by voxel overlap first, then centroid
  distance;
* a region may be absent for a small number of fresh selector updates before it
  is retired;
* the learned steering active set is the union of all currently remembered
  regions.

Safety truth is unchanged: the selector/global guard still audit the current
predicted candidate set point by point.  Persistence affects steering only.

For small updates of the same risk patch, the previous shared q_vis is also used
as an optimization warm start.  Large centroid jumps (e.g. the two distinct
risk modes observed in case_003) intentionally fall back to the nominal-deadline
seed.
"""

from __future__ import annotations

import math
import threading

import numpy as np
import rospy
from std_msgs.msg import Bool, Float64MultiArray

from vbc_deadline_waypoint_rolling_impl import RollingVbcDeadlineWaypointNode


class PersistentRollingVbcDeadlineWaypointNode(RollingVbcDeadlineWaypointNode):
    def __init__(self) -> None:
        # Defaults must exist before the base constructor creates subscribers and
        # timers: callbacks may fire immediately after subscription.
        self.persistent_region_max_diameter_m = 0.12
        self.persistent_region_match_distance_m = 0.12
        self.persistent_region_miss_tolerance = 2
        self.warm_start_centroid_distance_m = 0.10

        self._persistent_lock = threading.Lock()
        self._persistent_regions = []
        self._next_persistent_region_id = 1

        self._raw_active_set_size = 0
        self._persistent_matched_count = 0
        self._persistent_new_count = 0
        self._persistent_retained_missing_count = 0
        self._persistent_retired_count = 0

        self._last_success_active_set_xyz = None
        self._last_success_active_set_keys = None
        self._last_success_q_vis = None
        self._last_warm_start_used = False
        self._last_warm_start_centroid_distance_m = math.nan
        self._last_warm_start_jaccard = math.nan
        self._warm_seed_override = None
        self._warm_seed_pending = False

        super().__init__()

        self.persistent_region_max_diameter_m = float(rospy.get_param(
            "~persistent_region_max_diameter_m", 0.12))
        self.persistent_region_match_distance_m = float(rospy.get_param(
            "~persistent_region_match_distance_m",
            self.persistent_region_max_diameter_m))
        self.persistent_region_miss_tolerance = int(rospy.get_param(
            "~persistent_region_miss_tolerance", 2))
        self.warm_start_centroid_distance_m = float(rospy.get_param(
            "~warm_start_centroid_distance_m",
            2.0 * self.target_cell_resolution))

        if self.persistent_region_max_diameter_m <= 0.0:
            raise ValueError("~persistent_region_max_diameter_m must be positive")
        if self.persistent_region_match_distance_m <= 0.0:
            raise ValueError("~persistent_region_match_distance_m must be positive")
        if self.persistent_region_miss_tolerance < 0:
            raise ValueError("~persistent_region_miss_tolerance must be >= 0")
        if self.warm_start_centroid_distance_m <= 0.0:
            raise ValueError("~warm_start_centroid_distance_m must be positive")

        rospy.logwarn(
            "[vbc_waypoint_persistent] region_diam=%.3fm match=%.3fm "
            "miss_tolerance=%d warm_centroid=%.3fm",
            self.persistent_region_max_diameter_m,
            self.persistent_region_match_distance_m,
            self.persistent_region_miss_tolerance,
            self.warm_start_centroid_distance_m)

    @staticmethod
    def _centroid(points):
        if points is None or len(points) == 0:
            return None
        return np.mean(np.asarray(points, dtype=np.float64), axis=0)

    def _point_keys(self, points):
        return tuple(sorted({self._cell_key(p) for p in points}))

    @staticmethod
    def _jaccard(keys_a, keys_b):
        a = set(keys_a or ())
        b = set(keys_b or ())
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return float(len(a & b)) / float(len(a | b))

    @staticmethod
    def _cluster_diameter(points):
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] <= 1:
            return 0.0
        diff = pts[:, None, :] - pts[None, :, :]
        return float(np.max(np.linalg.norm(diff, axis=-1)))

    def _cluster_regions(self, points):
        """Complete-link agglomeration with a hard physical diameter bound."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        clusters = [[i] for i in range(pts.shape[0])]

        while True:
            best = None
            best_distance = math.inf
            centroids = [np.mean(pts[idx], axis=0) for idx in clusters]
            for a in range(len(clusters)):
                for b in range(a + 1, len(clusters)):
                    merged = clusters[a] + clusters[b]
                    if self._cluster_diameter(pts[merged]) > (
                            self.persistent_region_max_diameter_m + 1e-9):
                        continue
                    d = float(np.linalg.norm(centroids[a] - centroids[b]))
                    if d < best_distance:
                        best_distance = d
                        best = (a, b)
            if best is None:
                break
            a, b = best
            clusters[a].extend(clusters[b])
            del clusters[b]

        regions = []
        for idx in clusters:
            region_points = pts[idx].copy()
            regions.append({
                "points": region_points,
                "keys": self._point_keys(region_points),
                "centroid": np.mean(region_points, axis=0),
            })
        return regions

    def _match_score(self, persistent, incoming):
        jaccard = self._jaccard(persistent["keys"], incoming["keys"])
        distance = float(np.linalg.norm(
            persistent["centroid"] - incoming["centroid"]))
        if jaccard > 0.0:
            # Overlap dominates; centroid distance only breaks ties.
            return (2, jaccard, -distance)
        if distance <= self.persistent_region_match_distance_m:
            return (1, -distance, 0.0)
        return None

    def _update_persistent_regions(self, raw_points):
        incoming = self._cluster_regions(raw_points) if len(raw_points) else []
        used = set()
        matched_count = 0
        retained_missing = 0
        retired_count = 0

        updated = []
        for persistent in self._persistent_regions:
            best_j = None
            best_score = None
            for j, region in enumerate(incoming):
                if j in used:
                    continue
                score = self._match_score(persistent, region)
                if score is None:
                    continue
                if best_score is None or score > best_score:
                    best_score = score
                    best_j = j

            if best_j is not None:
                region = incoming[best_j]
                used.add(best_j)
                persistent["points"] = region["points"]
                persistent["keys"] = region["keys"]
                persistent["centroid"] = region["centroid"]
                persistent["miss_count"] = 0
                persistent["matched_now"] = True
                matched_count += 1
                updated.append(persistent)
            else:
                persistent["miss_count"] += 1
                persistent["matched_now"] = False
                if persistent["miss_count"] <= self.persistent_region_miss_tolerance:
                    retained_missing += 1
                    updated.append(persistent)
                else:
                    retired_count += 1

        new_count = 0
        for j, region in enumerate(incoming):
            if j in used:
                continue
            updated.append({
                "id": self._next_persistent_region_id,
                "points": region["points"],
                "keys": region["keys"],
                "centroid": region["centroid"],
                "miss_count": 0,
                "matched_now": True,
            })
            self._next_persistent_region_id += 1
            new_count += 1

        self._persistent_regions = updated
        self._persistent_matched_count = matched_count
        self._persistent_new_count = new_count
        self._persistent_retained_missing_count = retained_missing
        self._persistent_retired_count = retired_count

        # Union all remembered regions by confidence-map voxel key.  For a key
        # present in more than one tracker, prefer a currently matched point.
        by_key = {}
        for region in sorted(
                self._persistent_regions,
                key=lambda r: (not r["matched_now"], r["id"])):
            for point in region["points"]:
                key = self._cell_key(point)
                if key not in by_key:
                    by_key[key] = np.asarray(point, dtype=np.float64).copy()

        ordered_keys = tuple(sorted(by_key.keys()))
        if not ordered_keys:
            return np.zeros((0, 3), dtype=np.float64)
        return np.asarray([by_key[k] for k in ordered_keys], dtype=np.float64)

    def _active_set_callback(self, msg: Float64MultiArray) -> None:
        if msg is None:
            return
        values = np.asarray(list(msg.data), dtype=np.float64)
        if values.size == 0:
            raw_points = np.zeros((0, 3), dtype=np.float64)
        elif values.size % 3 != 0:
            rospy.logwarn_throttle(
                1.0,
                "[vbc_waypoint_persistent] active-set length %d not divisible by 3",
                int(values.size))
            return
        else:
            raw_points = values.reshape(-1, 3)
            if not np.all(np.isfinite(raw_points)):
                return

        # Canonicalize raw points before region tracking.
        raw_by_key = {}
        for point in raw_points:
            raw_by_key[self._cell_key(point)] = point.copy()
        raw_keys = tuple(sorted(raw_by_key.keys()))
        raw_points = (
            np.asarray([raw_by_key[k] for k in raw_keys], dtype=np.float64)
            if raw_keys else np.zeros((0, 3), dtype=np.float64))

        with self._persistent_lock:
            self._raw_active_set_size = int(raw_points.shape[0])
            persistent_points = self._update_persistent_regions(raw_points)
            region_count = len(self._persistent_regions)
            matched = self._persistent_matched_count
            retained = self._persistent_retained_missing_count
            new_count = self._persistent_new_count
            retired = self._persistent_retired_count

        persistent_msg = Float64MultiArray()
        persistent_msg.data = persistent_points.reshape(-1).astype(float).tolist()
        super()._active_set_callback(persistent_msg)

        rospy.loginfo_throttle(
            0.5,
            "[vbc_waypoint_persistent] raw_points=%d persistent_regions=%d "
            "persistent_points=%d matched=%d retained_missing=%d new=%d retired=%d",
            int(raw_points.shape[0]), region_count,
            int(persistent_points.shape[0]), matched, retained, new_count, retired)

    def _selection_active_callback(self, msg: Bool) -> None:
        super()._selection_active_callback(msg)
        if msg is None or bool(msg.data):
            return
        # End of a broker-selected steering episode: do not leak risk memory or
        # a warm start across measured-state replans / later episodes.
        with self._persistent_lock:
            self._persistent_regions = []
            self._persistent_matched_count = 0
            self._persistent_new_count = 0
            self._persistent_retained_missing_count = 0
            self._persistent_retired_count = 0
            self._raw_active_set_size = 0
            self._last_success_active_set_xyz = None
            self._last_success_active_set_keys = None
            self._last_success_q_vis = None
            self._last_warm_start_used = False
            self._last_warm_start_centroid_distance_m = math.nan
            self._last_warm_start_jaccard = math.nan
        empty = Float64MultiArray()
        empty.data = []
        # Bypass our persistent callback so the base cache is truly cleared.
        super()._active_set_callback(empty)

    def _sample_trajectory(self, trajectory, t):
        if self._warm_seed_pending and self._warm_seed_override is not None:
            self._warm_seed_pending = False
            return np.asarray(self._warm_seed_override, dtype=np.float64).copy()
        return super()._sample_trajectory(trajectory, t)

    def _generate_active_set_waypoint(
            self, points_xyz, trajectory, sweep_time_s, trajectory_received):
        points_np = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
        current_keys = self._point_keys(points_np)
        current_centroid = self._centroid(points_np)

        warm_used = False
        warm_distance = math.nan
        warm_jaccard = math.nan
        warm_q = None
        with self._persistent_lock:
            last_points = (
                None if self._last_success_active_set_xyz is None
                else self._last_success_active_set_xyz.copy())
            last_keys = self._last_success_active_set_keys
            last_q = (
                None if self._last_success_q_vis is None
                else self._last_success_q_vis.copy())

        if (last_points is not None and last_q is not None and
                current_centroid is not None and len(last_points) > 0):
            last_centroid = self._centroid(last_points)
            warm_distance = float(np.linalg.norm(current_centroid - last_centroid))
            warm_jaccard = self._jaccard(current_keys, last_keys)
            if warm_distance <= self.warm_start_centroid_distance_m:
                warm_used = True
                warm_q = last_q

        # Preserve the true nominal deadline configuration for diagnostics.  The
        # base generator calls self._sample_trajectory exactly once at startup;
        # when warm-starting we intercept that call with the previous q_vis.
        deadline_from_start = max(0.0, sweep_time_s - self.safety_margin_s)
        true_nominal_q = np.asarray(
            super()._sample_trajectory(trajectory, deadline_from_start),
            dtype=np.float64).copy()

        self._warm_seed_override = warm_q
        self._warm_seed_pending = bool(warm_used)
        try:
            result = super()._generate_active_set_waypoint(
                points_np, trajectory, sweep_time_s, trajectory_received)
        finally:
            self._warm_seed_pending = False
            self._warm_seed_override = None

        optimization_initial_q = np.asarray(
            result.get("q_deadline_nominal", true_nominal_q),
            dtype=np.float64)
        result["q_optimization_initial"] = optimization_initial_q.tolist()
        result["optimization_initial_source"] = (
            "previous_shared_q_vis_warm_start" if warm_used
            else "nominal_deadline_configuration")
        result["warm_start_used"] = bool(warm_used)
        result["warm_start_centroid_distance_m"] = float(warm_distance)
        result["warm_start_active_set_jaccard"] = float(warm_jaccard)
        result["warm_start_centroid_threshold_m"] = (
            self.warm_start_centroid_distance_m)

        # Correct nominal-labelled fields because the base routine intentionally
        # saw the warm seed as its q0 when warm_used=True.
        result["q_deadline_nominal"] = true_nominal_q.tolist()
        q_zero = np.asarray(result["q_zero"], dtype=np.float64)
        q_vis = np.asarray(result["q_vis"], dtype=np.float64)
        result["distance_qzero_from_nominal"] = float(
            np.linalg.norm(q_zero - true_nominal_q))
        result["distance_qvis_from_nominal"] = float(
            np.linalg.norm(q_vis - true_nominal_q))

        with self._persistent_lock:
            self._last_success_active_set_xyz = points_np.copy()
            self._last_success_active_set_keys = current_keys
            self._last_success_q_vis = q_vis.copy()
            self._last_warm_start_used = bool(warm_used)
            self._last_warm_start_centroid_distance_m = float(warm_distance)
            self._last_warm_start_jaccard = float(warm_jaccard)

        return result

    def _summary_message(
            self, selection_active, active, seen, ready, confidence,
            current_visibility, inside, deadline_from_start, deadline_abs,
            reason, trajectory_source, target_cell_key,
            active_set_size=0, shared_min_f=math.nan,
            shared_solution_mode="single_target"):
        msg = RollingVbcDeadlineWaypointNode._summary_message(
            selection_active=selection_active,
            active=active,
            seen=seen,
            ready=ready,
            confidence=confidence,
            current_visibility=current_visibility,
            inside=inside,
            deadline_from_start=deadline_from_start,
            deadline_abs=deadline_abs,
            reason=reason,
            trajectory_source=trajectory_source,
            target_cell_key=target_cell_key,
            active_set_size=active_set_size,
            shared_min_f=shared_min_f,
            shared_solution_mode=shared_solution_mode)

        with self._persistent_lock:
            region_count = len(self._persistent_regions)
            raw_size = self._raw_active_set_size
            matched = self._persistent_matched_count
            new_count = self._persistent_new_count
            retained = self._persistent_retained_missing_count
            retired = self._persistent_retired_count
            warm_used = self._last_warm_start_used
            warm_distance = self._last_warm_start_centroid_distance_m
            warm_jaccard = self._last_warm_start_jaccard

        msg.data += (
            f" persistent_region_count={region_count}"
            f" raw_active_set_size={raw_size}"
            f" persistent_matched_count={matched}"
            f" persistent_new_count={new_count}"
            f" persistent_retained_missing_count={retained}"
            f" persistent_retired_count={retired}"
            f" warm_start_used={int(warm_used)}"
            f" warm_start_centroid_distance_m={warm_distance:.6f}"
            f" warm_start_jaccard={warm_jaccard:.6f}"
        )
        return msg
