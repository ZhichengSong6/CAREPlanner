#!/usr/bin/env python3
"""Deadline-aware sequential spatial-region steering for CAREPlanner C4.5.

The exact VBC selector remains the safety authority and still audits every
low-confidence sweep point independently.  This module changes only the learned
steering side of the pipeline.

C4.3/C4.4 formed one shared active set from every spatial region in the earliest
unsafe temporal layer and asked the learned field for one common q_vis.  That is
stronger than VBC semantics: VBC only requires each region to become visible
before its own sweep deadline, not all regions to be simultaneously visible in
one configuration.

This class reuses the existing persistent spatial-region tracker but exposes only
ONE currently selected region to the learned q_vis generator.  The selected
region is retained while it remains a current VBC violation.  Once it disappears
from the fresh violating set, steering advances to the region containing the
selector's current representative target (the selector chooses the most urgent
violation by visibility/margin/sweep time), with nearest-centroid fallback.

Important: exact candidate VBC is unchanged.  A sequential steering choice can
never bypass verification or commit an unsafe trajectory.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import rospy
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool, Float64MultiArray

from vbc_deadline_waypoint_rolling_persistent_impl import (
    PersistentRollingVbcDeadlineWaypointNode,
)


class DeadlineSequentialRollingVbcWaypointNode(
        PersistentRollingVbcDeadlineWaypointNode):
    """Expose one persistent spatial risk region at a time to learned steering."""

    def __init__(self) -> None:
        # Must exist before super(): ROS callbacks may start immediately.
        self._sequential_selected_region_id: Optional[int] = None
        self._sequential_target_xyz: Optional[np.ndarray] = None
        self._sequential_selected_region_size = 0
        self._sequential_candidate_region_count = 0
        self._sequential_switch_count = 0
        self._sequential_retained_count = 0
        self._sequential_selected_target_distance_m = math.nan
        self._sequential_selection_reason = "none"

        super().__init__()

        self.sequential_target_match_distance_m = float(rospy.get_param(
            "~sequential_target_match_distance_m",
            self.persistent_region_max_diameter_m))
        if self.sequential_target_match_distance_m <= 0.0:
            raise ValueError("~sequential_target_match_distance_m must be positive")

        rospy.logwarn(
            "[vbc_waypoint_sequential] ENABLED: one spatial VBC region at a time; "
            "retain-until-resolved, target/urgency guided; target_match=%.3fm",
            self.sequential_target_match_distance_m)

    def _target_callback(self, msg: PointStamped) -> None:
        # Cache the selector's representative violation.  In temporal-cluster
        # mode that representative is chosen by visibility/margin/sweep urgency,
        # so its containing region is the deadline-aware next region.
        if msg is not None:
            xyz = self._target_array(msg)
            if np.all(np.isfinite(xyz)):
                self._sequential_target_xyz = xyz.copy()
        super()._target_callback(msg)

    @staticmethod
    def _region_points(region):
        pts = np.asarray(region.get("points", []), dtype=np.float64)
        if pts.size == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return pts.reshape(-1, 3)

    def _target_region(self, matched_regions):
        target = self._sequential_target_xyz
        if target is None or not matched_regions:
            return None, math.nan

        # First prefer a region that contains the representative target's voxel.
        target_key = self._cell_key(target)
        exact = [r for r in matched_regions if target_key in set(r.get("keys", ()))]
        if exact:
            region = min(exact, key=lambda r: int(r["id"]))
            return region, 0.0

        # The selector and Python tracker use the same physical active points but
        # may quantize/merge slightly differently; nearest centroid is robust.
        best = min(
            matched_regions,
            key=lambda r: float(np.linalg.norm(r["centroid"] - target)))
        distance = float(np.linalg.norm(best["centroid"] - target))
        if distance <= self.sequential_target_match_distance_m:
            return best, distance
        return None, distance

    def _canonical_region_points(self, region):
        by_key = {}
        for point in self._region_points(region):
            key = self._cell_key(point)
            if key not in by_key:
                by_key[key] = point.copy()
        keys = tuple(sorted(by_key.keys()))
        if not keys:
            return np.zeros((0, 3), dtype=np.float64)
        return np.asarray([by_key[k] for k in keys], dtype=np.float64)

    def _update_persistent_regions(self, raw_points):
        # Let the validated persistent tracker do spatial clustering + identity
        # matching first.  We intentionally discard its UNION return value.
        super()._update_persistent_regions(raw_points)

        matched = [
            r for r in self._persistent_regions
            if bool(r.get("matched_now", False)) and len(r.get("points", [])) > 0
        ]
        self._sequential_candidate_region_count = len(matched)

        previous_id = self._sequential_selected_region_id
        selected = None

        # Hysteresis with semantics: keep working on the same region while it is
        # still present in the *fresh* VBC violation set.  Do NOT keep a missing
        # region merely because the generic persistent tracker tolerates misses.
        if previous_id is not None:
            selected = next(
                (r for r in matched if int(r["id"]) == int(previous_id)), None)
        if selected is not None:
            self._sequential_retained_count += 1
            self._sequential_selection_reason = "retained_unresolved_region"
            target = self._sequential_target_xyz
            self._sequential_selected_target_distance_m = (
                float(np.linalg.norm(selected["centroid"] - target))
                if target is not None else math.nan)
        else:
            target_selected, distance = self._target_region(matched)
            if target_selected is not None:
                selected = target_selected
                self._sequential_selection_reason = "selector_urgent_target_region"
                self._sequential_selected_target_distance_m = float(distance)
            elif matched:
                # Deterministic fallback.  IDs are assigned when a spatial patch
                # first appears, so this also avoids arbitrary centroid flipping.
                selected = min(matched, key=lambda r: int(r["id"]))
                self._sequential_selection_reason = "oldest_pending_region_fallback"
                target = self._sequential_target_xyz
                self._sequential_selected_target_distance_m = (
                    float(np.linalg.norm(selected["centroid"] - target))
                    if target is not None else math.nan)
            else:
                self._sequential_selection_reason = "no_pending_region"
                self._sequential_selected_target_distance_m = math.nan

            new_id = None if selected is None else int(selected["id"])
            if new_id != previous_id:
                self._sequential_switch_count += 1
            self._sequential_selected_region_id = new_id

        if selected is None:
            self._sequential_selected_region_id = None
            self._sequential_selected_region_size = 0
            return np.zeros((0, 3), dtype=np.float64)

        self._sequential_selected_region_id = int(selected["id"])
        points = self._canonical_region_points(selected)
        self._sequential_selected_region_size = int(points.shape[0])
        return points

    def _selection_active_callback(self, msg: Bool) -> None:
        # Online C4.4/C4.5 keeps region identity across brief globally-safe probe
        # intervals.  Retirement is driven by future exact VBC active sets.
        # The online wrapper deliberately bypasses the base persistent-memory
        # clear; this method only resets the *active steering* state if called
        # directly outside that wrapper.
        super()._selection_active_callback(msg)
        if msg is not None and not bool(msg.data):
            self._sequential_selected_region_id = None
            self._sequential_selected_region_size = 0
            self._sequential_selection_reason = "selection_inactive"

    def _summary_message(
            self, selection_active, active, seen, ready, confidence,
            current_visibility, inside, deadline_from_start, deadline_abs,
            reason, trajectory_source, target_cell_key,
            active_set_size=0, shared_min_f=math.nan,
            shared_solution_mode="single_target"):
        msg = super()._summary_message(
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
        rid = -1 if self._sequential_selected_region_id is None else int(
            self._sequential_selected_region_id)
        msg.data += (
            " steering_policy=deadline_sequential_region"
            f" sequential_selected_region_id={rid}"
            f" sequential_selected_region_size={self._sequential_selected_region_size}"
            f" sequential_pending_region_count={self._sequential_candidate_region_count}"
            f" sequential_switch_count={self._sequential_switch_count}"
            f" sequential_retained_count={self._sequential_retained_count}"
            f" sequential_target_distance_m={self._sequential_selected_target_distance_m:.6f}"
            f" sequential_selection_reason={self._sequential_selection_reason}"
        )
        return msg
