#!/usr/bin/env python3
"""Fixed-target visibility-acquisition diagnostic.

Diagnostic-only mode used to answer one narrow question:
can the unchanged CAREPlanner safety/execution stack reach a hand-selected
visibility configuration and make the *real* confidence/ToF pipeline see one
fixed world point?

Natural VBC/GCDF active-set inputs are intentionally ignored in this mode.
The only visibility obligation is injected from explicit environment values.

Default formal behavior is untouched because this class is instantiated only
through region_schedule_mode=fixed_visibility_diagnostic.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import rospy
from std_msgs.msg import Bool

from vbc_visibility_acquisition_impl import VisibilityAcquisitionWaypointNode


def _env_vec(name: str, default: str, n: int) -> np.ndarray:
    raw = os.environ.get(name, default)
    vals = [float(x) for x in str(raw).replace(" ", "").split(",") if x != ""]
    arr = np.asarray(vals, dtype=np.float64)
    if arr.shape != (n,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain {n} finite comma-separated values")
    return arr


class FixedVisibilityDiagnosticWaypointNode(VisibilityAcquisitionWaypointNode):
    """One immutable target/q_vis obligation; clear only on actual confidence."""

    def __init__(self) -> None:
        # Timer/callbacks may fire while the parent constructor is still running.
        self._fixed_diag_ready = False
        self._fixed_diag_injected = False
        self._fixed_diag_ignored_active_sets = 0

        self._fixed_diag_target = _env_vec(
            "CARE_FIXED_VIS_TARGET", "0.1,0.05,0.15", 3)
        self._fixed_diag_q_vis = _env_vec(
            "CARE_FIXED_VIS_Q",
            "-0.8144537806510925,0.5605988502502441,1.0171856880187988,"
            "-2.180335760116577,0.12358028441667557,-1.7687498331069946,"
            "1.1331826448440552",
            7)
        # q_zero is not used as the REPAIR target in C4.7/C5.x.  For this
        # fixed diagnostic we publish the same known-valid configuration so no
        # stale projector artifact can leak into the run.
        self._fixed_diag_q_zero = _env_vec(
            "CARE_FIXED_VIS_Q_ZERO",
            ",".join(str(float(v)) for v in self._fixed_diag_q_vis),
            7)
        self._fixed_diag_obligation_id = int(
            os.environ.get("CARE_FIXED_VIS_OBLIGATION_ID", "1"))
        self._fixed_diag_label = str(os.environ.get(
            "CARE_FIXED_VIS_LABEL",
            "case026_fixed_target_robust_s7"))
        if self._fixed_diag_obligation_id < 1:
            raise ValueError("CARE_FIXED_VIS_OBLIGATION_ID must be >= 1")

        super().__init__()

        self._fixed_diag_ready = True
        rospy.logwarn(
            "[fixed_visibility_diag] ARMED label=%s target=%s q_vis=%s; "
            "natural active sets will be ignored",
            self._fixed_diag_label,
            np.array2string(self._fixed_diag_target, precision=6),
            np.array2string(self._fixed_diag_q_vis, precision=6))

    def _active_set_callback(self, msg) -> None:
        """Ignore natural VBC active sets; fixed target is the only obligation."""
        self._fixed_diag_ignored_active_sets += 1
        rospy.loginfo_throttle(
            1.0,
            "[fixed_visibility_diag] ignoring natural active-set messages "
            "count=%d",
            self._fixed_diag_ignored_active_sets)

    def _process_new_active_set(self) -> None:
        """No-op by construction: natural obligations are disabled."""
        return

    def _inject_fixed_obligation_if_ready(self) -> None:
        if not self._fixed_diag_ready or self._fixed_diag_injected:
            return
        measured = self._latest_measured_q
        if measured is None:
            return
        measured = np.asarray(measured, dtype=np.float64).reshape(7)
        if not np.all(np.isfinite(measured)):
            return

        target = self._fixed_diag_target.copy()
        q_vis = self._fixed_diag_q_vis.copy()
        q_zero = self._fixed_diag_q_zero.copy()
        now_s = rospy.Time.now().to_sec()
        oid = int(self._fixed_diag_obligation_id)
        key = self._cell_key(target)
        min_hit = self._rest_min_hit_time(measured, q_vis)

        ob = {
            "id": oid,
            "points": target.reshape(1, 3).copy(),
            "keys": (key,),
            "centroid": target.copy(),
            "q_vis_source_points": target.reshape(1, 3).copy(),
            "q_vis_source_keys": (key,),
            "q_vis_source_centroid": target.copy(),
            "q_vis_source_xyz_min": target.copy(),
            "q_vis_source_xyz_max": target.copy(),
            "geometry_match_update_count": 0,
            "geometry_match_change_count": 0,
            "geometry_changed_since_qvis": False,
            "last_match_centroid_shift_m": 0.0,
            "max_centroid_shift_from_qvis_m": 0.0,
            "last_geometry_match_source": "fixed_diagnostic",
            "refinement_depth": 0,
            "parent_obligation_id": -1,
            "root_obligation_id": oid,
            "refinement_reason": "fixed_diagnostic",
            "refinement_family_id": -1,
            "refinement_partition_anchor": target.copy(),
            "q_vis": q_vis.copy(),
            "q_zero": q_zero.copy(),
            # C4.7 REPAIR ignores nominal deadlines. Keep a finite bookkeeping
            # value for inherited ordering/diagnostics only.
            "deadline_abs_s": float(now_s + 3600.0),
            "discovered_sweep_time_s": math.nan,
            "discovered_at_ros_s": float(now_s),
            "last_seen_ros_s": float(now_s),
            "trajectory_source": "fixed_visibility_diagnostic",
            "min_hit_time_from_measured_rest_s": float(min_hit),
            "deadline_remaining_at_discovery_s": math.nan,
            "reachable_before_discovered_deadline_lower_bound": True,
            "final_f_min": 0.0,
            "shared_solution_mode": "fixed_visibility_diagnostic",
            "q_vis_generation_ms": 0.0,
            "fixed_visibility_diagnostic": True,
            "fixed_visibility_label": self._fixed_diag_label,
            "fixed_measured_seed_q": measured.copy(),
        }

        with self._obligation_lock:
            # This mode owns the whole obligation store. The list should be empty
            # because natural active sets are ignored; overwrite defensively.
            self._obligations = [ob]
            self._next_obligation_id = max(self._next_obligation_id, oid + 1)

        self._fixed_diag_injected = True
        self._acquisition_started = True
        self._acquisition_complete = False
        self.acquisition_complete_pub.publish(Bool(data=False))

        self.output_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        trace_path = Path(self.output_root) / (
            f"fixed_visibility_obligation_{oid:03d}_{stamp}.json")
        payload = {
            "mode": "fixed_visibility_diagnostic",
            "label": self._fixed_diag_label,
            "obligation_id": oid,
            "target_xyz": target.tolist(),
            "q_vis": q_vis.tolist(),
            "q_zero": q_zero.tolist(),
            "measured_seed_q": measured.tolist(),
            "min_hit_time_from_measured_rest_s": float(min_hit),
            "natural_active_sets_ignored_before_injection":
                int(self._fixed_diag_ignored_active_sets),
        }
        trace_path.write_text(json.dumps(payload, indent=2, allow_nan=True))

        rospy.logwarn(
            "[fixed_visibility_diag] INJECTED obligation=%d target=%s "
            "q_vis=%s Tmin_rest=%.3fs trace=%s",
            oid,
            np.array2string(target, precision=6),
            np.array2string(q_vis, precision=6),
            min_hit,
            str(trace_path))
        self._publish_schedule()

    def _maybe_generate(self) -> None:
        if not self._fixed_diag_ready:
            return
        self._inject_fixed_obligation_if_ready()
        self._update_actual_visibility_completion()
        self._publish_schedule()


__all__ = ["FixedVisibilityDiagnosticWaypointNode"]
