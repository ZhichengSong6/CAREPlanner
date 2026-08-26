#!/usr/bin/env python3
"""Online CAREPlanner visibility waypoint generator.

Modes:
  visibility_acquisition (C4.7/C4.8)
      REPAIR abandons nominal deadlines. Persist learned visibility goals until
      the real confidence map confirms they were seen.

  accumulated_multi_deadline (C4.6 baseline)
      Accumulate spatial visibility obligations across rejected candidates,
      generate each q_vis from measured q, and publish a multi-deadline schedule.

  deadline_sequential (C4.5 baseline)
      Keep one spatial region at a time across planner cycles.

  shared_persistent (C4.4 baseline)
      Union remembered regions and solve for one shared q_vis.

C4.8 repair-prefix verification is orthogonal to q_vis generation. It can be
selected either by private ROS params or by the one-command runner environment
variables C4_REPAIR_PREFIX_VERIFY, C4_REPAIR_PREFIX_S, C4_REPAIR_BRAKE_DT_S and
C4_REPAIR_HOLD_S. NORMAL candidates always keep full-horizon VBC verification.
"""

import os
import time

import numpy as np
import rospy
import torch

from evaluate_direct_vs_projection_ascent import model_value_and_grad_q
from vbc_deadline_waypoint_rolling_impl import RollingVbcDeadlineWaypointNode
from vbc_deadline_waypoint_rolling_persistent_impl import (
    PersistentRollingVbcDeadlineWaypointNode,
)
from vbc_deadline_waypoint_sequential_impl import (
    DeadlineSequentialRollingVbcWaypointNode,
)
from vbc_multi_deadline_obligation_impl import (
    AccumulatedMultiDeadlineWaypointNode,
)
from vbc_visibility_acquisition_impl import (
    VisibilityAcquisitionWaypointNode,
)


def _env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    return float(raw)


class _OnlineRuntimeMixin:
    def _run_model_warmup(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        tic = time.perf_counter()
        with torch.enable_grad():
            x = torch.zeros((8, 3), device=self.device, dtype=torch.float32)
            q_center = 0.5 * (self.q_min + self.q_max)
            q = q_center.reshape(1, -1).repeat(8, 1)
            for _ in range(2):
                model_value_and_grad_q(x, q, self.model)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - tic) * 1000.0
        rospy.logwarn(
            "[vbc_waypoint_online] ONLINE WARMUP READY: %.2f ms device=%s batch=8 passes=2",
            elapsed_ms, str(self.device))

    def _selection_active_callback(self, msg):
        RollingVbcDeadlineWaypointNode._selection_active_callback(self, msg)
        if msg is not None and not bool(msg.data):
            rospy.loginfo_throttle(
                0.5,
                "[vbc_waypoint_online] steering inactive; mode-specific memory retained")

    def _generate_active_set_waypoint(
            self, points_xyz, trajectory, sweep_time_s, trajectory_received):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        tic = time.perf_counter()
        result = super()._generate_active_set_waypoint(
            points_xyz, trajectory, sweep_time_s, trajectory_received)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - tic) * 1000.0
        result["generation_compute_ms"] = float(elapsed_ms)
        result["generation_device"] = str(self.device)
        result["oracle_diagnostics_enabled"] = bool(
            self.enable_oracle_diagnostics)
        rospy.logwarn(
            "[vbc_waypoint_online] generation %.2f ms device=%s points=%d oracle_diag=%d mode=%s",
            elapsed_ms, str(self.device), int(result["active_set_size"]),
            int(self.enable_oracle_diagnostics), result["shared_solution_mode"])
        return result


class OnlinePersistentWaypointNode(
        _OnlineRuntimeMixin, PersistentRollingVbcDeadlineWaypointNode):
    def __init__(self):
        super().__init__()
        self._run_model_warmup()


class OnlineDeadlineSequentialWaypointNode(
        _OnlineRuntimeMixin, DeadlineSequentialRollingVbcWaypointNode):
    def __init__(self):
        super().__init__()
        self._run_model_warmup()


class OnlineAccumulatedMultiDeadlineWaypointNode(
        _OnlineRuntimeMixin, AccumulatedMultiDeadlineWaypointNode):
    def __init__(self):
        self._c46_ready = False
        super().__init__()
        self._c46_ready = True
        self._publish_schedule()
        self._run_model_warmup()

    def _process_new_active_set(self) -> None:
        if not self._c46_ready:
            return
        with self._obligation_lock:
            serial = self._raw_active_set_serial
            if serial == self._processed_active_set_serial:
                return
            raw = self._raw_active_set.copy()
        if raw.shape[0] == 0:
            with self._obligation_lock:
                self._processed_active_set_serial = max(
                    self._processed_active_set_serial, serial)
            return
        with self._lock:
            sweep = self._sweep_time_s
            trajectory, trajectory_received, trajectory_source = (
                self._preferred_trajectory_locked())
        if (sweep is None or trajectory is None or
                self._latest_measured_q is None):
            return

        regions = self._cluster_regions(raw)
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
                all_regions_handled = False
                rospy.logerr(
                    "[vbc_multi_deadline] obligation generation failed; will retry active set: %s",
                    exc)
                continue
            with self._obligation_lock:
                matched = self._match_existing(region)
                if (matched is None and
                        len(self._obligations) < self.max_obligations):
                    self._obligations.append(new_ob)
                    self._schedule_new_obligations += 1

        if all_regions_handled:
            with self._obligation_lock:
                self._processed_active_set_serial = max(
                    self._processed_active_set_serial, serial)
        self._publish_schedule()

    def _maybe_generate(self) -> None:
        if not self._c46_ready:
            return
        self._process_new_active_set()
        self._publish_schedule()

    def _publish_state(self) -> None:
        if not self._c46_ready:
            return
        AccumulatedMultiDeadlineWaypointNode._publish_state(self)


class OnlineVisibilityAcquisitionWaypointNode(
        _OnlineRuntimeMixin, VisibilityAcquisitionWaypointNode):
    def __init__(self):
        self._c46_ready = False
        super().__init__()
        self._c46_ready = True
        self._publish_schedule()
        self._run_model_warmup()

    def _process_new_active_set(self) -> None:
        return OnlineAccumulatedMultiDeadlineWaypointNode._process_new_active_set(self)


def _configure_runtime_mode(mode: str) -> None:
    mpc_prefix = "/velocity_qp_mpc_waypoint_node/mpc/visibility_waypoint"
    multi = mode == "accumulated_multi_deadline"
    rospy.set_param(mpc_prefix + "/multi_deadline_enabled", bool(multi))
    rospy.set_param(
        mpc_prefix + "/schedule_topic",
        "/care_planner/active_sensing/visibility_waypoint_schedule")
    rospy.set_param(mpc_prefix + "/max_repair_waypoints", 8)

    manager_prefix = "/c4_4_verified_regime_manager"
    acquisition = mode == "visibility_acquisition"
    rospy.set_param(
        manager_prefix + "/repair_completion_gate_enabled", bool(acquisition))
    rospy.set_param(
        manager_prefix + "/repair_completion_topic",
        "/care_planner/active_sensing/visibility_acquisition_complete")

    prefix_default = _env_bool("C4_REPAIR_PREFIX_VERIFY", False)
    prefix_s_default = _env_float("C4_REPAIR_PREFIX_S", 0.15)
    brake_dt_default = _env_float("C4_REPAIR_BRAKE_DT_S", 0.05)
    hold_default = _env_float("C4_REPAIR_HOLD_S", 0.10)
    repair_prefix_enabled = bool(rospy.get_param(
        "~repair_prefix_verification_enabled", prefix_default))
    repair_prefix_s = float(rospy.get_param(
        "~repair_execution_prefix_s", prefix_s_default))
    repair_brake_dt_s = float(rospy.get_param(
        "~repair_brake_dt_s", brake_dt_default))
    repair_hold_s = float(rospy.get_param(
        "~repair_hold_s", hold_default))

    continuity_prefix = "/optimized_trajectory_continuity"
    rospy.set_param(
        continuity_prefix + "/repair_prefix_verification_enabled",
        repair_prefix_enabled)
    rospy.set_param(
        continuity_prefix + "/repair_execution_prefix_s", repair_prefix_s)
    rospy.set_param(
        continuity_prefix + "/repair_brake_dt_s", repair_brake_dt_s)
    rospy.set_param(
        continuity_prefix + "/repair_hold_s", repair_hold_s)
    rospy.set_param(
        continuity_prefix + "/repair_active_topic",
        "/care_planner/execution/predicted_vbc_recovery_triggered")

    rospy.logwarn(
        "[vbc_waypoint_online] preconfigured runtime: multi_deadline=%d "
        "repair_completion_gate=%d repair_prefix_verify=%d prefix=%.3fs "
        "brake_dt=%.3fs hold=%.3fs",
        int(multi), int(acquisition), int(repair_prefix_enabled),
        repair_prefix_s, repair_brake_dt_s, repair_hold_s)


def main():
    rospy.init_node("vbc_deadline_waypoint_rolling")
    if not rospy.has_param("~use_active_set"):
        rospy.set_param("~use_active_set", True)

    mode = str(rospy.get_param(
        "~region_schedule_mode", "accumulated_multi_deadline")).strip().lower()
    if mode not in (
            "visibility_acquisition", "accumulated_multi_deadline",
            "deadline_sequential", "shared_persistent"):
        raise ValueError(
            "~region_schedule_mode must be visibility_acquisition, accumulated_multi_deadline, deadline_sequential, or shared_persistent")

    _configure_runtime_mode(mode)

    if mode == "visibility_acquisition":
        rospy.logwarn(
            "[vbc_waypoint_online] C4.7/C4.8 region_schedule_mode=visibility_acquisition")
        OnlineVisibilityAcquisitionWaypointNode()
    elif mode == "accumulated_multi_deadline":
        rospy.logwarn(
            "[vbc_waypoint_online] C4.6 region_schedule_mode=accumulated_multi_deadline")
        OnlineAccumulatedMultiDeadlineWaypointNode()
    elif mode == "deadline_sequential":
        rospy.logwarn(
            "[vbc_waypoint_online] C4.5 region_schedule_mode=deadline_sequential")
        OnlineDeadlineSequentialWaypointNode()
    else:
        rospy.logwarn(
            "[vbc_waypoint_online] BASELINE region_schedule_mode=shared_persistent")
        OnlinePersistentWaypointNode()
    rospy.spin()


if __name__ == "__main__":
    main()
