#!/usr/bin/env python3
"""Online CAREPlanner visibility waypoint generator.

Modes:
  deadline_sequential (C4.5 default)
      Track spatial VBC regions persistently, but expose only one unresolved
      region at a time to learned q_vis generation.

  shared_persistent (C4.4 baseline)
      Union all remembered spatial regions and solve for one shared q_vis.

Both modes keep the same learned projection/root/ascent machinery and the same
exact downstream VBC verification.  This makes the scheduler change directly
comparable without also changing the learned field or safety semantics.
"""

import time

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


class _OnlineRuntimeMixin:
    def _run_model_warmup(self):
        """Pay first autograd / kernel initialization cost before online use."""
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
        # In planner-mode semantics, a brief globally-safe/probe interval must
        # not erase region identity.  Exact future VBC active sets retire solved
        # regions.  Bypass Persistent/Sequential episode-reset callbacks here.
        RollingVbcDeadlineWaypointNode._selection_active_callback(self, msg)
        if msg is not None and not bool(msg.data):
            rospy.loginfo_throttle(
                0.5,
                "[vbc_waypoint_online] steering inactive; region/warm-start "
                "memory retained for future exact-VBC updates")

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
            "[vbc_waypoint_online] generation %.2f ms device=%s "
            "points=%d oracle_diag=%d mode=%s",
            elapsed_ms, str(self.device), int(result["active_set_size"]),
            int(self.enable_oracle_diagnostics),
            result["shared_solution_mode"])
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


def main():
    rospy.init_node("vbc_deadline_waypoint_rolling")
    if not rospy.has_param("~use_active_set"):
        rospy.set_param("~use_active_set", True)

    mode = str(rospy.get_param(
        "~region_schedule_mode", "deadline_sequential")).strip().lower()
    if mode == "deadline_sequential":
        rospy.logwarn(
            "[vbc_waypoint_online] C4.5 region_schedule_mode=deadline_sequential")
        OnlineDeadlineSequentialWaypointNode()
    elif mode == "shared_persistent":
        rospy.logwarn(
            "[vbc_waypoint_online] BASELINE region_schedule_mode=shared_persistent")
        OnlinePersistentWaypointNode()
    else:
        raise ValueError(
            "~region_schedule_mode must be deadline_sequential or shared_persistent")
    rospy.spin()


if __name__ == "__main__":
    main()
