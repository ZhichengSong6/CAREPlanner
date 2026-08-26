#!/usr/bin/env python3
"""Online CAREPlanner visibility waypoint generator.

Modes:
  accumulated_multi_deadline (C4.6 default)
      Accumulate spatial visibility obligations across rejected candidates,
      generate each q_vis from measured q, and publish a multi-deadline schedule.

  deadline_sequential (C4.5 baseline)
      Keep one spatial region at a time across planner cycles.

  shared_persistent (C4.4 baseline)
      Union remembered regions and solve for one shared q_vis.

All modes keep exact candidate VBC downstream as the hard commit authority.
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
from vbc_multi_deadline_obligation_impl import (
    AccumulatedMultiDeadlineWaypointNode,
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
        # not erase steering memory. C4.6 clears accumulated obligations only on
        # an exact predicted-VBC SAFE verdict.
        RollingVbcDeadlineWaypointNode._selection_active_callback(self, msg)
        if msg is not None and not bool(msg.data):
            rospy.loginfo_throttle(
                0.5,
                "[vbc_waypoint_online] steering inactive; memory retained until exact predicted-VBC SAFE")

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
        super().__init__()
        self._run_model_warmup()


def main():
    rospy.init_node("vbc_deadline_waypoint_rolling")
    if not rospy.has_param("~use_active_set"):
        rospy.set_param("~use_active_set", True)

    mode = str(rospy.get_param(
        "~region_schedule_mode", "accumulated_multi_deadline")).strip().lower()
    if mode == "accumulated_multi_deadline":
        rospy.logwarn(
            "[vbc_waypoint_online] C4.6 region_schedule_mode=accumulated_multi_deadline")
        OnlineAccumulatedMultiDeadlineWaypointNode()
    elif mode == "deadline_sequential":
        rospy.logwarn(
            "[vbc_waypoint_online] C4.5 region_schedule_mode=deadline_sequential")
        OnlineDeadlineSequentialWaypointNode()
    elif mode == "shared_persistent":
        rospy.logwarn(
            "[vbc_waypoint_online] BASELINE region_schedule_mode=shared_persistent")
        OnlinePersistentWaypointNode()
    else:
        raise ValueError(
            "~region_schedule_mode must be accumulated_multi_deadline, deadline_sequential, or shared_persistent")
    rospy.spin()


if __name__ == "__main__":
    main()
