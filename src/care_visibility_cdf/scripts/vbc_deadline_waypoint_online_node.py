#!/usr/bin/env python3
"""Online C4.3 persistent waypoint generator with timing + startup warm-up."""

import time

import rospy
import torch

from evaluate_direct_vs_projection_ascent import model_value_and_grad_q
from vbc_deadline_waypoint_rolling_impl import RollingVbcDeadlineWaypointNode
from vbc_deadline_waypoint_rolling_persistent_impl import (
    PersistentRollingVbcDeadlineWaypointNode,
)


class OnlinePersistentWaypointNode(PersistentRollingVbcDeadlineWaypointNode):
    def __init__(self):
        super().__init__()
        self._run_model_warmup()

    def _run_model_warmup(self):
        """Pay first autograd / kernel initialization cost before online use."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        tic = time.perf_counter()
        with torch.enable_grad():
            x = torch.zeros((8, 3), device=self.device, dtype=torch.float32)
            q_center = 0.5 * (self.q_min + self.q_max)
            q = q_center.reshape(1, -1).repeat(8, 1)
            # Two passes cover the forward/autograd path used by projection and
            # remove the large first-call outlier from the execution window.
            for _ in range(2):
                model_value_and_grad_q(x, q, self.model)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - tic) * 1000.0
        rospy.logwarn(
            "[vbc_waypoint_online] ONLINE WARMUP READY: %.2f ms device=%s batch=8 passes=2",
            elapsed_ms, str(self.device))

    def _selection_active_callback(self, msg):
        # In continuous-planner mode, global-safe / steering-inactive is no
        # longer a measured-state replan boundary.  Let the base rolling class
        # deactivate the current waypoint, but deliberately bypass the old
        # Persistent implementation that cleared region + warm-start memory.
        RollingVbcDeadlineWaypointNode._selection_active_callback(self, msg)
        if msg is not None and not bool(msg.data):
            rospy.loginfo_throttle(
                0.5,
                "[vbc_waypoint_online] steering inactive; persistent/warm-start "
                "memory retained and retired only by future candidate active sets")

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
            "[vbc_waypoint_online] shared generation %.2f ms device=%s "
            "points=%d oracle_diag=%d mode=%s",
            elapsed_ms, str(self.device), int(result["active_set_size"]),
            int(self.enable_oracle_diagnostics),
            result["shared_solution_mode"])
        return result


def main():
    rospy.init_node("vbc_deadline_waypoint_rolling")
    if not rospy.has_param("~use_active_set"):
        rospy.set_param("~use_active_set", True)
    OnlinePersistentWaypointNode()
    rospy.spin()


if __name__ == "__main__":
    main()
