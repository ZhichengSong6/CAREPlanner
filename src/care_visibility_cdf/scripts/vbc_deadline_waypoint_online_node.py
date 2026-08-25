#!/usr/bin/env python3
"""Online C4.3 persistent waypoint generator with runtime timing diagnostics."""

import time

import rospy
import torch

from vbc_deadline_waypoint_rolling_persistent_impl import (
    PersistentRollingVbcDeadlineWaypointNode,
)


class OnlinePersistentWaypointNode(PersistentRollingVbcDeadlineWaypointNode):
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
