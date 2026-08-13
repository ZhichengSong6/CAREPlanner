#!/usr/bin/env python3
"""Runtime-fixed Stage-III node.

ROS Python message sequence fields can be tuples depending on the generated
message/runtime path.  The Stage-III core computes the QP correctly, but its
trajectory writer originally tried to mutate point.positions/velocities in
place.  This subclass keeps the core formulation unchanged and rewrites those
fields through mutable lists before assigning them back to the ROS message.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import rospy

from ncdf_stage3_trajectory_qp_core import NcdfStage3TrajectoryQPNode


class NcdfStage3TrajectoryQPRuntimeNode(NcdfStage3TrajectoryQPNode):
    def _apply_correction(self, traj, mapping, delta_q, delta_u):
        corrected = copy.deepcopy(traj)
        K = self.num_intervals
        dt = self.stage_dt

        for point in corrected.points:
            t = point.time_from_start.to_sec()
            if t <= 0.0 or t >= self.horizon_duration:
                dq_corr = np.zeros(7, dtype=np.float64)
                du_corr = np.zeros(7, dtype=np.float64)
            else:
                k = min(K - 1, int(math.floor(t / dt)))
                alpha = float((t - k * dt) / dt)
                dq_corr = (
                    (1.0 - alpha) * delta_q[:, k]
                    + alpha * delta_q[:, k + 1]
                )
                du_corr = delta_u[:, k]

            # rospy sequence fields may be tuples.  Convert to mutable lists,
            # modify them, then assign the whole sequence back to the message.
            positions = list(point.positions)
            for i, src in enumerate(mapping):
                if src >= len(positions):
                    raise RuntimeError(
                        f"trajectory point has no position index {src}"
                    )
                positions[src] = float(positions[src] + dq_corr[i])
            point.positions = positions

            if point.velocities:
                velocities = list(point.velocities)
                for i, src in enumerate(mapping):
                    if src < len(velocities):
                        velocities[src] = float(velocities[src] + du_corr[i])
                point.velocities = velocities

        return corrected


def main():
    rospy.init_node("ncdf_stage3_trajectory_qp")
    NcdfStage3TrajectoryQPRuntimeNode()
    rospy.spin()
