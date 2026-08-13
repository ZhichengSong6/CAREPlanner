#!/usr/bin/env python3
"""Stable entry point for the Stage-III NCDF trajectory QP.

This wrapper sanitizes obviously non-physical JointState velocities before they
enter the Stage-III first-step acceleration constraint.  The raw JointState
message is not modified; only the velocity used by the optimizer is guarded.
"""

import numpy as np
import rospy

from ncdf_stage3_trajectory_qp_runtime import NcdfStage3TrajectoryQPRuntimeNode


class NcdfStage3SafeRuntimeNode(NcdfStage3TrajectoryQPRuntimeNode):
    """Runtime Stage-III node with a defensive measured-velocity sanity guard."""

    _VELOCITY_SANITY_FACTOR = 1.5

    def _ordered_joint_state(self, msg):
        index = {name: i for i, name in enumerate(msg.name)}
        if any(name not in index for name in self.joint_names):
            return None

        q = np.zeros(7, dtype=np.float64)
        dq_raw = np.zeros(7, dtype=np.float64)
        for i, name in enumerate(self.joint_names):
            j = index[name]
            if j >= len(msg.position):
                return None
            q[i] = float(msg.position[j])
            if j < len(msg.velocity):
                dq_raw[i] = float(msg.velocity[j])

        if not np.all(np.isfinite(q)):
            return None

        limits = np.asarray(self.velocity_limits, dtype=np.float64).reshape(7)
        sanity_limits = self._VELOCITY_SANITY_FACTOR * limits
        nonphysical = (~np.isfinite(dq_raw)) | (np.abs(dq_raw) > sanity_limits)

        # Mild over-limit measurements are clipped.  Grossly non-physical
        # measurements (the Gazebo spikes observed at tens of rad/s) are replaced
        # by zero rather than by the velocity limit, because feeding a saturated
        # 2--2.5 rad/s value into the first-step acceleration constraint can still
        # make the QP infeasible when the arm is actually nearly stationary.
        dq = np.nan_to_num(dq_raw, nan=0.0, posinf=0.0, neginf=0.0)
        dq = np.clip(dq, -limits, limits)
        dq[nonphysical] = 0.0

        changed = nonphysical | (np.abs(dq_raw) > limits)
        if np.any(changed):
            details = ", ".join(
                f"{self.joint_names[i]}:raw={dq_raw[i]:+.3f}->used={dq[i]:+.3f},limit={limits[i]:.3f}"
                for i in np.flatnonzero(changed)
            )
            rospy.logwarn_throttle(
                0.5,
                "[ncdf_stage3] sanitized JointState velocity before QP: %s",
                details,
            )

        return q, dq


def main():
    rospy.init_node("ncdf_stage3_trajectory_qp")
    NcdfStage3SafeRuntimeNode()
    rospy.spin()


if __name__ == "__main__":
    main()
