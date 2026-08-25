#!/usr/bin/env python3
"""C4.3 entry point for the rolling visibility waypoint generator.

The default C4.3 path uses all-point shared steering plus short-horizon
spatial-region persistence and same-patch q_vis warm starts.  Passing
``_use_active_set:=false`` still retains the previous single-target rolling
ablation.
"""

import rospy

from vbc_deadline_waypoint_rolling_persistent_impl import (
    PersistentRollingVbcDeadlineWaypointNode,
)


def main() -> None:
    rospy.init_node("vbc_deadline_waypoint_rolling")
    if not rospy.has_param("~use_active_set"):
        rospy.set_param("~use_active_set", True)
    try:
        PersistentRollingVbcDeadlineWaypointNode()
    except Exception as exc:
        rospy.logfatal(
            "[vbc_waypoint_rolling] initialization failed: %s", exc)
        raise
    rospy.spin()


if __name__ == "__main__":
    main()
