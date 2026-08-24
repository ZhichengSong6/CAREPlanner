#!/usr/bin/env python3
"""C4.3 entry point for the rolling visibility waypoint generator.

The implementation lives in ``vbc_deadline_waypoint_rolling_impl.py``.  C4.3
now defaults to all-point active-set steering.  Passing ``_use_active_set:=false``
explicitly retains the previous single-target rolling ablation.
"""

import rospy

from vbc_deadline_waypoint_rolling_impl import RollingVbcDeadlineWaypointNode


def main() -> None:
    rospy.init_node("vbc_deadline_waypoint_rolling")
    if not rospy.has_param("~use_active_set"):
        rospy.set_param("~use_active_set", True)
    try:
        RollingVbcDeadlineWaypointNode()
    except Exception as exc:
        rospy.logfatal(
            "[vbc_waypoint_rolling] initialization failed: %s", exc)
        raise
    rospy.spin()


if __name__ == "__main__":
    main()
