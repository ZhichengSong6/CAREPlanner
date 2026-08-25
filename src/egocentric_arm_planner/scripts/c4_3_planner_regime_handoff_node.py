#!/usr/bin/env python3
"""Collapse legacy RECOVERY_HOLD into a one-cycle planner-regime handoff.

The legacy waypoint MPC enters RECOVERY_HOLD after global VBC clear and waits
for a measured-state task replan.  Continuous CAREPlanner no longer replans the
high-level EE task at this boundary.  This adapter turns the legacy
recovery-complete pulse into an immediate replan-ready acknowledgement so the
MPC resumes optimization against the still-fresh task reference on its next
cycle.  The committed optimized-trajectory continuity node masks that internal
one-cycle transition from the low-level tracker.
"""

import rospy
from std_msgs.msg import Bool


class PlannerRegimeHandoffNode:
    def __init__(self):
        self.complete_topic = str(rospy.get_param(
            "~recovery_complete_topic",
            "/care_planner/execution/visibility_recovery_complete"))
        self.ready_topic = str(rospy.get_param(
            "~replan_ready_topic",
            "/care_planner/execution/visibility_replan_ready"))
        self.delay_s = float(rospy.get_param("~delay_s", 0.005))
        if self.delay_s < 0.0:
            raise ValueError("~delay_s must be non-negative")
        self.pub = rospy.Publisher(self.ready_topic, Bool, queue_size=1)
        self.sub = rospy.Subscriber(
            self.complete_topic, Bool, self._complete_cb, queue_size=1)
        self._pending_timer = None
        rospy.logwarn(
            "[c4_3_planner_handoff] continuous mode: %s -> %s delay=%.3fs",
            self.complete_topic, self.ready_topic, self.delay_s)

    def _publish_ready(self, _event=None):
        self.pub.publish(Bool(data=True))
        rospy.logwarn(
            "[c4_3_planner_handoff] planner regime READY: keep current task "
            "reference; no measured-state EE-goal replan")
        self._pending_timer = None

    def _complete_cb(self, msg):
        if msg is None or not bool(msg.data):
            return
        if self._pending_timer is not None:
            try:
                self._pending_timer.shutdown()
            except Exception:
                pass
        if self.delay_s <= 0.0:
            self._publish_ready()
        else:
            self._pending_timer = rospy.Timer(
                rospy.Duration(self.delay_s), self._publish_ready, oneshot=True)


def main():
    rospy.init_node("c4_3_planner_regime_handoff")
    PlannerRegimeHandoffNode()
    rospy.spin()


if __name__ == "__main__":
    main()
