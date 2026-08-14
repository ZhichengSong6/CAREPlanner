#!/usr/bin/env python3

import subprocess
import threading
import time

import rospy
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory


class AutoKillPlannerAfterReference:
    def __init__(self):
        self.reference_topic = rospy.get_param(
            "~reference_topic", "/care_planner/command_trajectory_persistent"
        )
        self.planner_node = rospy.get_param(
            "~planner_node", "/receding_horizon_planner_node"
        )
        self.kill_delay = float(rospy.get_param("~kill_delay", 0.05))
        self.triggered = False
        self.lock = threading.Lock()

        self.event_pub = rospy.Publisher(
            "/care_planner/experiment/planner_killed", Bool, queue_size=1, latch=True
        )
        self.sub = rospy.Subscriber(
            self.reference_topic,
            JointTrajectory,
            self.reference_callback,
            queue_size=1,
        )

        rospy.loginfo(
            "[AutoKillPlanner] Waiting for first persistent reference on %s; "
            "will kill %s after %.3f s",
            self.reference_topic,
            self.planner_node,
            self.kill_delay,
        )

    def reference_callback(self, msg):
        if not msg.points:
            return

        with self.lock:
            if self.triggered:
                return
            self.triggered = True

        rospy.loginfo(
            "[AutoKillPlanner] First persistent reference received: points=%d, "
            "duration=%.6f s",
            len(msg.points),
            msg.points[-1].time_from_start.to_sec(),
        )

        # Run the kill outside the ROS callback thread so the subscriber can
        # return immediately. The small deterministic delay gives the first
        # persistent reference time to reach the MPC before the legacy planner
        # is removed from the experiment.
        thread = threading.Thread(target=self.kill_planner, daemon=True)
        thread.start()

    def kill_planner(self):
        if self.kill_delay > 0.0:
            rospy.sleep(self.kill_delay)

        rospy.logwarn("[AutoKillPlanner] Killing %s now.", self.planner_node)
        try:
            proc = subprocess.run(
                ["rosnode", "kill", self.planner_node],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=5.0,
                check=False,
            )
            output = (proc.stdout or "").strip()
            if output:
                rospy.loginfo("[AutoKillPlanner] rosnode output: %s", output)

            success = proc.returncode == 0
            self.event_pub.publish(Bool(data=success))
            if success:
                rospy.logwarn(
                    "[AutoKillPlanner] Planner killed successfully. Persistent "
                    "relay + PIQP MPC remain active."
                )
            else:
                rospy.logerr(
                    "[AutoKillPlanner] rosnode kill failed with return code %d",
                    proc.returncode,
                )
        except Exception as exc:  # pylint: disable=broad-except
            self.event_pub.publish(Bool(data=False))
            rospy.logerr("[AutoKillPlanner] Failed to kill planner: %s", exc)

        # Keep the latched experiment-event publisher alive briefly so rosbag
        # reliably receives the event before this helper exits.
        time.sleep(0.25)
        rospy.signal_shutdown("auto-kill experiment action complete")


def main():
    rospy.init_node("auto_kill_planner_after_reference")
    AutoKillPlannerAfterReference()
    rospy.spin()


if __name__ == "__main__":
    main()
