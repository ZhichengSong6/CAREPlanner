#!/usr/bin/env python3

import threading

import rospy
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import Bool, String


class PhaseB2ControlledTrialNode:
    """Publish one fixed EE goal and freeze the first trajectory-risk target.

    The node is intentionally simple and deterministic:
      1. wait until the one-shot initial trusted-free prior is ready;
      2. publish exactly one latched EE goal after the planner subscriber exists;
      3. wait for the first active-sensing candidate produced from the nominal
         task trajectory;
      4. freeze that candidate for the rest of the trial;
      5. republish the frozen target with a fresh timestamp at a fixed rate so
         the NCDF observer never treats it as stale.

    The trajectory-risk node is expected to publish candidates on a separate
    topic, typically /care_planner/active_sensing/target_candidate.  The frozen
    target is republished on /care_planner/active_sensing/target_point.
    """

    def __init__(self):
        self._lock = threading.Lock()

        self.base_frame = str(rospy.get_param("~base_frame", "base_link"))
        self.goal_topic = str(
            rospy.get_param("~goal_topic", "/care_planner/ee_target_pose")
        )
        self.candidate_topic = str(
            rospy.get_param(
                "~candidate_topic",
                "/care_planner/active_sensing/target_candidate",
            )
        )
        self.frozen_target_topic = str(
            rospy.get_param(
                "~frozen_target_topic",
                "/care_planner/active_sensing/target_point",
            )
        )
        self.require_initial_prior_ready = bool(
            rospy.get_param("~require_initial_prior_ready", True)
        )
        self.initial_prior_ready_topic = str(
            rospy.get_param(
                "~initial_prior_ready_topic",
                "/care_planner/confidence_map/initial_prior_ready",
            )
        )
        self.publish_rate = float(rospy.get_param("~publish_rate", 20.0))
        self.goal_delay = float(rospy.get_param("~goal_delay", 0.5))

        self.goal_x = float(rospy.get_param("~goal/x", 0.286881))
        self.goal_y = float(rospy.get_param("~goal/y", 0.0))
        self.goal_z = float(rospy.get_param("~goal/z", 0.560765))
        self.goal_qx = float(rospy.get_param("~goal/qx", 0.0))
        self.goal_qy = float(rospy.get_param("~goal/qy", 0.0))
        self.goal_qz = float(rospy.get_param("~goal/qz", 0.70710678))
        self.goal_qw = float(rospy.get_param("~goal/qw", 0.70710678))

        if self.publish_rate <= 0.0:
            raise ValueError("publish_rate must be positive")
        if self.goal_delay < 0.0:
            raise ValueError("goal_delay must be non-negative")

        self._startup_time = rospy.Time.now()
        self._goal_sent = False
        self._frozen_target = None
        self._initial_prior_ready = not self.require_initial_prior_ready

        # Latch the fixed goal so a late planner subscriber still receives the
        # exact same single target pose.  The planner therefore gets one fixed
        # EE goal per trial instead of repeated interactive-marker updates.
        self.goal_pub = rospy.Publisher(
            self.goal_topic, PoseStamped, queue_size=1, latch=True
        )
        self.target_pub = rospy.Publisher(
            self.frozen_target_topic, PointStamped, queue_size=1
        )
        self.frozen_pub = rospy.Publisher("~target_frozen", Bool, queue_size=1, latch=True)
        self.summary_pub = rospy.Publisher("~summary", String, queue_size=1, latch=True)

        self.candidate_sub = rospy.Subscriber(
            self.candidate_topic,
            PointStamped,
            self._candidate_callback,
            queue_size=1,
        )
        self.initial_prior_ready_sub = rospy.Subscriber(
            self.initial_prior_ready_topic,
            Bool,
            self._initial_prior_ready_callback,
            queue_size=1,
        )

        self.goal_timer = rospy.Timer(rospy.Duration(0.05), self._goal_timer_callback)
        self.target_timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate), self._target_timer_callback
        )

        self._publish_frozen_state(False)
        rospy.logwarn(
            "[phase_b2_trial] CONTROLLED A/B MODE: fixed EE goal + first risk target frozen"
        )
        rospy.loginfo(
            "[phase_b2_trial] fixed goal p=[%.6f, %.6f, %.6f] q=[%.8f, %.8f, %.8f, %.8f]",
            self.goal_x,
            self.goal_y,
            self.goal_z,
            self.goal_qx,
            self.goal_qy,
            self.goal_qz,
            self.goal_qw,
        )
        rospy.loginfo(
            "[phase_b2_trial] candidate=%s frozen_target=%s initial_prior_ready=%s require=%d",
            self.candidate_topic,
            self.frozen_target_topic,
            self.initial_prior_ready_topic,
            int(self.require_initial_prior_ready),
        )

    def _publish_frozen_state(self, frozen):
        msg = Bool()
        msg.data = bool(frozen)
        self.frozen_pub.publish(msg)

    def _initial_prior_ready_callback(self, msg):
        if msg is None or not bool(msg.data):
            return
        with self._lock:
            first = not self._initial_prior_ready
            self._initial_prior_ready = True
        if first:
            rospy.logwarn(
                "[phase_b2_trial] initial trusted-free prior READY; controlled motion may begin"
            )

    def _goal_timer_callback(self, _event):
        with self._lock:
            if self._goal_sent:
                return

            if not self._initial_prior_ready:
                rospy.logwarn_throttle(
                    1.0,
                    "[phase_b2_trial] waiting for one-shot initial trusted-free prior",
                )
                return

            elapsed = (rospy.Time.now() - self._startup_time).to_sec()
            if elapsed < self.goal_delay:
                return

            # Prefer waiting for a subscriber so the planning attempt starts
            # after the rest of the launch is up.  Because the publisher is
            # latched, the message is still safe if the subscriber connects
            # immediately after this callback.
            if self.goal_pub.get_num_connections() <= 0:
                rospy.logwarn_throttle(
                    1.0, "[phase_b2_trial] waiting for EE-goal subscriber"
                )
                return

            goal = PoseStamped()
            goal.header.stamp = rospy.Time.now()
            goal.header.frame_id = self.base_frame
            goal.pose.position.x = self.goal_x
            goal.pose.position.y = self.goal_y
            goal.pose.position.z = self.goal_z
            goal.pose.orientation.x = self.goal_qx
            goal.pose.orientation.y = self.goal_qy
            goal.pose.orientation.z = self.goal_qz
            goal.pose.orientation.w = self.goal_qw
            self.goal_pub.publish(goal)
            self._goal_sent = True

            rospy.logwarn(
                "[phase_b2_trial] FIXED EE GOAL PUBLISHED exactly once: "
                "p=[%.6f, %.6f, %.6f]",
                self.goal_x,
                self.goal_y,
                self.goal_z,
            )

    def _candidate_callback(self, msg):
        if msg is None:
            return

        with self._lock:
            # Ignore anything received before this trial's fixed goal is sent.
            if not self._goal_sent or self._frozen_target is not None:
                return

            if msg.header.frame_id and msg.header.frame_id.lstrip("/") != self.base_frame.lstrip("/"):
                rospy.logwarn_throttle(
                    1.0,
                    "[phase_b2_trial] ignoring candidate in frame '%s' (expected '%s')",
                    msg.header.frame_id,
                    self.base_frame,
                )
                return

            frozen = PointStamped()
            frozen.header.frame_id = self.base_frame
            frozen.header.stamp = rospy.Time.now()
            frozen.point.x = msg.point.x
            frozen.point.y = msg.point.y
            frozen.point.z = msg.point.z
            self._frozen_target = frozen

            self._publish_frozen_state(True)
            summary = String()
            summary.data = (
                "frozen_target=[{:.9f},{:.9f},{:.9f}] frame={}"
            ).format(
                frozen.point.x,
                frozen.point.y,
                frozen.point.z,
                self.base_frame,
            )
            self.summary_pub.publish(summary)

            rospy.logwarn(
                "[phase_b2_trial] ACTIVE-SENSING TARGET FROZEN for entire trial: "
                "x=%.9f y=%.9f z=%.9f",
                frozen.point.x,
                frozen.point.y,
                frozen.point.z,
            )

    def _target_timer_callback(self, _event):
        with self._lock:
            if self._frozen_target is None:
                return
            msg = PointStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = self.base_frame
            msg.point.x = self._frozen_target.point.x
            msg.point.y = self._frozen_target.point.y
            msg.point.z = self._frozen_target.point.z

        self.target_pub.publish(msg)


def main():
    rospy.init_node("phase_b2_controlled_trial")
    PhaseB2ControlledTrialNode()
    rospy.spin()


if __name__ == "__main__":
    main()
