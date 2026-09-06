#!/usr/bin/env python3

import re
import threading

import rospy
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger


class InitialBodyPriorInitializer:
    """Initialize the trusted-free body prior exactly once.

    The confidence-map service still performs the geometric body-sphere marking,
    but this coordinator owns *when* it is allowed to happen:

      1. wait until the confidence-map service exists;
      2. retry while robot/link TFs are incomplete;
      3. accept the first refresh with zero skipped body samples;
      4. latch `ready=True` and never call the refresh service again.

    This preserves the intended bootstrap semantics: the trusted-free region is
    constructed around the robot's initial pose and then remains fixed in the
    world for the first task. The confidence-map node stores it in a separate
    provenance layer, so a later multi-goal/session transition can remove only
    the bootstrap while preserving all genuine sensor-derived observations.
    """

    _TRANSFORMED_RE = re.compile(r"transformed_samples=(\d+)")
    _SKIPPED_RE = re.compile(r"skipped_samples=(\d+)")
    _UPDATED_RE = re.compile(r"updated_cells=(\d+)")

    def __init__(self):
        self._lock = threading.Lock()
        self._ready = False
        self._attempts = 0

        self.refresh_service = str(
            rospy.get_param(
                "~refresh_service",
                "/care_planner/confidence_map/refresh_body_prior",
            )
        )
        self.ready_topic = str(
            rospy.get_param(
                "~ready_topic",
                "/care_planner/confidence_map/initial_prior_ready",
            )
        )
        self.retry_rate = float(rospy.get_param("~retry_rate", 20.0))
        self.wait_timeout = float(rospy.get_param("~service_wait_timeout", 0.05))

        if self.retry_rate <= 0.0:
            raise ValueError("~retry_rate must be positive")
        if self.wait_timeout < 0.0:
            raise ValueError("~service_wait_timeout must be non-negative")

        self.ready_pub = rospy.Publisher(
            self.ready_topic, Bool, queue_size=1, latch=True
        )
        self.summary_pub = rospy.Publisher("~summary", String, queue_size=1, latch=True)
        self._publish_ready(False)

        self._proxy = None
        self._timer = rospy.Timer(
            rospy.Duration(1.0 / self.retry_rate), self._timer_callback
        )

        rospy.logwarn(
            "[initial_body_prior] ONE-SHOT bootstrap enabled: wait for complete TF, "
            "mark initial trusted-free region once, then never move it."
        )
        rospy.loginfo(
            "[initial_body_prior] refresh_service=%s ready_topic=%s retry_rate=%.1fHz",
            self.refresh_service,
            self.ready_topic,
            self.retry_rate,
        )

    def _publish_ready(self, value):
        msg = Bool()
        msg.data = bool(value)
        self.ready_pub.publish(msg)

    @classmethod
    def _extract_int(cls, regex, text, default=-1):
        match = regex.search(text or "")
        if match is None:
            return default
        return int(match.group(1))

    def _timer_callback(self, _event):
        with self._lock:
            if self._ready:
                return

        try:
            rospy.wait_for_service(self.refresh_service, timeout=self.wait_timeout)
        except rospy.ROSException:
            rospy.logwarn_throttle(
                1.0,
                "[initial_body_prior] waiting for confidence-map refresh service",
            )
            return

        if self._proxy is None:
            self._proxy = rospy.ServiceProxy(self.refresh_service, Trigger)

        try:
            response = self._proxy()
        except rospy.ServiceException as exc:
            rospy.logwarn_throttle(
                1.0, "[initial_body_prior] refresh service call failed: %s", exc
            )
            self._proxy = None
            return

        with self._lock:
            self._attempts += 1
            attempts = self._attempts

        text = response.message or ""

        # Disabled prior is also a valid terminal state: there is simply no
        # bootstrap region to wait for.
        if response.success and "disabled" in text.lower():
            with self._lock:
                self._ready = True
            self._publish_ready(True)
            summary = String()
            summary.data = "ready=1 disabled=1 attempts={} message={}".format(
                attempts, text.replace(" ", "_")
            )
            self.summary_pub.publish(summary)
            self._timer.shutdown()
            rospy.logwarn("[initial_body_prior] prior disabled; bootstrap marked ready")
            return

        transformed = self._extract_int(self._TRANSFORMED_RE, text)
        skipped = self._extract_int(self._SKIPPED_RE, text)
        updated = self._extract_int(self._UPDATED_RE, text)

        if response.success and transformed > 0 and skipped == 0:
            with self._lock:
                self._ready = True
            self._publish_ready(True)
            summary = String()
            summary.data = (
                "ready=1 attempts={} transformed_samples={} skipped_samples={} "
                "updated_cells={}"
            ).format(attempts, transformed, skipped, updated)
            self.summary_pub.publish(summary)
            self._timer.shutdown()
            rospy.logwarn(
                "[initial_body_prior] INITIAL TRUSTED-FREE REGION LOCKED: "
                "transformed_samples=%d skipped_samples=%d updated_cells=%d attempts=%d. "
                "No further body-prior refreshes will be issued.",
                transformed,
                skipped,
                updated,
                attempts,
            )
            return

        rospy.logwarn_throttle(
            0.5,
            "[initial_body_prior] bootstrap not ready yet: success=%d "
            "transformed_samples=%d skipped_samples=%d attempts=%d",
            int(bool(response.success)),
            transformed,
            skipped,
            attempts,
        )


def main():
    rospy.init_node("initial_body_prior_initializer")
    InitialBodyPriorInitializer()
    rospy.spin()


if __name__ == "__main__":
    main()
