#!/usr/bin/env python3
"""C4.3 rolling-target specialization of the validated VBC waypoint generator.

The learned projection/root/ascent implementation remains in
vbc_deadline_waypoint_node.py.  This node only changes target lifecycle:
  * a separate selection-active Bool explicitly turns steering on/off;
  * a changed target waits for its newly paired sweep time;
  * a fresh MPC predicted trajectory is preferred as the projection seed;
  * q_vis is generated once per selected target, not once per 20 Hz prediction;
  * releasing a target invalidates the cache so a later re-selection regenerates.

Recovery target locking is handled by phase_b2_controlled_trial_node.py; this
node therefore sees one stable target throughout Recovery/RECOVERY_HOLD.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import rospy
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool, Float64
from trajectory_msgs.msg import JointTrajectory

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from vbc_deadline_waypoint_node import (  # noqa: E402
    VbcDeadlineWaypointNode,
    _finite,
    _fmt,
    _vector_msg,
)


class RollingVbcDeadlineWaypointNode(VbcDeadlineWaypointNode):
    def __init__(self) -> None:
        # These defaults must exist before the base class creates its timer and
        # subscribes to overridden target callbacks.
        self._selection_active = False
        self._predicted_trajectory = None
        self._predicted_trajectory_received = None
        self._rolling_trajectory_source = "bootstrap"

        super().__init__()

        self.selection_active_topic = str(rospy.get_param(
            "~selection_active_topic",
            "/care_planner/active_sensing/target_selection_active"))
        self.predicted_trajectory_topic = str(rospy.get_param(
            "~predicted_trajectory_topic",
            "/care_planner/mpc/predicted_trajectory"))
        self.predicted_trajectory_timeout = float(rospy.get_param(
            "~predicted_trajectory_timeout", 0.20))
        if self.predicted_trajectory_timeout <= 0.0:
            raise ValueError("~predicted_trajectory_timeout must be positive")

        self.selection_active_sub = rospy.Subscriber(
            self.selection_active_topic, Bool,
            self._selection_active_callback, queue_size=1)
        self.predicted_trajectory_sub = rospy.Subscriber(
            self.predicted_trajectory_topic, JointTrajectory,
            self._predicted_trajectory_callback, queue_size=1)

        rospy.logwarn(
            "[vbc_waypoint_rolling] C4.3 mode: selection_active=%s predicted=%s timeout=%.3fs",
            self.selection_active_topic,
            self.predicted_trajectory_topic,
            self.predicted_trajectory_timeout)

    def _selection_active_callback(self, msg: Bool) -> None:
        if msg is None:
            return
        active = bool(msg.data)
        with self._lock:
            was_active = self._selection_active
            self._selection_active = active
            if not active:
                # A safe/no-target interval must explicitly disable the old
                # q_vis.  Clear the generation cache as well so selecting the
                # same voxel again later recomputes from the then-current state.
                self._generation_success = False
                self._generation_key = None
                self._summary = "selection_inactive"
            elif not was_active:
                self._seen_latched = False
                self._generation_key = None
                self._summary = "selection_active_waiting_pair"

    def _predicted_trajectory_callback(self, msg: JointTrajectory) -> None:
        if msg is None or not msg.points:
            return
        with self._lock:
            self._predicted_trajectory = msg
            self._predicted_trajectory_received = rospy.Time.now()

    def _target_callback(self, msg: PointStamped) -> None:
        if msg is None:
            return
        xyz = self._target_array(msg)
        if not _finite(xyz):
            return
        with self._lock:
            old_xyz = None if self._target_xyz is None else self._target_xyz.copy()
        super()._target_callback(msg)
        changed = old_xyz is None or np.linalg.norm(xyz - old_xyz) > self.target_change_tolerance
        if changed:
            with self._lock:
                # The broker publishes target first, then its paired sweep.
                # Clearing here prevents a one-cycle old-target/new-target mix.
                self._sweep_time_s = None
                self._summary = "new_rolling_target_waiting_sweep"
            rospy.logwarn("[vbc_waypoint_rolling] new target x*=%s", _fmt(xyz, 6))

    def _preferred_trajectory_locked(self):
        now = rospy.Time.now()
        if (self._predicted_trajectory is not None and
                self._predicted_trajectory_received is not None):
            age = (now - self._predicted_trajectory_received).to_sec()
            if 0.0 <= age <= self.predicted_trajectory_timeout:
                return (
                    self._predicted_trajectory,
                    self._predicted_trajectory_received,
                    "predicted")
        return self._trajectory, self._trajectory_received, "bootstrap"

    def _maybe_generate(self) -> None:
        with self._lock:
            selection_active = self._selection_active
            target = self._target
            target_xyz = None if self._target_xyz is None else self._target_xyz.copy()
            sweep = self._sweep_time_s
            old_key = self._generation_key
            trajectory, trajectory_received, trajectory_source = (
                self._preferred_trajectory_locked())

        if not selection_active:
            return
        if target is None or target_xyz is None or trajectory is None or sweep is None:
            return

        # Rolling predictions are stamped every control cycle.  Regenerating
        # q_vis on every stamp is unnecessary and expensive; q_vis is a
        # sufficient visible configuration for x*.  Recompute only when the
        # selected spatial target changes (or after selection was released).
        key = ("rolling", tuple(np.round(target_xyz, 7).tolist()))
        if old_key == key:
            return

        try:
            result = self._generate_waypoint(
                target, trajectory, sweep, trajectory_received)
        except Exception as exc:
            with self._lock:
                self._generation_key = key
                self._generation_success = False
                self._summary = "generation_failed:" + str(exc).replace(" ", "_")
            rospy.logerr("[vbc_waypoint_rolling] generation failed: %s", exc)
            return

        q_zero = np.asarray(result["q_zero"], dtype=np.float64)
        q_vis = np.asarray(result["q_vis"], dtype=np.float64)
        deadline_abs = float(result["deadline_absolute_ros_s"])
        deadline_from_start = float(result["deadline_from_start_s"])

        result["rolling_trajectory_source"] = trajectory_source
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.output_root / f"vbc_visibility_waypoint_{stamp}.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=True))

        with self._lock:
            self._generation_key = key
            self._q_zero = q_zero
            self._q_vis = q_vis
            self._deadline_abs_s = deadline_abs
            self._deadline_from_start_s = deadline_from_start
            self._generation_success = True
            self._rolling_trajectory_source = trajectory_source
            self._summary = "ready"

        self.zero_pub.publish(_vector_msg(q_zero))
        self.waypoint_pub.publish(_vector_msg(q_vis))
        dmsg = Float64(); dmsg.data = deadline_abs; self.deadline_pub.publish(dmsg)

        rospy.logwarn(
            "[vbc_waypoint_rolling] WAYPOINT READY source=%s x*=%s sweep=%.3f q_vis=%s",
            trajectory_source, _fmt(result["target_xyz"], 6), sweep, _fmt(q_vis, 5))
        rospy.logwarn("[vbc_waypoint_rolling] saved generation trace: %s", path)

    def _publish_state(self) -> None:
        with self._lock:
            selection_active = self._selection_active
            target = self._target
            generation_success = self._generation_success
            q_vis = None if self._q_vis is None else self._q_vis.copy()
            deadline_abs = self._deadline_abs_s
            deadline_from_start = self._deadline_from_start_s
            seen = self._seen_latched
            summary_reason = self._summary
            trajectory_source = self._rolling_trajectory_source

        if not selection_active:
            amsg = Bool(); amsg.data = False; self.active_pub.publish(amsg)
            msg = self._summary_message(
                active=False,
                seen=seen,
                ready=False,
                confidence=math.nan,
                current_visibility=math.nan,
                inside=False,
                deadline_from_start=deadline_from_start,
                deadline_abs=deadline_abs,
                reason=summary_reason,
                trajectory_source=trajectory_source)
            self.summary_pub.publish(msg)
            return

        confidence = math.nan
        current_visibility = math.nan
        inside = False
        if target is not None:
            response = self._query_confidence(target)
            if response is not None:
                confidence, current_visibility, inside = response
                if inside and confidence >= self.seen_threshold and not seen:
                    with self._lock:
                        self._seen_latched = True
                    seen = True
                    rospy.logwarn(
                        "[vbc_waypoint_rolling] target CONFIRMED SEEN -> waypoint objective OFF: confidence=%.4f",
                        confidence)

        active = bool(
            selection_active and generation_success and q_vis is not None and
            deadline_abs is not None and not seen)
        amsg = Bool(); amsg.data = active; self.active_pub.publish(amsg)
        if q_vis is not None:
            self.waypoint_pub.publish(_vector_msg(q_vis))
        if deadline_abs is not None:
            dmsg = Float64(); dmsg.data = float(deadline_abs); self.deadline_pub.publish(dmsg)

        msg = self._summary_message(
            active=active,
            seen=seen,
            ready=generation_success,
            confidence=confidence,
            current_visibility=current_visibility,
            inside=inside,
            deadline_from_start=deadline_from_start,
            deadline_abs=deadline_abs,
            reason=summary_reason,
            trajectory_source=trajectory_source)
        self.summary_pub.publish(msg)
        rospy.loginfo_throttle(0.5, "[vbc_waypoint_rolling] %s", msg.data)

    @staticmethod
    def _summary_message(
            active, seen, ready, confidence, current_visibility, inside,
            deadline_from_start, deadline_abs, reason, trajectory_source):
        from std_msgs.msg import String
        now = rospy.Time.now().to_sec()
        remaining = math.nan if deadline_abs is None else deadline_abs - now
        msg = String()
        msg.data = (
            f"active={int(active)} seen={int(seen)} ready={int(ready)} "
            f"selection_active={int(active or ready)} "
            f"confidence={confidence:.4f} current_visibility={current_visibility:.4f} "
            f"inside_map={int(inside)} "
            f"deadline_from_start={deadline_from_start if deadline_from_start is not None else math.nan:.6f} "
            f"deadline_remaining={remaining:.6f} "
            f"trajectory_source={trajectory_source} reason={reason}"
        )
        return msg


def main() -> None:
    rospy.init_node("vbc_deadline_waypoint_rolling")
    try:
        RollingVbcDeadlineWaypointNode()
    except Exception as exc:
        rospy.logfatal("[vbc_waypoint_rolling] initialization failed: %s", exc)
        raise
    rospy.spin()


if __name__ == "__main__":
    main()
