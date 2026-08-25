#!/usr/bin/env python3
"""C4.3 controlled-trial broker for continuous CAREPlanner execution.

This preserves the validated pre-T0 goal / target pairing logic from
PhaseB2ControlledTrialNode but removes the old emergency-controller handoff:
Recovery completion does not re-publish the EE goal, does not request a
measured-state task replan, and does not hard-lock the rolling target lifecycle.
The nominal task reference remains an upstream objective while CAREPlanner's
optimized trajectory is continuously tracked downstream.

catkin_install_python creates relay wrappers in devel/lib.  Importing
``phase_b2_controlled_trial_node`` by module name from another installed script
therefore resolves to the relay wrapper rather than to the source file that
contains PhaseB2ControlledTrialNode.  Load the source module explicitly from the
ROS package path so this subclass works both from source and from catkin's devel
space.
"""

from pathlib import Path
import importlib.util

import rospkg
import rospy


def _load_base_class():
    package_dir = Path(rospkg.RosPack().get_path("egocentric_arm_planner")).resolve()
    source_path = package_dir / "scripts" / "phase_b2_controlled_trial_node.py"
    if not source_path.is_file():
        raise ImportError(
            "cannot locate PhaseB2ControlledTrialNode source at {}".format(source_path)
        )
    spec = importlib.util.spec_from_file_location(
        "care_phase_b2_controlled_trial_source", str(source_path))
    if spec is None or spec.loader is None:
        raise ImportError("failed to create module spec for {}".format(source_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "PhaseB2ControlledTrialNode"):
        raise ImportError(
            "PhaseB2ControlledTrialNode missing from {}".format(source_path))
    return module.PhaseB2ControlledTrialNode


PhaseB2ControlledTrialNode = _load_base_class()


class PhaseC43ContinuousTrialNode(PhaseB2ControlledTrialNode):
    def _recovery_active_callback(self, msg):
        if msg is None or not self.rolling_target_mode:
            return
        active = bool(msg.data)
        with self._lock:
            was_active = self._recovery_episode_active
            self._recovery_episode_active = active
            # Continuous planner mode never freezes the target lifecycle merely
            # because the planner changed optimization regime.
            self._target_lock = False
            if active:
                self._safe_clear_streak = 0
                if self._selected_target is not None and self._candidate_active:
                    self._set_selected_active_locked(True)
                else:
                    self._publish_summary_locked()
            else:
                self._publish_frozen_state(False)
                self._publish_summary_locked()

        if active and not was_active:
            rospy.logwarn(
                "[phase_c4_3_trial] high-visibility planning regime ACTIVE; "
                "rolling target updates remain enabled")
        elif was_active and not active:
            rospy.logwarn(
                "[phase_c4_3_trial] high-visibility regime cleared; "
                "continuing CAREPlanner trajectory without task replan")

    def _recovery_complete_callback(self, msg):
        if msg is None or not bool(msg.data):
            return
        # Legacy MPC still emits this transition pulse internally.  In the
        # continuous planner architecture it is *not* a request to publish the
        # EE goal again or replace the current optimized trajectory.
        with self._lock:
            self._target_lock = False
            self._recovery_episode_active = False
            self._publish_frozen_state(False)
            self._publish_summary_locked()
        rospy.logwarn(
            "[phase_c4_3_trial] planner regime transition complete; "
            "measured-state EE-goal replan intentionally SKIPPED")

    def _replan_ready_callback(self, msg):
        if msg is None or not bool(msg.data) or not self.rolling_target_mode:
            return
        with self._lock:
            self._target_lock = False
            self._recovery_episode_active = False
            self._safe_clear_streak = 0
            self._publish_frozen_state(False)
            self._publish_summary_locked()
        rospy.loginfo(
            "[phase_c4_3_trial] continuous planner handoff ready; "
            "rolling selection remains unlocked")


def main():
    rospy.init_node("phase_b2_controlled_trial")
    PhaseC43ContinuousTrialNode()
    rospy.spin()


if __name__ == "__main__":
    main()
