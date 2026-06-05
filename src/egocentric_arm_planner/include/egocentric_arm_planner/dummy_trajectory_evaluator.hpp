#pragma once

#include "egocentric_arm_planner/planner_types.hpp"
#include "arm_trajectory/joint_trajectory.hpp"

#include <ros/ros.h>

namespace egocentric_arm_planner {

class DummyTrajectoryEvaluator {
public:
  DummyTrajectoryEvaluator() = default;

  bool initialize(const ros::NodeHandle& nh);

  bool initialized() const;

  PlannerStatus evaluate(const arm_trajectory::JointTrajectory& tau_task,
                         EvaluationResult& result);

private:
  bool initialized_ = false;

  double T_cmd_ = 0.5;
};

}  // namespace egocentric_arm_planner