#pragma once

#include "egocentric_arm_planner/planner_types.hpp"
#include "arm_trajectory/joint_trajectory.hpp"

#include <ros/ros.h>

#include <Eigen/Dense>

namespace egocentric_arm_planner {

class InterventionManager {
public:
  InterventionManager() = default;

  bool initialize(const ros::NodeHandle& nh);

  bool initialized() const;

  PlannerStatus decideCommand(const arm_trajectory::JointTrajectory& tau_task,
                              const EvaluationResult& eval,
                              const Eigen::VectorXd& q_current,
                              arm_trajectory::JointTrajectory& tau_cmd);

  const InterventionManagerConfig& config() const;

private:
  PlannerStatus makeTaskPrefixCommand(
      const arm_trajectory::JointTrajectory& tau_task,
      arm_trajectory::JointTrajectory& tau_cmd) const;

  PlannerStatus makeHoldCommand(const Eigen::VectorXd& q_current,
                                arm_trajectory::JointTrajectory& tau_cmd) const;

private:
  bool initialized_ = false;

  InterventionManagerConfig config_;
};

}  // namespace egocentric_arm_planner