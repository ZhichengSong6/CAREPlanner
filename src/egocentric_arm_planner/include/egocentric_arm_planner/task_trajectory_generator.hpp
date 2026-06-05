#pragma once

#include "egocentric_arm_planner/planner_types.hpp"

#include "arm_model/robot_model.hpp"
#include "arm_trajectory/joint_trajectory.hpp"

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <memory>
#include <string>

namespace egocentric_arm_planner {

class TaskTrajectoryGenerator {
public:
  TaskTrajectoryGenerator() = default;

  bool initialize(const ros::NodeHandle& nh,
                  const std::shared_ptr<arm_model::RobotModel>& robot_model);

  bool initialized() const;

  PlannerStatus generate(const Eigen::VectorXd& q_current,
                         const Eigen::VectorXd& dq_current,
                         const geometry_msgs::PoseStamped& target_pose_msg,
                         arm_trajectory::JointTrajectory& tau_task);

  const TaskTrajectoryGeneratorConfig& config() const;

  Eigen::VectorXd lastGoalQ() const;
  PlannerStatus lastStatus() const;
  std::string lastStatusString() const;

private:
  bool loadConfig(const ros::NodeHandle& nh);

  double computeTrajectoryDuration(const Eigen::VectorXd& q_current,
                                   const Eigen::VectorXd& q_goal) const;

private:
  bool initialized_ = false;

  TaskTrajectoryGeneratorConfig config_;

  std::shared_ptr<arm_model::RobotModel> robot_model_;

  Eigen::VectorXd last_goal_q_;
  PlannerStatus last_status_ = PlannerStatus::NOT_INITIALIZED;
};

}  // namespace egocentric_arm_planner