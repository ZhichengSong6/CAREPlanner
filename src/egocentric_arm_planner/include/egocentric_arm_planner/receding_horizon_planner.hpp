#pragma once

#include "arm_model/robot_model.hpp"
#include "arm_trajectory/joint_trajectory.hpp"

#include "egocentric_arm_planner/planner_types.hpp"
#include "egocentric_arm_planner/task_trajectory_generator.hpp"
#include "egocentric_arm_planner/dummy_trajectory_evaluator.hpp"
#include "egocentric_arm_planner/intervention_manager.hpp"

#include <ros/ros.h>

#include <sensor_msgs/JointState.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Float64MultiArray.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>

#include <Eigen/Dense>

#include <memory>
#include <mutex>
#include <string>

namespace egocentric_arm_planner {

class RecedingHorizonPlanner {
public:
  RecedingHorizonPlanner() = default;

  bool initialize(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);

private:
  void jointStateCallback(const sensor_msgs::JointStateConstPtr& msg);
  void targetPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg);
  void mpcCommandCallback(const std_msgs::Float64MultiArrayConstPtr& msg);

  void planningTimerCallback(const ros::TimerEvent& event);

  bool runOnePlanningStep();
  bool publishPersistentCommand();

  bool convertToRosTrajectory(
      const arm_trajectory::JointTrajectory& traj,
      const std::string& frame_id,
      trajectory_msgs::JointTrajectory& msg) const;

  bool hasValidInputs() const;

private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  ros::Subscriber joint_state_sub_;
  ros::Subscriber target_pose_sub_;
  ros::Subscriber mpc_command_sub_;

  ros::Publisher task_traj_pub_;
  ros::Publisher command_traj_pub_;

  ros::Timer planning_timer_;

  std::shared_ptr<arm_model::RobotModel> robot_model_;
  TaskTrajectoryGenerator task_generator_;
  DummyTrajectoryEvaluator dummy_evaluator_;
  InterventionManager intervention_manager_;

  mutable std::mutex data_mutex_;

  bool has_joint_state_ = false;
  bool has_target_pose_ = false;
  bool new_target_pending_ = false;
  bool has_persistent_command_ = false;
  bool has_mpc_command_ = false;
  bool has_previous_mpc_command_ = false;

  sensor_msgs::JointState latest_joint_state_;
  geometry_msgs::PoseStamped latest_target_pose_;
  Eigen::VectorXd latest_mpc_command_;
  Eigen::VectorXd previous_mpc_command_;
  ros::Time latest_mpc_command_received_;
  ros::Time previous_mpc_command_received_;

  // The validated old planner path is executed exactly once for each new EE
  // target. Only after generate/evaluate/intervention all succeed do we cache
  // the resulting command trajectory. Subsequent timer ticks publish an
  // advancing suffix of this same command instead of re-running IK.
  arm_trajectory::JointTrajectory persistent_command_;
  ros::Time persistent_command_start_time_;

  double planning_rate_ = 30.0;
  double mpc_command_timeout_ = 0.20;

  std::string joint_state_topic_ = "/joint_states";
  std::string target_pose_topic_ = "/care_planner/ee_target_pose";
  std::string task_trajectory_topic_ = "/care_planner/task_trajectory";
  std::string command_trajectory_topic_ = "/care_planner/command_trajectory";
  std::string mpc_command_topic_ =
      "/care_arm/arm_group_velocity_controller/command";

  bool publish_task_trajectory_ = true;

  double overrun_warn_ratio_ = 1.0;
};

}  // namespace egocentric_arm_planner
