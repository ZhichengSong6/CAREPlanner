#pragma once

#include <ros/ros.h>

#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64MultiArray.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>

#include <Eigen/Dense>

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace egocentric_arm_planner {

class TrajectoryExecutionManager {
public:
  TrajectoryExecutionManager() = default;

  bool initialize(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);

private:
  void jointStateCallback(const sensor_msgs::JointStateConstPtr& msg);
  void trajectoryCallback(const trajectory_msgs::JointTrajectoryConstPtr& msg);
  void executionTimerCallback(const ros::TimerEvent& event);

  bool loadConfig();

  bool extractMeasuredState(const sensor_msgs::JointState& msg,
                            Eigen::VectorXd& q,
                            Eigen::VectorXd& dq) const;

  bool trajectoryHasExpectedJoints(
      const trajectory_msgs::JointTrajectory& traj) const;

  bool buildTrajectoryJointIndexMap(
      const trajectory_msgs::JointTrajectory& traj,
      std::vector<int>& traj_index_for_control_joint) const;

  bool sampleTrajectory(const trajectory_msgs::JointTrajectory& traj,
                        const std::vector<int>& traj_index_for_control_joint,
                        double t,
                        Eigen::VectorXd& q_ref,
                        Eigen::VectorXd& dq_ref,
                        Eigen::VectorXd& ddq_ref) const;

  bool getPointVector(const trajectory_msgs::JointTrajectoryPoint& point,
                      const std::vector<double>& field,
                      const std::vector<int>& traj_index_for_control_joint,
                      Eigen::VectorXd& out,
                      bool allow_missing_as_zero) const;

  double getTrajectoryEndTime(
      const trajectory_msgs::JointTrajectory& traj) const;

  Eigen::VectorXd computeVelocityCommand(const Eigen::VectorXd& q_ref,
                                         const Eigen::VectorXd& dq_ref,
                                         const Eigen::VectorXd& q_measured) const;

  Eigen::VectorXd clampVelocityCommand(const Eigen::VectorXd& dq_cmd) const;

  void publishVelocityCommand(const Eigen::VectorXd& dq_cmd);

  void publishReferenceState(const Eigen::VectorXd& q_ref,
                             const Eigen::VectorXd& dq_ref);

  Eigen::VectorXd makeZeroVelocityCommand() const;


private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  ros::Subscriber joint_state_sub_;
  ros::Subscriber trajectory_sub_;

  ros::Publisher velocity_command_pub_;
  ros::Publisher reference_state_pub_;

  ros::Timer execution_timer_;

  mutable std::mutex data_mutex_;

  bool has_joint_state_ = false;
  sensor_msgs::JointState latest_joint_state_;
  Eigen::VectorXd q_measured_;
  Eigen::VectorXd dq_measured_;

  // Short-step reference used by the velocity backend.  The planner may publish
  // a full horizon trajectory, but this execution manager only tracks the
  // reference state at t = control_dt_ of the most recent trajectory.
  bool has_reference_step_ = false;
  bool has_received_trajectory_ = false;
  Eigen::VectorXd q_ref_step_;
  Eigen::VectorXd dq_ref_step_;
  Eigen::VectorXd ddq_ref_step_;
  ros::Time last_reference_time_;

  std::vector<std::string> joint_names_;

  double execution_rate_ = 100.0;
  double control_dt_ = 0.05;

  double max_start_error_ = 0.5;
  double max_tracking_error_ = 1.0;

  bool hold_when_no_trajectory_ = true;
  bool hold_when_tracking_error_large_ = true;
  bool reject_large_start_error_ = true;

  // When no trajectory has ever been received, optionally hold the zero joint
  // pose using the same position-feedback velocity law.  After a trajectory has
  // been received, stale/no-new-trajectory behavior holds the last reference,
  // not the zero pose.
  bool hold_initial_zero_pose_ = true;
  bool hold_last_reference_when_no_trajectory_ = true;
  double reference_timeout_ = 0.15;

  double position_feedback_gain_ = 1.0;
  double max_command_velocity_ = 0.2;

  std::string joint_state_topic_ = "/care_arm/joint_states";
  std::string input_trajectory_topic_ =
      "/care_planner/command_trajectory_candidate";
  std::string output_velocity_command_topic_ =
      "/care_arm/arm_group_velocity_controller/command";
  std::string reference_state_topic_ =
      "/care_planner/execution/reference_state";
};

}  // namespace egocentric_arm_planner
