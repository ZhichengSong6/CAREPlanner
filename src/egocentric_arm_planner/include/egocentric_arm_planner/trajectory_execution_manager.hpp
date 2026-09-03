#pragma once

#include <ros/ros.h>

#include <sensor_msgs/JointState.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/String.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>

#include <Eigen/Dense>

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace egocentric_arm_planner {

// Low-level trajectory tracker.
//
// C5.4 architecture:
//   local trajectory optimizer -> exact verifier/commit -> this tracker -> actuator
//
// Unlike the legacy C4 short-step backend, this class owns a complete committed
// trajectory and advances through it using elapsed wall/ROS time.  Therefore a
// planner is free to run event-triggered or at a low rate; execution does not
// require a fresh 20 Hz planning trajectory to keep moving.
class TrajectoryExecutionManager {
public:
  TrajectoryExecutionManager() = default;

  bool initialize(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);

private:
  void jointStateCallback(const sensor_msgs::JointStateConstPtr& msg);
  void trajectoryCallback(const trajectory_msgs::JointTrajectoryConstPtr& msg);
  void safetyHoldCallback(const std_msgs::BoolConstPtr& msg);
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

  void publishSummary(bool trajectory_active,
                      bool trajectory_complete,
                      uint32_t trajectory_seq,
                      uint64_t execution_stamp_ns,
                      double phase_s,
                      double remaining_s,
                      double tracking_error_inf,
                      int tracking_error_joint_index,
                      double tracking_error_q_ref,
                      double tracking_error_q_measured,
                      const std::string& source);

  void maybePublishReplanRequest(double tracking_error_inf);
  void maybePublishSmoothHandoffReplanRequest(
      bool trajectory_active,
      uint64_t execution_stamp_ns,
      double phase_s,
      double remaining_s);

  Eigen::VectorXd makeZeroVelocityCommand() const;

private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  ros::Subscriber joint_state_sub_;
  ros::Subscriber trajectory_sub_;
  ros::Subscriber safety_hold_sub_;

  ros::Publisher velocity_command_pub_;
  ros::Publisher reference_state_pub_;
  ros::Publisher summary_pub_;
  ros::Publisher replan_request_pub_;
  ros::Publisher smooth_replan_request_pub_;

  ros::Timer execution_timer_;

  mutable std::mutex data_mutex_;

  bool has_joint_state_ = false;
  bool external_safety_hold_ = false;
  ros::Time external_safety_hold_start_time_;
  unsigned long long external_safety_hold_event_count_ = 0;
  sensor_msgs::JointState latest_joint_state_;
  Eigen::VectorXd q_measured_;
  Eigen::VectorXd dq_measured_;

  // Complete committed trajectory.  A new message atomically replaces the
  // active trajectory; the controller then advances through it at execution_rate_.
  bool has_active_trajectory_ = false;
  bool has_received_trajectory_ = false;
  trajectory_msgs::JointTrajectory active_trajectory_;
  std::vector<int> active_trajectory_mapping_;
  ros::Time active_trajectory_start_time_;
  ros::Time active_trajectory_received_time_;
  double active_trajectory_duration_s_ = 0.0;
  uint32_t active_trajectory_seq_ = 0;
  uint64_t active_execution_stamp_ns_ = 0;

  // Last valid reference is retained after trajectory completion so the robot
  // holds its final configuration instead of snapping back to home.
  bool has_last_reference_ = false;
  Eigen::VectorXd last_q_ref_;
  Eigen::VectorXd last_dq_ref_;

  std::vector<std::string> joint_names_;

  double execution_rate_ = 100.0;
  // Legacy parameter kept for config compatibility; full-trajectory tracking
  // no longer samples a fixed short step.
  double control_dt_ = 0.05;

  double max_start_error_ = 0.5;
  double max_tracking_error_ = 1.0;

  bool hold_when_no_trajectory_ = true;
  bool hold_when_tracking_error_large_ = true;
  bool reject_large_start_error_ = true;

  bool hold_initial_zero_pose_ = true;
  bool hold_last_reference_when_no_trajectory_ = true;
  double reference_timeout_ = 0.15;

  double position_feedback_gain_ = 1.0;
  double max_command_velocity_ = 0.2;

  // Event-triggered replanning support. This is advisory only: the tracker
  // continues to enforce its own hold policy even if no planner subscribes.
  double replan_tracking_error_inf_ = 0.25;
  double replan_request_min_interval_s_ = 0.50;
  ros::Time last_replan_request_time_;

  // Smooth certified receding-horizon handoff. The current certified
  // trajectory always remains executable through its own braking/hold tail;
  // this look-ahead request only asks the planner to prepare a replacement
  // before that tail is normally reached.
  bool smooth_handoff_enabled_ = false;
  double smooth_replan_lead_time_s_ = 0.35;
  double smooth_replan_min_phase_s_ = 0.08;
  uint64_t last_smooth_replan_execution_stamp_ns_ = 0;
  unsigned long long smooth_replan_request_count_ = 0;

  std::string joint_state_topic_ = "/care_arm/joint_states";
  std::string input_trajectory_topic_ =
      "/care_planner/committed_trajectory";
  std::string output_velocity_command_topic_ =
      "/care_arm/arm_group_velocity_controller/command";
  std::string reference_state_topic_ =
      "/care_planner/execution/reference_state";
  std::string summary_topic_ =
      "/care_planner/execution/tracker_summary";
  std::string replan_request_topic_ =
      "/care_planner/local_planner/replan_request";
  std::string smooth_replan_request_topic_ =
      "/care_planner/local_planner/smooth_handoff_replan_request";
  std::string external_safety_hold_topic_ =
      "/care_planner/execution_gcdf/hard_hold";
};

}  // namespace egocentric_arm_planner
