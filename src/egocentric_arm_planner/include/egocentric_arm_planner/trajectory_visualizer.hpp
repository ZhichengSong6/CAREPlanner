#pragma once

#include "arm_model/robot_model.hpp"

#include <ros/ros.h>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace egocentric_arm_planner {

class TrajectoryVisualizer {
public:
  TrajectoryVisualizer() = default;

  bool initialize(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);

private:
  void targetPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg);
  void taskTrajectoryCallback(const trajectory_msgs::JointTrajectoryConstPtr& msg);
  void commandTrajectoryCallback(const trajectory_msgs::JointTrajectoryConstPtr& msg);

  void publishMarkers();

  bool trajectoryToEePath(const trajectory_msgs::JointTrajectory& traj,
                          std::vector<geometry_msgs::Point>& ee_points) const;

  bool buildQFromTrajectoryPoint(const trajectory_msgs::JointTrajectory& traj,
                                 std::size_t point_index,
                                 Eigen::VectorXd& q) const;

  visualization_msgs::Marker makeDeleteAllMarker() const;

  visualization_msgs::Marker makePathMarker(
      const std::string& ns,
      int id,
      const std::string& frame_id,
      const std::vector<geometry_msgs::Point>& points,
      double r,
      double g,
      double b,
      double a,
      double line_width) const;

  void appendTargetMarkers(const geometry_msgs::PoseStamped& target,
                           visualization_msgs::MarkerArray& marker_array) const;

private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  ros::Subscriber target_pose_sub_;
  ros::Subscriber task_traj_sub_;
  ros::Subscriber command_traj_sub_;

  ros::Publisher marker_pub_;

  std::shared_ptr<arm_model::RobotModel> robot_model_;

  mutable std::mutex data_mutex_;

  bool has_target_pose_ = false;
  bool has_task_traj_ = false;
  bool has_command_traj_ = false;

  geometry_msgs::PoseStamped latest_target_pose_;
  trajectory_msgs::JointTrajectory latest_task_traj_;
  trajectory_msgs::JointTrajectory latest_command_traj_;

  std::string target_pose_topic_ = "/care_planner/ee_target_pose";
  std::string task_trajectory_topic_ = "/care_planner/task_trajectory";
  std::string command_trajectory_topic_ = "/care_planner/command_trajectory";
  std::string marker_topic_ = "/care_planner/debug/markers";

  double path_line_width_ = 0.015;
  double command_line_width_ = 0.03;
  double target_axis_length_ = 0.12;
  double target_axis_width_ = 0.01;
  double target_sphere_radius_ = 0.035;
};

}  // namespace egocentric_arm_planner