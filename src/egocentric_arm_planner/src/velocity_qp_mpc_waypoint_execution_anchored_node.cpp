#include "egocentric_arm_planner/velocity_qp_mpc_waypoint.hpp"

#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>

#include <string>

int main(int argc, char** argv) {
  ros::init(argc, argv, "velocity_qp_mpc_waypoint_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  egocentric_arm_planner::VelocityQPMPCWaypoint mpc;
  if (!mpc.initialize(nh, pnh)) {
    ROS_FATAL("[VelocityQPMPCWaypointExecutionAnchored] initialization failed");
    return 1;
  }

  std::string executed_command_topic =
      "/care_arm/arm_group_velocity_controller/command";
  pnh.param<std::string>(
      "mpc/execution_anchor_velocity_command",
      executed_command_topic,
      executed_command_topic);

  const ros::Subscriber executed_command_sub = nh.subscribe<std_msgs::Float64MultiArray>(
      executed_command_topic,
      1,
      [&mpc](const std_msgs::Float64MultiArrayConstPtr& msg) {
        mpc.setExecutedCommandAnchor(msg);
      });
  (void)executed_command_sub;

  ROS_WARN_STREAM(
      "[VelocityQPMPCWaypointExecutionAnchored] first-step acceleration and "
      "smoothness are anchored to ACTUALLY EXECUTED command topic: "
      << executed_command_topic);

  ros::spin();
  return 0;
}
