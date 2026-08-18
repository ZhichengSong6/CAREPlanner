#include "egocentric_arm_planner/velocity_qp_mpc_waypoint.hpp"

#include <ros/ros.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "velocity_qp_mpc_waypoint_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  egocentric_arm_planner::VelocityQPMPCWaypoint mpc;
  if (!mpc.initialize(nh, pnh)) {
    ROS_FATAL("[velocity_qp_mpc_waypoint_node] Failed to initialize.");
    return 1;
  }

  ros::spin();
  return 0;
}
