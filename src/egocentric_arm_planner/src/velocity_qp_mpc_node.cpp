#include "egocentric_arm_planner/velocity_qp_mpc.hpp"

#include <ros/ros.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "velocity_qp_mpc_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  egocentric_arm_planner::VelocityQPMPC mpc;
  if (!mpc.initialize(nh, pnh)) {
    ROS_FATAL("[velocity_qp_mpc_node] Failed to initialize VelocityQPMPC.");
    return 1;
  }

  ros::spin();
  return 0;
}
