#include "egocentric_arm_planner/trajectory_execution_manager.hpp"

#include <ros/ros.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "trajectory_execution_manager_node");

  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  egocentric_arm_planner::TrajectoryExecutionManager manager;

  if (!manager.initialize(nh, pnh)) {
    ROS_FATAL("[trajectory_execution_manager_node] Failed to initialize.");
    return 1;
  }

  ros::spin();

  return 0;
}