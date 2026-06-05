#include "egocentric_arm_planner/receding_horizon_planner.hpp"

#include <ros/ros.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "receding_horizon_planner_node");

  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  egocentric_arm_planner::RecedingHorizonPlanner planner;

  if (!planner.initialize(nh, pnh)) {
    ROS_FATAL("[receding_horizon_planner_node] Failed to initialize planner.");
    return 1;
  }

  ros::spin();

  return 0;
}