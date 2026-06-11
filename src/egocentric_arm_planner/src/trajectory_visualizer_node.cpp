#include "egocentric_arm_planner/trajectory_visualizer.hpp"

#include <ros/ros.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "trajectory_visualizer_node");

  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  egocentric_arm_planner::TrajectoryVisualizer visualizer;

  if (!visualizer.initialize(nh, pnh)) {
    ROS_FATAL("[trajectory_visualizer_node] Failed to initialize trajectory visualizer.");
    return 1;
  }

  ros::spin();

  return 0;
}