#include "egocentric_arm_planner/local_sparse_scp_planner.hpp"

#include <ros/ros.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "local_sparse_scp_planner_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  egocentric_arm_planner::LocalSparseSCPPlanner planner;
  if (!planner.initialize(nh, pnh)) {
    ROS_FATAL("[local_sparse_scp_planner_node] initialization failed");
    return 1;
  }

  ros::AsyncSpinner spinner(2);
  spinner.start();
  ros::waitForShutdown();
  return 0;
}
