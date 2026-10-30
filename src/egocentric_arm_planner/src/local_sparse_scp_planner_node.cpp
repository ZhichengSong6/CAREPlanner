#include "egocentric_arm_planner/local_sparse_scp_planner.hpp"

#include <ros/ros.h>
#include <string>

namespace egocentric_arm_planner { int replaySparseQpFile(const char* path); }

int main(int argc, char** argv) {
  if (argc==3 && std::string(argv[1])=="--replay-qp")
    return egocentric_arm_planner::replaySparseQpFile(argv[2]);
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
