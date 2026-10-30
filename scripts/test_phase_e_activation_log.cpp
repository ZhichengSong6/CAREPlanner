#include "tracker_diagnostic_access.hpp"
#include <boost/make_shared.hpp>
#include <iostream>
#include <stdexcept>
int main(int argc,char** argv) {try {
 ros::init(argc,argv,"activation_log_test",ros::init_options::NoSigintHandler);
 ros::Time::init();ros::Time::setNow(ros::Time(100,0));
 egocentric_arm_planner::TrajectoryExecutionManager tracker;
 tracker.joint_names_={"joint1","joint2","joint3","joint4","wrist_joint1","wrist_joint2","wrist_joint3"};
 // No initialize(), timer, publishers, subscribers or actuator. The class's
 // default NodeHandles connect only to the test's private ROS master.
 for(int i=0;i<3;++i) {
  auto msg=boost::make_shared<trajectory_msgs::JointTrajectory>();
  msg->joint_names=tracker.joint_names_;msg->header.seq=i+1;msg->header.stamp=ros::Time(200+i,0);
  trajectory_msgs::JointTrajectoryPoint a;a.positions.assign(7,0.);a.velocities.assign(7,0.);a.accelerations.assign(7,0.);
  auto b=a;b.time_from_start=ros::Duration(.2);msg->points={a,b};
  tracker.trajectoryCallback(msg);
  if(tracker.active_execution_stamp_ns_!=msg->header.stamp.toNSec())throw std::runtime_error("wrong activation");
  ros::Time::setNow(ros::Time(100,10000000*(i+1)));
 }
 std::cout<<"PASS three accepted execution identities within30ms; no timer or publishers\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
