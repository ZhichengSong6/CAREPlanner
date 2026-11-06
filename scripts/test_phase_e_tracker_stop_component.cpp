#include "tracker_diagnostic_access.hpp"
#include <boost/make_shared.hpp>
#include <iostream>
#include <stdexcept>
using Tracker=egocentric_arm_planner::TrajectoryExecutionManager;
#define CHECK(x) do{if(!(x))throw std::runtime_error(#x);}while(false)
int main(int argc,char** argv){ros::init(argc,argv,"tracker_stop_component");try{
  ros::Time::setNow(ros::Time(30,0));ros::NodeHandle nh,pnh("~");Tracker t;CHECK(t.initialize(nh,pnh));t.execution_timer_.stop();
  auto measured=boost::make_shared<sensor_msgs::JointState>();measured->name=t.joint_names_;
  measured->position.assign(7,0.);measured->velocity.assign(7,0.);measured->header.stamp=ros::Time::now();t.jointStateCallback(measured);
  auto command=boost::make_shared<std_msgs::Float64MultiArray>();command->data.assign(7,0.);t.limiterOutputCallback(command);
  auto curve=boost::make_shared<trajectory_msgs::JointTrajectory>();curve->joint_names=t.joint_names_;
  curve->header.stamp=ros::Time(29,0);curve->points.resize(3);
  for(int k=0;k<3;++k){curve->points[k].positions.assign(7,k==0?0.:k==1?.005:.01);
    curve->points[k].velocities.assign(7,k==1?.01:0.);curve->points[k].time_from_start=ros::Duration(k);}
  auto incompatible=boost::make_shared<trajectory_msgs::JointTrajectory>(*curve);incompatible->points[1].velocities[5]=2.5;
  t.trajectoryCallback(incompatible);CHECK(!t.has_received_trajectory_);
  t.trajectoryCallback(curve);CHECK(t.has_active_trajectory_);
  ros::Time::setNow(ros::Time(30,10000000));measured->header.stamp=ros::Time::now();
  measured->position[1]=1.;measured->velocity[3]=-.16;t.jointStateCallback(measured);
  t.executionTimerCallback(ros::TimerEvent{});
  CHECK(!t.has_active_trajectory_ && t.stop_state_.token==29000000000ULL && !t.stop_state_.settled);
  auto next=boost::make_shared<trajectory_msgs::JointTrajectory>(*curve);next->header.stamp=ros::Time(30,20000000);
  t.trajectoryCallback(next);CHECK(t.stop_state_.token==29000000000ULL && !t.has_active_trajectory_);
  for(int k=2;k<10;++k){
    ros::Time::setNow(ros::Time(30,k*10000000));measured->header.stamp=ros::Time::now();
    measured->velocity.assign(7,0.);t.jointStateCallback(measured);t.limiterOutputCallback(command);
    t.executionTimerCallback(ros::TimerEvent{});CHECK(t.stop_state_.token==29000000000ULL);
  }
  CHECK(t.stop_state_.settled && !t.has_last_reference_);
  t.trajectoryCallback(curve);CHECK(!t.has_active_trajectory_); // old token remains tombstoned after settling
  for(auto& p:next->points){p.positions=measured->position;p.velocities.assign(7,0.);}
  t.trajectoryCallback(next);CHECK(t.has_active_trajectory_ && !t.stop_state_.token);
  auto old_stop=boost::make_shared<std_msgs::String>();old_stop->data="execution_stamp_ns=29000000000";
  t.executionVbcStopCallback(old_stop);CHECK(!t.execution_vbc_stop_stamp_ns_);
  std::cout<<"PASS actual tracker callbacks: over-speed reject, tracking abort, stopping identity, no restart, settled new token, stale stop rejection\n";
  ros::shutdown();return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';ros::shutdown();return 1;}}
