// Protocol component host with prescribed IDEAL hardware. This exercises the
// actual isolated plugin and ROS subscriptions. It is NOT a mechanical test.
#include <class_loader/class_loader.hpp>
#include <controller_interface/controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <ros/ros.h>
#include <rosgraph_msgs/Clock.h>
#include <sensor_msgs/JointState.h>
#include <std_srvs/Trigger.h>
#include <array>
#include <atomic>
#include <iostream>

int main(int argc,char** argv) {
  ros::init(argc,argv,"leased_controller_host");
  if(argc<2)return 2;
  ros::NodeHandle nh,controller_nh("/lease_test/controller");
  using Typed=controller_interface::Controller<hardware_interface::VelocityJointInterface>;
  class_loader::ClassLoader loader(argv[1]);
  auto base=loader.createInstance<controller_interface::ControllerBase>("care_velocity_controller::ExpiringJointVelocityController");
  auto controller=boost::dynamic_pointer_cast<Typed>(base);
  if(!controller)return 3;
  std::vector<std::string> names;
  if(!controller_nh.getParam("joints",names) || names.size()!=7)return 4;
  std::array<double,7> q{},v{},effort{},command{};
  hardware_interface::VelocityJointInterface hw;
  for(size_t j=0;j<7;++j) {
    hardware_interface::JointStateHandle state(names[j],&q[j],&v[j],&effort[j]);
    hw.registerHandle(hardware_interface::JointHandle(state,&command[j]));
  }
  if(!controller->init(&hw,controller_nh))return 5;
  auto clock=nh.advertise<rosgraph_msgs::Clock>("/clock",1);
  auto measured=nh.advertise<sensor_msgs::JointState>("/lease_test/joints",1);
  std::atomic<bool> restart{false};
  auto service=nh.advertiseService<std_srvs::Trigger::Request,std_srvs::Trigger::Response>(
      "/lease_test/restart",[&](std_srvs::Trigger::Request&,std_srvs::Trigger::Response& r){restart=true;r.success=true;return true;});
  ros::Time time(10,0);ros::Time::setNow(time);controller->starting(time);
  ros::AsyncSpinner spinner(1);spinner.start();ros::WallRate rate(1000.);
  uint64_t cycle=0;
  while(ros::ok()) {
    time+=ros::Duration(.001);ros::Time::setNow(time);
    for(size_t j=0;j<7;++j){v[j]=command[j];q[j]+=v[j]*.001;}
    if(restart.exchange(false))controller->starting(time);
    controller->update(time,ros::Duration(.001));
    rosgraph_msgs::Clock c;c.clock=time;clock.publish(c);
    if(++cycle%10==0) {
      sensor_msgs::JointState m;m.header.stamp=time;m.name=names;
      m.position.assign(q.begin(),q.end());m.velocity.assign(v.begin(),v.end());measured.publish(m);
    }
    rate.sleep();
  }
  controller->stopping(time);spinner.stop();
  std::cout<<"LEASE_CONTROLLER_HOST_CLEAN_EXIT prescribed ideal hardware only\n";
  return 0;
}
