#include <care_velocity_controller/expiring_velocity_core.hpp>
#include <care_velocity_controller/CommandStatus.h>
#include <controller_interface/controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.hpp>
#include <realtime_tools/realtime_buffer.h>
#include <realtime_tools/realtime_publisher.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_srvs/Trigger.h>
#include <atomic>
#include <memory>
#include <mutex>

namespace care_velocity_controller {
class ExpiringJointVelocityController : public controller_interface::Controller<hardware_interface::VelocityJointInterface> {
 public:
  bool init(hardware_interface::VelocityJointInterface* hw, ros::NodeHandle& nh) override {
    if (!nh.getParam("joints", names_) || names_.size()!=7) {
      ROS_ERROR("ExpiringJointVelocityController requires exactly seven ordered joints"); return false;
    }
    try {
      for (const auto& name : names_) joints_.push_back(hw->getHandle(name));
      std::vector<double> limits;
      if (!nh.getParam("velocity_limits", limits) || limits.size()!=7) return false;
      std::copy(limits.begin(), limits.end(), core_.velocity_limits.begin());
      if (!nh.getParam("acceleration_limits", limits) || limits.size()!=7) return false;
      std::copy(limits.begin(), limits.end(), core_.acceleration_limits.begin());
      nh.param("command_timeout", core_.timeout, .1);
      nh.param("max_braking_step", core_.max_step, .02);
      nh.param("stopped_velocity_tolerance", core_.stopped_tolerance, .001);
      core_.validate();
    } catch (const std::exception& exc) { ROS_ERROR("Controller configuration: %s",exc.what()); return false; }
    commands_.writeFromNonRT(ExpiringVelocityCore::Command());
    sub_=nh.subscribe("command_stamped",1,&ExpiringJointVelocityController::command,this,ros::TransportHints().tcpNoDelay());
    rearm_=nh.advertiseService("rearm",&ExpiringJointVelocityController::rearm,this);
    applied_.reset(new realtime_tools::RealtimePublisher<std_msgs::Float64MultiArray>(nh,"command",1));
    applied_->msg_.data.resize(7);
    status_.reset(new realtime_tools::RealtimePublisher<CommandStatus>(nh,"state",1));
    return true;
  }
  void starting(const ros::Time& time) override {
    core_.start(time.toSec()); rearm_requested_.store(false); last_publish_=time;
    integral_.fill(0.); integral_valid_=true; max_update_gap_=0.; integral_time_=time;
    for (auto& joint : joints_) joint.setCommand(0.);
  }
  void update(const ros::Time& time,const ros::Duration& period) override {
    ExpiringVelocityCore::Joints measured;
    // The previous output was held between the previous update and this pose.
    // Accumulate at control rate, independently of dropped status publications.
    const double elapsed=(time-integral_time_).toSec();
    if (elapsed < 0. || !std::isfinite(period.toSec()) ||
        std::abs(elapsed-period.toSec()) > 1e-8) integral_valid_=false;
    if (elapsed > 0.) {
      max_update_gap_=std::max(max_update_gap_,elapsed);
      for (size_t j=0;j<7;++j) integral_[j]+=core_.output[j]*elapsed;
    }
    integral_time_=time;
    for (size_t j=0;j<7;++j) measured[j]=joints_[j].getVelocity();
    core_.update(time.toSec(),period.toSec(),*commands_.readFromRT(),rearm_requested_.exchange(false),measured);
    for (size_t j=0;j<7;++j) joints_[j].setCommand(core_.output[j]);
    if (time < last_publish_ || (time-last_publish_).toSec() >= .01) {
      last_publish_=time;
      if (applied_->trylock()) {
        std::copy(core_.output.begin(),core_.output.end(),applied_->msg_.data.begin());
        applied_->unlockAndPublish();
      }
      if (status_->trylock()) {
        auto& m=status_->msg_; m.header.stamp=time; m.mode=core_.mode; m.clock_fault=core_.clock_fault;
        m.command_stamp.fromSec(std::max(0.,core_.command_stamp)); m.command_sequence=core_.command_sequence;
        m.fault_stamp.fromSec(std::max(0.,core_.fault_stamp)); m.stopped_stamp.fromSec(std::max(0.,core_.stopped_stamp));
        m.rejected_commands=core_.rejected; m.rearm_successes=core_.rearm_successes; m.rearm_rejections=core_.rearm_rejections;
        std::copy(core_.output.begin(),core_.output.end(),m.applied_velocity.begin());
        std::copy(measured.begin(),measured.end(),m.measured_velocity.begin());
        for (size_t j=0;j<7;++j) m.measured_position[j]=joints_[j].getPosition();
        std::copy(integral_.begin(),integral_.end(),m.applied_position_integral.begin());
        m.integral_valid=integral_valid_; m.max_update_gap=max_update_gap_;
        status_->unlockAndPublish();
      }
    }
  }
  void stopping(const ros::Time&) override {
    // Controller-manager lifecycle stop has no further update cycles available.
    // Lifecycle switching is outside the active-controller braking contract.
    for (auto& joint : joints_) joint.setCommand(0.);
  }
 private:
  void command(const sensor_msgs::JointStateConstPtr& msg) {
    std::lock_guard<std::mutex> lock(ingress_mutex_);
    ExpiringVelocityCore::Command c;
    c.ingress=++ingress_; c.stamp=msg->header.stamp.toSec(); c.sequence=msg->header.seq;
    c.valid=msg->name==names_ && msg->velocity.size()==7 && !msg->header.stamp.isZero();
    if (c.valid) std::copy(msg->velocity.begin(),msg->velocity.end(),c.velocity.begin());
    commands_.writeFromNonRT(c);
  }
  bool rearm(std_srvs::Trigger::Request&,std_srvs::Trigger::Response& response) {
    rearm_requested_.store(true);
    response.success=true;
    response.message="Rearm queued; inspect state.rearm_successes/rearm_rejections. Requires latched stop, fresh post-stop zero and measured rest.";
    return true;
  }
  ExpiringVelocityCore core_;
  std::vector<std::string> names_;
  std::vector<hardware_interface::JointHandle> joints_;
  realtime_tools::RealtimeBuffer<ExpiringVelocityCore::Command> commands_;
  std::unique_ptr<realtime_tools::RealtimePublisher<std_msgs::Float64MultiArray>> applied_;
  std::unique_ptr<realtime_tools::RealtimePublisher<CommandStatus>> status_;
  std::atomic<bool> rearm_requested_{false};
  std::mutex ingress_mutex_;
  uint64_t ingress_=0;
  ros::Time last_publish_;
  ros::Time integral_time_;
  ExpiringVelocityCore::Joints integral_{};
  bool integral_valid_=true;
  double max_update_gap_=0.;
  ros::Subscriber sub_;
  ros::ServiceServer rearm_;
};
}  // namespace care_velocity_controller
PLUGINLIB_EXPORT_CLASS(care_velocity_controller::ExpiringJointVelocityController,controller_interface::ControllerBase)
