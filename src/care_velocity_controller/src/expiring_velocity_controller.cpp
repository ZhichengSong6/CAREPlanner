#include <care_velocity_controller/expiring_velocity_core.hpp>
#include <care_velocity_controller/leased_velocity_core.hpp>
#include <care_velocity_controller/ExecutionLease.h>
#include <care_velocity_controller/LeasedVelocityCommand.h>
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
#include <random>

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
      nh.param("require_execution_lease",require_lease_,false);
      double lease_s=.1;
      nh.param("max_execution_lease_s",lease_s,.1);
      if (!std::isfinite(lease_s) || lease_s<=0. || lease_s>1.) return false;
      leased_.core=core_;
      leased_.max_lease_ns=static_cast<uint64_t>(lease_s*1e9);
      leased_.source_timeout_ns=static_cast<uint64_t>(core_.timeout*1e9);
    } catch (const std::exception& exc) { ROS_ERROR("Controller configuration: %s",exc.what()); return false; }
    commands_.writeFromNonRT(ExpiringVelocityCore::Command());
    if (require_lease_) {
      sub_=nh.subscribe("command_leased",1,&ExpiringJointVelocityController::leasedCommand,this,ros::TransportHints().tcpNoDelay());
      lease_sub_=nh.subscribe("execution_lease",10,&ExpiringJointVelocityController::lease,this,ros::TransportHints().tcpNoDelay());
    } else {
      sub_=nh.subscribe("command_stamped",1,&ExpiringJointVelocityController::command,this,ros::TransportHints().tcpNoDelay());
    }
    rearm_=nh.advertiseService("rearm",&ExpiringJointVelocityController::rearm,this);
    applied_.reset(new realtime_tools::RealtimePublisher<std_msgs::Float64MultiArray>(nh,"command",1));
    applied_->msg_.data.resize(7);
    status_.reset(new realtime_tools::RealtimePublisher<CommandStatus>(nh,"state",1));
    return true;
  }
  void starting(const ros::Time& time) override {
    core_.start(time.toSec());
    if (require_lease_) {
      std::random_device random;
      uint64_t session=(uint64_t(random())<<32)^random();
      if (!session) session=1;
      leased_.start(time.toNSec(),session); session_.store(session);
    }
    revoked_through_.store(0);
    std::atomic_store(&active_identity_,std::shared_ptr<const LeasedVelocityCore::Lease>());
    accepted_source_.store(0);
    rearm_requested_.store(false); last_publish_=time;
    integral_.fill(0.); integral_valid_=true; max_update_gap_=0.; integral_time_=time;
    for (auto& joint : joints_) joint.setCommand(0.);
  }
  void update(const ros::Time& time,const ros::Duration& period) override {
    auto& core=require_lease_ ? leased_.core : core_;
    ExpiringVelocityCore::Joints measured;
    // The previous output was held between the previous update and this pose.
    // Accumulate at control rate, independently of dropped status publications.
    const double elapsed=(time-integral_time_).toSec();
    if (elapsed < 0. || !std::isfinite(period.toSec()) ||
        std::abs(elapsed-period.toSec()) > 1e-8) integral_valid_=false;
    if (elapsed > 0.) {
      max_update_gap_=std::max(max_update_gap_,elapsed);
      for (size_t j=0;j<7;++j) integral_[j]+=core.output[j]*elapsed;
    }
    integral_time_=time;
    for (size_t j=0;j<7;++j) measured[j]=joints_[j].getVelocity();
    const bool rearm=rearm_requested_.exchange(false);
    if (require_lease_)
      leased_.update(time.toNSec(),period.toSec(),*leases_.readFromRT(),*leased_commands_.readFromRT(),
                     revoked_through_.load(),rearm,measured);
    else core.update(time.toSec(),period.toSec(),*commands_.readFromRT(),rearm,measured);
    if(require_lease_) {
      const auto previous=std::atomic_load(&active_identity_);
      if(!previous || previous->token!=leased_.active.token ||
          (leased_.active.token && !LeasedVelocityCore::same(*previous,leased_.active)))
        std::atomic_store(&active_identity_,std::shared_ptr<const LeasedVelocityCore::Lease>(new LeasedVelocityCore::Lease(leased_.active)));
      accepted_source_.store(leased_.source_sequence);
    }
    for (size_t j=0;j<7;++j) joints_[j].setCommand(core.output[j]);
    if (time < last_publish_ || (time-last_publish_).toSec() >= .01) {
      last_publish_=time;
      if (applied_->trylock()) {
        std::copy(core.output.begin(),core.output.end(),applied_->msg_.data.begin());
        applied_->unlockAndPublish();
      }
      if (status_->trylock()) {
        auto& m=status_->msg_; m.header.stamp=time; m.mode=core.mode; m.clock_fault=core.clock_fault;
        m.command_stamp.fromSec(std::max(0.,core.command_stamp)); m.command_sequence=core.command_sequence;
        m.fault_stamp.fromSec(std::max(0.,core.fault_stamp)); m.stopped_stamp.fromSec(std::max(0.,core.stopped_stamp));
        m.rejected_commands=core.rejected; m.rearm_successes=core.rearm_successes; m.rearm_rejections=core.rearm_rejections;
        std::copy(core.output.begin(),core.output.end(),m.applied_velocity.begin());
        std::copy(measured.begin(),measured.end(),m.measured_velocity.begin());
        for (size_t j=0;j<7;++j) m.measured_position[j]=joints_[j].getPosition();
        std::copy(integral_.begin(),integral_.end(),m.applied_position_integral.begin());
        m.integral_valid=integral_valid_; m.max_update_gap=max_update_gap_;
        m.execution_lease_required=require_lease_; m.controller_session=session_.load();
        std::copy(core.velocity_limits.begin(),core.velocity_limits.end(),m.velocity_limits.begin());
        std::copy(core.acceleration_limits.begin(),core.acceleration_limits.end(),m.acceleration_limits.begin());
        m.command_timeout_s=core.timeout;m.max_braking_step_s=core.max_step;
        m.max_execution_lease_s=leased_.max_lease_ns*1e-9;
        if (require_lease_) {
          m.execution_stamp_ns=leased_.active.token; m.lease_sequence=leased_.active.sequence;
          m.source_sequence=leased_.source_sequence; m.lease_valid_until.fromNSec(leased_.active.until_ns);
          m.command_stamp.fromNSec(leased_.generationNs());
          m.certificate_sha256=leased_.active.certificate;
          m.target_range_enforced=leased_.active.target_range_enforced;
          std::copy(leased_.active.target_abs_velocity.begin(),leased_.active.target_abs_velocity.end(),m.target_abs_velocity.begin());
          m.retired_execution_through_ns=leased_.retired_through; m.rejected_leases=leased_.rejected_leases;
        }
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
  static LeasedVelocityCore::Lease leaseValue(const ExecutionLease& msg) {
    LeasedVelocityCore::Lease value;
    value.session=msg.controller_session; value.token=msg.execution_stamp_ns;
    value.sequence=msg.lease_sequence; value.issued_ns=msg.issued_at.toNSec();
    value.until_ns=msg.valid_until.toNSec(); value.certificate=msg.certificate_sha256;
    value.target_range_enforced=msg.target_range_enforced;
    std::copy(msg.target_abs_velocity.begin(),msg.target_abs_velocity.end(),value.target_abs_velocity.begin());
    value.valid=true; return value;
  }
  void lease(const ExecutionLeaseConstPtr& msg) {
    std::lock_guard<std::mutex> lock(ingress_mutex_);
    if (msg->revoke) {
      // Sticky across queue overwrites: a following grant cannot conceal revoke.
      if (msg->controller_session==session_.load())
        revoked_through_.store(std::max(revoked_through_.load(),msg->execution_stamp_ns));
      return;
    }
    auto value=leaseValue(*msg); value.ingress=++lease_ingress_;
    leases_.writeFromNonRT(value);
  }
  void leasedCommand(const LeasedVelocityCommandConstPtr& msg) {
    std::lock_guard<std::mutex> lock(ingress_mutex_);
    LeasedVelocityCore::Command value;
    value.lease.session=msg->controller_session; value.lease.token=msg->execution_stamp_ns;
    value.lease.sequence=msg->lease_sequence; value.lease.until_ns=msg->valid_until.toNSec();
    value.lease.certificate=msg->certificate_sha256;
    value.lease.target_range_enforced=msg->target_range_enforced;
    std::copy(msg->target_abs_velocity.begin(),msg->target_abs_velocity.end(),value.lease.target_abs_velocity.begin());
    value.generated_ns=msg->header.stamp.toNSec(); value.source_sequence=msg->source_sequence;
    value.stop=msg->stop; value.ingress=++ingress_;
    value.valid=msg->joint_names==names_ && msg->velocity.size()==7;
    if (value.valid) std::copy(msg->velocity.begin(),msg->velocity.end(),value.velocity.begin());
    // Terminal stop cannot be overwritten by a later nonzero command callback.
    const auto applied=std::atomic_load(&active_identity_);
    const auto now=ros::Time::now().toNSec();
    const bool zero=std::all_of(value.velocity.begin(),value.velocity.end(),
        [](double v){return std::isfinite(v) && std::abs(v)<=1e-12;});
    if (value.valid && value.stop && applied && LeasedVelocityCore::same(value.lease,*applied) && zero &&
        value.generated_ns && value.generated_ns<=now && now-value.generated_ns<leased_.source_timeout_ns &&
        value.source_sequence>accepted_source_.load())
      revoked_through_.store(std::max(revoked_through_.load(),value.lease.token));
    leased_commands_.writeFromNonRT(value);
  }
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
  LeasedVelocityCore leased_;
  bool require_lease_=false;
  std::atomic<uint64_t> session_{0},revoked_through_{0};
  std::atomic<uint64_t> accepted_source_{0};
  std::shared_ptr<const LeasedVelocityCore::Lease> active_identity_;
  uint64_t lease_ingress_=0;
  realtime_tools::RealtimeBuffer<LeasedVelocityCore::Lease> leases_;
  realtime_tools::RealtimeBuffer<LeasedVelocityCore::Command> leased_commands_;
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
  ros::Subscriber sub_,lease_sub_;
  ros::ServiceServer rearm_;
};
}  // namespace care_velocity_controller
PLUGINLIB_EXPORT_CLASS(care_velocity_controller::ExpiringJointVelocityController,controller_interface::ControllerBase)
