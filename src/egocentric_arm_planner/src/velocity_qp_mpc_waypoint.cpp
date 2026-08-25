#include "egocentric_arm_planner/velocity_qp_mpc_waypoint.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_map>

namespace egocentric_arm_planner {
namespace {

bool finiteVector(const Eigen::VectorXd& x) {
  for (int i = 0; i < x.size(); ++i) {
    if (!std::isfinite(x[i])) return false;
  }
  return true;
}

bool finiteMatrix(const Eigen::MatrixXd& x) {
  for (int r = 0; r < x.rows(); ++r) {
    for (int c = 0; c < x.cols(); ++c) {
      if (!std::isfinite(x(r, c))) return false;
    }
  }
  return true;
}

}  // namespace

bool VelocityQPMPCWaypoint::initialize(const ros::NodeHandle& nh,
                                       const ros::NodeHandle& pnh) {
  nh_ = nh;
  pnh_ = pnh;

  if (!loadConfig()) return false;
  if (!loadPositionLimitsFromUrdf()) return false;
  if (!buildStaticQP()) return false;

  previous_command_ = Eigen::VectorXd::Zero(dof_);
  latest_waypoint_q_ = Eigen::VectorXd::Zero(dof_);

  joint_state_sub_ = nh_.subscribe(
      joint_state_topic_, 1, &VelocityQPMPCWaypoint::jointStateCallback, this);
  reference_sub_ = nh_.subscribe(
      reference_topic_, 1, &VelocityQPMPCWaypoint::referenceCallback, this);
  waypoint_active_sub_ = nh_.subscribe(
      waypoint_active_topic_, 1,
      &VelocityQPMPCWaypoint::waypointActiveCallback, this);
  waypoint_q_sub_ = nh_.subscribe(
      waypoint_q_topic_, 1,
      &VelocityQPMPCWaypoint::waypointQCallback, this);
  waypoint_deadline_sub_ = nh_.subscribe(
      waypoint_deadline_topic_, 1,
      &VelocityQPMPCWaypoint::waypointDeadlineCallback, this);
  verification_hold_sub_ = nh_.subscribe(
      verification_hold_topic_, 1,
      &VelocityQPMPCWaypoint::verificationHoldCallback, this);
  recovery_trigger_sub_ = nh_.subscribe(
      recovery_trigger_topic_, 1,
      &VelocityQPMPCWaypoint::recoveryTriggerCallback, this);
  recovery_clear_sub_ = nh_.subscribe(
      recovery_clear_topic_, 1,
      &VelocityQPMPCWaypoint::recoveryClearCallback, this);
  if (!planner_mode_semantics_) {
    replan_ready_sub_ = nh_.subscribe(
        replan_ready_topic_, 1,
        &VelocityQPMPCWaypoint::replanReadyCallback, this);
  }

  velocity_command_pub_ = nh_.advertise<std_msgs::Float64MultiArray>(
      velocity_command_topic_, 1);
  prediction_pub_ = nh_.advertise<trajectory_msgs::JointTrajectory>(
      prediction_topic_, 1);
  solve_time_pub_ = pnh_.advertise<std_msgs::Float32>("solve_time_ms", 1);
  summary_pub_ = pnh_.advertise<std_msgs::String>("summary", 1);
  recovery_active_pub_ = nh_.advertise<std_msgs::Bool>(
      recovery_active_topic_, 1, true);
  recovery_complete_pub_ = nh_.advertise<std_msgs::Bool>(
      recovery_complete_topic_, 1, false);

  publishRecoveryActive(false);

  timer_ = nh_.createTimer(
      ros::Duration(1.0 / rate_), &VelocityQPMPCWaypoint::timerCallback, this);

  ROS_WARN("[VelocityQPMPCWaypoint] This node owns the low-level velocity command topic. Do NOT run TrajectoryExecutionManager or another MPC simultaneously.");
  ROS_INFO_STREAM("[VelocityQPMPCWaypoint] solver=PIQP dense, rate=" << rate_
                  << " Hz, horizon=" << horizon_duration_
                  << " s, K=" << num_intervals_
                  << ", dt=" << dt_
                  << ", variables=" << n_u_
                  << ", constraints=" << n_constraints_);
  ROS_INFO_STREAM("[VelocityQPMPCWaypoint] reference=" << reference_topic_
                  << ", command=" << velocity_command_topic_);
  ROS_WARN_STREAM("[VelocityQPMPCWaypoint] VBC waypoint objective "
                  << (waypoint_enabled_ ? "ENABLED" : "DISABLED")
                  << ": weight=" << waypoint_weight_
                  << ", timeout=" << waypoint_timeout_
                  << " s, active=" << waypoint_active_topic_
                  << ", q_vis=" << waypoint_q_topic_
                  << ", deadline=" << waypoint_deadline_topic_);
  ROS_INFO_STREAM("[VelocityQPMPCWaypoint] verification hold topic="
                  << verification_hold_topic_);
  ROS_WARN_STREAM("[VelocityQPMPCWaypoint] recovery trigger="
                  << (use_external_recovery_trigger_ ? "external_predicted_vbc" : "deadline")
                  << ", topic=" << recovery_trigger_topic_
                  << ", recovery_clear="
                  << (use_external_recovery_clear_ ? "external_global_vbc" : "selected_target_inactive")
                  << ", clear_topic=" << recovery_clear_topic_
                  << ", recovery_weight="
                  << waypoint_weight_ * recovery_weight_scale_
                  << ", planner_mode_semantics="
                  << static_cast<int>(planner_mode_semantics_)
                  << ", complete=" << recovery_complete_topic_
                  << ", replan_ready=" << replan_ready_topic_);
  if (planner_mode_semantics_) {
    ROS_WARN("[VelocityQPMPCWaypoint] PLANNER MODE SEMANTICS ENABLED: REPAIR/NORMAL only select the next-candidate objective. Clear never enters recovery_hold, never publishes recovery_complete, and never waits for replan_ready. The committed execution trajectory is owned downstream and is untouched by this mode switch.");
  } else {
    ROS_INFO("[VelocityQPMPCWaypoint] Legacy Recovery lifecycle enabled: clear may enter recovery_hold and wait for replan_ready.");
  }
  ROS_INFO("[VelocityQPMPCWaypoint] Recovery/REPAIR removes nominal q/terminal/u-reference tracking but preserves hard joint-position, velocity, and acceleration constraints plus effort/smoothness regularization.");
  return true;
}

bool VelocityQPMPCWaypoint::loadJointVectorParam(
    const std::string& param_name,
    const Eigen::VectorXd& fallback,
    Eigen::VectorXd& value) const {
  XmlRpc::XmlRpcValue param;
  if (!pnh_.getParam(param_name, param)) {
    value = fallback;
    return true;
  }

  value = Eigen::VectorXd::Zero(dof_);
  if (param.getType() == XmlRpc::XmlRpcValue::TypeDouble ||
      param.getType() == XmlRpc::XmlRpcValue::TypeInt) {
    const double scalar = param.getType() == XmlRpc::XmlRpcValue::TypeDouble
                              ? static_cast<double>(param)
                              : static_cast<int>(param);
    value.setConstant(scalar);
    return true;
  }

  if (param.getType() == XmlRpc::XmlRpcValue::TypeArray) {
    if (param.size() != dof_) {
      ROS_ERROR_STREAM("[VelocityQPMPCWaypoint] " << param_name
                       << " array size must be " << dof_);
      return false;
    }
    for (int i = 0; i < dof_; ++i) {
      if (param[i].getType() == XmlRpc::XmlRpcValue::TypeDouble) {
        value[i] = static_cast<double>(param[i]);
      } else if (param[i].getType() == XmlRpc::XmlRpcValue::TypeInt) {
        value[i] = static_cast<int>(param[i]);
      } else {
        ROS_ERROR_STREAM("[VelocityQPMPCWaypoint] invalid array entry in "
                         << param_name);
        return false;
      }
    }
    return true;
  }

  if (param.getType() == XmlRpc::XmlRpcValue::TypeStruct) {
    for (int i = 0; i < dof_; ++i) {
      const std::string& name = joint_names_[static_cast<std::size_t>(i)];
      if (!param.hasMember(name)) {
        value[i] = fallback[i];
        continue;
      }
      const XmlRpc::XmlRpcValue& entry = param[name];
      if (entry.getType() == XmlRpc::XmlRpcValue::TypeDouble) {
        value[i] = static_cast<double>(entry);
      } else if (entry.getType() == XmlRpc::XmlRpcValue::TypeInt) {
        value[i] = static_cast<int>(entry);
      } else {
        ROS_ERROR_STREAM("[VelocityQPMPCWaypoint] invalid struct entry in "
                         << param_name << " for " << name);
        return false;
      }
    }
    return true;
  }

  ROS_ERROR_STREAM("[VelocityQPMPCWaypoint] unsupported parameter type for "
                   << param_name);
  return false;
}

bool VelocityQPMPCWaypoint::loadConfig() {
  if (!pnh_.getParam("joint_names", joint_names_)) {
    ROS_ERROR("[VelocityQPMPCWaypoint] Missing param: joint_names");
    return false;
  }
  if (joint_names_.size() != 7) {
    ROS_ERROR("[VelocityQPMPCWaypoint] Controller currently expects exactly 7 joints.");
    return false;
  }
  dof_ = static_cast<int>(joint_names_.size());

  pnh_.param<double>("mpc/rate", rate_, rate_);
  pnh_.param<double>("mpc/horizon_duration", horizon_duration_, horizon_duration_);
  pnh_.param<int>("mpc/num_intervals", num_intervals_, num_intervals_);
  pnh_.param<double>("mpc/q_tracking_weight", q_tracking_weight_, q_tracking_weight_);
  pnh_.param<double>("mpc/terminal_q_tracking_weight",
                     terminal_q_tracking_weight_, terminal_q_tracking_weight_);
  pnh_.param<double>("mpc/u_tracking_weight", u_tracking_weight_, u_tracking_weight_);
  pnh_.param<double>("mpc/u_smooth_weight", u_smooth_weight_, u_smooth_weight_);

  pnh_.param<bool>("mpc/visibility_waypoint/enabled",
                   waypoint_enabled_, waypoint_enabled_);
  pnh_.param<double>("mpc/visibility_waypoint/weight",
                     waypoint_weight_, waypoint_weight_);
  pnh_.param<double>("mpc/visibility_waypoint/timeout",
                     waypoint_timeout_, waypoint_timeout_);
  pnh_.param<double>("mpc/visibility_waypoint/horizon_slack",
                     waypoint_horizon_slack_, waypoint_horizon_slack_);
  pnh_.param<std::string>("mpc/visibility_waypoint/active_topic",
                          waypoint_active_topic_, waypoint_active_topic_);
  pnh_.param<std::string>("mpc/visibility_waypoint/q_topic",
                          waypoint_q_topic_, waypoint_q_topic_);
  pnh_.param<std::string>("mpc/visibility_waypoint/deadline_topic",
                          waypoint_deadline_topic_, waypoint_deadline_topic_);
  pnh_.param<std::string>("mpc/visibility_waypoint/verification_hold_topic",
                          verification_hold_topic_, verification_hold_topic_);
  pnh_.param<bool>("mpc/visibility_waypoint/use_external_recovery_trigger",
                   use_external_recovery_trigger_, use_external_recovery_trigger_);
  pnh_.param<std::string>("mpc/visibility_waypoint/recovery_trigger_topic",
                          recovery_trigger_topic_, recovery_trigger_topic_);
  pnh_.param<bool>("mpc/visibility_waypoint/use_external_recovery_clear",
                   use_external_recovery_clear_, use_external_recovery_clear_);
  pnh_.param<std::string>("mpc/visibility_waypoint/recovery_clear_topic",
                          recovery_clear_topic_, recovery_clear_topic_);
  pnh_.param<double>("mpc/visibility_waypoint/recovery_signal_timeout",
                     recovery_signal_timeout_, recovery_signal_timeout_);
  pnh_.param<bool>("mpc/visibility_waypoint/recovery_enabled",
                   recovery_enabled_, recovery_enabled_);
  pnh_.param<bool>("mpc/visibility_waypoint/planner_mode_semantics",
                   planner_mode_semantics_, planner_mode_semantics_);
  pnh_.param<double>("mpc/visibility_waypoint/recovery_weight_scale",
                     recovery_weight_scale_, recovery_weight_scale_);
  pnh_.param<std::string>("mpc/visibility_waypoint/recovery_active_topic",
                          recovery_active_topic_, recovery_active_topic_);
  pnh_.param<std::string>("mpc/visibility_waypoint/recovery_complete_topic",
                          recovery_complete_topic_, recovery_complete_topic_);
  pnh_.param<std::string>("mpc/visibility_waypoint/replan_ready_topic",
                          replan_ready_topic_, replan_ready_topic_);

  pnh_.param<double>("mpc/joint_position_margin",
                     joint_position_margin_, joint_position_margin_);
  pnh_.param<double>("mpc/joint_state_timeout",
                     joint_state_timeout_, joint_state_timeout_);
  pnh_.param<double>("mpc/reference_timeout",
                     reference_timeout_, reference_timeout_);

  pnh_.param<int>("mpc/piqp/max_iterations",
                  piqp_max_iterations_, piqp_max_iterations_);
  pnh_.param<double>("mpc/piqp/eps_abs", piqp_eps_abs_, piqp_eps_abs_);
  pnh_.param<double>("mpc/piqp/eps_rel", piqp_eps_rel_, piqp_eps_rel_);
  pnh_.param<bool>("mpc/piqp/verbose", piqp_verbose_, piqp_verbose_);
  pnh_.param<bool>("mpc/piqp/compute_timings",
                   piqp_compute_timings_, piqp_compute_timings_);

  pnh_.param<std::string>("robot_description_param",
                          robot_description_param_, robot_description_param_);
  pnh_.param<std::string>("mpc/joint_states",
                          joint_state_topic_, joint_state_topic_);
  pnh_.param<std::string>("mpc/reference_trajectory",
                          reference_topic_, reference_topic_);
  pnh_.param<std::string>("mpc/output_velocity_command",
                          velocity_command_topic_, velocity_command_topic_);
  pnh_.param<std::string>("mpc/predicted_trajectory",
                          prediction_topic_, prediction_topic_);

  if (rate_ <= 0.0 || horizon_duration_ <= 0.0 || num_intervals_ < 2 ||
      piqp_max_iterations_ <= 0 || piqp_eps_abs_ <= 0.0 || piqp_eps_rel_ < 0.0) {
    ROS_ERROR("[VelocityQPMPCWaypoint] Invalid timing/PIQP parameters.");
    return false;
  }
  if (std::min(std::min(q_tracking_weight_, terminal_q_tracking_weight_),
               std::min(u_tracking_weight_, u_smooth_weight_)) < 0.0) {
    ROS_ERROR("[VelocityQPMPCWaypoint] tracking/smoothing weights must be non-negative.");
    return false;
  }
  if (waypoint_weight_ < 0.0 || waypoint_timeout_ <= 0.0 ||
      waypoint_horizon_slack_ < 0.0 || recovery_weight_scale_ <= 0.0 ||
      recovery_signal_timeout_ <= 0.0) {
    ROS_ERROR("[VelocityQPMPCWaypoint] invalid visibility-waypoint/recovery parameters.");
    return false;
  }

  dt_ = horizon_duration_ / static_cast<double>(num_intervals_);
  control_period_ = 1.0 / rate_;

  Eigen::VectorXd default_velocity(7);
  default_velocity << 2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5;
  Eigen::VectorXd default_acceleration(7);
  default_acceleration << 8.0, 8.0, 10.0, 10.0, 15.0, 15.0, 15.0;

  if (!loadJointVectorParam("mpc/joint_velocity_limits", default_velocity,
                            velocity_limits_) ||
      !loadJointVectorParam("mpc/joint_acceleration_limits", default_acceleration,
                            acceleration_limits_)) {
    return false;
  }

  for (int i = 0; i < dof_; ++i) {
    if (!(velocity_limits_[i] > 0.0) || !(acceleration_limits_[i] > 0.0)) {
      ROS_ERROR("[VelocityQPMPCWaypoint] velocity/acceleration limits must be positive.");
      return false;
    }
  }
  return true;
}

bool VelocityQPMPCWaypoint::loadPositionLimitsFromUrdf() {
  std::string urdf;
  if (!nh_.getParam(robot_description_param_, urdf)) {
    ROS_ERROR_STREAM("[VelocityQPMPCWaypoint] Could not read "
                     << robot_description_param_);
    return false;
  }
  (void)urdf;

  Eigen::VectorXd default_min(7), default_max(7);
  default_min << -3.14, -2.30, -3.14, -2.65, -3.14, -3.14, -1.20;
  default_max <<  3.14,  2.30,  3.14,  2.65,  3.14,  3.14,  1.20;
  q_min_ = default_min;
  q_max_ = default_max;

  XmlRpc::XmlRpcValue limits;
  if (pnh_.getParam("mpc/joint_position_limits", limits) &&
      limits.getType() == XmlRpc::XmlRpcValue::TypeStruct) {
    for (int i = 0; i < dof_; ++i) {
      const std::string& name = joint_names_[static_cast<std::size_t>(i)];
      if (!limits.hasMember(name)) continue;
      const XmlRpc::XmlRpcValue& entry = limits[name];
      if (entry.getType() != XmlRpc::XmlRpcValue::TypeStruct ||
          !entry.hasMember("lower") || !entry.hasMember("upper")) {
        ROS_ERROR_STREAM("[VelocityQPMPCWaypoint] invalid position limits for "
                         << name);
        return false;
      }
      q_min_[i] = entry["lower"].getType() == XmlRpc::XmlRpcValue::TypeDouble
                      ? static_cast<double>(entry["lower"])
                      : static_cast<int>(entry["lower"]);
      q_max_[i] = entry["upper"].getType() == XmlRpc::XmlRpcValue::TypeDouble
                      ? static_cast<double>(entry["upper"])
                      : static_cast<int>(entry["upper"]);
    }
  }

  for (int i = 0; i < dof_; ++i) {
    q_min_[i] += joint_position_margin_;
    q_max_[i] -= joint_position_margin_;
    if (!(q_min_[i] < q_max_[i])) {
      ROS_ERROR_STREAM("[VelocityQPMPCWaypoint] invalid position limits for "
                       << joint_names_[static_cast<std::size_t>(i)]);
      return false;
    }
  }
  return true;
}

bool VelocityQPMPCWaypoint::buildStaticQP() {
  n_u_ = dof_ * num_intervals_;

  S_ = Eigen::MatrixXd::Zero(dof_ * num_intervals_, n_u_);
  for (int k = 0; k < num_intervals_; ++k) {
    for (int j = 0; j <= k; ++j) {
      S_.block(k * dof_, j * dof_, dof_, dof_).setIdentity();
      S_.block(k * dof_, j * dof_, dof_, dof_) *= dt_;
    }
  }
  S_terminal_ = S_.bottomRows(dof_);

  D_ = Eigen::MatrixXd::Zero(n_u_, n_u_);
  D_.topLeftCorner(dof_, dof_).setIdentity();
  for (int k = 1; k < num_intervals_; ++k) {
    D_.block(k * dof_, k * dof_, dof_, dof_).setIdentity();
    D_.block(k * dof_, (k - 1) * dof_, dof_, dof_) =
        -Eigen::MatrixXd::Identity(dof_, dof_);
  }

  acceleration_row0_ = 0;
  position_row0_ = n_u_;
  n_constraints_ = 2 * n_u_;
  G_ = Eigen::MatrixXd::Zero(n_constraints_, n_u_);
  G_.block(acceleration_row0_, 0, n_u_, n_u_) = D_;
  G_.block(position_row0_, 0, n_u_, n_u_) = S_;

  H_base_ = 2.0 * (
      q_tracking_weight_ * S_.transpose() * S_
      + terminal_q_tracking_weight_ *
            S_terminal_.transpose() * S_terminal_
      + u_tracking_weight_ * Eigen::MatrixXd::Identity(n_u_, n_u_)
      + u_smooth_weight_ * D_.transpose() * D_);
  H_base_.diagonal().array() += 1e-8;

  H_regularization_ = 2.0 * (
      u_tracking_weight_ * Eigen::MatrixXd::Identity(n_u_, n_u_)
      + u_smooth_weight_ * D_.transpose() * D_);
  H_regularization_.diagonal().array() += 1e-8;

  x_lower_ = Eigen::VectorXd::Zero(n_u_);
  x_upper_ = Eigen::VectorXd::Zero(n_u_);
  for (int k = 0; k < num_intervals_; ++k) {
    x_lower_.segment(k * dof_, dof_) = -velocity_limits_;
    x_upper_.segment(k * dof_, dof_) = velocity_limits_;
  }
  return finiteMatrix(H_base_) && finiteMatrix(H_regularization_) && finiteMatrix(G_);
}

void VelocityQPMPCWaypoint::jointStateCallback(
    const sensor_msgs::JointStateConstPtr& msg) {
  if (!msg) return;
  std::lock_guard<std::mutex> lock(data_mutex_);
  latest_joint_state_ = *msg;
  latest_joint_state_received_ = ros::Time::now();
  has_joint_state_ = true;
}

void VelocityQPMPCWaypoint::referenceCallback(
    const trajectory_msgs::JointTrajectoryConstPtr& msg) {
  if (!msg || msg->points.empty()) return;
  std::lock_guard<std::mutex> lock(data_mutex_);
  latest_reference_ = *msg;
  latest_reference_received_ = ros::Time::now();
  has_reference_ = true;
}

void VelocityQPMPCWaypoint::waypointActiveCallback(
    const std_msgs::BoolConstPtr& msg) {
  if (!msg) return;
  std::lock_guard<std::mutex> lock(data_mutex_);
  latest_waypoint_active_ = msg->data;
  latest_waypoint_active_received_ = ros::Time::now();
  has_waypoint_active_ = true;
}

void VelocityQPMPCWaypoint::waypointQCallback(
    const std_msgs::Float64MultiArrayConstPtr& msg) {
  if (!msg || msg->data.size() != static_cast<std::size_t>(dof_)) {
    ROS_WARN_THROTTLE(1.0,
                      "[VelocityQPMPCWaypoint] ignoring malformed q_vis");
    return;
  }
  Eigen::VectorXd q(dof_);
  for (int i = 0; i < dof_; ++i) {
    q[i] = msg->data[static_cast<std::size_t>(i)];
  }
  if (!finiteVector(q)) {
    ROS_WARN_THROTTLE(1.0,
                      "[VelocityQPMPCWaypoint] ignoring non-finite q_vis");
    return;
  }
  std::lock_guard<std::mutex> lock(data_mutex_);
  latest_waypoint_q_ = q;
  latest_waypoint_q_received_ = ros::Time::now();
  has_waypoint_q_ = true;
}

void VelocityQPMPCWaypoint::waypointDeadlineCallback(
    const std_msgs::Float64ConstPtr& msg) {
  if (!msg || !std::isfinite(msg->data) || msg->data <= 0.0) {
    ROS_WARN_THROTTLE(1.0,
                      "[VelocityQPMPCWaypoint] ignoring invalid absolute deadline");
    return;
  }
  std::lock_guard<std::mutex> lock(data_mutex_);
  latest_waypoint_deadline_abs_s_ = msg->data;
  latest_waypoint_deadline_received_ = ros::Time::now();
  has_waypoint_deadline_ = true;
}

void VelocityQPMPCWaypoint::verificationHoldCallback(
    const std_msgs::BoolConstPtr& msg) {
  if (!msg) return;
  std::lock_guard<std::mutex> lock(data_mutex_);
  latest_verification_hold_ = msg->data;
  latest_verification_hold_received_ = ros::Time::now();
  has_verification_hold_ = true;
}

void VelocityQPMPCWaypoint::recoveryTriggerCallback(
    const std_msgs::BoolConstPtr& msg) {
  if (!msg) return;
  std::lock_guard<std::mutex> lock(data_mutex_);
  latest_recovery_trigger_ = msg->data;
  latest_recovery_trigger_received_ = ros::Time::now();
  has_recovery_trigger_ = true;
}

void VelocityQPMPCWaypoint::recoveryClearCallback(
    const std_msgs::BoolConstPtr& msg) {
  if (!msg) return;
  std::lock_guard<std::mutex> lock(data_mutex_);
  latest_recovery_clear_ = msg->data;
  latest_recovery_clear_received_ = ros::Time::now();
  has_recovery_clear_ = true;
}

void VelocityQPMPCWaypoint::replanReadyCallback(
    const std_msgs::BoolConstPtr& msg) {
  if (!msg || !msg->data) return;
  std::lock_guard<std::mutex> lock(data_mutex_);
  replan_ready_received_ = true;
}

bool VelocityQPMPCWaypoint::extractMeasuredQ(
    const sensor_msgs::JointState& msg,
    Eigen::VectorXd& q) const {
  std::unordered_map<std::string, int> index;
  for (std::size_t i = 0; i < msg.name.size(); ++i) {
    index[msg.name[i]] = static_cast<int>(i);
  }
  q = Eigen::VectorXd::Zero(dof_);
  for (int i = 0; i < dof_; ++i) {
    const auto it = index.find(joint_names_[static_cast<std::size_t>(i)]);
    if (it == index.end() || it->second < 0 ||
        static_cast<std::size_t>(it->second) >= msg.position.size()) {
      return false;
    }
    q[i] = msg.position[static_cast<std::size_t>(it->second)];
  }
  return finiteVector(q);
}

bool VelocityQPMPCWaypoint::buildTrajectoryMapping(
    const trajectory_msgs::JointTrajectory& msg,
    std::vector<int>& mapping) const {
  std::unordered_map<std::string, int> index;
  for (std::size_t i = 0; i < msg.joint_names.size(); ++i) {
    index[msg.joint_names[i]] = static_cast<int>(i);
  }
  mapping.resize(static_cast<std::size_t>(dof_));
  for (int i = 0; i < dof_; ++i) {
    const auto it = index.find(joint_names_[static_cast<std::size_t>(i)]);
    if (it == index.end()) return false;
    mapping[static_cast<std::size_t>(i)] = it->second;
  }
  return true;
}

bool VelocityQPMPCWaypoint::sampleReferencePosition(
    const trajectory_msgs::JointTrajectory& msg,
    const std::vector<int>& mapping,
    double t,
    Eigen::VectorXd& q) const {
  if (msg.points.empty()) return false;
  q = Eigen::VectorXd::Zero(dof_);

  auto copyPoint = [&](const trajectory_msgs::JointTrajectoryPoint& point) {
    if (point.positions.size() < msg.joint_names.size()) return false;
    for (int j = 0; j < dof_; ++j) {
      const int idx = mapping[static_cast<std::size_t>(j)];
      q[j] = point.positions[static_cast<std::size_t>(idx)];
    }
    return finiteVector(q);
  };

  const double first_t = msg.points.front().time_from_start.toSec();
  const double last_t = msg.points.back().time_from_start.toSec();
  if (t <= first_t) return copyPoint(msg.points.front());
  if (t >= last_t) return copyPoint(msg.points.back());

  std::size_t hi = 1;
  while (hi < msg.points.size() && msg.points[hi].time_from_start.toSec() < t) {
    ++hi;
  }
  if (hi >= msg.points.size()) return copyPoint(msg.points.back());
  const std::size_t lo = hi - 1;
  const double t0 = msg.points[lo].time_from_start.toSec();
  const double t1 = msg.points[hi].time_from_start.toSec();
  const double h = t1 - t0;
  if (h <= 1e-9) return false;
  const double alpha = (t - t0) / h;

  if (msg.points[lo].positions.size() < msg.joint_names.size() ||
      msg.points[hi].positions.size() < msg.joint_names.size()) {
    return false;
  }
  for (int j = 0; j < dof_; ++j) {
    const int idx = mapping[static_cast<std::size_t>(j)];
    const double q0 = msg.points[lo].positions[static_cast<std::size_t>(idx)];
    const double q1 = msg.points[hi].positions[static_cast<std::size_t>(idx)];
    q[j] = (1.0 - alpha) * q0 + alpha * q1;
  }
  return finiteVector(q);
}

bool VelocityQPMPCWaypoint::buildReferenceHorizon(
    const trajectory_msgs::JointTrajectory& msg,
    Eigen::MatrixXd& q_ref,
    Eigen::MatrixXd& u_ref) const {
  std::vector<int> mapping;
  if (!buildTrajectoryMapping(msg, mapping)) return false;

  q_ref = Eigen::MatrixXd::Zero(dof_, num_intervals_ + 1);
  for (int k = 0; k <= num_intervals_; ++k) {
    Eigen::VectorXd q;
    if (!sampleReferencePosition(msg, mapping, k * dt_, q)) return false;
    q_ref.col(k) = q;
  }

  u_ref = Eigen::MatrixXd::Zero(dof_, num_intervals_);
  for (int k = 0; k < num_intervals_; ++k) {
    u_ref.col(k) = (q_ref.col(k + 1) - q_ref.col(k)) / dt_;
  }
  return true;
}

void VelocityQPMPCWaypoint::buildBounds(
    const Eigen::VectorXd& q_current,
    Eigen::VectorXd& lower,
    Eigen::VectorXd& upper) const {
  lower = Eigen::VectorXd::Zero(n_constraints_);
  upper = Eigen::VectorXd::Zero(n_constraints_);

  lower.segment(acceleration_row0_, dof_) =
      previous_command_ - acceleration_limits_ * control_period_;
  upper.segment(acceleration_row0_, dof_) =
      previous_command_ + acceleration_limits_ * control_period_;
  for (int k = 1; k < num_intervals_; ++k) {
    lower.segment(acceleration_row0_ + k * dof_, dof_) =
        -acceleration_limits_ * dt_;
    upper.segment(acceleration_row0_ + k * dof_, dof_) =
        acceleration_limits_ * dt_;
  }

  for (int k = 0; k < num_intervals_; ++k) {
    lower.segment(position_row0_ + k * dof_, dof_) = q_min_ - q_current;
    upper.segment(position_row0_ + k * dof_, dof_) = q_max_ - q_current;
  }
}

void VelocityQPMPCWaypoint::buildBaseCycleQP(
    const Eigen::VectorXd& q_current,
    const Eigen::MatrixXd& q_ref,
    const Eigen::MatrixXd& u_ref,
    Eigen::VectorXd& gradient,
    Eigen::VectorXd& lower,
    Eigen::VectorXd& upper) const {
  Eigen::VectorXd q_error_stack(n_u_);
  for (int k = 0; k < num_intervals_; ++k) {
    q_error_stack.segment(k * dof_, dof_) = q_current - q_ref.col(k + 1);
  }
  const Eigen::VectorXd q_terminal_error =
      q_current - q_ref.col(num_intervals_);

  Eigen::VectorXd u_ref_stack(n_u_);
  for (int k = 0; k < num_intervals_; ++k) {
    u_ref_stack.segment(k * dof_, dof_) = u_ref.col(k);
  }

  Eigen::VectorXd smooth_offset = Eigen::VectorXd::Zero(n_u_);
  smooth_offset.head(dof_) = -previous_command_;

  gradient = 2.0 * (
      q_tracking_weight_ * S_.transpose() * q_error_stack
      + terminal_q_tracking_weight_ *
            S_terminal_.transpose() * q_terminal_error
      - u_tracking_weight_ * u_ref_stack
      + u_smooth_weight_ * D_.transpose() * smooth_offset);

  buildBounds(q_current, lower, upper);
}

void VelocityQPMPCWaypoint::buildRegularizationCycleQP(
    const Eigen::VectorXd& q_current,
    Eigen::VectorXd& gradient,
    Eigen::VectorXd& lower,
    Eigen::VectorXd& upper) const {
  Eigen::VectorXd smooth_offset = Eigen::VectorXd::Zero(n_u_);
  smooth_offset.head(dof_) = -previous_command_;
  gradient = 2.0 * u_smooth_weight_ * D_.transpose() * smooth_offset;
  buildBounds(q_current, lower, upper);
}

bool VelocityQPMPCWaypoint::solveWithPIQP(
    const Eigen::MatrixXd& hessian,
    const Eigen::VectorXd& gradient,
    const Eigen::VectorXd& lower,
    const Eigen::VectorXd& upper,
    Eigen::VectorXd& solution,
    int& iterations,
    double& primal_residual,
    double& dual_residual,
    std::string& status_string) const {
  if (hessian.rows() != n_u_ || hessian.cols() != n_u_ ||
      gradient.size() != n_u_ || lower.size() != n_constraints_ ||
      upper.size() != n_constraints_ ||
      !finiteMatrix(hessian) || !finiteVector(gradient)) {
    status_string = "dimension_or_finite_error";
    return false;
  }
  for (int i = 0; i < n_constraints_; ++i) {
    if (!std::isfinite(lower[i]) || !std::isfinite(upper[i]) ||
        lower[i] > upper[i]) {
      status_string = "invalid_bounds";
      return false;
    }
  }

  piqp::DenseSolver<double> solver;
  auto& settings = solver.settings();
  settings.max_iter = piqp_max_iterations_;
  settings.eps_abs = piqp_eps_abs_;
  settings.eps_rel = piqp_eps_rel_;
  settings.verbose = piqp_verbose_;
  settings.compute_timings = piqp_compute_timings_;

  solver.setup(hessian, gradient,
               piqp::nullopt, piqp::nullopt,
               G_, lower, upper,
               x_lower_, x_upper_);
  const piqp::Status status = solver.solve();
  const auto& result = solver.result();

  iterations = static_cast<int>(result.info.iter);
  primal_residual = result.info.primal_res;
  dual_residual = result.info.dual_res;
  status_string = piqp::status_to_string(status);

  if (status != piqp::PIQP_SOLVED || !finiteVector(result.x)) {
    return false;
  }
  solution = result.x;
  return true;
}

Eigen::MatrixXd VelocityQPMPCWaypoint::reconstructPredictedQ(
    const Eigen::VectorXd& q_current,
    const Eigen::VectorXd& u_stack) const {
  Eigen::MatrixXd q_pred(dof_, num_intervals_ + 1);
  q_pred.col(0) = q_current;
  for (int k = 0; k < num_intervals_; ++k) {
    q_pred.col(k + 1) =
        q_pred.col(k) + dt_ * u_stack.segment(k * dof_, dof_);
  }
  return q_pred;
}

void VelocityQPMPCWaypoint::publishVelocity(const Eigen::VectorXd& command) {
  std_msgs::Float64MultiArray msg;
  msg.data.resize(static_cast<std::size_t>(dof_));
  for (int i = 0; i < dof_; ++i) {
    msg.data[static_cast<std::size_t>(i)] = command[i];
  }
  velocity_command_pub_.publish(msg);
}

void VelocityQPMPCWaypoint::publishSafeStop(const std::string& reason) {
  Eigen::VectorXd command = previous_command_;
  for (int i = 0; i < dof_; ++i) {
    const double step = acceleration_limits_[i] * control_period_;
    if (command[i] > step) command[i] -= step;
    else if (command[i] < -step) command[i] += step;
    else command[i] = 0.0;
  }
  publishVelocity(command);
  previous_command_ = command;
  ROS_WARN_STREAM_THROTTLE(
      1.0, "[VelocityQPMPCWaypoint] safe stop: " << reason);
}

void VelocityQPMPCWaypoint::publishPrediction(
    const Eigen::MatrixXd& q_pred,
    const Eigen::VectorXd& u_stack,
    const std::string& frame_id) {
  trajectory_msgs::JointTrajectory msg;
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = frame_id;
  msg.joint_names = joint_names_;
  msg.points.resize(static_cast<std::size_t>(num_intervals_ + 1));
  for (int k = 0; k <= num_intervals_; ++k) {
    auto& point = msg.points[static_cast<std::size_t>(k)];
    point.time_from_start = ros::Duration(k * dt_);
    point.positions.resize(static_cast<std::size_t>(dof_));
    point.velocities.resize(static_cast<std::size_t>(dof_));
    for (int j = 0; j < dof_; ++j) {
      point.positions[static_cast<std::size_t>(j)] = q_pred(j, k);
      point.velocities[static_cast<std::size_t>(j)] =
          k < num_intervals_ ? u_stack[k * dof_ + j] : 0.0;
    }
  }
  prediction_pub_.publish(msg);
}

void VelocityQPMPCWaypoint::publishRecoveryActive(bool active) {
  std_msgs::Bool msg;
  msg.data = active;
  recovery_active_pub_.publish(msg);
}

void VelocityQPMPCWaypoint::publishRecoveryComplete() {
  std_msgs::Bool msg;
  msg.data = true;
  recovery_complete_pub_.publish(msg);
}

void VelocityQPMPCWaypoint::timerCallback(const ros::TimerEvent&) {
  sensor_msgs::JointState joint_state;
  trajectory_msgs::JointTrajectory reference;
  ros::Time joint_received;
  ros::Time reference_received;

  bool has_wp_active = false;
  bool has_wp_q = false;
  bool has_wp_deadline = false;
  bool wp_active = false;
  ros::Time wp_active_received;
  ros::Time wp_q_received;
  ros::Time wp_deadline_received;
  Eigen::VectorXd q_vis;
  double deadline_abs_s = 0.0;

  bool has_verification_hold = false;
  bool verification_hold = false;
  ros::Time verification_hold_received;

  bool has_recovery_trigger = false;
  bool recovery_trigger = false;

  bool has_recovery_clear = false;
  bool recovery_clear = false;
  ros::Time recovery_clear_received;

  bool local_recovery_active = false;
  bool local_recovery_hold = false;
  bool local_replan_ready = false;

  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (!has_joint_state_) {
      publishSafeStop("waiting for JointState");
      return;
    }

    joint_state = latest_joint_state_;
    joint_received = latest_joint_state_received_;
    if (has_reference_) {
      reference = latest_reference_;
      reference_received = latest_reference_received_;
    }

    has_wp_active = has_waypoint_active_;
    has_wp_q = has_waypoint_q_;
    has_wp_deadline = has_waypoint_deadline_;
    wp_active = latest_waypoint_active_;
    wp_active_received = latest_waypoint_active_received_;
    wp_q_received = latest_waypoint_q_received_;
    wp_deadline_received = latest_waypoint_deadline_received_;
    if (has_waypoint_q_) q_vis = latest_waypoint_q_;
    deadline_abs_s = latest_waypoint_deadline_abs_s_;

    has_verification_hold = has_verification_hold_;
    verification_hold = latest_verification_hold_;
    verification_hold_received = latest_verification_hold_received_;

    has_recovery_trigger = has_recovery_trigger_;
    recovery_trigger = latest_recovery_trigger_;

    has_recovery_clear = has_recovery_clear_;
    recovery_clear = latest_recovery_clear_;
    recovery_clear_received = latest_recovery_clear_received_;

    local_recovery_active = recovery_active_;
    local_recovery_hold = recovery_hold_;
    local_replan_ready = replan_ready_received_;
  }

  const ros::Time now = ros::Time::now();
  if ((now - joint_received).toSec() > joint_state_timeout_) {
    publishSafeStop("stale JointState");
    return;
  }

  Eigen::VectorXd q_current;
  if (!extractMeasuredQ(joint_state, q_current)) {
    publishSafeStop("invalid JointState");
    return;
  }

  const bool reference_fresh =
      has_reference_ && !reference.points.empty() &&
      (now - reference_received).toSec() >= 0.0 &&
      (now - reference_received).toSec() <= reference_timeout_;

  const double verification_hold_age = has_verification_hold
      ? (now - verification_hold_received).toSec()
      : std::numeric_limits<double>::infinity();
  const bool verification_hold_active =
      has_verification_hold && verification_hold &&
      verification_hold_age >= 0.0 && verification_hold_age <= waypoint_timeout_;

  const double recovery_clear_age = has_recovery_clear
      ? (now - recovery_clear_received).toSec()
      : std::numeric_limits<double>::infinity();
  const bool external_recovery_clear_active =
      has_recovery_clear && recovery_clear &&
      recovery_clear_age >= 0.0 &&
      recovery_clear_age <= recovery_signal_timeout_;

  bool exited_recovery_hold = false;
  if (!planner_mode_semantics_ &&
      local_recovery_hold && local_replan_ready && reference_fresh) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    recovery_hold_ = false;
    replan_ready_received_ = false;
    recovery_complete_published_ = false;
    local_recovery_hold = false;
    local_replan_ready = false;
    exited_recovery_hold = true;
  }
  if (exited_recovery_hold) {
    ROS_WARN("[VelocityQPMPCWaypoint] RECOVERY REPLAN READY: resuming nominal task tracking on the new reference.");
  }

  double waypoint_age = -1.0;
  double deadline_remaining = std::numeric_limits<double>::quiet_NaN();
  bool waypoint_data_fresh = false;
  if (has_wp_active && has_wp_q && has_wp_deadline &&
      q_vis.size() == dof_ && finiteVector(q_vis) &&
      std::isfinite(deadline_abs_s)) {
    const double active_age = (now - wp_active_received).toSec();
    const double q_age = (now - wp_q_received).toSec();
    const double deadline_age = (now - wp_deadline_received).toSec();
    waypoint_age = std::max(active_age, std::max(q_age, deadline_age));
    deadline_remaining = deadline_abs_s - now.toSec();
    waypoint_data_fresh =
        active_age >= 0.0 && q_age >= 0.0 && deadline_age >= 0.0 &&
        waypoint_age <= waypoint_timeout_;
  }

  const bool recovery_requested = use_external_recovery_trigger_
      ? (has_recovery_trigger && recovery_trigger)
      : (std::isfinite(deadline_remaining) && deadline_remaining <= 0.0);
  const bool recovery_clear_requested = use_external_recovery_clear_
      ? external_recovery_clear_active
      : (has_wp_active && !wp_active);

  bool entered_recovery = false;
  bool completed_recovery = false;
  bool exited_repair_objective = false;
  if (waypoint_enabled_ && waypoint_weight_ > 0.0 && recovery_enabled_) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (recovery_active_) {
      if (recovery_clear_requested) {
        recovery_active_ = false;
        if (planner_mode_semantics_) {
          // C4.4 planner semantics: changing candidate objective must not alter
          // the already committed execution plan or request a new task plan.
          recovery_hold_ = false;
          replan_ready_received_ = false;
          recovery_complete_published_ = false;
          exited_repair_objective = true;
        } else {
          // Legacy controller semantics kept for frozen C4.2/C4.3 experiments.
          recovery_hold_ = true;
          replan_ready_received_ = false;
          completed_recovery = true;
          if (!recovery_complete_published_) {
            recovery_complete_published_ = true;
          }
        }
      }
    } else if ((planner_mode_semantics_ || !recovery_hold_) &&
               !verification_hold_active && waypoint_data_fresh && wp_active &&
               recovery_requested) {
      recovery_active_ = true;
      recovery_complete_published_ = false;
      entered_recovery = true;
    }

    local_recovery_active = recovery_active_;
    local_recovery_hold = recovery_hold_;
  }

  if (entered_recovery) {
    publishRecoveryActive(true);
    if (planner_mode_semantics_) {
      ROS_WARN_STREAM("[VelocityQPMPCWaypoint] REPAIR CANDIDATE OBJECTIVE ACTIVE via "
                      << (use_external_recovery_trigger_ ? "verified-regime trigger" : "deadline")
                      << "; committed execution trajectory remains owned by the verify/commit layer.");
    } else {
      ROS_ERROR_STREAM("[VelocityQPMPCWaypoint] VBC RECOVERY TRIGGERED WHILE UNSAFE via "
                       << (use_external_recovery_trigger_ ? "predicted-VBC trigger" : "deadline")
                       << "; nominal task tracking is suspended.");
    }
  }
  if (exited_repair_objective) {
    publishRecoveryActive(false);
    ROS_WARN("[VelocityQPMPCWaypoint] REPAIR CANDIDATE OBJECTIVE CLEARED -> NORMAL candidate objective. No recovery_hold, no recovery_complete/replan_ready handshake, and the current committed trajectory is unchanged.");
  }
  if (completed_recovery) {
    publishRecoveryActive(false);
    publishRecoveryComplete();
    ROS_WARN_STREAM("[VelocityQPMPCWaypoint] VISIBILITY RECOVERY EPISODE COMPLETE via "
                    << (use_external_recovery_clear_ ? "global predicted-VBC safe" : "selected target inactive/seen")
                    << "; holding/decelerating while the original task is replanned from measured q.");
  }

  if (!local_recovery_active && !local_recovery_hold &&
      !verification_hold_active && !reference_fresh) {
    publishSafeStop(has_reference_ ? "stale nominal reference" : "waiting for nominal reference");
    return;
  }

  Eigen::MatrixXd q_ref = Eigen::MatrixXd::Zero(dof_, num_intervals_ + 1);
  Eigen::MatrixXd u_ref = Eigen::MatrixXd::Zero(dof_, num_intervals_);
  if (reference_fresh) {
    if (!buildReferenceHorizon(reference, q_ref, u_ref)) {
      if (!local_recovery_active && !local_recovery_hold &&
          !verification_hold_active) {
        publishSafeStop("invalid nominal reference");
        return;
      }
      for (int k = 0; k <= num_intervals_; ++k) q_ref.col(k) = q_current;
    }
  } else {
    for (int k = 0; k <= num_intervals_; ++k) q_ref.col(k) = q_current;
  }

  Eigen::VectorXd nominal_gradient, dummy_lower, dummy_upper;
  buildBaseCycleQP(q_current, q_ref, u_ref,
                   nominal_gradient, dummy_lower, dummy_upper);
  const double base_gradient_inf = nominal_gradient.lpNorm<Eigen::Infinity>();

  Eigen::VectorXd gradient, lower, upper;
  Eigen::MatrixXd hessian;
  std::string control_mode = "normal";
  std::string waypoint_status = waypoint_enabled_ ? "waiting" : "disabled";
  int waypoint_k = -1;
  double waypoint_grid_time = -1.0;
  double waypoint_nominal_error_inf = std::numeric_limits<double>::quiet_NaN();
  double waypoint_pred_error_inf = std::numeric_limits<double>::quiet_NaN();
  double waypoint_linear_inf = 0.0;
  double waypoint_hessian_inf = 0.0;
  double applied_waypoint_weight = 0.0;

  if (local_recovery_hold) {
    control_mode = "recovery_hold";
    waypoint_status = "recovery_hold";
    hessian = H_regularization_;
    buildRegularizationCycleQP(q_current, gradient, lower, upper);
  } else if (local_recovery_active) {
    control_mode = "recovery";
    waypoint_status = "recovery";
    if (!waypoint_data_fresh || !wp_active || q_vis.size() != dof_) {
      publishSafeStop("recovery waiting for fresh active steering q_vis");
      return;
    }

    hessian = H_regularization_;
    buildRegularizationCycleQP(q_current, gradient, lower, upper);

    waypoint_k = num_intervals_;
    waypoint_grid_time = horizon_duration_;
    applied_waypoint_weight = waypoint_weight_ * recovery_weight_scale_;
    const Eigen::VectorXd waypoint_offset = q_current - q_vis;
    const Eigen::MatrixXd H_wp =
        2.0 * applied_waypoint_weight *
        S_terminal_.transpose() * S_terminal_;
    const Eigen::VectorXd g_wp =
        2.0 * applied_waypoint_weight *
        S_terminal_.transpose() * waypoint_offset;
    if (!finiteMatrix(H_wp) || !finiteVector(g_wp)) {
      publishSafeStop("invalid recovery visibility cost");
      return;
    }
    hessian += H_wp;
    gradient += g_wp;
    waypoint_linear_inf = g_wp.lpNorm<Eigen::Infinity>();
    waypoint_hessian_inf = H_wp.lpNorm<Eigen::Infinity>();
    waypoint_nominal_error_inf =
        (q_ref.col(num_intervals_) - q_vis).lpNorm<Eigen::Infinity>();
  } else if (verification_hold_active) {
    control_mode = "verification_hold";
    waypoint_status = "verification_hold";
    hessian = H_regularization_;
    buildRegularizationCycleQP(q_current, gradient, lower, upper);
  } else {
    hessian = H_base_;
    buildBaseCycleQP(q_current, q_ref, u_ref, gradient, lower, upper);

    if (waypoint_enabled_) {
      if (waypoint_weight_ <= 0.0) {
        waypoint_status = "zero_weight";
      } else if (!has_wp_active || !has_wp_q || !has_wp_deadline) {
        waypoint_status = "waiting";
      } else if (!wp_active) {
        waypoint_status = "inactive";
      } else if (q_vis.size() != dof_ || !finiteVector(q_vis) ||
                 !std::isfinite(deadline_abs_s)) {
        waypoint_status = "invalid";
      } else if (!waypoint_data_fresh) {
        waypoint_status = "stale";
      } else if (deadline_remaining >
                 horizon_duration_ + waypoint_horizon_slack_) {
        waypoint_status = "future";
      } else {
        control_mode = "intervention";
        if (deadline_remaining <= dt_) {
          waypoint_k = 1;
        } else {
          waypoint_k = static_cast<int>(
              std::floor(deadline_remaining / dt_ + 1e-9));
          waypoint_k = std::max(1, std::min(num_intervals_, waypoint_k));
        }
        waypoint_grid_time = waypoint_k * dt_;

        const Eigen::MatrixXd S_k =
            S_.block((waypoint_k - 1) * dof_, 0, dof_, n_u_);
        const Eigen::VectorXd waypoint_offset = q_current - q_vis;
        applied_waypoint_weight = waypoint_weight_;
        const Eigen::MatrixXd H_wp =
            2.0 * applied_waypoint_weight * S_k.transpose() * S_k;
        const Eigen::VectorXd g_wp =
            2.0 * applied_waypoint_weight * S_k.transpose() * waypoint_offset;

        if (!finiteMatrix(H_wp) || !finiteVector(g_wp)) {
          waypoint_status = "invalid_cost";
          control_mode = "normal";
        } else {
          hessian += H_wp;
          gradient += g_wp;
          waypoint_linear_inf = g_wp.lpNorm<Eigen::Infinity>();
          waypoint_hessian_inf = H_wp.lpNorm<Eigen::Infinity>();
          waypoint_nominal_error_inf =
              (q_ref.col(waypoint_k) - q_vis).lpNorm<Eigen::Infinity>();
          waypoint_status = "used";
        }
      }
    }
  }

  const ros::WallTime tic = ros::WallTime::now();
  Eigen::VectorXd solution;
  int iterations = 0;
  double primal = 0.0;
  double dual = 0.0;
  std::string piqp_status;
  if (!solveWithPIQP(hessian, gradient, lower, upper, solution,
                     iterations, primal, dual, piqp_status)) {
    publishSafeStop("PIQP " + piqp_status);
    return;
  }
  const double solve_ms = (ros::WallTime::now() - tic).toSec() * 1000.0;

  const Eigen::VectorXd command = solution.head(dof_);
  publishVelocity(command);
  previous_command_ = command;

  const Eigen::MatrixXd q_pred = reconstructPredictedQ(q_current, solution);
  publishPrediction(q_pred, solution,
                    reference_fresh ? reference.header.frame_id : std::string("base_link"));

  if (waypoint_k >= 1 && q_vis.size() == dof_ &&
      (waypoint_status == "used" || waypoint_status == "recovery")) {
    waypoint_pred_error_inf =
        (q_pred.col(waypoint_k) - q_vis).lpNorm<Eigen::Infinity>();
  }

  std_msgs::Float32 solve_msg;
  solve_msg.data = static_cast<float>(solve_ms);
  solve_time_pub_.publish(solve_msg);

  double pred_dev_inf = 0.0;
  for (int k = 0; k <= num_intervals_; ++k) {
    pred_dev_inf = std::max(
        pred_dev_inf,
        (q_pred.col(k) - q_ref.col(k)).lpNorm<Eigen::Infinity>());
  }
  const double tracking_inf =
      (q_current - q_ref.col(0)).lpNorm<Eigen::Infinity>();
  const double command_inf = command.lpNorm<Eigen::Infinity>();

  ++sequence_;
  std::ostringstream oss;
  oss << "seq=" << sequence_
      << " solver=PIQP"
      << " status=" << piqp_status
      << " solve=" << solve_ms << "ms"
      << " iter=" << iterations
      << " primal=" << primal
      << " dual=" << dual
      << " control_mode=" << control_mode
      << " planner_mode_semantics=" << static_cast<int>(planner_mode_semantics_)
      << " verification_hold=" << static_cast<int>(verification_hold_active)
      << " recovery_trigger=" << static_cast<int>(has_recovery_trigger && recovery_trigger)
      << " external_recovery_trigger=" << static_cast<int>(use_external_recovery_trigger_)
      << " recovery_clear=" << static_cast<int>(has_recovery_clear && recovery_clear)
      << " external_recovery_clear=" << static_cast<int>(use_external_recovery_clear_)
      << " recovery_clear_age=" << recovery_clear_age
      << " recovery_active=" << static_cast<int>(local_recovery_active)
      << " recovery_hold=" << static_cast<int>(local_recovery_hold)
      << " tracking_inf=" << tracking_inf
      << " command_inf=" << command_inf
      << " pred_dev_inf=" << pred_dev_inf
      << " vbc_wp=" << waypoint_status
      << " waypoint_weight=" << applied_waypoint_weight
      << " waypoint_age=" << waypoint_age
      << " deadline_remaining=" << deadline_remaining
      << " waypoint_k=" << waypoint_k
      << " waypoint_grid_time=" << waypoint_grid_time
      << " waypoint_nominal_error_inf=" << waypoint_nominal_error_inf
      << " waypoint_pred_error_inf=" << waypoint_pred_error_inf
      << " waypoint_linear_inf=" << waypoint_linear_inf
      << " waypoint_hessian_inf=" << waypoint_hessian_inf
      << " base_grad_inf=" << base_gradient_inf
      << " ref_horizon="
      << (reference_fresh && !reference.points.empty()
              ? reference.points.back().time_from_start.toSec()
              : 0.0);

  std_msgs::String summary;
  summary.data = oss.str();
  summary_pub_.publish(summary);
  ROS_INFO_STREAM_THROTTLE(
      0.5, "[VelocityQPMPCWaypoint] " << summary.data);
}

}  // namespace egocentric_arm_planner
