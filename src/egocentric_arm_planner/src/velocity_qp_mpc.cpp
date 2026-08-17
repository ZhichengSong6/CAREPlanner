#include "egocentric_arm_planner/velocity_qp_mpc.hpp"

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

}  // namespace

bool VelocityQPMPC::initialize(const ros::NodeHandle& nh,
                               const ros::NodeHandle& pnh) {
  nh_ = nh;
  pnh_ = pnh;

  if (!loadConfig()) return false;
  if (!loadPositionLimitsFromUrdf()) return false;
  if (!buildStaticQP()) return false;

  previous_command_ = Eigen::VectorXd::Zero(dof_);

  auto& settings = piqp_solver_.settings();
  settings.max_iter = piqp_max_iterations_;
  settings.eps_abs = piqp_eps_abs_;
  settings.eps_rel = piqp_eps_rel_;
  settings.verbose = piqp_verbose_;
  settings.compute_timings = piqp_compute_timings_;
  settings.preconditioner_reuse_on_update = true;

  joint_state_sub_ = nh_.subscribe(
      joint_state_topic_, 1, &VelocityQPMPC::jointStateCallback, this);
  reference_sub_ = nh_.subscribe(
      reference_topic_, 1, &VelocityQPMPC::referenceCallback, this);

  velocity_command_pub_ = nh_.advertise<std_msgs::Float64MultiArray>(
      velocity_command_topic_, 1);
  prediction_pub_ = nh_.advertise<trajectory_msgs::JointTrajectory>(
      prediction_topic_, 1);
  solve_time_pub_ = pnh_.advertise<std_msgs::Float32>("solve_time_ms", 1);
  summary_pub_ = pnh_.advertise<std_msgs::String>("summary", 1);

  timer_ = nh_.createTimer(
      ros::Duration(1.0 / rate_), &VelocityQPMPC::timerCallback, this);

  ROS_WARN("[VelocityQPMPC] Phase A owns the low-level velocity command topic. Do NOT run TrajectoryExecutionManager simultaneously.");
  ROS_INFO_STREAM("[VelocityQPMPC] solver=PIQP dense v0.6.2, rate=" << rate_
                  << " Hz, horizon=" << horizon_duration_
                  << " s, K=" << num_intervals_
                  << ", dt=" << dt_
                  << ", variables=" << n_u_
                  << ", general_constraints=" << n_constraints_);
  ROS_INFO_STREAM("[VelocityQPMPC] reference=" << reference_topic_
                  << ", command=" << velocity_command_topic_);
  return true;
}

bool VelocityQPMPC::loadJointVectorParam(const std::string& param_name,
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
      ROS_ERROR_STREAM("[VelocityQPMPC] " << param_name
                       << " array size must be " << dof_);
      return false;
    }
    for (int i = 0; i < dof_; ++i) {
      if (param[i].getType() == XmlRpc::XmlRpcValue::TypeDouble) {
        value[i] = static_cast<double>(param[i]);
      } else if (param[i].getType() == XmlRpc::XmlRpcValue::TypeInt) {
        value[i] = static_cast<int>(param[i]);
      } else {
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
        return false;
      }
    }
    return true;
  }

  ROS_ERROR_STREAM("[VelocityQPMPC] unsupported parameter type for " << param_name);
  return false;
}

bool VelocityQPMPC::loadConfig() {
  if (!pnh_.getParam("joint_names", joint_names_)) {
    ROS_ERROR("[VelocityQPMPC] Missing param: joint_names");
    return false;
  }
  if (joint_names_.size() != 7) {
    ROS_ERROR("[VelocityQPMPC] Phase A currently expects exactly 7 joints.");
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
    ROS_ERROR("[VelocityQPMPC] Invalid timing/PIQP parameters.");
    return false;
  }
  if (std::min(std::min(q_tracking_weight_, terminal_q_tracking_weight_),
               std::min(u_tracking_weight_, u_smooth_weight_)) < 0.0) {
    ROS_ERROR("[VelocityQPMPC] MPC weights must be non-negative.");
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
      ROS_ERROR("[VelocityQPMPC] velocity/acceleration limits must be positive.");
      return false;
    }
  }
  return true;
}

bool VelocityQPMPC::loadPositionLimitsFromUrdf() {
  std::string urdf;
  if (!nh_.getParam(robot_description_param_, urdf)) {
    ROS_ERROR_STREAM("[VelocityQPMPC] Could not read " << robot_description_param_);
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
        ROS_ERROR_STREAM("[VelocityQPMPC] invalid position limit entry for " << name);
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
      ROS_ERROR_STREAM("[VelocityQPMPC] invalid position limits for "
                       << joint_names_[static_cast<std::size_t>(i)]);
      return false;
    }
  }
  return true;
}

bool VelocityQPMPC::buildStaticQP() {
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

  // Velocity bounds are native PIQP variable bounds. Phase A has no nonlinear
  // CDF/NCDF linearization yet, so G contains only hard physical acceleration
  // and joint-position constraints. Nominal tracking is purely in the cost.
  acceleration_row0_ = 0;
  position_row0_ = acceleration_row0_ + n_u_;
  n_constraints_ = position_row0_ + n_u_;

  G_ = Eigen::MatrixXd::Zero(n_constraints_, n_u_);
  G_.block(acceleration_row0_, 0, n_u_, n_u_) = D_;
  G_.block(position_row0_, 0, n_u_, n_u_) = S_;

  H_ = 2.0 * (
      q_tracking_weight_ * S_.transpose() * S_
      + terminal_q_tracking_weight_ * S_terminal_.transpose() * S_terminal_
      + u_tracking_weight_ * Eigen::MatrixXd::Identity(n_u_, n_u_)
      + u_smooth_weight_ * D_.transpose() * D_);
  H_.diagonal().array() += 1e-8;

  x_lower_ = Eigen::VectorXd::Zero(n_u_);
  x_upper_ = Eigen::VectorXd::Zero(n_u_);
  for (int k = 0; k < num_intervals_; ++k) {
    x_lower_.segment(k * dof_, dof_) = -velocity_limits_;
    x_upper_.segment(k * dof_, dof_) = velocity_limits_;
  }
  return true;
}

void VelocityQPMPC::jointStateCallback(const sensor_msgs::JointStateConstPtr& msg) {
  if (!msg) return;
  std::lock_guard<std::mutex> lock(data_mutex_);
  latest_joint_state_ = *msg;
  latest_joint_state_received_ = ros::Time::now();
  has_joint_state_ = true;
}

void VelocityQPMPC::referenceCallback(
    const trajectory_msgs::JointTrajectoryConstPtr& msg) {
  if (!msg || msg->points.empty()) return;
  std::lock_guard<std::mutex> lock(data_mutex_);
  latest_reference_ = *msg;
  latest_reference_received_ = ros::Time::now();
  has_reference_ = true;
}

bool VelocityQPMPC::extractMeasuredQ(const sensor_msgs::JointState& msg,
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

bool VelocityQPMPC::buildTrajectoryMapping(
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

bool VelocityQPMPC::sampleReferencePosition(
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

bool VelocityQPMPC::buildReferenceHorizon(
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

void VelocityQPMPC::buildCycleQP(const Eigen::VectorXd& q_current,
                                 const Eigen::MatrixXd& q_ref,
                                 const Eigen::MatrixXd& u_ref,
                                 Eigen::VectorXd& gradient,
                                 Eigen::VectorXd& lower,
                                 Eigen::VectorXd& upper) const {
  Eigen::VectorXd q_error_stack(n_u_);
  for (int k = 0; k < num_intervals_; ++k) {
    q_error_stack.segment(k * dof_, dof_) = q_current - q_ref.col(k + 1);
  }
  const Eigen::VectorXd q_terminal_error = q_current - q_ref.col(num_intervals_);

  Eigen::VectorXd u_ref_stack(n_u_);
  for (int k = 0; k < num_intervals_; ++k) {
    u_ref_stack.segment(k * dof_, dof_) = u_ref.col(k);
  }

  Eigen::VectorXd smooth_offset = Eigen::VectorXd::Zero(n_u_);
  smooth_offset.head(dof_) = -previous_command_;

  gradient = 2.0 * (
      q_tracking_weight_ * S_.transpose() * q_error_stack
      + terminal_q_tracking_weight_ * S_terminal_.transpose() * q_terminal_error
      - u_tracking_weight_ * u_ref_stack
      + u_smooth_weight_ * D_.transpose() * smooth_offset);

  lower = Eigen::VectorXd::Zero(n_constraints_);
  upper = Eigen::VectorXd::Zero(n_constraints_);

  // D*u stores [u0, u1-u0, ...]. The first block is bounded around the previous
  // commanded velocity rather than raw JointState.velocity.
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

bool VelocityQPMPC::solveWithPIQP(const Eigen::VectorXd& gradient,
                                  const Eigen::VectorXd& lower,
                                  const Eigen::VectorXd& upper,
                                  Eigen::VectorXd& solution,
                                  int& iterations,
                                  double& primal_residual,
                                  double& dual_residual,
                                  std::string& status_string) {
  if (gradient.size() != n_u_ || lower.size() != n_constraints_ ||
      upper.size() != n_constraints_) {
    status_string = "dimension_mismatch";
    return false;
  }
  for (int i = 0; i < n_constraints_; ++i) {
    if (lower[i] > upper[i]) {
      status_string = "invalid_bounds";
      return false;
    }
  }

  if (!piqp_setup_done_) {
    piqp_solver_.setup(H_, gradient,
                       piqp::nullopt, piqp::nullopt,
                       G_, lower, upper,
                       x_lower_, x_upper_);
    piqp_setup_done_ = true;
  } else {
    // H, G and variable velocity bounds are static in Phase A. Keep PIQP's
    // factorization/preconditioner hot and update only the per-cycle linear cost
    // and general-constraint bounds.
    piqp_solver_.update(piqp::nullopt, gradient,
                        piqp::nullopt, piqp::nullopt,
                        piqp::nullopt, lower, upper);
  }

  const piqp::Status status = piqp_solver_.solve();
  const auto& result = piqp_solver_.result();
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

Eigen::MatrixXd VelocityQPMPC::reconstructPredictedQ(
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

void VelocityQPMPC::publishVelocity(const Eigen::VectorXd& command) {
  std_msgs::Float64MultiArray msg;
  msg.data.resize(static_cast<std::size_t>(dof_));
  for (int i = 0; i < dof_; ++i) {
    msg.data[static_cast<std::size_t>(i)] = command[i];
  }
  velocity_command_pub_.publish(msg);
}

void VelocityQPMPC::publishSafeStop(const std::string& reason) {
  Eigen::VectorXd command = previous_command_;
  for (int i = 0; i < dof_; ++i) {
    const double step = acceleration_limits_[i] * control_period_;
    if (command[i] > step) command[i] -= step;
    else if (command[i] < -step) command[i] += step;
    else command[i] = 0.0;
  }
  publishVelocity(command);
  previous_command_ = command;
  ROS_WARN_STREAM_THROTTLE(1.0, "[VelocityQPMPC] safe stop: " << reason);
}

void VelocityQPMPC::publishPrediction(const Eigen::MatrixXd& q_pred,
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

void VelocityQPMPC::timerCallback(const ros::TimerEvent&) {
  sensor_msgs::JointState joint_state;
  trajectory_msgs::JointTrajectory reference;
  ros::Time joint_received;
  ros::Time reference_received;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (!has_joint_state_) {
      publishSafeStop("waiting for JointState");
      return;
    }
    if (!has_reference_) {
      publishSafeStop("waiting for nominal reference");
      return;
    }
    joint_state = latest_joint_state_;
    reference = latest_reference_;
    joint_received = latest_joint_state_received_;
    reference_received = latest_reference_received_;
  }

  const ros::Time now = ros::Time::now();
  if ((now - joint_received).toSec() > joint_state_timeout_) {
    publishSafeStop("stale JointState");
    return;
  }
  if ((now - reference_received).toSec() > reference_timeout_) {
    publishSafeStop("stale nominal reference");
    return;
  }

  Eigen::VectorXd q_current;
  if (!extractMeasuredQ(joint_state, q_current)) {
    publishSafeStop("invalid JointState");
    return;
  }

  Eigen::MatrixXd q_ref;
  Eigen::MatrixXd u_ref;
  if (!buildReferenceHorizon(reference, q_ref, u_ref)) {
    publishSafeStop("invalid nominal reference");
    return;
  }

  Eigen::VectorXd gradient, lower, upper;
  buildCycleQP(q_current, q_ref, u_ref, gradient, lower, upper);

  const ros::WallTime tic = ros::WallTime::now();
  Eigen::VectorXd solution;
  int iterations = 0;
  double primal = 0.0;
  double dual = 0.0;
  std::string piqp_status;
  if (!solveWithPIQP(gradient, lower, upper, solution,
                     iterations, primal, dual, piqp_status)) {
    publishSafeStop("PIQP " + piqp_status);
    return;
  }
  const double solve_ms = (ros::WallTime::now() - tic).toSec() * 1000.0;

  const Eigen::VectorXd command = solution.head(dof_);
  publishVelocity(command);
  previous_command_ = command;

  const Eigen::MatrixXd q_pred = reconstructPredictedQ(q_current, solution);
  publishPrediction(q_pred, solution, reference.header.frame_id);

  std_msgs::Float32 solve_msg;
  solve_msg.data = static_cast<float>(solve_ms);
  solve_time_pub_.publish(solve_msg);

  double pred_dev_inf = 0.0;
  for (int k = 0; k <= num_intervals_; ++k) {
    pred_dev_inf = std::max(
        pred_dev_inf, (q_pred.col(k) - q_ref.col(k)).lpNorm<Eigen::Infinity>());
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
      << " tracking_inf=" << tracking_inf
      << " command_inf=" << command_inf
      << " pred_dev_inf=" << pred_dev_inf
      << " ref_horizon="
      << (reference.points.empty() ? 0.0
          : reference.points.back().time_from_start.toSec());

  std_msgs::String summary;
  summary.data = oss.str();
  summary_pub_.publish(summary);
  ROS_INFO_STREAM_THROTTLE(0.5, "[VelocityQPMPC] " << summary.data);
}

}  // namespace egocentric_arm_planner
