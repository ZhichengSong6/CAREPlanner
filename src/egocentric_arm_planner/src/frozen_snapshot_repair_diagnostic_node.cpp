#include <ros/ros.h>

#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/String.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>

#include <Eigen/Dense>
#include <piqp/piqp.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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

std::unordered_map<std::string, std::string> tokens(const std::string& text) {
  std::unordered_map<std::string, std::string> out;
  std::istringstream iss(text);
  std::string token;
  while (iss >> token) {
    const std::size_t p = token.find('=');
    if (p == std::string::npos || p == 0 || p + 1 >= token.size()) continue;
    out[token.substr(0, p)] = token.substr(p + 1);
  }
  return out;
}

bool parseBool01(const std::unordered_map<std::string, std::string>& fields,
                 const std::string& key,
                 bool& value) {
  const auto it = fields.find(key);
  if (it == fields.end()) return false;
  if (it->second == "1" || it->second == "true" || it->second == "True") {
    value = true;
    return true;
  }
  if (it->second == "0" || it->second == "false" || it->second == "False") {
    value = false;
    return true;
  }
  return false;
}

double parseDouble(const std::unordered_map<std::string, std::string>& fields,
                   const std::string& key,
                   double fallback = std::numeric_limits<double>::quiet_NaN()) {
  const auto it = fields.find(key);
  if (it == fields.end()) return fallback;
  try {
    return std::stod(it->second);
  } catch (...) {
    return fallback;
  }
}

std::string jsonNumber(double value) {
  if (!std::isfinite(value)) return "null";
  std::ostringstream oss;
  oss << std::setprecision(12) << value;
  return oss.str();
}

std::string jsonEscape(const std::string& value) {
  std::ostringstream oss;
  for (char c : value) {
    switch (c) {
      case '\\': oss << "\\\\"; break;
      case '"': oss << "\\\""; break;
      case '\n': oss << "\\n"; break;
      case '\r': oss << "\\r"; break;
      case '\t': oss << "\\t"; break;
      default: oss << c; break;
    }
  }
  return oss.str();
}

struct CandidateResult {
  double requested_horizon_s = 0.0;
  double actual_horizon_s = 0.0;
  int intervals = 0;
  bool qp_solved = false;
  double solve_ms = std::numeric_limits<double>::quiet_NaN();
  std::string solver_status = "not_run";
  double terminal_error_inf = std::numeric_limits<double>::quiet_NaN();
  double terminal_error_l2 = std::numeric_limits<double>::quiet_NaN();
  double terminal_motion_inf = std::numeric_limits<double>::quiet_NaN();
  double terminal_motion_l2 = std::numeric_limits<double>::quiet_NaN();
  double max_abs_velocity = std::numeric_limits<double>::quiet_NaN();
  bool vbc_done = false;
  bool vbc_unsafe = false;
  std::string vbc_reason = "not_checked";
  int vbc_violation_count = -1;
  int vbc_candidate_count = -1;
  double vbc_margin_s = std::numeric_limits<double>::quiet_NaN();
  double vbc_sweep_time_s = std::numeric_limits<double>::quiet_NaN();
  double vbc_see_time_s = std::numeric_limits<double>::quiet_NaN();
  trajectory_msgs::JointTrajectory trajectory;
};

class FrozenSnapshotRepairDiagnostic {
 public:
  FrozenSnapshotRepairDiagnostic()
      : nh_(), pnh_("~") {
    if (!loadConfig()) {
      throw std::runtime_error("failed to load frozen-snapshot diagnostic config");
    }

    joint_sub_ = nh_.subscribe(
        joint_state_topic_, 10,
        &FrozenSnapshotRepairDiagnostic::jointStateCb, this);
    actuator_sub_ = nh_.subscribe(
        actuator_topic_, 10,
        &FrozenSnapshotRepairDiagnostic::actuatorCb, this);
    q_vis_sub_ = nh_.subscribe(
        q_vis_topic_, 10,
        &FrozenSnapshotRepairDiagnostic::qVisCb, this);
    reference_sub_ = nh_.subscribe(
        reference_topic_, 5,
        &FrozenSnapshotRepairDiagnostic::referenceCb, this);
    mpc_summary_sub_ = nh_.subscribe(
        mpc_summary_topic_, 20,
        &FrozenSnapshotRepairDiagnostic::mpcSummaryCb, this);
    vbc_summary_sub_ = nh_.subscribe(
        vbc_summary_topic_, 20,
        &FrozenSnapshotRepairDiagnostic::vbcSummaryCb, this);

    candidate_pub_ = nh_.advertise<trajectory_msgs::JointTrajectory>(
        candidate_topic_, 1);
    summary_pub_ = nh_.advertise<std_msgs::String>(summary_topic_, 1, true);

    publishSummary("waiting_first_repair_cycle");
    ROS_WARN_STREAM(
        "[FrozenSnapshotRepairDiagnostic] waiting for first actual REPAIR MPC "
        "cycle; horizons=" << horizonString()
        << " dt=" << dt_
        << " output_dir=" << output_dir_);
  }

 private:
  bool loadJointVectorParam(const std::string& name,
                            const Eigen::VectorXd& fallback,
                            Eigen::VectorXd& value) {
    XmlRpc::XmlRpcValue param;
    if (!pnh_.getParam(name, param)) {
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
      if (param.size() != dof_) return false;
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
        const std::string& joint = joint_names_[static_cast<std::size_t>(i)];
        if (!param.hasMember(joint)) {
          value[i] = fallback[i];
          continue;
        }
        const XmlRpc::XmlRpcValue& entry = param[joint];
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
    return false;
  }

  bool loadPositionLimits() {
    Eigen::VectorXd default_min(7), default_max(7);
    default_min << -3.14, -2.30, -3.14, -2.65, -3.14, -3.14, -1.20;
    default_max <<  3.14,  2.30,  3.14,  2.65,  3.14,  3.14,  1.20;
    q_min_ = default_min;
    q_max_ = default_max;

    XmlRpc::XmlRpcValue limits;
    if (pnh_.getParam("mpc/joint_position_limits", limits) &&
        limits.getType() == XmlRpc::XmlRpcValue::TypeStruct) {
      for (int i = 0; i < dof_; ++i) {
        const std::string& joint = joint_names_[static_cast<std::size_t>(i)];
        if (!limits.hasMember(joint)) continue;
        const XmlRpc::XmlRpcValue& entry = limits[joint];
        if (entry.getType() != XmlRpc::XmlRpcValue::TypeStruct ||
            !entry.hasMember("lower") || !entry.hasMember("upper")) {
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
      if (!(q_min_[i] < q_max_[i])) return false;
    }
    return true;
  }

  bool loadConfig() {
    if (!pnh_.getParam("joint_names", joint_names_)) {
      ROS_ERROR("[FrozenSnapshotRepairDiagnostic] missing joint_names");
      return false;
    }
    dof_ = static_cast<int>(joint_names_.size());
    if (dof_ != 7) {
      ROS_ERROR("[FrozenSnapshotRepairDiagnostic] expected 7 joints");
      return false;
    }

    pnh_.param<double>("mpc/rate", rate_, 20.0);
    double base_horizon = 1.0;
    int base_intervals = 20;
    pnh_.param<double>("mpc/horizon_duration", base_horizon, 1.0);
    pnh_.param<int>("mpc/num_intervals", base_intervals, 20);
    if (!(rate_ > 0.0) || !(base_horizon > 0.0) || base_intervals < 2) return false;
    dt_ = base_horizon / static_cast<double>(base_intervals);
    control_period_ = 1.0 / rate_;

    pnh_.param<double>("mpc/u_tracking_weight", u_tracking_weight_, 2.0);
    pnh_.param<double>("mpc/u_smooth_weight", u_smooth_weight_, 1.0);
    pnh_.param<double>("mpc/visibility_waypoint/weight", waypoint_weight_, 3000.0);
    pnh_.param<double>("mpc/visibility_waypoint/recovery_weight_scale",
                       recovery_weight_scale_, 1.0);
    pnh_.param<double>("mpc/joint_position_margin", joint_position_margin_, 0.01);
    pnh_.param<int>("mpc/piqp/max_iterations", piqp_max_iterations_, 100);
    pnh_.param<double>("mpc/piqp/eps_abs", piqp_eps_abs_, 1e-6);
    pnh_.param<double>("mpc/piqp/eps_rel", piqp_eps_rel_, 1e-6);

    Eigen::VectorXd default_velocity(7), default_acceleration(7);
    default_velocity << 2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5;
    default_acceleration << 3.0, 3.0, 4.0, 4.0, 6.0, 6.0, 6.0;
    if (!loadJointVectorParam("mpc/joint_velocity_limits",
                              default_velocity, velocity_limits_) ||
        !loadJointVectorParam("mpc/joint_acceleration_limits",
                              default_acceleration, acceleration_limits_)) {
      return false;
    }
    if (!loadPositionLimits()) return false;

    XmlRpc::XmlRpcValue horizons;
    if (pnh_.getParam("horizons_s", horizons) &&
        horizons.getType() == XmlRpc::XmlRpcValue::TypeArray) {
      for (int i = 0; i < horizons.size(); ++i) {
        if (horizons[i].getType() == XmlRpc::XmlRpcValue::TypeDouble) {
          horizons_s_.push_back(static_cast<double>(horizons[i]));
        } else if (horizons[i].getType() == XmlRpc::XmlRpcValue::TypeInt) {
          horizons_s_.push_back(static_cast<int>(horizons[i]));
        }
      }
    }
    if (horizons_s_.empty()) horizons_s_ = {0.5, 1.0, 1.5, 2.0};
    horizons_s_.erase(
        std::remove_if(horizons_s_.begin(), horizons_s_.end(),
                       [](double h) { return !(h > 0.0) || !std::isfinite(h); }),
        horizons_s_.end());
    std::sort(horizons_s_.begin(), horizons_s_.end());
    if (horizons_s_.empty()) return false;

    pnh_.param<std::string>("joint_state_topic", joint_state_topic_,
                            "/care_arm/joint_states");
    pnh_.param<std::string>("actuator_topic", actuator_topic_,
                            "/care_arm/arm_group_velocity_controller/command");
    pnh_.param<std::string>("q_vis_topic", q_vis_topic_,
                            "/care_planner/active_sensing/visibility_waypoint_q");
    pnh_.param<std::string>("reference_topic", reference_topic_,
                            "/care_planner/command_trajectory_vbc_gated");
    pnh_.param<std::string>("mpc_summary_topic", mpc_summary_topic_,
                            "/velocity_qp_mpc_waypoint_node/summary");
    pnh_.param<std::string>("candidate_topic", candidate_topic_,
                            "/care_planner/diagnostic/frozen_candidate");
    pnh_.param<std::string>("vbc_summary_topic", vbc_summary_topic_,
                            "/care_planner/diagnostic/frozen_vbc_summary");
    pnh_.param<std::string>("summary_topic", summary_topic_,
                            "/care_planner/diagnostic/frozen_snapshot_summary");
    pnh_.param<std::string>("output_dir", output_dir_, ".");
    pnh_.param<std::string>("base_frame", base_frame_, "base_link");

    return waypoint_weight_ >= 0.0 && recovery_weight_scale_ > 0.0 &&
           u_tracking_weight_ >= 0.0 && u_smooth_weight_ >= 0.0 &&
           piqp_max_iterations_ > 0 && piqp_eps_abs_ > 0.0 &&
           piqp_eps_rel_ >= 0.0;
  }

  std::string horizonString() const {
    std::ostringstream oss;
    for (std::size_t i = 0; i < horizons_s_.size(); ++i) {
      if (i) oss << ",";
      oss << horizons_s_[i];
    }
    return oss.str();
  }

  bool extractQ(const sensor_msgs::JointState& msg, Eigen::VectorXd& q) const {
    std::unordered_map<std::string, std::size_t> index;
    for (std::size_t i = 0; i < msg.name.size(); ++i) index[msg.name[i]] = i;
    q = Eigen::VectorXd::Zero(dof_);
    for (int j = 0; j < dof_; ++j) {
      const auto it = index.find(joint_names_[static_cast<std::size_t>(j)]);
      if (it == index.end() || it->second >= msg.position.size()) return false;
      q[j] = msg.position[it->second];
    }
    return finiteVector(q);
  }

  void jointStateCb(const sensor_msgs::JointStateConstPtr& msg) {
    if (!msg) return;
    std::lock_guard<std::mutex> lock(mutex_);
    latest_joint_state_ = *msg;
    has_joint_state_ = true;
  }

  void actuatorCb(const std_msgs::Float64MultiArrayConstPtr& msg) {
    if (!msg || msg->data.size() != static_cast<std::size_t>(dof_)) return;
    Eigen::VectorXd u(dof_);
    for (int i = 0; i < dof_; ++i) u[i] = msg->data[static_cast<std::size_t>(i)];
    if (!finiteVector(u)) return;
    std::lock_guard<std::mutex> lock(mutex_);
    latest_actuator_ = u;
    has_actuator_ = true;
  }

  void qVisCb(const std_msgs::Float64MultiArrayConstPtr& msg) {
    if (!msg || msg->data.size() != static_cast<std::size_t>(dof_)) return;
    Eigen::VectorXd q(dof_);
    for (int i = 0; i < dof_; ++i) q[i] = msg->data[static_cast<std::size_t>(i)];
    if (!finiteVector(q)) return;
    std::lock_guard<std::mutex> lock(mutex_);
    latest_q_vis_ = q;
    has_q_vis_ = true;
  }

  void referenceCb(const trajectory_msgs::JointTrajectoryConstPtr& msg) {
    if (!msg || msg->points.empty()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    latest_reference_ = *msg;
    has_reference_ = true;
  }

  void mpcSummaryCb(const std_msgs::StringConstPtr& msg) {
    if (!msg) return;
    const auto fields = tokens(msg->data);
    const auto it_mode = fields.find("control_mode");
    if (it_mode == fields.end() || it_mode->second != "recovery") return;
    const auto it_sem = fields.find("planner_mode_semantics");
    if (it_sem == fields.end() || it_sem->second != "1") return;

    sensor_msgs::JointState joint;
    Eigen::VectorXd u_exec;
    Eigen::VectorXd q_vis;
    trajectory_msgs::JointTrajectory reference;
    bool have_reference = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (snapshot_started_ || snapshot_ready_) return;
      if (!has_joint_state_ || !has_actuator_ || !has_q_vis_) {
        ROS_WARN_THROTTLE(
            0.5,
            "[FrozenSnapshotRepairDiagnostic] REPAIR seen but snapshot inputs not ready");
        return;
      }
      snapshot_started_ = true;
      joint = latest_joint_state_;
      u_exec = latest_actuator_;
      q_vis = latest_q_vis_;
      have_reference = has_reference_;
      if (have_reference) reference = latest_reference_;
    }

    Eigen::VectorXd q0;
    if (!extractQ(joint, q0)) {
      std::lock_guard<std::mutex> lock(mutex_);
      snapshot_started_ = false;
      ROS_ERROR("[FrozenSnapshotRepairDiagnostic] failed to extract snapshot q");
      return;
    }

    takeSnapshotAndSolve(q0, u_exec, q_vis, reference, have_reference);
  }

  CandidateResult solveRepairCandidate(double requested_horizon,
                                       const Eigen::VectorXd& q0,
                                       const Eigen::VectorXd& u_exec,
                                       const Eigen::VectorXd& q_vis) const {
    CandidateResult out;
    out.requested_horizon_s = requested_horizon;
    const int K = std::max(2, static_cast<int>(std::llround(requested_horizon / dt_)));
    const int n_u = dof_ * K;
    out.intervals = K;
    out.actual_horizon_s = K * dt_;

    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(dof_ * K, n_u);
    for (int k = 0; k < K; ++k) {
      for (int j = 0; j <= k; ++j) {
        S.block(k * dof_, j * dof_, dof_, dof_).setIdentity();
        S.block(k * dof_, j * dof_, dof_, dof_) *= dt_;
      }
    }
    const Eigen::MatrixXd S_terminal = S.bottomRows(dof_);

    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(n_u, n_u);
    D.topLeftCorner(dof_, dof_).setIdentity();
    for (int k = 1; k < K; ++k) {
      D.block(k * dof_, k * dof_, dof_, dof_).setIdentity();
      D.block(k * dof_, (k - 1) * dof_, dof_, dof_) =
          -Eigen::MatrixXd::Identity(dof_, dof_);
    }

    const double weight = waypoint_weight_ * recovery_weight_scale_;
    Eigen::MatrixXd H = 2.0 * (
        u_tracking_weight_ * Eigen::MatrixXd::Identity(n_u, n_u) +
        u_smooth_weight_ * D.transpose() * D);
    H += 2.0 * weight * S_terminal.transpose() * S_terminal;
    H.diagonal().array() += 1e-8;

    Eigen::VectorXd smooth_offset = Eigen::VectorXd::Zero(n_u);
    smooth_offset.head(dof_) = -u_exec;
    Eigen::VectorXd g =
        2.0 * u_smooth_weight_ * D.transpose() * smooth_offset;
    g += 2.0 * weight * S_terminal.transpose() * (q0 - q_vis);

    const int n_constraints = 2 * n_u;
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(n_constraints, n_u);
    G.block(0, 0, n_u, n_u) = D;
    G.block(n_u, 0, n_u, n_u) = S;

    Eigen::VectorXd lower = Eigen::VectorXd::Zero(n_constraints);
    Eigen::VectorXd upper = Eigen::VectorXd::Zero(n_constraints);
    lower.head(dof_) = u_exec - acceleration_limits_ * control_period_;
    upper.head(dof_) = u_exec + acceleration_limits_ * control_period_;
    for (int k = 1; k < K; ++k) {
      lower.segment(k * dof_, dof_) = -acceleration_limits_ * dt_;
      upper.segment(k * dof_, dof_) = acceleration_limits_ * dt_;
    }
    for (int k = 0; k < K; ++k) {
      lower.segment(n_u + k * dof_, dof_) = q_min_ - q0;
      upper.segment(n_u + k * dof_, dof_) = q_max_ - q0;
    }

    Eigen::VectorXd x_lower = Eigen::VectorXd::Zero(n_u);
    Eigen::VectorXd x_upper = Eigen::VectorXd::Zero(n_u);
    for (int k = 0; k < K; ++k) {
      x_lower.segment(k * dof_, dof_) = -velocity_limits_;
      x_upper.segment(k * dof_, dof_) = velocity_limits_;
    }

    if (!finiteMatrix(H) || !finiteVector(g) || !finiteMatrix(G) ||
        !finiteVector(lower) || !finiteVector(upper)) {
      out.solver_status = "nonfinite_qp";
      return out;
    }

    piqp::DenseSolver<double> solver;
    auto& settings = solver.settings();
    settings.max_iter = piqp_max_iterations_;
    settings.eps_abs = piqp_eps_abs_;
    settings.eps_rel = piqp_eps_rel_;
    settings.verbose = false;
    settings.compute_timings = true;

    const ros::WallTime tic = ros::WallTime::now();
    solver.setup(H, g, piqp::nullopt, piqp::nullopt,
                 G, lower, upper, x_lower, x_upper);
    const piqp::Status status = solver.solve();
    out.solve_ms = (ros::WallTime::now() - tic).toSec() * 1000.0;
    out.solver_status = piqp::status_to_string(status);
    if (status != piqp::PIQP_SOLVED || !finiteVector(solver.result().x)) {
      return out;
    }

    const Eigen::VectorXd solution = solver.result().x;
    out.qp_solved = true;

    trajectory_msgs::JointTrajectory traj;
    traj.header.frame_id = base_frame_;
    traj.joint_names = joint_names_;
    traj.points.resize(static_cast<std::size_t>(K + 1));
    Eigen::VectorXd q = q0;
    double max_abs_velocity = 0.0;
    for (int k = 0; k <= K; ++k) {
      auto& point = traj.points[static_cast<std::size_t>(k)];
      point.time_from_start = ros::Duration(k * dt_);
      point.positions.resize(static_cast<std::size_t>(dof_));
      point.velocities.resize(static_cast<std::size_t>(dof_));
      for (int j = 0; j < dof_; ++j) {
        point.positions[static_cast<std::size_t>(j)] = q[j];
        const double velocity = k < K ? solution[k * dof_ + j] : 0.0;
        point.velocities[static_cast<std::size_t>(j)] = velocity;
        max_abs_velocity = std::max(max_abs_velocity, std::abs(velocity));
      }
      if (k < K) q += dt_ * solution.segment(k * dof_, dof_);
    }
    out.trajectory = traj;
    out.max_abs_velocity = max_abs_velocity;
    out.terminal_error_inf = (q - q_vis).lpNorm<Eigen::Infinity>();
    out.terminal_error_l2 = (q - q_vis).norm();
    out.terminal_motion_inf = (q - q0).lpNorm<Eigen::Infinity>();
    out.terminal_motion_l2 = (q - q0).norm();
    return out;
  }

  std::pair<double, double> reachableDisplacementRange(
      double T, double v0, double vmax, double amax) const {
    const double t_up = std::max(0.0, (vmax - v0) / amax);
    double d_max = 0.0;
    if (T <= t_up) {
      d_max = v0 * T + 0.5 * amax * T * T;
    } else {
      d_max = v0 * t_up + 0.5 * amax * t_up * t_up +
              vmax * (T - t_up);
    }

    const double t_down = std::max(0.0, (v0 + vmax) / amax);
    double d_min = 0.0;
    if (T <= t_down) {
      d_min = v0 * T - 0.5 * amax * T * T;
    } else {
      d_min = v0 * t_down - 0.5 * amax * t_down * t_down -
              vmax * (T - t_down);
    }
    return {d_min, d_max};
  }

  double minTimeOneJoint(double delta, double v0,
                         double vmax, double amax) const {
    if (std::abs(delta) <= 1e-10) return 0.0;
    double lo = 0.0;
    double hi = 0.05;
    for (int grow = 0; grow < 30; ++grow) {
      const auto range = reachableDisplacementRange(hi, v0, vmax, amax);
      if (delta >= range.first - 1e-10 && delta <= range.second + 1e-10) break;
      hi *= 2.0;
    }
    const auto final_range = reachableDisplacementRange(hi, v0, vmax, amax);
    if (delta < final_range.first - 1e-9 || delta > final_range.second + 1e-9) {
      return std::numeric_limits<double>::infinity();
    }
    for (int iter = 0; iter < 80; ++iter) {
      const double mid = 0.5 * (lo + hi);
      const auto range = reachableDisplacementRange(mid, v0, vmax, amax);
      if (delta >= range.first - 1e-12 && delta <= range.second + 1e-12) {
        hi = mid;
      } else {
        lo = mid;
      }
    }
    return hi;
  }

  double dynamicMinTimeToQvis(const Eigen::VectorXd& q0,
                              const Eigen::VectorXd& u_exec,
                              const Eigen::VectorXd& q_vis,
                              std::vector<double>& per_joint) const {
    per_joint.clear();
    double overall = 0.0;
    for (int j = 0; j < dof_; ++j) {
      const double t = minTimeOneJoint(
          q_vis[j] - q0[j], u_exec[j],
          velocity_limits_[j], acceleration_limits_[j]);
      per_joint.push_back(t);
      overall = std::max(overall, t);
    }
    return overall;
  }

  void takeSnapshotAndSolve(const Eigen::VectorXd& q0,
                            const Eigen::VectorXd& u_exec,
                            const Eigen::VectorXd& q_vis,
                            const trajectory_msgs::JointTrajectory& reference,
                            bool have_reference) {
    const ros::Time now = ros::Time::now();
    std::vector<CandidateResult> candidates;
    candidates.reserve(horizons_s_.size());
    for (double horizon : horizons_s_) {
      candidates.push_back(solveRepairCandidate(horizon, q0, u_exec, q_vis));
    }

    std::vector<double> per_joint;
    const double min_time = dynamicMinTimeToQvis(q0, u_exec, q_vis, per_joint);

    {
      std::lock_guard<std::mutex> lock(mutex_);
      snapshot_q0_ = q0;
      snapshot_u_exec_ = u_exec;
      snapshot_q_vis_ = q_vis;
      snapshot_time_ = now;
      snapshot_reference_duration_s_ =
          have_reference && !reference.points.empty()
              ? reference.points.back().time_from_start.toSec()
              : std::numeric_limits<double>::quiet_NaN();
      q_vis_distance_inf_ = (q_vis - q0).lpNorm<Eigen::Infinity>();
      q_vis_distance_l2_ = (q_vis - q0).norm();
      dynamic_min_time_to_q_vis_s_ = min_time;
      dynamic_min_time_per_joint_s_ = per_joint;
      candidates_ = std::move(candidates);
      snapshot_ready_ = true;
      snapshot_started_ = false;
      next_candidate_index_ = 0;
    }

    writeSnapshot();
    writeCandidateTrajectories();
    publishSummary("snapshot_solved_waiting_vbc_cycle");

    ROS_WARN_STREAM(
        "[FrozenSnapshotRepairDiagnostic] SNAPSHOT captured at first REPAIR "
        "cycle: |qvis-q0|inf=" << q_vis_distance_inf_
        << " dynamic_min_time=" << dynamic_min_time_to_q_vis_s_
        << " s; solved " << candidates_.size() << " frozen candidates");
    for (const auto& c : candidates_) {
      ROS_WARN_STREAM(
          "[FrozenSnapshotRepairDiagnostic] horizon=" << c.actual_horizon_s
          << " qp_solved=" << static_cast<int>(c.qp_solved)
          << " solve_ms=" << c.solve_ms
          << " terminal_error_inf=" << c.terminal_error_inf);
    }
  }

  void vbcSummaryCb(const std_msgs::StringConstPtr& msg) {
    if (!msg) return;
    const auto fields = tokens(msg->data);
    const auto it_source = fields.find("trajectory_source");
    const std::string source = it_source == fields.end() ? "" : it_source->second;
    bool violation = false;
    const bool have_violation = parseBool01(fields, "has_violation", violation);

    trajectory_msgs::JointTrajectory to_publish;
    bool publish_candidate = false;
    bool completed_now = false;

    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++selector_cycle_count_;
      if (!snapshot_ready_ || finalized_) return;

      if (outstanding_index_ >= 0) {
        if (source != "predicted" || !have_violation ||
            selector_cycle_count_ <= outstanding_dispatch_cycle_) {
          return;
        }
        CandidateResult& c = candidates_[static_cast<std::size_t>(outstanding_index_)];
        c.vbc_done = true;
        c.vbc_unsafe = violation;
        const auto it_reason = fields.find("reason");
        c.vbc_reason = it_reason == fields.end() ? "unknown" : it_reason->second;
        c.vbc_violation_count = static_cast<int>(parseDouble(fields, "violation_count", -1));
        c.vbc_candidate_count = static_cast<int>(parseDouble(fields, "candidate_count", -1));
        c.vbc_margin_s = parseDouble(fields, "margin_s");
        c.vbc_sweep_time_s = parseDouble(fields, "sweep_time_s");
        c.vbc_see_time_s = parseDouble(fields, "see_time_s");
        ROS_WARN_STREAM(
            "[FrozenSnapshotRepairDiagnostic] VBC horizon=" << c.actual_horizon_s
            << " safe=" << static_cast<int>(!c.vbc_unsafe)
            << " reason=" << c.vbc_reason
            << " violations=" << c.vbc_violation_count
            << " margin_s=" << c.vbc_margin_s);
        outstanding_index_ = -1;
      }

      while (next_candidate_index_ < candidates_.size() &&
             !candidates_[next_candidate_index_].qp_solved) {
        ++next_candidate_index_;
      }
      if (next_candidate_index_ < candidates_.size()) {
        CandidateResult& c = candidates_[next_candidate_index_];
        c.trajectory.header.seq = static_cast<uint32_t>(next_candidate_index_ + 1);
        c.trajectory.header.stamp = ros::Time::now();
        to_publish = c.trajectory;
        outstanding_index_ = static_cast<int>(next_candidate_index_);
        outstanding_dispatch_cycle_ = selector_cycle_count_;
        ++next_candidate_index_;
        publish_candidate = true;
      } else if (outstanding_index_ < 0) {
        finalized_ = true;
        completed_now = true;
      }
    }

    if (publish_candidate) {
      candidate_pub_.publish(to_publish);
      publishSummary("candidate_dispatched");
    }
    if (completed_now) {
      writeResults();
      publishSummary("complete");
      ROS_WARN_STREAM(
          "[FrozenSnapshotRepairDiagnostic] COMPLETE -> "
          << output_dir_ << "/frozen_snapshot_repair_results.json");
    }
  }

  void writeSnapshot() const {
    std::ofstream f(output_dir_ + "/frozen_snapshot.csv");
    if (!f) return;
    f << "joint,q0,u_exec,q_vis,dq_to_q_vis,min_time_to_q_vis_s\n";
    for (int j = 0; j < dof_; ++j) {
      const double tj = j < static_cast<int>(dynamic_min_time_per_joint_s_.size())
                            ? dynamic_min_time_per_joint_s_[static_cast<std::size_t>(j)]
                            : std::numeric_limits<double>::quiet_NaN();
      f << joint_names_[static_cast<std::size_t>(j)] << ","
        << std::setprecision(12) << snapshot_q0_[j] << ","
        << snapshot_u_exec_[j] << ","
        << snapshot_q_vis_[j] << ","
        << (snapshot_q_vis_[j] - snapshot_q0_[j]) << ","
        << tj << "\n";
    }
  }

  void writeCandidateTrajectories() const {
    for (const auto& c : candidates_) {
      if (!c.qp_solved) continue;
      std::ostringstream name;
      name << output_dir_ << "/candidate_h"
           << static_cast<int>(std::llround(c.actual_horizon_s * 1000.0))
           << "ms.csv";
      std::ofstream f(name.str());
      if (!f) continue;
      f << "t_s";
      for (const auto& joint : joint_names_) f << ",q_" << joint;
      for (const auto& joint : joint_names_) f << ",u_" << joint;
      f << "\n";
      for (const auto& p : c.trajectory.points) {
        f << std::setprecision(12) << p.time_from_start.toSec();
        for (double q : p.positions) f << "," << q;
        for (double u : p.velocities) f << "," << u;
        f << "\n";
      }
    }
  }

  void writeResults() const {
    std::ofstream csv(output_dir_ + "/frozen_snapshot_repair_results.csv");
    if (csv) {
      csv << "requested_horizon_s,actual_horizon_s,intervals,qp_solved,solve_ms,"
             "solver_status,terminal_error_inf,terminal_error_l2,terminal_motion_inf,"
             "terminal_motion_l2,max_abs_velocity,vbc_done,vbc_safe,vbc_reason,"
             "vbc_violation_count,vbc_candidate_count,vbc_margin_s,vbc_sweep_time_s,"
             "vbc_see_time_s\n";
      for (const auto& c : candidates_) {
        csv << std::setprecision(12)
            << c.requested_horizon_s << "," << c.actual_horizon_s << ","
            << c.intervals << "," << static_cast<int>(c.qp_solved) << ","
            << c.solve_ms << "," << c.solver_status << ","
            << c.terminal_error_inf << "," << c.terminal_error_l2 << ","
            << c.terminal_motion_inf << "," << c.terminal_motion_l2 << ","
            << c.max_abs_velocity << "," << static_cast<int>(c.vbc_done) << ","
            << static_cast<int>(c.vbc_done && !c.vbc_unsafe) << ","
            << c.vbc_reason << "," << c.vbc_violation_count << ","
            << c.vbc_candidate_count << "," << c.vbc_margin_s << ","
            << c.vbc_sweep_time_s << "," << c.vbc_see_time_s << "\n";
      }
    }

    std::ofstream json(output_dir_ + "/frozen_snapshot_repair_results.json");
    if (!json) return;
    json << "{\n";
    json << "  \"snapshot_ros_time_s\": " << jsonNumber(snapshot_time_.toSec()) << ",\n";
    json << "  \"dt_s\": " << jsonNumber(dt_) << ",\n";
    json << "  \"q_vis_distance_inf\": " << jsonNumber(q_vis_distance_inf_) << ",\n";
    json << "  \"q_vis_distance_l2\": " << jsonNumber(q_vis_distance_l2_) << ",\n";
    json << "  \"dynamic_min_time_to_q_vis_s\": "
         << jsonNumber(dynamic_min_time_to_q_vis_s_) << ",\n";
    json << "  \"snapshot_reference_duration_s\": "
         << jsonNumber(snapshot_reference_duration_s_) << ",\n";
    json << "  \"waypoint_weight\": "
         << jsonNumber(waypoint_weight_ * recovery_weight_scale_) << ",\n";
    json << "  \"candidates\": [\n";
    for (std::size_t i = 0; i < candidates_.size(); ++i) {
      const auto& c = candidates_[i];
      json << "    {\"requested_horizon_s\": " << jsonNumber(c.requested_horizon_s)
           << ", \"actual_horizon_s\": " << jsonNumber(c.actual_horizon_s)
           << ", \"intervals\": " << c.intervals
           << ", \"qp_solved\": " << (c.qp_solved ? "true" : "false")
           << ", \"solve_ms\": " << jsonNumber(c.solve_ms)
           << ", \"solver_status\": \"" << jsonEscape(c.solver_status) << "\""
           << ", \"terminal_error_inf\": " << jsonNumber(c.terminal_error_inf)
           << ", \"terminal_error_l2\": " << jsonNumber(c.terminal_error_l2)
           << ", \"terminal_motion_inf\": " << jsonNumber(c.terminal_motion_inf)
           << ", \"terminal_motion_l2\": " << jsonNumber(c.terminal_motion_l2)
           << ", \"vbc_done\": " << (c.vbc_done ? "true" : "false")
           << ", \"vbc_safe\": " << ((c.vbc_done && !c.vbc_unsafe) ? "true" : "false")
           << ", \"vbc_reason\": \"" << jsonEscape(c.vbc_reason) << "\""
           << ", \"vbc_violation_count\": " << c.vbc_violation_count
           << ", \"vbc_candidate_count\": " << c.vbc_candidate_count
           << ", \"vbc_margin_s\": " << jsonNumber(c.vbc_margin_s)
           << ", \"vbc_sweep_time_s\": " << jsonNumber(c.vbc_sweep_time_s)
           << ", \"vbc_see_time_s\": " << jsonNumber(c.vbc_see_time_s)
           << "}" << (i + 1 < candidates_.size() ? "," : "") << "\n";
    }
    json << "  ]\n";
    json << "}\n";
  }

  void publishSummary(const std::string& status) {
    std_msgs::String msg;
    std::ostringstream oss;
    std::lock_guard<std::mutex> lock(mutex_);
    oss << "status=" << status
        << " snapshot_ready=" << static_cast<int>(snapshot_ready_)
        << " finalized=" << static_cast<int>(finalized_)
        << " selector_cycles=" << selector_cycle_count_
        << " next_candidate_index=" << next_candidate_index_
        << " outstanding_index=" << outstanding_index_
        << " q_vis_distance_inf=" << q_vis_distance_inf_
        << " dynamic_min_time_to_q_vis_s=" << dynamic_min_time_to_q_vis_s_;
    msg.data = oss.str();
    summary_pub_.publish(msg);
  }

 private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber joint_sub_;
  ros::Subscriber actuator_sub_;
  ros::Subscriber q_vis_sub_;
  ros::Subscriber reference_sub_;
  ros::Subscriber mpc_summary_sub_;
  ros::Subscriber vbc_summary_sub_;
  ros::Publisher candidate_pub_;
  ros::Publisher summary_pub_;

  mutable std::mutex mutex_;
  sensor_msgs::JointState latest_joint_state_;
  Eigen::VectorXd latest_actuator_;
  Eigen::VectorXd latest_q_vis_;
  trajectory_msgs::JointTrajectory latest_reference_;
  bool has_joint_state_ = false;
  bool has_actuator_ = false;
  bool has_q_vis_ = false;
  bool has_reference_ = false;

  std::vector<std::string> joint_names_;
  int dof_ = 7;
  double rate_ = 20.0;
  double dt_ = 0.05;
  double control_period_ = 0.05;
  double u_tracking_weight_ = 2.0;
  double u_smooth_weight_ = 1.0;
  double waypoint_weight_ = 3000.0;
  double recovery_weight_scale_ = 1.0;
  double joint_position_margin_ = 0.01;
  int piqp_max_iterations_ = 100;
  double piqp_eps_abs_ = 1e-6;
  double piqp_eps_rel_ = 1e-6;
  Eigen::VectorXd velocity_limits_;
  Eigen::VectorXd acceleration_limits_;
  Eigen::VectorXd q_min_;
  Eigen::VectorXd q_max_;
  std::vector<double> horizons_s_;

  std::string joint_state_topic_;
  std::string actuator_topic_;
  std::string q_vis_topic_;
  std::string reference_topic_;
  std::string mpc_summary_topic_;
  std::string candidate_topic_;
  std::string vbc_summary_topic_;
  std::string summary_topic_;
  std::string output_dir_;
  std::string base_frame_;

  bool snapshot_started_ = false;
  bool snapshot_ready_ = false;
  bool finalized_ = false;
  ros::Time snapshot_time_;
  Eigen::VectorXd snapshot_q0_;
  Eigen::VectorXd snapshot_u_exec_;
  Eigen::VectorXd snapshot_q_vis_;
  double snapshot_reference_duration_s_ = std::numeric_limits<double>::quiet_NaN();
  double q_vis_distance_inf_ = std::numeric_limits<double>::quiet_NaN();
  double q_vis_distance_l2_ = std::numeric_limits<double>::quiet_NaN();
  double dynamic_min_time_to_q_vis_s_ = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> dynamic_min_time_per_joint_s_;
  std::vector<CandidateResult> candidates_;

  unsigned long long selector_cycle_count_ = 0;
  std::size_t next_candidate_index_ = 0;
  int outstanding_index_ = -1;
  unsigned long long outstanding_dispatch_cycle_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "frozen_snapshot_repair_diagnostic");
  try {
    FrozenSnapshotRepairDiagnostic node;
    ros::spin();
  } catch (const std::exception& e) {
    ROS_FATAL_STREAM("[FrozenSnapshotRepairDiagnostic] " << e.what());
    return 1;
  }
  return 0;
}
