#include "egocentric_arm_planner/trajectory_execution_manager.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace egocentric_arm_planner {

bool TrajectoryExecutionManager::initialize(const ros::NodeHandle& nh,
                                            const ros::NodeHandle& pnh) {
  nh_ = nh;
  pnh_ = pnh;

  if (!loadConfig()) {
    ROS_ERROR("[TrajectoryExecutionManager] Failed to load config.");
    return false;
  }

  if (joint_names_.empty()) {
    ROS_ERROR("[TrajectoryExecutionManager] joint_names is empty.");
    return false;
  }

  const int dof = static_cast<int>(joint_names_.size());
  q_measured_ = Eigen::VectorXd::Zero(dof);
  dq_measured_ = Eigen::VectorXd::Zero(dof);
  last_q_ref_ = Eigen::VectorXd::Zero(dof);
  last_dq_ref_ = Eigen::VectorXd::Zero(dof);

  joint_state_sub_ = nh_.subscribe(
      joint_state_topic_, 1,
      &TrajectoryExecutionManager::jointStateCallback, this);

  trajectory_sub_ = nh_.subscribe(
      input_trajectory_topic_, 1,
      &TrajectoryExecutionManager::trajectoryCallback, this);

  velocity_command_pub_ = nh_.advertise<std_msgs::Float64MultiArray>(
      output_velocity_command_topic_, 1);

  reference_state_pub_ = nh_.advertise<sensor_msgs::JointState>(
      reference_state_topic_, 1);

  summary_pub_ = nh_.advertise<std_msgs::String>(
      summary_topic_, 10);

  replan_request_pub_ = nh_.advertise<std_msgs::Bool>(
      replan_request_topic_, 10);

  execution_timer_ = nh_.createTimer(
      ros::Duration(1.0 / execution_rate_),
      &TrajectoryExecutionManager::executionTimerCallback,
      this);

  ROS_WARN(
      "[TrajectoryExecutionManager] C5.4 FULL-TRAJECTORY TRACKING ENABLED: "
      "execution advances through a committed trajectory independently of planner rate.");
  ROS_INFO_STREAM("[TrajectoryExecutionManager] execution_rate="
                  << execution_rate_
                  << " Hz input=" << input_trajectory_topic_
                  << " command=" << output_velocity_command_topic_
                  << " summary=" << summary_topic_);
  ROS_INFO_STREAM("[TrajectoryExecutionManager] Kp="
                  << position_feedback_gain_
                  << " vmax=" << max_command_velocity_
                  << " tracking_replan_threshold="
                  << replan_tracking_error_inf_);

  return true;
}

bool TrajectoryExecutionManager::loadConfig() {
  pnh_.param<double>("execution/rate", execution_rate_, execution_rate_);
  pnh_.param<double>("execution/control_dt", control_dt_, control_dt_);

  pnh_.param<double>("execution/max_start_error",
                     max_start_error_, max_start_error_);
  pnh_.param<double>("execution/max_tracking_error",
                     max_tracking_error_, max_tracking_error_);

  pnh_.param<bool>("execution/hold_when_no_trajectory",
                   hold_when_no_trajectory_, hold_when_no_trajectory_);
  pnh_.param<bool>("execution/hold_when_tracking_error_large",
                   hold_when_tracking_error_large_,
                   hold_when_tracking_error_large_);
  pnh_.param<bool>("execution/reject_large_start_error",
                   reject_large_start_error_,
                   reject_large_start_error_);
  pnh_.param<bool>("execution/hold_initial_zero_pose",
                   hold_initial_zero_pose_,
                   hold_initial_zero_pose_);
  pnh_.param<bool>("execution/hold_last_reference_when_no_trajectory",
                   hold_last_reference_when_no_trajectory_,
                   hold_last_reference_when_no_trajectory_);
  pnh_.param<double>("execution/reference_timeout",
                     reference_timeout_, reference_timeout_);

  // Backward-compatible fallback.
  pnh_.param<double>("execution/velocity_tracking_kp",
                     position_feedback_gain_,
                     position_feedback_gain_);
  pnh_.param<double>("execution/position_feedback_gain",
                     position_feedback_gain_,
                     position_feedback_gain_);
  pnh_.param<double>("execution/max_command_velocity",
                     max_command_velocity_,
                     max_command_velocity_);

  pnh_.param<double>("execution/replan_tracking_error_inf",
                     replan_tracking_error_inf_,
                     replan_tracking_error_inf_);
  pnh_.param<double>("execution/replan_request_min_interval",
                     replan_request_min_interval_s_,
                     replan_request_min_interval_s_);

  pnh_.param<std::string>("execution/joint_states",
                          joint_state_topic_, joint_state_topic_);
  pnh_.param<std::string>("execution/input_trajectory",
                          input_trajectory_topic_, input_trajectory_topic_);
  pnh_.param<std::string>("execution/output_velocity_command",
                          output_velocity_command_topic_,
                          output_velocity_command_topic_);
  pnh_.param<std::string>("execution/reference_state",
                          reference_state_topic_, reference_state_topic_);
  pnh_.param<std::string>("execution/summary_topic",
                          summary_topic_, summary_topic_);
  pnh_.param<std::string>("execution/replan_request_topic",
                          replan_request_topic_, replan_request_topic_);

  if (!pnh_.getParam("joint_names", joint_names_)) {
    ROS_ERROR("[TrajectoryExecutionManager] Missing param: joint_names");
    return false;
  }

  if (execution_rate_ <= 0.0) {
    ROS_ERROR("[TrajectoryExecutionManager] execution/rate must be positive.");
    return false;
  }
  if (control_dt_ <= 0.0) {
    ROS_ERROR("[TrajectoryExecutionManager] execution/control_dt must be positive.");
    return false;
  }
  if (max_command_velocity_ <= 0.0) {
    ROS_ERROR("[TrajectoryExecutionManager] execution/max_command_velocity must be positive.");
    return false;
  }
  if (position_feedback_gain_ < 0.0) {
    ROS_ERROR("[TrajectoryExecutionManager] execution/position_feedback_gain must be non-negative.");
    return false;
  }
  if (reference_timeout_ < 0.0 ||
      replan_request_min_interval_s_ < 0.0 ||
      replan_tracking_error_inf_ < 0.0) {
    ROS_ERROR("[TrajectoryExecutionManager] invalid timeout/replan configuration.");
    return false;
  }

  return true;
}

void TrajectoryExecutionManager::jointStateCallback(
    const sensor_msgs::JointStateConstPtr& msg) {
  if (!msg) return;

  Eigen::VectorXd q;
  Eigen::VectorXd dq;
  if (!extractMeasuredState(*msg, q, dq)) {
    ROS_WARN_THROTTLE(
        1.0,
        "[TrajectoryExecutionManager] Failed to extract measured joint state.");
    return;
  }

  std::lock_guard<std::mutex> lock(data_mutex_);
  latest_joint_state_ = *msg;
  q_measured_ = q;
  dq_measured_ = dq;
  has_joint_state_ = true;
}

void TrajectoryExecutionManager::trajectoryCallback(
    const trajectory_msgs::JointTrajectoryConstPtr& msg) {
  if (!msg || msg->points.empty()) {
    ROS_WARN_THROTTLE(
        1.0, "[TrajectoryExecutionManager] Received empty trajectory.");
    return;
  }
  if (!trajectoryHasExpectedJoints(*msg)) {
    ROS_WARN_THROTTLE(
        1.0,
        "[TrajectoryExecutionManager] Received trajectory with unexpected joints.");
    return;
  }

  std::vector<int> mapping;
  if (!buildTrajectoryJointIndexMap(*msg, mapping)) {
    ROS_WARN_THROTTLE(
        1.0,
        "[TrajectoryExecutionManager] Failed to build trajectory joint map.");
    return;
  }

  const double duration = getTrajectoryEndTime(*msg);
  if (!std::isfinite(duration) || duration < 0.0) {
    ROS_WARN_THROTTLE(
        1.0, "[TrajectoryExecutionManager] Invalid trajectory duration.");
    return;
  }

  Eigen::VectorXd q_start, dq_start, ddq_start;
  if (!sampleTrajectory(*msg, mapping, 0.0, q_start, dq_start, ddq_start)) {
    ROS_WARN_THROTTLE(
        1.0,
        "[TrajectoryExecutionManager] Failed to sample trajectory start.");
    return;
  }

  const ros::Time now = ros::Time::now();
  {
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (has_joint_state_ && reject_large_start_error_) {
      const double start_error =
          (q_start - q_measured_).lpNorm<Eigen::Infinity>();
      if (start_error > max_start_error_) {
        ROS_WARN_THROTTLE(
            1.0,
            "[TrajectoryExecutionManager] Reject trajectory: start error %.4f > %.4f",
            start_error, max_start_error_);
        return;
      }
    }

    active_trajectory_ = *msg;
    active_trajectory_mapping_ = mapping;
    active_trajectory_start_time_ = now;
    active_trajectory_received_time_ = now;
    active_trajectory_duration_s_ = duration;
    has_active_trajectory_ = true;
    has_received_trajectory_ = true;

    last_q_ref_ = q_start;
    last_dq_ref_ = dq_start;
    has_last_reference_ = true;
  }

  ROS_INFO_STREAM_THROTTLE(
      0.5,
      "[TrajectoryExecutionManager] accepted committed trajectory duration="
          << duration << " s points=" << msg->points.size());
}

void TrajectoryExecutionManager::executionTimerCallback(
    const ros::TimerEvent&) {
  const ros::Time now = ros::Time::now();

  Eigen::VectorXd q_measured;
  Eigen::VectorXd q_ref;
  Eigen::VectorXd dq_ref;
  Eigen::VectorXd ddq_ref;

  bool has_joint_state = false;
  bool has_received_trajectory = false;
  bool active = false;
  bool complete = false;
  bool have_reference = false;
  double phase_s = 0.0;
  double remaining_s = 0.0;
  std::string source = "none";

  {
    std::lock_guard<std::mutex> lock(data_mutex_);

    has_joint_state = has_joint_state_;
    has_received_trajectory = has_received_trajectory_;
    if (has_joint_state) q_measured = q_measured_;

    if (has_active_trajectory_) {
      phase_s = std::max(
          0.0, (now - active_trajectory_start_time_).toSec());
      const double sample_t =
          std::min(phase_s, active_trajectory_duration_s_);

      if (sampleTrajectory(
              active_trajectory_,
              active_trajectory_mapping_,
              sample_t,
              q_ref,
              dq_ref,
              ddq_ref)) {
        have_reference = true;
        source = "active_trajectory";
        remaining_s =
            std::max(0.0, active_trajectory_duration_s_ - phase_s);

        last_q_ref_ = q_ref;
        last_dq_ref_ = dq_ref;
        has_last_reference_ = true;

        if (phase_s >= active_trajectory_duration_s_ - 1e-9) {
          complete = true;
          has_active_trajectory_ = false;
          dq_ref.setZero();
          last_dq_ref_.setZero();
          source = "trajectory_complete_hold";
        } else {
          active = true;
        }
      } else {
        ROS_WARN_THROTTLE(
            1.0,
            "[TrajectoryExecutionManager] Failed to sample active trajectory.");
      }
    }

    if (!have_reference && has_last_reference_ &&
        hold_last_reference_when_no_trajectory_) {
      q_ref = last_q_ref_;
      dq_ref = Eigen::VectorXd::Zero(last_dq_ref_.size());
      have_reference = true;
      source = "last_reference_hold";
    }
  }

  if (!has_joint_state) {
    ROS_WARN_THROTTLE(
        1.0, "[TrajectoryExecutionManager] Waiting for joint state.");
    return;
  }

  if (!have_reference) {
    if (!hold_when_no_trajectory_) return;

    const Eigen::VectorXd dq_zero = makeZeroVelocityCommand();
    if (hold_initial_zero_pose_ && !has_received_trajectory) {
      q_ref = Eigen::VectorXd::Zero(
          static_cast<int>(joint_names_.size()));
      dq_ref = dq_zero;
      source = "initial_zero_hold";
      have_reference = true;
    } else {
      publishVelocityCommand(dq_zero);
      publishReferenceState(q_measured, dq_zero);
      publishSummary(false, false, 0.0, 0.0, 0.0, "zero_velocity_hold");
      return;
    }
  }

  const double tracking_error =
      (q_ref - q_measured).lpNorm<Eigen::Infinity>();

  maybePublishReplanRequest(tracking_error);

  if (tracking_error > max_tracking_error_) {
    ROS_WARN_THROTTLE(
        1.0,
        "[TrajectoryExecutionManager] Tracking error %.4f > %.4f; holding.",
        tracking_error, max_tracking_error_);

    if (hold_when_tracking_error_large_) {
      const Eigen::VectorXd dq_zero = makeZeroVelocityCommand();
      publishVelocityCommand(dq_zero);
      publishReferenceState(q_measured, dq_zero);
      publishSummary(active, complete, phase_s, remaining_s,
                     tracking_error, "tracking_error_hold");
      return;
    }
  }

  Eigen::VectorXd dq_cmd =
      computeVelocityCommand(q_ref, dq_ref, q_measured);
  dq_cmd = clampVelocityCommand(dq_cmd);

  publishVelocityCommand(dq_cmd);
  publishReferenceState(q_ref, dq_ref);
  publishSummary(active, complete, phase_s, remaining_s,
                 tracking_error, source);
}

bool TrajectoryExecutionManager::extractMeasuredState(
    const sensor_msgs::JointState& msg,
    Eigen::VectorXd& q,
    Eigen::VectorXd& dq) const {
  if (joint_names_.empty()) return false;

  std::unordered_map<std::string, std::size_t> name_to_index;
  for (std::size_t i = 0; i < msg.name.size(); ++i) {
    name_to_index[msg.name[i]] = i;
  }

  const int dof = static_cast<int>(joint_names_.size());
  q = Eigen::VectorXd::Zero(dof);
  dq = Eigen::VectorXd::Zero(dof);

  for (int i = 0; i < dof; ++i) {
    const auto it = name_to_index.find(joint_names_[i]);
    if (it == name_to_index.end()) return false;

    const std::size_t src_idx = it->second;
    if (src_idx >= msg.position.size()) return false;

    q[i] = msg.position[src_idx];
    dq[i] = src_idx < msg.velocity.size() ? msg.velocity[src_idx] : 0.0;
  }
  return true;
}

bool TrajectoryExecutionManager::trajectoryHasExpectedJoints(
    const trajectory_msgs::JointTrajectory& traj) const {
  if (traj.joint_names.empty()) return false;
  for (const auto& joint_name : joint_names_) {
    if (std::find(traj.joint_names.begin(),
                  traj.joint_names.end(),
                  joint_name) == traj.joint_names.end()) {
      return false;
    }
  }
  return true;
}

bool TrajectoryExecutionManager::buildTrajectoryJointIndexMap(
    const trajectory_msgs::JointTrajectory& traj,
    std::vector<int>& mapping) const {
  mapping.clear();
  mapping.reserve(joint_names_.size());
  for (const auto& joint_name : joint_names_) {
    const auto it = std::find(
        traj.joint_names.begin(), traj.joint_names.end(), joint_name);
    if (it == traj.joint_names.end()) return false;
    mapping.push_back(static_cast<int>(it - traj.joint_names.begin()));
  }
  return true;
}

bool TrajectoryExecutionManager::getPointVector(
    const trajectory_msgs::JointTrajectoryPoint&,
    const std::vector<double>& field,
    const std::vector<int>& mapping,
    Eigen::VectorXd& out,
    bool allow_missing_as_zero) const {
  const int dof = static_cast<int>(mapping.size());
  out = Eigen::VectorXd::Zero(dof);

  if (field.empty()) return allow_missing_as_zero;

  for (int i = 0; i < dof; ++i) {
    const int src_idx = mapping[static_cast<std::size_t>(i)];
    if (src_idx < 0 ||
        static_cast<std::size_t>(src_idx) >= field.size()) {
      if (!allow_missing_as_zero) return false;
      continue;
    }
    out[i] = field[static_cast<std::size_t>(src_idx)];
  }
  return true;
}

bool TrajectoryExecutionManager::sampleTrajectory(
    const trajectory_msgs::JointTrajectory& traj,
    const std::vector<int>& mapping,
    double t,
    Eigen::VectorXd& q_ref,
    Eigen::VectorXd& dq_ref,
    Eigen::VectorXd& ddq_ref) const {
  if (traj.points.empty() || mapping.empty()) return false;

  auto copyPoint =
      [&](const trajectory_msgs::JointTrajectoryPoint& p) -> bool {
        if (!getPointVector(p, p.positions, mapping, q_ref, false)) return false;
        getPointVector(p, p.velocities, mapping, dq_ref, true);
        getPointVector(p, p.accelerations, mapping, ddq_ref, true);
        return true;
      };

  if (traj.points.size() == 1) return copyPoint(traj.points.front());

  const double first_t = traj.points.front().time_from_start.toSec();
  const double last_t = traj.points.back().time_from_start.toSec();

  if (t <= first_t) return copyPoint(traj.points.front());
  if (t >= last_t) return copyPoint(traj.points.back());

  std::size_t hi = 1;
  while (hi < traj.points.size() &&
         traj.points[hi].time_from_start.toSec() < t) {
    ++hi;
  }
  if (hi >= traj.points.size()) return copyPoint(traj.points.back());

  const auto& p0 = traj.points[hi - 1];
  const auto& p1 = traj.points[hi];
  const double t0 = p0.time_from_start.toSec();
  const double t1 = p1.time_from_start.toSec();
  const double h = t1 - t0;
  if (h <= 1e-12) return false;

  const double s = (t - t0) / h;

  Eigen::VectorXd q0, q1, v0, v1, a0, a1;
  if (!getPointVector(p0, p0.positions, mapping, q0, false) ||
      !getPointVector(p1, p1.positions, mapping, q1, false)) {
    return false;
  }
  getPointVector(p0, p0.velocities, mapping, v0, true);
  getPointVector(p1, p1.velocities, mapping, v1, true);
  getPointVector(p0, p0.accelerations, mapping, a0, true);
  getPointVector(p1, p1.accelerations, mapping, a1, true);

  q_ref = (1.0 - s) * q0 + s * q1;
  dq_ref = (1.0 - s) * v0 + s * v1;
  ddq_ref = (1.0 - s) * a0 + s * a1;
  return true;
}

double TrajectoryExecutionManager::getTrajectoryEndTime(
    const trajectory_msgs::JointTrajectory& traj) const {
  return traj.points.empty()
             ? 0.0
             : traj.points.back().time_from_start.toSec();
}

Eigen::VectorXd TrajectoryExecutionManager::computeVelocityCommand(
    const Eigen::VectorXd& q_ref,
    const Eigen::VectorXd& dq_ref,
    const Eigen::VectorXd& q_measured) const {
  Eigen::VectorXd dq_cmd = Eigen::VectorXd::Zero(q_ref.size());
  if (dq_ref.size() == q_ref.size()) dq_cmd = dq_ref;
  dq_cmd += position_feedback_gain_ * (q_ref - q_measured);
  return dq_cmd;
}

Eigen::VectorXd TrajectoryExecutionManager::clampVelocityCommand(
    const Eigen::VectorXd& dq_cmd) const {
  Eigen::VectorXd out = dq_cmd;
  for (int i = 0; i < out.size(); ++i) {
    out[i] = std::max(
        -max_command_velocity_,
        std::min(max_command_velocity_, out[i]));
  }
  return out;
}

void TrajectoryExecutionManager::publishVelocityCommand(
    const Eigen::VectorXd& dq_cmd) {
  std_msgs::Float64MultiArray msg;
  msg.data.resize(static_cast<std::size_t>(dq_cmd.size()));
  for (int i = 0; i < dq_cmd.size(); ++i) {
    msg.data[static_cast<std::size_t>(i)] = dq_cmd[i];
  }
  velocity_command_pub_.publish(msg);
}

void TrajectoryExecutionManager::publishReferenceState(
    const Eigen::VectorXd& q_ref,
    const Eigen::VectorXd& dq_ref) {
  sensor_msgs::JointState msg;
  msg.header.stamp = ros::Time::now();
  msg.name = joint_names_;
  msg.position.resize(static_cast<std::size_t>(q_ref.size()));
  msg.velocity.resize(static_cast<std::size_t>(dq_ref.size()));
  for (int i = 0; i < q_ref.size(); ++i)
    msg.position[static_cast<std::size_t>(i)] = q_ref[i];
  for (int i = 0; i < dq_ref.size(); ++i)
    msg.velocity[static_cast<std::size_t>(i)] = dq_ref[i];
  reference_state_pub_.publish(msg);
}

void TrajectoryExecutionManager::publishSummary(
    bool trajectory_active,
    bool trajectory_complete,
    double phase_s,
    double remaining_s,
    double tracking_error_inf,
    const std::string& source) {
  std_msgs::String msg;
  std::ostringstream oss;
  oss << "TRACKER"
      << " active=" << static_cast<int>(trajectory_active)
      << " complete=" << static_cast<int>(trajectory_complete)
      << " phase_s=" << phase_s
      << " remaining_s=" << remaining_s
      << " tracking_error_inf=" << tracking_error_inf
      << " source=" << source;
  msg.data = oss.str();
  summary_pub_.publish(msg);
}

void TrajectoryExecutionManager::maybePublishReplanRequest(
    double tracking_error_inf) {
  if (replan_tracking_error_inf_ <= 0.0 ||
      tracking_error_inf <= replan_tracking_error_inf_) {
    return;
  }

  const ros::Time now = ros::Time::now();
  if (!last_replan_request_time_.isZero() &&
      (now - last_replan_request_time_).toSec() <
          replan_request_min_interval_s_) {
    return;
  }

  std_msgs::Bool msg;
  msg.data = true;
  replan_request_pub_.publish(msg);
  last_replan_request_time_ = now;

  ROS_WARN_STREAM_THROTTLE(
      0.5,
      "[TrajectoryExecutionManager] requested local replan: tracking_error_inf="
          << tracking_error_inf);
}

Eigen::VectorXd TrajectoryExecutionManager::makeZeroVelocityCommand() const {
  return Eigen::VectorXd::Zero(static_cast<int>(joint_names_.size()));
}

}  // namespace egocentric_arm_planner
