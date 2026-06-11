#include "egocentric_arm_planner/trajectory_execution_manager.hpp"

#include <algorithm>
#include <cmath>

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
  q_ref_step_ = Eigen::VectorXd::Zero(dof);
  dq_ref_step_ = Eigen::VectorXd::Zero(dof);
  ddq_ref_step_ = Eigen::VectorXd::Zero(dof);

  joint_state_sub_ = nh_.subscribe(
      joint_state_topic_,
      1,
      &TrajectoryExecutionManager::jointStateCallback,
      this);

  trajectory_sub_ = nh_.subscribe(
      input_trajectory_topic_,
      1,
      &TrajectoryExecutionManager::trajectoryCallback,
      this);

  velocity_command_pub_ = nh_.advertise<std_msgs::Float64MultiArray>(
      output_velocity_command_topic_,
      1);

  reference_state_pub_ = nh_.advertise<sensor_msgs::JointState>(
      reference_state_topic_,
      1);

  execution_timer_ = nh_.createTimer(
      ros::Duration(1.0 / execution_rate_),
      &TrajectoryExecutionManager::executionTimerCallback,
      this);

  ROS_INFO("[TrajectoryExecutionManager] Initialized with short-step velocity backend.");
  ROS_INFO_STREAM("[TrajectoryExecutionManager] execution_rate = "
                  << execution_rate_);
  ROS_INFO_STREAM("[TrajectoryExecutionManager] control_dt = "
                  << control_dt_);
  ROS_INFO_STREAM("[TrajectoryExecutionManager] joint_state_topic = "
                  << joint_state_topic_);
  ROS_INFO_STREAM("[TrajectoryExecutionManager] input_trajectory_topic = "
                  << input_trajectory_topic_);
  ROS_INFO_STREAM("[TrajectoryExecutionManager] output_velocity_command_topic = "
                  << output_velocity_command_topic_);
  ROS_INFO_STREAM("[TrajectoryExecutionManager] reference_state_topic = "
                  << reference_state_topic_);
  ROS_INFO_STREAM("[TrajectoryExecutionManager] position_feedback_gain = "
                  << position_feedback_gain_);
  ROS_INFO_STREAM("[TrajectoryExecutionManager] max_command_velocity = "
                  << max_command_velocity_);
  ROS_INFO_STREAM("[TrajectoryExecutionManager] hold_initial_zero_pose = "
                  << hold_initial_zero_pose_);
  ROS_INFO_STREAM("[TrajectoryExecutionManager] hold_last_reference_when_no_trajectory = "
                  << hold_last_reference_when_no_trajectory_);
  ROS_INFO_STREAM("[TrajectoryExecutionManager] reference_timeout = "
                  << reference_timeout_);

  return true;
}

bool TrajectoryExecutionManager::loadConfig() {
  pnh_.param<double>("execution/rate", execution_rate_, execution_rate_);

  pnh_.param<double>("execution/control_dt", control_dt_, control_dt_);

  pnh_.param<double>("execution/max_start_error",
                    max_start_error_,
                    max_start_error_);

  pnh_.param<double>("execution/max_tracking_error",
                    max_tracking_error_,
                    max_tracking_error_);

  pnh_.param<bool>("execution/hold_when_no_trajectory",
                  hold_when_no_trajectory_,
                  hold_when_no_trajectory_);

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
                    reference_timeout_,
                    reference_timeout_);


  // Backward-compatible fallback: old configs used execution/velocity_tracking_kp.
  pnh_.param<double>("execution/velocity_tracking_kp",
                    position_feedback_gain_,
                    position_feedback_gain_);

  pnh_.param<double>("execution/position_feedback_gain",
                    position_feedback_gain_,
                    position_feedback_gain_);

  pnh_.param<double>("execution/max_command_velocity",
                    max_command_velocity_,
                    max_command_velocity_);

  pnh_.param<std::string>("execution/joint_states",
                         joint_state_topic_,
                         joint_state_topic_);

  pnh_.param<std::string>("execution/input_trajectory",
                         input_trajectory_topic_,
                         input_trajectory_topic_);

  pnh_.param<std::string>("execution/output_velocity_command",
                         output_velocity_command_topic_,
                         output_velocity_command_topic_);

  pnh_.param<std::string>("execution/reference_state",
                         reference_state_topic_,
                         reference_state_topic_);

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

  if (reference_timeout_ < 0.0) {
    ROS_ERROR("[TrajectoryExecutionManager] execution/reference_timeout must be non-negative.");
    return false;
  }

  return true;
}

void TrajectoryExecutionManager::jointStateCallback(
    const sensor_msgs::JointStateConstPtr& msg) {
  if (!msg) {
    return;
  }

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
  if (!msg) {
    return;
  }

  if (msg->points.empty()) {
    ROS_WARN_THROTTLE(
        1.0,
        "[TrajectoryExecutionManager] Received empty trajectory.");
    return;
  }

  if (!trajectoryHasExpectedJoints(*msg)) {
    ROS_WARN_THROTTLE(
        1.0,
        "[TrajectoryExecutionManager] Received trajectory with unexpected joints.");
    return;
  }

  std::vector<int> traj_index_for_control_joint;
  if (!buildTrajectoryJointIndexMap(*msg, traj_index_for_control_joint)) {
    ROS_WARN_THROTTLE(
        1.0,
        "[TrajectoryExecutionManager] Failed to build trajectory joint index map.");
    return;
  }

  const double traj_end_time = getTrajectoryEndTime(*msg);
  const double sample_time = std::min(control_dt_, traj_end_time);

  Eigen::VectorXd q_step;
  Eigen::VectorXd dq_step;
  Eigen::VectorXd ddq_step;

  if (!sampleTrajectory(*msg,
                        traj_index_for_control_joint,
                        sample_time,
                        q_step,
                        dq_step,
                        ddq_step)) {
    ROS_WARN_THROTTLE(
        1.0,
        "[TrajectoryExecutionManager] Failed to sample short-step reference at t = %.4f.",
        sample_time);
    return;
  }

  std::lock_guard<std::mutex> lock(data_mutex_);

  const ros::Time now = ros::Time::now();

  if (has_joint_state_ && reject_large_start_error_) {
    Eigen::VectorXd q_start;
    Eigen::VectorXd dq_start;
    Eigen::VectorXd ddq_start;

    if (sampleTrajectory(*msg,
                         traj_index_for_control_joint,
                         0.0,
                         q_start,
                         dq_start,
                         ddq_start)) {
      const double start_error =
          (q_start - q_measured_).lpNorm<Eigen::Infinity>();

      if (start_error > max_start_error_) {
        ROS_WARN_THROTTLE(
            1.0,
            "[TrajectoryExecutionManager] Rejecting trajectory. "
            "Start error too large: %.4f > %.4f",
            start_error,
            max_start_error_);
        return;
      }
    }
  }

  q_ref_step_ = q_step;
  dq_ref_step_ = dq_step;
  ddq_ref_step_ = ddq_step;
  last_reference_time_ = now;
  has_reference_step_ = true;
  has_received_trajectory_ = true;
}

void TrajectoryExecutionManager::executionTimerCallback(
    const ros::TimerEvent& event) {
  (void)event;

  Eigen::VectorXd q_measured;
  Eigen::VectorXd q_ref;
  Eigen::VectorXd dq_ref;
  ros::Time last_reference_time;

  bool has_joint_state = false;
  bool has_reference_step = false;
  bool has_received_trajectory = false;

  {
    std::lock_guard<std::mutex> lock(data_mutex_);

    has_joint_state = has_joint_state_;
    if (has_joint_state) {
      q_measured = q_measured_;
    }

    has_reference_step = has_reference_step_;
    has_received_trajectory = has_received_trajectory_;
    if (has_reference_step) {
      q_ref = q_ref_step_;
      dq_ref = dq_ref_step_;
      last_reference_time = last_reference_time_;
    }
  }

  if (!has_joint_state) {
    ROS_WARN_THROTTLE(
        1.0,
        "[TrajectoryExecutionManager] Waiting for joint state.");
    return;
  }

  if (!has_reference_step) {
    if (!hold_when_no_trajectory_) {
      return;
    }

    const Eigen::VectorXd dq_zero = makeZeroVelocityCommand();

    if (hold_initial_zero_pose_) {
      // Before the first planner trajectory, actively hold the zero joint pose.
      // This is not used after motion; once a trajectory is received, the last
      // sampled reference is held instead.
      q_ref = Eigen::VectorXd::Zero(static_cast<int>(joint_names_.size()));
      dq_ref = dq_zero;
    } else {
      // Backward-compatible behavior: command zero velocity and publish the
      // measured state as the debug reference.
      publishVelocityCommand(dq_zero);
      publishReferenceState(q_measured, dq_zero);
      return;
    }
  }

  if (has_reference_step && has_received_trajectory &&
      reference_timeout_ > 0.0 && !last_reference_time.isZero()) {
    const double reference_age =
        (ros::Time::now() - last_reference_time).toSec();

    if (reference_age > reference_timeout_) {
      if (hold_last_reference_when_no_trajectory_) {
        // If the planner stops publishing, keep the last valid position
        // reference but remove feedforward velocity.  This prevents returning
        // to home after reaching a target while still resisting drift.
        dq_ref = Eigen::VectorXd::Zero(static_cast<int>(joint_names_.size()));
      } else if (hold_when_no_trajectory_) {
        const Eigen::VectorXd dq_zero = makeZeroVelocityCommand();
        publishVelocityCommand(dq_zero);
        publishReferenceState(q_measured, dq_zero);
        return;
      } else {
        return;
      }
    }
  }

  const double tracking_error =
      (q_ref - q_measured).lpNorm<Eigen::Infinity>();

  if (tracking_error > max_tracking_error_) {
    ROS_WARN_THROTTLE(
        1.0,
        "[TrajectoryExecutionManager] Tracking error too large: %.4f > %.4f. "
        "Publishing zero velocity for safety.",
        tracking_error,
        max_tracking_error_);

    if (hold_when_tracking_error_large_) {
      const Eigen::VectorXd dq_zero = makeZeroVelocityCommand();
      publishVelocityCommand(dq_zero);
      publishReferenceState(q_measured, dq_zero);
      return;
    }
  }

  Eigen::VectorXd dq_cmd =
      computeVelocityCommand(q_ref, dq_ref, q_measured);

  dq_cmd = clampVelocityCommand(dq_cmd);

  publishVelocityCommand(dq_cmd);
  publishReferenceState(q_ref, dq_ref);
}

bool TrajectoryExecutionManager::extractMeasuredState(
    const sensor_msgs::JointState& msg,
    Eigen::VectorXd& q,
    Eigen::VectorXd& dq) const {
  if (joint_names_.empty()) {
    return false;
  }

  std::unordered_map<std::string, std::size_t> name_to_index;
  for (std::size_t i = 0; i < msg.name.size(); ++i) {
    name_to_index[msg.name[i]] = i;
  }

  const int dof = static_cast<int>(joint_names_.size());
  q = Eigen::VectorXd::Zero(dof);
  dq = Eigen::VectorXd::Zero(dof);

  for (int i = 0; i < dof; ++i) {
    const auto it = name_to_index.find(joint_names_[i]);
    if (it == name_to_index.end()) {
      ROS_WARN_THROTTLE(
          1.0,
          "[TrajectoryExecutionManager] Missing joint in JointState: %s",
          joint_names_[i].c_str());
      return false;
    }

    const std::size_t src_idx = it->second;

    if (src_idx < msg.position.size()) {
      q[i] = msg.position[src_idx];
    } else {
      ROS_WARN_THROTTLE(
          1.0,
          "[TrajectoryExecutionManager] JointState has no position for: %s",
          joint_names_[i].c_str());
      return false;
    }

    if (src_idx < msg.velocity.size()) {
      dq[i] = msg.velocity[src_idx];
    } else {
      dq[i] = 0.0;
    }
  }

  return true;
}

bool TrajectoryExecutionManager::trajectoryHasExpectedJoints(
    const trajectory_msgs::JointTrajectory& traj) const {
  if (traj.joint_names.empty()) {
    return false;
  }

  for (const auto& joint_name : joint_names_) {
    const auto it = std::find(
        traj.joint_names.begin(),
        traj.joint_names.end(),
        joint_name);

    if (it == traj.joint_names.end()) {
      ROS_WARN_THROTTLE(
          1.0,
          "[TrajectoryExecutionManager] Trajectory missing joint: %s",
          joint_name.c_str());
      return false;
    }
  }

  return true;
}

bool TrajectoryExecutionManager::buildTrajectoryJointIndexMap(
    const trajectory_msgs::JointTrajectory& traj,
    std::vector<int>& traj_index_for_control_joint) const {
  traj_index_for_control_joint.clear();
  traj_index_for_control_joint.reserve(joint_names_.size());

  for (const auto& joint_name : joint_names_) {
    const auto it = std::find(
        traj.joint_names.begin(),
        traj.joint_names.end(),
        joint_name);

    if (it == traj.joint_names.end()) {
      return false;
    }

    traj_index_for_control_joint.push_back(
        static_cast<int>(it - traj.joint_names.begin()));
  }

  return true;
}

bool TrajectoryExecutionManager::getPointVector(
    const trajectory_msgs::JointTrajectoryPoint& point,
    const std::vector<double>& field,
    const std::vector<int>& traj_index_for_control_joint,
    Eigen::VectorXd& out,
    bool allow_missing_as_zero) const {
  const int dof = static_cast<int>(traj_index_for_control_joint.size());
  out = Eigen::VectorXd::Zero(dof);

  if (field.empty()) {
    return allow_missing_as_zero;
  }

  for (int i = 0; i < dof; ++i) {
    const int src_idx = traj_index_for_control_joint[i];

    if (src_idx < 0 || static_cast<std::size_t>(src_idx) >= field.size()) {
      if (allow_missing_as_zero) {
        out[i] = 0.0;
      } else {
        return false;
      }
    } else {
      out[i] = field[static_cast<std::size_t>(src_idx)];
    }
  }

  return true;
}

bool TrajectoryExecutionManager::sampleTrajectory(
    const trajectory_msgs::JointTrajectory& traj,
    const std::vector<int>& traj_index_for_control_joint,
    double t,
    Eigen::VectorXd& q_ref,
    Eigen::VectorXd& dq_ref,
    Eigen::VectorXd& ddq_ref) const {
  if (traj.points.empty()) {
    return false;
  }

  if (traj_index_for_control_joint.empty()) {
    return false;
  }

  if (traj.points.size() == 1) {
    const auto& p = traj.points.front();

    if (!getPointVector(p,
                        p.positions,
                        traj_index_for_control_joint,
                        q_ref,
                        false)) {
      return false;
    }

    getPointVector(p,
                   p.velocities,
                   traj_index_for_control_joint,
                   dq_ref,
                   true);

    getPointVector(p,
                   p.accelerations,
                   traj_index_for_control_joint,
                   ddq_ref,
                   true);

    return true;
  }

  const double first_t = traj.points.front().time_from_start.toSec();
  const double last_t = traj.points.back().time_from_start.toSec();

  if (t <= first_t) {
    const auto& p = traj.points.front();

    if (!getPointVector(p,
                        p.positions,
                        traj_index_for_control_joint,
                        q_ref,
                        false)) {
      return false;
    }

    getPointVector(p,
                   p.velocities,
                   traj_index_for_control_joint,
                   dq_ref,
                   true);

    getPointVector(p,
                   p.accelerations,
                   traj_index_for_control_joint,
                   ddq_ref,
                   true);

    return true;
  }

  if (t >= last_t) {
    const auto& p = traj.points.back();

    if (!getPointVector(p,
                        p.positions,
                        traj_index_for_control_joint,
                        q_ref,
                        false)) {
      return false;
    }

    getPointVector(p,
                   p.velocities,
                   traj_index_for_control_joint,
                   dq_ref,
                   true);

    getPointVector(p,
                   p.accelerations,
                   traj_index_for_control_joint,
                   ddq_ref,
                   true);

    return true;
  }

  std::size_t idx1 = 1;
  while (idx1 < traj.points.size() &&
         traj.points[idx1].time_from_start.toSec() < t) {
    ++idx1;
  }

  if (idx1 >= traj.points.size()) {
    return false;
  }

  const std::size_t idx0 = idx1 - 1;

  const auto& p0 = traj.points[idx0];
  const auto& p1 = traj.points[idx1];

  const double t0 = p0.time_from_start.toSec();
  const double t1 = p1.time_from_start.toSec();

  const double h = t1 - t0;
  if (h <= 1e-12) {
    return false;
  }

  const double s = (t - t0) / h;

  Eigen::VectorXd q0;
  Eigen::VectorXd q1;
  Eigen::VectorXd dq0;
  Eigen::VectorXd dq1;
  Eigen::VectorXd ddq0;
  Eigen::VectorXd ddq1;

  if (!getPointVector(p0,
                      p0.positions,
                      traj_index_for_control_joint,
                      q0,
                      false)) {
    return false;
  }

  if (!getPointVector(p1,
                      p1.positions,
                      traj_index_for_control_joint,
                      q1,
                      false)) {
    return false;
  }

  getPointVector(p0,
                 p0.velocities,
                 traj_index_for_control_joint,
                 dq0,
                 true);

  getPointVector(p1,
                 p1.velocities,
                 traj_index_for_control_joint,
                 dq1,
                 true);

  getPointVector(p0,
                 p0.accelerations,
                 traj_index_for_control_joint,
                 ddq0,
                 true);

  getPointVector(p1,
                 p1.accelerations,
                 traj_index_for_control_joint,
                 ddq1,
                 true);

  q_ref = (1.0 - s) * q0 + s * q1;
  dq_ref = (1.0 - s) * dq0 + s * dq1;
  ddq_ref = (1.0 - s) * ddq0 + s * ddq1;

  return true;
}

double TrajectoryExecutionManager::getTrajectoryEndTime(
    const trajectory_msgs::JointTrajectory& traj) const {
  if (traj.points.empty()) {
    return 0.0;
  }

  return traj.points.back().time_from_start.toSec();
}

Eigen::VectorXd TrajectoryExecutionManager::computeVelocityCommand(
    const Eigen::VectorXd& q_ref,
    const Eigen::VectorXd& dq_ref,
    const Eigen::VectorXd& q_measured) const {
  Eigen::VectorXd dq_cmd = Eigen::VectorXd::Zero(q_ref.size());

  if (dq_ref.size() == q_ref.size()) {
    dq_cmd = dq_ref;
  }

  dq_cmd += position_feedback_gain_ * (q_ref - q_measured);

  return dq_cmd;
}

Eigen::VectorXd TrajectoryExecutionManager::clampVelocityCommand(
    const Eigen::VectorXd& dq_cmd) const {
  Eigen::VectorXd dq_limited = dq_cmd;

  for (int i = 0; i < dq_limited.size(); ++i) {
    if (dq_limited[i] > max_command_velocity_) {
      dq_limited[i] = max_command_velocity_;
    } else if (dq_limited[i] < -max_command_velocity_) {
      dq_limited[i] = -max_command_velocity_;
    }
  }

  return dq_limited;
}

Eigen::VectorXd TrajectoryExecutionManager::makeZeroVelocityCommand() const {
  return Eigen::VectorXd::Zero(static_cast<int>(joint_names_.size()));
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

  for (int i = 0; i < q_ref.size(); ++i) {
    msg.position[static_cast<std::size_t>(i)] = q_ref[i];
  }

  for (int i = 0; i < dq_ref.size(); ++i) {
    msg.velocity[static_cast<std::size_t>(i)] = dq_ref[i];
  }

  reference_state_pub_.publish(msg);
}

}  // namespace egocentric_arm_planner
