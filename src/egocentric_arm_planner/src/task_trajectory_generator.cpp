#include "egocentric_arm_planner/task_trajectory_generator.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <XmlRpcValue.h>

namespace egocentric_arm_planner {

namespace {

constexpr double kEps = 1e-9;

std::string jointScalarVectorToString(const std::vector<std::string>& names,
                                      const std::vector<double>& v,
                                      int precision = 6) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << "[";
  for (int i = 0; i < static_cast<int>(v.size()); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    if (i < static_cast<int>(names.size())) {
      oss << names[static_cast<std::size_t>(i)] << ":";
    } else {
      oss << "j" << i << ":";
    }
    oss << v[static_cast<std::size_t>(i)];
  }
  oss << "]";
  return oss.str();
}

bool readXmlRpcNumber(const XmlRpc::XmlRpcValue& value, double& out) {
  if (value.getType() == XmlRpc::XmlRpcValue::TypeInt) {
    out = static_cast<int>(value);
    return true;
  }
  if (value.getType() == XmlRpc::XmlRpcValue::TypeDouble) {
    out = static_cast<double>(value);
    return true;
  }
  return false;
}

bool loadJointLimitVector(const ros::NodeHandle& nh,
                          const std::string& param_name,
                          const std::vector<std::string>& joint_names,
                          const double fallback_value,
                          std::vector<double>& limits) {
  const int dof = static_cast<int>(joint_names.size());
  limits.assign(static_cast<std::size_t>(dof), fallback_value);

  XmlRpc::XmlRpcValue param;
  if (!nh.getParam(param_name, param)) {
    return true;
  }

  if (param.getType() == XmlRpc::XmlRpcValue::TypeStruct) {
    for (int i = 0; i < dof; ++i) {
      const std::string& joint_name = joint_names[static_cast<std::size_t>(i)];
      if (!param.hasMember(joint_name)) {
        ROS_WARN_STREAM("[TaskTrajectoryGenerator] Missing " << param_name
                        << " for " << joint_name
                        << ". Falling back to nominal value = " << fallback_value);
        continue;
      }

      double value = fallback_value;
      if (!readXmlRpcNumber(param[joint_name], value)) {
        ROS_ERROR_STREAM("[TaskTrajectoryGenerator] " << param_name << " for "
                         << joint_name << " must be int or double.");
        return false;
      }
      limits[static_cast<std::size_t>(i)] = value;
    }
    return true;
  }

  if (param.getType() == XmlRpc::XmlRpcValue::TypeArray) {
    if (param.size() != dof) {
      ROS_ERROR_STREAM("[TaskTrajectoryGenerator] " << param_name
                       << " array size = " << param.size()
                       << ", but dof = " << dof);
      return false;
    }

    for (int i = 0; i < dof; ++i) {
      double value = fallback_value;
      if (!readXmlRpcNumber(param[i], value)) {
        ROS_ERROR_STREAM("[TaskTrajectoryGenerator] " << param_name << "["
                         << i << "] must be int or double.");
        return false;
      }
      limits[static_cast<std::size_t>(i)] = value;
    }
    return true;
  }

  ROS_ERROR_STREAM("[TaskTrajectoryGenerator] " << param_name
                   << " must be a map or an array.");
  return false;
}

}  // namespace

bool TaskTrajectoryGenerator::initialize(
    const ros::NodeHandle& nh,
    const std::shared_ptr<arm_model::RobotModel>& robot_model) {
  robot_model_ = robot_model;

  if (!robot_model_) {
    ROS_ERROR("[TaskTrajectoryGenerator] robot_model is null.");
    last_status_ = PlannerStatus::MISSING_ROBOT_MODEL;
    initialized_ = false;
    return false;
  }

  if (!robot_model_->initialized()) {
    ROS_ERROR("[TaskTrajectoryGenerator] robot_model is not initialized.");
    last_status_ = PlannerStatus::MISSING_ROBOT_MODEL;
    initialized_ = false;
    return false;
  }

  if (!loadConfig(nh)) {
    ROS_ERROR("[TaskTrajectoryGenerator] Failed to load config.");
    last_status_ = PlannerStatus::UNKNOWN_ERROR;
    initialized_ = false;
    return false;
  }

  initialized_ = true;
  last_status_ = PlannerStatus::SUCCESS;

  ROS_INFO("[TaskTrajectoryGenerator] Initialized.");
  ROS_INFO_STREAM("[TaskTrajectoryGenerator] T_plan = " << config_.T_plan);
  ROS_INFO_STREAM("[TaskTrajectoryGenerator] trajectory_dt = " << config_.trajectory_dt);
  ROS_INFO("[TaskTrajectoryGenerator] interpolation = quintic measured-start-velocity/acceleration, zero-end-velocity/acceleration");
  ROS_INFO_STREAM("[TaskTrajectoryGenerator] enable_time_scaling = "
                  << static_cast<int>(config_.enable_time_scaling));
  ROS_INFO_STREAM("[TaskTrajectoryGenerator] min_plan_duration = "
                  << config_.min_plan_duration);
  ROS_INFO_STREAM("[TaskTrajectoryGenerator] max_plan_duration = "
                  << config_.max_plan_duration);
  ROS_INFO_STREAM("[TaskTrajectoryGenerator] enforce_velocity_acceleration_limits = "
                  << static_cast<int>(config_.enforce_velocity_acceleration_limits));
  ROS_INFO_STREAM("[TaskTrajectoryGenerator] joint_velocity_limits = "
                  << jointScalarVectorToString(robot_model_->jointNames(), config_.joint_velocity_limits));
  ROS_INFO_STREAM("[TaskTrajectoryGenerator] joint_acceleration_limits = "
                  << jointScalarVectorToString(robot_model_->jointNames(), config_.joint_acceleration_limits));

  return true;
}

bool TaskTrajectoryGenerator::initialized() const {
  return initialized_;
}

bool TaskTrajectoryGenerator::loadConfig(const ros::NodeHandle& nh) {
  nh.param<double>("task_generator/T_plan", config_.T_plan, config_.T_plan);
  nh.param<double>("task_generator/trajectory_dt", config_.trajectory_dt, config_.trajectory_dt);
  nh.param<bool>("task_generator/enable_time_scaling", config_.enable_time_scaling, config_.enable_time_scaling);
  nh.param<double>("task_generator/nominal_max_joint_velocity", config_.nominal_max_joint_velocity, config_.nominal_max_joint_velocity);
  nh.param<double>("task_generator/nominal_max_joint_acceleration", config_.nominal_max_joint_acceleration, config_.nominal_max_joint_acceleration);
  nh.param<double>("task_generator/min_plan_duration", config_.min_plan_duration, config_.min_plan_duration);
  nh.param<double>("task_generator/max_plan_duration", config_.max_plan_duration, config_.max_plan_duration);
  nh.param<bool>("task_generator/enforce_velocity_acceleration_limits", config_.enforce_velocity_acceleration_limits, config_.enforce_velocity_acceleration_limits);
  nh.param<int>("task_generator/duration_limit_check_iterations", config_.duration_limit_check_iterations, config_.duration_limit_check_iterations);
  nh.param<double>("task_generator/duration_limit_check_margin", config_.duration_limit_check_margin, config_.duration_limit_check_margin);
  nh.param<bool>("task_generator/reject_large_joint_jump", config_.reject_large_joint_jump, config_.reject_large_joint_jump);
  nh.param<double>("task_generator/max_joint_jump_inf_norm", config_.max_joint_jump_inf_norm, config_.max_joint_jump_inf_norm);

  const int dof = robot_model_ ? robot_model_->nq() : 0;
  const auto& joint_names = robot_model_->jointNames();

  if (!loadJointLimitVector(nh,
                            "task_generator/joint_velocity_limits",
                            joint_names,
                            config_.nominal_max_joint_velocity,
                            config_.joint_velocity_limits)) {
    return false;
  }

  if (!loadJointLimitVector(nh,
                            "task_generator/joint_acceleration_limits",
                            joint_names,
                            config_.nominal_max_joint_acceleration,
                            config_.joint_acceleration_limits)) {
    return false;
  }

  if (config_.T_plan <= 0.0) {
    ROS_ERROR("[TaskTrajectoryGenerator] T_plan must be positive.");
    return false;
  }

  if (config_.trajectory_dt <= 0.0) {
    ROS_ERROR("[TaskTrajectoryGenerator] trajectory_dt must be positive.");
    return false;
  }

  if (config_.nominal_max_joint_velocity <= 1e-6) {
    ROS_ERROR("[TaskTrajectoryGenerator] nominal_max_joint_velocity is too small.");
    return false;
  }

  if (config_.nominal_max_joint_acceleration <= 1e-6) {
    ROS_ERROR("[TaskTrajectoryGenerator] nominal_max_joint_acceleration is too small.");
    return false;
  }

  if (static_cast<int>(config_.joint_velocity_limits.size()) != dof ||
      static_cast<int>(config_.joint_acceleration_limits.size()) != dof) {
    ROS_ERROR_STREAM("[TaskTrajectoryGenerator] limit vector size mismatch. dof = " << dof
                     << ", velocity_limits.size = " << config_.joint_velocity_limits.size()
                     << ", acceleration_limits.size = " << config_.joint_acceleration_limits.size());
    return false;
  }

  for (int i = 0; i < dof; ++i) {
    const double v = config_.joint_velocity_limits[static_cast<std::size_t>(i)];
    const double a = config_.joint_acceleration_limits[static_cast<std::size_t>(i)];
    if (v <= 1e-6) {
      ROS_ERROR_STREAM("[TaskTrajectoryGenerator] joint velocity limit for "
                       << joint_names[static_cast<std::size_t>(i)]
                       << " is too small: " << v);
      return false;
    }
    if (a <= 1e-6) {
      ROS_ERROR_STREAM("[TaskTrajectoryGenerator] joint acceleration limit for "
                       << joint_names[static_cast<std::size_t>(i)]
                       << " is too small: " << a);
      return false;
    }
  }

  if (config_.min_plan_duration <= 0.0 ||
      config_.max_plan_duration < config_.min_plan_duration) {
    ROS_ERROR("[TaskTrajectoryGenerator] Invalid min/max plan duration.");
    return false;
  }

  if (config_.duration_limit_check_iterations < 0) {
    config_.duration_limit_check_iterations = 0;
  }
  config_.duration_limit_check_margin = std::max(1.0, config_.duration_limit_check_margin);

  return true;
}

PlannerStatus TaskTrajectoryGenerator::generate(
    const Eigen::VectorXd& q_current,
    const Eigen::VectorXd& dq_current,
    const geometry_msgs::PoseStamped& target_pose_msg,
    arm_trajectory::JointTrajectory& tau_task) {
  const int nq = robot_model_ ? robot_model_->nq() : q_current.size();
  return generate(q_current,
                  dq_current,
                  Eigen::VectorXd::Zero(nq),
                  target_pose_msg,
                  tau_task);
}

PlannerStatus TaskTrajectoryGenerator::generate(
    const Eigen::VectorXd& q_current,
    const Eigen::VectorXd& dq_current,
    const Eigen::VectorXd& ddq_current,
    const geometry_msgs::PoseStamped& target_pose_msg,
    arm_trajectory::JointTrajectory& tau_task) {
  tau_task.clear();

  if (!initialized_) {
    last_status_ = PlannerStatus::NOT_INITIALIZED;
    return last_status_;
  }

  if (!robot_model_) {
    last_status_ = PlannerStatus::MISSING_ROBOT_MODEL;
    return last_status_;
  }

  if (q_current.size() != robot_model_->nq()) {
    ROS_ERROR_STREAM("[TaskTrajectoryGenerator] q_current size mismatch. q_current.size = "
                     << q_current.size() << ", robot_model.nq = " << robot_model_->nq());
    last_status_ = PlannerStatus::MISSING_CURRENT_STATE;
    return last_status_;
  }

  Eigen::VectorXd dq_start = Eigen::VectorXd::Zero(robot_model_->nq());
  if (dq_current.size() == robot_model_->nv()) {
    dq_start = dq_current;
  } else {
    ROS_WARN_STREAM_THROTTLE(
        1.0,
        "[TaskTrajectoryGenerator] dq_current size mismatch. dq_current.size = "
            << dq_current.size() << ", robot_model.nv = " << robot_model_->nv()
            << ". Falling back to zero start velocity.");
  }

  Eigen::VectorXd ddq_start = Eigen::VectorXd::Zero(robot_model_->nq());
  if (ddq_current.size() == robot_model_->nv()) {
    ddq_start = clampAccelerationToLimits(ddq_current);
  } else {
    ROS_WARN_STREAM_THROTTLE(
        1.0,
        "[TaskTrajectoryGenerator] ddq_current size mismatch. ddq_current.size = "
            << ddq_current.size() << ", robot_model.nv = " << robot_model_->nv()
            << ". Falling back to zero start acceleration.");
  }

  if (!target_pose_msg.pose.orientation.w &&
      !target_pose_msg.pose.orientation.x &&
      !target_pose_msg.pose.orientation.y &&
      !target_pose_msg.pose.orientation.z) {
    ROS_ERROR("[TaskTrajectoryGenerator] target pose has zero quaternion.");
    last_status_ = PlannerStatus::MISSING_TARGET;
    return last_status_;
  }

  if (!target_pose_msg.header.frame_id.empty() &&
      target_pose_msg.header.frame_id != robot_model_->baseFrame()) {
    ROS_WARN_STREAM_THROTTLE(
        1.0,
        "[TaskTrajectoryGenerator] target_pose frame_id = "
            << target_pose_msg.header.frame_id
            << ", but RobotModel base_frame = "
            << robot_model_->baseFrame()
            << ". Phase I assumes target pose is already expressed in base_frame.");
  }

  Eigen::Isometry3d T_base_target = Eigen::Isometry3d::Identity();
  if (!robot_model_->poseMsgToEigen(target_pose_msg, T_base_target)) {
    ROS_ERROR("[TaskTrajectoryGenerator] Failed to convert target pose to Eigen.");
    last_status_ = PlannerStatus::MISSING_TARGET;
    return last_status_;
  }

  Eigen::VectorXd q_goal;
  const bool ik_ok = robot_model_->solveIK(T_base_target, q_current, q_goal);

  if (!ik_ok || q_goal.size() != robot_model_->nq() || !q_goal.allFinite()) {
    ROS_WARN("[TaskTrajectoryGenerator] IK failed.");
    last_status_ = PlannerStatus::IK_FAILED;
    return last_status_;
  }

  const Eigen::VectorXd q_error = q_goal - q_current;
  const double joint_jump_inf = q_error.lpNorm<Eigen::Infinity>();
  if (config_.reject_large_joint_jump &&
      joint_jump_inf > config_.max_joint_jump_inf_norm) {
    ROS_WARN_STREAM("[TaskTrajectoryGenerator] Rejecting large IK jump. ||q_goal - q_current||_inf = "
                    << joint_jump_inf
                    << ", threshold = "
                    << config_.max_joint_jump_inf_norm);
    last_status_ = PlannerStatus::IK_FAILED;
    return last_status_;
  }

  const Eigen::VectorXd dq_goal_zero = Eigen::VectorXd::Zero(robot_model_->nq());
  const Eigen::VectorXd ddq_goal_zero = Eigen::VectorXd::Zero(robot_model_->nq());

  double duration = computeTrajectoryDuration(q_current, dq_start, q_goal, dq_goal_zero);

  bool generated = false;
  for (int iter = 0; iter <= config_.duration_limit_check_iterations; ++iter) {
    try {
      tau_task = arm_trajectory::JointTrajectory::makeQuinticBoundaryVelocityAcceleration(
          q_current,
          dq_start,
          ddq_start,
          q_goal,
          dq_goal_zero,
          ddq_goal_zero,
          duration,
          config_.trajectory_dt);
    } catch (const std::exception& e) {
      ROS_ERROR_STREAM("[TaskTrajectoryGenerator] Failed to generate trajectory: " << e.what());
      last_status_ = PlannerStatus::INVALID_TRAJECTORY;
      return last_status_;
    }

    if (tau_task.empty()) {
      ROS_ERROR("[TaskTrajectoryGenerator] Generated empty trajectory.");
      last_status_ = PlannerStatus::INVALID_TRAJECTORY;
      return last_status_;
    }

    generated = true;

    if (!config_.enforce_velocity_acceleration_limits) {
      break;
    }

    double scale = 1.0;
    Eigen::VectorXd q_s, dq_s, ddq_s;
    const double check_dt = std::max(0.005, std::min(config_.trajectory_dt, 0.02));
    for (double t = 0.0; t <= duration + 1e-9; t += check_dt) {
      if (!tau_task.sample(std::min(t, duration), q_s, dq_s, ddq_s)) {
        continue;
      }
      for (int i = 0; i < dq_s.size(); ++i) {
        const double v_limit = std::max(config_.joint_velocity_limits[static_cast<std::size_t>(i)], 1e-6);
        const double a_limit = std::max(config_.joint_acceleration_limits[static_cast<std::size_t>(i)], 1e-6);
        scale = std::max(scale, std::abs(dq_s[i]) / v_limit);
        scale = std::max(scale, std::sqrt(std::abs(ddq_s[i]) / a_limit));
      }
    }

    if (scale <= 1.0 + 1e-3) {
      break;
    }

    if (iter == config_.duration_limit_check_iterations) {
      ROS_WARN_STREAM_THROTTLE(
          0.5,
          "[TaskTrajectoryGenerator] Duration limit check reached max iterations. "
              << "remaining_scale=" << scale << ", duration=" << duration);
      break;
    }

    const double new_duration = std::min(config_.max_plan_duration,
                                         std::max(config_.min_plan_duration,
                                                  duration * config_.duration_limit_check_margin * scale));
    if (new_duration <= duration + 1e-6) {
      break;
    }
    duration = new_duration;
  }

  if (!generated) {
    last_status_ = PlannerStatus::INVALID_TRAJECTORY;
    return last_status_;
  }

  last_goal_q_ = q_goal;
  last_status_ = PlannerStatus::SUCCESS;

  Eigen::Isometry3d T_final;
  if (robot_model_->getEndEffectorPose(q_goal, T_final)) {
    const double final_pos_error =
        (T_base_target.translation() - T_final.translation()).norm();

    ROS_INFO_STREAM_THROTTLE(
        1.0,
        "[TaskTrajectoryGenerator] Generated tau_task. duration = "
            << duration
            << ", points = "
            << tau_task.size()
            << ", final position error = "
            << final_pos_error);
  }

  return last_status_;
}

Eigen::VectorXd TaskTrajectoryGenerator::clampAccelerationToLimits(
    const Eigen::VectorXd& ddq) const {
  Eigen::VectorXd out = ddq;
  const int n = std::min(static_cast<int>(config_.joint_acceleration_limits.size()),
                         static_cast<int>(out.size()));
  for (int i = 0; i < n; ++i) {
    const double limit = std::max(config_.joint_acceleration_limits[static_cast<std::size_t>(i)], 1e-6);
    out[i] = std::max(-limit, std::min(limit, out[i]));
  }
  return out;
}

double TaskTrajectoryGenerator::estimateJointDuration(
    double q0,
    double dq0,
    double q1,
    double dq1,
    double velocity_limit,
    double acceleration_limit) const {
  const double d_raw = q1 - q0;
  const double dir = (d_raw >= 0.0) ? 1.0 : -1.0;
  double d = std::abs(d_raw);
  double v0 = dir * dq0;
  double vf = dir * dq1;
  const double vmax = std::max(velocity_limit, 1e-6);
  const double amax = std::max(acceleration_limit, 1e-6);

  if (d < 1e-9) {
    return std::abs(v0 - vf) / amax;
  }

  double extra_time = 0.0;
  if (v0 < 0.0) {
    // Current velocity is moving away from the goal. First brake to zero.
    extra_time += -v0 / amax;
    d += 0.5 * v0 * v0 / amax;
    v0 = 0.0;
  }
  if (vf < 0.0) {
    vf = 0.0;
  }

  v0 = std::min(v0, vmax);
  vf = std::min(vf, vmax);

  const double d_acc = std::max(0.0, (vmax * vmax - v0 * v0) / (2.0 * amax));
  const double d_dec = std::max(0.0, (vmax * vmax - vf * vf) / (2.0 * amax));

  double t = 0.0;
  if (d_acc + d_dec <= d) {
    t = (vmax - v0) / amax
        + (d - d_acc - d_dec) / vmax
        + (vmax - vf) / amax;
  } else {
    const double v_peak_sq = std::max(0.0, amax * d + 0.5 * (v0 * v0 + vf * vf));
    const double v_peak = std::sqrt(v_peak_sq);
    t = std::max(0.0, (v_peak - v0) / amax)
        + std::max(0.0, (v_peak - vf) / amax);
  }

  return extra_time + std::max(0.0, t);
}

double TaskTrajectoryGenerator::computeTrajectoryDuration(
    const Eigen::VectorXd& q_current,
    const Eigen::VectorXd& dq_start,
    const Eigen::VectorXd& q_goal,
    const Eigen::VectorXd& dq_goal) const {
  double duration = config_.T_plan;

  if (config_.enable_time_scaling) {
    duration = 0.0;
    const Eigen::VectorXd q_error = q_goal - q_current;
    for (int i = 0; i < q_error.size(); ++i) {
      const double v_limit = std::max(config_.joint_velocity_limits[static_cast<std::size_t>(i)], 1e-6);
      const double a_limit = std::max(config_.joint_acceleration_limits[static_cast<std::size_t>(i)], 1e-6);
      const double t_vel = std::abs(q_error[i]) / v_limit;
      const double t_acc = std::sqrt(5.8 * std::abs(q_error[i]) / a_limit);
      const double t_state = estimateJointDuration(q_current[i],
                                                   dq_start[i],
                                                   q_goal[i],
                                                   dq_goal[i],
                                                   v_limit,
                                                   a_limit);
      duration = std::max(duration, std::max(t_state, std::max(t_vel, t_acc)));
    }
  }

  duration = std::max(duration, config_.min_plan_duration);
  duration = std::min(duration, config_.max_plan_duration);
  return duration;
}

const TaskTrajectoryGeneratorConfig& TaskTrajectoryGenerator::config() const {
  return config_;
}

Eigen::VectorXd TaskTrajectoryGenerator::lastGoalQ() const {
  return last_goal_q_;
}

PlannerStatus TaskTrajectoryGenerator::lastStatus() const {
  return last_status_;
}

std::string TaskTrajectoryGenerator::lastStatusString() const {
  return plannerStatusToString(last_status_);
}

}  // namespace egocentric_arm_planner
