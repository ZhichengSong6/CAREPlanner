#include "egocentric_arm_planner/task_trajectory_generator.hpp"

#include <algorithm>
#include <cmath>

namespace egocentric_arm_planner {

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
  ROS_INFO("[TaskTrajectoryGenerator] interpolation = quintic zero-velocity-zero-acceleration");
  ROS_INFO_STREAM("[TaskTrajectoryGenerator] enable_time_scaling = "
                  << static_cast<int>(config_.enable_time_scaling));

  return true;
}

bool TaskTrajectoryGenerator::initialized() const {
  return initialized_;
}

bool TaskTrajectoryGenerator::loadConfig(const ros::NodeHandle& nh) {
  nh.param<double>("task_generator/T_plan",
                   config_.T_plan,
                   config_.T_plan);

  nh.param<double>("task_generator/trajectory_dt",
                   config_.trajectory_dt,
                   config_.trajectory_dt);

  nh.param<bool>("task_generator/enable_time_scaling",
                 config_.enable_time_scaling,
                 config_.enable_time_scaling);

  nh.param<double>("task_generator/nominal_max_joint_velocity",
                   config_.nominal_max_joint_velocity,
                   config_.nominal_max_joint_velocity);

  nh.param<double>("task_generator/min_plan_duration",
                   config_.min_plan_duration,
                   config_.min_plan_duration);

  nh.param<double>("task_generator/max_plan_duration",
                   config_.max_plan_duration,
                   config_.max_plan_duration);

  nh.param<bool>("task_generator/reject_large_joint_jump",
                 config_.reject_large_joint_jump,
                 config_.reject_large_joint_jump);

  nh.param<double>("task_generator/max_joint_jump_inf_norm",
                   config_.max_joint_jump_inf_norm,
                   config_.max_joint_jump_inf_norm);

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

  if (config_.min_plan_duration <= 0.0 ||
      config_.max_plan_duration < config_.min_plan_duration) {
    ROS_ERROR("[TaskTrajectoryGenerator] Invalid min/max plan duration.");
    return false;
  }

  return true;
}

PlannerStatus TaskTrajectoryGenerator::generate(
    const Eigen::VectorXd& q_current,
    const Eigen::VectorXd& dq_current,
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

  if (dq_current.size() != robot_model_->nv()) {
    ROS_WARN_STREAM_THROTTLE(
        1.0,
        "[TaskTrajectoryGenerator] dq_current size mismatch. dq_current.size = "
            << dq_current.size() << ", robot_model.nv = " << robot_model_->nv()
            << ". Phase I generator does not use dq_current yet.");
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

  const Eigen::VectorXd dq_goal = q_goal - q_current;
  const double joint_jump_inf = dq_goal.lpNorm<Eigen::Infinity>();

  if (config_.reject_large_joint_jump &&
      joint_jump_inf > config_.max_joint_jump_inf_norm) {
    ROS_WARN_STREAM("[TaskTrajectoryGenerator] Rejecting large IK jump. ||q_goal - q_current||_inf = "
                    << joint_jump_inf
                    << ", threshold = "
                    << config_.max_joint_jump_inf_norm);
    last_status_ = PlannerStatus::IK_FAILED;
    return last_status_;
  }

  const double duration = computeTrajectoryDuration(q_current, q_goal);

  try {
    tau_task =
        arm_trajectory::JointTrajectory::makeQuinticZeroVelocityAcceleration(
            q_current,
            q_goal,
            duration,
            config_.trajectory_dt);
  } catch (const std::exception& e) {
    ROS_ERROR_STREAM("[TaskTrajectoryGenerator] Failed to generate trajectory: "
                    << e.what());
    last_status_ = PlannerStatus::INVALID_TRAJECTORY;
    return last_status_;
  }

  if (tau_task.empty()) {
    ROS_ERROR("[TaskTrajectoryGenerator] Generated empty trajectory.");
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

double TaskTrajectoryGenerator::computeTrajectoryDuration(
    const Eigen::VectorXd& q_current,
    const Eigen::VectorXd& q_goal) const {
  double duration = config_.T_plan;

  if (config_.enable_time_scaling) {
    const double max_joint_motion =
        (q_goal - q_current).lpNorm<Eigen::Infinity>();

    const double required_duration =
        max_joint_motion / config_.nominal_max_joint_velocity;

    duration = std::max(duration, required_duration);
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