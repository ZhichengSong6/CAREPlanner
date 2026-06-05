#include "egocentric_arm_planner/intervention_manager.hpp"

namespace egocentric_arm_planner {

bool InterventionManager::initialize(const ros::NodeHandle& nh) {
  nh.param<double>("intervention/T_cmd",
                   config_.T_cmd,
                   config_.T_cmd);

  nh.param<double>("intervention/hold_duration",
                   config_.hold_duration,
                   config_.hold_duration);

  nh.param<double>("intervention/command_dt",
                   config_.command_dt,
                   config_.command_dt);

  if (config_.T_cmd <= 0.0) {
    ROS_ERROR("[InterventionManager] T_cmd must be positive.");
    initialized_ = false;
    return false;
  }

  if (config_.hold_duration <= 0.0) {
    ROS_ERROR("[InterventionManager] hold_duration must be positive.");
    initialized_ = false;
    return false;
  }

  if (config_.command_dt <= 0.0) {
    ROS_ERROR("[InterventionManager] command_dt must be positive.");
    initialized_ = false;
    return false;
  }

  initialized_ = true;

  ROS_INFO_STREAM("[InterventionManager] Initialized.");
  ROS_INFO_STREAM("[InterventionManager] T_cmd = " << config_.T_cmd);
  ROS_INFO_STREAM("[InterventionManager] hold_duration = " << config_.hold_duration);
  ROS_INFO_STREAM("[InterventionManager] command_dt = " << config_.command_dt);

  return true;
}

bool InterventionManager::initialized() const {
  return initialized_;
}

PlannerStatus InterventionManager::decideCommand(
    const arm_trajectory::JointTrajectory& tau_task,
    const EvaluationResult& eval,
    const Eigen::VectorXd& q_current,
    arm_trajectory::JointTrajectory& tau_cmd) {
  tau_cmd.clear();

  if (!initialized_) {
    ROS_ERROR("[InterventionManager] decideCommand called before initialization.");
    return PlannerStatus::NOT_INITIALIZED;
  }

  if (!eval.valid) {
    ROS_WARN_STREAM("[InterventionManager] Invalid evaluation result. Holding. Message: "
                    << eval.message);
    return makeHoldCommand(q_current, tau_cmd);
  }

  switch (eval.mode) {
    case InterventionMode::EXECUTE_TASK_PREFIX:
      return makeTaskPrefixCommand(tau_task, tau_cmd);

    case InterventionMode::HOLD:
      return makeHoldCommand(q_current, tau_cmd);

    case InterventionMode::ACTIVE_SENSING:
      ROS_WARN("[InterventionManager] ACTIVE_SENSING not implemented in Phase I. Holding.");
      return makeHoldCommand(q_current, tau_cmd);

    case InterventionMode::RETREAT:
      ROS_WARN("[InterventionManager] RETREAT not implemented in Phase I. Holding.");
      return makeHoldCommand(q_current, tau_cmd);

    case InterventionMode::REPLAN:
      ROS_WARN("[InterventionManager] REPLAN not implemented in Phase I. Holding.");
      return makeHoldCommand(q_current, tau_cmd);

    default:
      ROS_WARN("[InterventionManager] Unknown intervention mode. Holding.");
      return makeHoldCommand(q_current, tau_cmd);
  }
}

PlannerStatus InterventionManager::makeTaskPrefixCommand(
    const arm_trajectory::JointTrajectory& tau_task,
    arm_trajectory::JointTrajectory& tau_cmd) const {
  tau_cmd.clear();

  if (tau_task.empty()) {
    ROS_ERROR("[InterventionManager] Cannot create task prefix from empty tau_task.");
    return PlannerStatus::INVALID_TRAJECTORY;
  }

  const double t0 = tau_task.startTime();
  const double t1 = std::min(tau_task.endTime(), t0 + config_.T_cmd);

  tau_cmd = tau_task.truncate(t0, t1);

  if (tau_cmd.empty()) {
    ROS_ERROR("[InterventionManager] Generated empty tau_cmd from task prefix.");
    return PlannerStatus::INVALID_TRAJECTORY;
  }

  return PlannerStatus::SUCCESS;
}

PlannerStatus InterventionManager::makeHoldCommand(
    const Eigen::VectorXd& q_current,
    arm_trajectory::JointTrajectory& tau_cmd) const {
  tau_cmd.clear();

  if (q_current.size() <= 0) {
    ROS_ERROR("[InterventionManager] Cannot hold because q_current is empty.");
    return PlannerStatus::MISSING_CURRENT_STATE;
  }

  try {
    tau_cmd = arm_trajectory::JointTrajectory::makeHold(
        q_current,
        config_.hold_duration,
        config_.command_dt);
  } catch (const std::exception& e) {
    ROS_ERROR_STREAM("[InterventionManager] Failed to generate hold command: "
                     << e.what());
    return PlannerStatus::INVALID_TRAJECTORY;
  }

  if (tau_cmd.empty()) {
    ROS_ERROR("[InterventionManager] Generated empty hold trajectory.");
    return PlannerStatus::INVALID_TRAJECTORY;
  }

  return PlannerStatus::SUCCESS;
}

const InterventionManagerConfig& InterventionManager::config() const {
  return config_;
}

}  // namespace egocentric_arm_planner