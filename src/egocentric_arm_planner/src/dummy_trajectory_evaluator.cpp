#include "egocentric_arm_planner/dummy_trajectory_evaluator.hpp"

namespace egocentric_arm_planner {

bool DummyTrajectoryEvaluator::initialize(const ros::NodeHandle& nh) {
  nh.param<double>("intervention/T_cmd", T_cmd_, T_cmd_);

  if (T_cmd_ <= 0.0) {
    ROS_ERROR("[DummyTrajectoryEvaluator] T_cmd must be positive.");
    initialized_ = false;
    return false;
  }

  initialized_ = true;

  ROS_INFO_STREAM("[DummyTrajectoryEvaluator] Initialized. T_cmd = " << T_cmd_);

  return true;
}

bool DummyTrajectoryEvaluator::initialized() const {
  return initialized_;
}

PlannerStatus DummyTrajectoryEvaluator::evaluate(
    const arm_trajectory::JointTrajectory& tau_task,
    EvaluationResult& result) {
  result = EvaluationResult();

  if (!initialized_) {
    result.valid = false;
    result.mode = InterventionMode::HOLD;
    result.message = "DummyTrajectoryEvaluator is not initialized.";
    return PlannerStatus::NOT_INITIALIZED;
  }

  if (tau_task.empty()) {
    result.valid = false;
    result.mode = InterventionMode::HOLD;
    result.message = "tau_task is empty.";
    return PlannerStatus::INVALID_TRAJECTORY;
  }

  result.valid = true;
  result.command_horizon_safe = true;

  result.t_risk = std::numeric_limits<double>::infinity();
  result.t_safe = std::numeric_limits<double>::infinity();
  result.t_switch = std::numeric_limits<double>::infinity();

  result.min_known_obstacle_clearance = std::numeric_limits<double>::infinity();
  result.min_risk_clearance = std::numeric_limits<double>::infinity();

  result.dominant_risk_type = RiskType::NONE;
  result.risk_trend = RiskTrend::NONE;
  result.affected_link_index = -1;
  result.recoverable_by_active_sensing = false;

  result.mode = InterventionMode::EXECUTE_TASK_PREFIX;
  result.message = "Dummy evaluator: all safe.";

  return PlannerStatus::SUCCESS;
}

}  // namespace egocentric_arm_planner