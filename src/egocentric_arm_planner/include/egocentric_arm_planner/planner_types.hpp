#pragma once

#include <string>
#include <limits>

namespace egocentric_arm_planner {

enum class PlannerStatus {
  SUCCESS = 0,

  NOT_INITIALIZED,
  MISSING_ROBOT_MODEL,
  MISSING_CURRENT_STATE,
  MISSING_TARGET,

  IK_FAILED,
  INVALID_TRAJECTORY,
  EVALUATION_FAILED,
  INTERVENTION_FAILED,

  UNKNOWN_ERROR
};

inline std::string plannerStatusToString(const PlannerStatus status) {
  switch (status) {
    case PlannerStatus::SUCCESS:
      return "SUCCESS";
    case PlannerStatus::NOT_INITIALIZED:
      return "NOT_INITIALIZED";
    case PlannerStatus::MISSING_ROBOT_MODEL:
      return "MISSING_ROBOT_MODEL";
    case PlannerStatus::MISSING_CURRENT_STATE:
      return "MISSING_CURRENT_STATE";
    case PlannerStatus::MISSING_TARGET:
      return "MISSING_TARGET";
    case PlannerStatus::IK_FAILED:
      return "IK_FAILED";
    case PlannerStatus::INVALID_TRAJECTORY:
      return "INVALID_TRAJECTORY";
    case PlannerStatus::EVALUATION_FAILED:
      return "EVALUATION_FAILED";
    case PlannerStatus::INTERVENTION_FAILED:
      return "INTERVENTION_FAILED";
    case PlannerStatus::UNKNOWN_ERROR:
      return "UNKNOWN_ERROR";
    default:
      return "UNRECOGNIZED_STATUS";
  }
}

enum class InterventionMode {
  EXECUTE_TASK_PREFIX = 0,
  ACTIVE_SENSING,
  RETREAT,
  REPLAN,
  HOLD
};

inline std::string interventionModeToString(const InterventionMode mode) {
  switch (mode) {
    case InterventionMode::EXECUTE_TASK_PREFIX:
      return "EXECUTE_TASK_PREFIX";
    case InterventionMode::ACTIVE_SENSING:
      return "ACTIVE_SENSING";
    case InterventionMode::RETREAT:
      return "RETREAT";
    case InterventionMode::REPLAN:
      return "REPLAN";
    case InterventionMode::HOLD:
      return "HOLD";
    default:
      return "UNRECOGNIZED_INTERVENTION_MODE";
  }
}

enum class RiskType {
  NONE = 0,
  KNOWN_OCCUPIED,
  UNKNOWN,
  STALE,
  LOW_CONFIDENCE,
  RGB_RISK,
  NEAR_FIELD_BLIND
};

inline std::string riskTypeToString(const RiskType risk_type) {
  switch (risk_type) {
    case RiskType::NONE:
      return "NONE";
    case RiskType::KNOWN_OCCUPIED:
      return "KNOWN_OCCUPIED";
    case RiskType::UNKNOWN:
      return "UNKNOWN";
    case RiskType::STALE:
      return "STALE";
    case RiskType::LOW_CONFIDENCE:
      return "LOW_CONFIDENCE";
    case RiskType::RGB_RISK:
      return "RGB_RISK";
    case RiskType::NEAR_FIELD_BLIND:
      return "NEAR_FIELD_BLIND";
    default:
      return "UNRECOGNIZED_RISK_TYPE";
  }
}

enum class RiskTrend {
  NONE = 0,
  INCREASING,
  DECREASING,
  FLAT
};

inline std::string riskTrendToString(const RiskTrend trend) {
  switch (trend) {
    case RiskTrend::NONE:
      return "NONE";
    case RiskTrend::INCREASING:
      return "INCREASING";
    case RiskTrend::DECREASING:
      return "DECREASING";
    case RiskTrend::FLAT:
      return "FLAT";
    default:
      return "UNRECOGNIZED_RISK_TREND";
  }
}

struct TaskTrajectoryGeneratorConfig {
  double T_plan = 2.0;
  double trajectory_dt = 0.02;

  bool enable_time_scaling = true;
  double nominal_max_joint_velocity = 0.5;

  double min_plan_duration = 0.5;
  double max_plan_duration = 5.0;

  bool reject_large_joint_jump = true;
  double max_joint_jump_inf_norm = 3.14;
};

struct EvaluationResult {
  bool valid = false;

  bool command_horizon_safe = true;

  double t_risk = std::numeric_limits<double>::infinity();
  double t_safe = std::numeric_limits<double>::infinity();
  double t_switch = std::numeric_limits<double>::infinity();

  double min_known_obstacle_clearance =
      std::numeric_limits<double>::infinity();

  double min_risk_clearance =
      std::numeric_limits<double>::infinity();

  RiskType dominant_risk_type = RiskType::NONE;
  RiskTrend risk_trend = RiskTrend::NONE;

  int affected_link_index = -1;

  bool recoverable_by_active_sensing = false;

  InterventionMode mode = InterventionMode::EXECUTE_TASK_PREFIX;

  std::string message;
};

struct InterventionManagerConfig {
  double T_cmd = 0.5;
  double hold_duration = 0.5;
  double command_dt = 0.02;
};

}  // namespace egocentric_arm_planner