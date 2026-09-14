#pragma once
#include "arm_trajectory/joint_trajectory.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace egocentric_arm_planner {
// Keep the existing receiver tolerance. Duration scaling is NOT acceptance.
constexpr double kTaskDynamicsRatioTolerance = 1.001;
struct TaskDynamicsCheck {
  bool valid = false;
  double velocity_ratio = 0.0;
  double acceleration_ratio = 0.0;
  bool accepted() const {
    return valid && velocity_ratio <= kTaskDynamicsRatioTolerance &&
           acceleration_ratio <= kTaskDynamicsRatioTolerance;
  }
  double durationScale() const {
    return std::max(1.0, std::max(velocity_ratio, std::sqrt(acceleration_ratio)));
  }
};

inline TaskDynamicsCheck checkTaskDynamics(
    const arm_trajectory::JointTrajectory& trajectory,
    const std::vector<double>& velocity_limits,
    const std::vector<double>& acceleration_limits, double trajectory_dt) {
  TaskDynamicsCheck out;
  if (trajectory.empty() || !std::isfinite(trajectory_dt) || trajectory_dt <= 0 ||
      velocity_limits.size() != acceleration_limits.size() ||
      velocity_limits.size() != static_cast<std::size_t>(trajectory.dof())) return out;
  for (std::size_t i = 0; i < velocity_limits.size(); ++i) {
    if (!std::isfinite(velocity_limits[i]) || velocity_limits[i] <= 0 ||
        !std::isfinite(acceleration_limits[i]) || acceleration_limits[i] <= 0) return out;
  }
  const double start = trajectory.startTime(), end = trajectory.endTime();
  if (!std::isfinite(start) || !std::isfinite(end) || end < start) return out;
  const double dt = std::max(0.005, std::min(trajectory_dt, 0.02));
  const int steps = static_cast<int>(std::ceil((end-start)/dt));
  Eigen::VectorXd q, dq, ddq;
  for (int step = 0; step <= steps; ++step) {
    if (!trajectory.sample(std::min(start + step*dt, end), q, dq, ddq) ||
        dq.size() != trajectory.dof() || ddq.size() != trajectory.dof() ||
        !q.allFinite() || !dq.allFinite() || !ddq.allFinite()) return out;
    for (int i = 0; i < dq.size(); ++i) {
      out.velocity_ratio = std::max(out.velocity_ratio, std::abs(dq[i])/velocity_limits[i]);
      out.acceleration_ratio = std::max(out.acceleration_ratio, std::abs(ddq[i])/acceleration_limits[i]);
    }
  }
  out.valid = true;
  return out;
}
}  // namespace egocentric_arm_planner
