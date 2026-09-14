#pragma once
#include <Eigen/Core>
#include <algorithm>
#include <cmath>

namespace egocentric_arm_planner {
// Linearization seed only. It is NOT a safe/executable trajectory certificate.
inline bool measuredBrakingSeed(const Eigen::VectorXd& measured,
                               const Eigen::VectorXd& command,
                               const Eigen::VectorXd& acceleration_limits,
                               int intervals, double dt,
                               Eigen::MatrixXd& q, Eigen::MatrixXd& u) {
  if (!measured.size() || command.size() != measured.size() ||
      acceleration_limits.size() != measured.size() || intervals < 1 ||
      !std::isfinite(dt) || dt <= 0 || !measured.allFinite() ||
      !command.allFinite() || !acceleration_limits.allFinite() ||
      (acceleration_limits.array() <= 0).any()) return false;
  q.resize(measured.size(), intervals+1); u.resize(measured.size(), intervals);
  q.col(0) = measured;
  Eigen::VectorXd velocity = command;
  for (int k=0; k<intervals; ++k) {
    for (int j=0; j<velocity.size(); ++j)
      velocity[j] = std::copysign(std::max(0., std::abs(velocity[j])-acceleration_limits[j]*dt), velocity[j]);
    u.col(k) = velocity;
    q.col(k+1) = q.col(k) + dt*velocity;
  }
  return q.allFinite() && u.allFinite();
}
}  // namespace egocentric_arm_planner
