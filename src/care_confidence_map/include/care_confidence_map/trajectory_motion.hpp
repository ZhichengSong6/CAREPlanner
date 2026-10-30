#pragma once
#include <Eigen/Core>
#include <trajectory_msgs/JointTrajectory.h>
#include <cmath>
#include <vector>

namespace care_confidence_map {
// q(s) = (1-s)q0 + s q1 + 4s(1-s)c. A trapezoid-integrated
// velocity ramp has c = h(v0-v1)/8. Legacy position knots remain linear.
inline Eigen::VectorXd trajectoryCurvature(
    const trajectory_msgs::JointTrajectoryPoint& a,
    const trajectory_msgs::JointTrajectoryPoint& b,
    const std::vector<int>& mapping) {
  Eigen::VectorXd c=Eigen::VectorXd::Zero(mapping.size());
  const double h=(b.time_from_start-a.time_from_start).toSec();
  if (!(h>0) || !std::isfinite(h)) return c;
  for (const int j:mapping) {
    if (j<0 || std::size_t(j)>=a.positions.size() || std::size_t(j)>=b.positions.size() ||
        std::size_t(j)>=a.velocities.size() || std::size_t(j)>=b.velocities.size()) return c;
    const double delta=b.positions[j]-a.positions[j];
    const double integral=.5*h*(a.velocities[j]+b.velocities[j]);
    if (!std::isfinite(integral) || std::abs(delta-integral)>1e-9) return c;
  }
  for(std::size_t i=0;i<mapping.size();++i) {
    const int j=mapping[i];c[i]=h*(a.velocities[j]-b.velocities[j])/8.;
  }
  return c;
}
inline std::vector<Eigen::VectorXd> trajectoryCurvatures(
    const trajectory_msgs::JointTrajectory& trajectory,
    const std::vector<int>& mapping) {
  std::vector<Eigen::VectorXd> out;
  for(std::size_t k=1;k<trajectory.points.size();++k)
    out.push_back(trajectoryCurvature(trajectory.points[k-1],trajectory.points[k],mapping));
  return out;
}
}  // namespace care_confidence_map
