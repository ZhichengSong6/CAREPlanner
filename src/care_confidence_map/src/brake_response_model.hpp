#pragma once

#include <care_confidence_map/trajectory_motion.hpp>
#include <algorithm>
#include <vector>

namespace care_confidence_map {

// Predict the two coherent velocity responses used by the research candidate
// gate. The native verifier maps both curves to full-body continuous sweeps.
inline std::vector<std::vector<Eigen::VectorXd>> predictBrakeResponse(
    const trajectory_msgs::JointTrajectory& trajectory, const std::vector<int>& mapping,
    const Eigen::VectorXd& position, const Eigen::VectorXd& velocity,
    const Eigen::VectorXd& command, double state_age) {
  constexpr double step = .01, acceleration = .5, gain = 2.5, max_command = 2.;
  constexpr double residual_horizon = .30;
  const auto& points = trajectory.points;
  const double end = points.back().time_from_start.toSec();
  std::vector<Eigen::VectorXd> positions;
  for (const auto& point : points) {
    Eigen::VectorXd ordered(mapping.size());
    for (std::size_t j = 0; j < mapping.size(); ++j)
      ordered[j] = point.positions[mapping[j]];
    positions.push_back(std::move(ordered));
  }
  const auto curvature = trajectoryCurvatures(trajectory, mapping);
  std::vector<std::vector<Eigen::VectorXd>> paths;
  for (const double velocity_response : {.025, .060}) {
    const double response = -std::expm1(-step / velocity_response);
    Eigen::VectorXd q = position + state_age * velocity;
    Eigen::VectorXd v = velocity, u = command;
    std::vector<Eigen::VectorXd> path{position, q};
    std::size_t segment = 0;
    for (double time = step / 2; time < end + residual_horizon; time += step) {
      Eigen::VectorXd request = Eigen::VectorXd::Zero(q.size());
      if (time < end && points.size() > 1) {
        while (points[segment + 1].time_from_start.toSec() < time) ++segment;
        const double left = points[segment].time_from_start.toSec();
        const double duration = points[segment + 1].time_from_start.toSec() - left;
        const double s = std::max(0., (time - left) / duration);
        const auto& a = positions[segment];
        const auto& b = positions[segment + 1];
        const auto& c = curvature[segment];
        const Eigen::VectorXd reference = (1. - s) * a + s * b + (4. * s * (1. - s)) * c;
        request = (b - a + (4. - 8. * s) * c) / duration + gain * (reference - q);
        request = request.cwiseMax(-max_command).cwiseMin(max_command);
      }
      u += (request - u).cwiseMax(-acceleration * step).cwiseMin(acceleration * step);
      const Eigen::VectorXd next = v + response * (u - v);
      q += (.5 * step) * (v + next);
      path.push_back(q);
      v = next;
      if (time >= end && v.lpNorm<Eigen::Infinity>() < .001 &&
          u.lpNorm<Eigen::Infinity>() < .001) break;
    }
    paths.push_back(std::move(path));
  }
  return paths;
}

}  // namespace care_confidence_map
