#pragma once
#include <Eigen/Core>

namespace egocentric_arm_planner {
// Liveness policy, not a feasibility/safety threshold. Solver success is not
// execution progress; reference identity or measured motion resets the episode.
struct QPNoProgress {
  int failures = 0;
  int max_failures = 5;
  double motion_epsilon = 0.01;
  Eigen::VectorXd anchor;
  bool blocked() const { return failures >= max_failures; }
  void reset() { failures = 0; anchor.resize(0); }
  bool observe(const Eigen::VectorXd& measured) {
    if (!measured.size() || !measured.allFinite()) return false;
    if (anchor.size() != measured.size() ||
        (measured - anchor).lpNorm<Eigen::Infinity>() > motion_epsilon) {
      failures = 0;
      anchor = measured;
      return true;
    }
    return false;
  }
  bool fail(const Eigen::VectorXd& measured) {
    observe(measured);
    if (!blocked()) ++failures;
    return blocked();
  }
};
}  // namespace egocentric_arm_planner
