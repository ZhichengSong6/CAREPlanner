#pragma once

#include <Eigen/Core>
#include <cmath>

namespace egocentric_arm_planner {

// Liveness evidence from distinct, already active certified executions.
// This policy does not classify obstacles or authorize a mode transition.
struct NormalActiveNoProgress {
  double min_duration_s = 8.0;
  double max_sample_gap_s = 1.0;
  double motion_epsilon = 0.01;
  int min_execution_stamps = 5;
  Eigen::VectorXd anchor;
  double first_time_s = -1.0, last_time_s = -1.0;
  unsigned long long last_execution_stamp_ns = 0;
  int execution_stamps = 0;
  bool emitted = false;

  void reset() {
    anchor.resize(0);
    first_time_s = last_time_s = -1.0;
    last_execution_stamp_ns = 0;
    execution_stamps = 0;
    emitted = false;
  }

  double duration(double now_s) const {
    return first_time_s >= 0.0 ? now_s - first_time_s : 0.0;
  }

  bool observe(double now_s, unsigned long long execution_stamp_ns,
               const Eigen::VectorXd& measured, bool target_far) {
    if (!target_far || !execution_stamp_ns || measured.size() == 0 ||
        !measured.allFinite() || !std::isfinite(now_s)) {
      reset();
      return false;
    }
    if (anchor.size() != measured.size() || first_time_s < 0.0 ||
        now_s < last_time_s || now_s - last_time_s > max_sample_gap_s ||
        (measured - anchor).lpNorm<Eigen::Infinity>() > motion_epsilon) {
      reset();
      anchor = measured;
      first_time_s = now_s;
    }
    last_time_s = now_s;
    if (execution_stamp_ns > last_execution_stamp_ns) {
      last_execution_stamp_ns = execution_stamp_ns;
      ++execution_stamps;
    }
    if (!emitted && execution_stamps >= min_execution_stamps &&
        duration(now_s) >= min_duration_s) {
      emitted = true;
      return true;
    }
    return false;
  }
};

}  // namespace egocentric_arm_planner
