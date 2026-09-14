#pragma once
#include <algorithm>
#include <cmath>

namespace egocentric_arm_planner {
// A persistent acquisition target is a terminal objective, not an expiring
// task deadline. Real multi-deadline callers retain their original indexing.
inline int visibilityObjectiveStep(bool terminal, double deadline, double now,
                                   double dt, int intervals) {
  if (intervals < 1 || !std::isfinite(dt) || dt <= 0.0) return -1;
  if (terminal) return intervals;
  if (!std::isfinite(deadline) || !std::isfinite(now)) return -1;
  const double step = std::ceil((deadline - now) / dt);
  if (step > intervals) return -1;
  return step <= 1.0 ? 1 : static_cast<int>(step);
}
}  // namespace egocentric_arm_planner
