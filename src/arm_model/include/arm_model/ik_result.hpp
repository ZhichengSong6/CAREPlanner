#pragma once

#include <cmath>
#include <limits>

namespace arm_model {
enum class IKStatus { FAILED, APPROXIMATE, CONVERGED };
struct IKResult {
  IKStatus status = IKStatus::FAILED;
  double position_error = std::numeric_limits<double>::infinity();
  double rotation_error = std::numeric_limits<double>::infinity();
  int iterations = 0;
  bool converged() const { return status == IKStatus::CONVERGED; }
};

// The historical 5x exit remains distinguishable, never an exact success.
inline IKStatus classifyIK(double p, double r, double p_tol, double r_tol) {
  if (!std::isfinite(p) || !std::isfinite(r) || p < 0 || r < 0 ||
      !std::isfinite(p_tol) || !std::isfinite(r_tol) || p_tol <= 0 || r_tol <= 0)
    return IKStatus::FAILED;
  if (p < p_tol && r < r_tol) return IKStatus::CONVERGED;
  if (p < 5 * p_tol && r < 5 * r_tol) return IKStatus::APPROXIMATE;
  return IKStatus::FAILED;
}
}  // namespace arm_model
