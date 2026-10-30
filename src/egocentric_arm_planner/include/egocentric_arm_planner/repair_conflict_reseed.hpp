#pragma once
#include <Eigen/Core>
#include <cmath>

namespace egocentric_arm_planner {

// Internal relinearization only, within the existing SCP iteration budget.
// A measured hold already used as the center has nothing to restore. Every
// new center must be queried and hard-solved before it can become a candidate.
struct RepairConflictReseed {
  bool used = false;

  bool consume(bool current_fresh_repair, int conflict_rows,
               const Eigen::MatrixXd& center, const Eigen::MatrixXd& seed,
               int iteration, int max_iterations, double minimum_change) {
    if (!current_fresh_repair || conflict_rows <= 0 || used ||
        iteration < 0 || iteration + 1 >= max_iterations ||
        center.rows() == 0 || center.cols() < 2 ||
        center.rows() != seed.rows() || center.cols() != seed.cols() ||
        !center.allFinite() || !seed.allFinite() ||
        !std::isfinite(minimum_change) || minimum_change < 0. ||
        (center.col(0)-seed.col(0)).lpNorm<Eigen::Infinity>() > 1e-12 ||
        (center-seed).cwiseAbs().maxCoeff() <= minimum_change) {
      return false;
    }
    used = true;
    return true;
  }
};

}  // namespace egocentric_arm_planner
