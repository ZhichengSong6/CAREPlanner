#pragma once
#include <Eigen/SparseCore>
#include <cmath>

namespace egocentric_arm_planner {
// Necessary condition only: no conflicts does NOT prove QP feasibility.
inline int lowerBoxConflicts(const Eigen::SparseMatrix<double>& G,
                             const Eigen::VectorXd& lower,
                             const Eigen::VectorXd& xl,
                             const Eigen::VectorXd& xu, int first_row) {
  Eigen::VectorXd best = Eigen::VectorXd::Zero(G.rows());
  for (int col=0; col<G.outerSize(); ++col)
    for (Eigen::SparseMatrix<double>::InnerIterator it(G,col); it; ++it)
      if (it.value()!=0.) best[it.row()] += it.value()*(it.value()>0 ? xu[col] : xl[col]);
  int conflicts=0;
  for (int row=first_row; row<G.rows(); ++row)
    if (std::isfinite(best[row]) && lower[row]-best[row]>1e-8) ++conflicts;
  return conflicts;
}
}  // namespace egocentric_arm_planner
