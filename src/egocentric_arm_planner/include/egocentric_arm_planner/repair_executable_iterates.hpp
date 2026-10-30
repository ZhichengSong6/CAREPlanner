#pragma once
#include <Eigen/Core>
#include <cmath>
#include <string>

namespace egocentric_arm_planner {

// Internal candidate-generation state only. Restoring an iterate never grants
// publication: a fresh query and complete hard solve must follow it. The last
// existing SCP iteration can recheck fixed q/u rather than spend a new solve.
struct RepairExecutableIterates {
  std::string context;
  Eigen::MatrixXd q, u;
  double step_scale = 1.;
  bool recheck_only = false;

  void bind(const std::string& current) {
    if (context == current) return;
    *this = RepairExecutableIterates{};
    context = current;
  }
  bool available() const { return q.rows() > 0; }
  bool retain(const Eigen::MatrixXd& candidate_q,
              const Eigen::MatrixXd& candidate_u) {
    if (candidate_q.rows() < 1 || candidate_q.cols() < 2 ||
        candidate_u.rows() != candidate_q.rows() ||
        candidate_u.cols()+1 != candidate_q.cols() ||
        !candidate_q.allFinite() || !candidate_u.allFinite()) return false;
    q = candidate_q; u = candidate_u;
    return true;
  }
  void rejectTrial() { step_scale *= .5; }
  void invalidateCurve() {
    q.resize(0,0); u.resize(0,0); recheck_only=false;
  }
  bool restore(const Eigen::VectorXd& start,
               Eigen::MatrixXd* next_q, Eigen::MatrixXd* next_u) const {
    if (!available() || !next_q || !next_u || start.size() != q.rows() ||
        !start.allFinite() || (start-q.col(0)).lpNorm<Eigen::Infinity>() > 1e-12)
      return false;
    *next_q = q; *next_u = u;
    return true;
  }
  bool reserveRecheck(int consumed, int maximum,
                     const Eigen::VectorXd& start,
                     Eigen::MatrixXd* next_q, Eigen::MatrixXd* next_u) {
    if (maximum < 2 || consumed != maximum-1 ||
        !restore(start,next_q,next_u)) return false;
    recheck_only = true;
    return true;
  }
};
}  // namespace egocentric_arm_planner
