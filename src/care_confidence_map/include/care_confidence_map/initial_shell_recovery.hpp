#pragma once
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <map>
#include <tuple>

namespace care_confidence_map {
// A point's first recovery baseline is diagnostic progress history. It survives
// rejected candidates and target changes, but cannot prevent a safe escape
// from an already regressed measured state. The owner's finite attempt ledger
// remains responsible for terminating recovery without progress.
// This ledger grants no execution permission: the complete continuous path,
// braking, other unknowns and current measured state must also pass their gates.
struct InitialShellRecovery {
  using Key = std::tuple<long long,long long,long long>;
  static constexpr double tolerance_m() { return 1e-6; }
  std::map<Key,double> baselines;
  static Key key(const Eigen::Vector3d& p) {
    return {std::llround(p.x()*1e5),std::llround(p.y()*1e5),std::llround(p.z()*1e5)};
  }
  bool floor(const Eigen::Vector3d& p, double start, double desired,
             double execution, double* required, bool* recovering,
             bool owned_margin_recovery = false) {
    if (!required || !recovering || !p.allFinite() ||
        p.cwiseAbs().maxCoeff()>1e6 || !std::isfinite(start) ||
        !std::isfinite(desired) || !std::isfinite(execution) ||
        execution<0. || desired<=execution || start<=execution+tolerance_m()) return false;
    // Explicit owned-point recovery may consume the planning reserve even
    // when its initial state is just outside it. Full native certification
    // and the owner's progress/attempt ledger still govern execution.
    *recovering=owned_margin_recovery || start<=desired+1e-12;
    *required=desired;
    if (!*recovering) return true;
    const auto k=key(p);
    auto it=baselines.find(k);
    if (it==baselines.end()) {
      if (baselines.size()>=32) return false;
      it=baselines.emplace(k,start).first;
    }
    *required=execution;
    return start>*required+1e-12;
  }
  void observedFree(const Eigen::Vector3d& p) {
    if (p.allFinite() && p.cwiseAbs().maxCoeff()<=1e6) baselines.erase(key(p));
  }
};
}  // namespace care_confidence_map
