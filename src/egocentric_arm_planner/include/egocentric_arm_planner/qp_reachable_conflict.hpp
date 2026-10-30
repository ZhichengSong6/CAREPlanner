#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>

namespace egocentric_arm_planner {

// An upper bound for one linearized CDF row at step k. The per-joint interval
// contains every state reachable from measured q0 under the velocity limit,
// joint limit, and the current SCP trust box. Coupled constraints can only
// shrink that interval, so an upper bound below the hard margin proves that
// this row alone is infeasible. This is never an executable relaxation.
inline double reachableCdfUpperBound(
    double d, const Eigen::VectorXd& g, const Eigen::VectorXd& qlin,
    const Eigen::VectorXd& qbar_k, const Eigen::VectorXd& q0,
    const Eigen::VectorXd& qmin, const Eigen::VectorXd& qmax,
    const Eigen::VectorXd& velocity, double position_margin,
    double trust_radius, double dt, int k,
    const Eigen::VectorXd* initial_velocity=nullptr,
    const Eigen::VectorXd* acceleration=nullptr) {
  const int n = g.size();
  if (k < 1 || !(dt > 0.) || !(trust_radius >= 0.) ||
      qlin.size() != n || qbar_k.size() != n || q0.size() != n ||
      qmin.size() != n || qmax.size() != n || velocity.size() != n ||
      !std::isfinite(d) || !std::isfinite(position_margin) ||
      !g.allFinite() || !qlin.allFinite() || !qbar_k.allFinite() ||
      !q0.allFinite() || !qmin.allFinite() || !qmax.allFinite() ||
      !velocity.allFinite()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double upper = d;
  for (int j = 0; j < n; ++j) {
    if (velocity[j] < 0.) return std::numeric_limits<double>::quiet_NaN();
    const double reach = static_cast<double>(k) * dt * velocity[j];
    double reachable_lo=q0[j]-reach,reachable_hi=q0[j]+reach;
    if(initial_velocity || acceleration) {
      if(!initial_velocity || !acceleration || initial_velocity->size()!=n ||
          acceleration->size()!=n || !initial_velocity->allFinite() || !acceleration->allFinite() ||
          (*acceleration)[j]<=0.)return std::numeric_limits<double>::quiet_NaN();
      const double t=k*dt,drift=(*initial_velocity)[j]*t,spread=.5*(*acceleration)[j]*t*t;
      reachable_lo=std::max(reachable_lo,q0[j]+drift-spread);
      reachable_hi=std::min(reachable_hi,q0[j]+drift+spread);
    }
    const double lo = std::max({qmin[j] + position_margin,qbar_k[j] - trust_radius,reachable_lo});
    const double hi = std::min({qmax[j] - position_margin,qbar_k[j] + trust_radius,reachable_hi});
    // An empty joint interval is a separate bound failure, not evidence that
    // this particular CDF witness is the cause.
    if (lo > hi) return std::numeric_limits<double>::quiet_NaN();
    upper += g[j] * ((g[j] >= 0. ? hi : lo) - qlin[j]);
  }
  return upper;
}

}  // namespace egocentric_arm_planner
