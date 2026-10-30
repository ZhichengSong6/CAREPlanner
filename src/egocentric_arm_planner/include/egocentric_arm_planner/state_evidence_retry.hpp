#pragma once
#include <Eigen/Core>
#include <cmath>
namespace egocentric_arm_planner {
// A failed certificate gets one fresh-state retry after a short stable window.
// This observes state only; it never commands a stop or extends a deadline.
struct StateEvidenceRetryWait {
  Eigen::VectorXd anchor;
  double start=-1.,last=-1.;
  void reset(){anchor.resize(0);start=last=-1.;}
  bool observe(double stamp,const Eigen::VectorXd& q,const Eigen::VectorXd& v,
               const Eigen::VectorXd& actuator) {
    if(!std::isfinite(stamp) || q.size()!=7 || v.size()!=7 || actuator.size()!=7 ||
        !q.allFinite() || !v.allFinite() || !actuator.allFinite() ||
        v.cwiseAbs().maxCoeff()>.01 || actuator.cwiseAbs().maxCoeff()>.01 || stamp<=last) {
      reset();return false;
    }
    if(start<0. || (q-anchor).lpNorm<Eigen::Infinity>()>.001){start=stamp;anchor=q;}
    last=stamp;
    return stamp-start>=.1;
  }
};
}  // namespace egocentric_arm_planner
