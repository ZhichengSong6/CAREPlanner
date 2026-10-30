#pragma once
#include <Eigen/Core>
#include <cmath>

namespace egocentric_arm_planner {
inline bool fixedStartUnknownBlocked(double distance,double floor,bool ordinary) {
  return ordinary && std::isfinite(distance) && std::isfinite(floor) && distance<floor-1e-12;
}
// Retry suppression, not motion authority. Target/head/token changes cannot
// change a fixed body's clearance. Fresh changed geometry/evidence may requery.
struct FixedStartClearanceHold {
  bool armed=false;
  Eigen::VectorXd anchor;
  Eigen::Vector3d point=Eigen::Vector3d::Zero();
  double recorded=-1.,distance=0.,floor=0.;
  void record(const Eigen::VectorXd& q,const Eigen::Vector3d& p,double d,double f,double now) {
    armed=true;anchor=q;point=p;distance=d;floor=f;recorded=now;
  }
  bool observe(double stamp,double now,const Eigen::VectorXd& q) {
    if(!armed || !std::isfinite(stamp) || stamp<=recorded || now<stamp || now-stamp>.05 ||
        q.size()!=anchor.size() || !q.allFinite() || (q-anchor).lpNorm<Eigen::Infinity>()<=.001)return false;
    armed=false;return true;
  }
  bool evidenceRecheck(double stamp,double now) {
    if(!armed || !std::isfinite(stamp) || stamp<=recorded || now<stamp || now-stamp>.5)return false;
    armed=false;return true;
  }
  void resolvedFree(const Eigen::Vector3d& p) {
    if(armed && (point-p).lpNorm<Eigen::Infinity>()<=1e-5)armed=false;
  }
};
}  // namespace egocentric_arm_planner
