#pragma once
#include <egocentric_arm_planner/safe_frontier_recovery.hpp>

namespace egocentric_arm_planner {
// Objective inputs only. Point roles and their future-only reserve targets
// must survive the failed-QP -> geometry handoff. Complete native proof and
// the existing execution gates still establish every applicable hard floor.
struct GeometricRestoreObjective {
  Eigen::VectorXd visibility_gradient = Eigen::VectorXd::Zero(7);
  Eigen::VectorXd observation_mask = Eigen::VectorXd::Ones(7);
  std::vector<double> point_floor_m, point_reserve_m;
};

// A short objective proposal, never a safety certificate or a hard-QP result.
// The receiving pipeline reconstructs the measured acceleration ramp and
// certifies its complete map, braking, connection and executable curve.
inline bool geometricRestoreProposal(const Eigen::VectorXd& q0,
    const Eigen::VectorXd& desired, const std::vector<Eigen::VectorXd>& normals,
    const std::vector<double>& distances, const std::vector<Eigen::VectorXd>& failed,
    const Eigen::VectorXd& lower, const Eigen::VectorXd& upper,
    const Eigen::VectorXd& velocity_limits, double position_margin,
    int intervals, double dt, Eigen::MatrixXd* q, Eigen::MatrixXd* u,
    double prefix_s = 0., const GeometricRestoreObjective* objective = nullptr) {
  if(!q || !u || intervals<1 || dt<=0. || normals.size()!=distances.size() ||
      q0.size()!=7 || lower.size()!=7 || upper.size()!=7 || velocity_limits.size()!=7 ||
      !q0.allFinite() || desired.size()!=7 || !desired.allFinite() ||
      !lower.allFinite() || !upper.allFinite() || !velocity_limits.allFinite() ||
      (velocity_limits.array()<=0.).any())return false;
  if(objective && (objective->visibility_gradient.size()!=7 ||
      objective->observation_mask.size()!=7 || !objective->visibility_gradient.allFinite() ||
      !objective->observation_mask.allFinite() ||
      (objective->observation_mask.array()<0.).any() ||
      (objective->observation_mask.array()>1.).any() || objective->observation_mask.sum()<=0. ||
      ((Eigen::VectorXd::Ones(7)-objective->observation_mask).cwiseProduct(
          objective->visibility_gradient)).norm()>1e-9 ||
      objective->point_floor_m.size()!=distances.size() ||
      objective->point_reserve_m.size()!=distances.size()))return false;
  std::vector<double> slack;
  std::vector<Eigen::VectorXd> steering_normals;
  for(std::size_t i=0;i<distances.size();++i) {
    const double d=distances[i];
    const double floor=objective ? objective->point_floor_m[i] : (d<=.020000000001 ? .01 : .02);
    const double reserve=objective ? objective->point_reserve_m[i] : 0.;
    if(!std::isfinite(d) || !std::isfinite(floor) || !std::isfinite(reserve) ||
        normals[i].size()!=7 || !normals[i].allFinite() || reserve<0.)return false;
    // A satisfied zero-Jacobian row cannot steer a first-order proposal.
    // Mounted ground rows legitimately have a zero/nonpenetration floor;
    // their nanometre slack must not be interpreted as an UNKNOWN10mm floor.
    // Reject a violated constant row. Full native ground/curve validation
    // remains responsible for every body, including these mounting contacts.
    if(normals[i].norm()<=1e-12) {
      if(d<floor-1e-12)return false;
      continue;
    }
    if(floor<.01 || d<=floor+1e-6)return false;
    // Reserve improves proposal quality; a deficient reserve at the fixed
    // measured start is not an additional hard constraint on that start.
    slack.push_back(d-floor-reserve);
    steering_normals.push_back(normals[i]);
  }
  const Eigen::VectorXd gradient=objective ? objective->visibility_gradient : Eigen::VectorXd(desired-q0);
  const Eigen::VectorXd mask=objective ? objective->observation_mask : Eigen::VectorXd::Ones(7);
  Eigen::VectorXd target=boundedObservationTarget(q0,desired,steering_normals,slack,
      gradient,mask,failed,.03);
  if(target.size()!=7)return false;
  const int motion_intervals=prefix_s>0. ? std::max(1,std::min(intervals,int(std::floor(prefix_s/dt+1e-8)))) : intervals;
  const double horizon=motion_intervals*dt;
  for(int j=0;j<7;++j) {
    if(q0[j]<lower[j]+position_margin || q0[j]>upper[j]-position_margin)return false;
    target[j]=std::max(lower[j]+position_margin,std::min(upper[j]-position_margin,target[j]));
    const double travel=horizon*velocity_limits[j];
    target[j]=q0[j]+std::max(-travel,std::min(travel,target[j]-q0[j]));
  }
  const Eigen::VectorXd delta=target-q0;
  if(delta.lpNorm<Eigen::Infinity>()<1e-5)return false;
  q->resize(7,intervals+1);u->resize(7,intervals);
  for(int k=0;k<=intervals;++k)q->col(k)=q0+(double(std::min(k,motion_intervals))/motion_intervals)*delta;
  for(int k=0;k<intervals;++k)u->col(k)=k<motion_intervals ? Eigen::VectorXd(delta/horizon) : Eigen::VectorXd::Zero(7);
  return q->allFinite() && u->allFinite();
}
}  // namespace egocentric_arm_planner
