#pragma once
#include "egocentric_arm_planner/repair_executable_view.hpp"

namespace egocentric_arm_planner {
// Original nodal motion and discrete braking, plus all prefix velocity-zero
// extrema. These are local SCP proposal rows; the receiver still checks the
// independently rebuilt continuous curve against its original joint limits.
inline bool normalExecutableLimitSamples(const Eigen::MatrixXd& q,
    const Eigen::MatrixXd& u,double dt,RepairExecutableSettings settings,
    const Eigen::VectorXd& initial_velocity,std::vector<RepairExecutableSample>* samples,
    std::string* error=nullptr) {
  settings.initial_velocity=initial_velocity;settings.prefixes={.25};
  if(!repairExecutableSamples(q,u,dt,settings,samples,error))return false;
  const double prefix=std::min(.25,u.cols()*dt);
  for(int k=0;k<u.cols() && k*dt<prefix-1e-9;++k) {
    const Eigen::VectorXd before=k ? u.col(k-1).eval() : initial_velocity;
    for(int j=0;j<q.rows();++j) {
      const double dv=u(j,k)-before[j];if(std::abs(dv)<1e-12)continue;
      const double s=-before[j]/dv;
      if(!(s>0. && s<std::min(1.,(prefix-k*dt)/dt)))continue;
      const double bend=.5*dt*s*(1.-s);
      RepairExecutableSample p;p.time_s=(k+s)*dt;p.prefix_s=prefix;
      p.q=(1.-s)*q.col(k)+s*q.col(k+1)+bend*(before-u.col(k));
      p.q_weights={{k,1.-s},{k+1,s}};
      if(k)p.u_weights.push_back({k-1,Eigen::VectorXd::Constant(q.rows(),bend)});
      p.u_weights.push_back({k,Eigen::VectorXd::Constant(q.rows(),-bend)});
      samples->push_back(std::move(p));
    }
  }
  samples->erase(std::remove_if(samples->begin(),samples->end(),
      [](const RepairExecutableSample& s){return s.time_s<=0.;}),samples->end());
  return true;
}
struct NormalExecutableLimitRow {
  RepairExecutableSample sample;
  bool lower = true;
  bool upper = true;
  bool prefix_position = false;
};

struct NormalLimitPadding { double lower, upper; };

// A fixed start may satisfy the physical margin while lying inside the extra
// numerical reserve. At an early extremum of ANOTHER joint, imposing the full
// reserve on every joint can be impossible even at maximum inward acceleration.
// Only drop that extra reserve when the original first-interval motion model
// proves it unreachable. Exact continuous checks still own physical acceptance.
inline NormalLimitPadding normalExecutableLimitPadding(
    const NormalExecutableLimitRow& row, int joint,
    const Eigen::VectorXd& q0, const Eigen::VectorXd& v0,
    const Eigen::VectorXd& speed, const Eigen::VectorXd& acceleration,
    const Eigen::VectorXd& lower, const Eigen::VectorXd& upper,
    double dt, double nominal) {
  NormalLimitPadding padding{nominal,nominal};
  const double t=row.sample.time_s;
  if(!row.prefix_position || !(t>0. && t<dt) || !(nominal>0.) ||
      joint<0 || joint>=q0.size() || q0.size()!=v0.size() ||
      q0.size()!=speed.size() || q0.size()!=acceleration.size() ||
      q0.size()!=lower.size() || q0.size()!=upper.size())return padding;
  const int j=joint;
  if(!std::isfinite(q0[j]) || !std::isfinite(v0[j]) ||
      !(acceleration[j]>0.) || !(speed[j]>0.) ||
      q0[j]<lower[j] || q0[j]>upper[j] || std::abs(v0[j])>speed[j])return padding;
  const double next_max=std::min(speed[j],v0[j]+acceleration[j]*dt);
  const double next_min=std::max(-speed[j],v0[j]-acceleration[j]*dt);
  const double best_upper=q0[j]+v0[j]*t+.5*(next_max-v0[j])/dt*t*t;
  const double best_lower=q0[j]+v0[j]*t+.5*(next_min-v0[j])/dt*t*t;
  if(row.lower && best_upper>=lower[j] && best_upper<lower[j]+nominal-1e-12)
    padding.lower=0.;
  if(row.upper && best_lower<=upper[j] && best_lower>upper[j]-nominal+1e-12)
    padding.upper=0.;
  return padding;
}

// A finite maximum of affine functions equals the original trapezoidal stop
// displacement for positive speed. Use both signs to constrain its full range.
// Prefix sample rows are affine positions at fixed times, including the current
// analytic velocity-zero times. No brake derivative or iterate velocity box is
// needed for these joint-limit rows. Final receiver extrema checks still apply.
inline bool normalExecutableLimitRows(const Eigen::MatrixXd& q,
    const Eigen::MatrixXd& u, double dt, RepairExecutableSettings settings,
    const Eigen::VectorXd& initial_velocity,
    std::vector<NormalExecutableLimitRow>* rows, std::string* error=nullptr) {
  if (!rows) { if(error)*error="Missing normal limit rows"; return false; }
  rows->clear();
  std::vector<RepairExecutableSample> samples;
  if (!normalExecutableLimitSamples(q,u,dt,settings,initial_velocity,&samples,error)) return false;
  const double prefix=std::min(.25,u.cols()*dt);
  RepairExecutableSample endpoint;
  bool found=false;
  for (const auto& sample:samples) if(!sample.brake) {
    rows->push_back({sample,true,true,true});
    if (std::abs(sample.time_s-prefix)<1e-9) { endpoint=sample; found=true; }
  }
  if(!found) { if(error)*error="Missing prefix endpoint";return false; }
  const int left=std::min<int>(u.cols()-1,std::floor(prefix/dt));
  const double alpha=std::min(1.,std::max(0.,prefix/dt-left));
  const Eigen::VectorXd before=left ? u.col(left-1).eval() : initial_velocity;
  const Eigen::VectorXd velocity=(1.-alpha)*before+alpha*u.col(left);
  for(int n=1;n<=settings.max_brake_steps;++n) {
    const double slope=settings.brake_dt*(n-.5);
    const Eigen::VectorXd intercept=.5*settings.brake_dt*settings.brake_dt*n*(n-1)*settings.acceleration;
    for(int sign:{-1,1}) {
      RepairExecutableSample sample=endpoint;
      sample.q+=slope*velocity+sign*intercept;
      if(left && 1.-alpha!=0.)sample.u_weights.push_back({left-1,Eigen::VectorXd::Constant(q.rows(),slope*(1.-alpha))});
      if(alpha!=0.)sample.u_weights.push_back({left,Eigen::VectorXd::Constant(q.rows(),slope*alpha)});
      // upper: q + slope*v - intercept <= hi
      // lower: q + slope*v + intercept >= lo
      rows->push_back({std::move(sample),sign>0,sign<0});
    }
  }
  return true;
}
} // namespace egocentric_arm_planner
