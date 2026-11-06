#pragma once
#include "egocentric_arm_planner/repair_executable_view.hpp"
#include <cstdint>
namespace egocentric_arm_planner {
struct NormalCurveKnot { Eigen::VectorXd q,v; std::int64_t ns; };
struct NormalCurveRange {
  bool valid=false; double violation=0.,time_s=0.; int joint=-1;
};
// Exact coordinated ramp and clipped stop used by the Python receiver. The
// immutable advertised boundary and the actual construction boundary differ
// deliberately: a preview never substitutes for a fresh execution audit.
inline bool normalExecutableCurve(const Eigen::MatrixXd& q,const Eigen::MatrixXd& u,
    double dt,const RepairExecutableSettings& settings,const Eigen::VectorXd& advertised_v,
    const Eigen::VectorXd& start_q,const Eigen::VectorXd& start_v,
    std::vector<NormalCurveKnot>* knots,std::string* error=nullptr) {
  auto fail=[&](const char* s){if(error)*error=s;return false;};
  const int n=q.rows(),K=u.cols();
  if(!knots || n<1 || K<1 || q.cols()!=K+1 || u.rows()!=n ||
      advertised_v.size()!=n || start_q.size()!=n || start_v.size()!=n ||
      settings.acceleration.size()!=n || !q.allFinite() || !u.allFinite() ||
      !advertised_v.allFinite() || !start_q.allFinite() || !start_v.allFinite() ||
      !settings.acceleration.allFinite() || (settings.acceleration.array()<=0.).any() ||
      !(dt>0.) || !(settings.brake_dt>0.) || settings.max_brake_steps<1 || settings.max_brake_steps>20)
    return fail("normal_curve_inputs");
  auto before=[&](int k)->Eigen::VectorXd{return k?u.col(k-1).eval():advertised_v;};
  bool nodal=true;
  for(int k=0;k<K;++k)if((q.col(k+1)-q.col(k)-.5*dt*(before(k)+u.col(k))).cwiseAbs().maxCoeff()>1e-6 ||
      ((u.col(k)-before(k)).cwiseAbs()-settings.acceleration*dt).maxCoeff()>1e-6)nodal=false;
  auto at=[&](std::int64_t ns,bool velocity)->Eigen::VectorXd {
    const double x=std::min(K*dt,ns*1e-9)/dt;
    const int k=std::min(K-1,int(std::floor(x)));const double s=std::min(1.,std::max(0.,x-k));
    if(velocity)return ((1.-s)*before(k)+s*u.col(k)).eval();
    Eigen::VectorXd p=(1.-s)*q.col(k)+s*q.col(k+1);
    if((q.col(k+1)-q.col(k)-.5*dt*(before(k)+u.col(k))).cwiseAbs().maxCoeff()<=1e-9)
      p+=.5*dt*s*(1.-s)*(before(k)-u.col(k));
    return p;
  };
  knots->clear();knots->push_back({start_q,start_v,0});
  const std::int64_t end=std::llround(std::min(.25,K*dt)*1e9),hns=std::max<std::int64_t>(1,std::llround(settings.brake_dt*1e9));
  auto append=[&](const Eigen::VectorXd& next,std::int64_t ns){
    const auto& p=knots->back();const double h=(ns-p.ns)*1e-9;
    Eigen::VectorXd position=p.q+.5*h*(p.v+next);knots->push_back({position,next,ns});
  };
  for(int k=1;knots->back().ns<end;++k) {
    const std::int64_t boundary=std::min(end,std::int64_t(std::llround(k*dt*1e9)));
    while(knots->back().ns<boundary) {
      const auto ns=std::min(boundary,knots->back().ns+hns);const double h=(ns-knots->back().ns)*1e-9;
      const Eigen::VectorXd desired=nodal?at(ns,true):((at(ns,false)-at(knots->back().ns,false))/h).eval();
      const Eigen::VectorXd v=knots->back().v,delta=desired-v;double fraction=1.;
      for(int j=0;j<n;++j)if(std::abs(delta[j])>1e-12)fraction=std::min(fraction,settings.acceleration[j]*h/std::abs(delta[j]));
      append(v+fraction*delta,ns);
    }
  }
  for(int step=0;step<settings.max_brake_steps && knots->back().v.cwiseAbs().maxCoeff()>1e-6;++step) {
    Eigen::VectorXd next=knots->back().v;
    for(int j=0;j<n;++j){const double dv=settings.acceleration[j]*hns*1e-9;next[j]=next[j]>dv?next[j]-dv:next[j]<-dv?next[j]+dv:0.;}
    append(next,knots->back().ns+hns);
  }
  if(knots->back().v.cwiseAbs().maxCoeff()>1e-5)return fail("normal_curve_brake_budget");
  knots->push_back({knots->back().q,Eigen::VectorXd::Zero(n),knots->back().ns+100000000});
  return true;
}
inline NormalCurveRange normalCurveRange(const std::vector<NormalCurveKnot>& knots,
    const Eigen::VectorXd& lo,const Eigen::VectorXd& hi,const Eigen::VectorXd& speed) {
  NormalCurveRange out;if(knots.empty() || lo.size()!=hi.size() || speed.size()!=lo.size())return out;
  out.valid=true;
  auto check=[&](int j,double q,double t){const double d=std::max(lo[j]-q,q-hi[j]);if(d>0.)out.valid=false;if(d>out.violation){out.violation=d;out.joint=j;out.time_s=t;}};
  for(std::size_t k=0;k<knots.size();++k)for(int j=0;j<lo.size();++j) {
    const auto& p=knots[k];check(j,p.q[j],p.ns*1e-9);
    if(std::abs(p.v[j])>speed[j]+1e-9)out.valid=false;
    if(k){const auto& a=knots[k-1];const double h=(p.ns-a.ns)*1e-9,dv=p.v[j]-a.v[j];
      if(std::abs(dv)>1e-12){const double t=-a.v[j]*h/dv;if(t>0. && t<h)check(j,a.q[j]+a.v[j]*t+.5*dv/h*t*t,a.ns*1e-9+t);}}
  }
  return out;
}
}
