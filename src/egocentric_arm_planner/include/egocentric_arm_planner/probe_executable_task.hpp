#pragma once
#include "egocentric_arm_planner/executable_task_objective.hpp"
#include <cstdint>

namespace egocentric_arm_planner {
struct ProbeTaskSettings {
  Eigen::VectorXd acceleration;
  double prefix=.25,velocity_cap=.5,time_scale_max=5.,brake_dt=.05,hold=.1;
  int brake_steps=20;
};
struct ProbeTaskKnot { Eigen::VectorXd q,v; int64_t ns; };

// Receiver's legacy PROBE message: advertised velocity u[k], final zero.
// This is a read-only objective model. It authorizes no motion.
inline bool probeStoppedView(const Eigen::MatrixXd& q,const Eigen::MatrixXd& u,
    double dt,const Eigen::VectorXd& measured_v,const ProbeTaskSettings& s,
    Eigen::VectorXd* stopped,std::vector<ProbeTaskKnot>* trace=nullptr) {
  const int n=q.rows(),K=u.cols();
  if(!stopped || n<1 || K<1 || q.cols()!=K+1 || u.rows()!=n ||
      measured_v.size()!=n || s.acceleration.size()!=n || !q.allFinite() ||
      !u.allFinite() || !measured_v.allFinite() || !s.acceleration.allFinite() ||
      (s.acceleration.array()<=0.).any() || !(dt>0.) || !(s.prefix>0.) ||
      !(s.velocity_cap>0.) || !(s.time_scale_max>=1.) || !(s.brake_dt>0.) || s.brake_steps<1)return false;
  const double scale=std::min(s.time_scale_max,std::max(1.,u.cwiseAbs().maxCoeff()/s.velocity_cap));
  std::vector<int64_t> times(K+1);Eigen::MatrixXd velocities(n,K+1);
  for(int k=0;k<=K;++k) {
    // Match rospy.Duration(float) after scaling the original nanosecond knots.
    times[k]=static_cast<int64_t>((std::llround(k*dt*1e9)*1e-9*scale)*1e9);
    if(k<K)velocities.col(k)=u.col(k)/scale;else velocities.col(k).setZero();
    if(k && times[k]<=times[k-1])return false;
  }
  bool nodal=true;
  for(int k=0;k<K;++k) {
    const double h=(times[k+1]-times[k])*1e-9;
    if((q.col(k+1)-q.col(k)-.5*h*(velocities.col(k)+velocities.col(k+1))).cwiseAbs().maxCoeff()>1e-6 ||
       ((velocities.col(k+1)-velocities.col(k)).cwiseAbs()-s.acceleration*h).maxCoeff()>1e-6)nodal=false;
  }
  auto at=[&](int64_t ns,bool velocity)->Eigen::VectorXd {
    if(ns<=0)return velocity ? velocities.col(0).eval() : q.col(0).eval();
    if(ns>=times.back())return velocity ? velocities.col(K).eval() : q.col(K).eval();
    int k=0;while(times[k+1]<ns)++k;
    const double h=(times[k+1]-times[k])*1e-9,alpha=double(ns-times[k])/double(times[k+1]-times[k]);
    if(velocity)return ((1.-alpha)*velocities.col(k)+alpha*velocities.col(k+1)).eval();
    Eigen::VectorXd p=(1.-alpha)*q.col(k)+alpha*q.col(k+1);
    // _interpolate_point recognizes each segment independently at 1e-9.
    if((q.col(k+1)-q.col(k)-.5*h*(velocities.col(k)+velocities.col(k+1))).cwiseAbs().maxCoeff()<=1e-9)
      p+=.5*h*alpha*(1.-alpha)*(velocities.col(k)-velocities.col(k+1));
    return p;
  };
  const int64_t prefix=std::min(times.back(),std::max<int64_t>(1000,std::llround(s.prefix*scale*1e9)));
  const int64_t step=std::max<int64_t>(1,std::llround(s.brake_dt*1e9));
  Eigen::VectorXd position=q.col(0),v=measured_v;int64_t t=0;
  if(trace){trace->clear();trace->push_back({position,v,t});}
  auto append=[&](){if(trace)trace->push_back({position,v,t});};
  for(int boundary=1;t<prefix;++boundary) {
    const int64_t end=std::min(prefix,times[boundary]);
    while(t<end) {
      const int64_t next_t=t+std::min(step,end-t);const double h=(next_t-t)*1e-9;
      const Eigen::VectorXd desired=nodal ? at(next_t,true) : ((at(next_t,false)-at(t,false))/h).eval();
      const Eigen::VectorXd delta=desired-v;double fraction=1.;
      for(int j=0;j<n;++j)if(std::abs(delta[j])>1e-12)fraction=std::min(fraction,s.acceleration[j]*h/std::abs(delta[j]));
      const Eigen::VectorXd next=v+fraction*delta;
      position+=.5*h*(v+next);v=next;t=next_t;append();
    }
  }
  for(int b=0;b<s.brake_steps && v.cwiseAbs().maxCoeff()>1e-6;++b) {
    const double h=step*1e-9;Eigen::VectorXd next(n);
    for(int j=0;j<n;++j)next[j]=std::copysign(std::max(0.,std::abs(v[j])-s.acceleration[j]*h),v[j]);
    position+=.5*h*(v+next);v=next;t+=step;append();
  }
  if(v.cwiseAbs().maxCoeff()>1e-5)return false;
  if(s.hold>1e-9){t+=std::llround(s.hold*1e9);v.setZero();append();}
  *stopped=position;return position.allFinite();
}

// Differentiate the same executed prefix and brake model for either task residual.
inline bool probeStoppedPositionJacobian(const Eigen::MatrixXd& q, const Eigen::MatrixXd& u,
    double dt, const Eigen::VectorXd& measured_v, const ProbeTaskSettings& settings,
    Eigen::VectorXd* stopped, Eigen::MatrixXd* jacobian) {
  if (!jacobian || !probeStoppedView(q, u, dt, measured_v, settings, stopped)) return false;
  const int n = q.rows(), steps = u.cols(), position_columns = n * steps;
  constexpr double eps = 1e-5;
  jacobian->resize(n, 2 * position_columns);
  for (int col = 0; col < jacobian->cols(); ++col) {
    auto qp = q, qm = q, up = u, um = u;
    if (col < position_columns) {
      qp(col % n, col / n + 1) += eps;
      qm(col % n, col / n + 1) -= eps;
    } else {
      const int index = col - position_columns;
      up(index % n, index / n) += eps;
      um(index % n, index / n) -= eps;
    }
    Eigen::VectorXd plus, minus;
    if (!probeStoppedView(qp, up, dt, measured_v, settings, &plus) ||
        !probeStoppedView(qm, um, dt, measured_v, settings, &minus)) return false;
    jacobian->col(col) = (plus - minus) / (2. * eps);
  }
  return jacobian->allFinite();
}

inline bool probeTaskResidualJacobian(const PrimitiveObservationGuard& geometry,
    const Eigen::MatrixXd& q, const Eigen::MatrixXd& u, double dt,
    const Eigen::VectorXd& measured_v, const ProbeTaskSettings& settings,
    const std::string& frame, const Eigen::Isometry3d& goal,
    Eigen::Matrix<double, 6, 1>* error, Eigen::MatrixXd* jacobian) {
  Eigen::VectorXd stop;
  Eigen::MatrixXd stopped_jacobian;
  if (!error || !jacobian ||
      !probeStoppedPositionJacobian(q, u, dt, measured_v, settings, &stop, &stopped_jacobian) ||
      !taskPoseResidual(geometry, stop, frame, goal, error)) return false;
  constexpr double eps = 1e-5;
  Eigen::MatrixXd fk(6, q.rows());
  for (int j = 0; j < q.rows(); ++j) {
    auto plus = stop, minus = stop;
    plus[j] += eps;
    minus[j] -= eps;
    Eigen::Matrix<double, 6, 1> ep, em;
    if (!taskPoseResidual(geometry, plus, frame, goal, &ep) ||
        !taskPoseResidual(geometry, minus, frame, goal, &em)) return false;
    fk.col(j) = (ep - em) / (2. * eps);
  }
  *jacobian = fk * stopped_jacobian;
  return jacobian->allFinite();
}
} // namespace egocentric_arm_planner
