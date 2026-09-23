#include "egocentric_arm_planner/measured_braking_seed.hpp"
#include "egocentric_arm_planner/qp_box_conflict.hpp"
#include <cassert>
#include <iostream>
int main() {
  Eigen::SparseMatrix<double> G(2,2);
  G.insert(0,0)=1.; G.insert(0,1)=-2.; G.insert(1,1)=1.;
  Eigen::VectorXd xl=Eigen::VectorXd::Constant(2,-1.), xu=-xl, lower(2);
  lower<<3.,1.;
  assert(egocentric_arm_planner::lowerBoxConflicts(G,lower,xl,xu,0)==0);
  lower[0]=3.001;
  assert(egocentric_arm_planner::lowerBoxConflicts(G,lower,xl,xu,0)==1);
  assert(egocentric_arm_planner::lowerBoxConflicts(G,lower,xl,xu,1)==0);
  Eigen::VectorXd q0 = Eigen::VectorXd::Constant(7,.2), v=Eigen::VectorXd::Zero(7), a=Eigen::VectorXd::Constant(7,3.);
  Eigen::MatrixXd q,u;
  assert(egocentric_arm_planner::measuredBrakingSeed(q0,v,a,20,.05,q,u));
  assert((q.col(20)-q0).norm()==0 && u.norm()==0);
  v[0]=.6; v[1]=-.6;
  assert(egocentric_arm_planner::measuredBrakingSeed(q0,v,a,20,.05,q,u));
  auto prev=v;
  for(int k=0;k<20;++k) {
    assert((q.col(k+1)-q.col(k)-.05*u.col(k)).norm()<1e-12);
    assert(((u.col(k)-prev).cwiseAbs().array()<=a.array()*.05+1e-12).all());
    prev=u.col(k);
  }
  assert(u.col(19).norm()==0);
  assert(std::abs(q(0,20)-.245)<1e-12 && std::abs(q(1,20)-.155)<1e-12);
  v[0]=NAN; assert(!egocentric_arm_planner::measuredBrakingSeed(q0,v,a,20,.05,q,u));
  std::cout<<"PASS measured seed: hold, braking, dynamics, sign, endpoint, nonfinite\n";
}
