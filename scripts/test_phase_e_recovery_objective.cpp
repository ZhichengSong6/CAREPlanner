#include <egocentric_arm_planner/geometric_restore_proposal.hpp>
#include <cassert>
#include <iostream>

int main() {
  using namespace egocentric_arm_planner;
  const Eigen::VectorXd zero=Eigen::VectorXd::Zero(7),lo=Eigen::VectorXd::Constant(7,-2.),hi=-lo;
  const Eigen::VectorXd speed=Eigen::VectorXd::Ones(7);
  const std::vector<Eigen::VectorXd> normal{Eigen::VectorXd::Unit(7,0)};
  const Eigen::VectorXd old_goal=.5*Eigen::VectorXd::Unit(7,1);
  GeometricRestoreObjective objective;
  objective.visibility_gradient=-Eigen::VectorXd::Unit(7,1);
  objective.observation_mask=Eigen::VectorXd::Unit(7,1);
  objective.point_floor_m={.02};objective.point_reserve_m={0.};
  Eigen::MatrixXd q,u,old_q,old_u;
  assert(geometricRestoreProposal(zero,old_goal,normal,{.05},{},lo,hi,speed,.01,20,.05,&old_q,&old_u,.25));
  const Eigen::VectorXd local=-.05*Eigen::VectorXd::Unit(7,1);
  assert(geometricRestoreProposal(zero,local,normal,{.05},{},lo,hi,speed,.01,20,.05,&q,&u,.25,&objective));
  assert(old_q(1,20)>0. && q(1,20)<0.);
  assert(objective.visibility_gradient.dot(q.col(20))>0.);
  assert((q.col(5)-q.col(20)).norm()==0. && u.rightCols(15).norm()==0.);
  for(int k=0;k<20;++k)assert((q.col(k+1)-q.col(k)-.05*u.col(k)).norm()<1e-12);

  // With no compatible guide, do not label the old goal vector visibility.
  objective.visibility_gradient.setZero();
  assert(geometricRestoreProposal(zero,old_goal,normal,{.05},{},lo,hi,speed,.01,20,.05,&q,&u,.25,&objective));
  assert(q(0,20)>0. && q(1,20)==0.);

  // An ordinary UNKNOWN point does not inherit the owned10mm proposal floor.
  assert(!geometricRestoreProposal(zero,local,normal,{.0124},{},lo,hi,speed,.01,20,.05,&q,&u,.25,&objective));
  objective.point_floor_m={.01};
  assert(geometricRestoreProposal(zero,local,normal,{.0124},{},lo,hi,speed,.01,20,.05,&q,&u,.25,&objective));
  assert(!geometricRestoreProposal(zero,local,normal,{.009},{},lo,hi,speed,.01,20,.05,&q,&u,.25,&objective));

  // A future reserve below its target at the start remains a soft objective.
  // Geometry can use a joint outside the observation mask to obtain it.
  objective.point_floor_m={.02};objective.point_reserve_m={.002};
  objective.visibility_gradient=Eigen::VectorXd::Unit(7,1);
  const std::vector<Eigen::VectorXd> wrist_normal{Eigen::VectorXd::Unit(7,6)};
  assert(geometricRestoreProposal(zero,old_goal,wrist_normal,{.0201},{},lo,hi,speed,.01,20,.05,&q,&u,.25,&objective));
  assert(q(6,20)>0. && objective.observation_mask[6]==0.);

  auto invalid=objective;invalid.visibility_gradient[6]=1.;
  assert(!geometricRestoreProposal(zero,old_goal,wrist_normal,{.05},{},lo,hi,speed,.01,20,.05,&q,&u,.25,&invalid));
  invalid=objective;invalid.point_floor_m.clear();
  assert(!geometricRestoreProposal(zero,old_goal,wrist_normal,{.05},{},lo,hi,speed,.01,20,.05,&q,&u,.25,&invalid));
  invalid=objective;invalid.point_reserve_m[0]=-1.;
  assert(!geometricRestoreProposal(zero,old_goal,wrist_normal,{.05},{},lo,hi,speed,.01,20,.05,&q,&u,.25,&invalid));
  std::cout<<"PASS guidance direction, geometry-only fallback, point roles, future reserve, full-arm escape and prefix dynamics\n";
}
