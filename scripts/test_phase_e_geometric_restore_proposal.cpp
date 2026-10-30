#include <egocentric_arm_planner/geometric_restore_proposal.hpp>
#include <cassert>
int main() {
  using namespace egocentric_arm_planner;
  const Eigen::VectorXd zero=Eigen::VectorXd::Zero(7),lo=Eigen::VectorXd::Constant(7,-2.),hi=-lo;
  Eigen::VectorXd goal=Eigen::VectorXd::Constant(7,.5);
  std::vector<Eigen::VectorXd> normals{Eigen::VectorXd::Unit(7,0)};
  Eigen::MatrixXd q,u;
  assert(geometricRestoreProposal(zero,goal,normals,{.0124},{},lo,hi,Eigen::VectorXd::Ones(7),.01,20,.05,&q,&u));
  assert(q.cols()==21 && u.cols()==20 && q.col(0).norm()==0.);
  assert(q(0,20)>0.); // deficient shell favors outward gain
  for(int k=0;k<20;++k)assert((q.col(k+1)-q.col(k)-.05*u.col(k)).norm()<1e-12);
  assert(q.cwiseAbs().maxCoeff()<=.03000001);
  assert(!geometricRestoreProposal(zero,goal,normals,{.009},{},lo,hi,Eigen::VectorXd::Ones(7),.01,20,.05,&q,&u));
  assert(!geometricRestoreProposal(zero,goal,{}, {},{},lo,hi,Eigen::VectorXd::Ones(7),.01,20,.05,&q,&u));
  assert(!geometricRestoreProposal(zero,goal,normals,{.03},{},lo,hi,zero,.01,20,.05,&q,&u));
  assert(geometricRestoreProposal(zero,goal,normals,{.03},{},lo,hi,Eigen::VectorXd::Ones(7),.01,20,.05,&q,&u,.25));
  assert((q.col(5)-q.col(20)).norm()==0. && q.col(5).norm()>0.);
  assert(u.rightCols(15).norm()==0.);
}
