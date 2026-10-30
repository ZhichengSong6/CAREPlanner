#pragma once
#include "egocentric_arm_planner/primitive_observation_guard.hpp"
#include "egocentric_arm_planner/repair_executable_view.hpp"

namespace egocentric_arm_planner {
// Soft task objectives on the same nodal prefix + discrete brake used by the
// receiver. A useful proposal still needs the independent full safety audits.
struct ExecutableTaskTarget {
  RepairExecutableSample stopped;
  Eigen::Matrix<double,6,1> error;
  Eigen::MatrixXd jacobian;
};
inline bool taskPoseResidual(const PrimitiveObservationGuard& geometry,
    const Eigen::VectorXd& q,const std::string& frame,const Eigen::Isometry3d& goal,
    Eigen::Matrix<double,6,1>* error) {
  Eigen::Isometry3d pose;
  if(!error || !goal.matrix().allFinite() || !geometry.framePose(q,frame,&pose))return false;
  // Position/orientation ratio follows the original 2 cm / 0.2 rad task
  // tolerances. These are objective units, not acceptance thresholds.
  error->head<3>()=(pose.translation()-goal.translation())/.1;
  const Eigen::AngleAxisd rotation(goal.linear().transpose()*pose.linear());
  error->tail<3>()=rotation.axis()*rotation.angle();
  return error->allFinite();
}
inline bool executableTaskTargets(const PrimitiveObservationGuard& geometry,
    const Eigen::MatrixXd& q,const Eigen::MatrixXd& u,double dt,
    const RepairExecutableSettings& settings,const std::string& frame,
    const Eigen::Isometry3d& goal,std::vector<ExecutableTaskTarget>* targets,
    std::string* error=nullptr) {
  if(!targets)return false;targets->clear();
  std::vector<RepairExecutableSample> samples;
  if(!repairExecutableSamples(q,u,dt,settings,&samples,error))return false;
  for(double prefix:settings.prefixes) {
    const auto it=std::find_if(samples.rbegin(),samples.rend(),[&](const RepairExecutableSample& s){
      return std::abs(s.prefix_s-std::min(prefix,u.cols()*dt))<1e-9;});
    if(it==samples.rend())return false;
    ExecutableTaskTarget target;target.stopped=*it;
    if(!taskPoseResidual(geometry,it->q,frame,goal,&target.error))return false;
    target.jacobian.resize(6,q.rows());
    for(int j=0;j<q.rows();++j) {
      Eigen::VectorXd plus=it->q,minus=it->q;plus[j]+=1e-5;minus[j]-=1e-5;
      Eigen::Matrix<double,6,1> ep,em;
      if(!taskPoseResidual(geometry,plus,frame,goal,&ep) ||
         !taskPoseResidual(geometry,minus,frame,goal,&em))return false;
      target.jacobian.col(j)=(ep-em)/2e-5;
    }
    if(!target.jacobian.allFinite())return false;
    targets->push_back(std::move(target));
  }
  return true;
}
} // namespace egocentric_arm_planner
