#include <egocentric_arm_planner/ik_goal_cache.hpp>
#include <cassert>
#include <limits>
int main() {
  egocentric_arm_planner::IKGoalCache cache;
  Eigen::Isometry3d pose=Eigen::Isometry3d::Identity();
  pose.translation()=Eigen::Vector3d(.334,.218,.637);
  const Eigen::VectorXd q=Eigen::VectorXd::LinSpaced(7,-.3,.3);
  assert(!cache.matches("base_link",pose,7));
  cache.store("base_link",pose,q);
  assert(cache.matches("base_link",pose,7));
  assert(!cache.matches("world",pose,7));
  assert(!cache.matches("base_link",pose,6));
  auto changed=pose; changed.translation().x()+=.001;
  assert(!cache.matches("base_link",changed,7));
  changed=pose;changed.rotate(Eigen::AngleAxisd(.01,Eigen::Vector3d::UnitZ()));
  assert(!cache.matches("base_link",changed,7));
  Eigen::Quaterniond positive(pose.rotation()), negative;
  negative.coeffs()=-positive.coeffs();changed=pose;changed.linear()=negative.toRotationMatrix();
  assert(cache.matches("base_link",changed,7));
  Eigen::VectorXd invalid=q; invalid[0]=std::numeric_limits<double>::quiet_NaN();
  cache.store("world",changed,invalid);
  assert(cache.matches("base_link",pose,7));
  assert((cache.q-q).norm()==0.);
}
