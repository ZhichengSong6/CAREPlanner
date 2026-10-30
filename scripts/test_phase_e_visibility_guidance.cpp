#include "egocentric_arm_planner/visibility_guidance.hpp"
#include "egocentric_arm_planner/safe_frontier_recovery.hpp"
#include <cassert>
#include <iostream>
using namespace egocentric_arm_planner;
int main() {
  VisibilityGuidanceState g;
  g.sequence=2; g.epoch=4; g.token="care_obs_v1_test"; g.region="care_region_v1_test";
  g.sensor=2; g.stamp=1.; g.q_eval=Eigen::VectorXd::Zero(7); g.target=g.q_eval;
  g.mask=Eigen::VectorXd::Zero(7); g.mask.head(3).setOnes();
  g.gradient=g.mask; const auto measured=g.q_eval;
  const auto accepts=[&](){return g.compatible("care_obs_v1_test","care_region_v1_test",
      2,g.mask,4,measured,1.1);};
  assert(accepts()); g.sensor=6; assert(!accepts()); g.sensor=2;
  g.token="care_obs_v1_old"; assert(!accepts()); g.token="care_obs_v1_test";
  g.epoch=5; assert(!accepts()); g.epoch=4;
  g.stamp=.7; assert(!accepts()); g.stamp=1.2; assert(!accepts()); g.stamp=1.;
  g.q_eval[0]=.03; assert(!accepts()); g.q_eval[0]=0.;
  g.gradient[6]=1.; assert(!accepts()); g.gradient[6]=0.;
  assert(accepts());
  const auto zero=Eigen::VectorXd::Zero(7).eval();
  const auto x=Eigen::VectorXd::Unit(7,0).eval(), y=Eigen::VectorXd::Unit(7,1).eval();
  const auto full=Eigen::VectorXd::Ones(7).eval();
  auto target=boundedObservationTarget(zero,-.05*x,{x},{0.},y,full,{},.03);
  assert(target.size()==7 && target[1]>.029 && std::abs(target[0])<1e-9);
  auto outward=boundedObservationTarget(zero,-.05*x,{x},{-.03},y,full,{},.03);
  assert(outward.size()==7 && outward[0]>.029);
  auto alternative=boundedObservationTarget(zero,-.05*x,{x},{0.},y,full,{y},.03);
  assert(alternative.size()==7 && alternative.normalized().dot(y)<.98);
  SafeFrontierRecovery r; r.select("one",4);r.advance();r.detour_direction=y;
  r.advance(); assert(r.failed_directions.size()==1 && r.attempts==2);
  r.select("two",4);r.select("one",4);
  assert(r.failed_directions.size()==1 && r.attempts==2);
  r.monitor_detour_observation=true;r.setObservationGoal("care_obs_v1_test",zero,full);
  r.noteClearanceFeedback("care_obs_v1_test","same",1.,1.,-.1);
  r.noteClearanceFeedback("care_obs_v1_old","same",1.1,1.1,1.);
  assert(!r.clearance_progress_pending);
  r.noteClearanceFeedback("care_obs_v1_test","same",1.2,1.2,-.09);
  assert(r.clearance_progress_pending && r.attempts==2 && r.segments==0);
  r.segments=5;r.detour_direction.resize(0);
  r.select("two",4);r.select("one",4);assert(r.segments==5);
  std::cout<<"PASS: guidance identity, age, epoch, anchor, mask, direction ranking and budget persistence\n";
}
