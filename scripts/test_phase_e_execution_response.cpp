#include "egocentric_arm_planner/execution_response_monitor.hpp"
#include "egocentric_arm_planner/safe_frontier_recovery.hpp"
#include <cassert>
#include <iostream>
using namespace egocentric_arm_planner;
int main() {
  const auto zero=Eigen::VectorXd::Zero(7).eval();
  auto velocity=zero;velocity[0]=.1;
  ExecutionResponseMonitor no_response;
  for(int i=0;i<=110;++i) no_response.observe(i*.01,zero,velocity,true);
  assert(no_response.blocked && no_response.faults==1);
  // Neither a new token/mode nor a zero request makes actuation responsive.
  no_response.observe(2.,zero,zero,false);assert(no_response.blocked);
  auto moved=zero;moved[0]=.002;
  assert(no_response.observe(2.1,moved,zero,false) && !no_response.blocked);
  ExecutionResponseMonitor duplicate;
  for(int i=0;i<=110;++i) {
    duplicate.observe(i*.01,zero,velocity,true);
    duplicate.observe(i*.01,zero,velocity,true);
  }
  assert(duplicate.blocked);
  const auto faults=duplicate.faults;
  assert(duplicate.observe(2.,zero,zero,false,true) && !duplicate.blocked);
  assert(duplicate.faults==faults); // explicit safety waiting is not a motion claim
  for(int i=0;i<=110;++i) duplicate.observe(3.+i*.01,zero,velocity,true,true);
  assert(!duplicate.blocked);
  for(int scenario=0;scenario<5;++scenario) {
    ExecutionResponseMonitor monitor;
    for(int i=0;i<=250;++i) {
      auto q=zero;auto u=velocity;
      if(scenario==0) q[0]=i*.001; // actual response
      if(scenario==1) u[0]=.001; // tiny constrained QP motion
      if(scenario==2) u[0]=(i%2?-.1:.1); // opposing requests
      if(scenario==3 && i==50) {monitor.observe(5.,q,u,true);continue;} // sample gap
      if(scenario==4) u=zero; // safe hold
      monitor.observe(i*.01,q,u,scenario!=3);
      assert(!monitor.blocked);
    }
  }
  SafeFrontierRecovery recovery;recovery.select("target",1);
  recovery.setObservationGoal("care_obs_v1_target",Eigen::VectorXd::Ones(7),Eigen::VectorXd::Ones(7));
  const int generation=recovery.generation;
  for(int i=1;i<=6;++i) {
    recovery.remember(i,.1);recovery.certify(i,i);
    recovery.clearWindow();recovery.clearObservationWindow();
  }
  assert(recovery.attempts==0 && recovery.generation==generation && recovery.certified.size()==6);
  recovery.observation_window_start=1.;recovery.observation_commits=4;
  recovery.pauseProgress(1.5);recovery.pauseProgress(1.7);recovery.resumeProgress(2.);
  assert(std::abs(recovery.observation_window_start-1.5)<1e-12);
  assert(recovery.observation_commits==4 && recovery.attempts==0);
  std::cout<<"PASS sustained nonresponse, response, tiny commands, reversal, stale samples, holds, preserved recovery identity/budget\n";
}
