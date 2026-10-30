#include "egocentric_arm_planner/repair_no_progress.hpp"
#include <cassert>
#include <iostream>
#include <limits>
using egocentric_arm_planner::RepairNoProgress;
int main() {
  RepairNoProgress p;
  Eigen::VectorXd q = Eigen::VectorXd::Zero(7);
  assert(p.observe(q)); p.select("target_a");
  auto first = p.active;
  for (int i=1; i<=5; ++i) {
    assert(p.fail(p.active) == (i==5 ? RepairNoProgress::Failure::Exhausted : RepairNoProgress::Failure::Retry));
    assert(p.failures()==i);
    p.select("target_a"); assert(!p.observe(q));
  }
  assert(p.blocked()); assert(p.fail(first)==RepairNoProgress::Failure::Exhausted);
  p.select("target_b"); assert(!p.blocked());
  assert(p.fail(first)==RepairNoProgress::Failure::Stale);
  assert(p.failures()==0); p.select("target_a"); assert(p.blocked());
  assert(!p.targetCurrent(first));  // A -> B -> A does not resurrect old work
  auto exhausted=p.active;
  q[0]=std::numeric_limits<double>::quiet_NaN(); assert(!p.observe(q)); assert(p.blocked());
  q[0]=.005; assert(!p.observe(q)); assert(p.blocked());
  q[0]=.011; assert(p.observe(q)); assert(!p.blocked());
  assert(p.targetCurrent(exhausted) && !p.current(exhausted));
  assert(p.fail(exhausted)==RepairNoProgress::Failure::Stale); assert(p.failures()==0);
  for (int i=0; i<64; ++i) { p.select(std::to_string(i)); p.fail(p.active); }
  p.select("overflow"); assert(p.blocked()); assert(p.counts.size()==64);
  p.select("0"); assert(p.failures()==1);  // no eviction of old budgets
  std::cout << "PASS REPAIR target budgets, stale tickets, measured progress, bounded memory\n";
}
