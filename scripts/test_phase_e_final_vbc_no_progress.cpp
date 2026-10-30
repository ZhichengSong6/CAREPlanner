#include "egocentric_arm_planner/final_vbc_no_progress.hpp"
#include <cassert>
#include <iostream>
using namespace egocentric_arm_planner;
int main() {
  using F = RepairNoProgress::Failure;
  FinalVbcNoProgress g;
  Eigen::VectorXd q=Eigen::VectorXd::Zero(7);
  g.progress.observe(q); g.progress.select("A");
  assert(g.reject(42,1)==F::Stale);
  for (int i=1;i<=5;++i) {
    assert(g.remember(i,1,i));
    assert(!g.remember(i,1,i));
    assert(g.reject(i,1)==(i==5?F::Exhausted:F::Retry));
    assert(g.reject(i,1)==F::Stale);
    assert(g.progress.failures()==i);
  }
  assert(g.progress.blocked());
  g.progress.select("B"); assert(!g.progress.blocked());
  g.progress.select("A"); assert(g.progress.blocked());
  q[0]=.005; assert(!g.progress.observe(q)); assert(g.progress.blocked());
  g.remember(10,1,10);
  q[0]=.02; assert(g.progress.observe(q)); assert(!g.progress.blocked());
  assert(g.reject(10,1)==F::Stale);  // old measured episode
  g.remember(11,1,11); g.progress.select("B"); g.progress.select("A");
  assert(g.reject(11,1)==F::Stale);  // A -> B -> A is not the same revision
  g.remember(12,1,12); assert(g.reject(12,2)==F::Stale); // old mode epoch
  for(int i=20;i<200;++i) g.remember(i,2,i);
  assert(g.pending.size()==128); assert(g.reject(20,2)==F::Stale);
  assert(!g.remember(0,2,200));
  std::cout << "PASS final VBC budget, duplicate, unknown, target/mode/progress staleness, bounded memory\n";
}
