#include "egocentric_arm_planner/normal_active_no_progress.hpp"
#include <cassert>
#include <iostream>

using egocentric_arm_planner::NormalActiveNoProgress;

int main() {
  NormalActiveNoProgress guard;
  Eigen::VectorXd q = Eigen::VectorXd::Zero(7);
  assert(!guard.observe(0.0, 1, q, true));
  for (int i = 2; i <= 5; ++i)
    assert(!guard.observe(0.5 * i, i, q, true));
  assert(!guard.observe(3.186, 5, q, true)); // successful CASE011 short hold
  assert(guard.observe(8.1, 6, q, true) == false); // stale sample gap resets
  for (int i = 1; i <= 8; ++i)
    assert(!guard.observe(8.1 + 0.9 * i, 6 + i, q, true));
  assert(guard.observe(16.2, 15, q, true));
  assert(!guard.observe(16.3, 15, q, true)); // one event per episode
  q[0] = 0.011;
  assert(!guard.observe(16.4, 16, q, true)); // measured motion resets
  assert(guard.execution_stamps == 1);
  assert(!guard.observe(16.5, 17, q, false)); // target reached resets
  assert(guard.execution_stamps == 0);
  std::cout << "normal active no progress PASS\n";
}
