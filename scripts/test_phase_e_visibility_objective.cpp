#include "egocentric_arm_planner/visibility_objective_step.hpp"
#include <cassert>
#include <iostream>
#include <limits>
using egocentric_arm_planner::visibilityObjectiveStep;
int main() {
  // Simulate increasing GCDF/SCP latency: terminal must stay at k=20.
  for (double now : {10., 10.16, 10.51, 11.5, 100.})
    assert(visibilityObjectiveStep(true, 11., now, .05, 20) == 20);
  // Regression witness for the former fake-deadline conversion.
  assert(visibilityObjectiveStep(false, 11., 10.16, .05, 20) == 17);
  assert(visibilityObjectiveStep(false, 11., 11.5, .05, 20) == 1);
  // Genuine legacy deadlines retain ceil, expired=>first, beyond=>skip.
  assert(visibilityObjectiveStep(false, 10.125, 10., .05, 20) == 3);
  assert(visibilityObjectiveStep(false, 12., 10., .05, 20) == -1);
  assert(visibilityObjectiveStep(false, 9., 10., .05, 20) == 1);
  assert(visibilityObjectiveStep(false, std::numeric_limits<double>::quiet_NaN(), 10., .05, 20) == -1);
  assert(visibilityObjectiveStep(true, 11., 10., 0., 20) == -1);
  std::cout << "PASS terminal index invariant; legacy deadlines preserved\n";
}
