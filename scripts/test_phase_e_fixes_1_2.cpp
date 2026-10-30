#include "egocentric_arm_planner/task_reference_retry.hpp"
#include "egocentric_arm_planner/trajectory_dynamics.hpp"
#include <cassert>
#include <iostream>
#include <limits>
using namespace egocentric_arm_planner;

int main() {
  for (double ratio : {1.0, 1.001, 1.00127, 1.00155, 1.00184, 1.002001}) {
    arm_trajectory::JointTrajectory trajectory(7);
    auto q = Eigen::VectorXd::Zero(7).eval();
    auto v = Eigen::VectorXd::Constant(7, 0.7).eval();
    auto a = Eigen::VectorXd::Constant(7, ratio).eval();
    assert(trajectory.addPoint(0.0, q, v, a));
    assert(trajectory.addPoint(0.021, q, v, a));
    const auto result = checkTaskDynamics(trajectory, std::vector<double>(7,1),
                                        std::vector<double>(7,1),0.02);
    assert(result.valid && result.accepted() == (ratio <= 1.001));
  }
  assert(!checkTaskDynamics(arm_trajectory::JointTrajectory(7), {}, {},0.02).accepted());
  TaskDynamicsCheck invalid;
  invalid.valid = true; invalid.acceleration_ratio = std::numeric_limits<double>::quiet_NaN();
  assert(!invalid.accepted());
  std::cout << "PASS dynamics: historical rejection ratios, exact boundary, empty/NaN\n";

  TaskReferenceRetry retry;
  retry.start(0); auto id = retry.id;
  assert(retry.claim(0)); assert(!retry.claim(0.1));
  assert(retry.fail(id,0.1)); assert(!retry.due(0.5)); assert(retry.claim(0.6));
  assert(retry.fail(id,0.7)); assert(retry.claim(1.2));
  assert(!retry.fail(id,1.3)); assert(!retry.active && retry.attempts==3);
  assert(!retry.claim(4));
  std::cout << "PASS retry: delayed retries and bounded exhaustion\n";
  retry.start(10); id = retry.id; assert(retry.claim(10));
  assert(retry.expired(15)); assert(!retry.succeed(id,15)); assert(!retry.fail(id,15));
  std::cout << "PASS retry: deadline includes in-flight result\n";
  retry.start(20); id = retry.id; assert(retry.claim(20));
  retry.start(21); assert(retry.id > id);
  assert(!retry.fail(id,21.1)); assert(!retry.succeed(id,21.1));
  assert(retry.active && retry.attempts==0 && retry.claim(21.2));
  assert(retry.succeed(retry.id,21.3)); assert(!retry.due(22));
  std::cout << "PASS retry: superseded results discarded, success stops retries\n";

  TaskReferenceReceipt receipt;
  assert(receipt.reference(10));  // legacy publisher compatibility
  assert(receipt.status(1,20,false)); assert(receipt.pending);
  assert(!receipt.reference(19)); assert(receipt.reference(21));
  assert(receipt.status(1,20,false)); assert(!receipt.pending);
  assert(receipt.status(2,30,true)); assert(receipt.failed && !receipt.pending);
  assert(!receipt.reference(31)); assert(!receipt.status(1,20,false));
  assert(receipt.status(3,40,false)); assert(!receipt.failed);
  assert(receipt.reference(41));
  assert(receipt.status(4,50,true)); assert(!receipt.reference(61));
  assert(receipt.failed);  // do not plan until newer request status arrives
  assert(receipt.status(5,60,false)); assert(!receipt.failed && !receipt.pending);
  std::cout << "PASS receipt: stale reference/status, cross-topic ordering, exhausted hold, new request\n";
}
