#include <egocentric_arm_planner/state_evidence_retry.hpp>
#include <cassert>
int main() {
  egocentric_arm_planner::StateEvidenceRetryWait wait;
  const auto q=Eigen::VectorXd::Zero(7).eval();
  assert(!wait.observe(1.,q,q,q));
  assert(!wait.observe(1.05,q,q,q));
  assert(wait.observe(1.11,q,q,q));
  assert(!wait.observe(1.12,q,Eigen::VectorXd::Constant(7,.1),q));
  assert(!wait.observe(1.13,q,q,q));
  assert(!wait.observe(1.14,Eigen::VectorXd::Constant(7,.002),q,q));
  assert(!wait.observe(1.13,q,q,q)); // reordered sample invalidates window
  assert(!wait.observe(1.2,q,q,Eigen::VectorXd::Constant(7,.1)));
  assert(!wait.observe(1.3,q,q,q));
  assert(wait.observe(1.41,q,q,q));
}
