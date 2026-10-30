#include "arm_model/ik_result.hpp"
#include "egocentric_arm_planner/qp_no_progress.hpp"
#include "egocentric_arm_planner/qp_backtrack_policy.hpp"
#include "egocentric_arm_planner/qp_snapshot.hpp"
#include <cassert>
#include <iostream>
#include <sstream>

int main() {
  using namespace arm_model;
  assert(classifyIK(.009, .19, .01, .2) == IKStatus::CONVERGED);
  assert(classifyIK(.02254155991957572, .42862482122823703, .01, .2) == IKStatus::APPROXIMATE);
  assert(classifyIK(.01, .2, .01, .2) == IKStatus::APPROXIMATE);
  assert(classifyIK(.05, 1., .01, .2) == IKStatus::FAILED);
  assert(classifyIK(NAN, 0, .01, .2) == IKStatus::FAILED);
  assert(classifyIK(0, INFINITY, .01, .2) == IKStatus::FAILED);
  assert(classifyIK(0, 0, 0, .2) == IKStatus::FAILED);
  egocentric_arm_planner::QPNoProgress state;
  Eigen::VectorXd q = Eigen::VectorXd::Zero(7);
  for (int i=0; i<4; ++i) assert(!state.fail(q));
  assert(state.fail(q));
  assert(state.fail(q)); assert(state.failures == 5);
  q[0] = .009; assert(!state.observe(q)); assert(state.blocked());
  q[0] = .011; assert(state.observe(q)); assert(!state.blocked());
  assert(state.failures == 0);
  state.fail(q); state.reset(); assert(state.failures == 0);
  egocentric_arm_planner::QPBacktrackPolicy backtrack;
  assert(backtrack.consume(true, true, 3, "max iterations reached"));
  assert(backtrack.attempts == 1 && backtrack.used());
  assert(!backtrack.consume(true, true, 3, "max iterations reached"));
  assert(!backtrack.consume(true, true, 3, "solved"));
  backtrack.reset();
  assert(!backtrack.consume(false, true, 3, "primal infeasible"));
  assert(!backtrack.consume(true, false, 3, "primal infeasible"));
  assert(!backtrack.consume(true, true, 0, "primal infeasible"));
  assert(backtrack.consume(true, true, 2, "primal infeasible"));
  Eigen::SparseMatrix<double> a(2,3); a.insert(1,2)=1.2345678901234567;
  std::ostringstream s;
  egocentric_arm_planner::jsonSparse(s,a);
  assert(s.str().find("1.2345678901234567") != std::string::npos);
  s.str(""); egocentric_arm_planner::jsonNumber(s,INFINITY); assert(s.str()=="\"inf\"");
  std::cout << "PASS IK classification/finite boundaries; QP bounded state/reset/backtrack; snapshot precision\n";
}
