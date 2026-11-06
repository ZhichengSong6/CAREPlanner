#pragma once
#include <utility>

namespace egocentric_arm_planner {

// The live worker and an immutable preview use the same bounded local choice.
// Neither this policy nor a preview may publish motion or update SCP history.
struct ObservationDecisionPolicy {
  bool repair = false;
  bool probe = false;
  bool observation_phase = false;
  bool recheck_only = false;
  bool free_joint_task_enabled = false;
  bool frontier_active = false;
  int iteration = 0;
  int max_iterations = 0;
  int recovery_attempt = 0;
  int schedule_size = 0;
  bool mask_has_free_joint = false;

  bool canTryFreeJoints() const {
    return repair && !probe && observation_phase &&
        iteration + 2 <= max_iterations && !recheck_only &&
        free_joint_task_enabled && frontier_active && recovery_attempt == 0 &&
        schedule_size == 1 && mask_has_free_joint;
  }
};

template<class Result> struct ObservationDecisionResult {
  Result result;
  bool free_joint_trial_ran = false;
  bool free_joint_trial_used = false;
};

template<class Solve, class Improves>
auto observationLocalDecision(const ObservationDecisionPolicy& policy,
                              Solve solve, Improves improves)
    -> ObservationDecisionResult<decltype(solve(false))> {
  ObservationDecisionResult<decltype(solve(false))> decision;
  decision.result = solve(false);
  if (decision.result.solved && decision.result.primitive_executable_valid &&
      !decision.result.primitive_live_rebase_required && policy.canTryFreeJoints()) {
    decision.free_joint_trial_ran = true;
    auto trial = solve(true);
    const double total_ms = decision.result.setup_and_solve_ms + trial.setup_and_solve_ms;
    if (improves(decision.result, trial)) {
      decision.result = std::move(trial);
      decision.free_joint_trial_used = true;
    }
    decision.result.setup_and_solve_ms = total_ms;
  }
  return decision;
}

}  // namespace egocentric_arm_planner
