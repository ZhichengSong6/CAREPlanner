#pragma once

#include <string>

namespace egocentric_arm_planner {

// Suppress repeated solves only for the exact failed observation. A changed
// observation requires a fresh hard QP, while recovery budgets remain separate.
struct RepairCandidateFailure {
  std::string target_key;
  std::string observation_token;
  unsigned long long mode_epoch = 0;
  unsigned long long measured_progress = 0;

  void record(const std::string& target, const std::string& token,
              unsigned long long epoch, unsigned long long progress) {
    target_key = target;
    observation_token = token;
    mode_epoch = epoch;
    measured_progress = progress;
  }

  bool blocks(const std::string& target, const std::string& token,
              unsigned long long epoch, unsigned long long progress) const {
    return !target_key.empty() && target_key == target &&
        observation_token == token && mode_epoch == epoch &&
        measured_progress == progress;
  }
};

}  // namespace egocentric_arm_planner
