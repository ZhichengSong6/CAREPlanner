#pragma once
#include "egocentric_arm_planner/repair_no_progress.hpp"
#include <cstdint>

namespace egocentric_arm_planner {
// Separate liveness budget, never an execution/safety certificate.
struct FinalVbcNoProgress {
  struct Candidate {
    RepairNoProgress::Ticket ticket;
    unsigned long long mode_epoch;
    unsigned long long plan_sequence;
  };
  RepairNoProgress progress;
  std::map<std::uint64_t, Candidate> pending;
  static constexpr std::size_t max_pending = 128;
  bool remember(std::uint64_t stamp, unsigned long long mode, unsigned long long plan) {
    if (!stamp || pending.count(stamp)) return false;
    while (pending.size() >= max_pending) pending.erase(pending.begin());
    pending.emplace(stamp, Candidate{progress.active, mode, plan});
    return true;
  }
  RepairNoProgress::Failure reject(std::uint64_t stamp, unsigned long long mode) {
    auto it = pending.find(stamp);
    if (it == pending.end()) return RepairNoProgress::Failure::Stale;
    const auto candidate = it->second;
    pending.erase(it);  // duplicate outcome cannot count, even after target churn
    if (candidate.mode_epoch != mode) return RepairNoProgress::Failure::Stale;
    return progress.fail(candidate.ticket);
  }
};
}  // namespace egocentric_arm_planner
