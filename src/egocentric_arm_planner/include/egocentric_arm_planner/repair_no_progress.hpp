#pragma once
#include <Eigen/Core>
#include <map>
#include <string>

namespace egocentric_arm_planner {
// REPAIR liveness only. Never a feasibility certificate or execution permit.
// Keep old targets' counters until real measured motion; toggles, successful
// solves, repeated references and external replan pulses do not reset them.
struct RepairNoProgress {
  struct Ticket {
    std::string target;
    unsigned long long revision = 0, progress = 0;
  };
  enum class Failure { Stale, Retry, Exhausted };
  static constexpr int max_failures = 5;
  static constexpr std::size_t max_targets = 64;
  std::map<std::string, int> counts;
  Eigen::VectorXd anchor;
  Ticket active;
  void select(const std::string& target) {
    if (target != active.target) { active.target = target; ++active.revision; }
  }
  bool observe(const Eigen::VectorXd& q) {
    if (q.size() != 7 || !q.allFinite()) return false;
    if (anchor.size() != 7 || (q-anchor).lpNorm<Eigen::Infinity>() > 0.01) {
      anchor = q; counts.clear(); ++active.progress; return true;
    }
    return false;
  }
  bool targetCurrent(const Ticket& t) const {
    return t.target == active.target && t.revision == active.revision;
  }
  bool current(const Ticket& t) const {
    return targetCurrent(t) && t.progress == active.progress;
  }
  int failures() const {
    const auto it = counts.find(active.target);
    // No eviction: target churn cannot silently reopen exhausted targets.
    return it != counts.end() ? it->second :
        (counts.size() >= max_targets ? max_failures : 0);
  }
  bool blocked() const { return !active.target.empty() && failures() >= max_failures; }
  Failure fail(const Ticket& t) {
    if (active.target.empty() || !current(t)) return Failure::Stale;
    if (blocked()) return Failure::Exhausted;
    ++counts[active.target];
    return blocked() ? Failure::Exhausted : Failure::Retry;
  }
};
}  // namespace egocentric_arm_planner
