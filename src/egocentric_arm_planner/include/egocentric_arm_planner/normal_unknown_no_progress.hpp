#pragma once
#include "egocentric_arm_planner/normal_active_no_progress.hpp"
#include <map>
#include <string>

namespace egocentric_arm_planner {
// Scheduling evidence only. No safety verdict or permission to execute.
struct NormalUnknownNoProgress {
  struct Evidence {
    std::string context;
    unsigned long long query = 0;
    double raw_time_s = 0.;
  };
  NormalActiveNoProgress progress;
  std::map<unsigned long long, Evidence> pending, certified;
  std::string episode_context;
  double max_candidate_span = .005;
  double max_evidence_age_s = 2.;
  Evidence last;

  NormalUnknownNoProgress() {
    progress.min_duration_s = 2.;
    progress.min_execution_stamps = 3;
    progress.motion_epsilon = .005;
  }
  void reset() {
    pending.clear(); certified.clear(); episode_context.clear();
    progress.reset(); last = Evidence{};
  }
  void remember(unsigned long long raw, unsigned long long query,
                const std::string& context, double span, bool blocked_unknown) {
    if (!raw || !query || context.empty() || !blocked_unknown ||
        !std::isfinite(span) || span < 0. || span > max_candidate_span) return;
    while (pending.size() >= 32) pending.erase(pending.begin());
    pending.emplace(raw, Evidence{context, query, raw*1e-9});
  }
  void certify(unsigned long long raw, unsigned long long execution,
               const std::string& context) {
    const auto it = pending.find(raw);
    if (it == pending.end()) return;
    const auto e = it->second;
    pending.erase(it);  // duplicates cannot mint new execution identities
    if (!execution || e.context != context || execution < raw) return;
    while (certified.size() >= 32) certified.erase(certified.begin());
    certified.emplace(execution, e);
  }
  bool observe(double now, unsigned long long execution, const Eigen::VectorXd& q,
               const std::string& context, bool eligible) {
    const auto it = certified.find(execution);
    if (!eligible || context.empty() || it == certified.end() ||
        it->second.context != context || !std::isfinite(now) ||
        now < it->second.raw_time_s || now-it->second.raw_time_s > max_evidence_age_s) {
      progress.reset(); episode_context.clear(); return false;
    }
    if (episode_context != context) {
      progress.reset(); episode_context = context;
    }
    last = it->second;
    return progress.observe(now, execution, q, true);
  }
};
}  // namespace egocentric_arm_planner
