#pragma once
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace egocentric_arm_planner {
// Call only AFTER the complete response/request identity and live repair
// ticket have matched. A point-scoped VBC guard needs unanimous map evidence.
inline bool resolvedFreePoint(const std::vector<std::string>& status,
    const std::vector<double>& points, const Eigen::Vector3d& point) {
  if (points.size()!=3*status.size() || !point.allFinite()) return false;
  bool found = false;
  for (std::size_t i=0; i<status.size(); ++i) {
    if (status[i]!="resolved_free" && status[i]!="evaluated_unknown" &&
        status[i]!="evaluated_occupied") return false;
    const Eigen::Vector3d p(points[3*i],points[3*i+1],points[3*i+2]);
    if (!p.allFinite()) return false;
    if ((p-point).lpNorm<Eigen::Infinity>()<=1e-5) {
      if (status[i]!="resolved_free") return false;
      found = true;
    }
  }
  return found;
}
// Steering/liveness only. No method authorizes execution or lowers a margin.
struct SafeFrontierRecovery {
  // Initial sensor mount group plus four distinct replacement groups.
  static constexpr int max_attempts = 5;
  static constexpr int max_segments = 6;
  struct Candidate { int generation; double span; bool following_frontier; };
  struct Budget { int attempts, segments; bool active, final_stage_replacement_used; };
  std::string key;
  unsigned long long epoch = 0, last_execution = 0;
  int attempts = 0, generation = 0, tiny_commits = 0, rejected = 0;
  int segments = 0;
  bool active = false, following_frontier = false;
  bool final_stage_replacement_used = false;
  double window_start = -1.0;
  double last_observed_time = -1.0, last_raw_span = 0.0;
  double last_stall_duration = 0.0, last_stall_motion = 0.0;
  int last_stall_commits = 0;
  Eigen::VectorXd anchor, window_anchor, target;
  std::map<unsigned long long, Candidate> pending, certified;
  std::map<std::string, Budget> budgets;

  void invalidateMotion() {
    ++generation; target.resize(0); anchor.resize(0); clearWindow();
    last_observed_time = -1.;
    pending.clear(); certified.clear(); following_frontier = false;
  }

  void select(const std::string& next, unsigned long long mode) {
    if (next == key && mode == epoch) return;
    if (mode != epoch) budgets.clear();
    else if (!key.empty() && (budgets.count(key) || budgets.size()<64))
      budgets[key] = Budget{attempts, segments, active, final_stage_replacement_used};
    key = next; epoch = mode;
    attempts = segments = rejected = 0; active = false; following_frontier = false;
    final_stage_replacement_used = false;
    last_stall_duration = last_stall_motion = last_raw_span = 0.; last_stall_commits = 0;
    const auto saved = budgets.find(key);
    if (saved != budgets.end()) {
      attempts = saved->second.attempts; segments = saved->second.segments;
      active = saved->second.active;
      final_stage_replacement_used = saved->second.final_stage_replacement_used;
    } else if (budgets.size() >= 64) {
      active = true; attempts = max_attempts + 1; // bounded, no eviction/retry loop
    }
    invalidateMotion(); last_execution = 0;
  }
  bool blocked() const { return active && attempts > max_attempts; }
  int stage() const { return active ? attempts : 0; }
  // A sensor replacement spends the SAME recovery ledger, before generation.
  // Token changes, rejected proposals and timeouts never refund this charge.
  bool reserveReplacement() {
    if (!active || attempts < 1 || attempts > max_attempts ||
        segments >= max_segments) return false;
    // A fifth direction may already have been charged by measured-stall
    // recovery. It still gets one sensor-group experiment before hold.
    if (attempts == max_attempts) {
      if (final_stage_replacement_used) return false;
      final_stage_replacement_used = true;
    } else {
      ++attempts;
      final_stage_replacement_used = attempts == max_attempts;
    }
    rejected = 0;
    invalidateMotion();  // preserves attempts and the already used segments
    return true;
  }
  // A repeated final-VBC rejection can be the first event that activates the
  // recovery ledger.  Once active, reject() may already have advanced the
  // direction on the same rejection; do not charge that direction twice when
  // the replacement handshake arrives.
  bool reserveReplacementAfterVbc(bool direction_already_advanced) {
    if (segments >= max_segments || blocked()) return false;
    if (direction_already_advanced) {
      if (!active || attempts < 1 || attempts > max_attempts) return false;
      if (attempts == max_attempts) {
        if (final_stage_replacement_used) return false;
        final_stage_replacement_used = true;
      }
      return true;
    }
    if (!active) advance();  // the rejected original q_vis is direction one
    return reserveReplacement();
  }
  // Scheduling an already-live point owner before its parent does not create
  // a steering candidate. It may use the authenticated handshake without
  // changing this target's finite candidate/segment ledger.
  bool authorizeOwnerPromotion() const { return !blocked(); }
  void clearWindow() { tiny_commits = 0; window_start = -1.; window_anchor.resize(0); }
  // An authenticated abort invalidates this execution and its partial window.
  void abort(unsigned long long execution) {
    if (!certified.erase(execution)) return;
    last_execution = std::max(last_execution, execution);
    clearWindow();
  }
  void advance() {
    active = true; ++attempts; ++generation; rejected = 0; segments = 0;
    final_stage_replacement_used = false;
    following_frontier = false;
    target.resize(0); clearWindow(); pending.clear(); certified.clear();
  }
  void remember(unsigned long long raw, double span) {
    if (!raw || key.empty() || !std::isfinite(span) || span < 0.) return;
    while (pending.size() >= 64) pending.erase(pending.begin());
    pending.emplace(raw, Candidate{generation, span, following_frontier});
  }
  bool certify(unsigned long long raw, unsigned long long execution) {
    const auto it = pending.find(raw);
    if (!execution || it == pending.end() || it->second.generation != generation ||
        execution <= last_execution || certified.count(execution)) return false;
    while (certified.size() >= 64) certified.erase(certified.begin());
    certified.emplace(execution, it->second); pending.erase(it); return true;
  }
  bool reject(unsigned long long raw) {
    const auto it = pending.find(raw);
    if (it == pending.end()) return false;
    const bool current = it->second.generation == generation;
    pending.erase(it);
    if (current && active && !blocked() && ++rejected >= 2) { advance(); return true; }
    return false;
  }
  // Called only with fresh measured state and a matching, non-aborted tracker
  // execution that has actually advanced to >= 50 ms (or completed).
  bool observe(unsigned long long execution, double now, const Eigen::VectorXd& q) {
    if (q.size() != 7 || !q.allFinite() || !std::isfinite(now)) return false;
    const auto it = certified.find(execution);
    if (it == certified.end() || execution < last_execution ||
        it->second.generation != generation || now <= last_observed_time) return false;
    const Candidate c = it->second;
    const bool new_execution = execution != last_execution;
    last_execution = execution; last_observed_time = now; last_raw_span = c.span;
    certified.erase(certified.begin(), it); // retain current identity for later measured samples
    if (anchor.size() != 7) anchor = q;
    // Movement is a local waypoint transition, NEVER observation completion.
    // Continue the detour across commits, with a finite segment budget.
    if ((q-anchor).lpNorm<Eigen::Infinity>() >= .01) {
      anchor = q; clearWindow();
      if (!active || blocked()) return false;
      if (!c.following_frontier && ++segments >= max_segments) advance();
      else { ++generation; target.resize(0); pending.clear(); certified.clear(); }
      return true;
    }
    // The raw candidate may be much larger than the certified/executed prefix.
    // Only measured motion ends a stall window, including across handoffs.
    // Keep a separate anchor so small steps can accumulate into a .01 rad segment.
    if (window_start < 0.) { window_start = now; window_anchor = q; }
    if ((q-window_anchor).lpNorm<Eigen::Infinity>() > .001) {
      clearWindow(); return false;
    }
    if (new_execution) ++tiny_commits;
    if (tiny_commits >= 6 && now-window_start >= 1.0 && !blocked()) {
      last_stall_duration = now-window_start;
      last_stall_motion = (q-window_anchor).lpNorm<Eigen::Infinity>();
      last_stall_commits = tiny_commits;
      advance(); return true;
    }
    return false;
  }
};

// Tangential search around the currently tight CDF halfspaces. The candidate
// target is only an objective: all original QP constraints remain unchanged.
inline Eigen::VectorXd boundedTangentTarget(
    const Eigen::VectorXd& q0, const Eigen::VectorXd& desired,
    const std::vector<Eigen::VectorXd>& gradients, int attempt, double step) {
  if (q0.size()!=7 || desired.size()!=7 || !q0.allFinite() || !desired.allFinite() ||
      attempt<1 || attempt>SafeFrontierRecovery::max_attempts ||
      !std::isfinite(step) || step<=0. || gradients.empty()) return Eigen::VectorXd();
  Eigen::MatrixXd projector = Eigen::MatrixXd::Identity(7,7);
  std::vector<Eigen::VectorXd> basis;
  for (const auto& g : gradients) {
    if (g.size()!=7 || !g.allFinite() || g.norm()<1e-9) return Eigen::VectorXd();
    Eigen::VectorXd v = g.normalized();
    for (const auto& b : basis) v -= b.dot(v)*b;
    if (v.norm()>1e-7) basis.push_back(v.normalized());
  }
  for (const auto& b : basis) projector -= b*b.transpose();
  std::vector<Eigen::VectorXd> directions;
  for (int j=0;j<7;++j) {
    Eigen::VectorXd v = projector.col(j);
    if (v.norm()>.1) { directions.push_back(v); directions.push_back(-v); }
  }
  if (directions.empty()) return Eigen::VectorXd();
  Eigen::VectorXd v = directions[(attempt-1)%directions.size()];
  // If a feasible projected goal component exists, keep a small forward bias.
  const Eigen::VectorXd forward = projector*(desired-q0);
  if (forward.norm()>1e-6) v += .2*forward.normalized();
  if (v.lpNorm<Eigen::Infinity>()<1e-8) return Eigen::VectorXd();
  v *= std::min(.05,step)/v.lpNorm<Eigen::Infinity>();
  for (const auto& g : gradients)
    if (g.dot(v)<-1e-8) return Eigen::VectorXd();
  return q0+v;
}
}  // namespace egocentric_arm_planner
