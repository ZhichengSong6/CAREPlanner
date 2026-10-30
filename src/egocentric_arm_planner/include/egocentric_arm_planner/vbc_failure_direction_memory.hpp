#pragma once
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace egocentric_arm_planner {
// Failed steering evidence only; never a certificate or an attempt budget.
// Target and points arrive on separate topics. Keep compatible history through
// that transition, but only consume it after an exact CURRENT candidate fails.
struct VbcFailureDirectionMemory {
  using Points = std::vector<Eigen::Vector3d>;
  using Cells = std::vector<std::array<float,3>>;
  struct Context {
    std::uint64_t mode = 0, reference = 0;
    long long owner = -1;
    std::string token;
    Eigen::VectorXd target, mask;
    Points points;
    bool enabled = false;
  };
  struct Candidate { Context context; std::uint64_t query = 0; };
  struct Match { Points repeated; std::uint64_t previous_raw = 0; };
  std::map<std::uint64_t,Candidate> pending;
  Context observed, history_context;
  Points history_points;
  Eigen::VectorXd anchor;
  std::uint64_t history_raw = 0, largest_raw = 0;
  double history_time = 0., last_measurement = 0.;
  static constexpr std::size_t max_pending = 32, max_points = 256;

  static Cells cells(const Points& points) {
    Cells result;
    if (points.empty() || points.size()>max_points) return result;
    for (const auto& p:points) {
      std::array<float,3> cell;
      for (int j=0;j<3;++j) {
        cell[j]=static_cast<float>(p[j]);
        if (!std::isfinite(cell[j])) return {};
      }
      result.push_back(cell);
    }
    std::sort(result.begin(),result.end());
    result.erase(std::unique(result.begin(),result.end()),result.end());
    return result;
  }
  static bool valid(const Context& c) {
    const auto prefix="care_obs_v1_"+std::to_string(c.owner)+"_";
    return c.enabled && c.owner>=0 && c.reference &&
        c.token.compare(0,prefix.size(),prefix)==0 && c.token.size()>prefix.size() &&
        c.target.size()==7 && c.mask.size()==7 && c.target.allFinite() && c.mask.allFinite() &&
        (c.mask.array()>=0.).all() && (c.mask.array()<=1.).all() && !cells(c.points).empty();
  }
  static bool sameDirection(const Context& a,const Context& b) {
    return valid(a) && valid(b) && a.mode==b.mode && a.reference==b.reference && a.owner==b.owner &&
        (a.target.array()==b.target.array()).all() && (a.mask.array()==b.mask.array()).all();
  }
  static bool contains(const Context& grown,const Context& old) {
    const auto a=cells(grown.points), b=cells(old.points);
    return !a.empty() && !b.empty() && std::includes(a.begin(),a.end(),b.begin(),b.end());
  }
  static bool exact(const Context& a,const Context& b) {
    return sameDirection(a,b) && a.token==b.token && cells(a.points)==cells(b.points);
  }
  void clearHistory() {
    history_raw=0; history_points.clear(); anchor.resize(0);
  }
  void reset() {
    clearHistory(); pending.clear(); observed=Context{}; last_measurement=0.;
    // Keep raw monotonicity across target churn and duplicate callbacks.
  }
  void observeContext(const Context& c) {
    if (!valid(c) || (valid(observed) &&
        (!sameDirection(c,observed) || !contains(c,observed)))) {
      clearHistory(); pending.clear();
    }
    observed=c;
  }
  void observeMeasured(const Eigen::VectorXd& q,double stamp) {
    if (q.size()!=7 || !q.allFinite() || !std::isfinite(stamp) || stamp<=0. ||
        (last_measurement && stamp<last_measurement)) {
      clearHistory(); last_measurement=0.; return;
    }
    if (last_measurement && stamp-last_measurement>.2) clearHistory();
    last_measurement=stamp;
    if (anchor.size()==7 && (q-anchor).lpNorm<Eigen::Infinity>()>.005) clearHistory();
  }
  bool remember(std::uint64_t raw,std::uint64_t query,const Context& c) {
    observeContext(c);
    if (!valid(c) || !query || query>raw || raw<=largest_raw) return false;
    largest_raw=raw;
    while (pending.size()>=max_pending) pending.erase(pending.begin());
    pending.emplace(raw,Candidate{c,query}); return true;
  }
  void accepted(std::uint64_t raw,const Context& c) {
    const auto it=pending.find(raw);
    if (it==pending.end()) return;
    if (exact(it->second.context,c)) clearHistory();
    pending.erase(it);
  }
  Match reject(std::uint64_t raw,std::uint64_t audited,std::uint64_t evidence_stamp,
               std::uint64_t query,const Context& c,const Points& points,
               const Eigen::VectorXd& measured,double now) {
    Match result;
    observeContext(c);
    const auto it=pending.find(raw);
    if (it==pending.end()) return result;
    const auto candidate=it->second; pending.erase(it);
    if (!exact(candidate.context,c) || query!=candidate.query || !audited ||
        evidence_stamp!=audited || audited<raw || !std::isfinite(now) ||
        now<audited*1e-9 || now-query*1e-9>.5 ||
        measured.size()!=7 || !measured.allFinite() || !last_measurement ||
        now<last_measurement || now-last_measurement>.2 || cells(points).empty()) return result;
    if (anchor.size()==7 && (measured-anchor).lpNorm<Eigen::Infinity>()>.005) clearHistory();
    const bool usable=history_raw && raw>history_raw && now>=history_time && now-history_time<=2. &&
        sameDirection(c,history_context) && contains(c,history_context);
    // Only a strict growth across token identities adds to legacy repeat logic.
    // Equal geometry + new identity, shrinkage, pose/mask change cannot inherit.
    if (usable && c.token!=history_context.token &&
        cells(c.points).size()>cells(history_context.points).size()) {
      for (const auto& p:points)
        if (std::any_of(history_points.begin(),history_points.end(),[&](const Eigen::Vector3d& old) {
              return (p-old).lpNorm<Eigen::Infinity>()<=1e-5;
            })) result.repeated.push_back(p);
      if (!result.repeated.empty()) result.previous_raw=history_raw;
    }
    if (!usable || (c.token!=history_context.token && cells(c.points)==cells(history_context.points)))
      anchor=measured;
    history_context=c; history_points=points; history_raw=raw; history_time=now;
    return result;
  }
};
}  // namespace egocentric_arm_planner
