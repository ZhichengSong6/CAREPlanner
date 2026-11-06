#pragma once
#include <Eigen/Core>
#include "egocentric_arm_planner/observation_action_progress.hpp"
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
  struct Budget { int attempts, segments; bool active, final_stage_replacement_used;
    std::vector<Eigen::VectorXd> failed_directions; std::string direction_token;
    int unapplied_replacements = 0; };
  std::string key;
  unsigned long long epoch = 0, last_execution = 0;
  int attempts = 0, generation = 0, tiny_commits = 0, rejected = 0;
  int segments = 0;
  int unapplied_replacements = 0;
  struct ReplacementCharge {
    std::string key; unsigned long long epoch = 0;
    int before = 0, after = 0, generation = 0, segments = 0;
    bool final_before = false, final_after = false, valid = false;
  };
  ReplacementCharge replacementCharge() const {
    ReplacementCharge c; c.key=key; c.epoch=epoch; c.before=attempts;
    c.final_before=final_stage_replacement_used; return c;
  }
  void sealReplacementCharge(ReplacementCharge& c) const {
    c.after=attempts; c.final_after=final_stage_replacement_used;
    c.generation=generation; c.segments=segments; c.valid=true;
  }
  bool canRetryUnappliedReplacement() const { return unapplied_replacements < 3; }
  bool refundUnappliedReplacement(ReplacementCharge& c) {
    if(!c.valid || c.key!=key || c.epoch!=epoch || c.after!=attempts ||
        c.final_after!=final_stage_replacement_used || c.generation!=generation ||
        c.segments!=segments)return false;
    // Retain a genuine earlier direction failure (including initial activation).
    // Only the unused reservation is refunded; motion/segment evidence is kept.
    attempts=std::max(1,c.before); final_stage_replacement_used=c.final_before;
    ++unapplied_replacements; c.valid=false; return true;
  }
  bool active = false, following_frontier = false;
  bool final_stage_replacement_used = false;
  double window_start = -1.0;
  double last_observed_time = -1.0, last_raw_span = 0.0;
  double last_stall_duration = 0.0, last_stall_motion = 0.0;
  int last_stall_commits = 0;
  std::string last_stall_reason = "none", observation_token;
  Eigen::VectorXd observation_goal, observation_mask;
  double observation_window_start = -1., observation_best_gap = -1.;
  int observation_commits = 0;
  double feedback_query_time = -1.;
  Eigen::Vector3d feedback_best = Eigen::Vector3d::Constant(-1.);
  bool feedback_progress_pending = false;
  bool monitor_detour_observation = false;
  Eigen::VectorXd detour_direction;
  std::vector<Eigen::VectorXd> failed_directions;
  std::string failed_direction_token;
  std::string clearance_token, clearance_signature;
  double clearance_query_time=-1., clearance_best=-1.;
  bool clearance_progress_pending=false;
  // Preserve this independent view-progress window across local detour
  // generations. Small pose/clearance gains cannot indefinitely hide a view
  // that is still outside the FOV. Same original six-commit/one-second gate.
  std::string geometry_signature;
  double geometry_query_time=-1., geometry_best=0., view_window_start=-1.;
  bool geometry_progress_pending=false;
  int view_window_commits=0;
  std::string last_advance_reason="none";
  bool observation_replacement_due=false;
  bool persistent_observation_monitor=false;
  bool action_progress_enabled=false;
  ObservationActionProgress action_progress;
  std::map<std::string,ObservationActionProgress> action_budgets;
  bool view_replacement_waiting=false, view_trial_pending=false;
  std::string view_replacement_key;
  unsigned long long view_replacement_epoch=0;
  int executed_view_trials=0;
  // Only an authenticated sensor/view replacement can arm a new trial.
  // Publishing another token or suspending the same view cannot renew it.
  void beginViewReplacement() {
    action_progress.suspend();
    view_replacement_waiting=true;view_trial_pending=false;
    view_replacement_key=key;view_replacement_epoch=epoch;
  }
  void finishViewReplacement(bool applied) {
    if(!view_replacement_waiting)return;
    action_progress.suspend();
    view_trial_pending=applied && view_replacement_waiting &&
        view_replacement_key==key && view_replacement_epoch==epoch;
    view_replacement_waiting=false;
  }
  Eigen::VectorXd geometry_measured_q;
  double geometry_latest_margin=0.;
  std::string observation_region;
  double unseen_window_start=-1.,unseen_best_fraction=-1.,unseen_best_gap=-1.;
  int unseen_window_commits=0;
  void setObservationRegion(const std::string& region) {
    if(region==observation_region)return;
    observation_region=region;unseen_window_start=-1.;unseen_best_fraction=-1.;unseen_best_gap=-1.;unseen_window_commits=0;
  }
  double progress_paused_at = -1.;
  Eigen::VectorXd anchor, window_anchor, target;
  std::map<unsigned long long, Candidate> pending, certified;
  std::map<std::string, Budget> budgets;

  void pauseProgress(double now) {
    if (!std::isfinite(now)) return;
    if (progress_paused_at>=0. && now>progress_paused_at) {
      const double dt=now-progress_paused_at;
      if (window_start>=0.) window_start+=dt;
      if (observation_window_start>=0.) observation_window_start+=dt;
      if (view_window_start>=0.) view_window_start+=dt;
      if (unseen_window_start>=0.) unseen_window_start+=dt;
    }
    if(action_progress_enabled)action_progress.suspend();
    progress_paused_at=now;
  }
  void resumeProgress(double now) {
    if (progress_paused_at<0.) return;
    pauseProgress(now); progress_paused_at=-1.;
  }

  void clearObservationWindow(bool clear_view=true) {
    observation_window_start = -1.; observation_best_gap = -1.;
    observation_commits = 0; feedback_progress_pending = false;
    if (clear_view) {view_window_start=-1.;view_window_commits=0;geometry_progress_pending=false;}
  }
  void setObservationGoal(const std::string& token, const Eigen::VectorXd& q,
                          const Eigen::VectorXd& mask) {
    const bool valid = token.compare(0,12,"care_obs_v1_")==0 && q.size()==7 &&
        mask.size()==7 && q.allFinite() && mask.allFinite() &&
        (mask.array()>=0.).all() && (mask.array()<=1.).all() && mask.sum()>0.;
    if (!valid) {
      observation_token.clear(); observation_goal.resize(0); observation_mask.resize(0);
      geometry_signature.clear();geometry_query_time=-1.;
      clearObservationWindow(); feedback_query_time=-1.; feedback_best.setConstant(-1.); return;
    }
    if (token==observation_token && observation_goal.size()==7 &&
        (q-observation_goal).norm()==0. && (mask-observation_mask).norm()==0.) return;
    if(token!=failed_direction_token) {
      failed_directions.clear();detour_direction.resize(0);failed_direction_token=token;
    }
    unseen_best_gap=-1.; // changed steering pose changes the distance metric, not the region budget
    observation_token=token; observation_goal=q; observation_mask=mask;
    geometry_signature.clear();geometry_query_time=-1.;geometry_progress_pending=false;
    clearance_token.clear(); clearance_query_time=-1.; clearance_progress_pending=false;
    clearObservationWindow(); feedback_query_time=-1.; feedback_best.setConstant(-1.);
  }
  // Actual measured-start collision-query progress is a scheduling hint only.
  // It never completes observations, refunds attempts, or extends segment caps.
  void noteClearanceFeedback(const std::string& token, const std::string& signature,
                            double query_time, double now, double clearance) {
    if (token!=observation_token || signature.empty() || !std::isfinite(clearance) ||
        !std::isfinite(query_time) || !std::isfinite(now) || query_time<=clearance_query_time ||
        now-query_time<0. || now-query_time>.5) return;
    if (token!=clearance_token || signature!=clearance_signature) {
      clearance_token=token; clearance_signature=signature; clearance_best=clearance;
    } else if (clearance-clearance_best>=.001) {
      clearance_progress_pending=true; clearance_best=clearance;
    }
    clearance_query_time=query_time;
  }
  // Real map-query feedback may extend a progress window, never complete an
  // obligation or grant motion. Strict token/time checks also reject latched data.
  bool noteObservationFeedback(const std::string& token, double query_time,
                               double now, const Eigen::Vector3d& metrics) {
    if (token!=observation_token || token.empty() || !std::isfinite(query_time) ||
        !std::isfinite(now) || query_time<=feedback_query_time || now-query_time<0. ||
        now-query_time>.5 || !metrics.allFinite() ||
        (metrics.array()<0.).any() || (metrics.array()>1.).any()) return false;
    const bool initialized = feedback_query_time>=0.;
    if (initialized && (metrics-feedback_best).maxCoeff()>=.001)
      feedback_progress_pending=true;
    feedback_best=feedback_best.cwiseMax(metrics); feedback_query_time=query_time;
    return true;
  }
  bool noteGeometryFeedback(const std::string& token,const std::string& signature,
                            double query_time,double now,double margin,
                            const Eigen::VectorXd& measured_q=Eigen::VectorXd()) {
    if(token!=observation_token || token.empty() || signature.size()!=64 ||
        signature.find_first_not_of("0123456789abcdef")!=std::string::npos ||
        !std::isfinite(query_time) || !std::isfinite(now) || !std::isfinite(margin) ||
        query_time<=geometry_query_time || now-query_time<0. || now-query_time>.2)return false;
    if(signature!=geometry_signature) {
      geometry_signature=signature;geometry_best=margin;
      view_window_start=-1.;view_window_commits=0;geometry_progress_pending=false;
    } else if(margin-geometry_best>=.001) {
      geometry_best=margin;geometry_progress_pending=true;
    }
    geometry_measured_q=measured_q;geometry_latest_margin=margin;
    geometry_query_time=query_time;return true;
  }
  bool observeGoalProgress(bool new_execution, double now, const Eigen::VectorXd& q) {
    const double gap=((q-observation_goal).cwiseProduct(observation_mask)).norm();
    if (observation_window_start<0.) {
      observation_window_start=now; observation_best_gap=gap;
    }
    const bool map_progress=feedback_progress_pending && now-feedback_query_time>=0. &&
        now-feedback_query_time<=.5;
    const bool sensed_progress=map_progress || (clearance_progress_pending &&
        now-clearance_query_time>=0. && now-clearance_query_time<=.5);
    feedback_progress_pending=false;
    clearance_progress_pending=false;
    const bool fresh_geometry=geometry_query_time>=0. && now-geometry_query_time>=0. &&
        now-geometry_query_time<=.5;
    if(active && monitor_detour_observation && fresh_geometry) {
      if(view_window_start<0. || map_progress || geometry_progress_pending) {
        view_window_start=now;view_window_commits=0;
      }
      geometry_progress_pending=false;
      if(new_execution)++view_window_commits;
      if(view_window_commits>=6 && now-view_window_start>=1. && !blocked()) {
        last_stall_duration=now-view_window_start;last_stall_motion=0.;
        last_stall_commits=view_window_commits;last_stall_reason="observation_no_progress";
        view_window_start=now;view_window_commits=0;
        advanceObservation(true,"view_geometry_stall");return true;
      }
    }
    if (observation_best_gap-gap>=.001 || sensed_progress) {
      observation_window_start=now; observation_commits=0;
      observation_best_gap=std::min(observation_best_gap,gap);
    }
    if (new_execution) ++observation_commits;
    if (observation_commits<6 || now-observation_window_start<1.0 || blocked()) return false;
    last_stall_duration=now-observation_window_start;
    last_stall_motion=std::max(0.,observation_best_gap-gap);
    last_stall_commits=observation_commits; last_stall_reason="observation_no_progress";
    advanceObservation(true,"observation_stall"); return true;
  }

  void invalidateMotion() {
    progress_paused_at=-1.;
    ++generation; target.resize(0); anchor.resize(0); clearWindow();
    last_observed_time = -1.;
    clearObservationWindow(false);
    pending.clear(); certified.clear(); following_frontier = false;
  }

  void select(const std::string& next, unsigned long long mode) {
    if (next == key && mode == epoch) return;
    view_replacement_waiting=false;view_trial_pending=false;
    if(action_progress_enabled) {
      action_progress.suspend();
      if(mode!=epoch)action_budgets.clear();
      else if(!key.empty() && (action_budgets.count(key)||action_budgets.size()<64))
        action_budgets[key]=action_progress;
      const auto saved=action_budgets.find(next);
      action_progress=saved==action_budgets.end() ? ObservationActionProgress() : saved->second;
      action_progress.suspend();
    }
    if (mode != epoch) budgets.clear();
    else if (!key.empty() && (budgets.count(key) || budgets.size()<64))
      budgets[key] = Budget{attempts, segments, active, final_stage_replacement_used, failed_directions, failed_direction_token, unapplied_replacements};
    key = next; epoch = mode;
    observation_region.clear();unseen_window_start=-1.;unseen_best_fraction=-1.;unseen_best_gap=-1.;unseen_window_commits=0;
    monitor_detour_observation=false;observation_replacement_due=false;
    attempts = segments = rejected = unapplied_replacements = 0; active = false; following_frontier = false;
    final_stage_replacement_used = false;
    failed_directions.clear();failed_direction_token.clear(); detour_direction.resize(0);
    last_stall_duration = last_stall_motion = last_raw_span = 0.; last_stall_commits = 0;
    const auto saved = budgets.find(key);
    if (saved != budgets.end()) {
      attempts = saved->second.attempts; segments = saved->second.segments;
      active = saved->second.active;
      unapplied_replacements = saved->second.unapplied_replacements;
      final_stage_replacement_used = saved->second.final_stage_replacement_used;
      failed_directions=saved->second.failed_directions;
      failed_direction_token=saved->second.direction_token;
    } else if (budgets.size() >= 64) {
      active = true; attempts = max_attempts + 1; // bounded, no eviction/retry loop
    }
    invalidateMotion(); last_execution = 0;
    clearObservationWindow();geometry_signature.clear();geometry_query_time=-1.;
    last_advance_reason="none";
    observation_token.clear(); observation_goal.resize(0); observation_mask.resize(0);
    feedback_query_time=-1.; feedback_best.setConstant(-1.); last_stall_reason="none";
  }
  bool blocked() const { return active && attempts > max_attempts; }
  int stage() const { return active ? attempts : 0; }
  // A sensor replacement spends the SAME recovery ledger, before generation.
  // Token changes, rejected proposals and timeouts never refund this charge.
  bool canReserveReplacement() const {
    return active && attempts >= 1 && attempts <= max_attempts &&
        segments < max_segments &&
        (attempts < max_attempts || !final_stage_replacement_used);
  }
  bool reserveReplacement() {
    if (!canReserveReplacement()) return false;
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
    clearObservationWindow();
  }
  void advance(bool mark_direction_failed=true,const char* reason="direction_failure") {
    last_advance_reason=reason;
    if (mark_direction_failed && detour_direction.size()==7 && detour_direction.allFinite() &&
        detour_direction.norm()>1e-9 && failed_directions.size()<max_attempts)
      failed_directions.push_back(detour_direction.normalized());
    detour_direction.resize(0);
    active = true; ++attempts; ++generation; rejected = 0; segments = 0;
    final_stage_replacement_used = false;
    following_frontier = false;
    target.resize(0); clearWindow(); pending.clear(); certified.clear();
    clearObservationWindow(false);
  }
  void advanceObservation(bool failed_direction,const char* reason) {
    // The existing fifth stage has one final replacement reservation. Ask
    // for it before crossing into the exhausted sentinel; no budget is added.
    if(attempts==max_attempts && canReserveReplacement()) {
      invalidateMotion();last_advance_reason=reason;
    } else advance(failed_direction,reason);
    observation_replacement_due=true;
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
  // A direct path counterexample may activate the same finite recovery ledger
  // before any motion is certified. Generic unavailable evidence does not.
  bool reject(unsigned long long raw, bool activate_recovery = false) {
    const auto it = pending.find(raw);
    if (it == pending.end()) return false;
    const bool current = it->second.generation == generation;
    pending.erase(it);
    if (current && (active || activate_recovery) && !blocked() && ++rejected >= 2) { advance(); return true; }
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
    observation_replacement_due=false;
    const bool new_execution = execution != last_execution;
    last_execution = execution; last_observed_time = now; last_raw_span = c.span;
    certified.erase(certified.begin(), it); // retain current identity for later measured samples
    // Ordinary observation steering and rejoining count progress toward the
    // stable published q_vis, not the rolling local frontier or any wrist motion.
    // A bounded detour keeps the existing whole-arm segment/stall accounting.
    const bool observation_monitor=observation_goal.size()==7 &&
        (!active || c.following_frontier || monitor_detour_observation || persistent_observation_monitor);
    // A geometry-only restore still serves the same unseen region. Candidate
    // generations and local clearance gains cannot reset this actual-map window.
    if(action_progress_enabled && observation_monitor && !observation_region.empty() &&
        feedback_query_time>=0. && now-feedback_query_time>=0. && now-feedback_query_time<=.5) {
      if(view_replacement_waiting) { action_progress.suspend(); return false; }
      if(view_trial_pending) {
        action_progress.nextTrial();view_trial_pending=false;++executed_view_trials;
      }
      const double gap=((q-observation_goal).cwiseProduct(observation_mask)).norm();
      const bool geometry_live=geometry_query_time>=0. && now-geometry_query_time>=0. &&
          now-geometry_query_time<=.2 && geometry_measured_q.size()==7 &&
          (q-geometry_measured_q).lpNorm<Eigen::Infinity>()<=.02;
      if(!blocked() && action_progress.observe(new_execution,now,gap,feedback_best[0],
          observation_token+geometry_signature,geometry_latest_margin,geometry_live)) {
        last_stall_duration=action_progress.elapsed;last_stall_commits=action_progress.commits;
        last_stall_motion=0.;last_stall_reason="observation_no_progress";
        advanceObservation(true,"executed_view_no_progress");
        action_progress.nextTrial();return true;
      }
    }
    if(!action_progress_enabled && persistent_observation_monitor && active && !c.following_frontier &&
        observation_goal.size()==7 && !observation_region.empty() &&
        feedback_query_time>=0. && now-feedback_query_time>=0. && now-feedback_query_time<=.5 && feedback_best[0]<1.) {
      const double gap=((q-observation_goal).cwiseProduct(observation_mask)).norm();
      if(unseen_window_start<0. || feedback_best[0]-unseen_best_fraction>=.001 ||
          (unseen_best_gap>=0. && unseen_best_gap-gap>=.001)) {
        unseen_window_start=now;unseen_window_commits=0;
        unseen_best_fraction=std::max(unseen_best_fraction,feedback_best[0]);
      }
      unseen_best_gap=unseen_best_gap<0. ? gap : std::min(unseen_best_gap,gap);
      if(new_execution)++unseen_window_commits;
      if(unseen_window_commits>=6 && now-unseen_window_start>=1. && !blocked()) {
        last_stall_duration=now-unseen_window_start;last_stall_commits=unseen_window_commits;
        last_stall_reason="observation_no_progress";
        unseen_window_start=now;unseen_window_commits=0;
        advanceObservation(true,"unseen_execution_window");return true;
      }
    }
    if (!action_progress_enabled && observation_monitor && observeGoalProgress(new_execution,now,q)) return true;
    if (!observation_monitor) clearObservationWindow(false);
    if (anchor.size() != 7) anchor = q;
    // Movement is a local waypoint transition, NEVER observation completion.
    // Continue the detour across commits, with a finite segment budget.
    if ((q-anchor).lpNorm<Eigen::Infinity>() >= .01) {
      anchor = q; clearWindow();
      if (!active || blocked()) return false;
      if (!c.following_frontier && ++segments >= max_segments) {
        const bool unseen_approach=observation_monitor && monitor_detour_observation &&
            geometry_query_time>=0. && now-geometry_query_time>=0. && now-geometry_query_time<=.5 &&
            geometry_best<0. && feedback_query_time>=0. && now-feedback_query_time>=0. &&
            now-feedback_query_time<=.5 && feedback_best.maxCoeff()==0.;
        if(unseen_approach) {
          last_stall_reason="observation_no_progress";
          advanceObservation(false,"unseen_approach_budget");
        } else advance(false,"segment_budget");
      }
      else { ++generation; target.resize(0); pending.clear(); certified.clear(); }
      return true;
    }
    if (observation_monitor) return false;
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
      last_stall_reason = "measured_motion_stall";
      advance(); return true;
    }
    return false;
  }
};

// A nearby one-sided CDF guard needs a detour only when the proposed forward
// step does not improve it and would leave it below its required margin.
// An improving step may still be insufficient from a deficient start: this
// chooses an objective only; the complete hard QP must establish feasibility.
inline bool recoveryForwardBlocked(double clearance, double along) {
  if (!std::isfinite(clearance) || !std::isfinite(along)) return true;
  return clearance < .005 && along <= 1e-9 && clearance + along < 0.;
}

// Tangential search around the currently blocking CDF halfspaces. The candidate
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

// A bounded objective-direction pool; feasibility is still established ONLY
// by the original full hard QP. Outward directions complement tangent ones.
inline Eigen::VectorXd boundedObservationTarget(
    const Eigen::VectorXd& q0, const Eigen::VectorXd& desired,
    const std::vector<Eigen::VectorXd>& normals, const std::vector<double>& slack,
    const Eigen::VectorXd& vis_gradient, const Eigen::VectorXd& mask,
    const std::vector<Eigen::VectorXd>& failed, double step) {
  if (q0.size()!=7 || desired.size()!=7 || vis_gradient.size()!=7 || mask.size()!=7 ||
      !q0.allFinite() || !desired.allFinite() || !vis_gradient.allFinite() ||
      !mask.allFinite() || normals.empty() || slack.size()!=normals.size() ||
      !std::isfinite(step) || step<=0.) return Eigen::VectorXd();
  Eigen::MatrixXd projector=Eigen::MatrixXd::Identity(7,7);
  std::vector<Eigen::VectorXd> basis;
  Eigen::VectorXd outward=Eigen::VectorXd::Zero(7);
  for (const auto& g:normals) {
    if (g.size()!=7 || !g.allFinite() || g.norm()<1e-9) return Eigen::VectorXd();
    outward+=g.normalized(); Eigen::VectorXd v=g.normalized();
    for (const auto& b:basis) v-=b.dot(v)*b;
    if (v.norm()>1e-7) basis.push_back(v.normalized());
  }
  for (const auto& b:basis) projector-=b*b.transpose();
  // Maximum pool size stays fourteen, the old +/- seven-coordinate pool.
  std::vector<Eigen::VectorXd> pool={projector*vis_gradient, outward,
      projector*(desired-q0), vis_gradient.cwiseProduct(mask)};
  for (int j=0;j<7 && pool.size()<14;++j) {
    const Eigen::VectorXd v=projector.col(j);
    if (v.norm()>.1) { pool.push_back(v); if (pool.size()<14) pool.push_back(-v); }
  }
  double best=-1e100; Eigen::VectorXd chosen;
  for (auto v:pool) {
    if (!v.allFinite() || v.lpNorm<Eigen::Infinity>()<1e-8) continue;
    v*=std::min(.03,step)/v.lpNorm<Eigen::Infinity>();
    bool tried=false;
    for (const auto& old:failed)
      if (old.size()==7 && old.allFinite() && old.norm()>1e-9 &&
          old.normalized().dot(v.normalized())>.98) tried=true;
    if (tried) continue;
    double violation=0., clearance_gain=1e100;
    for (std::size_t i=0;i<normals.size();++i) {
      if (!std::isfinite(slack[i])) return Eigen::VectorXd();
      const double gain=normals[i].dot(v)/normals[i].norm();
      violation+=std::max(0.,-(slack[i]/normals[i].norm()+gain));
      clearance_gain=std::min(clearance_gain,gain);
    }
    const double vis_gain=vis_gradient.norm()>1e-9 ?
        vis_gradient.normalized().dot(v):0.;
    const Eigen::VectorXd delta=(desired-q0).cwiseProduct(mask);
    const double goal_gain=delta.norm()>1e-9 ? delta.normalized().dot(v):0.;
    const double score=-100.*violation+vis_gain+clearance_gain+.2*goal_gain;
    if (score>best+1e-10) {best=score;chosen=q0+v;}
  }
  return chosen;
}
}  // namespace egocentric_arm_planner
