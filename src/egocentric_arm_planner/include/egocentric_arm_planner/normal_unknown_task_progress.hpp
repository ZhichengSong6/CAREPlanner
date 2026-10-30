#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <string>

namespace egocentric_arm_planner {

// Scheduling evidence only: actual task progress while the same live UNKNOWN
// obligation constrains certified NORMAL executions. Never authorizes motion.
struct NormalUnknownTaskProgress {
  struct Evidence {
    std::string context;
    unsigned long long raw = 0, query = 0;
    double raw_time_s = 0.;
  };
  std::map<unsigned long long, Evidence> pending, certified;
  Evidence last;
  std::string episode_context;
  double first_time_s = -1., last_time_s = -1.;
  double credited_error = -1., current_error = -1.;
  unsigned long long last_execution_stamp_ns = 0;
  int execution_stamps = 0;
  bool emitted = false;

  // Same time, distinct-execution and freshness limits as the existing
  // UNKNOWN stationary detector. Task units use the existing goal tolerances.
  double min_duration_s = 2., max_sample_gap_s = 1., max_evidence_age_s = 2.;
  int min_execution_stamps = 3;
  double position_tolerance_m = .02, orientation_tolerance_rad = .2;
  double progress_epsilon = .25;

  unsigned long long event_sequence = 0;
  std::string event = "none", reason = "none", event_context = "none";
  Evidence reset_evidence;
  unsigned long long reset_execution = 0;
  int reset_execution_stamps = 0;
  double reset_duration_s = 0., reset_credited_error = -1., reset_current_error = -1.;

  double duration(double now) const {
    return first_time_s >= 0. ? now-first_time_s : 0.;
  }
  void note(const std::string& what, const std::string& why,
            const std::string& context) {
    ++event_sequence; event = what; reason = why;
    event_context = context.empty() ? "none" : context;
  }
  void resetWindow(const std::string& why) {
    if (first_time_s >= 0.) {
      reset_evidence=last;reset_execution=last_execution_stamp_ns;
      reset_execution_stamps=execution_stamps;reset_duration_s=duration(last_time_s);
      reset_credited_error=credited_error;reset_current_error=current_error;
      note("reset", why, episode_context);
    }
    episode_context.clear(); first_time_s = last_time_s = -1.;
    credited_error = current_error = -1.;
    last_execution_stamp_ns = 0; execution_stamps = 0; emitted = false;
  }
  void reset(const std::string& why) {
    resetWindow(why); pending.clear(); certified.clear(); last = Evidence{};
  }
  bool remember(unsigned long long raw, unsigned long long query,
                const std::string& context, bool blocked_unknown) {
    if (!raw || !query || context.empty() || !blocked_unknown) return false;
    while (pending.size() >= 32) pending.erase(pending.begin());
    return pending.emplace(raw, Evidence{context,raw,query,raw*1e-9}).second;
  }
  bool certify(unsigned long long raw, unsigned long long execution,
               const std::string& context) {
    const auto it = pending.find(raw);
    if (it == pending.end()) return false;
    const auto e = it->second;
    pending.erase(it);  // A duplicate verdict cannot mint another identity.
    if (!execution || e.context != context || execution < raw) return false;
    while (certified.size() >= 32) certified.erase(certified.begin());
    return certified.emplace(execution,e).second;
  }
  bool qualified(double now, unsigned long long execution,
                 const std::string& context) const {
    const auto it = certified.find(execution);
    return !context.empty() && it != certified.end() &&
        it->second.context == context && std::isfinite(now) &&
        now >= it->second.raw_time_s &&
        now-it->second.raw_time_s <= max_evidence_age_s;
  }
  bool observe(double now, unsigned long long execution, double position_error_m,
               double orientation_error_rad, const std::string& context,
               bool eligible) {
    if (!eligible || !qualified(now,execution,context)) {
      resetWindow(!eligible ? "ineligible" : "execution_evidence_missing_or_stale");
      return false;
    }
    if (!std::isfinite(position_error_m) || !std::isfinite(orientation_error_rad) ||
        position_error_m < 0. || orientation_error_rad < 0.) {
      resetWindow("invalid_task_pose"); return false;
    }
    const double error = std::max(position_error_m/position_tolerance_m,
                                 orientation_error_rad/orientation_tolerance_rad);
    if (!std::isfinite(error)) { resetWindow("invalid_task_pose"); return false; }
    if (error <= 1.) { resetWindow("goal_reached"); return false; }
    last = certified.at(execution);
    if (first_time_s < 0. || episode_context != context || now < last_time_s ||
        now-last_time_s > max_sample_gap_s) {
      const std::string why = first_time_s < 0. ? "first_sample" :
          episode_context != context ? "context_changed" : "sample_time_gap";
      resetWindow(why);
      episode_context = context; first_time_s = now; credited_error = error;
      note("start",why,context);
    } else if (error <= credited_error-progress_epsilon) {
      // A retreat and return to a previously reached error earns no progress.
      credited_error = error; first_time_s = now;
      last_execution_stamp_ns = 0; execution_stamps = 0; emitted = false;
      note("progress","new_best_task_error",context);
    }
    current_error = error; last_time_s = now;
    if (execution > last_execution_stamp_ns) {
      last_execution_stamp_ns = execution; ++execution_stamps;
    }
    if (!emitted && execution_stamps >= min_execution_stamps &&
        duration(now) >= min_duration_s) {
      emitted = true; note("stalled","no_new_best_task_error",context);
      return true;
    }
    return false;
  }
};
}  // namespace egocentric_arm_planner
