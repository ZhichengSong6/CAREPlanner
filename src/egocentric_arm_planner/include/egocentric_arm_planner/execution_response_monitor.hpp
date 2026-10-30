#pragma once
#include <Eigen/Core>
#include <cmath>

namespace egocentric_arm_planner {
// Liveness diagnosis only. A fault pauses proposal/recovery accounting; it
// neither authorizes a trajectory nor changes any tracking or safety gate.
struct ExecutionResponseMonitor {
  bool blocked = false;
  unsigned long long faults = 0;
  double last_time = -1., window_start = -1.;
  double expected_travel = 0., measured_motion = 0.;
  Eigen::VectorXd anchor, previous_u, integral, fault_anchor;

  void clearWindow() {
    last_time = window_start = -1.; anchor.resize(0); previous_u.resize(0);
    integral.resize(0);
  }
  // eligible means fresh measured joints, a current non-aborted tracker and
  // fresh desired velocity. An explicit safety/limiter hold makes it false.
  bool observe(double now, const Eigen::VectorXd& q, const Eigen::VectorXd& u,
               bool eligible, bool inhibited=false) {
    if (!std::isfinite(now) || q.size()!=7 || !q.allFinite()) return false;
    // A known safety/limiter hold withdraws the motion request. It cannot be
    // judged as actuator nonresponse; planning may be needed to end that hold.
    if (inhibited) {
      const bool changed=blocked; blocked=false; clearWindow(); return changed;
    }
    if (blocked) {
      if ((q-fault_anchor).lpNorm<Eigen::Infinity>() > .001) {
        blocked=false; clearWindow(); return true;
      }
      return false;
    }
    if (!eligible || u.size()!=7 || !u.allFinite()) { clearWindow(); return false; }
    // Multiple tracker callbacks can share one /clock timestamp. They cannot
    // repeatedly erase an accumulating response window or add command travel.
    if (last_time>=0. && now<=last_time) return false;
    if (window_start<0. || now-last_time>.2) {
      anchor=q; previous_u=u; integral=Eigen::VectorXd::Zero(7);
      last_time=window_start=now; expected_travel=measured_motion=0.; return false;
    }
    integral += .5*(previous_u+u)*(now-last_time);
    previous_u=u; last_time=now;
    measured_motion=(q-anchor).lpNorm<Eigen::Infinity>();
    expected_travel=integral.lpNorm<Eigen::Infinity>();
    // Reuse the sensing liveness window and displacement scales. A tiny QP
    // hold, alternating requests with no net travel, or genuine small motion
    // remains a planning/sensing condition and keeps its original recovery.
    if (measured_motion > .001) { clearWindow(); return false; }
    if (now-window_start < 1.) return false;
    if (expected_travel >= .01) {
      blocked=true; fault_anchor=q; ++faults; return true;
    }
    clearWindow(); return false;
  }
};
}  // namespace egocentric_arm_planner
