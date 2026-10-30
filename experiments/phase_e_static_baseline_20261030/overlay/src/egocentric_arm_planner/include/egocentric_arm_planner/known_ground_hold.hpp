#pragma once
#include <cmath>
#include <string>

namespace egocentric_arm_planner {
// Scheduling only. A wakeup requests a fresh solve; it grants no execution.
struct KnownGroundHold {
  enum State { Idle, Waiting, RequeryPending, Exhausted, Cancelled };
  State state=Idle;
  unsigned long long goal=0, execution_at_entry=0;
  unsigned wakeups=0;
  std::string context,reason="idle";
  double ros_deadline=0.,wall_deadline=0.,last_check=-1.,last_stamp=-1.,stable_since=-1.;
  unsigned stable_samples=0;
  bool transaction=false;
  void bind(unsigned long long revision) {
    if(goal==revision)return;
    *this=KnownGroundHold();goal=revision;
  }
  bool blocked() const { return state==Waiting || state==Exhausted; }
  bool enter(unsigned long long revision,const std::string& identity,double ros,double wall,
             unsigned long long execution) {
    bind(revision);context=identity;stable_since=-1.;stable_samples=0;last_stamp=-1.;last_check=-1.;
    if(!goal || identity.empty() || wakeups>=2) {
      state=Exhausted;reason="ground_wakeup_budget";return false;
    }
    if(!transaction) {
      transaction=true;ros_deadline=ros+1.;wall_deadline=wall+1.5;execution_at_entry=execution;
    }
    state=Waiting;reason="ground_wait_clearance";return true;
  }
  void cancel(const std::string& why) {
    if(state==Waiting || state==RequeryPending) {state=Cancelled;reason=why;}
    stable_since=-1.;stable_samples=0;
  }
  bool due(double wall) const { return state==Waiting && (last_check<0. || wall-last_check>=.05); }
  bool observe(const std::string& identity,double ros,double wall,double stamp,bool valid,double clearance,double required=.02) {
    if(state!=Waiting)return false;
    last_check=wall;
    if(identity!=context) {cancel("ground_context_changed");return false;}
    if(ros>=ros_deadline || wall>=wall_deadline) {
      state=Exhausted;reason="ground_wait_timeout";return false;
    }
    if(!valid || !std::isfinite(stamp) || stamp<=0. || stamp<=last_stamp || ros<stamp || ros-stamp>.05 ||
        !std::isfinite(clearance) || !std::isfinite(required) || clearance<required) {
      stable_since=-1.;stable_samples=0;return false;
    }
    last_stamp=stamp;
    if(stable_since<0.)stable_since=stamp;
    ++stable_samples;
    if(stable_samples<3 || stamp-stable_since<.1-1e-9)return false;
    ++wakeups;state=RequeryPending;reason="ground_clearance_restored";return true;
  }
  void completed(unsigned long long execution) {
    if(state==RequeryPending && execution>execution_at_entry) {
      transaction=false;state=Idle;reason="ground_new_execution_completed";
    }
  }
};
}
