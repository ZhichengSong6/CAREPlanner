#pragma once
#include <algorithm>
#include <cmath>
#include <string>

namespace egocentric_arm_planner {
// Performance of certified, actually executed short motions. Geometry only
// earns one bounded approach extension. Map observation renews a view's trial;
// the owner starts a fresh trial when an authenticated new view is executed.
// No method here certifies motion, marks a voxel seen, or changes a constraint.
struct ObservationActionProgress {
  double elapsed=0., budget=0., last_time=-1., best_seen=-1.;
  double first_margin=0., last_margin=0., fov_gain=0.;
  int commits=0, geometry_samples=0, renewals=0;
  bool extended=false, initialized=false;
  std::string geometry_id, decision="idle";
  void suspend() { last_time=-1.; }
  void nextTrial() {
    elapsed=budget=0.; last_time=-1.; commits=geometry_samples=0;
    initialized=extended=false; geometry_id.clear(); fov_gain=0.;
  }
  bool observe(bool new_execution,double now,double gap,double seen,
      const std::string& signature,double margin,bool geometry_valid) {
    if(!std::isfinite(now)||!std::isfinite(gap)||gap<0.||
        !std::isfinite(seen)||seen<0.||seen>1.)return false;
    const bool observed=best_seen>=0. && seen-best_seen>=.001;
    if(!initialized || observed) {
      if(observed)++renewals;
      nextTrial(); initialized=true;
      // A finite pre-FOV approach is expected. Distance sizes this initial
      // allowance, but subsequent milliradian steps cannot renew it.
      budget=std::min(3.5,std::max(1.5,1.+gap/.25));
    }
    best_seen=std::max(best_seen,seen);
    if(last_time>=0. && now>last_time)elapsed+=now-last_time;
    last_time=now;
    if(new_execution)++commits;
    if(geometry_valid && std::isfinite(margin) && !signature.empty()) {
      if(signature!=geometry_id) {
        geometry_id=signature;first_margin=margin;geometry_samples=0;
      }
      last_margin=margin;fov_gain=margin-first_margin;++geometry_samples;
    }
    decision="approach";
    if(seen>=1.) { decision="observed";return false; }
    if(commits<4 || elapsed<budget)return false;
    // This is measured geometry along executed motion, never the endpoint
    // q_vis score or the unexecuted QP's predicted gain. One extension only.
    if(!extended && geometry_valid && geometry_samples>=3 && fov_gain>=.02) {
      budget+=1.5;extended=true;decision="useful_executed_approach";return false;
    }
    decision="replace_unproductive_view";return true;
  }
};
} // namespace egocentric_arm_planner
