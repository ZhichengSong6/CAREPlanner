#pragma once
#include <Eigen/Core>
#include <cmath>
#include <cstdint>
#include <string>

namespace egocentric_arm_planner {
// Lifecycle and identity only. Observed settling is not a stopping certificate
// and never authorizes renewed motion or resurrects the aborted trajectory.
struct ExecutionStopState {
  std::uint64_t token=0;
  std::uint32_t seq=0;
  double phase=0.,remaining=0.,started=0.,settling_since=-1.,last_sample=-1.;
  bool settled=false;
  std::string reason;
  void begin(std::uint64_t stamp,std::uint32_t sequence,double phase_s,
             double remaining_s,double now,const std::string& cause) {
    token=stamp;seq=sequence;phase=phase_s;remaining=remaining_s;started=now;
    reason=cause;settled=false;settling_since=last_sample=-1.;
  }
  void observe(double now,double sample,const Eigen::VectorXd& measured_velocity,
      const Eigen::VectorXd& limiter_output,bool state_fresh,bool output_fresh) {
    if(!token)return;
    const bool stopped=state_fresh && output_fresh && std::isfinite(now) && std::isfinite(sample) &&
        sample>=started && sample<=now && measured_velocity.size()>0 &&
        limiter_output.size()==measured_velocity.size() && measured_velocity.allFinite() && limiter_output.allFinite() &&
        measured_velocity.lpNorm<Eigen::Infinity>()<=.001 && limiter_output.lpNorm<Eigen::Infinity>()<=1e-9;
    if(!stopped){settled=false;settling_since=-1.;last_sample=-1.;return;}
    if(sample<=last_sample)return;
    if(settling_since<0. || sample-last_sample>.05)settling_since=sample;
    last_sample=sample;
    if(sample-settling_since>=.05-1e-9)settled=true;
  }
  const char* state()const{return token?(settled?"STOPPED_HOLD":"STOPPING"):"NONE";}
};
}  // namespace egocentric_arm_planner
