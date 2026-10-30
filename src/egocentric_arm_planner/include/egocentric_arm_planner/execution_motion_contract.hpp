#pragma once
#include <care_confidence_map/trajectory_motion.hpp>
#include <algorithm>
#include <cmath>
#include <string>

namespace egocentric_arm_planner {
// Independent actuator-facing rejection of an inadmissible committed curve.
// The normal path rejects the same curve before GCDF/VBC certification.
inline std::string executionMotionViolation(const trajectory_msgs::JointTrajectory& t,
    const std::vector<int>& mapping,const std::vector<double>& velocity,
    const std::vector<double>& acceleration,double tolerance=1e-6) {
  const std::size_t n=mapping.size();
  if(!n || t.points.empty() || velocity.size()!=n || acceleration.size()!=n)return "motion_limit_dimensions";
  for(std::size_t j=0;j<n;++j)
    if(!std::isfinite(velocity[j]) || velocity[j]<=0. || !std::isfinite(acceleration[j]) || acceleration[j]<=0.)
      return "invalid_motion_limits";
  if(t.points.front().time_from_start.toSec()!=0.)return "motion_missing_measured_start";
  for(std::size_t k=0;k<t.points.size();++k) {
    const auto& p=t.points[k];const double now=p.time_from_start.toSec();
    for(std::size_t j=0;j<n;++j) {
      const int i=mapping[j];
      if(i<0 || std::size_t(i)>=p.positions.size() || std::size_t(i)>=p.velocities.size() ||
          !std::isfinite(p.positions[i]) || !std::isfinite(p.velocities[i]))return "invalid_motion_state";
      if(std::abs(p.velocities[i])>velocity[j]+tolerance)return "reference_velocity_limit";
    }
    if(k) {
      const auto& a=t.points[k-1];const double h=now-a.time_from_start.toSec();
      if(!std::isfinite(h) || h<=1e-12)return "nonincreasing_motion_time";
      const auto curve=care_confidence_map::trajectoryCurvature(a,p,mapping);
      for(std::size_t j=0;j<n;++j) {
        const int i=mapping[j];const double secant=(p.positions[i]-a.positions[i])/h;
        const double v0=secant+4.*curve[j]/h,v1=secant-4.*curve[j]/h;
        if(std::max(std::abs(v0),std::abs(v1))>velocity[j]+tolerance)return "continuous_velocity_limit";
        if(std::abs(8.*curve[j]/(h*h))>acceleration[j]+tolerance)return "continuous_acceleration_limit";
        if(std::max(std::abs(v0-a.velocities[i]),std::abs(v1-p.velocities[i]))>tolerance)
          return "motion_velocity_discontinuity";
      }
    }
  }
  for(const int i:mapping)if(std::abs(t.points.back().velocities[i])>tolerance)return "motion_missing_terminal_stop";
  return {};
}
}  // namespace egocentric_arm_planner
