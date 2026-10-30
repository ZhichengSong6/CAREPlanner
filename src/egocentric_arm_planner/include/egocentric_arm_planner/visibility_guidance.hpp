#pragma once
#include <Eigen/Core>
#include <cmath>
#include <string>

namespace egocentric_arm_planner {
struct VisibilityGuidanceState {
  std::string token, region;
  unsigned long long sequence=0, epoch=0;
  int sensor=-1;
  double stamp=-1., value=0.;
  bool drive=false;
  bool geometry_valid=false;
  std::string geometry_signature;
  double conservative_fov_margin=0.;
  Eigen::VectorXd q_eval, target, mask, gradient;
  bool compatible(const std::string& wanted_token, const std::string& wanted_region,
      int wanted_sensor, const Eigen::VectorXd& wanted_mask, unsigned long long mode,
      const Eigen::VectorXd& measured, double now) const {
    return sequence && epoch==mode && token==wanted_token && region==wanted_region &&
      token.compare(0,12,"care_obs_v1_")==0 && region.compare(0,15,"care_region_v1_")==0 &&
      sensor>=0 && sensor<8 && sensor==wanted_sensor &&
      std::isfinite(now) && std::isfinite(stamp) && now-stamp>=0. && now-stamp<=.2 &&
      std::isfinite(value) && q_eval.size()==7 && target.size()==7 && mask.size()==7 &&
      gradient.size()==7 && measured.size()==7 && wanted_mask.size()==7 &&
      q_eval.allFinite() && target.allFinite() && mask.allFinite() && gradient.allFinite() &&
      measured.allFinite() && wanted_mask.allFinite() && (mask.array()>=0.).all() &&
      (mask.array()<=1.).all() && mask.sum()>0. &&
      (mask-wanted_mask).norm()<1e-9 &&
      (measured-q_eval).lpNorm<Eigen::Infinity>()<=.02 &&
      (target-q_eval).lpNorm<Eigen::Infinity>()<=.2+1e-9 &&
      ((Eigen::VectorXd::Ones(7)-mask).cwiseProduct(gradient)).norm()<1e-9;
  }
};
}
