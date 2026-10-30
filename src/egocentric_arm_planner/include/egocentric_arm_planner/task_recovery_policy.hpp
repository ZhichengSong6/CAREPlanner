#pragma once
#include "egocentric_arm_planner/task_observation_context.hpp"
#include "egocentric_arm_planner/visibility_guidance.hpp"
#include <openssl/sha.h>

namespace egocentric_arm_planner {
inline std::string recoveryContentSha(const std::string& kind,const std::string& raw) {
  std::string bytes="care_recovery_"+kind+"_v1";bytes.push_back('\0');bytes+=raw;
  unsigned char hash[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char*>(bytes.data()),bytes.size(),hash);
  std::ostringstream out;out<<std::hex<<std::setfill('0');
  for(auto c:hash)out<<std::setw(2)<<static_cast<unsigned>(c);
  return out.str();
}
inline bool recoveryTreeEqual(const boost::property_tree::ptree& a,const boost::property_tree::ptree& b) {
  if(a.data()!=b.data() || a.size()!=b.size())return false;
  auto x=a.begin(),y=b.begin();
  for(;x!=a.end();++x,++y)if(x->first!=y->first || !recoveryTreeEqual(x->second,y->second))return false;
  return true;
}
struct TaskRecoveryPolicy {
  std::string target_sha,sha,binding,regime;
  uint64_t measured_ns=0,until=0;
  bool present=false;
  double frontier_weight=0.,qvis_weight=1.;
  Eigen::VectorXd q_eval,target,mask;
  VisibilityGuidanceState guidance;
  bool has_guidance=false;
  static Eigen::VectorXd vector(const boost::property_tree::ptree& tree,size_t size=7) {
    Eigen::VectorXd out(size);size_t n=0;
    for(const auto& cell:tree) {
      if(!cell.first.empty() || n>=size)throw std::runtime_error("invalid_recovery_policy_vector");
      const double value=cell.second.get_value<double>();
      if(!std::isfinite(value))throw std::runtime_error("nonfinite_recovery_policy_vector");
      out[n++]=value;
    }
    if(n!=size)throw std::runtime_error("invalid_recovery_policy_vector");
    return out;
  }
  static TaskRecoveryPolicy parse(const boost::property_tree::ptree& payload,const std::string& binding) {
    TaskRecoveryPolicy out;
    const auto selection=payload.get_child_optional("selection");
    if(!selection)return out; // Legacy target-only bootstrap has no soft policy.
    const auto& s=*selection;const auto raw=s.get<std::string>("target_content");
    const auto digest=s.get<std::string>("target_sha256");
    boost::property_tree::ptree decoded;std::istringstream in(raw);read_json(in,decoded);
    if(s.size()!=5 || s.get<int>("schema")!=1 || raw.size()>32768 ||
       digest!=recoveryContentSha("target",raw) || !recoveryTreeEqual(decoded,payload.get_child("target")))
      throw std::runtime_error("invalid_recovery_selection");
    out.target_sha=decoded.data()=="null" ? std::string() : digest;
    const auto content=s.get<std::string>("policy_content");out.sha=s.get<std::string>("policy_sha256");
    if(content.empty()) {
      if(!out.sha.empty())throw std::runtime_error("invalid_empty_recovery_policy");
      return out;
    }
    if(decoded.data()=="null" || out.sha!=recoveryContentSha("policy",content))
      throw std::runtime_error("recovery_policy_digest_mismatch");
    boost::property_tree::ptree p;std::istringstream policy(content);read_json(policy,p);
    out.binding=p.get<std::string>("task_context");out.regime=p.get<std::string>("task_regime");
    out.measured_ns=p.get<uint64_t>("measured_ns");out.until=p.get<uint64_t>("valid_until_ns");
    if(p.size()!=9 || p.get<int>("schema")!=1 || out.binding!=binding ||
       p.get<std::string>("target_sha256")!=digest || !out.measured_ns || out.until<=out.measured_ns ||
       out.until-out.measured_ns>200000000ULL ||
       !std::regex_match(out.regime,std::regex("[A-Za-z0-9_.:-]{1,128}:[1-9][0-9]*:(NORMAL|REPAIR|PROBE_NORMAL)")))
      throw std::runtime_error("invalid_recovery_policy_binding");
    out.q_eval=vector(p.get_child("q_eval"));const auto& f=p.get_child("frontier");
    out.target=vector(f.get_child("target"));out.mask=vector(f.get_child("joint_mask"));
    out.frontier_weight=f.get<double>("frontier_weight_scale");out.qvis_weight=f.get<double>("qvis_weight_scale");
    const auto wanted_mask=vector(decoded.get_child("joint_mask"));
    if(f.size()!=4 || (out.mask-wanted_mask).norm()>1e-9 || (out.mask.array()<0.).any() ||
       (out.mask.array()>1.).any() || out.mask.sum()<=0. ||
       !std::isfinite(out.frontier_weight) || !std::isfinite(out.qvis_weight) ||
       out.frontier_weight<0. || out.qvis_weight<0. || out.frontier_weight>1e6 || out.qvis_weight>1e6)
      throw std::runtime_error("invalid_recovery_frontier");
    const auto& g=p.get_child("guidance");
    if(g.data()!="null") {
      auto& v=out.guidance;v.token=decoded.get<std::string>("observation_token");
      v.region=decoded.get<std::string>("region_token");v.sequence=g.get<uint64_t>("sequence");
      v.sensor=g.get<int>("sensor_id");v.stamp=out.measured_ns*1e-9;v.q_eval=out.q_eval;v.mask=out.mask;
      v.target=vector(g.get_child("target"));v.gradient=vector(g.get_child("gradient"));
      v.value=g.get<double>("value");v.drive=g.get<bool>("drive_enabled");
      v.geometry_valid=g.get<bool>("geometry_valid");v.geometry_signature=g.get<std::string>("geometry_signature");
      v.conservative_fov_margin=g.get<double>("conservative_fov_margin");
      if(g.size()!=9 || v.sequence>=9007199254740992ULL || v.geometry_signature.size()>512 ||
         !std::isfinite(v.conservative_fov_margin) || !v.compatible(v.token,v.region,
           decoded.get<int>("sensor_id"),wanted_mask,0,out.q_eval,v.stamp))
        throw std::runtime_error("invalid_recovery_guidance");
      out.has_guidance=true;
    }
    out.present=true;return out;
  }
  bool current(const std::string& task,const std::string& mode,const std::string& adopted,
               const Eigen::VectorXd& measured,uint64_t now) const {
    return present && binding==task && regime==mode && target_sha==adopted &&
      measured.size()==7 && measured.allFinite() && now>=measured_ns && now<until &&
      (measured-q_eval).lpNorm<Eigen::Infinity>()<=.02;
  }
};
}
