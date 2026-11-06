#pragma once
#include <trajectory_msgs/JointTrajectory.h>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>

namespace egocentric_arm_planner {
// Scheduling memory only. No entry authorizes motion or replaces a verifier.
struct NormalTerminalRetention {
  using Trajectory = trajectory_msgs::JointTrajectory;
  struct Proof { std::uint64_t raw=0,audit=0; std::string context; };
  static constexpr std::size_t capacity=16;
  std::map<std::uint64_t,std::string> pending;
  std::map<std::uint64_t,Proof> proofs;
  std::map<std::uint64_t,Trajectory> audits,commits;
  std::uint64_t newest_commit=0,armed=0;
  std::string armed_context,reason="idle";
  std::uint64_t arm_count=0,revoke_count=0,suppressed_count=0;

  template<class Map> static void trim(Map& m) {
    while(m.size()>capacity)m.erase(m.begin());
  }
  static bool sameMotion(const Trajectory& a,const Trajectory& b) {
    if(a.header.frame_id!=b.header.frame_id || a.joint_names!=b.joint_names || a.points.size()!=b.points.size())return false;
    for(std::size_t k=0;k<a.points.size();++k) {
      const auto& x=a.points[k];const auto& y=b.points[k];
      if(x.positions!=y.positions || x.velocities!=y.velocities || x.accelerations!=y.accelerations ||
          x.effort!=y.effort || x.time_from_start!=y.time_from_start)return false;
    }
    return true;
  }
  static bool zeroTail(const Trajectory& t) {
    if(t.joint_names.size()!=7 || t.points.size()<2)return false;
    const auto& end=t.points.back();
    if(end.positions.size()!=7 || end.time_from_start.toNSec()<100000000)return false;
    const auto start=end.time_from_start.toNSec()-100000000;
    bool covered=false;
    for(auto it=t.points.rbegin();it!=t.points.rend();++it) {
      if(it->positions.size()!=7 || it->velocities.size()!=7)return false;
      for(int j=0;j<7;++j)if(!std::isfinite(it->positions[j]) || !std::isfinite(it->velocities[j]) ||
          std::abs(it->positions[j]-end.positions[j])>1e-9 || std::abs(it->velocities[j])>1e-8)return false;
      if(it->time_from_start.toNSec()<=start){covered=true;break;}
    }
    for(std::size_t k=1;k<t.points.size();++k)
      if(t.points[k].time_from_start<=t.points[k-1].time_from_start)return false;
    return covered;
  }
  // A planning request revokes a scheduling lease, but does not revoke an
  // already published audit/commit fact. Keep bounded in-flight provenance so
  // delayed callbacks can join it; context, newest token, exact motion, tail
  // and the caller's tracker/pose checks still decide whether it can arm.
  void replan(const std::string& why) {
    if(armed)++revoke_count;
    armed=0;armed_context.clear();reason=why;
  }
  void invalidate(const std::string& why) {
    if(armed)++revoke_count;
    armed=0;armed_context.clear();pending.clear();proofs.clear();audits.clear();commits.clear();reason=why;
  }
  void candidate(std::uint64_t raw,const std::string& context) {
    if(raw && !context.empty()){pending.emplace(raw,context);trim(pending);}
  }
  void audit(const Trajectory& t) {
    const auto id=t.header.stamp.toNSec();if(!id)return;
    const auto old=audits.find(id);
    if(old!=audits.end() && !sameMotion(old->second,t)){invalidate("audit_payload_changed");return;}
    audits.emplace(id,t);trim(audits);
  }
  void commit(const Trajectory& t) {
    const auto id=t.header.stamp.toNSec();if(!id || id<newest_commit)return;
    const auto old=commits.find(id);
    if(old!=commits.end() && !sameMotion(old->second,t)){invalidate("commit_payload_changed");return;}
    if(id>newest_commit){newest_commit=id;armed=0;armed_context.clear();reason="new_execution";}
    commits.emplace(id,t);trim(commits);
  }
  void certify(std::uint64_t raw,std::uint64_t audit,std::uint64_t execution,const std::string& context) {
    const auto p=pending.find(raw);
    if(!raw || !audit || !execution || p==pending.end() || context.empty() || p->second!=context)return;
    for(const auto& old:proofs)if(old.second.raw==raw &&
        (old.first!=execution || old.second.audit!=audit))return;
    proofs.emplace(execution,Proof{raw,audit,context});trim(proofs);
  }
  template<class Eligible> void tryArm(const std::string& context,Eligible eligible) {
    if(context.empty() || armed==newest_commit)return;
    const auto p=proofs.find(newest_commit);const auto c=commits.find(newest_commit);
    if(p==proofs.end() || c==commits.end() || p->second.context!=context)return;
    const auto ticket=pending.find(p->second.raw);const auto a=audits.find(p->second.audit);
    if(ticket==pending.end() || ticket->second!=context || a==audits.end() ||
        !sameMotion(a->second,c->second) || !zeroTail(c->second) || !eligible(c->second))return;
    armed=newest_commit;armed_context=context;reason="certified_goal_tail";++arm_count;
  }
};
}  // namespace egocentric_arm_planner
