#pragma once
#include <pinocchio/fwd.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <geometry_msgs/PoseStamped.h>
#include <ros/time.h>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace egocentric_arm_planner {
// Planning lineage only. This state never grants execution permission.
struct TaskObservationContext {
  std::string binding, owner, id, goal;
  std::uint64_t generation=0, sequence=0, stamp=0, until=0, started=0;
  bool active=false;
  mutable bool closed=true;
  mutable std::uint64_t last_now=0;
  geometry_msgs::PoseStamped target;
  std::set<std::string> owners;
  bool current(const std::string& value, const ros::Time& now) const {
    const auto ns=now.toNSec();
    if(until && (ns>until || ns<stamp || (last_now && ns<last_now)))closed=true;
    last_now=std::max(last_now,ns);
    return active && !closed && ns>=stamp && ns<=until && value==binding;
  }
  bool update(const std::string& raw, const ros::Time& now) {
    using boost::property_tree::ptree;
    if(raw.size()>32768)throw std::runtime_error("task_context_too_large");
    ptree e,b;std::istringstream input(raw);read_json(input,e);
    const auto incoming=e.get<std::string>("candidate_binding");
    std::istringstream inner(incoming);read_json(inner,b);
    const auto next_owner=e.get<std::string>("owner_session");
    const auto next_id=e.get<std::string>("context_id");
    const auto next_generation=e.get<std::uint64_t>("generation");
    const auto next_sequence=e.get<std::uint64_t>("event_seq");
    const auto next_stamp=e.get<std::uint64_t>("stamp_ns");
    const auto next_until=e.get<std::uint64_t>("valid_until_ns");
    const auto next_started=e.get<std::uint64_t>("started_ns");
    const auto next_active=e.get<bool>("active");
    if(e.get<int>("schema")!=1 || b.get<int>("schema")!=1 || incoming.size()>16384 ||
       next_owner.empty() || next_owner.size()>128 || next_id.empty() || next_id.size()>128 ||
       !next_generation || !next_sequence || !next_started || next_started>next_stamp ||
       next_stamp>now.toNSec() || next_until<now.toNSec() || next_until<=next_stamp ||
       next_until-next_stamp>2000000000ULL ||
       b.get<std::string>("owner_session")!=next_owner || b.get<std::string>("context_id")!=next_id ||
       b.get<std::uint64_t>("generation")!=next_generation ||
       b.get<std::uint64_t>("started_ns")!=next_started ||
       b.get<std::string>("goal_request_id")!=e.get<std::string>("goal_request_id"))
      throw std::runtime_error("invalid_task_context");
    geometry_msgs::PoseStamped pose;pose.header.frame_id=b.get<std::string>("frame_id");
    if(pose.header.frame_id.empty() || b.get<std::string>("target.frame_id")!=pose.header.frame_id)
      throw std::runtime_error("invalid_task_frame");
    const auto vector=[&](const std::string& key,std::size_t size) {
      std::vector<double> values;
      for(const auto& x:b.get_child(key)) {const double v=x.second.get_value<double>();
        if(!std::isfinite(v))throw std::runtime_error("nonfinite_task_geometry");values.push_back(v);}
      if(values.size()!=size)throw std::runtime_error("invalid_task_geometry");return values;
    };
    const auto p=vector("target.position",3),q=vector("target.orientation",4);
    const auto lo=vector("region_min",3),hi=vector("region_max",3);
    for(int j=0;j<3;++j)if(lo[j]>=hi[j])throw std::runtime_error("invalid_task_region");
    const double norm=q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3];
    if(std::abs(norm-1.)>1e-8)throw std::runtime_error("invalid_task_quaternion");
    pose.pose.position.x=p[0];pose.pose.position.y=p[1];pose.pose.position.z=p[2];
    pose.pose.orientation.x=q[0];pose.pose.orientation.y=q[1];
    pose.pose.orientation.z=q[2];pose.pose.orientation.w=q[3];
    const bool same=owner==next_owner && generation==next_generation && id==next_id;
    const bool expired=until && (now.toNSec()>until || now.toNSec()<stamp);
    if(expired || (last_now && now.toNSec()<last_now))closed=true;
    if(!owner.empty()) {
      if(next_stamp<stamp)throw std::runtime_error("task_event_regressed");
      if(next_owner==owner) {
        if(next_sequence<=sequence || next_generation<generation ||
           (next_generation==generation && (!same || incoming!=binding || ((closed||expired)&&next_active))))
          throw std::runtime_error("task_epoch_closed_or_mutated");
      } else if(owners.count(next_owner) || next_stamp<=stamp)
        throw std::runtime_error("old_task_owner");
    }
    if(!owners.count(next_owner) && owners.size()>=16)throw std::runtime_error("task_owner_capacity");
    const bool changed=!same || active!=next_active;
    binding=incoming;owner=next_owner;id=next_id;goal=b.get<std::string>("goal_request_id");
    generation=next_generation;sequence=next_sequence;stamp=next_stamp;until=next_until;started=next_started;
    active=next_active;closed=!active;target=pose;owners.insert(owner);
    last_now=now.toNSec();
    return changed;
  }
};

struct TaskPlanningEvent {
  boost::property_tree::ptree payload;
  std::string source,kind,binding;
  std::uint64_t sequence=0,stamp=0,until=0;
};

inline std::string taskControlRecoveryTarget(const boost::property_tree::ptree& p) {
  const auto value=p.get_optional<std::string>("recovery_target_sha256");
  if((p.size()!=8 && !(p.size()==9 && value)) ||
     (value && !std::regex_match(*value,std::regex("([0-9a-f]{64})?"))))
    throw std::runtime_error("invalid_control_recovery_target");
  return value ? *value : std::string();
}

struct TaskPlanningEventReceiver {
  std::map<std::pair<std::string,std::string>,std::uint64_t> sequences;
  std::set<std::string> sessions;
  TaskPlanningEvent parse(const std::string& raw,const TaskObservationContext& context,
                          const ros::Time& now,const std::string& kind) const {
    using boost::property_tree::ptree;
    if(raw.size()>65536)throw std::runtime_error("task_event_too_large");
    ptree tree;std::istringstream input(raw);read_json(input,tree);
    TaskPlanningEvent event;
    event.kind=tree.get<std::string>("kind");event.source=tree.get<std::string>("source_session");
    event.sequence=tree.get<std::uint64_t>("event_seq");event.stamp=tree.get<std::uint64_t>("stamp_ns");
    event.until=tree.get<std::uint64_t>("valid_until_ns");event.binding=tree.get<std::string>("task_context");
    event.payload=tree.get_child("payload");
    if(tree.size()!=8 || tree.get<int>("schema")!=1 || event.kind!=kind ||
       !std::regex_match(event.source,std::regex("[A-Za-z0-9_.:-]{1,128}")) ||
       !event.sequence || event.sequence>9007199254740991ULL || !event.stamp ||
       event.stamp<context.started || event.stamp>now.toNSec() || event.until<now.toNSec() ||
       event.until<event.stamp || event.until-event.stamp>250000000ULL ||
       !context.current(event.binding,now))throw std::runtime_error("invalid_task_event");
    const auto key=std::make_pair(event.source,event.kind);const auto previous=sequences.find(key);
    if(previous!=sequences.end() && event.sequence<=previous->second)
      throw std::runtime_error("replayed_task_event");
    if((!sessions.count(event.source) && sessions.size()>=16) ||
       (previous==sequences.end() && sequences.size()>=128))throw std::runtime_error("task_event_capacity");
    return event;
  }
  void remember(const TaskPlanningEvent& event) {
    sessions.insert(event.source);sequences[{event.source,event.kind}]=event.sequence;
  }
};

inline std::string taskJsonQuote(const std::string& value) {
  std::ostringstream out;out<<'"';
  for(unsigned char c:value) {
    if(c=='"' || c=='\\')out<<'\\'<<c;
    else if(c<32)out<<"\\u"<<std::hex<<std::setfill('0')<<std::setw(4)<<static_cast<int>(c)<<std::dec;
    else out<<c;
  }
  out<<'"';return out.str();
}

inline std::string taskPlanningStatus(const std::string& name,const std::string& text,
    const TaskObservationContext& context,const std::string& source,std::uint64_t sequence,
    const ros::Time& now,const std::string& regime) {
  if(!context.current(context.binding,now))return {};
  std::ostringstream out;out<<"{\"event_seq\":"<<sequence
    <<",\"kind\":\"planner_status\",\"payload\":{\"name\":"<<taskJsonQuote(name)
    <<",\"task_regime\":"<<taskJsonQuote(regime)<<",\"text\":"<<taskJsonQuote(text)
    <<"},\"schema\":1,\"source_session\":"<<taskJsonQuote(source)
    <<",\"stamp_ns\":"<<now.toNSec()<<",\"task_context\":"<<taskJsonQuote(context.binding)
    <<",\"valid_until_ns\":"<<std::min<std::uint64_t>(context.until,now.toNSec()+250000000ULL)<<'}';
  const auto result=out.str();return result.size()<=65536 ? result : std::string();
}
}
