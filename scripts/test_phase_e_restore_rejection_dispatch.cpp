#include "planner_diagnostic_access.hpp"
#include <boost/property_tree/json_parser.hpp>
#include <cassert>
#include <fstream>
#include <iostream>
using Planner=egocentric_arm_planner::LocalSparseSCPPlanner;
using boost::property_tree::ptree;
Eigen::VectorXd vec(const ptree& t){Eigen::VectorXd v(t.size());int j=0;for(const auto& x:t)v[j++]=x.second.get_value<double>();return v;}
int main(int argc,char** argv){
 ros::init(argc,argv,"restore_dispatch_test",ros::init_options::NoSigintHandler);
 try {
  if(argc!=3)return 2;ros::NodeHandle nh,pnh("~");Planner p;
  if(!p.initialize(nh,pnh))throw std::runtime_error("initialize failed");p.stopWorker();p.timer_.stop();
  ptree fixture;boost::property_tree::read_json(argv[1],fixture);auto target=vec(fixture.get_child("target"));
  p.repair_mode_=true;p.probe_mode_=false;p.repair_observation_phase_=true;
  p.candidate_replacement_enabled_=true;p.latest_vbc_obligation_id_=9;
  p.latest_vbc_obligation_points_={Eigen::Vector3d(.25,-.05,.95)};
  p.latest_single_waypoint_active_=true;p.has_single_waypoint_q_=true;
  p.latest_single_waypoint_q_=target;p.latest_single_waypoint_joint_mask_=Eigen::VectorXd::Ones(7);
  std::ofstream out(argv[2]);out<<'[';
  for(int group=0;group<3;++group){
   p.mode_epoch_=20+group;p.latest_observation_token_="care_obs_v1_9_dispatch_"+std::to_string(group);
   p.selectRepairTargetLocked();
   assert(!p.safe_frontier_recovery_.active && p.safe_frontier_recovery_.attempts==0);
   p.geometric_restore_scope_=std::to_string(p.mode_epoch_)+":"+p.safe_frontier_recovery_.key;
   const std::string reason=group==0?"geometric_restore_certificate_unavailable":"geometric_restore_path_counterexample";
   for(int i=0;i<12;++i){
    const unsigned long long raw=10000+100*group+i;
    assert(p.final_vbc_no_progress_.remember(raw,p.mode_epoch_+(group==2?1:0),p.plan_sequence_));
    p.safe_frontier_recovery_.remember(raw,.02);
    p.safe_frontier_recovery_.detour_direction=Eigen::VectorXd::Unit(7,(i/2)%7);
    std_msgs::StringPtr msg(new std_msgs::String);
    msg->data="result=protocol_invalid safety_gate=vbc_protocol committed=0 raw_candidate_stamp_ns="+
      std::to_string(raw)+" audited_trajectory_stamp_ns="+std::to_string(raw+1000)+" vbc_protocol_reason="+reason;
    p.finalVerificationCallback(msg);
    const int before=p.safe_frontier_recovery_.attempts;
    p.finalVerificationCallback(msg); // duplicate cannot consume another attempt
    assert(p.safe_frontier_recovery_.attempts==before);
   }
   if(group)out<<',';
   out<<"{\"group\":"<<group<<",\"reason\":\""<<reason<<"\",\"active\":"<<(p.safe_frontier_recovery_.active?"true":"false")
      <<",\"attempts\":"<<p.safe_frontier_recovery_.attempts<<",\"blocked\":"<<(p.safe_frontier_recovery_.blocked()?"true":"false")
      <<",\"failed_directions\":"<<p.safe_frontier_recovery_.failed_directions.size()
      <<",\"plan_requested\":"<<(p.plan_requested_?"true":"false")<<'}';
   if(group!=1)assert(!p.safe_frontier_recovery_.active && p.safe_frontier_recovery_.attempts==0);
  }
  out<<"]\n";ros::shutdown();return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';ros::shutdown();return 1;}
}
