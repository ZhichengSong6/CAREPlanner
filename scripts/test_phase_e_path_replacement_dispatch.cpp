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
  // Exact rejected identities dispatch to the original reservation callback.
  for(int group=0;group<10;++group){
   p.mode_epoch_=120+group;p.latest_observation_token_="care_obs_v1_9_path_"+std::to_string(group);
   p.candidate_replacement_trigger_pending_=false;p.candidate_replacement_pending_=false;
   p.candidate_replacement_retry_pending_=false;
   p.selectRepairTargetLocked();
   assert(!p.safe_frontier_recovery_.active && p.safe_frontier_recovery_.attempts==0);
   p.last_query_ros_time_=ros::Time::now();p.current_query_stamp_=p.last_query_ros_time_;
   if(group>=7){for(int j=0;j<(group==7?3:4);++j)p.safe_frontier_recovery_.advance();}
   const bool was_active=p.safe_frontier_recovery_.active;
   const int start=p.safe_frontier_recovery_.attempts;
   const auto count=p.candidate_replacement_trigger_count_;
   unsigned long long raw=0,audit=0;
   for(int i=0;i<2;++i){
    raw=100000+100*group+i;audit=raw+1000;
    assert(p.final_vbc_no_progress_.remember(raw,p.mode_epoch_+(group==2?1:0),p.plan_sequence_));
    p.safe_frontier_recovery_.remember(raw,.02);
    p.safe_frontier_recovery_.detour_direction=Eigen::VectorXd::Unit(7,i);
    std_msgs::StringPtr msg(new std_msgs::String);
    msg->data="result=protocol_invalid safety_gate=vbc_protocol committed=0 raw_candidate_stamp_ns="+
      std::to_string(raw)+" audited_trajectory_stamp_ns="+std::to_string(audit)+" vbc_protocol_reason="+
      (group==0?"geometric_restore_certificate_unavailable":"geometric_restore_path_counterexample");
    p.finalVerificationCallback(msg);const auto before=p.candidate_replacement_trigger_count_;
    p.finalVerificationCallback(msg);assert(before==p.candidate_replacement_trigger_count_);
   }
   const bool triggered=p.candidate_replacement_trigger_count_>count;
   bool granted=false;
   if(group==0 || group==2){assert(!triggered && !p.safe_frontier_recovery_.active);}
   else {
    assert(triggered && p.candidate_replacement_trigger_pending_ && !p.plan_requested_);
    assert(p.candidate_replacement_trigger_raw_==raw);
    assert(p.candidate_replacement_trigger_direction_advanced_==was_active);
    const auto epoch=p.mode_epoch_;const auto query=p.current_query_stamp_.toNSec();
    const auto token=p.latest_observation_token_;
    std_msgs::StringPtr reserve(new std_msgs::String);
    reserve->data="action=reserve request_id="+std::to_string(group+1)+
      " mode_epoch="+std::to_string(epoch+(group==5?1:0))+" query_stamp_ns="+std::to_string(query)+
      " observation_token="+(group==4?"care_obs_v1_stale":token)+" obligation_id=9 reason=geometric_restore_path_counterexample"
      " trigger_raw_candidate_stamp_ns="+std::to_string(raw+(group==3?1:0));
    if(group==6)p.last_query_ros_time_=ros::Time::now()-ros::Duration(1.01);
    if(group==9)p.safe_frontier_recovery_.segments=6;
    p.candidateReplacementCallback(reserve);granted=p.candidate_replacement_pending_;
    const bool expected=group==1 || group==7 || group==8;
    assert(granted==expected);
    if(granted){
     assert(p.safe_frontier_recovery_.attempts==(was_active?start+1:2));
     assert(!p.candidate_replacement_trigger_pending_);
     const auto attempts=p.safe_frontier_recovery_.attempts;
     p.candidateReplacementCallback(reserve);assert(p.safe_frontier_recovery_.attempts==attempts);
     if(group==8){assert(attempts==5 && p.safe_frontier_recovery_.final_stage_replacement_used);
      assert(!p.safe_frontier_recovery_.reserveReplacementAfterVbc(true));}
     std_msgs::StringPtr finish(new std_msgs::String);
     finish->data="action=finish request_id="+std::to_string(group+1)+
       " new_token=none generation_result=no_geometric_candidate";
     p.candidateReplacementCallback(finish);assert(!p.candidate_replacement_pending_);
     assert(p.safe_frontier_recovery_.attempts==attempts); // no refund on generation failure
    }
   }
   if(group)out<<',';
   out<<"{\"group\":"<<group<<",\"triggered\":"<<triggered<<",\"granted\":"<<granted
      <<",\"attempts\":"<<p.safe_frontier_recovery_.attempts<<'}';
  }
  out<<"]\n";ros::shutdown();return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';ros::shutdown();return 1;}
}
