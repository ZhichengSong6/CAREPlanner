#define main resource_dispatch_main
#include "test_phase_e_proof_resource_dispatch.cpp"
#undef main
#include <thread>
#include <chrono>
#ifndef HANDOFF_TEST_FUNCTION
#define HANDOFF_TEST_FUNCTION main
#endif
int HANDOFF_TEST_FUNCTION(int argc,char** argv) {
  ros::init(argc,argv,"restore_dispatch_test",ros::init_options::NoSigintHandler);
  ros::NodeHandle nh,pnh("~");
  if(resource_dispatch_main(argc,argv))return 2;
  Planner p;assert(p.initialize(nh,pnh));p.stopWorker();p.timer_.stop();
  ptree fixture;boost::property_tree::read_json(argv[1],fixture);auto target=vec(fixture.get_child("target"));
  p.repair_mode_=true;p.probe_mode_=false;p.repair_observation_phase_=true;
  p.candidate_replacement_enabled_=true;p.initial_shell_execution_margin_m_=.01;
  p.latest_vbc_obligation_id_=9;p.latest_vbc_obligation_points_={Eigen::Vector3d(.25,-.05,.95)};
  p.latest_single_waypoint_active_=true;p.has_single_waypoint_q_=true;
  p.latest_single_waypoint_q_=target;p.latest_single_waypoint_joint_mask_=Eigen::VectorXd::Ones(7);
  std::ofstream out(std::string(argv[2])+".handoff.json");out<<'[';
  for(int group=0;group<12;++group) {
    p.mode_epoch_=500+group;p.latest_observation_token_="care_obs_v1_handoff_"+std::to_string(group);
    p.candidate_replacement_trigger_pending_=p.candidate_replacement_pending_=p.candidate_replacement_retry_pending_=false;
    p.state_evidence_retry_pending_.clear();p.state_residual_retry_raw_=0;p.state_evidence_retry_used_.clear();
    p.selectRepairTargetLocked();p.last_query_ros_time_=ros::Time::now();p.current_query_stamp_=p.last_query_ros_time_;
    const auto key=std::to_string(p.mode_epoch_)+":"+p.safe_frontier_recovery_.key;
    if(group==6)p.state_evidence_retry_used_[key]=true;
    unsigned long long raw=0,audit=0;std_msgs::StringPtr last;
    for(int i=0;i<((group==8 || group>=9)?1:2);++i) {
      raw=300000+group*100+i;audit=raw+1000;
      assert(p.final_vbc_no_progress_.remember(raw,p.mode_epoch_,p.plan_sequence_));
      p.safe_frontier_recovery_.remember(raw,.02);
      p.safe_frontier_recovery_.detour_direction=Eigen::VectorXd::Unit(7,i);
      const std::string body="{\"schema\":1,\"kind\":\"handoff_residual_insufficient\",\"trajectory_stamp_ns\":\""+
        std::to_string(audit+(group==1?1:0))+"\",\"raw_candidate_stamp_ns\":\""+std::to_string(raw)+
        "\",\"mode_epoch\":\""+std::to_string(p.mode_epoch_+(group==2?1:0))+"\",\"observation_token\":\""+
        (group==3?"care_obs_v1_wrong":p.latest_observation_token_)+
        "\",\"stage\":\"map_path_proof\",\"floor_m\":0.02,\"point\":[0.25,0,0.95],\"ordinary_unknown\":true,"
        "\"interval_index\":3,\"required_residual_m\":"+(group==4?"0.00001":"0.0004")+
        ",\"certified_residual_m\":0.00004,\"requires_reconstruction\":"+((group==5 || group==6 || group==8)?"true":"false")+
        (group==9?",\"initial_state_changed\":true":group==10?",\"initial_state_changed\":\"invalid\"":
         group==11?",\"initial_state_changed\":true,\"initial_state_changed\":false":"")+"}";
      last.reset(new std_msgs::String);last->data="result=timeout safety_gate=state_evidence committed=0 failure_kind=geometric_certificate_state_changed raw_candidate_stamp_ns="+
        std::to_string(raw)+" audited_trajectory_stamp_ns="+std::to_string(audit)+" handoff_residual_evidence="+body;
      p.finalVerificationCallback(last);const int attempts=p.safe_frontier_recovery_.attempts;
      const auto trigger=p.candidate_replacement_trigger_count_;p.finalVerificationCallback(last);
      assert(p.safe_frontier_recovery_.attempts==attempts && p.candidate_replacement_trigger_count_==trigger);
      if((group>=1 && group<=4) || group>=10)assert(p.final_vbc_no_progress_.pending.count(raw));
      if(group==5 && i==0) {
        assert(!p.state_evidence_retry_pending_.empty() && p.state_residual_retry_raw_==raw);
        p.state_evidence_retry_deadline_=ros::WallTime::now()-ros::WallDuration(.01);
        p.timerCallback(ros::TimerEvent{});
        assert(p.state_evidence_retry_pending_.empty() && p.plan_requested_ && p.safe_frontier_recovery_.rejected==1);
      }
    }
    if(group==8) {
      assert(!p.state_evidence_retry_pending_.empty());
      // Another callback cannot bypass the matching state wait or start work.
      const auto deadline=p.state_evidence_retry_deadline_;
      p.requestPlanLocked("repair_trajectory_complete");
      assert(p.plan_requested_ && p.state_evidence_retry_deadline_==deadline);
      assert(!p.startPlan() && !p.plan_running_);
      p.latest_executed_command_=Eigen::VectorXd::Zero(7);
      for(int i=0;i<3;++i) {
        sensor_msgs::JointStatePtr state(new sensor_msgs::JointState);state->name=p.joint_names_;
        state->position.assign(target.data(),target.data()+7);state->velocity.assign(7,0.);
        state->header.stamp=ros::Time::now();p.latest_executed_command_received_=state->header.stamp;
        p.jointStateCallback(state);if(i<2)std::this_thread::sleep_for(std::chrono::milliseconds(60));
      }
      assert(p.state_evidence_retry_pending_.empty() && p.plan_requested_ && !p.state_residual_retry_raw_);
      assert(!p.safe_frontier_recovery_.pending.count(raw));
    } else if(group==9) {
      if(group==9) {
        assert(!p.state_evidence_retry_pending_.empty());
        p.state_evidence_retry_deadline_=ros::WallTime::now()-ros::WallDuration(.01);
        p.timerCallback(ros::TimerEvent{});
      }
      assert(p.plan_requested_ && !p.candidate_replacement_trigger_pending_);
      assert(p.safe_frontier_recovery_.attempts==0 && p.safe_frontier_recovery_.rejected==1);
      assert(p.state_evidence_retry_used_.count(key));
    } else if((group>=1 && group<=4) || group>=10)assert(!p.candidate_replacement_trigger_pending_);
    else {
      assert(p.candidate_replacement_trigger_pending_ && p.candidate_replacement_trigger_raw_==raw);
      if(group==0)assert(p.restore_path_guard_points_.size()==1 && p.restore_path_guard_points_[0].proposal_reserve_m==.0004);
      std_msgs::StringPtr reserve(new std_msgs::String);
      reserve->data="action=reserve request_id="+std::to_string(group+100)+" mode_epoch="+std::to_string(p.mode_epoch_)+
        " query_stamp_ns="+std::to_string(p.current_query_stamp_.toNSec())+" observation_token="+p.latest_observation_token_+
        " obligation_id=9 reason=geometric_restore_handoff_residual_insufficient trigger_raw_candidate_stamp_ns="+std::to_string(raw);
      if(group==7)p.safe_frontier_recovery_.segments=6;
      p.candidateReplacementCallback(reserve);
      if(p.candidate_replacement_pending_!=(group!=7))std::cerr<<"handoff reserve failed group="<<group<<" attempts="<<p.safe_frontier_recovery_.attempts<<" key="<<p.safe_frontier_recovery_.key<<"\n";
      assert(p.candidate_replacement_pending_==(group!=7));
      assert(p.safe_frontier_recovery_.attempts==(group==7?1:2));
    }
    if(group)out<<',';out<<"{\"group\":"<<group<<",\"passed\":true}";
  }
  out<<']';std::cout<<"PASS 12 handoff callback groups plus10 existing resource groups\n";
  return 0;
}
