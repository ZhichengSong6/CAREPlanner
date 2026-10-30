#define HANDOFF_TEST_FUNCTION existing_handoff_main
#include "test_phase_e_handoff_dispatch.cpp"

int main(int argc,char** argv) {
  if(argc!=4)return 2;
  if(existing_handoff_main(3,argv))return 3;
  ros::NodeHandle nh,pnh("~");
  ptree fixture;boost::property_tree::read_json(argv[1],fixture);
  const auto target=vec(fixture.get_child("target"));
  ptree records;boost::property_tree::read_json(argv[3],records);
  const auto configure=[&](Planner& p,unsigned long long epoch,const std::string& token) {
    assert(p.initialize(nh,pnh));p.stopWorker();p.timer_.stop();
    p.repair_mode_=true;p.probe_mode_=false;p.repair_observation_phase_=true;
    p.candidate_replacement_enabled_=true;p.initial_shell_execution_margin_m_=.01;
    p.latest_vbc_obligation_id_=3;p.latest_vbc_obligation_points_={Eigen::Vector3d(.25,0.,.9)};
    p.latest_single_waypoint_active_=true;p.has_single_waypoint_q_=true;
    p.latest_single_waypoint_q_=target;p.latest_single_waypoint_joint_mask_=Eigen::VectorXd::Ones(7);
    p.mode_epoch_=epoch;p.latest_observation_token_=token;p.selectRepairTargetLocked();
    p.last_query_ros_time_=ros::Time::now();p.current_query_stamp_=p.last_query_ros_time_;
  };
  const auto deliver=[&](Planner& p,unsigned long long raw,unsigned long long audit,const std::string& evidence) {
    assert(p.final_vbc_no_progress_.remember(raw,p.mode_epoch_,p.plan_sequence_));
    p.safe_frontier_recovery_.remember(raw,.02);
    std_msgs::StringPtr msg(new std_msgs::String);
    msg->data="result=timeout safety_gate=state_evidence committed=0 failure_kind=geometric_certificate_state_changed raw_candidate_stamp_ns="+
        std::to_string(raw)+" audited_trajectory_stamp_ns="+std::to_string(audit)+" handoff_residual_evidence="+evidence;
    p.finalVerificationCallback(msg);
    const auto attempts=p.safe_frontier_recovery_.attempts;const auto rejected=p.safe_frontier_recovery_.rejected;
    p.finalVerificationCallback(msg);
    assert(p.safe_frontier_recovery_.attempts==attempts && p.safe_frontier_recovery_.rejected==rejected);
  };
  int replayed=0;
  for(const auto& entry:records) {
    const auto& f=entry.second;Planner p;
    configure(p,f.get<unsigned long long>("epoch"),f.get<std::string>("token"));
    const auto key=std::to_string(p.mode_epoch_)+":"+p.safe_frontier_recovery_.key;
    p.state_evidence_retry_used_[key]=true;
    deliver(p,f.get<unsigned long long>("raw"),f.get<unsigned long long>("audit"),f.get<std::string>("evidence"));
    assert(p.plan_requested_ && p.state_evidence_retry_pending_.empty());
    assert(p.safe_frontier_recovery_.rejected==1 && p.safe_frontier_recovery_.attempts==0);
    assert(p.restore_path_guard_points_.size()==1 && p.repair_unknown_witnesses_.empty());
    assert(p.restore_path_guard_points_[0].proposal_reserve_m==f.get<double>("required"));
    assert(p.state_evidence_retry_used_.size()==1);++replayed;
  }
  assert(replayed==6);
  // Permanent failure is finite: no new state wait, no invented context and
  // no retry past the original five directions. No motion is published here.
  Planner p;configure(p,700,"care_obs_v1_3_bounded");p.candidate_replacement_enabled_=false;
  p.selectRepairTargetLocked();
  const auto key=std::to_string(p.mode_epoch_)+":"+p.safe_frontier_recovery_.key;
  p.state_evidence_retry_used_[key]=true;
  for(int i=0;i<12;++i) {
    const unsigned long long raw=900000+i,audit=raw+1000;
    const auto evidence="{\"schema\":1,\"kind\":\"handoff_residual_insufficient\",\"trajectory_stamp_ns\":\""+
      std::to_string(audit)+"\",\"raw_candidate_stamp_ns\":\""+std::to_string(raw)+
      "\",\"mode_epoch\":\"700\",\"observation_token\":\"care_obs_v1_3_bounded\",\"stage\":\"map_path_proof\","
      "\"floor_m\":0.02,\"point\":[0.25,-0.05,0.9],\"ordinary_unknown\":true,\"interval_index\":5,"
      "\"required_residual_m\":0.001,\"certified_residual_m\":0.00006,\"requires_reconstruction\":false,\"initial_state_changed\":true}";
    p.plan_requested_=false;deliver(p,raw,audit,evidence);
    assert(p.state_evidence_retry_used_.size()==1 && p.state_evidence_retry_pending_.empty());
    if(i<11)assert(p.plan_requested_);
  }
  assert(p.safe_frontier_recovery_.blocked() && !p.plan_requested_);
  assert(p.safe_frontier_recovery_.attempts==6); // five directions exhausted; sixth is the existing terminal sentinel
  std::cout<<"PASS six exact recorded residuals, duplicates, finite five-direction exhaustion; inherited22 callback groups\n";
}
