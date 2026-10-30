#include "planner_diagnostic_access.hpp"
#include <boost/property_tree/json_parser.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
using Planner=egocentric_arm_planner::LocalSparseSCPPlanner;
using boost::property_tree::ptree;

void require(bool value,const char* name) {
  if(!value)throw std::runtime_error(name);
}
std_msgs::String::Ptr rejection(unsigned long long raw=33349000000ULL) {
  auto msg=boost::make_shared<std_msgs::String>();
  msg->data="result=unsafe safety_gate=vbc committed=0 raw_candidate_stamp_ns="+
      std::to_string(raw)+" audited_trajectory_stamp_ns="+std::to_string(raw+4000000)+
      " vbc_evidence={\"violations\":[{\"point\":[-0.15,-0.2,0.85]}]}";
  return msg;
}
struct Fixture {
  Planner p;
  Fixture(ros::NodeHandle& nh) {
    p.dof_=7;p.mode_epoch_=11;p.plan_sequence_=299;
    p.joint_names_={"joint1","joint2","joint3","joint4","wrist_joint1","wrist_joint2","wrist_joint3"};
    p.repair_mode_=p.plan_repair_mode_=p.repair_observation_phase_=true;
    p.latest_single_waypoint_active_=p.has_single_waypoint_q_=true;
    p.latest_single_waypoint_q_=Eigen::VectorXd::Constant(7,.5);
    p.latest_single_waypoint_joint_mask_=Eigen::VectorXd::Ones(7);
    p.latest_vbc_obligation_id_=13;p.latest_vbc_obligation_points_={Eigen::Vector3d(-.05,-.25,.85)};
    p.latest_observation_token_="care_obs_v1_13_fixed";
    p.candidate_replacement_enabled_=p.safe_frontier_recovery_enabled_=true;
    p.visibility_obligation_cdf_margin_base_=.05;p.vbc_feedback_cdf_margin_step_=.15;
    p.vbc_feedback_cdf_margin_max_=.3;
    p.summary_pub_=nh.advertise<std_msgs::String>("/continuation_test/summary",100);
    p.candidate_replacement_trigger_pub_=nh.advertise<std_msgs::String>("/continuation_test/trigger",100);
    p.selectRepairTargetLocked();
    p.final_vbc_no_progress_.progress.observe(Eigen::VectorXd::Zero(7));
    p.repair_no_progress_.observe(Eigen::VectorXd::Zero(7));
    p.safe_frontier_recovery_.active=true;p.safe_frontier_recovery_.attempts=5;
    p.plan_running_=p.waiting_for_cdf_=true;
    p.pending_batch_=boost::make_shared<care_collision_cdf::CollisionCDFConstraintBatch>();
  }
  void remember(unsigned long long raw=33349000000ULL) {
    require(p.final_vbc_no_progress_.remember(raw,p.mode_epoch_,p.plan_sequence_),"remember candidate");
    p.safe_frontier_recovery_.remember(raw,.04);
  }
  void motion() {
    auto q=boost::make_shared<sensor_msgs::JointState>();q->header.stamp=ros::Time::now();
    q->name=p.joint_names_;q->position.assign(7,.02);q->velocity.assign(7,0.);
    p.jointStateCallback(q);
  }
  void external() {
    auto m=boost::make_shared<std_msgs::Bool>();m->data=true;p.replanRequestCallback(m);
  }
};

int main(int argc,char** argv) {
  ros::init(argc,argv,"final_vbc_continuation_test",ros::init_options::NoSigintHandler);
  try {
    require(argc==3,"output and before/fixed selector required");
    const bool fixed=std::string(argv[2])=="fixed";
    ros::NodeHandle nh;ptree report,checks;
    auto record=[&](const std::string& name,const Fixture& f) {
      ptree row;row.put("name",name);row.put("requested",f.p.plan_requested_);
      row.put("reason",f.p.plan_request_reason_);row.put("attempts",f.p.safe_frontier_recovery_.attempts);
      row.put("failures",f.p.final_vbc_no_progress_.progress.failures());checks.push_back({"",row});
    };
    for(int order=0;order<3;++order) {
      Fixture f(nh);f.remember();f.motion();
      if(order==1)f.external();
      if(order==2) {
        f.external();require(f.p.plan_requested_,"pre-rejection external request queued");
        // A worker has consumed that request and is waiting for its CDF reply.
        f.p.plan_requested_=false;f.p.plan_running_=f.p.waiting_for_cdf_=true;
      }
      f.p.finalVerificationCallback(rejection());
      require(f.p.plan_requested_==(fixed || order==1),"stale feedback must schedule unless old callback cancels consumed request");
      require(!f.p.plan_running_ && !f.p.waiting_for_cdf_ && !f.p.pending_batch_,"rejected feedback invalidates outstanding solve");
      require(f.p.final_vbc_no_progress_.progress.failures()==0,"old progress failure not charged to current progress");
      require(f.p.repair_unknown_witnesses_.size()==1 && f.p.repair_unknown_witnesses_[0].safety_margin==.2,"point safety guard retained");
      require(f.p.safe_frontier_recovery_.attempts==5,"no refunded or extra direction");
      const auto seq=f.p.plan_sequence_;f.p.finalVerificationCallback(rejection());
      require(f.p.plan_sequence_==seq && f.p.repair_unknown_witnesses_[0].safety_margin==.2,"duplicate causes no second mutation");
      record("stale_progress_external_order_"+std::to_string(order),f);
      f.external();require(f.p.plan_requested_,"later external request can coalesce");
      require(f.p.safe_frontier_recovery_.attempts==5,"external request never buys direction");
    }
    {
      Fixture f(nh);f.remember();f.p.finalVerificationCallback(rejection());
      require(f.p.plan_requested_ && f.p.final_vbc_no_progress_.progress.failures()==1,"fresh rejection preserves normal retry accounting");
      record("fresh_rejection",f);
    }
    {
      Fixture f(nh);f.remember();f.motion();f.p.safe_frontier_recovery_.rejected=1;
      f.p.finalVerificationCallback(rejection());
      require(!f.p.plan_requested_ && f.p.safe_frontier_recovery_.attempts==6,"second rejection exhausts original fifth direction");
      if(fixed)require(f.p.plan_request_reason_=="repair_candidate_replacement_exhausted_hold","exhaustion explicitly reported");
      f.external();require(!f.p.plan_requested_ && f.p.safe_frontier_recovery_.attempts==6,"external pulse cannot reopen exhausted ledger");
      record("stale_progress_exhausted_direction",f);
    }
    {
      Fixture f(nh);f.remember();f.motion();f.p.execution_response_monitor_.blocked=true;
      f.p.finalVerificationCallback(rejection());require(!f.p.plan_requested_,"actuator hold remains closed");
      if(fixed)require(f.p.plan_request_reason_=="execution_command_no_response_hold","actuator hold explicitly reported");
      record("actuator_hold",f);
    }
    {
      Fixture f(nh);f.remember();f.motion();f.p.candidate_replacement_pending_=true;
      f.p.finalVerificationCallback(rejection());require(!f.p.plan_requested_,"pending replacement owns continuation");
      if(fixed)require(f.p.plan_request_reason_=="repair_candidate_replacement_wait","replacement wait explicitly reported");
      record("replacement_pending",f);
    }
    {
      Fixture f(nh);f.remember();f.p.final_vbc_no_progress_.progress.counts[f.p.repairTargetKeyLocked()]=4;
      f.p.finalVerificationCallback(rejection());require(!f.p.plan_requested_ && f.p.final_vbc_no_progress_.progress.failures()==5,"fifth fresh failure holds");
      f.external();require(!f.p.plan_requested_,"Bool pulse cannot reopen failure hold");record("failure_budget_exhausted",f);
    }
    {
      Fixture f(nh);f.remember();f.p.latest_single_waypoint_q_[0]=.6;f.p.selectRepairTargetLocked();
      f.p.finalVerificationCallback(rejection());
      require(f.p.final_vbc_no_progress_.progress.failures()==0,"old target failure not charged to new target");
      require(f.p.plan_requested_==fixed,"same-mode old target point feedback re-queries current target");
      require(!f.p.candidate_replacement_trigger_pending_,"old target cannot trigger current replacement");
      record("target_revision_changed",f);
    }
    for(const auto& kind:{"unknown_raw","old_mode","probe","safe","wrong_gate","malformed"}) {
      Fixture f(nh);f.remember();auto m=rejection();
      const std::string k=kind;
      if(k=="unknown_raw")m=rejection(33350000000ULL);
      if(k=="old_mode")++f.p.mode_epoch_;
      if(k=="probe")f.p.probe_mode_=true;
      if(k=="safe")m->data.replace(7,6,"safe");
      if(k=="wrong_gate")m->data.replace(m->data.find("safety_gate=vbc"),15,"safety_gate=gcdf");
      if(k=="malformed")m->data+=" committed=0";
      f.p.finalVerificationCallback(m);
      require(f.p.repair_unknown_witnesses_.empty() && !f.p.plan_requested_,"foreign/positive/malformed outcome has no rejection side effects");
      record(k,f);
    }
    report.put("status","PASS");report.put("variant",fixed?"fixed":"before");report.add_child("checks",checks);
    boost::property_tree::write_json(argv[1],report);
    ros::shutdown();return 0;
  } catch(const std::exception& e) {
    std::cerr<<"FAIL "<<e.what()<<'\n';ros::shutdown();return 1;
  }
}
