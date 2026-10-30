// Actual planner callbacks, native FK and immutable audit/commit identities.
#include "planner_diagnostic_access.hpp"
#include <boost/make_shared.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
using Planner=egocentric_arm_planner::LocalSparseSCPPlanner;
using Motion=trajectory_msgs::JointTrajectory;
using boost::property_tree::ptree;
#define CHECK(x) do {if(!(x))throw std::runtime_error(#x);}while(false)
std_msgs::StringConstPtr text(const std::string& s){auto p=boost::make_shared<std_msgs::String>();p->data=s;return p;}
std_msgs::BoolConstPtr boolean(bool b){auto p=boost::make_shared<std_msgs::Bool>();p->data=b;return p;}
std::vector<double> vector(const ptree& t){std::vector<double> v;for(const auto& x:t)v.push_back(x.second.get_value<double>());return v;}
int main(int argc,char** argv) {
  ros::init(argc,argv,"normal_terminal_retention_test",ros::init_options::NoSigintHandler);
  try {
    CHECK(argc==3);ptree f;boost::property_tree::read_json(argv[1],f);
    ros::NodeHandle nh,pnh("~");Planner p;CHECK(p.initialize(nh,pnh));p.stopWorker();p.timer_.stop();
    Motion fixture;fixture.header.frame_id="base_link";fixture.joint_names=p.joint_names_;
    for(const auto& v:f.get_child("curve")) {
      trajectory_msgs::JointTrajectoryPoint point;point.positions=vector(v.second.get_child("q"));point.velocities=vector(v.second.get_child("v"));
      point.accelerations=vector(v.second.get_child("a"));point.time_from_start=ros::Duration(v.second.get<double>("t"));fixture.points.push_back(point);
    }
    const auto pos=vector(f.get_child("goal_position")),quat=vector(f.get_child("goal_orientation_xyzw"));
    auto goal=boost::make_shared<geometry_msgs::PoseStamped>();goal->header.frame_id="base_link";
    goal->pose.position.x=pos[0];goal->pose.position.y=pos[1];goal->pose.position.z=pos[2];
    goal->pose.orientation.x=quat[0];goal->pose.orientation.y=quat[1];goal->pose.orientation.z=quat[2];goal->pose.orientation.w=quat[3];
    std::uint64_t raw=0,audit=0,exec=0;int index=0;Motion current;std::string outcome;std::vector<std::string> passed;
    auto tracker=[&](bool complete,bool aborted=false) {
      return text("active="+std::to_string(!complete)+" complete="+std::to_string(complete)+
          " execution_stamp_ns="+std::to_string(exec)+" execution_aborted="+std::to_string(aborted)+
          " phase_s="+(complete?"1.1":"0.8")+" source="+(complete?"trajectory_complete_hold":"active_trajectory"));
    };
    auto prepare=[&](bool certify=true) {
      const ros::Time now(100+10*index++,0);ros::Time::setNow(now);
      raw=now.toNSec()-30000000;audit=raw+10000000;exec=audit+10000000;
      p.terminal_retention_=egocentric_arm_planner::NormalTerminalRetention{};
      p.terminal_tracker_completed_=false;p.terminal_tracker_aborted_=false;
      p.repair_mode_=p.probe_mode_=false;p.normal_reference_refresh_pending_=false;
      p.vbc_execution_hold_=p.gcdf_execution_hold_=false;p.execution_response_monitor_={};
      p.mode_epoch_=p.plan_mode_epoch_=7;p.latest_execution_complete_=false;
      p.task_no_progress_.reset();p.normal_active_no_progress_.reset();p.normal_unknown_no_progress_.reset();
      p.has_reference_=true;p.latest_reference_=fixture;p.latest_reference_.header.stamp=now-ros::Duration(1.);
      p.terminalGoalCallback(goal);
      p.has_joint_state_=true;p.latest_joint_state_.name=p.joint_names_;p.latest_joint_state_.position=fixture.points.back().positions;
      p.latest_joint_state_.velocity.assign(7,0.);p.latest_joint_state_.header.stamp=now;p.latest_joint_state_received_=now;
      p.latest_executed_command_=Eigen::VectorXd::Zero(7);p.latest_executed_command_received_=now;
      p.latest_desired_command_=Eigen::VectorXd::Zero(7);p.latest_desired_command_received_=now;
      p.terminal_retention_.candidate(raw,p.terminalContextLocked());
      auto a=boost::make_shared<Motion>(fixture);a->header.stamp.fromNSec(audit);p.terminalAuditCallback(a);
      auto c=boost::make_shared<Motion>(fixture);c->header.stamp.fromNSec(exec);p.terminalCommitCallback(c);current=*c;
      outcome="result=safe safety_gate=vbc committed=1 verification_view=full_horizon raw_candidate_stamp_ns="+std::to_string(raw)+
          " audited_trajectory_stamp_ns="+std::to_string(audit)+" execution_stamp_ns="+std::to_string(exec);
      if(certify)p.finalVerificationCallback(text(outcome));
      p.executionSummaryCallback(tracker(false));p.plan_requested_=false;
    };
    prepare();CHECK(p.terminal_retention_.armed==exec);p.smoothReplanRequestCallback(boolean(true));CHECK(!p.plan_requested_);
    p.executionSummaryCallback(tracker(true));CHECK(!p.plan_requested_ && p.terminal_retention_.suppressed_count==2);
    p.executionSummaryCallback(text("active=0 complete=0 execution_stamp_ns=0 execution_aborted=0 phase_s=0 source=last_reference_hold"));
    CHECK(p.retainTerminalLocked());passed.push_back("smooth_and_complete_then_last_reference_hold");
    prepare();p.finalVerificationCallback(text(outcome));CHECK(p.terminal_retention_.arm_count==1);
    auto old=boost::make_shared<Motion>(current);old->header.stamp.fromNSec(exec-1);p.terminalCommitCallback(old);CHECK(p.terminal_retention_.armed==exec);
    p.terminal_retention_.certify(raw,audit,exec+1,p.terminalContextLocked());CHECK(!p.terminal_retention_.proofs.count(exec+1));passed.push_back("duplicate_and_old_identity_idempotent");
    prepare(false);CHECK(!p.terminal_retention_.armed);p.finalVerificationCallback(text("result=safe safety_gate=vbc committed=1 verification_view=full_horizon raw_candidate_stamp_ns="+std::to_string(raw+1)+" audited_trajectory_stamp_ns="+std::to_string(audit)+" execution_stamp_ns="+std::to_string(exec)));
    CHECK(!p.terminal_retention_.armed);passed.push_back("unowned_raw_outcome_cannot_arm");
    prepare(false);auto mismatch=boost::make_shared<Motion>(current);mismatch->points[0].positions[0]+=.001;
    p.terminal_retention_.commits[exec]=*mismatch;p.finalVerificationCallback(text(outcome));CHECK(!p.terminal_retention_.armed);passed.push_back("audit_commit_motion_mismatch");
    prepare(false);p.terminal_retention_.audits.clear();p.finalVerificationCallback(text(outcome));CHECK(!p.terminal_retention_.armed);
    auto delayed=boost::make_shared<Motion>(fixture);delayed->header.stamp.fromNSec(audit);p.terminalAuditCallback(delayed);CHECK(p.terminal_retention_.armed==exec);passed.push_back("independent_topic_delivery_order");
    prepare();p.latest_joint_state_.header.stamp-=ros::Duration(0,50000001);CHECK(!p.retainTerminalLocked());
    p.latest_joint_state_.header.stamp=ros::Time::now();p.finalVerificationCallback(text(outcome));CHECK(!p.terminal_retention_.armed);passed.push_back("stale_state_and_late_positive_cannot_rearm");
    prepare();p.terminal_tracker_received_-=ros::Duration(0,50000001);CHECK(!p.retainTerminalLocked());passed.push_back("stale_tracker");
    prepare();p.latest_executed_command_received_-=ros::Duration(0,50000001);CHECK(!p.retainTerminalLocked());passed.push_back("stale_actuator_evidence");
    prepare();p.replanRequestCallback(boolean(true));CHECK(p.plan_requested_ && !p.terminal_retention_.armed);passed.push_back("external_replan_remains_live");
    prepare();auto changed=boost::make_shared<geometry_msgs::PoseStamped>(*goal);changed->pose.position.x+=.001;p.terminalGoalCallback(changed);CHECK(!p.terminal_retention_.armed);passed.push_back("changed_goal");
    prepare();p.recoveryCallback(boolean(true));CHECK(!p.terminal_retention_.armed && p.repair_mode_);passed.push_back("enter_repair");
    prepare();p.probeActiveCallback(boolean(true));CHECK(!p.terminal_retention_.armed && p.probe_mode_);passed.push_back("enter_probe");
    prepare();p.executionSummaryCallback(tracker(false,true));CHECK(!p.terminal_retention_.armed);passed.push_back("tracker_abort");
    prepare();p.vbcExecutionHoldCallback(boolean(true));CHECK(!p.terminal_retention_.armed);p.vbcExecutionHoldCallback(boolean(false));p.finalVerificationCallback(text(outcome));CHECK(!p.terminal_retention_.armed);passed.push_back("vbc_hold_and_delayed_outcome");
    prepare();p.gcdfExecutionHoldCallback(boolean(true));CHECK(!p.terminal_retention_.armed);passed.push_back("gcdf_hold");
    prepare();auto replacement=boost::make_shared<Motion>(current);replacement->header.stamp.fromNSec(exec+1);p.terminalCommitCallback(replacement);CHECK(!p.terminal_retention_.armed);p.finalVerificationCallback(text(outcome));CHECK(!p.terminal_retention_.armed);passed.push_back("new_unqualified_execution");
    prepare();for(auto& x:p.latest_joint_state_.position)x+=.5;p.executionSummaryCallback(tracker(true));CHECK(!p.terminal_retention_.armed && p.plan_requested_);passed.push_back("completed_outside_goal_replans");
    prepare(false);auto fake=current;fake.points[fake.points.size()-2].velocities[0]=.1;p.terminal_retention_.commits[exec]=fake;
    auto fake_a=fake;fake_a.header.stamp.fromNSec(audit);p.terminal_retention_.audits[audit]=fake_a;p.finalVerificationCallback(text(outcome));CHECK(!p.terminal_retention_.armed);passed.push_back("fake_zero_marker_rejected");
    prepare();p.latest_joint_state_.header.stamp+=ros::Duration(.001);CHECK(!p.retainTerminalLocked());passed.push_back("future_measured_stamp");
    std::ofstream out(argv[2]);out<<"{\"status\":\"PASS\",\"checks\":[";
    for(std::size_t i=0;i<passed.size();++i){if(i)out<<',';out<<'"'<<passed[i]<<'"';}out<<"]}\n";
    std::cout<<"PASS "<<passed.size()<<" production callback controls\n";
  } catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
