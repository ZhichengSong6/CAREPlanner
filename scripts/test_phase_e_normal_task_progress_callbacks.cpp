#include "planner_diagnostic_access.hpp"
#include <boost/make_shared.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
using Planner=egocentric_arm_planner::LocalSparseSCPPlanner;
#define CHECK(x) do { if(!(x))throw std::runtime_error(#x); } while(false)
std_msgs::StringConstPtr message(const std::string& text) {
  auto m=boost::make_shared<std_msgs::String>();m->data=text;return m;
}
int main(int argc,char** argv) { try {
  ros::init(argc,argv,"task_progress_callback_test",ros::init_options::NoSigintHandler);
  CHECK(argc==2);ros::NodeHandle nh,pnh("~");std::ofstream out(argv[1]);
  const char* names[]={"large_motion_stall","missing_unknown","uncertified","wrong_view",
    "wrong_execution","stale_joint_state","safety_hold","goal_reached","real_orientation_progress",
    "changing_goal","changing_owner","changing_reference","abort","invalid_goal","stall_then_real_progress"};
  out<<"{\"status\":\"PASS\",\"scenarios\":[";
  for(int mode=0;mode<15;++mode) {
    ros::Time::setNow(ros::Time(1000+mode*10,0));
    Planner p;CHECK(p.initialize(nh,pnh));p.stopWorker();p.timer_.stop();
    p.plan_repair_mode_=p.plan_probe_mode_=p.repair_mode_=p.probe_mode_=false;
    p.normal_reference_refresh_pending_=false;
    p.latest_single_waypoint_active_=p.has_single_waypoint_q_=true;
    p.latest_single_waypoint_q_=Eigen::VectorXd::Constant(7,.2);
    p.latest_single_waypoint_joint_mask_=Eigen::VectorXd::Ones(7);
    p.latest_observation_token_="care_obs_v1_test";
    p.latest_vbc_obligation_id_=1;
    const Eigen::Vector3d point(.15,0,.3);p.latest_vbc_obligation_points_={point};
    Eigen::VectorXd zero=Eigen::VectorXd::Zero(7),goal_q=zero;
    if(mode==8 || mode==14)goal_q[6]=.8;
    Eigen::Isometry3d target;CHECK(p.primitive_observation_guard_.framePose(goal_q,"EE_link",&target));
    if(mode!=7 && mode!=8 && mode!=14)target.translation().x()+=.5;
    auto goal=boost::make_shared<geometry_msgs::PoseStamped>();goal->header.frame_id="base_link";
    goal->pose.position.x=target.translation().x();goal->pose.position.y=target.translation().y();goal->pose.position.z=target.translation().z();
    Eigen::Quaterniond rotation(target.linear());goal->pose.orientation.x=rotation.x();goal->pose.orientation.y=rotation.y();
    goal->pose.orientation.z=rotation.z();goal->pose.orientation.w=rotation.w();
    if(mode==13)goal->header.frame_id="invalid";
    p.terminalGoalCallback(goal);
    auto reference=boost::make_shared<trajectory_msgs::JointTrajectory>();reference->joint_names=p.joint_names_;
    reference->header.stamp=ros::Time::now();trajectory_msgs::JointTrajectoryPoint end;
    end.positions.assign(7,0.);end.positions[0]=.4;end.time_from_start=ros::Duration(1.);
    reference->points={end};p.referenceCallback(reference);
    p.plan_q_ref_=Eigen::MatrixXd::Zero(7,21);p.plan_q_ref_.row(0).setConstant(.4);
    unsigned long long execution=0;int stalls=0,large_candidates=0;
    unsigned long long captured_query=0;
    for(int i=0;i<40;++i) {
      const double elapsed=i*.1;ros::Time now;now.fromSec(1000+mode*10+elapsed);ros::Time::setNow(now);
      if(i && i%15==0) {
        if(mode==9){goal->pose.position.x+=.1;p.terminalGoalCallback(goal);}
        if(mode==10){auto wp=boost::make_shared<std_msgs::Float64MultiArray>();wp->data.assign(7,.2);
          std_msgs::MultiArrayDimension dim;dim.label="care_obs_v1_test_"+std::to_string(i);dim.size=7;dim.stride=7;
          wp->layout.dim={dim};p.singleWaypointQCallback(wp);}
        if(mode==11){reference->header.stamp=now;p.referenceCallback(reference);}
      }
      if(mode==6){auto hold=boost::make_shared<std_msgs::Bool>();hold->data=true;p.vbcExecutionHoldCallback(hold);}
      if(i%8==0) {
        p.current_query_stamp_=p.last_query_ros_time_=now-ros::Duration(.01);
        Eigen::MatrixXd q=Eigen::MatrixXd::Zero(7,21),u=Eigen::MatrixXd::Zero(7,20);
        for(int k=0;k<21;++k)q(6,k)=k*.15/20.;
        care_collision_cdf::CollisionCDFConstraintBatch batch;batch.header.stamp=p.current_query_stamp_;
        p.publishCandidateTrajectory(q,u,"base_link","none",batch,q,u,
            mode==1?std::vector<Eigen::Vector3d>{}:std::vector<Eigen::Vector3d>{point},{},false);
        const auto raw=now.toNSec();execution=raw+1;
        CHECK(p.normal_unknown_no_progress_.pending.empty());
        if(mode!=1 && mode!=13){CHECK(p.normal_unknown_task_progress_.pending.count(raw));++large_candidates;}
        if(mode!=2) {
          const auto verdict_execution=mode==4?raw-1:execution;
          p.finalVerificationCallback(message("result=safe safety_gate=vbc committed=1 verification_view="+
            std::string(mode==3?"repair_prefix":"full_horizon")+" raw_candidate_stamp_ns="+std::to_string(raw)+
            " audited_trajectory_stamp_ns="+std::to_string(raw)+" execution_stamp_ns="+std::to_string(verdict_execution)));
        }
      }
      Eigen::VectorXd measured=zero;measured[6]=mode==8?.1*elapsed:.02*std::sin(elapsed*5.);
      if(mode==14)measured[6]=std::max(0.,elapsed-2.5)*.4;
      auto state=boost::make_shared<sensor_msgs::JointState>();state->header.stamp=now-ros::Duration(mode==5?.3:0.);
      state->name=p.joint_names_;state->position.assign(measured.data(),measured.data()+7);p.jointStateCallback(state);
      p.executionSummaryCallback(message("active=1 complete=0 source=active_trajectory execution_aborted="+
        std::string(mode==12?"1":"0")+" phase_s="+std::to_string(now.toSec()-execution*1e-9)+
        " execution_stamp_ns="+std::to_string(execution)));
      if(p.normal_unknown_task_progress_.event=="stalled") {
        CHECK(p.task_no_progress_.blocked());captured_query=p.normal_unknown_task_progress_.last.query;
        ++stalls;
      }
    }
    CHECK((stalls>0)==(mode==0 || mode==14));
    CHECK(p.normal_measured_stall_count_==static_cast<unsigned long long>(mode==0 || mode==14));
    if(mode==14)CHECK(!p.normal_task_stall_pending_ && !p.task_no_progress_.blocked());
    CHECK(p.max_scp_iterations_==6 && p.safe_frontier_recovery_.attempts==0 && p.safe_frontier_recovery_.segments==0);
    if(mode)out<<',';
    out<<"{\"name\":\""<<names[mode]<<"\",\"stalled\":"<<(stalls?"true":"false")
       <<",\"large_candidates_admitted\":"<<large_candidates<<",\"last_query\":"<<captured_query<<'}';
  }
  out<<"]}\n";std::cout<<"PASS 15 real planner callback scenarios; no actuator\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
