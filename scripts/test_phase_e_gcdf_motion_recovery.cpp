#include "planner_diagnostic_access.hpp"
#include <boost/property_tree/json_parser.hpp>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <iostream>
using boost::property_tree::ptree;
using Planner=egocentric_arm_planner::LocalSparseSCPPlanner;
Eigen::VectorXd vec(const ptree& row){Eigen::VectorXd v(row.size());int j=0;for(const auto& x:row)v[j++]=x.second.get_value<double>();return v;}
void vectorJson(std::ostream& out,const Eigen::VectorXd& v){out<<'[';for(int j=0;j<v.size();++j){if(j)out<<',';out<<v[j];}out<<']';}
void matrixJson(std::ostream& out,const Eigen::MatrixXd& m){out<<'[';for(int k=0;k<m.cols();++k){if(k)out<<',';vectorJson(out,m.col(k));}out<<']';}
int main(int argc,char** argv){
  ros::init(argc,argv,"gcdf_motion_test",ros::init_options::NoSigintHandler);
  try {
    assert(argc==3);ros::NodeHandle nh,pnh("~");Planner p;
    assert(p.initialize(nh,pnh));p.stopWorker();p.timer_.stop();
    ptree fixture;boost::property_tree::read_json(argv[1],fixture);
    std::ofstream out(argv[2]);out<<std::setprecision(17)<<"{\"cases\":[";bool first=true;
    for(const auto& item:fixture.get_child("cases")) {
      const auto& f=item.second;Eigen::MatrixXd raw(7,21),q(7,21),u(7,20);int k=0;
      for(const auto& row:f.get_child("raw_q"))raw.col(k++)=vec(row.second);
      const Eigen::VectorXd v0=Eigen::VectorXd::Zero(7),point=vec(f.get_child("point"));
      std::vector<Eigen::Vector3d> points{point};
      care_collision_cdf::CollisionCDFConstraintBatch b;b.dof=7;
#if CARE_BEFORE
      q=raw;for(int k=0;k<20;++k)u.col(k)=(raw.col(k+1)-raw.col(k))/.05;
      const auto& rows=f.get_child("raw_rows");
#else
      q=raw.col(0).replicate(1,21);u.setZero();
      const auto& rows=f.get_child("physical_hold_rows");
#endif
      k=1;for(const auto& row:rows) {
        b.original_timestep.push_back(k++);b.source_type.push_back(0);
        b.distance.push_back(row.second.get<double>("distance"));
        const auto gradient=vec(row.second.get_child("gradient")),center=vec(row.second.get_child("q"));
        for(int j=0;j<7;++j){b.gradient_flat.push_back(gradient[j]);b.q_linearization_flat.push_back(center[j]);}
        for(int j=0;j<3;++j)b.point_flat.push_back(point[j]);
      }
      b.num_pairs=20;
      auto r=p.solveSparseSubproblem(b,q,u,raw,u,v0,{},Planner::FrontierObjective{},points,true,false,.15,10.);
#if CARE_BEFORE
      assert(r.solved);
#else
      assert(!r.solved && r.first_step_conflicting_cdf_rows>0 && r.first_step_conflicting_occupied_rows==0);
      assert(!r.geometric_restore_guard_points.empty());
#endif
      if(!first)out<<',';first=false;
      out<<"{\"raw_stamp_ns\":\""<<f.get<std::string>("raw_stamp_ns")<<"\",\"solved\":"<<(r.solved?"true":"false")
         <<",\"status\":\""<<r.status<<"\",\"cdf_rows\":"<<r.selected_cdf_rows
         <<",\"first_physical_conflicts\":"<<r.first_step_conflicting_cdf_rows<<'}';
    }
    out<<']';
#if !CARE_BEFORE
    // Controlled safe rows isolate physical QP -> nodal proposal -> actual
    // constructor. A brake-only hard conflict must not disappear from the QP.
    Eigen::MatrixXd q=Eigen::MatrixXd::Zero(7,21),u=Eigen::MatrixXd::Zero(7,20),reference=q;
    for(int k=1;k<=20;++k)reference(0,k)=.02;
    const Eigen::VectorXd v0=Eigen::VectorXd::Zero(7);
    auto settings=p.repair_executable_settings_;settings.prefixes={.25};settings.initial_velocity=v0;
    p.repair_executable_settings_.prefixes={.25};
    std::vector<egocentric_arm_planner::RepairExecutableSample> samples;
    assert(egocentric_arm_planner::repairCdfExecutableSamples(q,u,.05,settings,&samples));
    care_collision_cdf::CollisionCDFConstraintBatch b;b.dof=7;
    const int total=20+samples.size();
    for(int k=1;k<=total;++k) {
      b.original_timestep.push_back(k);b.source_type.push_back(0);b.distance.push_back(.5);
      b.point_flat.insert(b.point_flat.end(),{-.15,.15,.5});
      for(int j=0;j<7;++j){b.gradient_flat.push_back(0.);b.q_linearization_flat.push_back(0.);}
    }
    b.num_pairs=total;
    auto good=p.solveSparseSubproblem(b,q,u,reference,u,v0,{},Planner::FrontierObjective{},{Eigen::Vector3d(-.15,.15,.5)},true,false,.15,10.);
    assert(good.solved && good.selected_cdf_rows==total && good.max_slack==0.);
    auto bad=b;bad.distance.back()=-1.;
    const auto rejected=p.solveSparseSubproblem(bad,q,u,reference,u,v0,{},Planner::FrontierObjective{},{Eigen::Vector3d(-.15,.15,.5)},true,false,.15,10.);
    assert(!rejected.solved && rejected.selected_cdf_rows==total);
    assert(egocentric_arm_planner::repairCdfExecutableSamples(good.q,good.u,.05,settings,&samples));
    out<<",\"safe_motion\":{\"q\":";matrixJson(out,good.q);out<<",\"u\":";matrixJson(out,good.u);
    out<<",\"added_cdf_rows\":"<<total-20<<",\"brake_only_conflict_rejected\":true,\"samples\":[";
    for(std::size_t i=0;i<samples.size();++i){if(i)out<<',';out<<"{\"t\":"<<samples[i].time_s<<",\"q\":";vectorJson(out,samples[i].q);out<<'}';}out<<"]}";

    // Run actual production feedback/reservation callbacks without actuators.
    const Eigen::Vector3d point(-.15,.15,.5);
    auto context=[&](int epoch,bool replacement) {
      p.repair_mode_=true;p.probe_mode_=false;p.mode_epoch_=epoch;p.repair_observation_phase_=true;
      p.candidate_replacement_enabled_=replacement;p.latest_vbc_obligation_id_=10;
      p.latest_vbc_obligation_points_={point};p.latest_observation_token_="care_obs_v1_10_fixture";
      p.latest_single_waypoint_active_=true;p.has_single_waypoint_q_=true;p.latest_single_waypoint_q_=Eigen::VectorXd::Ones(7);
      p.latest_single_waypoint_joint_mask_=Eigen::VectorXd::Ones(7);p.selectRepairTargetLocked();
      // Separate test episodes have explicit measured motion. Token/epoch
      // changes alone are never used to refund a no-progress ledger.
      const auto measured=Eigen::VectorXd::Constant(7,epoch*.02).eval();
      p.repair_no_progress_.observe(measured);p.final_vbc_no_progress_.progress.observe(measured);
      p.gcdf_feedback_pending_.clear();p.final_vbc_no_progress_.pending.clear();p.final_gcdf_previous_points_.clear();
      p.candidate_replacement_pending_=p.candidate_replacement_trigger_pending_=false;
      p.current_frame_id_="base_link";p.plan_requested_=true;
    };
    int raw_counter=0;
    auto feedback=[&]() {
      ros::Time now(100+ ++raw_counter,0);ros::Time::setNow(now);
      const auto raw=now.toNSec();p.current_query_stamp_=now;p.last_query_ros_time_=now;
      assert(p.final_vbc_no_progress_.remember(raw,p.mode_epoch_,1));
      p.gcdf_feedback_pending_.emplace(raw,Planner::GcdfCandidate{
          {p.repair_no_progress_.active,p.mode_epoch_,1},p.latest_observation_token_,now,ros::WallTime::now()+ros::WallDuration(.5)});
      care_collision_cdf::CollisionCDFRejectionFeedbackPtr msg(new care_collision_cdf::CollisionCDFRejectionFeedback);
      msg->raw_candidate_stamp=now;msg->header.stamp=now;msg->header.frame_id="base_link";
      msg->point_flat={point.x(),point.y(),point.z()};return msg;
    };
    context(11,false);
    for(int i=0;i<98;++i){auto msg=feedback();p.gcdfRejectionFeedbackCallback(msg);const int n=p.final_vbc_no_progress_.progress.failures();p.gcdfRejectionFeedbackCallback(msg);assert(p.final_vbc_no_progress_.progress.failures()==n);}
    assert(p.final_vbc_no_progress_.progress.failures()==5 && !p.plan_requested_);
    context(12,true);
    const auto first_feedback=feedback();p.gcdfRejectionFeedbackCallback(first_feedback);assert(!p.candidate_replacement_trigger_pending_);
    auto second_feedback=feedback();p.gcdfRejectionFeedbackCallback(second_feedback);
    assert(p.candidate_replacement_trigger_pending_ && p.candidate_replacement_trigger_reason_=="final_gcdf_repeat");
    std_msgs::StringPtr request(new std_msgs::String);
    const auto raw=second_feedback->raw_candidate_stamp.toNSec();
    request->data="action=reserve request_id=1 reason=final_gcdf_repeat mode_epoch=12 query_stamp_ns="+std::to_string(raw)+
        " observation_token=care_obs_v1_10_fixture obligation_id=10 trigger_raw_candidate_stamp_ns="+std::to_string(raw);
    p.candidateReplacementCallback(request);
    assert(p.candidate_replacement_pending_ && p.safe_frontier_recovery_.attempts==2);
    p.gcdfRejectionFeedbackCallback(second_feedback);assert(p.safe_frontier_recovery_.attempts==2);
    for(int kind=0;kind<5;++kind) {
      context(20+kind,false);auto msg=feedback();const int count=p.final_vbc_no_progress_.progress.failures();
      if(kind==0)p.latest_observation_token_="care_obs_v1_10_stale";
      if(kind==1)++p.mode_epoch_;
      if(kind==2)++p.repair_no_progress_.active.progress;
      if(kind==3)p.gcdf_feedback_pending_.begin()->second.deadline=ros::WallTime::now()-ros::WallDuration(.01);
      if(kind==4)msg->header.stamp=msg->header.stamp+ros::Duration(.01);
      p.gcdfRejectionFeedbackCallback(msg);assert(p.final_vbc_no_progress_.progress.failures()==count);
    }
    out<<",\"feedback_identity_and_caps\":\"PASS\"";
#endif
    out<<",\"status\":\"PASS\"}\n";ros::shutdown();return 0;
  } catch(const std::exception& e){std::cerr<<e.what()<<'\n';ros::shutdown();return 1;}
}
