#include "planner_diagnostic_access.hpp"
#include <boost/property_tree/json_parser.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cassert>
#include <thread>
#include <chrono>
using boost::property_tree::ptree;
using Planner=egocentric_arm_planner::LocalSparseSCPPlanner;
Eigen::VectorXd vec(const ptree& t) {Eigen::VectorXd v(t.size());int j=0;for(const auto& x:t)v[j++]=x.second.get_value<double>();return v;}
void matrix(std::ostream& out,const Eigen::MatrixXd& m) {
  out<<'[';for(int k=0;k<m.cols();++k){if(k)out<<',';out<<'[';for(int j=0;j<m.rows();++j){if(j)out<<',';out<<m(j,k);}out<<']';}out<<']';
}
int main(int argc,char** argv) {
  ros::init(argc,argv,"initial_shell_qp",ros::init_options::NoSigintHandler);
  try {
    if(argc!=3 && argc!=4)throw std::runtime_error("fixture/results and optional coupled snapshot required");
    ros::NodeHandle nh,pnh("~");Planner p;
    if(!p.initialize(nh,pnh))throw std::runtime_error("initialize failed");p.stopWorker();p.timer_.stop();
    ptree f;boost::property_tree::read_json(argv[1],f);
    const auto q0=vec(f.get_child("q0")),target=vec(f.get_child("target")),previous=vec(f.get_child("previous_command"));
    p.repair_unknown_witnesses_.clear();
    for(const auto& w:f.get_child("witnesses")) {
      Planner::RepairWitness r{vec(w.second.get_child("point")),1,w.second.get<double>("margin"),true,-1.,.3};
      r.has_primitive_guard=true;r.primitive_guard.point=r.point;
      r.primitive_guard.sweeping_link=w.second.get<std::string>("link");
      r.primitive_guard.collision_name=w.second.get<std::string>("collision_name");
      r.primitive_guard.collision_index=w.second.get<int>("collision_index");
      p.repair_unknown_witnesses_.push_back(r);
    }
    care_collision_cdf::CollisionCDFConstraintBatch b;b.dof=7;
    std::vector<Eigen::Vector3d> points;
    for(const auto& w:p.repair_unknown_witnesses_)points.push_back(w.point);
    for(int k=1;k<=20;++k)for(const auto& row:f.get_child("rows")) {
      const auto& r=row.second;b.original_timestep.push_back(k);b.source_type.push_back(0);
      b.distance.push_back(r.get<double>("d"));
      for(int j=0;j<3;++j)b.point_flat.push_back(vec(r.get_child("point"))[j]);
      for(int j=0;j<7;++j){b.gradient_flat.push_back(vec(r.get_child("g"))[j]);b.q_linearization_flat.push_back(q0[j]);}
    }
    b.num_pairs=b.distance.size();
    const Eigen::MatrixXd q=q0.replicate(1,21),u=Eigen::MatrixXd::Zero(7,20),reference=target.replicate(1,21);
    Planner::DeadlineWaypoint wp;wp.id=9;wp.terminal_objective=true;wp.q=target;wp.joint_mask=Eigen::VectorXd::Ones(7);
    std::ofstream out(argv[2]);out<<std::setprecision(17)<<'[';
    for(int enabled=0;enabled<=1;++enabled) {
      p.initial_shell_recovery_enabled_=enabled;p.initial_shell_execution_margin_m_=.01;
      const auto r=p.solveSparseSubproblem(b,q,u,reference,u,previous,{wp},Planner::FrontierObjective{},points,true,false,.15,10.);
      if(enabled)out<<',';
      out<<"{\"enabled\":"<<enabled<<",\"solved\":"<<(r.solved?"true":"false")<<",\"status\":\""<<r.status
          <<"\",\"cdf_rows\":"<<r.selected_cdf_rows<<",\"max_slack\":"<<r.max_slack
          <<",\"shell_points\":"<<r.primitive_shell_recovery_points<<",\"penalty_rows\":"<<r.primitive_shell_penalty_rows
          <<",\"executable_valid\":"<<(r.primitive_executable_valid?"true":"false");
      if(r.solved){out<<",\"q\":";matrix(out,r.q);out<<",\"u\":";matrix(out,r.u);}out<<'}';
    }
    p.primitive_observation_guard_enabled_=false;
    auto contradictory=b;
    for(std::size_t i=0;i<contradictory.distance.size();++i) {
      double norm=0.;for(int j=0;j<7;++j)norm+=std::abs(contradictory.gradient_flat[7*i+j]);
      contradictory.distance[i]=-1.-2.*norm;
    }
    for(int occupied=0;occupied<=1;++occupied) {
      contradictory.source_type.assign(contradictory.num_pairs,occupied);
      const auto r=p.solveSparseSubproblem(contradictory,q,u,reference,u,previous,{wp},Planner::FrontierObjective{},points,true,false,.15,10.);
      assert(!r.solved && r.first_step_conflicting_cdf_rows>0 && r.later_conflicting_cdf_rows>0);
      assert((r.first_step_conflicting_occupied_rows>0)==bool(occupied));
      assert(occupied || !r.geometric_restore_guard_points.empty());
      out<<",{\"classification\":\""<<(occupied?"occupied_start":"unknown_start")
         <<"\",\"first\":"<<r.first_step_conflicting_cdf_rows<<",\"later\":"<<r.later_conflicting_cdf_rows<<'}';
    }
    // Exercise the actual failure dispatcher: the old hard-QP hold must not
    // suppress an explicitly bounded alternate recovery, and a state failure
    // must not increment geometry/witness failure counters.
    p.repair_mode_=true;p.probe_mode_=false;p.repair_observation_phase_=true;p.mode_epoch_=2;
    p.candidate_replacement_enabled_=true;
    p.latest_vbc_obligation_id_=9;p.latest_vbc_obligation_points_=points;
    p.latest_observation_token_="care_obs_v1_9_fixture";p.latest_single_waypoint_active_=true;
    p.has_single_waypoint_q_=true;p.latest_single_waypoint_q_=target;p.latest_single_waypoint_joint_mask_=Eigen::VectorXd::Ones(7);
    p.selectRepairTargetLocked();
    p.repair_candidate_failure_.record(p.repairTargetKeyLocked(),p.latest_observation_token_,p.mode_epoch_,p.repair_no_progress_.active.progress);
    p.requestPlanLocked("ordinary_failed_query");assert(!p.plan_requested_);
    p.geometric_restore_scope_=std::to_string(p.mode_epoch_)+":"+p.safe_frontier_recovery_.key;
    p.requestPlanLocked("bounded_geometry_resume");assert(p.plan_requested_);
    const auto before_attempts=p.safe_frontier_recovery_.attempts;
    const auto before_failures=p.final_vbc_no_progress_.progress.failures();
    assert(p.final_vbc_no_progress_.remember(123,2,1));
    std_msgs::StringPtr failed(new std_msgs::String);
    failed->data="result=timeout safety_gate=state_evidence committed=0 raw_candidate_stamp_ns=123 audited_trajectory_stamp_ns=124 failure_kind=certificate_state_stale";
    p.finalVerificationCallback(failed);
    assert(!p.state_evidence_retry_pending_.empty() && !p.plan_requested_);
    assert(p.safe_frontier_recovery_.attempts==before_attempts && p.final_vbc_no_progress_.progress.failures()==before_failures);
    for(int i=0;i<3;++i) {
      std_msgs::Float64MultiArrayPtr cmd(new std_msgs::Float64MultiArray);cmd->data.assign(7,0.);p.executedCommandCallback(cmd);
      sensor_msgs::JointStatePtr state(new sensor_msgs::JointState);state->name=p.joint_names_;
      state->position.assign(q0.data(),q0.data()+7);state->velocity.assign(7,0.);state->header.stamp=ros::Time::now();
      p.jointStateCallback(state);
      if(i<2)std::this_thread::sleep_for(std::chrono::milliseconds(60));
    }
    assert(p.state_evidence_retry_pending_.empty() && p.plan_requested_);
    p.finalVerificationCallback(failed);assert(p.safe_frontier_recovery_.attempts==before_attempts);
    out<<",{\"failure_dispatch\":\"PASS\"}";
    p.primitive_observation_guard_enabled_=true;p.repair_prefix_binding_enabled_=true;
    egocentric_arm_planner::RepairPrefixSelection prefix;
    prefix.session=std::string(32,'a');prefix.revision=1;prefix.prefix_s=.25;
    prefix.brake_dt_s=.05;prefix.brake_steps=20;prefix.acceleration={3.,3.,4.,4.,6.,6.,6.};
    const auto margin=p.solveSparseSubproblem(b,q,u,reference,u,previous,{wp},Planner::FrontierObjective{},
        points,true,false,.15,10.,false,false,1.,false,&prefix);
    assert(margin.solved && margin.max_slack==0. && margin.selected_cdf_rows==b.num_pairs);
    assert(!margin.primitive_margin_recovery_points.empty() &&
        margin.primitive_margin_recovery_points.size()<=p.repair_unknown_witnesses_.size());
    for(const auto& point:margin.primitive_margin_recovery_points)
      assert(std::any_of(p.repair_unknown_witnesses_.begin(),p.repair_unknown_witnesses_.end(),
          [&](const Planner::RepairWitness& w){return (w.point-point).lpNorm<Eigen::Infinity>()<=1e-5;}));
    out<<",{\"owned_margin_hard_qp\":\"PASS\",\"cdf_rows\":"<<margin.selected_cdf_rows
       <<",\"owned_points\":"<<margin.primitive_margin_recovery_points.size()<<'}';
    if(argc==4) {
      ptree snapshot;boost::property_tree::read_json(argv[3],snapshot);
      const auto& s=snapshot.get_child("subproblem");
      auto read_matrix=[&](const char* key){const auto& a=s.get_child(key);Eigen::MatrixXd m(7,a.size());int k=0;for(const auto& row:a)m.col(k++)=vec(row.second);return m;};
      const auto qb=read_matrix("q_bar"),ub=read_matrix("u_bar"),qr=read_matrix("q_ref");
      care_collision_cdf::CollisionCDFConstraintBatch coupled;coupled.dof=7;
      std::vector<Eigen::Vector3d> owned;
      for(const auto& item:s.get_child("cdf_rows")) {
        const auto& row=item.second;coupled.original_timestep.push_back(row.get<int>("timestep"));
        coupled.source_type.push_back(row.get<int>("source"));coupled.distance.push_back(row.get<double>("distance"));
        const Eigen::Vector3d point=vec(row.get_child("point"));
        for(int j=0;j<3;++j)coupled.point_flat.push_back(point[j]);
        const auto g=vec(row.get_child("gradient")),center=vec(row.get_child("q_linearization"));
        for(int j=0;j<7;++j){coupled.gradient_flat.push_back(g[j]);coupled.q_linearization_flat.push_back(center[j]);}
        if(std::none_of(owned.begin(),owned.end(),[&](const Eigen::Vector3d& old){return (old-point).lpNorm<Eigen::Infinity>()<1e-5;}))owned.push_back(point);
      }
      coupled.num_pairs=coupled.distance.size();p.repair_unknown_witnesses_.clear();
      const auto r=p.solveSparseSubproblem(coupled,qb,ub,qr,ub,vec(s.get_child("previous_command")),{},
          Planner::FrontierObjective{},owned,true,false,.15,10.,false,false,1.,false,&prefix);
      assert(!r.solved && r.first_step_conflicting_cdf_rows==0 && !r.geometric_restore_guard_points.empty());
      out<<",{\"coupled_unknown_restore_points\":\"PASS\",\"first_step_conflicts\":"<<r.first_step_conflicting_cdf_rows
         <<",\"owned_points\":"<<r.geometric_restore_guard_points.size()<<'}';
    }
    out<<"]\n";
    ros::shutdown();return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';ros::shutdown();return 1;}
}
