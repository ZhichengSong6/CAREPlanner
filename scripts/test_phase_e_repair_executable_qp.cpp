#include "planner_diagnostic_access.hpp"
#include <boost/property_tree/json_parser.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <boost/make_shared.hpp>
#include <atomic>
#include <chrono>
#include <thread>
using boost::property_tree::ptree;
using Planner=egocentric_arm_planner::LocalSparseSCPPlanner;
Eigen::VectorXd vector(const ptree& t) {
  Eigen::VectorXd v(t.size());int j=0;for(const auto& x:t)v[j++]=x.second.get_value<double>();return v;
}
Eigen::MatrixXd matrix(const ptree& t) {
  Eigen::MatrixXd m(t.front().second.size(),t.size());int k=0;for(const auto& x:t)m.col(k++)=vector(x.second);return m;
}
void matrixJson(std::ostream& o,const Eigen::MatrixXd& m) {
  o<<'[';for(int k=0;k<m.cols();++k){if(k)o<<',';o<<'[';for(int j=0;j<m.rows();++j){if(j)o<<',';o<<m(j,k);}o<<']';}o<<']';
}
int main(int argc,char** argv) {
  ros::init(argc,argv,"repair_executable_test",ros::init_options::NoSigintHandler);
  try {
    if(argc!=3)throw std::runtime_error("fixture/results required");
    ros::NodeHandle nh,pnh("~");Planner p;if(!p.initialize(nh,pnh))throw std::runtime_error("initialize failed");
    p.stopWorker();p.timer_.stop();
    ptree fixture;boost::property_tree::read_json(argv[1],fixture);
    Eigen::MatrixXd q=matrix(fixture.get_child("q")),u=matrix(fixture.get_child("u"));
    const auto qr=q,ur=u;
    const Eigen::Vector3d point=vector(fixture.get_child("point"));
    std::vector<egocentric_arm_planner::PrimitiveObservationPoint> evidence;
    if(!egocentric_arm_planner::parsePrimitiveObservationEvidence(fixture.get<std::string>("evidence"),
        fixture.get<unsigned long long>("audited_stamp_ns"),&evidence))throw std::runtime_error("evidence invalid");
    Planner::RepairWitness w{point,1,.05,true,-1.,.3};w.has_primitive_guard=true;w.primitive_guard=evidence.front();
    p.repair_unknown_witnesses_={w};
    // Identical controlled objective on the real recorded q/u. Safe constant
    // CDF rows isolate the executable geometry mechanism; they are not the
    // original nonlinear CDF batch, whose complete successful QP was not saved.
    p.repair_task_tracking_scale_=1.;p.q_tracking_weight_=1000.;p.terminal_q_tracking_weight_=0.;
    p.u_tracking_weight_=1000.;p.u_reference_tracking_enabled_=true;p.u_smooth_weight_=0.;p.handoff_velocity_weight_=0.;
#if CARE_NEW
    p.repair_executable_view_enabled_=true;
    p.repair_executable_settings_.prefixes={.25,.15,.10,.05};
    p.repair_executable_settings_.acceleration=vector(fixture.get_child("acceleration"));
    p.repair_executable_settings_.brake_dt=.05;p.repair_executable_settings_.max_brake_steps=20;
#endif
    std::ofstream output(argv[2]);output<<std::setprecision(17)<<'[';
    for(int iteration=0;iteration<6;++iteration) {
      care_collision_cdf::CollisionCDFConstraintBatch b;b.dof=7;
      for(int k=1;k<=20;++k) {
        b.original_timestep.push_back(k);b.source_type.push_back(0);b.distance.push_back(.5);
        for(int j=0;j<3;++j)b.point_flat.push_back(point[j]);
        for(int j=0;j<7;++j){b.gradient_flat.push_back(0.);b.q_linearization_flat.push_back(q(j,k));}
      }
      b.num_pairs=20;
      const auto r=p.solveSparseSubproblem(b,q,u,qr,ur,ur.col(0),{},Planner::FrontierObjective{},
                                           {point},true,false,.15,10.);
      if(iteration)output<<',';
      output<<"{\"iteration\":"<<iteration<<",\"solved\":"<<(r.solved?"true":"false")
            <<",\"status\":\""<<r.status<<"\",\"cdf_rows\":"<<r.selected_cdf_rows
            <<",\"geometry_rows\":"<<r.primitive_observation_rows<<",\"max_slack\":"<<r.max_slack;
#if CARE_NEW
      output<<",\"executable_rows\":"<<r.primitive_executable_rows
            <<",\"executable_samples\":"<<r.primitive_executable_samples
            <<",\"executable_valid\":"<<(r.primitive_executable_valid?"true":"false");
#endif
      if(r.solved){output<<",\"q\":";matrixJson(output,r.q);output<<",\"u\":";matrixJson(output,r.u);q=r.q;u=r.u;}
      output<<'}';
      if(!r.solved)break;
#if CARE_NEW
      if(r.primitive_executable_valid)break;
#else
      break;
#endif
    }
    output<<"]\n";
#if CARE_NEW
    // Exercise the actual worker at its last allowed iteration. A nonlinear
    // miss may not become a candidate merely because the convex QP solved.
    p.candidate_replacement_enabled_=true;
    p.scp_step_tolerance_inf_=100.;
    std::atomic<int> candidates{0};
    auto sub=nh.subscribe<trajectory_msgs::JointTrajectory>(p.candidate_trajectory_topic_,10,
        [&](const trajectory_msgs::JointTrajectoryConstPtr&){++candidates;});
    ros::AsyncSpinner spinner(1);spinner.start();
    std::ofstream worker_output(std::string(argv[2])+".worker.json");worker_output<<std::setprecision(17)<<'[';
    const Eigen::MatrixXd worker_reference=qr,worker_velocity=ur;
    auto constant_batch=[&](const Eigen::MatrixXd& seed) {
      care_collision_cdf::CollisionCDFConstraintBatch b;b.dof=7;
      for(int k=1;k<=20;++k){b.original_timestep.push_back(k);b.source_type.push_back(0);b.distance.push_back(.5);
        for(int j=0;j<3;++j)b.point_flat.push_back(point[j]);for(int j=0;j<7;++j){b.gradient_flat.push_back(0.);b.q_linearization_flat.push_back(seed(j,k));}}
      b.num_pairs=20;return b;
    };
    const Eigen::MatrixXd seed=qr.col(0).replicate(1,21);
    Eigen::MatrixXd velocities=Eigen::MatrixXd::Zero(7,20);
    for(int j=0;j<7;++j)velocities.row(j).setConstant(.5*p.repair_executable_settings_.acceleration[j]*.05);
    // Controlled construction failure: the stationary center fits one brake
    // step, but the solved velocity needs more. This tests the worker's actual
    // rejection path. The production launch's unchanged cap remains twenty.
    p.repair_executable_settings_.max_brake_steps=1;
    for(int scenario=0;scenario<3;++scenario) {
      p.stopWorker();
      auto initial=p.solveSparseSubproblem(constant_batch(seed),seed,velocities,worker_reference,worker_velocity,ur.col(0),{},Planner::FrontierObjective{},
          {point},true,false,.15,10.);
      if(!initial.solved || initial.primitive_executable_valid)throw std::runtime_error("controlled brake-cap failure not reproduced");
      auto b=constant_batch(seed);b.header.stamp=ros::Time::now();
      const auto stamp=b.header.stamp;
      {
        std::lock_guard<std::mutex> lock(p.mutex_);
        p.repair_mode_=p.plan_repair_mode_=true;p.probe_mode_=p.plan_probe_mode_=false;
        p.mode_epoch_=1;p.plan_mode_epoch_=scenario==1?0:1;
        p.plan_sequence_=100+scenario;p.plan_running_=true;p.waiting_for_cdf_=false;
        p.plan_repair_observation_phase_=p.repair_observation_phase_=true;
        p.repair_witness_requires_reobserve_=false;
        p.repair_no_progress_=egocentric_arm_planner::RepairNoProgress{};p.repair_no_progress_.select("executable_worker");
        p.plan_repair_ticket_=p.repair_no_progress_.active;
        p.plan_observation_token_=p.latest_observation_token_=scenario==2 ? "care_obs_v1_5_executable_test" : "executable_token";
        p.latest_vbc_obligation_id_=scenario==2 ? 5 : -1;
        p.safe_frontier_recovery_enabled_=true;
        p.latest_vbc_obligation_points_={point};p.plan_q_bar_=seed;p.plan_u_bar_=velocities;
        p.plan_q_ref_=worker_reference;p.plan_u_ref_=worker_velocity;p.plan_previous_command_=ur.col(0);
        p.plan_schedule_.clear();p.plan_frontier_=Planner::FrontierObjective{};
        p.trust_radius_=.15;p.scp_iteration_=p.max_scp_iterations_-1;
        p.current_query_stamp_=stamp;p.last_query_ros_time_=stamp;p.current_plan_start_wall_=ros::WallTime::now();
        p.pending_batch_=boost::make_shared<care_collision_cdf::CollisionCDFConstraintBatch>(b);
      }
      p.startWorker();const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(5);
      bool done=false;
      while(std::chrono::steady_clock::now()<end) {
        {std::lock_guard<std::mutex> lock(p.mutex_);done=!p.plan_running_;}
        if(done)break;std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
      p.stopWorker();
      if(!done)throw std::runtime_error("worker timeout");
      if((scenario==1 || !initial.primitive_executable_valid) && candidates.load()!=0)
        throw std::runtime_error("invalid/stale worker published candidate");
      if(p.current_query_stamp_!=stamp)throw std::runtime_error("exhausted/stale worker exceeded query budget");
      if(scenario)worker_output<<',';
      worker_output<<"{\"scenario\":"<<scenario<<",\"test_brake_cap\":1,\"initial_solved\":"<<(initial.solved?"true":"false")
        <<",\"initial_executable_valid\":"<<(initial.primitive_executable_valid?"true":"false")
        <<",\"candidates\":"<<candidates.load()<<",\"iteration_after\":"<<p.scp_iteration_
        <<",\"replacement_pending\":"<<(p.candidate_replacement_trigger_pending_?"true":"false")<<'}';
      if(p.candidate_replacement_trigger_pending_!=(scenario==2))throw std::runtime_error("replacement eligibility mismatch");
    }
    worker_output<<"]\n";
    // The fixed start just outside 20 mm must not be tightened into a
    // constant infeasible row. Only variable executable positions get padding.
    if(fixture.count("boundary_q0")) {
      p.repair_executable_settings_.max_brake_steps=20;
      const Eigen::MatrixXd boundary=vector(fixture.get_child("boundary_q0")).replicate(1,21);
      const Eigen::MatrixXd zero_velocity=Eigen::MatrixXd::Zero(7,20);
      const auto b=p.solveSparseSubproblem(constant_batch(boundary),boundary,zero_velocity,boundary,zero_velocity,
          Eigen::VectorXd::Zero(7),{},Planner::FrontierObjective{},{point},true,false,.15,10.);
      if(!b.solved || !b.primitive_executable_valid)throw std::runtime_error("valid fixed start rejected by numerical padding");
      std::ofstream o(std::string(argv[2])+".boundary.json");
      o<<std::setprecision(17)<<"{\"solved\":true,\"cdf_rows\":"<<b.selected_cdf_rows
        <<",\"executable_valid\":true,\"min_distance_m\":"<<b.primitive_executable_min_distance_m
        <<",\"numerical_tightening_m\":"<<b.primitive_executable_numerical_tightening_m<<"}\n";
    }
#endif
    ros::shutdown();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
