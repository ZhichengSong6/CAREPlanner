#include "planner_diagnostic_access.hpp"
#include <boost/property_tree/json_parser.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <boost/make_shared.hpp>
#include <stdexcept>
using Planner=egocentric_arm_planner::LocalSparseSCPPlanner;
#define CHECK(x) do { if(!(x))throw std::runtime_error(#x); } while(false)
Eigen::VectorXd vec(const boost::property_tree::ptree& t) {
  Eigen::VectorXd v(t.size());int i=0;for(const auto& e:t)v[i++]=e.second.get_value<double>();return v;
}
int main(int argc,char** argv) {try {
  ros::init(argc,argv,"probe_execution_model_test",ros::init_options::NoSigintHandler);
  CHECK(argc==3);Planner p;ros::NodeHandle nh,pnh("~");CHECK(p.initialize(nh,pnh));
  p.stopWorker();p.timer_.stop();p.repair_prefix_binding_enabled_=false;
  boost::property_tree::ptree data;boost::property_tree::read_json(argv[1],data);
  std::ofstream out(argv[2]);out<<std::setprecision(17)<<"{\"status\":\"PASS\",\"probe_cases\":[";
  int cases=0,controls=0;
  for(const auto& entry:data.get_child("cases")) {
    if(cases==3)break;
    const auto q0=vec(entry.second.get_child("current.branch_seed_q"));
    const auto goal=vec(entry.second.get_child("context.q_task"));
    trajectory_msgs::JointTrajectory reference;reference.joint_names=p.joint_names_;
    reference.header.stamp=ros::Time::now();reference.points.resize(2);
    reference.points[0].positions.assign(q0.data(),q0.data()+7);
    reference.points[1].positions.assign(goal.data(),goal.data()+7);
    reference.points[1].time_from_start=ros::Duration(1.);
    Eigen::MatrixXd qr,ur,q,u;
    CHECK(p.buildReferenceHorizon(q0,reference,reference.header.stamp+ros::Duration(2.),true,qr,ur,q,u));
    CHECK(ur.cwiseAbs().maxCoeff()<=.5+1e-9 && u.cwiseAbs().maxCoeff()<=.5+1e-9);
    CHECK((qr.col(5)-q0).lpNorm<Eigen::Infinity>()<=.125+1e-9);
    CHECK((qr.rightCols(15).colwise()-qr.col(5)).cwiseAbs().maxCoeff()<1e-12);
    care_collision_cdf::CollisionCDFConstraintBatch b;b.dof=7;b.num_pairs=0;
    p.primitive_observation_guard_enabled_=false;
    const auto r=p.solveSparseSubproblem(b,q,u,qr,ur,Eigen::VectorXd::Zero(7),{},
        Planner::FrontierObjective{}, {},false,true,.15,10.);
    CHECK(r.solved && r.max_slack==0. && r.u.cwiseAbs().maxCoeff()<=.5+1e-6);
    const auto msg=p.makeTrajectoryMessage(r.q,r.u,"base_link",ros::Time::now());
    if(cases++)out<<',';
    out<<"{\"max_velocity\":"<<r.u.cwiseAbs().maxCoeff()<<",\"raw\":{\"joint_names\":[";
    for(int j=0;j<7;++j){if(j)out<<',';out<<'"'<<msg.joint_names[j]<<'"';}
    out<<"],\"points\":[";
    for(size_t k=0;k<msg.points.size();++k){if(k)out<<',';const auto& pt=msg.points[k];
      out<<"{\"time\":"<<pt.time_from_start.toSec()<<",\"positions\":[";
      for(int j=0;j<7;++j){if(j)out<<',';out<<pt.positions[j];}
      out<<"],\"velocities\":[";for(int j=0;j<7;++j){if(j)out<<',';out<<pt.velocities[j];}out<<"]}";
    }out<<"]}}";
    p.repair_mode_=true;p.probe_mode_=false;p.latest_observation_token_="care_obs_v1_warm";
    p.repair_no_progress_.select("same_target");
    const auto reset=[&](){p.repair_warm_q_=r.q;p.repair_warm_key_=p.repairWarmStartKeyLocked();
      p.repair_warm_stamp_=ros::Time::now();p.repair_warm_iterations_=2;};
    auto current=q0;current[0]+=.005;const Eigen::VectorXd v=Eigen::VectorXd::Constant(7,.005);
    int charged=0;Eigen::MatrixXd rq,ru;reset();
    CHECK(p.takeRepairWarmStartLocked(current,v,rq,ru,charged));CHECK(charged==2);
    CHECK((rq.col(0)-current).norm()==0. && ru.col(19).lpNorm<Eigen::Infinity>()<=1e-6);
    for(int k=0;k<20;++k){
      Eigen::VectorXd before=v;if(k)before=ru.col(k-1);
      CHECK((rq.col(k+1)-rq.col(k)-.5*p.dt_*(before+ru.col(k))).lpNorm<Eigen::Infinity>()<1e-10);
    }
    CHECK(!p.takeRepairWarmStartLocked(current,v,rq,ru,charged));++controls;
    reset();p.latest_observation_token_+="changed";
    CHECK(!p.takeRepairWarmStartLocked(current,v,rq,ru,charged));++controls;
    reset();p.repair_warm_stamp_=ros::Time::now()-ros::Duration(.21);
    CHECK(!p.takeRepairWarmStartLocked(current,v,rq,ru,charged));++controls;
    reset();p.repair_warm_iterations_=6;
    CHECK(!p.takeRepairWarmStartLocked(current,v,rq,ru,charged));++controls;
    reset();p.repair_warm_q_(1,2)=std::numeric_limits<double>::quiet_NaN();
    CHECK(!p.takeRepairWarmStartLocked(current,v,rq,ru,charged));++controls;
    reset();p.probe_mode_=true;
    CHECK(!p.takeRepairWarmStartLocked(current,v,rq,ru,charged));++controls;p.probe_mode_=false;
    reset();p.latest_vbc_obligation_points_seq_++;
    CHECK(!p.takeRepairWarmStartLocked(current,v,rq,ru,charged));++controls;
  }
  CHECK(cases==3);
  // Actual stale-progress worker: keep the solved shape, charge its QPs,
  // then consume it once at a different measured start. No candidate executes.
  const auto& seed=data.get_child("cases").front().second;
  const auto q0=vec(seed.get_child("current.branch_seed_q"));
  const auto target=vec(seed.get_child("current.selected_q_vis"));
  const auto mask=vec(seed.get_child("current.selected_joint_mask"));
  p.repair_executable_view_enabled_=false;p.safe_frontier_recovery_enabled_=false;
  p.candidate_replacement_enabled_=false;p.repair_mode_=p.plan_repair_mode_=true;
  p.probe_mode_=p.plan_probe_mode_=false;p.repair_observation_phase_=p.plan_repair_observation_phase_=true;
  p.latest_single_waypoint_active_=p.has_single_waypoint_q_=true;
  p.latest_single_waypoint_q_=target;p.latest_single_waypoint_joint_mask_=mask;
  p.latest_observation_token_="care_obs_v1_warm_worker";p.selectRepairTargetLocked();
  p.worker_stop_=false;p.plan_running_=true;++p.plan_sequence_;
  p.plan_repair_ticket_=p.repair_no_progress_.active;++p.repair_no_progress_.active.progress;
  p.plan_mode_epoch_=p.mode_epoch_;p.plan_observation_token_=p.latest_observation_token_;p.scp_iteration_=0;
  p.plan_q_bar_=q0.replicate(1,21);p.plan_u_bar_=Eigen::MatrixXd::Zero(7,20);
  p.plan_q_ref_=vec(seed.get_child("context.q_task")).replicate(1,21);p.plan_u_ref_=p.plan_u_bar_;
  p.plan_q_current_=q0;p.plan_previous_command_=Eigen::VectorXd::Zero(7);
  p.plan_schedule_.clear();Planner::DeadlineWaypoint wp;wp.q=target;wp.joint_mask=mask;wp.terminal_objective=true;
  p.plan_schedule_.push_back(wp);p.plan_frontier_=Planner::FrontierObjective{};
  p.plan_frontier_.active=true;p.plan_frontier_.q=target;p.plan_frontier_.joint_mask=mask;
  p.plan_frontier_.frontier_weight_scale=1.;p.plan_frontier_.qvis_weight_scale=0.;
  p.repair_witness_requires_reobserve_=false;p.repair_witness_overflow_=false;p.repair_unknown_witnesses_.clear();
  p.plan_has_hard_solution_=false;p.current_frame_id_="base_link";
  p.current_plan_start_wall_=ros::WallTime::now();p.current_query_stamp_=ros::Time::now();
  auto batch=boost::make_shared<care_collision_cdf::CollisionCDFConstraintBatch>();
  batch->dof=7;batch->num_pairs=0;batch->header.stamp=p.current_query_stamp_;
  p.startWorker();{std::lock_guard<std::mutex> lock(p.mutex_);p.pending_batch_=batch;}p.worker_cv_.notify_one();
  bool cached=false;
  for(int i=0;i<200;++i){
    {std::lock_guard<std::mutex> lock(p.mutex_);cached=!p.repair_warm_key_.empty();}
    if(cached)break;std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  p.stopWorker();CHECK(cached && !p.plan_running_ && p.repair_warm_iterations_==2);
  Eigen::MatrixXd warmed_q,warmed_u;int spent=0;Eigen::VectorXd current=q0;current[0]+=.005;
  CHECK(p.takeRepairWarmStartLocked(current,Eigen::VectorXd::Zero(7),warmed_q,warmed_u,spent));
  CHECK(spent==2 && (warmed_q.col(0)-current).norm()==0.);
  out<<"],\"actual_stale_worker_controls\":1,\"warm_start_negative_controls\":"<<controls
      <<",\"qualification\":\"Production reference/QP and fresh-q/v warm-shape rebuild with empty CDF rows; no robot or execution permission.\"}\n";
  std::cout<<"PASS "<<cases<<" production probe QPs, "<<controls<<" warm-shape controls\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
