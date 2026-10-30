#include "planner_diagnostic_access.hpp"
#include <boost/property_tree/json_parser.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <boost/make_shared.hpp>
using Planner=egocentric_arm_planner::LocalSparseSCPPlanner;
using boost::property_tree::ptree;
#define CHECK(x) do {if(!(x))throw std::runtime_error(#x);}while(false)
Eigen::VectorXd vec(const ptree& t) {Eigen::VectorXd v(t.size());int i=0;for(const auto& x:t)v[i++]=x.second.get_value<double>();return v;}
void matrix(std::ostream& out,const Eigen::MatrixXd& m) {
 out<<'[';for(int k=0;k<m.cols();++k){if(k)out<<',';out<<'[';for(int j=0;j<m.rows();++j){if(j)out<<',';out<<m(j,k);}out<<']';}out<<']';
}
int main(int argc,char** argv) {try {
 ros::init(argc,argv,"free_joint_task_test",ros::init_options::NoSigintHandler);
 CHECK(argc==3);ros::NodeHandle nh,pnh("~");Planner p;CHECK(p.initialize(nh,pnh));p.stopWorker();p.timer_.stop();
 p.repair_prefix_binding_enabled_=false;p.primitive_observation_guard_enabled_=false;
 p.repair_free_joint_task_tracking_scale_=1.;CHECK(p.max_scp_iterations_==6);
 ptree input;boost::property_tree::read_json(argv[1],input);
 std::ofstream out(argv[2]);out<<std::setprecision(17)<<"{\"status\":\"PASS\",\"cases\":[";
 int count=0,used=0,controls=0;
 for(const auto& entry:input.get_child("cases")) {
  const auto& f=entry.second;const auto q0=vec(f.get_child("current.branch_seed_q"));
  const auto goal=vec(f.get_child("context.q_task")),qvis=vec(f.get_child("current.selected_q_vis")),mask=vec(f.get_child("current.selected_joint_mask"));
  Eigen::MatrixXd q=q0.replicate(1,21),u=Eigen::MatrixXd::Zero(7,20),ref=goal.replicate(1,21);
  const auto previous=Eigen::VectorXd::Zero(7);
  CHECK(p.buildNormalMotionSeed(q0,previous,q,q,u));
  care_collision_cdf::CollisionCDFConstraintBatch b;b.dof=7;b.num_pairs=0;
  Planner::DeadlineWaypoint wp;wp.id=1;wp.terminal_objective=true;wp.q=qvis;wp.joint_mask=mask;
  Planner::FrontierObjective frontier;frontier.active=true;frontier.joint_mask=mask;
  const auto delta=(qvis-q0).cwiseProduct(mask);
  frontier.q=q0+delta*std::min(1.,.05/std::max(1e-9,delta.lpNorm<Eigen::Infinity>()));
  frontier.qvis_weight_scale=0.;frontier.frontier_weight_scale=1.;
  const auto baseline=p.solveSparseSubproblem(b,q,u,ref,u,previous,{wp},frontier,{},true,false,.15,10.);
  const auto trial=p.solveSparseSubproblem(b,q,u,ref,u,previous,{wp},frontier,{},true,false,.15,10.,false,false,1.,false,nullptr,true);
  CHECK(baseline.solved && trial.solved && baseline.max_slack==0. && trial.max_slack==0.);
  const bool accepted=p.freeJointTaskImproves(baseline,trial,{wp},frontier,ref);used+=accepted;
  if((mask.array()==1.).all())CHECK(!accepted);
  if(accepted) {
   auto bad=trial;int related=-1;for(int j=0;j<7;++j)if(mask[j]==1.){related=j;break;}
   CHECK(related>=0);bad.q(related,2)+=.001;
   CHECK(!p.freeJointTaskImproves(baseline,bad,{wp},frontier,ref));++controls;
   bad=trial;bad.q(0,0)=std::numeric_limits<double>::quiet_NaN();
   CHECK(!p.freeJointTaskImproves(baseline,bad,{wp},frontier,ref));++controls;
   bad=trial;bad.max_slack=.001;CHECK(!p.freeJointTaskImproves(baseline,bad,{wp},frontier,ref));++controls;
   bad=trial;bad.primitive_executable_valid=false;
   CHECK(!p.freeJointTaskImproves(baseline,bad,{wp},frontier,ref));++controls;
   bad=trial;bad.primitive_live_rebase_required=true;
   CHECK(!p.freeJointTaskImproves(baseline,bad,{wp},frontier,ref));++controls;
   auto recovery=frontier;recovery.recovery_attempt=1;
   CHECK(!p.freeJointTaskImproves(baseline,trial,{wp},recovery,ref));++controls;
   auto changed=wp;changed.joint_mask.setOnes();
   CHECK(!p.freeJointTaskImproves(baseline,trial,{changed},frontier,ref));++controls;
  }
  if(count++)out<<',';
  out<<"{\"tag\":\""<<f.get<std::string>("tag")<<"\",\"id\":"<<f.get<int>("id")
     <<",\"sensor\":"<<f.get<int>("selected_sensor")<<",\"used\":"<<(accepted?"true":"false")
     <<",\"q_primary\":";matrix(out,baseline.q);out<<",\"q_trial\":";matrix(out,trial.q);
  out<<",\"u_primary\":";matrix(out,baseline.u);out<<",\"u_trial\":";matrix(out,trial.u);out<<'}';
 }
 CHECK(count==13 && used>0 && controls>0);
 // Exercise the actual asynchronous worker at the two iteration boundaries.
 // No subscribers grant motion; these are controlled empty-CDF budget cases.
 const auto& seed=input.get_child("cases").front().second;
 const auto q0=vec(seed.get_child("current.branch_seed_q"));
 const auto target=vec(seed.get_child("current.selected_q_vis"));
 const auto mask=vec(seed.get_child("current.selected_joint_mask"));
 p.repair_executable_view_enabled_=false;p.safe_frontier_recovery_enabled_=false;
 p.candidate_replacement_enabled_=false;p.repair_mode_=p.plan_repair_mode_=true;
 p.probe_mode_=p.plan_probe_mode_=false;p.repair_observation_phase_=p.plan_repair_observation_phase_=true;
 p.latest_single_waypoint_active_=p.has_single_waypoint_q_=true;
 p.latest_single_waypoint_q_=target;p.latest_single_waypoint_joint_mask_=mask;
 p.latest_observation_token_="care_obs_v1_budget_test";p.selectRepairTargetLocked();
 int budget_controls=0;
 for(int start:{4,5}) {
  p.worker_stop_=false;p.plan_running_=true;++p.plan_sequence_;
  p.plan_repair_ticket_=p.repair_no_progress_.active;p.plan_mode_epoch_=p.mode_epoch_;
  p.plan_observation_token_=p.latest_observation_token_;p.scp_iteration_=start;
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
  const auto solves=p.solve_count_;p.startWorker();
  {std::lock_guard<std::mutex> lock(p.mutex_);p.pending_batch_=batch;}
  p.worker_cv_.notify_one();
  bool completed=false;
  for(int i=0;i<300;++i) {
   {std::lock_guard<std::mutex> lock(p.mutex_);completed=p.scp_iteration_==6;}
   if(completed)break;std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  p.stopWorker();CHECK(completed && p.scp_iteration_==6);
  CHECK(p.solve_count_-solves==static_cast<unsigned>(6-start));++budget_controls;
 }
 out<<"],\"accepted_trials\":"<<used<<",\"negative_controls\":"<<controls
    <<",\"worker_budget_controls\":"<<budget_controls
    <<",\"qualification\":\"Production QP at saved measured-q seeds with controlled zero incoming velocity and no CDF rows; two actual worker iteration-boundary controls; not historical map or live audit.\"}\n";
 std::cout<<"PASS "<<count<<" saved-seed production QP pairs, "<<used<<" accepted free-joint improvements, "<<controls<<" negative controls\n";
 return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
