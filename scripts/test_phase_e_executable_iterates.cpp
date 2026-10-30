#include "planner_diagnostic_access.hpp"
#include <boost/property_tree/json_parser.hpp>
#include <boost/make_shared.hpp>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <thread>
using Planner=egocentric_arm_planner::LocalSparseSCPPlanner;
using boost::property_tree::ptree;

Eigen::VectorXd vec(const ptree& t) {
  Eigen::VectorXd v(t.size());int i=0;
  for(const auto& x:t)v[i++]=x.second.get_value<double>();return v;
}
Eigen::MatrixXd mat(const ptree& t) {
  Eigen::MatrixXd m(t.front().second.size(),t.size());int k=0;
  for(const auto& x:t)m.col(k++)=vec(x.second);return m;
}
void matrix(std::ostream& o,const Eigen::MatrixXd& m) {
  o<<'[';for(int k=0;k<m.cols();++k){if(k)o<<',';o<<'[';
    for(int j=0;j<m.rows();++j){if(j)o<<',';o<<m(j,k);}o<<']';}o<<']';
}
int main(int argc,char** argv) {
  ros::init(argc,argv,"feasible_iterate_test",ros::init_options::NoSigintHandler);
  try {
    if(argc!=3)throw std::runtime_error("fixtures/results required");
    ptree fixtures;boost::property_tree::read_json(argv[1],fixtures);
    ptree safe,bad;
    for(const auto& f:fixtures) {
      if(f.second.get<std::string>("name")=="plan77_input2")safe=f.second;
      if(f.second.get<std::string>("name")=="plan77_input3")bad=f.second;
    }
    const auto qs=mat(safe.get_child("q")),us=mat(safe.get_child("u"));
    const auto qb=mat(bad.get_child("q")),ub=mat(bad.get_child("u"));
    ros::NodeHandle nh,pnh("~");Planner p;
    if(!p.initialize(nh,pnh))throw std::runtime_error("initialize failed");
    p.stopWorker();p.timer_.stop();
    std::vector<Planner::RepairWitness> witnesses;
    std::vector<Eigen::Vector3d> points;
    for(const auto& x:safe.get_child("points")) {
      const auto point=vec(x.second.get_child("point"));
      Planner::RepairWitness w{point,1,.05,true,-1.,.3};
      w.has_primitive_guard=true;w.primitive_guard.point=point;
      w.primitive_guard.sweeping_link=x.second.get<std::string>("sweeping_link");
      w.primitive_guard.collision_name=x.second.get<std::string>("collision_name");
      w.primitive_guard.collision_index=x.second.get<int>("collision_index");
      witnesses.push_back(w);points.push_back(point);
    }
    p.repair_task_tracking_scale_=1.;p.q_tracking_weight_=1000.;
    p.terminal_q_tracking_weight_=0.;p.u_tracking_weight_=1000.;
    p.u_reference_tracking_enabled_=true;p.u_smooth_weight_=0.;
    p.handoff_velocity_weight_=0.;p.candidate_replacement_enabled_=false;
    p.scp_step_tolerance_inf_=1e-12;
    p.repair_unknown_witnesses_=witnesses;
    auto batch=[&](const Eigen::MatrixXd& center,bool hostile=false) {
      care_collision_cdf::CollisionCDFConstraintBatch b;b.dof=7;
      for(int k=1;k<=20;++k)for(const auto& point:points) {
        b.original_timestep.push_back(k);b.source_type.push_back(hostile
          ?care_collision_cdf::CollisionCDFConstraintBatch::SOURCE_OCCUPIED
          :care_collision_cdf::CollisionCDFConstraintBatch::SOURCE_UNKNOWN);
        b.distance.push_back(hostile?-.5:.5);
        for(int j=0;j<3;++j)b.point_flat.push_back(point[j]);
        for(int j=0;j<7;++j){b.gradient_flat.push_back(0.);b.q_linearization_flat.push_back(center(j,k));}
      }
      b.num_pairs=b.distance.size();b.header.stamp=ros::Time::now();return b;
    };
    const auto initial=p.solveSparseSubproblem(batch(qs),qs,us,qb,ub,ub.col(0),{},
      Planner::FrontierObjective{},points,true,false,.15,10.);
    if(!initial.solved || initial.primitive_executable_valid)
      throw std::runtime_error("same-input invalid brake proposal not reproduced");
    std::atomic<int> candidates{0};
    auto sub=nh.subscribe<trajectory_msgs::JointTrajectory>(p.candidate_trajectory_topic_,20,
      [&](const trajectory_msgs::JointTrajectoryConstPtr&){++candidates;});
    ros::AsyncSpinner spinner(1);spinner.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    auto setup=[&](int consumed,int scenario) {
      p.stopWorker();
      std::lock_guard<std::mutex> lock(p.mutex_);
      p.repair_mode_=p.plan_repair_mode_=true;p.probe_mode_=p.plan_probe_mode_=false;
      p.mode_epoch_=p.plan_mode_epoch_=1;p.plan_sequence_=100+scenario;
      p.plan_running_=true;p.waiting_for_cdf_=false;p.plan_requested_=false;
      p.plan_repair_observation_phase_=p.repair_observation_phase_=true;
      p.repair_witness_requires_reobserve_=false;p.repair_unknown_witnesses_=witnesses;
      p.repair_no_progress_=egocentric_arm_planner::RepairNoProgress{};
      p.repair_no_progress_.select("feasible_iterate_target");p.plan_repair_ticket_=p.repair_no_progress_.active;
      p.plan_observation_token_=p.latest_observation_token_="controlled_observation";
      p.latest_vbc_obligation_points_=points;
      p.plan_q_bar_=qs;p.plan_u_bar_=us;p.plan_q_ref_=qb;p.plan_u_ref_=ub;
      p.plan_previous_command_=ub.col(0);p.plan_schedule_.clear();p.plan_frontier_=Planner::FrontierObjective{};
      p.trust_radius_=.15;p.scp_iteration_=consumed;p.max_scp_iterations_=6;
      p.plan_has_hard_solution_=true;p.plan_previous_q_bar_=qs;p.plan_previous_u_bar_=us;
      p.previous_query_min_distance_=std::numeric_limits<double>::quiet_NaN();
      p.current_plan_start_wall_=ros::WallTime::now();
#if CARE_NEW
      p.plan_executable_iterates_=egocentric_arm_planner::RepairExecutableIterates{};
      p.plan_executable_iterates_.bind(p.repairExecutableContextLocked());
      p.plan_executable_iterates_.retain(qs,us);
#endif
    };
    auto drive=[&](bool hostile=false) {
      const auto b=batch(p.plan_q_bar_,hostile);const auto stamp=b.header.stamp;
      {
        std::lock_guard<std::mutex> lock(p.mutex_);
        p.current_query_stamp_=stamp;p.last_query_ros_time_=stamp;
        p.repair_witness_requires_reobserve_=false;
        p.pending_batch_=boost::make_shared<care_collision_cdf::CollisionCDFConstraintBatch>(b);
      }
      p.startWorker();bool done=false;
      const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
      while(std::chrono::steady_clock::now()<deadline) {
        {std::lock_guard<std::mutex> lock(p.mutex_);
          done=!p.plan_running_ || p.current_query_stamp_!=stamp;}
        if(done)break;std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
      p.stopWorker();
      if(!done)throw std::runtime_error("worker timeout");
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      return p.current_query_stamp_!=stamp;
    };
    std::ofstream out(argv[2]);out<<std::setprecision(17)<<"{\"controlled_initial_min_m\":"
      <<initial.primitive_executable_min_distance_m<<",\"controlled_initial_cdf_rows\":"
      <<initial.selected_cdf_rows<<",\"scenarios\":[";
    // A retained recorded iterate is seeded explicitly. Both variants then
    // run their actual worker and PIQP on identical q/u/objective/CDF inputs.
    for(int scenario=0;scenario<4;++scenario) {
      setup(scenario==0?1:scenario==3?5:4,scenario);
      const int before=candidates.load();
#if CARE_NEW
      if(scenario==3)p.plan_executable_iterates_=egocentric_arm_planner::RepairExecutableIterates{};
#endif
      bool query=drive();
      const auto restored=p.plan_q_bar_;
      const int consumed_first=p.scp_iteration_;
      double step_scale=1.;bool reserved=false;
#if CARE_NEW
      step_scale=p.plan_executable_iterates_.step_scale;
      reserved=p.plan_executable_iterates_.recheck_only;
#endif
      bool second_query=false;
      if((scenario==1 || scenario==2) && query)second_query=drive(scenario==2);
      if(scenario)out<<',';
      out<<"{\"scenario\":"<<scenario<<",\"first_query\":"<<(query?"true":"false")
        <<",\"second_query\":"<<(second_query?"true":"false")
        <<",\"first_consumed\":"<<consumed_first<<",\"consumed\":"<<p.scp_iteration_
        <<",\"step_scale\":"<<step_scale<<",\"reserved\":"<<(reserved?"true":"false")
        <<",\"restore_inf_rad\":"<<(restored-qs).cwiseAbs().maxCoeff()
        <<",\"candidates\":"<<candidates.load()-before<<",\"center\":";matrix(out,restored);out<<'}';
      if(scenario==0 && candidates.load()!=before)throw std::runtime_error("invalid trial published");
      if((scenario==2 || scenario==3) && candidates.load()!=before)throw std::runtime_error("uncertified/exhausted candidate published");
      if(p.scp_iteration_>6 || ((scenario==1 || scenario==2 || scenario==3) && second_query))
        throw std::runtime_error("SCP budget exceeded");
#if CARE_NEW
      if(scenario==0 && (!query || step_scale!=.5 || (restored-qs).cwiseAbs().maxCoeff()>1e-12))
        throw std::runtime_error("failed trial not restored/shrunk");
      if(scenario==1 && (!reserved || candidates.load()-before!=1 || p.scp_iteration_!=6))
        throw std::runtime_error("fresh final recheck failed");
      if(scenario==3 && query)throw std::runtime_error("last iteration opened extra query");
#endif
    }
    out<<"],\"guard_controls\":[";
#if CARE_NEW
    // Rechecking never suppresses CDF rows or accepts a missing fixed start.
    const auto valid=p.solveSparseSubproblem(batch(qs),qs,us,qb,ub,ub.col(0),{},
      Planner::FrontierObjective{},points,true,false,.15,10.,false,false,.5,true);
    const auto hostile=p.solveSparseSubproblem(batch(qs,true),qs,us,qb,ub,ub.col(0),{},
      Planner::FrontierObjective{},points,true,false,.15,10.,false,false,.5,true);
    const auto shrunk=p.solveSparseSubproblem(batch(qs),qs,us,qb,ub,ub.col(0),{},
      Planner::FrontierObjective{},points,true,false,.075,10.,false,false,.5,false);
    if(!valid.solved || !valid.primitive_executable_valid || valid.max_slack!=0. ||
        hostile.solved || hostile.selected_cdf_rows!=80 || valid.selected_cdf_rows!=80)
      throw std::runtime_error("hard CDF recheck controls failed");
    if(!shrunk.solved)throw std::runtime_error("shrunk hard trial failed");
    double maximum_quantum=0.;
    for(int k:{1,2,3,5})for(int j=0;j<7;++j)
      maximum_quantum=std::max(maximum_quantum,std::abs(shrunk.u(j,k)-us(j,k))/
        (p.repair_executable_settings_.acceleration[j]*.05));
    if(maximum_quantum>.5001 || (shrunk.q-qs).cwiseAbs().maxCoeff()>.0750001)
      throw std::runtime_error("nonlinear failure did not reduce actual QP bounds");
    auto& ledger=p.plan_executable_iterates_;
    ledger.bind("control");ledger.retain(qs,us);
    Eigen::MatrixXd rq,ru;
    if(ledger.reserveRecheck(6,6,qs.col(0),&rq,&ru))throw std::runtime_error("seventh query reserved");
    auto wrong_start=Eigen::VectorXd(qs.col(0));wrong_start[0]+=.001;
    if(ledger.restore(wrong_start,&rq,&ru))throw std::runtime_error("changed start restored");
    ledger.bind("different_target_or_witness");
    if(ledger.available() || ledger.recheck_only || ledger.step_scale!=1.)
      throw std::runtime_error("changed identity reused ledger");
    out<<"{\"fixed_recheck_valid\":true,\"fresh_hostile_cdf_rejected\":true,\"cdf_rows\":80,\"occupied_rows\":"
      <<hostile.selected_occupied_cdf_rows<<",\"max_slack\":0,\"seventh_query_rejected\":true,\"changed_start_rejected\":true,\"changed_identity_cleared\":true,\"shrunk_velocity_update_max_quanta\":"<<maximum_quantum<<'}';
#endif
    out<<"]}\n";spinner.stop();
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
