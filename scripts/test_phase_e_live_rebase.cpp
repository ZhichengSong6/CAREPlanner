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
Eigen::VectorXd vec(const ptree& t){Eigen::VectorXd v(t.size());int i=0;for(const auto& x:t)v[i++]=x.second.get_value<double>();return v;}
Eigen::MatrixXd mat(const ptree& t){Eigen::MatrixXd m(t.front().second.size(),t.size());int k=0;for(const auto& x:t)m.col(k++)=vec(x.second);return m;}
#define CHECK(x) do {if(!(x))throw std::runtime_error(#x);}while(false)
int main(int argc,char** argv){
 try {
  ros::init(argc,argv,"live_rebase_test",ros::init_options::NoSigintHandler);
  CHECK(argc==3);ptree fixtures;boost::property_tree::read_json(argv[1],fixtures);
  ros::NodeHandle nh,pnh("~");Planner p;CHECK(p.initialize(nh,pnh));p.stopWorker();p.timer_.stop();
  p.repair_prefix_binding_enabled_=false;p.candidate_replacement_enabled_=false;
  p.repair_task_tracking_scale_=1.;p.q_tracking_weight_=1000.;p.terminal_q_tracking_weight_=0.;
  p.u_tracking_weight_=1000.;p.u_reference_tracking_enabled_=true;p.u_smooth_weight_=0.;p.handoff_velocity_weight_=0.;
  p.scp_step_tolerance_inf_=1.;
  std::atomic<int> candidates{0};auto sub=nh.subscribe<trajectory_msgs::JointTrajectory>(p.candidate_trajectory_topic_,20,
   [&](const trajectory_msgs::JointTrajectoryConstPtr&){++candidates;});
  ros::AsyncSpinner spinner(1);spinner.start();std::this_thread::sleep_for(std::chrono::milliseconds(100));
  std::ofstream out(argv[2]);out<<std::setprecision(17)<<"{\"scenarios\":[";bool first=true;int serial=0;
  for(const auto& entry:fixtures){const auto& fixture=entry.second;
   const auto q=mat(fixture.get_child("q")),u=mat(fixture.get_child("u"));const auto measured=vec(fixture.get_child("measured"));
   p.repair_executable_settings_.prefixes={fixture.get<double>("prefix")};
   std::vector<Planner::RepairWitness> witnesses;std::vector<Eigen::Vector3d> points;
   for(const auto& item:fixture.get_child("points")){const auto& x=item.second;const auto point=vec(x.get_child("point"));
    Planner::RepairWitness w{point,1,.05,true,-1.,.3};w.has_primitive_guard=true;w.primitive_guard.point=point;
    w.primitive_guard.sweeping_link=x.get<std::string>("sweeping_link");w.primitive_guard.collision_name=x.get<std::string>("source_collision_name");
    w.primitive_guard.collision_index=x.get<int>("source_collision_index");witnesses.push_back(w);points.push_back(point);
   }
   auto batch=[&](bool hostile){care_collision_cdf::CollisionCDFConstraintBatch b;b.dof=7;b.header.stamp=ros::Time::now();
    for(int k=1;k<=20;++k)for(const auto& point:points){b.original_timestep.push_back(k);b.source_type.push_back(0);b.distance.push_back(hostile?-.5:.5);
     for(int j=0;j<3;++j)b.point_flat.push_back(point[j]);for(int j=0;j<7;++j){b.gradient_flat.push_back(0.);b.q_linearization_flat.push_back(p.plan_q_bar_(j,k));}}
    b.num_pairs=b.distance.size();return b;};
   for(int scenario=0;scenario<4;++scenario){p.stopWorker();
    {std::lock_guard<std::mutex> lock(p.mutex_);
     p.repair_mode_=p.plan_repair_mode_=true;p.probe_mode_=p.plan_probe_mode_=false;p.mode_epoch_=p.plan_mode_epoch_=1;
     p.plan_sequence_=100+(++serial);p.plan_running_=true;p.waiting_for_cdf_=p.plan_requested_=false;
     p.repair_observation_phase_=p.plan_repair_observation_phase_=true;p.repair_witness_requires_reobserve_=p.repair_witness_overflow_=false;
     p.repair_unknown_witnesses_=witnesses;p.repair_no_progress_=egocentric_arm_planner::RepairNoProgress{};
     p.final_vbc_no_progress_=egocentric_arm_planner::FinalVbcNoProgress{};p.safe_frontier_recovery_=egocentric_arm_planner::SafeFrontierRecovery{};
     p.repair_candidate_failure_=egocentric_arm_planner::RepairCandidateFailure{};
     p.latest_observation_token_=p.plan_observation_token_="care_obs_v1_1_controlled";
     p.latest_vbc_obligation_points_=points;p.latest_vbc_obligation_id_=1;p.latest_single_waypoint_active_=p.has_single_waypoint_q_=true;
     p.latest_single_waypoint_q_=q.col(20);p.latest_single_waypoint_joint_mask_=Eigen::VectorXd::Ones(7);p.selectRepairTargetLocked();
     p.plan_repair_ticket_=p.repair_no_progress_.active;p.repair_witness_ticket_=p.plan_repair_ticket_;p.repair_witness_mode_epoch_=1;
     p.plan_q_bar_=p.plan_previous_q_bar_=q;p.plan_u_bar_=p.plan_previous_u_bar_=u;p.plan_q_current_=q.col(0);p.plan_q_ref_=q;p.plan_u_ref_=u;
     p.plan_previous_command_=p.latest_executed_command_=u.col(0);p.plan_schedule_.clear();p.plan_frontier_=Planner::FrontierObjective{};
     p.plan_has_hard_solution_=false;p.plan_executable_iterates_=egocentric_arm_planner::RepairExecutableIterates{};
     p.scp_iteration_=scenario==1?5:0;p.max_scp_iterations_=6;p.trust_radius_=.15;p.current_plan_start_wall_=ros::WallTime::now();
     p.previous_query_min_distance_=std::numeric_limits<double>::quiet_NaN();p.has_joint_state_=true;
     p.latest_joint_state_.header.stamp=ros::Time::now();p.latest_joint_state_.name=p.joint_names_;
     const Eigen::VectorXd actual=scenario==2?q.col(0).eval():measured;
     p.latest_joint_state_.position.assign(actual.data(),actual.data()+7);
    }
    auto b=batch(false);
    if(scenario==3){
     // Each first-step row fits the individual box; the pair is contradictory.
     // Preset the old capture count to prove the latest failure survives it.
     p.repair_hard_qp_snapshots_written_=6;
     b.distance[0]=0.;b.gradient_flat[0]=10.;
     b.original_timestep.push_back(1);b.source_type.push_back(0);b.distance.push_back(0.);
     for(int j=0;j<3;++j)b.point_flat.push_back(points[0][j]+(j==0?.05:0.));
     for(int j=0;j<7;++j){b.gradient_flat.push_back(j==0?-10.:0.);b.q_linearization_flat.push_back(p.plan_q_bar_(j,1));}
     ++b.num_pairs;
    }
    const auto stamp=b.header.stamp;const int before=candidates.load();
    {std::lock_guard<std::mutex> lock(p.mutex_);p.current_query_stamp_=p.last_query_ros_time_=stamp;
     p.current_query_wall_=ros::WallTime::now();p.pending_batch_=boost::make_shared<care_collision_cdf::CollisionCDFConstraintBatch>(b);}
    p.startWorker();bool done=false;const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    while(std::chrono::steady_clock::now()<deadline){{std::lock_guard<std::mutex> lock(p.mutex_);done=!p.plan_running_||p.current_query_stamp_!=stamp;}
     if(done)break;std::this_thread::sleep_for(std::chrono::milliseconds(1));}
    p.stopWorker();CHECK(done);std::this_thread::sleep_for(std::chrono::milliseconds(30));
    const bool query=p.current_query_stamp_!=stamp;const int emitted=candidates.load()-before;
    const double root_error=(p.plan_q_bar_.col(0)-measured).lpNorm<Eigen::Infinity>();
    if(scenario==0){
#if CARE_FIXED
     CHECK(query && emitted==0 && root_error<1e-12 && p.scp_iteration_==1);
#else
     CHECK(!query && emitted==1);
#endif
    }else if(scenario==1){
#if CARE_FIXED
     CHECK(!query && emitted==0 && p.scp_iteration_==6);
#else
     CHECK(!query && emitted==1);
#endif
    }else if(scenario==2){CHECK(!query && emitted==1);}else{CHECK(emitted==0);}
    if(!first)out<<',';first=false;out<<"{\"case\":\""<<fixture.get<std::string>("case")<<"\",\"scenario\":"<<scenario
     <<",\"new_query\":"<<(query?"true":"false")<<",\"candidates\":"<<emitted<<",\"consumed\":"<<p.scp_iteration_
     <<",\"new_root_error\":"<<root_error<<'}';
   }
  }
  out<<"]}";spinner.stop();return 0;
 }catch(const std::exception& e){ROS_ERROR_STREAM(e.what());return 1;}
}
