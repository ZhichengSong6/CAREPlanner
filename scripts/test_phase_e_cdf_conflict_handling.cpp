#include "planner_diagnostic_access.hpp"
#include <boost/property_tree/json_parser.hpp>
#include <boost/make_shared.hpp>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <atomic>
#include <stdexcept>

using Planner = egocentric_arm_planner::LocalSparseSCPPlanner;
using boost::property_tree::ptree;
void require(bool good, const char* why) { if (!good) throw std::runtime_error(why); }
Eigen::VectorXd vec(const ptree& a) {
  Eigen::VectorXd v(a.size());int j=0;
  for(const auto& x:a)v[j++]=x.second.get_value<double>();
  return v;
}
care_collision_cdf::CollisionCDFConstraintBatch plane(const Eigen::MatrixXd& q, bool impossible) {
  care_collision_cdf::CollisionCDFConstraintBatch b;b.dof=7;b.num_pairs=20;
  for(int k=1;k<=20;++k) {
    b.original_timestep.push_back(k);b.source_type.push_back(0);
    b.distance.push_back((impossible ? -.2 : .1)-q(0,k));
    for(double x:{1.,1.,1.})b.point_flat.push_back(x);
    for(int j=0;j<7;++j) {b.gradient_flat.push_back(j==0 ? -1. : 0.);b.q_linearization_flat.push_back(q(j,k));}
  }
  return b;
}

int main(int argc,char** argv) {
  ros::init(argc,argv,"cdf_conflict_validation",ros::init_options::NoSigintHandler);
  try {
    require(argc==3,"fixture and result paths required");
    ros::NodeHandle nh,pnh("~");Planner p;
    require(p.initialize(nh,pnh),"planner initialize failed");p.stopWorker();p.timer_.stop();
    std::ofstream output(argv[2]);output<<std::setprecision(17)<<"{\"frozen_cases\":[";
    ptree input;boost::property_tree::read_json(argv[1],input);bool first=true;
    const auto zero=Eigen::VectorXd::Zero(7).eval();
    const auto u=Eigen::MatrixXd::Zero(7,20).eval();
    for(const auto& item:input.get_child("cases")) {
      const auto& c=item.second;const auto cid=c.get<std::string>("case_id");
      if(cid!="005" && cid!="009" && cid!="027")continue;
      const auto q0=vec(c.get_child("q0"));const auto q=q0.replicate(1,21).eval();
      care_collision_cdf::CollisionCDFConstraintBatch b;b.dof=7;
      std::vector<Eigen::Vector3d> points;p.repair_unknown_witnesses_.clear();
      for(const auto& it:c.get_child("logged_rows")) {
        const auto& row=it.second;const Eigen::Vector3d point=vec(row.get_child("point"));
        if(std::none_of(points.begin(),points.end(),[&](const Eigen::Vector3d& x){return (x-point).norm()<1e-5;})) {
          points.push_back(point);p.repair_unknown_witnesses_.push_back(Planner::RepairWitness{point,1,.05});
        }
        b.original_timestep.push_back(row.get<int>("k"));b.source_type.push_back(0);
        b.distance.push_back(row.get<double>("d"));
        for(const auto& x:row.get_child("point"))b.point_flat.push_back(x.second.get_value<double>());
        for(const auto& x:row.get_child("g"))b.gradient_flat.push_back(x.second.get_value<double>());
        for(const auto& x:row.get_child("qlin"))b.q_linearization_flat.push_back(x.second.get_value<double>());
      }
      b.num_pairs=b.distance.size();
      const auto r=p.solveSparseSubproblem(b,q,u,q,u,zero,{},Planner::FrontierObjective{},points,true,false,.15,10.);
      require(!r.solved && r.selected_cdf_rows==b.num_pairs && r.reachable_conflicting_repair_witness_rows==1,"frozen hard conflict lost");
      require(std::abs(r.worst_repair_witness_reachable_deficit-c.get<double>("production_worst_deficit"))<1e-6,"frozen deficit mismatch");
#if CARE_NEW
      require(r.first_step_conflicting_repair_witness_rows==1 && r.later_conflicting_repair_witness_rows==0,"incorrect frozen stage");
      require(!r.repair_hard_qp_snapshot.empty(),"missing exact primary snapshot");
      std::istringstream ss(r.repair_hard_qp_snapshot);ptree snapshot;boost::property_tree::read_json(ss,snapshot);
      require(snapshot.get_child("cdf_rows").size()==static_cast<std::size_t>(b.num_pairs),"snapshot omitted rows");
      const auto oldmargin=p.solveSparseSubproblem(b,q,u,q,u,zero,{},Planner::FrontierObjective{},points,true,false,.15,10.,false,true);
      require(oldmargin.repair_hard_qp_snapshot.empty(),"rollback diagnostic written as primary");
#endif
      if(!first)output<<',';first=false;
      output<<"{\"case_id\":\""<<cid<<"\",\"solved\":false,\"cdf_rows\":"<<r.selected_cdf_rows
        <<",\"conflicts\":"<<r.reachable_conflicting_repair_witness_rows<<",\"deficit\":"<<r.worst_repair_witness_reachable_deficit<<'}';
    }
    output<<"],\"worker_cases\":[";first=true;
    std::atomic<int> candidates{0};
    auto sub=nh.subscribe<trajectory_msgs::JointTrajectory>(p.candidate_trajectory_topic_,10,
        [&](const trajectory_msgs::JointTrajectoryConstPtr&){++candidates;});
    ros::AsyncSpinner spinner(1);spinner.start();
    int serial=0;
    auto worker=[&](const std::string& id,bool shifted,bool impossible,int iteration,bool stale,bool used,bool expect_reseed) {
      p.stopWorker();
      Eigen::MatrixXd q=Eigen::MatrixXd::Zero(7,21);
      if(shifted)for(int k=1;k<=20;++k)q(0,k)=std::min(.3,.1*k);
      auto b=plane(q,impossible);b.header.stamp=ros::Time::now();
      require(!b.header.stamp.isZero(),"no ROS clock");
      const ros::Time stamp=b.header.stamp;
      {
        std::lock_guard<std::mutex> lock(p.mutex_);
        p.repair_mode_=true;p.probe_mode_=false;p.plan_repair_mode_=true;p.plan_probe_mode_=false;
        p.mode_epoch_=1;p.plan_mode_epoch_=stale ? 0 : 1;
        p.plan_sequence_=++serial;p.plan_running_=true;p.waiting_for_cdf_=false;
        p.plan_repair_observation_phase_=true;p.repair_observation_phase_=true;
        p.repair_witness_requires_reobserve_=false;
        p.repair_no_progress_=egocentric_arm_planner::RepairNoProgress{};
        p.repair_no_progress_.select("fixed_worker");p.plan_repair_ticket_=p.repair_no_progress_.active;
        p.plan_observation_token_="fixed_token";p.latest_observation_token_="fixed_token";
        p.latest_vbc_obligation_points_={Eigen::Vector3d(1.,1.,1.)};
        p.repair_unknown_witnesses_={Planner::RepairWitness{Eigen::Vector3d(1.,1.,1.),1,.05}};
        p.plan_q_bar_=q;p.plan_u_bar_=u;p.plan_q_ref_=Eigen::MatrixXd::Zero(7,21);p.plan_u_ref_=u;
        p.plan_previous_command_=zero;p.plan_schedule_.clear();p.plan_frontier_=Planner::FrontierObjective{};
        p.trust_radius_=.05;p.previous_query_min_distance_=NAN;p.scp_iteration_=iteration;
        p.current_query_stamp_=stamp;p.last_query_ros_time_=stamp;p.current_plan_start_wall_=ros::WallTime::now();
        p.pending_batch_=boost::make_shared<care_collision_cdf::CollisionCDFConstraintBatch>(b);
#if CARE_NEW
        p.plan_repair_conflict_reseed_=egocentric_arm_planner::RepairConflictReseed{};
        p.plan_repair_conflict_reseed_.used=used;
#endif
      }
      const auto before=p.solve_count_;p.startWorker();
      const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(5);bool done=false;
      while(std::chrono::steady_clock::now()<end) {
        {
          std::lock_guard<std::mutex> lock(p.mutex_);
          done=stale ? !p.plan_running_ : (p.solve_count_>before &&
              (!p.plan_running_ || p.current_query_stamp_!=stamp));
        }
        if(done)break;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
      require(done,"worker failed to finish");p.stopWorker();
      const bool requery=p.current_query_stamp_!=stamp;
#if CARE_NEW
      require(requery==expect_reseed,"worker requery decision incorrect");
      require(p.scp_iteration_==iteration+(expect_reseed ? 1 : 0),"SCP budget changed");
      if(expect_reseed)require(p.plan_q_bar_.cwiseAbs().maxCoeff()==0.,"seed not measured hold");
#else
      require(!requery,"before worker unexpectedly requerying");
#endif
      require(candidates.load()==0,"failed QP published executable candidate");
      require(p.repair_unknown_witnesses_.size()==1 && p.repair_unknown_witnesses_[0].safety_margin==.05,"witness changed");
      if(!first)output<<',';first=false;
      output<<"{\"id\":\""<<id<<"\",\"requery\":"<<(requery?"true":"false")
        <<",\"iteration_after\":"<<p.scp_iteration_<<",\"candidate_count\":"<<candidates.load()<<'}';
    };
    worker("later_center_conflict",true,false,1,false,false,true);
    worker("held_start_conflict",false,true,0,false,false,false);
    worker("no_remaining_scp_budget",true,false,p.max_scp_iterations_-1,false,false,false);
    worker("stale_mode_result",true,false,1,true,false,false);
    worker("already_reseeded_conflict",true,false,1,false,true,false);
    for(int i=0;i<7;++i)worker("snapshot_limit_"+std::to_string(i),false,true,0,false,false,false);
#if CARE_NEW
    // Supersession after the worker decision must not arm an old seed using
    // the replacement plan identity. This exercises the actual publisher.
    p.plan_running_=true;
    const auto guarded_stamp=p.current_query_stamp_;
    const Eigen::MatrixXd guarded_q=Eigen::MatrixXd::Zero(7,21);
    auto ticket=p.repair_no_progress_.active;
    require(p.publishQueryTrajectory(guarded_q,u,"base_link",p.plan_sequence_-1,p.mode_epoch_,&ticket).isZero(),"old sequence armed new query");
    require(p.publishQueryTrajectory(guarded_q,u,"base_link",p.plan_sequence_,p.mode_epoch_+1,&ticket).isZero(),"old mode armed new query");
    ++ticket.progress;
    require(p.publishQueryTrajectory(guarded_q,u,"base_link",p.plan_sequence_,p.mode_epoch_,&ticket).isZero(),"old progress armed new query");
    require(p.current_query_stamp_==guarded_stamp,"stale query modified identity");
    p.plan_running_=false;
#endif
    output<<"],\"fresh_hard_solve\":";
    const Eigen::MatrixXd held=Eigen::MatrixXd::Zero(7,21);const auto fresh=plane(held,false);
    const auto r=p.solveSparseSubproblem(fresh,held,u,held,u,zero,{},Planner::FrontierObjective{},
        {Eigen::Vector3d(1.,1.,1.)},true,false,.05,10.);
    require(r.solved && r.selected_cdf_rows==20 && r.max_slack==0.,"fresh hard solve failed");
    for(int k=1;k<=20;++k)require(.1-r.q(0,k)>=.05-1e-5,"original margin violated");
    output<<"{\"solved\":true,\"cdf_rows\":20,\"max_slack\":0},\"candidate_count\":"<<candidates.load();
#if CARE_NEW
    require(p.repair_hard_qp_snapshots_written_.load()==6,"snapshot bound changed");
    output<<",\"snapshot_count\":"<<p.repair_hard_qp_snapshots_written_.load();
#endif
    output<<"}\n";spinner.stop();ros::shutdown();return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';ros::shutdown();return 1;}
}
