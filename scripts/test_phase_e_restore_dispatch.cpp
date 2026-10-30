// Exercise the real worker, hard QP and candidate publication with saved O5 rows.
#include "planner_diagnostic_access.hpp"
#include <boost/property_tree/json_parser.hpp>
#include <boost/make_shared.hpp>
#include <fstream>
#include <iomanip>
#include <set>
#include <stdexcept>

using Planner = egocentric_arm_planner::LocalSparseSCPPlanner;
using boost::property_tree::ptree;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (false)

Eigen::VectorXd vec(const ptree& t) {
  Eigen::VectorXd v(t.size()); int j=0;
  for (const auto& x:t) v[j++]=x.second.get_value<double>();
  return v;
}
Eigen::MatrixXd matrix(const ptree& t) {
  Eigen::MatrixXd m(7,t.size()); int k=0;
  for (const auto& x:t) m.col(k++)=vec(x.second);
  return m;
}
std_msgs::StringConstPtr message(const std::string& s) {
  auto m=boost::make_shared<std_msgs::String>(); m->data=s; return m;
}

int main(int argc,char** argv) {
  ros::init(argc,argv,"restore_dispatch_test",ros::init_options::NoSigintHandler);
  try {
    CHECK(argc==3); ptree fixtures; boost::property_tree::read_json(argv[1],fixtures);
    ros::NodeHandle nh,pnh("~"); Planner p; CHECK(p.initialize(nh,pnh));
    p.stopWorker(); p.timer_.stop();
    std::vector<std::string> events,triggers;
    std::vector<trajectory_msgs::JointTrajectory> candidates;
    auto es=nh.subscribe<std_msgs::String>(p.summary_topic_,100,[&](const std_msgs::StringConstPtr& m){events.push_back(m->data);});
    auto ts=nh.subscribe<std_msgs::String>("/care_planner/local_planner/candidate_replacement_trigger",100,[&](const std_msgs::StringConstPtr& m){triggers.push_back(m->data);});
    auto cs=nh.subscribe<trajectory_msgs::JointTrajectory>(p.candidate_trajectory_topic_,100,[&](const trajectory_msgs::JointTrajectoryConstPtr& m){candidates.push_back(*m);});
    ros::WallDuration(.25).sleep(); ros::spinOnce();
    std::ofstream out(argv[2]); out<<std::setprecision(17)<<"{\"cases\":[";
    int index=0;
    for (const auto& item:fixtures.get_child("cases")) {
      const auto& f=item.second; const auto& s=f.get_child("subproblem");
      const auto q=matrix(s.get_child("q_bar")),u=matrix(s.get_child("u_bar"));
      const auto qvis=vec(f.get_child("q_vis"));
      const std::string token=f.get<std::string>("token");
      ros::Time now(1000+index*10,0); ros::Time::setNow(now);
      p.repair_mode_=p.plan_repair_mode_=true; p.probe_mode_=p.plan_probe_mode_=false;
      p.repair_observation_phase_=p.plan_repair_observation_phase_=true;
      p.safe_frontier_recovery_enabled_=p.candidate_replacement_enabled_=true;
      p.mode_epoch_=p.plan_mode_epoch_=3; p.plan_sequence_=100+index;
      p.plan_requested_=false; p.plan_running_=p.waiting_for_cdf_=true;
      p.candidate_replacement_pending_=p.candidate_replacement_trigger_pending_=p.candidate_replacement_retry_pending_=false;
      p.safe_frontier_recovery_=egocentric_arm_planner::SafeFrontierRecovery{};
      p.repair_no_progress_=egocentric_arm_planner::RepairNoProgress{};
      p.final_vbc_no_progress_=egocentric_arm_planner::FinalVbcNoProgress{};
      p.repair_candidate_failure_=egocentric_arm_planner::RepairCandidateFailure{};
      p.plan_repair_conflict_reseed_=egocentric_arm_planner::RepairConflictReseed{};
      p.plan_executable_iterates_=egocentric_arm_planner::RepairExecutableIterates{};
      p.repair_unknown_witnesses_.clear(); p.latest_vbc_obligation_points_.clear();
      p.gcdf_feedback_pending_.clear();p.geometric_restore_scope_.clear();
      p.latest_observation_token_=p.plan_observation_token_=token;p.latest_vbc_obligation_id_=5;
      p.latest_single_waypoint_active_=p.has_single_waypoint_q_=true;
      p.latest_single_waypoint_q_=qvis;p.latest_single_waypoint_joint_mask_=Eigen::VectorXd::Ones(7);
      p.latest_schedule_.clear();p.plan_schedule_.clear();
      care_collision_cdf::CollisionCDFConstraintBatchPtr batch(new care_collision_cdf::CollisionCDFConstraintBatch);
      batch->header.stamp=now-ros::Duration(.01);batch->header.frame_id="base_link";batch->dof=7;
      std::set<std::string> seen;
      for (const auto& x:s.get_child("cdf_rows")) {
        const auto& r=x.second;const Eigen::Vector3d point=vec(r.get_child("point"));
        batch->original_timestep.push_back(r.get<int>("timestep"));batch->source_type.push_back(r.get<int>("source"));
        batch->distance.push_back(r.get<double>("distance"));
        for(int j=0;j<3;++j)batch->point_flat.push_back(point[j]);
        const auto g=vec(r.get_child("gradient")),center=vec(r.get_child("q_linearization"));
        for(int j=0;j<7;++j){batch->gradient_flat.push_back(g[j]);batch->q_linearization_flat.push_back(center[j]);}
        std::ostringstream key;key<<std::setprecision(17)<<point.transpose();
        if(seen.insert(key.str()).second){
          p.latest_vbc_obligation_points_.push_back(point);
          Planner::RepairWitness w{point,1,r.get<double>("margin")};
          // Explicit control: previous native-CDF ladder is 0.15. Saved
          // snapshots contain the current 0.20 rows, not the full old ledger.
          w.previous_margin=.15;p.repair_unknown_witnesses_.push_back(w);
        }
      }
      batch->num_pairs=batch->distance.size();
      p.selectRepairTargetLocked();p.safe_frontier_recovery_.active=true;
      p.safe_frontier_recovery_.attempts=index+1;
      p.plan_repair_ticket_=p.repair_no_progress_.active;
      p.repair_witness_requires_reobserve_=p.repair_witness_overflow_=false;
      p.current_query_stamp_=batch->header.stamp;p.last_query_ros_time_=batch->header.stamp;
      p.current_query_wall_=p.current_plan_start_wall_=ros::WallTime::now();
      p.plan_q_current_=q.col(0);p.plan_q_bar_=q;p.plan_u_bar_=u;
      p.plan_q_ref_=matrix(s.get_child("q_ref"));p.plan_u_ref_=Eigen::MatrixXd::Zero(7,20);
      p.plan_previous_command_=vec(s.get_child("previous_command"));
      p.scp_iteration_=f.get<int>("scp_iteration");p.trust_radius_=s.get<double>("trust_radius");
      p.plan_frontier_=Planner::FrontierObjective{};
      p.plan_frontier_.active=true;p.plan_frontier_.frontier_weight_scale=1.;p.plan_frontier_.qvis_weight_scale=0.;
      p.plan_frontier_.q=q.col(0)+(qvis-q.col(0)).cwiseMax(-.05).cwiseMin(.05);
      p.plan_frontier_.joint_mask=Eigen::VectorXd::Ones(7);p.plan_frontier_.recovery_attempt=index+1;
      p.plan_frontier_.recovery_target=p.plan_frontier_.q;
      p.has_joint_state_=true;p.latest_joint_state_.header.stamp=now;
      p.latest_joint_state_.name=p.joint_names_;p.latest_joint_state_.position.assign(q.col(0).data(),q.col(0).data()+7);
      p.latest_joint_state_.velocity.assign(p.plan_previous_command_.data(),p.plan_previous_command_.data()+7);
      p.latest_joint_state_received_=now;
      auto& prefix=p.latest_prefix_selection_;prefix.header.stamp=now;prefix.session=std::string(32,'a');
      prefix.revision=index+1;prefix.prefix_s=.25;prefix.brake_dt_s=.05;prefix.brake_steps=20;
      prefix.acceleration={3.,3.,4.,4.,6.,6.,6.};p.plan_prefix_selection_=prefix;
      const auto nc=candidates.size(),nt=triggers.size(),ne=events.size();
      p.startWorker();{std::lock_guard<std::mutex> lock(p.mutex_);p.pending_batch_=batch;}p.worker_cv_.notify_one();
      const auto deadline=ros::WallTime::now()+ros::WallDuration(8.);
      bool published=false;
      while(ros::WallTime::now()<deadline){
        ros::spinOnce();
        for(std::size_t i=ne;i<events.size();++i)if(events[i].find("event=geometric_restore_candidate_published ")!=std::string::npos)published=true;
        if(published && candidates.size()>nc)break;
        ros::WallDuration(.005).sleep();
      }
      p.stopWorker();ros::WallDuration(.03).sleep();ros::spinOnce();
      CHECK(published && candidates.size()==nc+1);
      const bool competing=triggers.size()>nt;
#if CARE_BEFORE
      CHECK(competing && p.candidate_replacement_trigger_pending_);
#else
      CHECK(!competing && !p.candidate_replacement_trigger_pending_ && !p.candidate_replacement_pending_);
      CHECK(!p.plan_requested_);
      const auto raw=candidates.back().header.stamp.toNSec();
      CHECK(p.final_vbc_no_progress_.pending.count(raw)==1);
      // A delayed reservation without a live trigger cannot steal this path.
      std::ostringstream stale;stale<<"action=reserve request_id="<<index+1
          <<" reason=repair_qp_failure mode_epoch=3 query_stamp_ns="<<batch->header.stamp.toNSec()
          <<" observation_token="<<token<<" obligation_id=5 trigger_raw_candidate_stamp_ns=0";
      const int before=p.safe_frontier_recovery_.attempts;
      p.candidateReplacementCallback(message(stale.str()));
      CHECK(!p.candidate_replacement_pending_ && p.safe_frontier_recovery_.attempts==before);
      // A real terminal certification failure remains finite and idempotent.
      const std::string rejection="result=protocol_invalid committed=0 raw_candidate_stamp_ns="+std::to_string(raw)+
          " audited_trajectory_stamp_ns="+std::to_string(raw+1)+" vbc_protocol_reason=geometric_restore_certificate_unavailable";
      p.finalVerificationCallback(message(rejection));const int charged=p.safe_frontier_recovery_.attempts;
      p.finalVerificationCallback(message(rejection));CHECK(p.safe_frontier_recovery_.attempts==charged);
      CHECK(charged==before && p.safe_frontier_recovery_.rejected==1);
      const auto second=raw+2;
      CHECK(p.final_vbc_no_progress_.remember(second,p.mode_epoch_,p.plan_sequence_));
      p.safe_frontier_recovery_.remember(second,.03);
      const std::string second_rejection="result=protocol_invalid committed=0 raw_candidate_stamp_ns="+std::to_string(second)+
          " audited_trajectory_stamp_ns="+std::to_string(second+1)+" vbc_protocol_reason=geometric_restore_certificate_unavailable";
      p.finalVerificationCallback(message(second_rejection));
      CHECK(p.safe_frontier_recovery_.attempts==before+1);
      p.finalVerificationCallback(message(second_rejection));
      CHECK(p.safe_frontier_recovery_.attempts==before+1 && p.safe_frontier_recovery_.attempts<=5);
#endif
      if(index)out<<',';
      out<<"{\"source_query\":\""<<s.get<std::string>("query_stamp_ns")<<"\",\"raw_candidate\":\""<<f.get<std::string>("raw_stamp_ns")
         <<"\",\"candidate_published\":true,\"competing_replacement\":"<<(competing?"true":"false")
         <<",\"selected_cdf_rows\":"<<batch->num_pairs<<",\"attempts\":"<<p.safe_frontier_recovery_.attempts<<'}';
      ++index;
    }
    out<<"],\"status\":\"PASS\"}\n";CHECK(index==4);ros::shutdown();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';ros::shutdown();return 1;}
}
