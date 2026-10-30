#include "planner_diagnostic_access.hpp"
#include <boost/property_tree/json_parser.hpp>
#include <boost/make_shared.hpp>
#include <fstream>
#include <iomanip>
#include <thread>
using Planner=egocentric_arm_planner::LocalSparseSCPPlanner;
using boost::property_tree::ptree;
Eigen::VectorXd vector(const ptree& t){Eigen::VectorXd q(t.size());int i=0;for(const auto& x:t)q[i++]=x.second.get_value<double>();return q;}
#define CHECK(x) do{if(!(x))throw std::runtime_error(#x);}while(false)
std_msgs::StringConstPtr message(const std::string& s){auto m=boost::make_shared<std_msgs::String>();m->data=s;return m;}
int main(int argc,char** argv){try{
 ros::init(argc,argv,"observation_target_test",ros::init_options::NoSigintHandler);CHECK(argc==3);
 ptree fixture;boost::property_tree::read_json(argv[1],fixture);
 const auto qvis=vector(fixture.get_child("q_vis")),recorded_q=vector(fixture.get_child("measured"));
 const std::string token=fixture.get<std::string>("token");std::vector<Eigen::Vector3d> points;
 for(const auto& x:fixture.get_child("points"))points.push_back(vector(x.second));
 ros::NodeHandle nh,pnh("~");Planner p;CHECK(p.initialize(nh,pnh));p.stopWorker();p.timer_.stop();
 std::vector<std::string> triggers;int candidates=0;
 auto sub=nh.subscribe<std_msgs::String>("/care_planner/local_planner/candidate_replacement_trigger",30,[&](const std_msgs::StringConstPtr& m){triggers.push_back(m->data);});
 auto cs=nh.subscribe<trajectory_msgs::JointTrajectory>(p.candidate_trajectory_topic_,30,[&](const trajectory_msgs::JointTrajectoryConstPtr&){++candidates;});
 ros::WallDuration(.2).sleep();ros::spinOnce();
 const std::vector<std::string> names={"recorded_last_direction","first_activation","active_next_direction","budget_exhausted","already_seen","stale_feedback","foreign_feedback_token","stale_query","far_from_goal","real_goal_progress","duplicate_execution","aborted_execution","safety_hold","stale_measured","feedback_progress","pending_reservation"};
 std::ofstream out(argv[2]);out<<std::setprecision(17)<<"{\"scenarios\":[";
 for(int scenario=0;scenario<int(names.size());++scenario){
  const int prior=scenario==1?0:scenario==2?2:scenario==3?5:4;
  ros::Time::setNow(ros::Time(1000+scenario*10,0));
  p.repair_mode_=true;p.probe_mode_=false;p.mode_epoch_=3;p.repair_observation_phase_=true;
  p.safe_frontier_recovery_enabled_=p.candidate_replacement_enabled_=true;
  p.plan_sequence_=scenario+10;p.plan_running_=p.waiting_for_cdf_=p.plan_requested_=false;
  p.pending_batch_.reset();p.latest_schedule_.clear();p.has_reference_=false;
  p.candidate_replacement_pending_=scenario==15;p.candidate_replacement_trigger_pending_=false;p.candidate_replacement_retry_pending_=false;p.candidate_replacement_last_id_=0;
  p.safe_frontier_recovery_=egocentric_arm_planner::SafeFrontierRecovery{};
  p.repair_no_progress_=egocentric_arm_planner::RepairNoProgress{};p.final_vbc_no_progress_=egocentric_arm_planner::FinalVbcNoProgress{};
  p.repair_candidate_failure_=egocentric_arm_planner::RepairCandidateFailure{};p.execution_response_monitor_=egocentric_arm_planner::ExecutionResponseMonitor{};
  p.latest_observation_token_=token;p.latest_vbc_obligation_id_=4;p.latest_vbc_obligation_points_=points;
  p.latest_single_waypoint_active_=p.has_single_waypoint_q_=true;p.latest_single_waypoint_q_=qvis;
  p.latest_single_waypoint_joint_mask_=vector(fixture.get_child("obligation.q_vis_joint_mask"));
  p.selectRepairTargetLocked();p.safe_frontier_recovery_.attempts=prior;p.safe_frontier_recovery_.active=prior>0;
  p.latest_frontier_.active=true;p.latest_execution_complete_=false;p.vbc_execution_hold_=scenario==12;p.gcdf_execution_hold_=false;
  p.latest_limiter_summary_received_=ros::Time();p.latest_executed_command_=Eigen::VectorXd::Zero(7);p.latest_desired_command_=Eigen::VectorXd::Zero(7);
  const auto trigger_count=p.candidate_replacement_trigger_count_;const std::size_t before=triggers.size();
  for(int i=0;i<6;++i){
   ros::Time::setNow(ros::Time(1000+scenario*10,i*250000000));const auto now=ros::Time::now();
   Eigen::VectorXd q=recorded_q;if(scenario==8)q[0]+=.1;if(scenario==9)q[0]+=.012-.002*i;
   p.has_joint_state_=true;p.latest_joint_state_.header.stamp=now-ros::Duration(scenario==13?.3:0.);
   p.latest_joint_state_.name=p.joint_names_;p.latest_joint_state_.position.assign(q.data(),q.data()+7);
   p.latest_joint_state_received_=now;p.latest_desired_command_received_=now;
   p.current_query_stamp_=p.last_query_ros_time_=now-ros::Duration(scenario==7?2.:0.);
   p.current_query_wall_=ros::WallTime::now();
   std::ostringstream feedback;feedback<<std::setprecision(17)<<"policy=visibility_acquisition active_query_status=ok active_obligation_id=4 active_point_count=3 active_inside_map_count=3 active_finite_confidence_count=3 active_finite_current_visibility_count=3 active_seen="<<(scenario==4?1:0)
    <<" active_seen_fraction="<<(scenario==4?1.:scenario==14?double(i)/9.:2./3.)<<" active_min_confidence=0 active_max_current_visibility=1 active_query_ros_s="<<(now.toSec()-(scenario==5?1.:0.))<<" active_feedback_token="<<(scenario==6?"care_obs_v1_stale":token);
   p.observationProgressCallback(message(feedback.str()));
   const auto execution=(scenario==10?ros::Time(1000+scenario*10,0):now).toNSec()-100000000ULL;
   p.safe_frontier_recovery_.following_frontier=true;
   p.safe_frontier_recovery_.remember(execution-1,.02);p.safe_frontier_recovery_.certify(execution-1,execution);
   std::ostringstream tracker;tracker<<"active=1 complete=0 execution_aborted="<<(scenario==11?1:0)<<" execution_stamp_ns="<<execution<<" phase_s=0.1";
   p.executionSummaryCallback(message(tracker.str()));
   ros::WallDuration(.005).sleep();ros::spinOnce();
  }
  ros::WallDuration(.02).sleep();ros::spinOnce();
  const bool expected=scenario<=2;const bool queued=p.candidate_replacement_trigger_pending_;
#if CARE_FIXED
  CHECK(queued==expected);
#else
  CHECK(!queued);
#endif
  const int attempts_before_grant=p.safe_frontier_recovery_.attempts;bool granted=false;
  if(queued){
   CHECK(p.candidate_replacement_trigger_reason_=="observation_no_progress");
   CHECK(p.candidate_replacement_trigger_raw_==0 && !p.plan_requested_);
   CHECK(p.candidate_replacement_trigger_count_==trigger_count+1);
   std::ostringstream reserve;reserve<<"action=reserve request_id=1 reason=observation_no_progress mode_epoch=3 query_stamp_ns="<<p.current_query_stamp_.toNSec()<<" observation_token="<<token<<" obligation_id=4 trigger_raw_candidate_stamp_ns=0";
   p.candidateReplacementCallback(message(reserve.str()));granted=p.candidate_replacement_pending_;CHECK(granted);
   CHECK(p.safe_frontier_recovery_.attempts==(scenario==0?5:scenario==1?2:3));
   const int charged=p.safe_frontier_recovery_.attempts;
   p.candidateReplacementCallback(message(reserve.str()));CHECK(p.safe_frontier_recovery_.attempts==charged);
   p.candidateReplacementCallback(message("action=finish request_id=1 new_token=none generation_result=no_geometric_candidate"));
   CHECK(!p.candidate_replacement_pending_ && !p.candidate_replacement_retry_pending_);
   std::string duplicate=reserve.str();duplicate.replace(duplicate.find("request_id=1"),12,"request_id=2");
   p.candidateReplacementCallback(message(duplicate));CHECK(!p.candidate_replacement_pending_ && p.safe_frontier_recovery_.attempts==charged);
   if(scenario==0)CHECK(!p.safe_frontier_recovery_.canReserveReplacement());
  }
  CHECK(candidates==0);CHECK(p.max_scp_iterations_==6);
  if(scenario)out<<',';out<<"{\"name\":\""<<names[scenario]<<"\",\"prior_attempts\":"<<prior<<",\"after_stall\":"<<attempts_before_grant<<",\"triggered\":"<<(queued?"true":"false")<<",\"granted\":"<<(granted?"true":"false")<<",\"after_grant\":"<<p.safe_frontier_recovery_.attempts<<",\"trigger_messages\":"<<triggers.size()-before<<'}';
 }
 out<<"],\"candidates\":"<<candidates<<",\"triggers\":[";
 for(std::size_t i=0;i<triggers.size();++i){if(i)out<<',';out<<'"'<<triggers[i]<<'"';}out<<"]}\n";
 std::cout<<"PASS 16 native observation-stall scenarios\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
