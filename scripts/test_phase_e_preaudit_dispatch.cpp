#include "planner_diagnostic_access.hpp"
#include <boost/property_tree/json_parser.hpp>
#include <cassert>
#include <fstream>
#include <iostream>
using Planner=egocentric_arm_planner::LocalSparseSCPPlanner;
using boost::property_tree::ptree;
Eigen::VectorXd values(const ptree& t){Eigen::VectorXd v(t.size());int i=0;for(const auto& x:t)v[i++]=x.second.get_value<double>();return v;}
int main(int argc,char** argv) {
  if(argc!=4)return 2;
  ros::init(argc,argv,"restore_dispatch_test",ros::init_options::NoSigintHandler);
  ros::NodeHandle nh,pnh("~");ptree input;boost::property_tree::read_json(argv[3],input);
  int controls=0,replayed=0;
  for(const auto& record:input) {
    const auto& f=record.second;
    for(int group=0;group<11;++group) {
      Planner p;assert(p.initialize(nh,pnh));p.stopWorker();p.timer_.stop();
      p.repair_prefix_binding_enabled_=true;p.repair_mode_=true;p.probe_mode_=false;
      p.repair_observation_phase_=true;p.candidate_replacement_enabled_=false;
      p.mode_epoch_=f.get<unsigned long long>("epoch");p.latest_observation_token_=f.get<std::string>("token");
      p.latest_vbc_obligation_id_=3;p.latest_vbc_obligation_points_={Eigen::Vector3d(.25,0.,.95)};
      p.latest_single_waypoint_active_=p.has_single_waypoint_q_=true;
      p.latest_single_waypoint_q_=values(f.get_child("points").front().second.get_child("q"));
      p.latest_single_waypoint_joint_mask_=Eigen::VectorXd::Ones(7);
      auto& prefix=p.latest_prefix_selection_;prefix.session=std::string(32,'a');prefix.revision=1;prefix.prefix_s=.25;
      prefix.brake_dt_s=p.repair_executable_settings_.brake_dt;prefix.brake_steps=p.repair_executable_settings_.max_brake_steps;
      prefix.acceleration.assign(p.repair_executable_settings_.acceleration.data(),
          p.repair_executable_settings_.acceleration.data()+7);
      p.selectRepairTargetLocked();
      const auto once=std::to_string(p.mode_epoch_)+":"+p.safe_frontier_recovery_.key;
      p.state_evidence_retry_used_[once]=true;
      const int count=group==10?12:1;
      for(int n=0;n<count;++n) {
        egocentric_arm_planner::CandidateTransferPtr event(new egocentric_arm_planner::CandidateTransfer);
        event->action="preaudit_expired";event->session=f.get<std::string>("session");
        event->receiver_boot_id=f.get<std::string>("boot");event->sequence=f.get<unsigned long long>("sequence");
        event->payload_sha256=f.get<std::string>("digest");event->planned_prefix_context="controlled-original-prefix-context";
        auto& raw=event->trajectory;const auto id=f.get<unsigned long long>("raw")+n;
        raw.header.stamp.fromNSec(id);raw.header.frame_id="base_link";raw.joint_names=p.joint_names_;
        for(const auto& item:f.get_child("points")) {
          const auto q=values(item.second.get_child("q")),v=values(item.second.get_child("v"));
          trajectory_msgs::JointTrajectoryPoint point;point.positions.assign(q.data(),q.data()+q.size());
          point.velocities.assign(v.data(),v.data()+v.size());point.time_from_start.fromNSec(item.second.get<unsigned long long>("ns"));
          raw.points.push_back(point);
        }
        p.admission_pending_.emplace(id,Planner::AdmissionCandidate{*event,prefix,p.mode_epoch_,p.latest_observation_token_,p.safe_frontier_recovery_.key});
        assert(p.final_vbc_no_progress_.remember(id,p.mode_epoch_,p.plan_sequence_));p.safe_frontier_recovery_.remember(id,.02);
        p.plan_requested_=false;
        if(group==1)event->trajectory.points[0].positions[0]+=.001;
        if(group==2)event->planned_prefix_context+="wrong";
        if(group==3)++p.mode_epoch_;
        if(group==4)p.latest_observation_token_+="wrong";
        if(group==5)++prefix.revision;
        if(group==6)++p.safe_frontier_recovery_.generation;
        if(group==7)p.final_vbc_no_progress_.pending.erase(id);
        if(group==8)event->action="expired";
        if(group==9)event->session="bad";
        if(group==0) {
          // Actual production subscription, with its worker stopped and no
          // robot or actuator process. Only the finite disposition runs.
          auto pub=nh.advertise<egocentric_arm_planner::CandidateTransfer>("/care_planner/local_planner/candidate_trajectory/admission_terminal",1);
          const auto until=ros::WallTime::now()+ros::WallDuration(2.);
          while(!pub.getNumSubscribers() && ros::WallTime::now()<until){ros::spinOnce();ros::WallDuration(.005).sleep();}
          assert(pub.getNumSubscribers());pub.publish(event);
          while(p.admission_pending_.count(id) && ros::WallTime::now()<until){ros::spinOnce();ros::WallDuration(.005).sleep();}
        } else p.admissionTerminalCallback(event);
        const auto attempts=p.safe_frontier_recovery_.attempts;const auto rejected=p.safe_frontier_recovery_.rejected;
        p.admissionTerminalCallback(event);assert(attempts==p.safe_frontier_recovery_.attempts && rejected==p.safe_frontier_recovery_.rejected);
        if(group>0 && group<10)assert(!p.plan_requested_);
        else assert(p.plan_requested_==(n<11));
        assert(p.repair_unknown_witnesses_.empty() && p.restore_path_guard_points_.empty());
        assert(p.state_evidence_retry_used_.size()==1 && p.state_evidence_retry_pending_.empty());
      }
      if(group==10)assert(p.safe_frontier_recovery_.attempts==6 && p.safe_frontier_recovery_.blocked());
      ++controls;
    }
    ++replayed;
  }
  std::ofstream out(argv[2]);out<<"{\"status\":\"PASS\",\"recorded_payloads\":"<<replayed<<",\"controls\":"<<controls
      <<",\"real_ros_subscription\":true,\"finite_five_directions\":true,\"no_audit_or_collision_witness\":true}";
  std::cout<<"PASS "<<controls<<" controls on "<<replayed<<" recorded raw payloads\n";
}
