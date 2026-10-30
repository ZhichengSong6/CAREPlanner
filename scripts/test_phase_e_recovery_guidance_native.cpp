#include "planner_diagnostic_access.hpp"
#include <egocentric_arm_planner/geometric_restore_proposal.hpp>
#include <egocentric_arm_planner/qp_snapshot.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>

using Planner=egocentric_arm_planner::LocalSparseSCPPlanner;
using boost::property_tree::ptree;
using namespace egocentric_arm_planner;
#define CHECK(x) do {if(!(x))throw std::runtime_error(#x);}while(false)
Eigen::VectorXd vec(const ptree& t) {
  Eigen::VectorXd v(t.size());int i=0;for(const auto& x:t)v[i++]=x.second.get_value<double>();return v;
}
int main(int argc,char** argv) {
  try {
    ros::init(argc,argv,"recovery_guidance_native",ros::init_options::NoSigintHandler);
    CHECK(argc==3);ros::NodeHandle nh,pnh("~");Planner p;CHECK(p.initialize(nh,pnh));p.stopWorker();p.timer_.stop();
    ros::Time::setNow(ros::Time(100.));
    ptree fixtures;boost::property_tree::read_json(argv[1],fixtures);
    std::ofstream output(argv[2]);output<<"{\"scenarios\":[";bool first=true;int checks=0;
    for(const auto& entry:fixtures) {
      const auto& f=entry.second;const auto measured=vec(f.get_child("q"));
      p.latest_observation_token_=f.get<std::string>("token");p.latest_observation_region_=f.get<std::string>("region");
      p.latest_observation_sensor_=f.get<int>("sensor");p.mode_epoch_=73;
      p.has_joint_state_=p.has_single_waypoint_q_=p.latest_single_waypoint_active_=true;
      p.latest_single_waypoint_q_=vec(f.get_child("q_vis"));p.latest_single_waypoint_joint_mask_=vec(f.get_child("mask"));
      p.latest_joint_state_.name=p.joint_names_;p.latest_joint_state_.position.assign(measured.data(),measured.data()+7);
      const auto velocity=vec(f.get_child("velocity"));p.latest_joint_state_.velocity.assign(velocity.data(),velocity.data()+7);
      p.latest_executed_command_=velocity;
      Planner::SparseSolveResult failure;
      for(const auto& point:f.get_child("restore_points"))failure.geometric_restore_guard_points.push_back(vec(point.second));
      p.restore_path_guard_scope_.clear();p.restore_path_guard_points_.clear();
      p.safe_frontier_recovery_=SafeFrontierRecovery{};p.safe_frontier_recovery_.attempts=2;p.safe_frontier_recovery_.segments=1;
      VisibilityGuidanceState guide;
      guide.token=p.latest_observation_token_;guide.region=p.latest_observation_region_;guide.sensor=p.latest_observation_sensor_;
      guide.epoch=p.mode_epoch_;guide.sequence=9;guide.value=f.get<double>("model_value");guide.drive=true;
      guide.q_eval=measured;guide.target=vec(f.get_child("target"));guide.gradient=vec(f.get_child("gradient"));
      guide.mask=p.latest_single_waypoint_joint_mask_;
      Eigen::MatrixXd plain_q,plain_u;
      for(int variant=0;variant<13;++variant) {
        const auto now=ros::Time::now();p.latest_joint_state_.header.stamp=now;
        p.rolling_visibility_guidance_enabled_=variant!=1;p.latest_visibility_guidance_=guide;p.latest_visibility_guidance_.stamp=now.toSec();
        if(variant==2)p.latest_visibility_guidance_.stamp-=.201;
        if(variant==3)p.latest_visibility_guidance_.stamp+=.001;
        if(variant==4)p.latest_visibility_guidance_.token="care_obs_v1_foreign";
        if(variant==5)p.latest_visibility_guidance_.region="care_region_v1_foreign";
        if(variant==6)p.latest_visibility_guidance_.epoch++;
        if(variant==7)p.latest_visibility_guidance_.sensor=(guide.sensor+1)%8;
        if(variant==8)p.latest_visibility_guidance_.q_eval[0]+=.021;
        if(variant==9)p.latest_visibility_guidance_.mask[0]=1.-guide.mask[0];
        if(variant==10)p.latest_visibility_guidance_.drive=false;
        if(variant==11)p.latest_joint_state_.header.stamp=now-ros::Duration(.051);
        Eigen::MatrixXd q,u;
        Eigen::MatrixXd legacy_q,legacy_u;bool legacy_proposed=false;
        const bool proposed=p.buildGeometricRestoreProposalLocked(failure,
            variant==12 ? "care_obs_v1_foreign" : p.latest_observation_token_,f.get<double>("prefix"),&q,&u);
        CHECK(p.safe_frontier_recovery_.attempts==2 && p.safe_frontier_recovery_.segments==1);
        if(variant>=11)CHECK(!proposed);
        else {
          ptree record;boost::property_tree::read_json(std::string(std::getenv("ROOT_OUT"))+"/recovery_proposal_latest.json",record);
          CHECK(record.get<bool>("guidance_used")== (variant==0));
          const auto desired=vec(record.get_child("desired"));
          const auto expected=variant==0 ? guide.target : p.latest_single_waypoint_q_;
          CHECK((desired-expected).norm()<1e-12);
          if(variant==1){plain_q=q;plain_u=u;}
          if(variant>1 && proposed)CHECK((q-plain_q).norm()<1e-12 && (u-plain_u).norm()<1e-12);
          if(proposed) {
            CHECK((q.col(0)-measured).norm()<1e-12);
            CHECK((q.col(3)-q.col(20)).norm()<1e-12 && u.rightCols(17).norm()==0.);
            for(int k=0;k<20;++k)CHECK((q.col(k+1)-q.col(k)-.05*u.col(k)).norm()<1e-12);
          }
          if(variant==0) {
            std::vector<Eigen::VectorXd> normals;std::vector<double> distances;
            for(const auto& item:record.get_child("rows")) {
              normals.push_back(vec(item.second.get_child("gradient")));
              distances.push_back(item.second.get<double>("distance_m"));
            }
            legacy_proposed=geometricRestoreProposal(measured,p.latest_single_waypoint_q_,normals,distances,{},
                p.q_min_,p.q_max_,p.velocity_limits_,p.joint_position_margin_,20,.05,
                &legacy_q,&legacy_u,f.get<double>("prefix"));
          }
        }
        if(!first)output<<',';first=false;output<<"{\"tag\":\""<<f.get<std::string>("tag")<<"\",\"variant\":"<<variant
            <<",\"proposed\":"<<(proposed ? "true":"false");
        if(proposed){output<<",\"endpoint\":";jsonVector(output,q.col(20));}
        if(variant==0) {
          output<<",\"legacy_proposed\":"<<(legacy_proposed ? "true":"false");
          if(legacy_proposed){output<<",\"legacy_endpoint\":";jsonVector(output,legacy_q.col(20));}
        }
        output<<'}';++checks;
      }
      // A duplicated point must retain the stricter ordinary role/reserve.
      p.latest_joint_state_.header.stamp=ros::Time::now();p.rolling_visibility_guidance_enabled_=false;
      p.restore_path_guard_scope_=std::to_string(p.mode_epoch_)+":"+p.latest_observation_token_;
      PrimitiveObservationPoint guard;guard.point=failure.geometric_restore_guard_points.front();
      guard.ordinary_path_guard=true;guard.proposal_reserve_m=.0004;p.restore_path_guard_points_={guard};
      Eigen::MatrixXd q,u;p.buildGeometricRestoreProposalLocked(failure,p.latest_observation_token_,.15,&q,&u);
      ptree record;boost::property_tree::read_json(std::string(std::getenv("ROOT_OUT"))+"/recovery_proposal_latest.json",record);
      const auto& row=record.get_child("rows").front().second;
      CHECK(row.get<double>("floor_m")==.02 && row.get<double>("reserve_m")==.0004);++checks;
      // A transient ordinary corridor point from this failed QP must survive
      // even when no owned restore point or persistent path guard exists.
      p.restore_path_guard_points_.clear();p.restore_path_guard_scope_.clear();
      failure.geometric_restore_guard_points.clear();failure.geometric_restore_path_guards={guard};
      p.buildGeometricRestoreProposalLocked(failure,p.latest_observation_token_,.15,&q,&u);
      ptree transient;boost::property_tree::read_json(std::string(std::getenv("ROOT_OUT"))+"/recovery_proposal_latest.json",transient);
      CHECK(transient.get_child("rows").size()==1);
      const auto& ordinary=transient.get_child("rows").front().second;
      CHECK(ordinary.get<double>("floor_m")==.02 && ordinary.get<double>("reserve_m")==.0004);++checks;
    }
    output<<"],\"checks\":"<<checks<<",\"status\":\"PASS\"}\n";
    std::cout<<"PASS "<<checks<<" native saved-pose/identity/age/mask/point-role controls\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
