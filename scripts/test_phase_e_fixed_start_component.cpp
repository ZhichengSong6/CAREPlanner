#include "planner_diagnostic_access.hpp"
#include <boost/property_tree/json_parser.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
using Planner=egocentric_arm_planner::LocalSparseSCPPlanner;
#define CHECK(x) do{if(!(x))throw std::runtime_error(#x);}while(false)
Eigen::VectorXd vec(const boost::property_tree::ptree& t){Eigen::VectorXd v(t.size());int i=0;for(const auto& x:t)v[i++]=x.second.get_value<double>();return v;}
Eigen::MatrixXd matrix(const boost::property_tree::ptree& t){Eigen::MatrixXd v(7,t.size());int i=0;for(const auto& x:t)v.col(i++)=vec(x.second);return v;}
int main(int argc,char** argv){ros::init(argc,argv,"fixed_start_component");try{
  CHECK(argc==2);ros::NodeHandle nh,pnh("~");Planner p;CHECK(p.initialize(nh,pnh));p.stopWorker();p.timer_.stop();
  p.repair_prefix_binding_enabled_=false;p.repair_unknown_witnesses_.clear();
  p.mode_epoch_=14;p.latest_observation_token_="care_obs_v1_fixed";
  p.restore_path_guard_scope_="14:care_obs_v1_fixed";
  egocentric_arm_planner::PrimitiveObservationPoint point;point.point=Eigen::Vector3d(.3,-.25,.65);
  point.ordinary_path_guard=true;p.restore_path_guard_points_={point};
  boost::property_tree::ptree data;boost::property_tree::read_json(argv[1],data);
  care_collision_cdf::CollisionCDFConstraintBatch batch;batch.dof=7;batch.num_pairs=0;
  int captures=0;
  for(const auto& item:data.get_child("snapshots")){
    const auto& s=item.second.get_child("subproblem");
    const auto q=matrix(s.get_child("q_bar")),u=matrix(s.get_child("u_bar")),reference=matrix(s.get_child("q_ref"));
    const auto result=p.solveSparseSubproblem(batch,q,u,reference,u,vec(s.get_child("previous_command")),{},
        Planner::FrontierObjective{},{},true,false,.15,10.);
    CHECK(!result.solved && result.fixed_start_unknown_blocked);
    CHECK(result.status=="fixed_start_unknown_clearance" && result.iterations==0);
    CHECK(result.fixed_start_unknown_floor_m==.02 && result.fixed_start_unknown_distance_m<.02);
    CHECK((result.fixed_start_unknown_point-point.point).norm()<1e-12);++captures;
  }
  CHECK(captures==5);
  const Eigen::MatrixXd q=Eigen::MatrixXd::Zero(7,21),u=Eigen::MatrixXd::Zero(7,20);
  const auto control=p.solveSparseSubproblem(batch,q,u,q,u,Eigen::VectorXd::Zero(7),{},Planner::FrontierObjective{},{},true,false,.15,10.);
  CHECK(!control.fixed_start_unknown_blocked && control.solved);
  p.fixed_start_clearance_hold_.record(q.col(0),point.point,.00939,.02,ros::Time::now().toSec());
  const auto attempts=p.safe_frontier_recovery_.attempts;
  for(int k=0;k<8;++k){p.latest_observation_token_="care_obs_v1_changed_"+std::to_string(k);
    p.requestPlanLocked("single_visibility_waypoint_q_changed");CHECK(!p.plan_requested_);}
  CHECK(p.safe_frontier_recovery_.attempts==attempts && p.plan_request_reason_=="fixed_start_unknown_clearance_hold");
  std::cout<<"PASS 5 recorded starts rejected before PIQP, safe-start real PIQP control, 8 head/token changes suppressed\n";
  ros::shutdown();return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';ros::shutdown();return 1;}}
