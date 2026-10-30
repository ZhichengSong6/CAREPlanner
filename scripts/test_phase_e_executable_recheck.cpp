#include "planner_diagnostic_access.hpp"
#include <boost/property_tree/json_parser.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
using Planner=egocentric_arm_planner::LocalSparseSCPPlanner;
int main(int argc,char** argv) {
 ros::init(argc,argv,"recheck_test",ros::init_options::NoSigintHandler);
 try {
  if(argc!=3)throw std::runtime_error("query/results required");
  ros::NodeHandle nh,pnh("~");Planner p;
  if(!p.initialize(nh,pnh))throw std::runtime_error("initialize failed");p.stopWorker();p.timer_.stop();
  p.primitive_observation_guard_enabled_=false;p.repair_prefix_binding_enabled_=false;p.cdf_slack_enabled_=false;
  boost::property_tree::ptree row;boost::property_tree::read_json(argv[1],row);
  Eigen::MatrixXd q(7,21),u(7,20);Eigen::VectorXd previous(7);
  const auto read=[&](const std::string& key){return std::stod(row.get<std::string>(boost::property_tree::ptree::path_type(key,'/')));};
  for(int k=0;k<=20;++k)for(int j=0;j<7;++j) {
    q(j,k)=read("field.points"+std::to_string(k)+".positions"+std::to_string(j));
    const double v=read("field.points"+std::to_string(k)+".velocities"+std::to_string(j));
    if(k==0)previous[j]=v;else u(j,k-1)=v;
  }
  care_collision_cdf::CollisionCDFConstraintBatch batch;batch.dof=7;batch.num_pairs=0;
  std::ofstream out(argv[2]);out<<std::setprecision(17)<<'[';
  for(int mode=0;mode<7;++mode) {
    auto qb=q,ub=u;auto b=batch;
    if(mode==1)ub.col(19).setZero();
    if(mode==2)ub(0,19)=1e-4;
    if(mode==3) {
      b.num_pairs=1;b.original_timestep={1};b.source_type={1};b.distance={-1.};
      b.point_flat={.1,.2,.3};b.gradient_flat.assign(7,0.);b.q_linearization_flat.assign(q.col(1).data(),q.col(1).data()+7);
    }
    if(mode==4) {
      qb(0,20)=p.q_max_[0]-p.joint_position_margin_+1e-4;
    }
    if(mode==5)ub(0,19)=.5e-7;
    if(mode==6)ub(0,19)=2e-7;
    const auto r=p.solveSparseSubproblem(b,qb,ub,qb,ub,previous,{},Planner::FrontierObjective{}, {},true,false,.15,10.,false,false,1.,true);
    const bool expected=mode==0 || mode==1 || mode==5;
    if(r.solved!=expected)throw std::runtime_error("unexpected recheck result mode="+std::to_string(mode)+" status="+r.status);
    if(r.solved && ((r.q-qb).cwiseAbs().maxCoeff()>1e-7 || (r.u-ub).cwiseAbs().maxCoeff()>1e-7))
      throw std::runtime_error("fixed-candidate tolerance changed");
    if(mode==3 && r.status=="executable_recheck_bound_error")
      throw std::runtime_error("CDF control did not reach full hard problem");
    if(mode)out<<',';
    out<<"{\"mode\":"<<mode<<",\"solved\":"<<(r.solved?"true":"false")<<",\"status\":\""<<r.status
       <<"\",\"piqp_iter\":"<<r.iterations;
    if(r.solved)out<<",\"dq_inf\":"<<(r.q-qb).cwiseAbs().maxCoeff()<<",\"du_inf\":"<<(r.u-ub).cwiseAbs().maxCoeff()
       <<",\"terminal_speed_inf\":"<<r.u.col(19).cwiseAbs().maxCoeff();
    out<<'}';
  }
  out<<"]\n";ros::shutdown();return 0;
 } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';ros::shutdown();return 1;}
}
