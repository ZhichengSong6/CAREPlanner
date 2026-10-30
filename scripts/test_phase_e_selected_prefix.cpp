#include "planner_diagnostic_access.hpp"
#include <boost/property_tree/json_parser.hpp>
#include <boost/make_shared.hpp>
#include <fstream>
#include <iomanip>
using Planner=egocentric_arm_planner::LocalSparseSCPPlanner;
using boost::property_tree::ptree;
Eigen::VectorXd vec(const ptree& t){Eigen::VectorXd v(t.size());int j=0;for(const auto& x:t)v[j++]=x.second.get_value<double>();return v;}
Eigen::MatrixXd mat(const ptree& t){Eigen::MatrixXd m(t.front().second.size(),t.size());int k=0;for(const auto& x:t)m.col(k++)=vec(x.second);return m;}
void matrix(std::ostream& o,const Eigen::MatrixXd& m){o<<'[';for(int k=0;k<m.cols();++k){if(k)o<<',';o<<'[';for(int j=0;j<m.rows();++j){if(j)o<<',';o<<m(j,k);}o<<']';}o<<']';}
int main(int argc,char** argv){
  ros::init(argc,argv,"selected_prefix_test",ros::init_options::NoSigintHandler);
  try{
    ptree fixtures;boost::property_tree::read_json(argv[1],fixtures);ptree safe,bad;
    for(const auto& f:fixtures){if(f.second.get<std::string>("name")=="plan77_input2")safe=f.second;
      if(f.second.get<std::string>("name")=="plan77_input3")bad=f.second;}
    const auto qs=mat(safe.get_child("q")),us=mat(safe.get_child("u"));
    const auto qb=mat(bad.get_child("q")),ub=mat(bad.get_child("u"));
    ros::NodeHandle nh,pnh("~");Planner p;if(!p.initialize(nh,pnh))throw std::runtime_error("initialize");
    p.stopWorker();p.timer_.stop();std::vector<Eigen::Vector3d> points;
    for(const auto& x:safe.get_child("points")){
      const auto point=vec(x.second.get_child("point"));Planner::RepairWitness w{point,1,.05,true,-1.,.3};
      w.has_primitive_guard=true;w.primitive_guard.point=point;
      w.primitive_guard.sweeping_link=x.second.get<std::string>("sweeping_link");
      w.primitive_guard.collision_name=x.second.get<std::string>("collision_name");
      w.primitive_guard.collision_index=x.second.get<int>("collision_index");
      p.repair_unknown_witnesses_.push_back(w);points.push_back(point);
    }
    p.repair_task_tracking_scale_=1.;p.q_tracking_weight_=1000.;p.terminal_q_tracking_weight_=0.;
    p.u_tracking_weight_=1000.;p.u_reference_tracking_enabled_=true;p.u_smooth_weight_=0.;p.handoff_velocity_weight_=0.;
    care_collision_cdf::CollisionCDFConstraintBatch b;b.dof=7;
    for(int k=1;k<=20;++k)for(const auto& point:points){b.original_timestep.push_back(k);
      b.source_type.push_back(care_collision_cdf::CollisionCDFConstraintBatch::SOURCE_UNKNOWN);b.distance.push_back(.5);
      for(int j=0;j<3;++j)b.point_flat.push_back(point[j]);
      for(int j=0;j<7;++j){b.gradient_flat.push_back(0.);b.q_linearization_flat.push_back(qs(j,k));}}
    b.num_pairs=b.distance.size();b.header.stamp=ros::Time::now();
#if CARE_NEW
    egocentric_arm_planner::RepairPrefixSelection c;c.session=std::string(32,'a');c.revision=1;c.prefix_s=.25;
    c.acceleration={3,3,4,4,6,6,6};c.brake_dt_s=.05;c.brake_steps=20;c.header.stamp=ros::Time::now();
    p.repair_prefix_binding_enabled_=true;p.latest_prefix_selection_=c;p.plan_prefix_selection_=c;
#endif
    auto solve=[&](bool hostile,bool fixed){auto batch=b;if(hostile)for(auto& d:batch.distance)d=-.5;
      if(fixed)for(std::size_t i=0;i<batch.original_timestep.size();++i)for(int j=0;j<7;++j)
        batch.q_linearization_flat[7*i+j]=qb(j,batch.original_timestep[i]);
      return p.solveSparseSubproblem(batch,fixed?qb:qs,fixed?ub:us,qb,ub,ub.col(0),{},Planner::FrontierObjective{},
        points,true,false,.15,10.,false,false,1.,fixed
#if CARE_NEW
        ,&c
#endif
      );};
    const auto proposal=solve(false,false),fixed=solve(false,true),hostile=solve(true,false);
    std::ofstream out(argv[2]);out<<std::setprecision(17)<<"{\"proposal\":{\"solved\":"<<proposal.solved
      <<",\"executable_valid\":"<<proposal.primitive_executable_valid<<",\"min_m\":"<<proposal.primitive_executable_min_distance_m
      <<",\"cdf_rows\":"<<proposal.selected_cdf_rows<<",\"geometry_rows\":"<<proposal.primitive_executable_rows
      <<",\"max_slack\":"<<proposal.max_slack<<",\"q\":";matrix(out,proposal.q);out<<",\"u\":";matrix(out,proposal.u);
    out<<"},\"fixed\":{\"solved\":"<<fixed.solved<<",\"executable_valid\":"<<fixed.primitive_executable_valid
      <<",\"status\":\""<<fixed.status<<"\"},\"hostile\":{\"solved\":"<<hostile.solved
      <<",\"cdf_rows\":"<<hostile.selected_cdf_rows<<"}";
#if CARE_NEW
    auto invalid_selection=c;invalid_selection.acceleration[0]=1.;
    if(p.prefixSelectionCompatible(invalid_selection))throw std::runtime_error("wrong params accepted");
    p.repair_mode_=false;auto update=boost::make_shared<egocentric_arm_planner::RepairPrefixSelection>(c);
    update->revision=2;update->prefix_s=.15;p.repairPrefixSelectionCallback(update);
    if(p.prefixSelectionCurrentLocked(c))throw std::runtime_error("stale current");
    update->revision=1;update->prefix_s=.25;p.repairPrefixSelectionCallback(update);
    if(p.latest_prefix_selection_.revision!=2)throw std::runtime_error("old revision restored");
    update->session=std::string(32,'b');p.repairPrefixSelectionCallback(update);
    update->session=c.session;update->revision=999;p.repairPrefixSelectionCallback(update);
    if(p.latest_prefix_selection_.session!=std::string(32,'b'))throw std::runtime_error("closed boot replay");
    auto missing=p.solveSparseSubproblem(b,qs,us,qb,ub,ub.col(0),{},Planner::FrontierObjective{},points,true,false,.15,10.);
    if(missing.solved)throw std::runtime_error("missing prefix accepted");
    out<<",\"identity_controls\":true";
#endif
    out<<"}\n";return 0;
  }catch(const std::exception& e){ROS_ERROR_STREAM(e.what());return 1;}
}
