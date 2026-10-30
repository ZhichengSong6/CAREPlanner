#include "planner_diagnostic_access.hpp"
#include <boost/property_tree/json_parser.hpp>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
using boost::property_tree::ptree;
using Planner=egocentric_arm_planner::LocalSparseSCPPlanner;
Eigen::VectorXd vec(const ptree& t){Eigen::VectorXd v(t.size());int j=0;for(const auto& x:t)v[j++]=x.second.get_value<double>();return v;}
Eigen::MatrixXd mat(const ptree& t){Eigen::MatrixXd q(7,t.size());int k=0;for(const auto& x:t)q.col(k++)=vec(x.second);return q;}
void matrix(std::ostream& out,const Eigen::MatrixXd& m){out<<'[';for(int k=0;k<m.cols();++k){if(k)out<<',';out<<'[';for(int j=0;j<m.rows();++j){if(j)out<<',';out<<m(j,k);}out<<']';}out<<']';}
int main(int argc,char** argv){
  ros::init(argc,argv,"submit_corridor_test",ros::init_options::NoSigintHandler);
  try{
    ros::NodeHandle nh,pnh("~");Planner p;assert(p.initialize(nh,pnh));p.stopWorker();p.timer_.stop();
    ptree f;boost::property_tree::read_json(argv[1],f);
    const auto reference=mat(f.get_child("q")),velocities=mat(f.get_child("v"));
    const Eigen::Vector3d point=vec(f.get_child("point"));
    const Eigen::MatrixXd input_u=velocities.rightCols(20);
    const Eigen::VectorXd previous=velocities.col(0);
    p.repair_prefix_binding_enabled_=true;p.initial_shell_recovery_enabled_=true;
    p.latest_vbc_obligation_points_.clear();p.repair_unknown_witnesses_.clear();
    egocentric_arm_planner::RepairPrefixSelection prefix;
    prefix.session=std::string(32,'a');prefix.revision=1;prefix.prefix_s=.25;
    prefix.brake_dt_s=.05;prefix.brake_steps=20;prefix.acceleration={3.,3.,4.,4.,6.,6.,6.};
    Planner::DeadlineWaypoint wp;wp.id=1;wp.terminal_objective=true;
    wp.q=reference.col(20);wp.joint_mask=Eigen::VectorXd::Ones(7);
    std::ofstream out(argv[2]);out<<std::setprecision(17)<<"{\"status\":\"PASS\",\"cases\":[";
    for(int variant=0;variant<5;++variant){
      p.unknown_corridor_guard_enabled_=variant!=0;
      p.latest_vbc_obligation_points_.clear();if(variant==3)p.latest_vbc_obligation_points_.push_back(point);
      auto qb=reference,ub=input_u;Planner::SparseSolveResult result;
      for(int iteration=0;iteration<6;++iteration){
        care_collision_cdf::CollisionCDFConstraintBatch b;b.dof=7;
        // Controlled native-CDF rows are deliberately safely screened. The
        // physical UNKNOWN identity still needs an analytic prefix guard.
        if(variant!=2)for(int k=1;k<=20;++k){
          b.original_timestep.push_back(k);b.source_type.push_back(variant==4?1:0);
          b.distance.push_back(.06);for(int j=0;j<3;++j)b.point_flat.push_back(point[j]);
          for(int j=0;j<7;++j){b.gradient_flat.push_back(0.);b.q_linearization_flat.push_back(qb(j,k));}
        }
        b.num_pairs=b.distance.size();
        const std::vector<Eigen::Vector3d> owned=variant==3?std::vector<Eigen::Vector3d>{point}:std::vector<Eigen::Vector3d>{};
        result=p.solveSparseSubproblem(b,qb,ub,reference,input_u,previous,{wp},Planner::FrontierObjective{},owned,
            true,false,.15,10.,false,false,1.,false,&prefix);
        assert(result.solved && result.max_slack==0.);
        qb=result.q;ub=result.u;
        if(variant!=1 || result.primitive_executable_valid)break;
      }
      if(variant==1){
        assert(result.primitive_observation_points==1 && result.primitive_executable_rows>0);
        assert(result.primitive_executable_valid && result.primitive_executable_min_distance_m>=.02);
        assert(result.selected_cdf_rows==0);
        assert(result.geometric_restore_guard_points.empty());
      }else if(variant==3){
        assert(result.primitive_observation_points==0); // owned semantics retained
        assert(result.selected_cdf_rows>0);
      }else assert(result.primitive_observation_points==0);
      if(variant)out<<',';
      out<<"{\"variant\":"<<variant<<",\"passed\":true,\"primitive_points\":"<<result.primitive_observation_points
         <<",\"cdf_rows\":"<<result.selected_cdf_rows<<",\"executable_rows\":"<<result.primitive_executable_rows;
      if(variant==1){out<<",\"minimum_m\":"<<result.primitive_executable_min_distance_m<<",\"q\":";matrix(out,result.q);out<<",\"u\":";matrix(out,result.u);}
      out<<'}';
    }
    out<<"],\"qualification\":\"Actual production QP and saved002A state/reference/point with controlled clear native-CDF rows. Not the historical full QP or a continuous execution certificate.\"}\n";
    ros::shutdown();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';ros::shutdown();return 1;}
}
