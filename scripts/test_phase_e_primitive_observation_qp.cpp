// Access shim is generated from the current production header by the driver.
#include <care_confidence_map/vbc_primitive_model.hpp>
#include "planner_diagnostic_access.hpp"
#include <boost/property_tree/json_parser.hpp>
#include <fstream>
#include <iomanip>

using Planner = egocentric_arm_planner::LocalSparseSCPPlanner;
using boost::property_tree::ptree;
Eigen::VectorXd vec(const ptree& values) {
  Eigen::VectorXd v(values.size()); int j = 0;
  for (const auto& item : values) v[j++] = item.second.get_value<double>();
  return v;
}
void matrix(std::ostream& out, const Eigen::MatrixXd& m) {
  out << '[';
  for (int k=0;k<m.cols();++k) {
    if(k)out<<','; out<<'[';
    for(int j=0;j<m.rows();++j) { if(j)out<<','; out<<m(j,k); }
    out<<']';
  }
  out << ']';
}
int main(int argc, char** argv) {
  ros::init(argc,argv,"primitive_observation_qp",ros::init_options::NoSigintHandler);
  try {
    if(argc!=3)throw std::runtime_error("fixture and result paths required");
    ros::NodeHandle nh,pnh("~"); Planner p;
    if(!p.initialize(nh,pnh))throw std::runtime_error("planner initialization failed");
    p.stopWorker();p.timer_.stop();
    ptree input;boost::property_tree::read_json(argv[1],input);
    care_confidence_map::TrajectoryRiskEvaluator fk;
    care_confidence_map::VbcPrimitiveModel model;
    std::string error,urdf,primitive;
    pnh.getParam("local_planner/primitive_observation_guard/fk_urdf",urdf);
    pnh.getParam("local_planner/primitive_observation_guard/primitive_urdf",primitive);
    if(!fk.initializeKinematics(urdf,"base_link",&error) ||
       !model.load(primitive,{"base_link","link1"},&error))throw std::runtime_error(error);
    std::ofstream output(argv[2]);output<<std::setprecision(17)<<'[';bool first=true;
    for(const auto& item : input) {
      const auto& t=item.second;
      p.primitive_observation_guard_enabled_=t.get<bool>("geometry_enabled");
      p.repair_unknown_witnesses_.clear();
      std::vector<Eigen::Vector3d> points;
      for(const auto& w:t.get_child("witnesses")) {
        Planner::RepairWitness witness{vec(w.second.get_child("point")),1,w.second.get<double>("margin"),true,-1.,.3};
        const auto evidence=w.second.get_optional<std::string>("evidence");
        if(evidence) {
          std::vector<egocentric_arm_planner::PrimitiveObservationPoint> parsed;
          if(!egocentric_arm_planner::parsePrimitiveObservationEvidence(*evidence,26193000000ULL,&parsed))
            throw std::runtime_error("Invalid fixture evidence");
          witness.has_primitive_guard=true;witness.primitive_guard=parsed.front();
        }
        points.push_back(witness.point);p.repair_unknown_witnesses_.push_back(witness);
      }
      care_collision_cdf::CollisionCDFConstraintBatch b;b.dof=7;
      for(const auto& item:t.get_child("rows")) {
        const auto& r=item.second;
        b.original_timestep.push_back(r.get<int>("k"));b.source_type.push_back(0);
        b.distance.push_back(r.get<double>("d"));
        for(const auto& x:r.get_child("point"))b.point_flat.push_back(x.second.get_value<double>());
        for(const auto& x:r.get_child("g"))b.gradient_flat.push_back(x.second.get_value<double>());
        for(const auto& x:r.get_child("qlin"))b.q_linearization_flat.push_back(x.second.get_value<double>());
      }
      b.num_pairs=b.distance.size();
      const auto q0=vec(t.get_child("q0")),target=vec(t.get_child("target"));
      Eigen::MatrixXd q=q0.replicate(1,21),u=Eigen::MatrixXd::Zero(7,20);
      Eigen::MatrixXd reference=target.replicate(1,21);
      Planner::DeadlineWaypoint waypoint;waypoint.id=5;waypoint.terminal_objective=true;
      waypoint.q=target;waypoint.joint_mask=Eigen::VectorXd::Ones(7);
      const auto r=p.solveSparseSubproblem(b,q,u,reference,u,Eigen::VectorXd::Zero(7),
          {waypoint},Planner::FrontierObjective{},points,true,false,.15,10.);
      if(!first)output<<',';first=false;
      output<<"{\"id\":\""<<t.get<std::string>("id")<<"\",\"status\":\""<<r.status
          <<"\",\"solved\":"<<(r.solved?"true":"false")<<",\"cdf_rows\":"<<r.selected_cdf_rows
          <<",\"geometry_rows\":"<<r.primitive_observation_rows<<",\"max_slack\":"<<r.max_slack
          <<",\"reachable_cdf_conflicts\":"<<r.reachable_conflicting_repair_witness_rows;
      if(r.solved) {
        std::vector<Eigen::VectorXd> path;
        for(int k=0;k<r.q.cols();++k)path.push_back(r.q.col(k));
        int remaining=256;
        const bool outside=model.provePointOutsideContinuousTrajectory(fk,path,points.front(),.020,&remaining);
        output<<",\"outside_20mm_continuous\":"<<(outside?"true":"false")
            <<",\"proof_fk\":"<<256-remaining<<",\"q\":";
        matrix(output,r.q);output<<",\"u\":";matrix(output,r.u);
      }
      output<<'}';
    }
    output<<"]\n";ros::shutdown();return 0;
  } catch(const std::exception& e) {
    std::cerr<<e.what()<<'\n';ros::shutdown();return 1;
  }
}
