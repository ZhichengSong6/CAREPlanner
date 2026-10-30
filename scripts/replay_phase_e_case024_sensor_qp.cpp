// Diagnostic access shim links the unchanged production planner object.
// No candidate publication, no actuator, no map or VBC certificate.
#include "planner_diagnostic_access.hpp"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <fstream>
#include <iomanip>
using boost::property_tree::ptree;
using Planner=egocentric_arm_planner::LocalSparseSCPPlanner;
Eigen::VectorXd vec(const ptree& t) {
  Eigen::VectorXd v(t.size()); int i=0;
  for(const auto& x:t) v[i++]=x.second.get_value<double>(); return v;
}
Eigen::MatrixXd mat(const ptree& t) {
  Eigen::MatrixXd m(7,t.size()); int i=0;
  for(const auto& x:t) m.col(i++)=vec(x.second); return m;
}
void matrix(std::ostream& o,const Eigen::MatrixXd& m) {
  o<<'['; for(int k=0;k<m.cols();++k) { if(k)o<<',';o<<'[';
    for(int j=0;j<m.rows();++j){if(j)o<<',';o<<m(j,k);}o<<']';}o<<']';
}
int main(int argc,char** argv) {
  ros::init(argc,argv,"case024_sensor_qp",ros::init_options::NoSigintHandler);
  if(argc!=3)return 2;
  ros::NodeHandle nh,pnh("~"); Planner p;
  if(!p.initialize(nh,pnh))return 3;
  p.stopWorker(); p.timer_.stop();
  ptree input;boost::property_tree::read_json(argv[1],input);
  std::ofstream out(argv[2]);out<<std::setprecision(17)<<'[';bool first=true;
  for(const auto& item:input) {
    const auto& t=item.second;
    p.repair_unknown_witnesses_.clear();
    for(const auto& w:t.get_child("witnesses"))
      p.repair_unknown_witnesses_.push_back(Planner::RepairWitness{
        vec(w.second.get_child("point")),1,w.second.get<double>("margin"),true,-1.,.3});
    care_collision_cdf::CollisionCDFConstraintBatch b;b.dof=7;
    for(const auto& w:t.get_child("rows")) {
      b.original_timestep.push_back(w.second.get<int>("k"));b.source_type.push_back(0);
      b.distance.push_back(w.second.get<double>("d"));
      for(const auto& x:w.second.get_child("point"))b.point_flat.push_back(x.second.get_value<double>());
      for(const auto& x:w.second.get_child("g"))b.gradient_flat.push_back(x.second.get_value<double>());
      for(const auto& x:w.second.get_child("qlin"))b.q_linearization_flat.push_back(x.second.get_value<double>());
    }
    b.num_pairs=b.distance.size();
    const auto q=mat(t.get_child("qbar")),u=mat(t.get_child("ubar"));
    Planner::FrontierObjective f;f.active=true;f.frontier_weight_scale=1.;f.qvis_weight_scale=0.;
    f.q=vec(t.get_child("frontier"));f.joint_mask=vec(t.get_child("mask"));
    f.recovery_attempt=t.get<int>("attempt");
    std::vector<Eigen::Vector3d> points;
    for(const auto& w:t.get_child("witnesses"))points.push_back(vec(w.second.get_child("point")));
    auto r=p.solveSparseSubproblem(b,q,u,q,u,vec(t.get_child("previous")),{},f,points,
                                  true,false,t.get<double>("radius"),10.);
    if(!first)out<<',';first=false;
    out<<"{\"id\":\""<<t.get<std::string>("id")<<"\",\"status\":\""<<r.status
       <<"\",\"rows\":"<<r.selected_cdf_rows<<",\"max_slack\":"<<r.max_slack
       <<",\"primal\":"<<r.primal_residual<<",\"dual\":"<<r.dual_residual
       <<",\"blocking_normals\":"<<r.recovery_normals<<",\"q\":";
    matrix(out,r.q);out<<",\"u\":";matrix(out,r.u);out<<'}';
  }
  out<<"]\n";ros::shutdown();return 0;
}
