// Read-only production-library geometry probe; no ROS node or map writes.
#include <care_confidence_map/vbc_primitive_model.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <iomanip>
#include <iostream>
using boost::property_tree::ptree;
Eigen::VectorXd vec(const ptree& x) {
  Eigen::VectorXd v(x.size()); int j=0;
  for(const auto& a:x) v[j++]=a.second.get_value<double>();
  if(!v.allFinite()) throw std::runtime_error("nonfinite"); return v;
}
int main() {
 try {
  ptree in; boost::property_tree::read_json(std::cin,in);
  care_confidence_map::TrajectoryRiskEvaluator fk;
  care_confidence_map::VbcPrimitiveModel model; std::string error;
  if(!fk.initializeKinematics(in.get<std::string>("urdf"),"base_link",&error) ||
     !model.load(in.get<std::string>("primitive_urdf"),{"base_link","link1"},&error))
    throw std::runtime_error(error);
  std::vector<Eigen::VectorXd> q; std::vector<double> times; std::vector<int> indices;
  for(const auto& row:in.get_child("trajectory")) {
    q.push_back(vec(row.second.get_child("q"))); indices.push_back(indices.size());
    times.push_back(row.second.get<double>("time_s"));
  }
  std::vector<Eigen::Vector3d> points;
  for(const auto& p:in.get_child("points")) points.push_back(vec(p.second));
  std::cout<<std::setprecision(17)<<"["; bool first=true;
  for(int mode=0;mode<3;++mode) {
    std::vector<care_confidence_map::VbcPrimitiveFrame> frames;
    std::vector<int> ids; std::vector<double> ts;
    if(mode==0) {
      if(!model.computeTrajectory(fk,q,&frames,&error)) throw std::runtime_error(error);
      ts=times; ids=indices;
    } else if(!model.computeContinuousTrajectory(fk,q,indices,times,&frames,&ids,&ts,&error,mode==2))
      throw std::runtime_error(error);
    for(size_t k=0;k<frames.size();++k) for(size_t j=0;j<points.size();++j) {
      double best=1e100,raw=0,bound=0; std::string name;
      for(size_t i=0;i<frames[k].primitives.size();++i) {
        const auto& p=frames[k].primitives[i]; double d=p.signedDistance(points[j]);
        double b=frames[k].motion_bounds.empty()?0:frames[k].motion_bounds[i];
        if(d-b<best) {best=d-b;raw=d;bound=b;name=p.collision_name;}
      }
      if(!first)std::cout<<',';first=false;
      std::cout<<"{\"mode\":"<<mode<<",\"frame\":"<<k<<",\"original_k\":"<<ids[k]
        <<",\"entry_s\":"<<ts[k]<<",\"point_index\":"<<j<<",\"distance_m\":"<<raw
        <<",\"motion_bound_m\":"<<bound<<",\"enclosed_distance_m\":"<<best
        <<",\"primitive\":"<<std::quoted(name)<<"}";
    }
  }
  std::cout<<"]\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
