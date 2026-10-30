// Offline FK diagnostic only: no ROS node, publisher, controller or map writes.
// Use the exact already-built production body-sample/FK library.
#include <care_confidence_map/trajectory_risk_evaluator.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
using boost::property_tree::ptree;
Eigen::VectorXd numbers(const ptree& t) {
  Eigen::VectorXd q(t.size()); int i=0;
  for (const auto& v:t) q[i++]=std::stod(v.second.data());
  if (!q.allFinite()) throw std::runtime_error("nonfinite input");
  return q;
}
int main() {
  try {
    ptree input; boost::property_tree::read_json(std::cin,input);
    care_confidence_map::TrajectoryRiskEvaluator model;
    std::string error;
    if (!model.initialize(input.get<std::string>("urdf"),input.get<std::string>("body_samples"),
                          "base_link",&error)) throw std::runtime_error(error);
    const std::vector<std::string> expected={"joint1","joint2","joint3","joint4",
        "wrist_joint1","wrist_joint2","wrist_joint3"};
    if (model.activeJointNames()!=expected) throw std::runtime_error("joint ordering mismatch");
    std::set<std::string> ignored;
    for (const auto& link:input.get_child("ignored_links")) ignored.insert(link.second.data());
    const auto lower=numbers(input.get_child("map_lower"));
    const auto upper=numbers(input.get_child("map_upper"));
    const double inflation=input.get<double>("body_inflation_m");
    if (lower.size()!=3 || upper.size()!=3 || !std::isfinite(inflation) || inflation<0)
      throw std::runtime_error("invalid map/inflation input");
    std::cout<<std::setprecision(17)<<"["; bool first=true;
    for (const auto& request:input.get_child("queries")) {
      const auto q=numbers(request.second.get_child("q"));
      const auto p=numbers(request.second.get_child("point"));
      if (q.size()!=7 || p.size()!=3) throw std::runtime_error("dimension mismatch");
      care_confidence_map::TrajectoryFrameSamples frame;
      if (!model.computeSamplesForConfiguration(q,0,&frame,&error)) throw std::runtime_error(error);
      double best=std::numeric_limits<double>::infinity();
      const care_confidence_map::TrajectoryBodySample* winner=nullptr;
      for (const auto& sample:frame.samples) {
        if (ignored.count(sample.link_name)) continue;
        bool inside=true;
        for (int j=0;j<3;++j) inside=inside && sample.center_base[j]>=lower[j] && sample.center_base[j]<upper[j];
        if (!inside) continue;
        // Match PointCloud2 FLOAT32 anchors and the selector's FLOAT32 result.
        Eigen::Vector3d center=sample.center_base.cast<float>().cast<double>();
        const double radius=static_cast<float>(sample.radius+inflation);
        const double clearance=static_cast<float>((center-p).norm()-radius);
        if (clearance<best) { best=clearance; winner=&sample; }
      }
      if (!winner) throw std::runtime_error("no in-map eligible body sample");
      if (!first) std::cout<<','; first=false;
      std::cout<<"{\"clearance_m\":"<<best<<",\"link\":\""<<winner->link_name
        <<"\",\"sample_index\":"<<winner->sample_index_in_link
        <<",\"source_collision_index\":"<<winner->source_collision_index
        <<",\"raw_radius_m\":"<<winner->radius<<"}";
    }
    std::cout<<"]\n";
    return 0;
  } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 2; }
}
