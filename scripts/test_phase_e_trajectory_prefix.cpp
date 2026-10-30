// Read independently generated intervals and call the actual tracker sampler.
#include "tracker_diagnostic_access.hpp"
#include <boost/property_tree/json_parser.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
using boost::property_tree::ptree;
using Tracker=egocentric_arm_planner::TrajectoryExecutionManager;
std::vector<double> values(const ptree& node) {
  std::vector<double> result;
  for(const auto& item:node)result.push_back(item.second.get_value<double>());
  return result;
}
int main(int argc,char** argv) {
  try {
    if(argc!=3)throw std::runtime_error("input and result required");
    ptree input;boost::property_tree::read_json(argv[1],input);
    std::size_t trajectories=0,pieces=0,samples=0;
    for(const auto& row:input) {
      const auto& r=row.second;
      trajectory_msgs::JointTrajectory t;
      for(const auto& n:r.get_child("record.joint_names"))t.joint_names.push_back(n.second.get_value<std::string>());
      std::vector<int> mapping;
      for(const auto& n:r.get_child("controller_joint_names")) {
        auto i=std::find(t.joint_names.begin(),t.joint_names.end(),n.second.get_value<std::string>());
        if(i==t.joint_names.end())throw std::runtime_error("joint mapping failed");
        mapping.push_back(i-t.joint_names.begin());
      }
      for(const auto& item:r.get_child("record.points")) {
        const auto& p=item.second;trajectory_msgs::JointTrajectoryPoint point;
        point.time_from_start.fromNSec(p.get<std::int64_t>("time_ns"));
        point.positions=values(p.get_child("positions"));point.velocities=values(p.get_child("velocities"));
        point.accelerations=values(p.get_child("accelerations"));t.points.push_back(point);
      }
      for(const auto& item:r.get_child("construction.segments")) {
        const auto& s=item.second;
        const auto lo=values(s.get_child("q_low")),hi=values(s.get_child("q_high"));
        const double begin=s.get<double>("phase_begin_s"),end=s.get<double>("phase_end_s");
        for(int k=0;k<=256;++k) {
          const double phase=begin+(end-begin)*k/256.;Eigen::VectorXd q,v,a;
          if(!Tracker::sampleTrajectory(t,mapping,phase,q,v,a))throw std::runtime_error("actual sampler rejected fixture");
          for(int j=0;j<7;++j)if(q[j]<lo[j] || q[j]>hi[j])
            throw std::runtime_error("actual tracker sample outside generated interval");
          ++samples;
        }
        ++pieces;
      }
      ++trajectories;
    }
    std::ofstream out(argv[2]);out<<"{\"status\":\"PASS_ACTUAL_TRACKER_INTERVAL_REPLAY\",\"trajectories\":"
      <<trajectories<<",\"pieces\":"<<pieces<<",\"samples\":"<<samples<<",\"runtime_admission\":false}\n";
    std::cout<<"PASS actual sampler "<<trajectories<<" trajectories, "<<pieces<<" pieces, "<<samples<<" samples\n";
    return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
