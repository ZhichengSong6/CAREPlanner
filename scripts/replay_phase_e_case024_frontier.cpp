// Offline diagnostic: invoke the unchanged production tangent target function.
#include <egocentric_arm_planner/safe_frontier_recovery.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <iostream>
#include <iomanip>
using boost::property_tree::ptree;
Eigen::VectorXd vec(const ptree& t) {
  Eigen::VectorXd v(t.size()); int i=0;
  for (const auto& item:t) v[i++]=std::stod(item.second.data());
  return v;
}
int main(int argc,char** argv) {
  if(argc!=2) return 2;
  ptree root; boost::property_tree::read_json(argv[1],root);
  std::cout<<std::setprecision(17)<<"["; bool first=true;
  for(const auto& item:root) {
    const auto& t=item.second;
    std::vector<Eigen::VectorXd> gradients;
    for(const auto& g:t.get_child("gradients")) gradients.push_back(vec(g.second));
    const auto q=egocentric_arm_planner::boundedTangentTarget(vec(t.get_child("q0")),
        vec(t.get_child("desired")),gradients,t.get<int>("attempt"),t.get<double>("step"));
    if(!first) std::cout<<','; first=false;
    std::cout<<"[";
    for(int i=0;i<q.size();++i) { if(i) std::cout<<','; std::cout<<q[i]; }
    std::cout<<"]";
  }
  std::cout<<"]\n";
}
