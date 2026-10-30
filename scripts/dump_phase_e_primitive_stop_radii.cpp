// Inspect the existing compiled VBC model; no ROS master or callbacks required.
#include <care_confidence_map/vbc_primitive_model.hpp>
#include <iomanip>
#include <iostream>
int main(int argc,char** argv) {
  if(argc!=2) return 2;
  care_confidence_map::VbcPrimitiveModel model; std::string error;
  if(!model.load(argv[1],{},&error)) { std::cerr<<error; return 1; }
  const std::vector<std::string> names{"joint1","joint2","joint3","joint4","wrist_joint1","wrist_joint2","wrist_joint3"};
  std::vector<std::vector<double>> columns;
  for(int j=0;j<7;++j) {
    Eigen::VectorXd delta=Eigen::VectorXd::Zero(7); delta[j]=1.;
    columns.push_back(model.displacementBounds(names,delta));
  }
  std::cout<<std::setprecision(17);
  for(size_t i=0;i<model.primitives().size();++i) {
    const auto& p=model.primitives()[i];
    std::cout<<"{\"link\":"<<std::quoted(p.link_name)<<",\"name\":"<<std::quoted(p.collision_name)<<",\"radii\":[";
    for(int j=0;j<7;++j) { if(j)std::cout<<','; std::cout<<columns[j][i]; }
    std::cout<<"]}\n";
  }
}
