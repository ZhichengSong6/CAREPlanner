#include <care_confidence_map/vbc_primitive_model.hpp>
#include <care_confidence_map/initial_shell_recovery.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
using boost::property_tree::ptree;
Eigen::VectorXd vec(const ptree& tree) {
  Eigen::VectorXd v(tree.size()); int j=0;
  for(const auto& e:tree)v[j++]=e.second.get_value<double>(); return v;
}
void require(bool ok,const char* why) { if(!ok)throw std::runtime_error(why); }
int main(int argc,char** argv) {
  try {
    require(argc==3,"root and fixture required"); const std::string root=argv[1];
    ptree f;boost::property_tree::read_json(argv[2],f);
    care_confidence_map::TrajectoryRiskEvaluator fk;
    care_confidence_map::VbcPrimitiveModel model;
    std::string error;
    require(fk.initializeKinematics(root+"/src/arm_description/urdf/Arm.urdf","base_link",&error),error.c_str());
    require(model.load(root+"/src/arm_description/urdf/Arm_with_self_filter_collision.urdf",{"base_link","link1"},&error),error.c_str());
    std::vector<Eigen::Vector3d> points;
    for(const auto& e:f.get_child("points")) points.push_back(vec(e.second));
    const auto q0=vec(f.get_child("q0"));
    std::vector<care_confidence_map::VbcPrimitiveFrame> frames;
    require(model.computeTrajectory(fk,{q0},&frames,&error),error.c_str());
    care_confidence_map::InitialShellRecovery ledger;
    std::vector<double> floors;std::vector<bool> recovering;
    for(const auto& p:points) {
      double d=1e9;for(const auto& b:frames.front().primitives)d=std::min(d,b.signedDistance(p));
      double floor;bool rec;
      require(ledger.floor(p,d,.02,.01,&floor,&rec),"fixture start outside recovery domain");
      floors.push_back(floor);recovering.push_back(rec);
    }
    std::cout<<std::setprecision(17);
    for(const auto& e:f.get_child("proposals")) {
      std::vector<Eigen::VectorXd> path;for(const auto& q:e.second.get_child("q"))path.push_back(vec(q.second));
      int budget=256;bool safe=true;
      for(std::size_t j=0;j<points.size();++j)
        safe=model.provePointOutsideContinuousTrajectory(fk,path,points[j],floors[j],&budget)&&safe;
      const auto label=e.second.get<std::string>("label");
      require(safe,"trajectory inside the execution floor should remain geometrically admissible");
      std::cout<<"{\"label\":\""<<label<<"\",\"continuous_safe\":"<<safe<<",\"fk\":"<<256-budget<<"}\n";
    }
    const auto p=points.front(); double floor;bool rec;
    care_confidence_map::InitialShellRecovery check;
    require(check.floor(p,.0198,.02,.01,&floor,&rec),"valid shell rejected");
    require(check.floor(p,.0197,.02,.01,&floor,&rec) && floor==.01,"safe escape from regressed start rejected");
    require(check.baselines.begin()->second==.0198,"replan reset progress history");
    require(!check.floor(p,.01,.02,.01,&floor,&rec),"execution floor admitted");
    check.observedFree(p);
    require(check.floor(p,.0197,.02,.01,&floor,&rec),"fresh free retirement failed");
    int none=0;require(!model.provePointOutsideContinuousTrajectory(fk,{q0},p,.01,&none),"budget fail-open");
    std::cout<<"{\"ledger\":\"PASS\"}\n";
    return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
