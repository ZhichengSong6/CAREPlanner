#include <egocentric_arm_planner/repair_executable_view.hpp>
#include <egocentric_arm_planner/repair_executable_iterates.hpp>
#include <care_confidence_map/initial_shell_recovery.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
using boost::property_tree::ptree;
Eigen::VectorXd vec(const ptree& t){Eigen::VectorXd v(t.size());int j=0;for(const auto& x:t)v[j++]=x.second.get_value<double>();return v;}
Eigen::MatrixXd mat(const ptree& t){Eigen::MatrixXd m(t.front().second.size(),t.size());int k=0;for(const auto& x:t)m.col(k++)=vec(x.second);return m;}
int main(int argc,char** argv){
  if(argc!=3)return 2;
  care_confidence_map::InitialShellRecovery ledger;double floor;bool recovering;
  auto require=[](bool ok){if(!ok)throw std::runtime_error("margin/iterate contract failed");};
  require(ledger.floor(Eigen::Vector3d::Zero(),.0205357,.02,.01,&floor,&recovering) && floor==.02 && !recovering);
  require(ledger.floor(Eigen::Vector3d::Zero(),.0205357,.02,.01,&floor,&recovering,true) && floor==.01 && recovering);
  require(!ledger.floor(Eigen::Vector3d::Zero(),.00999,.02,.01,&floor,&recovering,true));
  egocentric_arm_planner::RepairExecutableIterates iterate;
  iterate.bind("same_owner");iterate.retain(Eigen::MatrixXd::Zero(7,2),Eigen::MatrixXd::Zero(7,1));
  iterate.rejectTrial();iterate.invalidateCurve();iterate.bind("same_owner");
  require(iterate.step_scale==.5 && !iterate.available());
  iterate.rejectTrial();require(iterate.step_scale==.25);
  iterate.bind("new_owner");require(iterate.step_scale==1.);
  ptree fixtures;boost::property_tree::read_json(argv[1],fixtures);
  std::ofstream out(argv[2]);out<<std::setprecision(17)<<'[';bool first=true;
  for(const auto& item:fixtures){
    const auto& f=item.second;egocentric_arm_planner::RepairExecutableSettings s;
    s.acceleration=vec(f.get_child("acceleration"));s.initial_velocity=vec(f.get_child("initial_velocity"));
    s.prefixes={f.get<double>("prefix")};
    std::vector<egocentric_arm_planner::RepairExecutableSample> samples;std::string error;
    const bool ok=egocentric_arm_planner::repairLiveExecutableSamples(mat(f.get_child("q")),mat(f.get_child("u")),
        f.get<double>("dt",.05),s,vec(f.get_child("measured_q")),vec(f.get_child("measured_v")),&samples,&error);
    if(!first)out<<',';first=false;
    out<<"{\"ok\":"<<(ok?"true":"false")<<",\"error\":\""<<error<<"\",\"samples\":[";
    for(std::size_t i=0;i<samples.size();++i){if(i)out<<',';out<<"{\"t\":"<<samples[i].time_s<<",\"q\":[";
      for(int j=0;j<samples[i].q.size();++j){if(j)out<<',';out<<samples[i].q[j];}out<<"]}";}
    out<<"]}";
  }
  out<<"]\n";
}
