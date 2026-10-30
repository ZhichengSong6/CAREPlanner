#include <egocentric_arm_planner/repair_executable_view.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
using boost::property_tree::ptree;
Eigen::VectorXd vector(const ptree& t) {
  Eigen::VectorXd v(t.size()); int i=0;
  for(const auto& x:t)v[i++]=x.second.get_value<double>(); return v;
}
Eigen::MatrixXd matrix(const ptree& t) {
  Eigen::MatrixXd m(t.front().second.size(),t.size()); int k=0;
  for(const auto& x:t)m.col(k++)=vector(x.second); return m;
}
void vectorJson(std::ostream& o,const Eigen::VectorXd& v) {
  o<<'[';for(int j=0;j<v.size();++j){if(j)o<<',';o<<v[j];}o<<']';
}
int main(int argc,char** argv) {
  if(argc!=3)return 2;
  ptree tests;boost::property_tree::read_json(argv[1],tests);
  std::ofstream o(argv[2]);o<<std::setprecision(17)<<'[';bool first=true;
  for(const auto& item:tests) {
    const auto& t=item.second;
    egocentric_arm_planner::RepairExecutableSettings settings;
    settings.acceleration=vector(t.get_child("acceleration"));
    settings.brake_dt=t.get<double>("brake_dt",.05);
    settings.max_brake_steps=t.get<int>("max_brake_steps",20);
    if(t.get_child_optional("initial_velocity"))settings.initial_velocity=vector(t.get_child("initial_velocity"));
    for(const auto& p:t.get_child("prefixes"))settings.prefixes.push_back(p.second.get_value<double>());
    std::vector<egocentric_arm_planner::RepairExecutableSample> samples;
    std::string error;
    const bool ok=egocentric_arm_planner::repairExecutableSamples(matrix(t.get_child("q")),
        matrix(t.get_child("u")),t.get<double>("dt",.05),settings,&samples,&error);
    if(!first)o<<',';first=false;
    o<<"{\"ok\":"<<(ok?"true":"false")<<",\"error\":\""<<error<<"\",\"samples\":[";
    bool fs=true;
    for(const auto& s:samples) {
      if(!fs)o<<',';fs=false;
      o<<"{\"prefix\":"<<s.prefix_s<<",\"time\":"<<s.time_s<<",\"q\":";vectorJson(o,s.q);
      o<<",\"q_weights\":[";bool ft=true;
      for(const auto& t:s.q_weights){if(!ft)o<<',';ft=false;o<<'['<<t.first<<','<<t.second<<']';}
      o<<"],\"u_weights\":[";ft=true;
      for(const auto& t:s.u_weights){if(!ft)o<<',';ft=false;o<<'['<<t.first<<',';vectorJson(o,t.second);o<<']';}
      o<<"]}";
    }
    o<<"]}";
  }
  o<<"]\n";
}
