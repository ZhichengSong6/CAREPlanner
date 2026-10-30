// Exact rejected-path fixture replay against the production geometry library.
#include <care_confidence_map/vbc_primitive_model.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <iostream>
#include <limits>
using boost::property_tree::ptree;
Eigen::VectorXd vec(const ptree& x) {
  Eigen::VectorXd v(x.size()); int j=0;
  for(const auto& a:x) v[j++]=a.second.get_value<double>();
  return v;
}
void check(bool ok,const char* why) {if(!ok) throw std::runtime_error(why);}
int main(int argc,char** argv) {
 try {
  check(argc==2,"fixture argument required");
  ptree in; boost::property_tree::read_json(argv[1],in);
  bool first=true; std::cout<<"[";
  for(const auto& row:in.get_child("fixtures")) {
   const auto& f=row.second; const auto& request=f.get_child("request");
   care_confidence_map::TrajectoryRiskEvaluator fk;
   care_confidence_map::VbcPrimitiveModel model; std::string error;
   check(fk.initializeKinematics(request.get<std::string>("urdf"),"base_link",&error),error.c_str());
   check(model.load(request.get<std::string>("primitive_urdf"),{"base_link","link1"},&error),error.c_str());
   std::vector<Eigen::VectorXd> q;
   for(const auto& pose:request.get_child("trajectory")) q.push_back(vec(pose.second.get_child("q")));
   int budget=256,diagnostic_budget=256,excluded=0,total=0;
   for(const auto& p:request.get_child("points")) {
    Eigen::Vector3d point=vec(p.second); ++total;
    const bool plain=model.provePointOutsideContinuousTrajectory(fk,q,point,.04,&budget);
    std::string diagnostic;double residual;
    const bool detailed=model.provePointOutsideContinuousTrajectory(fk,q,point,.04,&diagnostic_budget,{},&residual,&diagnostic);
    check(plain==detailed && budget==diagnostic_budget,"diagnostics changed proof decision or FK accounting");
    check(!diagnostic.empty(),"missing proof diagnostic");
    excluded+=plain;
   }
   const auto stamp=f.get<std::string>("audited_ns");
   const bool clear=(stamp=="25284000000" || stamp=="25434000000");
   check((excluded==total)==clear,"clearance classification differs from independent fine-enclosure oracle");
   check(budget>=0 && budget<=256,"budget escaped bounds");
   Eigen::Vector3d far(10,10,10);
   int b=0; check(!model.provePointOutsideContinuousTrajectory(fk,q,far,.04,&b),"zero budget allowed proof");
   b=1; check(!model.provePointOutsideContinuousTrajectory(fk,q,far,.04,&b),"partial path allowed proof");
   b=256; check(model.provePointOutsideContinuousTrajectory(fk,{q[0]},far,.04,&b),"clear single pose rejected");
   std::vector<care_confidence_map::VbcPrimitiveFrame> frames;
   check(model.computeTrajectory(fk,{q[0]},&frames,&error),error.c_str());
   const Eigen::Vector3d inside=frames[0].primitives[0].center;
   b=256; check(!model.provePointOutsideContinuousTrajectory(fk,{q[0]},inside,.04,&b),"inside single pose excluded");
   std::string reason;
   b=0;check(!model.provePointOutsideContinuousTrajectory(fk,q,far,.04,&b,{},nullptr,&reason) &&
       reason=="fk_budget_exhausted","budget exhaustion mislabeled");
   b=256;check(!model.provePointOutsideContinuousTrajectory(fk,{q[0]},inside,.04,&b,{},nullptr,&reason) &&
       reason=="distance_within_margin","geometry failure mislabeled");
   b=256; check(!model.provePointOutsideContinuousTrajectory(fk,q,inside,.04,&b),"initial collision excluded");
   auto invalid=q;invalid[0][0]=std::numeric_limits<double>::quiet_NaN();
   b=256;check(!model.provePointOutsideContinuousTrajectory(fk,invalid,far,.04,&b),"NaN accepted");
   b=256;check(!model.provePointOutsideContinuousTrajectory(fk,{},far,.04,&b),"empty path accepted");
   b=256;check(!model.provePointOutsideContinuousTrajectory(fk,{Eigen::VectorXd::Zero(6)},far,.04,&b),"invalid dimension accepted");
   b=256;check(!model.provePointOutsideContinuousTrajectory(fk,q,far,-.01,&b),"negative margin accepted");
   check(!model.provePointOutsideContinuousTrajectory(fk,q,far,.04,nullptr),"missing budget accepted");
   if(!first)std::cout<<',';first=false;
   std::cout<<"{\"audited_ns\":\""<<stamp<<"\",\"points\":"<<total
     <<",\"excluded\":"<<excluded<<",\"fk_used\":"<<256-budget<<"}";
  }
  std::cout<<"]\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
