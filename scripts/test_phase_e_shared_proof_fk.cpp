#include <care_confidence_map/vbc_primitive_model.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <iostream>
#include <iomanip>
#include <stdexcept>
using Vec=Eigen::VectorXd;
using boost::property_tree::ptree;
Vec vector(const ptree& t) { Vec v(t.size()); int j=0; for(const auto& x:t)v[j++]=x.second.get_value<double>(); return v; }
void require(bool value,const char* message) { if(!value)throw std::runtime_error(message); }
int main(int argc,char** argv) {
  require(argc==3,"root and recorded curve required");
  ptree record;boost::property_tree::read_json(argv[2],record);
  const std::string root=argv[1];std::string error,reason;
  care_confidence_map::TrajectoryRiskEvaluator fk;care_confidence_map::VbcPrimitiveModel model;
  require(fk.initializeKinematics(root+"/src/arm_description/urdf/Arm.urdf","base_link",&error),"FK load");
  require(model.load(root+"/src/arm_description/urdf/Arm_with_self_filter_collision.urdf",{"base_link","link1"},&error),"model load");
  std::vector<Vec> q,v,bends;std::vector<double> times;
  for(const auto& x:record.get_child("context.q"))q.push_back(vector(x.second));
  for(const auto& x:record.get_child("context.v"))v.push_back(vector(x.second));
  for(const auto& x:record.get_child("context.times_s"))times.push_back(x.second.get_value<double>());
  for(std::size_t k=1;k<q.size();++k) {
    const double dt=times[k]-times[k-1];Vec bend=Vec::Zero(7);
    if((q[k]-q[k-1]-.5*dt*(v[k]+v[k-1])).lpNorm<Eigen::Infinity>()<=1e-9)bend=dt*(v[k-1]-v[k])/8.;
    bends.push_back(bend);
  }
  std::vector<care_confidence_map::VbcPrimitiveFrame> frames;
  require(model.computeTrajectory(fk,{q.front()},&frames,&error),"initial geometry");
  std::vector<Eigen::Vector3d> points;
  // Controlled near-body grid. Every included point individually has a full
  // continuous 20 mm proof on the exact recorded curve, not a sample-only test.
  for(int iz=3;iz<=22 && points.size()<64;++iz)
    for(int ix=-10;ix<=10 && points.size()<64;++ix)
      for(int iy=-10;iy<=10 && points.size()<64;++iy) {
        Eigen::Vector3d p(.05*ix,.05*iy,.05*iz);double d=1e9;
        for(const auto& body:frames.front().primitives)d=std::min(d,body.signedDistance(p));
        if(d<.025 || d>.15)continue;
        int budget=256;double slack;
        if(model.provePointOutsideContinuousTrajectory(fk,q,p,.02,&budget,bends,&slack))points.push_back(p);
      }
  require(points.size()==64,"controlled point count");
  int old_budget=256,new_budget=256,old_proved=0,new_proved=0;
  care_confidence_map::VbcPrimitiveModel::ProofFkCache cache;
  double slack;
  for(const auto& p:points) {
    old_proved+=model.provePointOutsideContinuousTrajectory(fk,q,p,.02,&old_budget,bends,&slack,&reason);
    new_proved+=model.provePointOutsideContinuousTrajectory(fk,q,p,.02,&new_budget,bends,&slack,&reason,&cache);
  }
  require(old_budget==0 && old_proved<64,"uncached workload should exhaust original allowance");
  require(new_proved==64 && new_budget>0,"shared FK should complete the same workload within 256");
  // Reuse only geometry; stronger margins and an unsafe point still fail.
  const auto inside=frames.front().primitives.front().center;
  require(!model.provePointOutsideContinuousTrajectory(fk,q,inside,.02,&new_budget,bends,&slack,&reason,&cache),"cached unsafe point admitted");
  require(reason=="distance_within_margin","unsafe reason");
  require(!model.provePointOutsideContinuousTrajectory(fk,q,points.front(),1.,&new_budget,bends,&slack,&reason,&cache),"cached result reused across margins");
  require(reason=="distance_within_margin","stronger margin reason");
  // The measured brake is separately proved with the same remaining ledger.
  Vec actual=vector(record.get_child("actual")),velocity=vector(record.get_child("velocity"));
  Vec accel=vector(record.get_child("context.acceleration"));
  std::vector<Vec> brake{actual},brake_bends;
  for(int step=0;step<20 && velocity.cwiseAbs().maxCoeff()>1e-6;++step) {
    Vec next=velocity;
    for(int j=0;j<7;++j)next[j]=std::copysign(std::max(0.,std::abs(velocity[j])-accel[j]*.05),velocity[j]);
    brake.push_back(brake.back()+.025*(velocity+next));brake_bends.push_back(.05*(velocity-next)/8.);velocity=next;
  }
  const Eigen::Vector3d recorded_point=vector(record.get_child("point"));
  bool old_brake=model.provePointOutsideContinuousTrajectory(fk,brake,recorded_point,.01,&old_budget,brake_bends,&slack,&reason);
  require(!old_brake && reason=="fk_budget_exhausted","old exhausted brake control");
  require(model.provePointOutsideContinuousTrajectory(fk,brake,recorded_point,.01,&new_budget,brake_bends,&slack,&reason,&cache),"separate measured brake proof");
  std::cout<<std::setprecision(17)<<"{\"status\":\"PASS\",\"points\":"<<points.size()
    <<",\"old_proved\":"<<old_proved<<",\"new_proved\":"<<new_proved
    <<",\"old_remaining_fk\":"<<old_budget<<",\"new_remaining_fk\":"<<new_budget
    <<",\"cache_hits\":"<<cache.hits<<",\"cache_misses\":"<<cache.misses
    <<",\"separate_brake_clearance_m\":"<<slack
    <<",\"unsafe_point_rejected\":true,\"stronger_margin_rechecked\":true}\n";
}
