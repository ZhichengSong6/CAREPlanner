#include <care_confidence_map/vbc_primitive_model.hpp>
#include <ros/ros.h>
#include <fstream>
#include <iostream>
#include <set>
using namespace care_confidence_map;
struct Voxel {
 Eigen::Vector3d point_base,sample_center_base;
 std::string link_name,source_type,source_collision_name;
 int source_collision_index,sample_index_in_link,sweep_eval_timestep,sweep_original_timestep;
 double raw_sample_radius_m,swept_radius_m,point_center_distance_m,primitive_signed_distance_m,sweep_time_s;
};
void check(bool ok,const std::string& why) {if(!ok)throw std::runtime_error(why);}
int main(int argc,char**argv) {
 ros::init(argc,argv,"continuous_primitive_offline",ros::init_options::AnonymousName|ros::init_options::NoSigintHandler);
 try {
  check(argc==3,"usage: test_continuous_primitive_sweep REPO FIXTURE");
  std::string root=argv[1],error;TrajectoryRiskEvaluator fk;VbcPrimitiveModel model;
  check(fk.initializeKinematics(root+"/src/arm_description/urdf/Arm.urdf","base_link",&error),error);
  check(model.load(root+"/src/arm_description/urdf/Arm_with_self_filter_collision.urdf",{"base_link","link1"},&error),error);
  std::ifstream file(argv[2]); check(bool(file),"fixture missing");
  int fixtures;file>>fixtures;check(fixtures==4,"four real counterexamples required");
  const VbcGrid grid{{-.95,-.95,0},{.95,.95,1.15},.05};
  for(int f=0;f<fixtures;++f) {
   int n;Eigen::Vector3d point;file>>n>>point.x()>>point.y()>>point.z();
   std::vector<Eigen::VectorXd> q;std::vector<double> times;std::vector<int> indices;
   for(int i=0;i<n;++i) {double t;file>>t;Eigen::VectorXd x(7);for(int j=0;j<7;++j)file>>x[j];q.push_back(x);times.push_back(t);indices.push_back(i);}
   check(bool(file),"truncated saved fixture");
   std::vector<VbcPrimitiveFrame> discrete,bounded;std::vector<int> bi;std::vector<double> bt;
   check(model.computeTrajectory(fk,q,&discrete,&error),error);
   // Exercise the optional analytic enclosure explicitly.  Runtime VBC
   // defaults to midpoint sampling without this extra inflation.
   check(model.computeContinuousTrajectory(fk,q,indices,times,&bounded,&bi,&bt,&error,true),error);
   std::vector<VbcPrimitiveFrame> sampled;
   std::vector<int> si; std::vector<double> st;
   check(model.computeContinuousTrajectory(fk,q,indices,times,&sampled,&si,&st,&error,false),error);
   check(sampled.size()==bounded.size() && si==bi && st==bt,
         "disabling motion bound changed adaptive sample schedule");
   for(const auto& frame:sampled) for(double bound:frame.motion_bounds)
     check(bound==0.0,"disabled motion bound was not removed");
   auto includes=[&](const std::vector<Voxel>& vv){for(const auto& v:vv)if((v.point_base-point).norm()<1e-8)return true;return false;};
   check(!includes(buildPrimitiveSweptVoxels<Voxel>(discrete,indices,times,grid,0)),"baseline no longer reproduces missed point");
   check(includes(buildPrimitiveSweptVoxels<Voxel>(bounded,bi,bt,grid,0)),"continuous enclosure missed real counterexample even without static margin");
   check(includes(buildPrimitiveSweptVoxels<Voxel>(bounded,bi,bt,grid,.010)),"10mm continuous enclosure missed real point");
   // Dense independent spot checks supplement the analytical radius proof:
   // actual solid centre/corners must lie inside their interval enclosure.
   for(std::size_t k=0;k+1<q.size();++k)for(int j=0;j<=10;++j) {
    double t=times[k]+(j/10.)*(times[k+1]-times[k]);
    auto it=std::upper_bound(bt.begin(),bt.end(),t);std::size_t ix=it==bt.begin()?0:std::distance(bt.begin(),it)-1;
    std::vector<VbcPrimitiveFrame> actual;
    check(model.computeTrajectory(fk,{q[k]+(j/10.)*(q[k+1]-q[k])},&actual,&error),error);
    for(std::size_t pi=0;pi<actual[0].primitives.size();++pi) {
     const auto& a=actual[0].primitives[pi];const auto& ref=bounded[ix].primitives[pi];
     std::vector<Eigen::Vector3d> local{Eigen::Vector3d::Zero()};
     if(a.kind==PrimitiveKind::Box)for(int x:{-1,1})for(int y:{-1,1})for(int z:{-1,1})local.emplace_back(x*a.half_size.x(),y*a.half_size.y(),z*a.half_size.z());
     else if(a.kind==PrimitiveKind::Cylinder)for(int z:{-1,1})for(int d=0;d<8;++d)local.emplace_back(a.radius*std::cos(d*M_PI/4),a.radius*std::sin(d*M_PI/4),z*a.half_length);
     else for(int axis=0;axis<3;++axis)for(int sign:{-1,1})local.push_back(sign*a.radius*Eigen::Vector3d::Unit(axis));
     for(const auto& p:local)check(ref.signedDistance(a.center+a.rotation*p)<=bounded[ix].motion_bounds[pi]+1e-9,"motion bound failed sampled surface oracle");
    }
   }
   for(std::size_t i=0;i<bt.size();++i)check(bt[i]>=times[bi[i]] && bt[i]<times[bi[i]+1],"entry time grants late visibility deadline");
   std::cout<<"PASS saved counterexample "<<f<<" knots="<<n<<" enclosures="<<bounded.size()<<"\n";
  }
  std::vector<Eigen::VectorXd> q(2,Eigen::VectorXd::Zero(7));q[1][0]=1e8;
  std::vector<VbcPrimitiveFrame> bounded;std::vector<int> bi;std::vector<double> bt;
  check(!model.computeContinuousTrajectory(fk,q,{0,1},{0,.05},&bounded,&bi,&bt,&error)&&bounded.empty(),"capacity emitted partial geometry");
  q[1].setZero();check(!model.computeContinuousTrajectory(fk,q,{0,1},{0,0},&bounded,&bi,&bt,&error),"duplicate knot time accepted");
  q[1][0]=std::numeric_limits<double>::quiet_NaN();check(!model.computeContinuousTrajectory(fk,q,{0,1},{0,.05},&bounded,&bi,&bt,&error),"NaN trajectory accepted");
  auto names=fk.activeJointNames();names[0]="missing_joint";bool rejected=false;try{model.displacementBounds(names,Eigen::VectorXd::Zero(7));}catch(...){rejected=true;}check(rejected,"unbounded kinematic joint accepted");
  std::cout<<"PASS capacity/invalid inputs fail closed; interval surface coverage and conservative timing\n";
  return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
