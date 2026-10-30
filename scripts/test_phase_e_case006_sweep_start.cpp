// CASE006 regression against production geometry, no ROS master required.
#include <care_confidence_map/vbc_primitive_model.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <fstream>
#include <iostream>
#include <set>
using namespace care_confidence_map;
using boost::property_tree::ptree;
struct Voxel {
 Eigen::Vector3d point_base,sample_center_base;
 std::string link_name,source_type,source_collision_name;
 int source_collision_index,sample_index_in_link,sweep_eval_timestep,sweep_original_timestep;
 double raw_sample_radius_m,swept_radius_m,point_center_distance_m,primitive_signed_distance_m,sweep_time_s;
};
void check(bool condition,const std::string& why) {if(!condition)throw std::runtime_error(why);}
int main(int argc,char** argv) {
 try {
  check(argc==2,"usage: test_case006_sweep INPUT_JSON");
  ptree in;boost::property_tree::read_json(argv[1],in);std::string error;
  TrajectoryRiskEvaluator fk;VbcPrimitiveModel model;
  check(fk.initializeKinematics(in.get<std::string>("urdf"),"base_link",&error),error);
  check(model.load(in.get<std::string>("primitive_urdf"),{"base_link","link1"},&error),error);
  std::vector<Eigen::VectorXd> q;std::vector<int> ids;std::vector<double> times;
  for(const auto& row:in.get_child("trajectory")) {
    Eigen::VectorXd v(7);int j=0;for(const auto& x:row.second.get_child("q"))v[j++]=x.second.get_value<double>();
    check(j==7,"q dimensions");q.push_back(v);ids.push_back(ids.size());times.push_back(row.second.get<double>("time_s"));
  }
  const Eigen::Vector3d point(.05,-.2,.75);
  const VbcGrid grid{{-.95,-.95,0},{.95,.95,1.15},.05};
  std::vector<VbcPrimitiveFrame> bounded,sampled,static_frames;
  std::vector<int> bi,si;std::vector<double> bt,st;
  // Omitted optional flag must now choose the complete enclosure.
  check(model.computeContinuousTrajectory(fk,q,ids,times,&bounded,&bi,&bt,&error),error);
  check(model.computeContinuousTrajectory(fk,q,ids,times,&sampled,&si,&st,&error,false),error);
  check(model.computeTrajectory(fk,q,&static_frames,&error),error);
  auto match=[&](const std::vector<Voxel>& vv) -> const Voxel* {
    for(const auto& v:vv)if((v.point_base-point).norm()<1e-8)return &v;return nullptr;
  };
  const auto old=buildPrimitiveSweptVoxels<Voxel>(sampled,si,st,grid,.04);
  const auto fixed=buildPrimitiveSweptVoxels<Voxel>(bounded,bi,bt,grid,.04);
  check(!match(old),"legacy midpoint counterexample not reproduced");
  check(match(fixed) && match(fixed)->sweep_time_s==0,"initial point missing or deadline late");
  check(!match(buildPrimitiveSweptVoxels<Voxel>(bounded,bi,bt,grid,.02)),"20mm shell unexpectedly contains test point");
  // Every original knot and dense interior pose's 40mm grid support must be
  // included, with an entry timestamp no later than that pose.
  int spot_checks=0;
  for(std::size_t k=0;k+1<q.size();++k)for(int j=0;j<=8;++j) {
    const double a=j/8.,t=times[k]+a*(times[k+1]-times[k]);
    std::vector<VbcPrimitiveFrame> exact;
    check(model.computeTrajectory(fk,{q[k]+a*(q[k+1]-q[k])},&exact,&error),error);
    const auto voxels=buildPrimitiveSweptVoxels<Voxel>(exact,{0},{t},grid,.04);
    for(const auto& v:voxels) {
      auto it=std::find_if(fixed.begin(),fixed.end(),[&](const Voxel& x){return (x.point_base-v.point_base).norm()<1e-8;});
      check(it!=fixed.end() && it->sweep_time_s<=t+1e-12,"interval support/timing containment failed");++spot_checks;
    }
  }
  for(int count:{1,2}) {
    std::vector<Eigen::VectorXd> hold(count,q.front());std::vector<int> ix(count,0);std::vector<double> t(count,0);
    if(count==2){ix[1]=1;t[1]=.1;}
    check(model.computeContinuousTrajectory(fk,hold,ix,t,&bounded,&bi,&bt,&error),error);
    check(match(buildPrimitiveSweptVoxels<Voxel>(bounded,bi,bt,grid,.04)),"static recovery origin disappeared");
  }
  q[1][0]=1e8;
  check(!model.computeContinuousTrajectory(fk,{q[0],q[1]},{0,1},{0,.05},&bounded,&bi,&bt,&error)
        &&bounded.empty()&&bi.empty()&&bt.empty(),"capacity did not reject atomically");
  std::cout<<"{\"status\":\"PASS\",\"dense_grid_support_checks\":"<<spot_checks
    <<",\"old_missed_initial_point\":true,\"default_enclosure_recovers_point\":true,\"earliest_s\":0,"
    <<"\"hold_and_single_pose\":true,\"capacity_fail_closed\":true}\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
