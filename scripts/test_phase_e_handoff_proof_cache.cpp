#include <care_confidence_map/vbc_primitive_model.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <cassert>
#include <fstream>
#include <iostream>
using Vec=Eigen::VectorXd;using boost::property_tree::ptree;
Vec vec(const ptree& t){Vec v(t.size());int j=0;for(const auto& x:t)v[j++]=x.second.get_value<double>();return v;}
int main(int argc,char** argv) {
 if(argc!=3)return 2;std::string root=argv[1],error;
 care_confidence_map::TrajectoryRiskEvaluator fk;care_confidence_map::VbcPrimitiveModel model;
 assert(fk.initializeKinematics(root+"/src/arm_description/urdf/Arm.urdf","base_link",&error));
 assert(model.load(root+"/src/arm_description/urdf/Arm_with_self_filter_collision.urdf",{"base_link","link1"},&error));
 ptree fixture;boost::property_tree::read_json(argv[2],fixture);
 std::vector<Vec> q,v,c;std::vector<double> t;
 for(const auto& x:fixture.get_child("context.q"))q.push_back(vec(x.second));
 for(const auto& x:fixture.get_child("context.v"))v.push_back(vec(x.second));
 for(const auto& x:fixture.get_child("context.times_s"))t.push_back(x.second.get_value<double>());
 for(std::size_t i=1;i<q.size();++i){double h=t[i]-t[i-1];Vec bend=Vec::Zero(7);if((q[i]-q[i-1]-.5*h*(v[i]+v[i-1])).lpNorm<Eigen::Infinity>()<=1e-9)bend=h*(v[i-1]-v[i])/8.;c.push_back(bend);}
 q.insert(q.begin(),vec(fixture.get_child("certificate.measured_q")));c.insert(c.begin(),Vec::Zero(7));
 Eigen::Vector3d point=vec(fixture.get_child("point"));double floor=fixture.get<double>("floor"),target=fixture.get<double>("required_residual");
 care_confidence_map::VbcPrimitiveModel::ProofFkCache cache;int remaining=256;double slack=0.;std::string reason;
 const auto prove=[&](double desired){return model.provePointOutsideContinuousTrajectory(fk,q,point,floor,&remaining,c,&slack,&reason,&cache,desired);};
 assert(prove(0.));double baseline=slack;int baseline_fk=256-remaining;
 assert(prove(target));double stronger=slack;assert(stronger>=baseline && remaining>=0);
 int before=remaining;assert(prove(0.) && slack>=stronger && remaining==before);
 // A fully spent ledger may replay already proved immutable geometry.
 remaining=0;assert(prove(0.) && slack>=stronger && remaining==0);
 // A changed measured bridge still needs new geometry and cannot borrow the old cover.
 auto changed=q;changed.front()[0]+=1e-4;
 assert(!model.provePointOutsideContinuousTrajectory(fk,changed,point,floor,&remaining,c,&slack,&reason,&cache));
 assert(reason=="fk_budget_exhausted");
 // Revalidate its one changed bridge while preserving all immutable subinterval strength.
 remaining=256-cache.misses;assert(model.provePointOutsideContinuousTrajectory(fk,changed,point,floor,&remaining,c,&slack,&reason,&cache));
 assert(cache.misses<=256 && slack>=std::min(stronger,baseline));
 auto bend=c;bend.back()[0]+=.01;remaining=0;
 assert(!model.provePointOutsideContinuousTrajectory(fk,q,point,floor,&remaining,bend,&slack,&reason,&cache));
 // A new floor or point must be evaluated against geometry again.
 remaining=256-cache.misses;
 assert(!model.provePointOutsideContinuousTrajectory(fk,q,point,1.,&remaining,c,&slack,&reason,&cache));
 assert(reason=="distance_within_margin");
 std::vector<care_confidence_map::VbcPrimitiveFrame> frames;assert(model.computeTrajectory(fk,{q[1]},&frames,&error));
 Eigen::Vector3d inside=frames[0].primitives.back().center;
 assert(!model.provePointOutsideContinuousTrajectory(fk,{q[1]},inside,0.,&remaining,{},&slack,&reason,&cache));
 // An interrupted optional split retains the complete parent cover.
 care_confidence_map::VbcPrimitiveModel::ProofFkCache partial;remaining=256;
 assert(model.provePointOutsideContinuousTrajectory(fk,q,point,floor,&remaining,c,&slack,&reason,&partial));
 double parent=slack;remaining=1;
 assert(model.provePointOutsideContinuousTrajectory(fk,q,point,floor,&remaining,c,&slack,&reason,&partial,.002));
 assert(slack>=parent);double partial_slack=slack;remaining=0;
 assert(model.provePointOutsideContinuousTrajectory(fk,q,point,floor,&remaining,c,&slack,&reason,&partial) && slack>=partial_slack);
 // Reloading even the same model invalidates cached frames and covers.
 assert(model.load(root+"/src/arm_description/urdf/Arm_with_self_filter_collision.urdf",{"base_link","link1"},&error));
 assert(!model.provePointOutsideContinuousTrajectory(fk,q,point,floor,&remaining,c,&slack,&reason,&cache));
 std::cout<<"PASS baseline_fk="<<baseline_fk<<" baseline_m="<<baseline<<" stronger_m="<<stronger
          <<" demand_m="<<target<<" attainable="<<(stronger>target)<<" covers/replay/bridge/floor/point/curvature/partial/model_reload\n";
}
