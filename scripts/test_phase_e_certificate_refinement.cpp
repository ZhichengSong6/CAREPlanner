#include <care_confidence_map/vbc_primitive_model.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <iostream>
#include <iomanip>

using Vec=Eigen::VectorXd;
using boost::property_tree::ptree;
using care_confidence_map::VbcPrimitiveModel;
using care_confidence_map::VbcPrimitiveFrame;
Vec vec(const ptree& t) {
  Vec v(t.size());int i=0;for(const auto& x:t)v[i++]=x.second.get_value<double>();return v;
}
void check(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
int main(int argc,char** argv) {
 try {
  check(argc==3,"usage: test_certificate_refinement REPO RECORDS");
  std::string root=argv[1],error;ptree rows;boost::property_tree::read_json(argv[2],rows);
  care_confidence_map::TrajectoryRiskEvaluator fk;VbcPrimitiveModel model;
  check(fk.initializeKinematics(root+"/src/arm_description/urdf/Arm.urdf","base_link",&error),error.c_str());
  check(model.load(root+"/src/arm_description/urdf/Arm_with_self_filter_collision.urdf",{"base_link","link1"},&error),error.c_str());
  std::cout<<std::setprecision(17);int rowid=0,unsafe=0,unattainable=0;
  for(const auto& row:rows) {
    const auto& c=row.second.get_child("context");const auto& e=row.second.get_child("evidence");
    const Vec point_vector=vec(row.second.get_child("test_point"));
    const Eigen::Vector3d point=point_vector;
    const double floor=row.second.get<double>("test_floor",.02);
    std::vector<Vec> q,v,curve;std::vector<double> ts;
    for(const auto& x:c.get_child("q"))q.push_back(vec(x.second));
    for(const auto& x:c.get_child("v"))v.push_back(vec(x.second));
    for(const auto& x:c.get_child("times_s"))ts.push_back(x.second.get_value<double>());
    for(std::size_t k=1;k<q.size();++k) {
      const double dt=ts[k]-ts[k-1];Vec bend=Vec::Zero(7);
      if((q[k]-q[k-1]-.5*dt*(v[k]+v[k-1])).cwiseAbs().maxCoeff()<=1e-9)
        bend=dt*(v[k-1]-v[k])/8.;
      curve.push_back(bend);
    }
    q.insert(q.begin(),vec(e.get_child("measured_q")));curve.insert(curve.begin(),Vec::Zero(7));
    // Dense samples are an independent upper bound on true minimum clearance,
    // not an execution certificate or reconstruction of the historical map.
    double upper=1e6;
    for(std::size_t k=1;k<q.size();++k)for(int i=0;i<=200;++i) {
      const double u=i/200.;Vec pose=q[k-1]+u*(q[k]-q[k-1])+4*u*(1-u)*curve[k-1];
      std::vector<VbcPrimitiveFrame> frames;check(model.computeTrajectory(fk,{pose},&frames,&error),error.c_str());
      for(const auto& p:frames.front().primitives)upper=std::min(upper,p.signedDistance(point)-floor);
    }
    VbcPrimitiveModel::ProofFkCache base_cache;int base_budget=256;double base=0.;
    bool safe=model.provePointOutsideContinuousTrajectory(fk,q,point,floor,&base_budget,curve,&base,&error,&base_cache);
    if(!safe) {
      int budget=256;double slack=0.;VbcPrimitiveModel::ProofFkCache cache;
      check(!model.provePointOutsideContinuousTrajectory(fk,q,point,floor,&budget,curve,&slack,&error,&cache,.002),
            "optional quality erased an unproved/unsafe motion");
      ++unsafe;++rowid;continue;
    }
    if(upper<.002)++unattainable;
    for(int allowance:{0,1,4,8,16,32,base_budget}) {
      if(allowance>base_budget)continue;
      auto cache=base_cache;int budget=allowance;double slack=0.;
      check(model.provePointOutsideContinuousTrajectory(fk,q,point,floor,&budget,curve,&slack,&error,&cache,.002),
            "lost complete original certificate during optional refinement");
      check(budget>=0 && budget<=allowance,"FK work exceeded allowance");
      check(slack>=base-1e-12 && slack<=upper+1e-12,"invalid or weaker clearance bound");
      if(allowance==0)check(cache.misses==base_cache.misses,"zero budget computed FK");
      std::cout<<"{\"row\":"<<rowid<<",\"base_m\":"<<base<<",\"upper_m\":"<<upper
          <<",\"allowance\":"<<allowance<<",\"used_fk\":"<<allowance-budget<<",\"refined_m\":"<<slack<<"}\n";
    }
    // Deterministic interruption before or between the two child proofs must
    // keep the parent, never return a certificate for only part of the path.
    for(int callbacks:{0,1,2,3}) {
      auto cache=base_cache;int budget=base_budget,calls=0;double slack=0.;
      check(model.provePointOutsideContinuousTrajectory(fk,q,point,floor,&budget,curve,&slack,&error,&cache,.002,
          [&](){return calls++<callbacks;}),"deadline lost the original complete proof");
      check(slack>=base-1e-12 && slack<=upper+1e-12,"deadline emitted a partial interval proof");
      if(callbacks<=1)check(std::abs(slack-base)<1e-12,"partial children replaced parent");
      if(callbacks==0)check(budget==base_budget,"expired optional deadline spent FK");
    }
    // A sample strictly inside the requested floor is a real counterexample.
    int budget=256;double slack=0.;VbcPrimitiveModel::ProofFkCache cache;
    check(!model.provePointOutsideContinuousTrajectory(fk,q,point,floor+upper+.0001,&budget,
        curve,&slack,&error,&cache,.002),"refinement accepted a sampled margin violation");
    ++rowid;
  }
  check(unsafe>0 && unattainable>0,"fixtures must include unsafe and unattainable targets");
  std::cout<<"{\"status\":\"PASS\",\"records\":"<<rowid<<",\"unsafe_controls\":"<<unsafe
      <<",\"unattainable_targets\":"<<unattainable<<"}\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
