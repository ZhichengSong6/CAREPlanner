#include <care_confidence_map/vbc_primitive_model.hpp>
#include <care_confidence_map/primitive_probe_grid.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace care_confidence_map;
void check(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
double referenceDistance(const VbcPrimitive& p,const Eigen::Vector3d& x) {
  const Eigen::Vector3d q=p.rotation.transpose()*(x-p.center);
  if(p.kind==PrimitiveKind::Sphere)return q.norm()-p.radius;
  if(p.kind==PrimitiveKind::Cylinder) {
    Eigen::Vector2d d(q.head<2>().norm()-p.radius,std::abs(q.z())-p.half_length);
    return d.cwiseMax(0.).norm()+std::min(0.,d.maxCoeff());
  }
  const Eigen::Vector3d d=q.cwiseAbs()-p.half_size;
  return d.cwiseMax(0.).norm()+std::min(0.,d.maxCoeff());
}
int main(int argc,char** argv) {
  try {
    check(argc==3,"ROOT OUTPUT required");std::string root=argv[1],out=argv[2],error;
    const auto urdf=root+"/src/arm_description/urdf/Arm.urdf";
    const auto collision=root+"/src/arm_description/urdf/Arm_with_self_filter_collision.urdf";
    TrajectoryRiskEvaluator old,primitive;
    check(old.initialize(urdf,root+"/src/care_confidence_map/config/body_samples.yaml","base_link",&error),error);
    check(primitive.initializePrimitives(urdf,collision,"base_link",.05,Eigen::Vector3d(-.95,-.95,0.),&error),error);
    check(primitive.bodySampleModel().size()==0,"primitive evaluator loaded YAML");
    check(old.hasLegacyBodySamples() && !primitive.hasLegacyBodySamples(),"legacy allocation boundary");
    VbcPrimitiveModel model;check(model.load(collision,{"base_link"},&error),error);
    // Caller supplies the unchanged sensor frames from trajectory_risk.yaml.
    std::vector<std::string> sensors;
    std::ifstream sensor_file(out+"/sensor_frames.txt");std::string name;
    while(std::getline(sensor_file,name))if(!name.empty())sensors.push_back(name);
    check(sensors.size()==8,"need eight actual sensor frames");
    for(auto* e:{&old,&primitive})check(e->prepareFastAudit(sensors,{"base_link","link1"},&error),error);
    check(primitive.fastAuditBodySampleCount()==21,"primitive fast audit geometry count");
    std::ofstream timing(out+"/timings.jsonl"),coverage(out+"/coverage.jsonl");timing<<std::setprecision(17);
    for(int pose=0;pose<3;++pose) {
      Eigen::VectorXd q(7);for(int j=0;j<7;++j)q[j]=pose==0?0.:.35*std::sin((pose+1)*(j+1));
      std::vector<VbcPrimitiveFrame> shapes;check(model.computeTrajectory(primitive,{q},&shapes,&error),error);
      const auto result=primitive.computeTrajectorySamples({q});check(result.success,result.message);
      using Key=std::tuple<std::string,int,int,int>;std::set<Key> actual,expected;
      for(const auto& p:result.frames[0].samples) {
        check(p.radius==0. && p.source_type.find("primitive_")==0 && p.sample_index_in_link==-1,"grid probe disguised as sphere");
        actual.emplace(p.link_name,std::llround((p.center_base.x()+.95)/.05),
          std::llround((p.center_base.y()+.95)/.05),std::llround(p.center_base.z()/.05));
      }
      for(int x=-5;x<46;++x)for(int y=-5;y<46;++y)for(int z=-10;z<36;++z) {
        const Eigen::Vector3d point(-.95+.05*x,-.95+.05*y,.05*z);
        for(const auto& p:shapes[0].primitives)if(referenceDistance(p,point)<=1e-12)
          expected.emplace(p.link_name,x,y,z);
      }
      check(actual==expected,"discrete primitive probes differ from independent full-grid oracle");
      const auto baseline=old.computeTrajectorySamples({q});check(baseline.success,baseline.message);
      coverage<<"{\"pose\":"<<pose<<",\"samples\":"<<baseline.total_samples<<",\"primitive_points\":"<<result.total_samples<<"}\n";
      for(int target=0;target<30;++target) {
        const Eigen::Vector3d point(.3*std::sin(target),.2*std::cos(target),.025*target);
        double expected_distance=INFINITY;
        for(const auto& p:shapes[0].primitives)if(p.link_name!="link1")
          expected_distance=std::min(expected_distance,referenceDistance(p,point));
        double d;std::vector<FastAuditSensorPose> sensors_out;
        check(primitive.evaluateFastAuditForConfiguration(q,point,&sensors_out,&d,&error),error);
        check(std::abs(d-expected_distance)<1e-12 && sensors_out.size()==8,"analytic primitive fast audit mismatch");
      }
      for(int rep=-3;rep<40;++rep)for(int order=0;order<2;++order) {
        bool use_primitive=(rep+order+4)%2;auto& e=use_primitive?primitive:old;
        auto begin=std::chrono::steady_clock::now();auto samples=e.computeTrajectorySamples({q});
        double probe_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        check(samples.success,samples.message);
        begin=std::chrono::steady_clock::now();double d;std::vector<FastAuditSensorPose> s;
        check(e.evaluateFastAuditForConfiguration(q,Eigen::Vector3d(.1,.1,.4),&s,&d,&error),error);
        double audit_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        if(rep>=0)timing<<"{\"pose\":"<<pose<<",\"primitive\":"<<use_primitive<<",\"probe_ms\":"<<probe_ms<<",\"audit_ms\":"<<audit_ms<<"}\n";
      }
    }
    auto q=Eigen::VectorXd::Zero(7).eval();q[0]=NAN;
    check(!primitive.computeTrajectorySamples({q}).success,"nonfinite q accepted");
    check(!primitive.computeTrajectorySamples({Eigen::VectorXd::Zero(6)}).success,"q6 accepted");
    check(!primitive.computeTrajectorySamples({}).success,"empty trajectory accepted");
    VbcPrimitive p;p.kind=PrimitiveKind::Sphere;p.radius=.1;p.center=Eigen::Vector3d(2.,0.,0.);
    const auto points=primitiveProbeGrid({p},.05,Eigen::Vector3d(-.95,-.95,0.));
    check(!points.empty() && points[0].point.x()>1.,"outside-map diagnostic geometry clipped");
    bool rejected=false;try{primitiveProbeGrid({p},0.,Eigen::Vector3d::Zero());}catch(const std::exception&){rejected=true;}
    check(rejected,"zero resolution accepted");
    rejected=false;try{primitiveProbeGrid({p},1e-9,Eigen::Vector3d::Zero());}catch(const std::exception&){rejected=true;}
    check(rejected,"excessive geometry silently truncated");
    std::cout<<"PASS independent probe oracle, 90 analytic target distances, raw q, YAML independence, fail-closed, 240 paired timings\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}
}
