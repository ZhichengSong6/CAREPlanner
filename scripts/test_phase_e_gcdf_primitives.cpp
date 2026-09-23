// Exercise the actual selector (not a reimplementation of its budgets/sorting).
#include <care_confidence_map/vbc_primitive_model.hpp>
#define main care_gcdf_selector_original_main
#include "../src/care_collision_cdf/src/cpp_forbidden_voxel_gpu_shadow_node.cpp"
#undef main
#include <set>
#include <iostream>

namespace care_collision_cdf {
struct GcdfPrimitiveTest {
  using Node=CppForbiddenVoxelGpuShadow;
  using Channel=Node::Channel;
  using Key=std::pair<int,std::array<float,3>>;
  static void check(bool ok,const std::string& why) {if(!ok) throw std::runtime_error(why);}
  static std::set<Key> keys(const std::vector<PairMeta>& pairs) {
    std::set<Key> out;for(const auto& p:pairs) out.emplace(p.original_timestep,p.point);return out;
  }
  static sensor_msgs::PointCloud2 sampleCloud(const std::vector<Anchor>& anchors) {
    sensor_msgs::PointCloud2 c;c.height=1;c.width=anchors.size();
    sensor_msgs::PointCloud2Modifier mod(c);
    mod.setPointCloud2Fields(15,"x",1,7,"y",1,7,"z",1,7,"q0",1,7,"q1",1,7,"q2",1,7,"q3",1,7,
        "q4",1,7,"q5",1,7,"q6",1,7,"confidence",1,7,"current_visibility",1,7,"radius",1,7,
        "eval_timestep",1,5,"original_timestep",1,5);
    mod.resize(anchors.size());
    for(std::size_t i=0;i<anchors.size();++i) {
      const auto& a=anchors[i];float f[13]={};
      std::copy(a.center.begin(),a.center.end(),f);std::copy(a.q.begin(),a.q.end(),f+3);f[12]=a.radius;
      int32_t k[]={a.eval_timestep,a.original_timestep};
      std::memcpy(c.data.data()+i*c.point_step,f,52);std::memcpy(c.data.data()+i*c.point_step+52,k,8);
    }
    return c;
  }
  static void geometryChecks() {
    using namespace care_confidence_map;
    for(int kind=0;kind<3;++kind) for(double inflation:{0.,.015}) for(double margin:{0.,.025,.12}) {
      GcdfPrimitiveAnchor a;a.shape.kind=static_cast<PrimitiveKind>(kind);
      a.shape.center=Eigen::Vector3d(.08,-.1,.13);
      a.shape.rotation=Eigen::AngleAxisd(.71,Eigen::Vector3d(1,2,3).normalized()).toRotationMatrix();
      a.shape.half_size=Eigen::Vector3d(.15,.06,.09);a.shape.radius=.11;a.shape.half_length=.13;
      a.inflation=inflation;a.eval_timestep=0;a.original_timestep=1;a.q[6]=.123456789f;
      auto c=encodePrimitiveAnchors(std_msgs::Header(),{a,a});
      const auto decoded=decodePrimitiveAnchors(c);
      check(decoded.size()==2 && decoded[0].q==a.q && decoded[0].shape.center==a.shape.center &&
        decoded[0].shape.rotation==a.shape.rotation,"wire precision");
      auto bad=c;bad.width--;bad.row_step-=bad.point_step;bad.data.resize(bad.row_step);
      bool rejected=false;try{decodePrimitiveAnchors(bad);}catch(const std::exception&){rejected=true;}
      check(rejected,"partial shape set accepted");
      for(int fault=0;fault<5;++fault) {
        bad=c;
        if(fault==0) bad.fields[0].datatype=7;
        if(fault==1) bad.data.resize(bad.data.size()-1);
        if(fault==2) bad.is_bigendian=true;
        if(fault==3) {double nan=NAN;std::memcpy(bad.data.data()+56,&nan,8);}
        if(fault==4) {int32_t version=99;std::memcpy(bad.data.data(),&version,4);}
        rejected=false;try{decodePrimitiveAnchors(bad);}catch(const std::exception&){rejected=true;}
        check(rejected,"malformed shape cloud accepted");
      }
      auto second=a;second.eval_timestep=1;second.original_timestep=2;
      bad=encodePrimitiveAnchors(std_msgs::Header(),{a,second});
      bad.width--;bad.row_step-=bad.point_step;bad.data.resize(bad.row_step);
      rejected=false;try{decodePrimitiveAnchors(bad);}catch(const std::exception&){rejected=true;}
      check(rejected,"entire missing timestep accepted");
      // Also exercise a primitive centre outside the clipped grid.
      for(double shift:{0.,.4}) {
        a.shape.center.x()+=shift;
        Eigen::Vector3d lo(-.2,-.3,-.1);Eigen::Vector3i dims(11,13,15);
        std::set<std::array<int,3>> fast,reference;
        visitPrimitiveProximity(a.shape,inflation,margin,lo,dims,.04,
          [&](int x,int y,int z,float){fast.insert({x,y,z});});
        for(int x=0;x<dims.x();++x)for(int y=0;y<dims.y();++y)for(int z=0;z<dims.z();++z)
          if(a.shape.signedDistance(lo+.04*Eigen::Vector3d(x,y,z))<=inflation+margin+1e-12)
            reference.insert({x,y,z});
        check(fast==reference,"AABB missed primitive grid centres");
      }
    }
  }
  static void occupiedCoverage(Node& node) {
    using namespace care_confidence_map;
    MapIndex map; map.present.assign(node.grid_size_,1); map.low.assign(node.grid_size_,0);
    map.source_type.assign(node.grid_size_,0); map.confidence.assign(node.grid_size_,0);
    map.current_visibility.assign(node.grid_size_,0);
    const int x=node.coordToIndex(.35,node.x_min_,node.nx_);
    const int y=node.coordToIndex(-.1,node.y_min_,node.ny_);
    const int ground=node.linearIndex(x,y,0), unknown=node.linearIndex(x,y,1);
    map.low[ground]=map.low[unknown]=1; map.source_type[ground]=1;
    GcdfPrimitiveAnchor a; a.shape.kind=PrimitiveKind::Sphere; a.shape.radius=.01;
    a.shape.center={.35,-.1,.063891538}; a.inflation=.015;
    a.eval_timestep=a.original_timestep=0;
    Anchor anchor; anchor.primitive=std::make_shared<GcdfPrimitiveAnchor>(a);
    anchor.eval_timestep=anchor.original_timestep=0;
    const int per_step=node.max_pairs_per_step_, total=node.max_pairs_;
    node.max_pairs_per_step_=1; node.max_pairs_=2;
    auto pairs=node.buildPairs({anchor},map,Channel::FINAL,.025,nullptr,nullptr);
    check(pairs.size()==2,"worst occupied lost behind UNKNOWN at capacity");
    bool physical=false, epistemic=false;
    for(const auto& pair:pairs) {
      if(pair.source_type==1) {
        physical=true;check(pair.approx_body_clearance_m>.025,"fixture is not outside old broadphase");
        check(pair.approx_body_clearance_m-.5*std::sqrt(3.)*.05<0,"fixture not E5 unsafe");
      } else epistemic=true;
    }
    check(physical && epistemic,"physical witness evicted learned UNKNOWN constraint");
    node.max_pairs_=1;
    auto full=node.buildPairs({anchor},map,Channel::FINAL,.025,nullptr,nullptr);
    check(full.size()==1 && full.front().source_type==1,"global capacity lost physical witness or exceeded GPU capacity");
    auto local=node.buildPairs({anchor},map,Channel::LOCAL,.025,nullptr,nullptr);
    check(local.size()==1 && local.front().source_type==0,"LOCAL UNKNOWN band changed");
    auto execution=node.buildPairs({anchor},map,Channel::EXECUTION,.12,nullptr,nullptr);
    check(execution.size()==1 && execution.front().source_type==1,"E5 worst obstacle lost");
    node.max_pairs_per_step_=per_step; node.max_pairs_=total;
    std::cout<<"PASS: FINAL occupied band, both pair caps, E5 witness, unchanged LOCAL band\n";
  }
  static void occupiedSaved(Node& node,care_confidence_map::TrajectoryRiskEvaluator& fk,
      care_confidence_map::VbcPrimitiveModel& model,const std::string& path) {
    using namespace care_confidence_map;
    std::ifstream f(path);check(bool(f),"saved Case011 fixture missing");int n;f>>n;
    std::vector<Eigen::VectorXd> q;
    for(int i=0;i<n;++i){Eigen::VectorXd v(7);for(int j=0;j<7;++j)f>>v[j];q.push_back(v);}
    check(bool(f),"saved Case011 fixture truncated");
    std::vector<VbcPrimitiveFrame> frames;std::string error;
    check(model.computeTrajectory(fk,q,&frames,&error),error);
    std::vector<Anchor> anchors;
    for(int i=0;i<n;++i)for(const auto& p:frames[i].primitives){
      auto a=std::make_shared<GcdfPrimitiveAnchor>();a->shape=p;a->inflation=.015;
      a->eval_timestep=a->original_timestep=i;
      Anchor b;b.eval_timestep=b.original_timestep=i;b.primitive=a;
      for(int j=0;j<7;++j)b.q[j]=a->q[j]=q[i][j];anchors.push_back(b);
    }
    MapIndex map;map.low.assign(node.grid_size_,0);map.present.assign(node.grid_size_,1);
    map.source_type.assign(node.grid_size_,0);map.confidence.assign(node.grid_size_,0);map.current_visibility.assign(node.grid_size_,0);
    int index=node.linearIndex(node.coordToIndex(.35,node.x_min_,node.nx_),node.coordToIndex(-.1,node.y_min_,node.ny_),0);
    map.low[index]=1;map.source_type[index]=1;
    check(node.buildPairs(anchors,map,Channel::LOCAL,.025,nullptr,nullptr).empty(),"old Case011 miss not reproduced");
    auto pairs=node.buildPairs(anchors,map,Channel::FINAL,.025,nullptr,nullptr);
    check(!pairs.empty(),"real Case011 occupied point still omitted");
    double d=1.;for(const auto& p:pairs)d=std::min(d,double(p.approx_body_clearance_m));
    check(std::abs(d-.038891538)<1e-6,"real Case011 clearance mismatch");
    check(d-.5*std::sqrt(3.)*.05<0,"real Case011 not rejected by E5 contract");
    std::cout<<"PASS saved Case011 #37: min centre="<<d<<" volume="<<d-.5*std::sqrt(3.)*.05<<"\n";
  }
  static void run(const std::string& root,const std::string& output,int reps) {
    using namespace care_confidence_map;
    geometryChecks();
    ros::NodeHandle pnh("~");pnh.setParam("output_jsonl",output+"/unused-selector.jsonl");
    Node node;occupiedCoverage(node);std::string error;
    TrajectoryRiskEvaluator samples,fk;VbcPrimitiveModel primitives;
    const auto urdf=root+"/src/arm_description/urdf/Arm_with_self_filter_collision.urdf";
    check(samples.initialize(urdf,root+"/src/care_confidence_map/config/body_samples.yaml","base_link",&error),error);
    check(fk.initializeKinematics(urdf,"base_link",&error),error);
    check(primitives.load(urdf,{"base_link","link1"},&error),error);
    check(primitives.primitives().size()==21,"active primitive count changed");
    occupiedSaved(node,fk,primitives,root+"/scripts/fixtures/gcdf_case011.txt");
    std::ofstream out(output+"/timings.jsonl");out<<std::setprecision(17);
    std::vector<Eigen::VectorXd> endpoints(2,Eigen::VectorXd(7));
    endpoints[0]<<-.22706325988282663,.4568144338672266,-.5396616798361127,-.8349629656679631,
        .23558290158128645,-.27101622546461174,-.5447906370079348;
    endpoints[1]<<-.14339464665723076,.30510076687065185,-.517879483698979,-.9921681614422065,
        .09036627929530537,.0207999567150217,-.29492401753490216;
    // Explicit offline interpolation fixture, not a claimed recorded trajectory.
    for(int pose=0;pose<2;++pose)for(int count:{1,21})for(int map_mode=0;map_mode<3;++map_mode) {
      std::vector<Eigen::VectorXd> q;
      for(int k=0;k<count;++k) q.push_back(endpoints[pose]*(count==1?1.:double(k)/20));
      MapIndex map;map.present.assign(node.grid_size_,1);map.low.assign(node.grid_size_,1);
      map.source_type.assign(node.grid_size_,0);map.confidence.assign(node.grid_size_,0.);
      map.current_visibility.assign(node.grid_size_,0.);
      for(std::size_t i=0;i<node.grid_size_;++i) {
        if(map_mode==1) map.low[i]=i%13==0;
        if(map_mode==2) map.source_type[i]=i%3==0?1:0;
      }
      const auto channel=count==1?Node::Channel::EXECUTION:Node::Channel::LOCAL;
      const double margin=count==1?.12:.025;
      std::set<Key> baseline;
      for(int rep=-2;rep<reps;++rep)for(int turn=0;turn<2;++turn) {
        const int backend=(turn+rep+2)%2;
        const auto begin=Clock::now();std::vector<Anchor> anchors;
        sensor_msgs::PointCloud2 cloud;
        if(backend==0) {
          const auto frames=samples.computeTrajectorySamples(q);check(frames.success,frames.message);
          for(const auto& f:frames.frames)for(const auto& s:f.samples) {
            if(s.link_name=="base_link" || s.link_name=="link1" ||
                s.center_base.x()<node.x_min_ || s.center_base.x()>node.x_max_ ||
                s.center_base.y()<node.y_min_ || s.center_base.y()>node.y_max_ ||
                s.center_base.z()<node.z_min_ || s.center_base.z()>node.z_max_) continue;
            Anchor a;for(int j=0;j<3;++j)a.center[j]=s.center_base[j];
            for(int j=0;j<7;++j)a.q[j]=f.q[j];a.radius=s.radius+.015;
            a.eval_timestep=a.original_timestep=f.timestep_index;anchors.push_back(a);
          }
          cloud=sampleCloud(anchors);
        } else {
          std::vector<VbcPrimitiveFrame> frames;check(primitives.computeTrajectory(fk,q,&frames,&error),error);
          std::vector<GcdfPrimitiveAnchor> wire;
          for(std::size_t k=0;k<frames.size();++k)for(const auto& p:frames[k].primitives) {
            GcdfPrimitiveAnchor a;a.shape=p;a.inflation=.015;a.eval_timestep=a.original_timestep=k;
            for(int j=0;j<7;++j)a.q[j]=q[k][j];wire.push_back(a);
          }
          cloud=encodePrimitiveAnchors(std_msgs::Header(),wire);
        }
        node.geometry_backend_=backend==0?"samples":"primitive";
        anchors=node.decodeAnchors(cloud);check(!anchors.empty(),"roundtrip decode failed");
        const auto selected_begin=Clock::now();std::size_t raw=0;int active=0;
        auto pairs=node.buildPairs(anchors,map,channel,margin,&raw,&active);
        const auto done=Clock::now();
        auto selected=keys(pairs);check(selected.size()==pairs.size(),"duplicate pair");
        if(backend==0) baseline=selected;
        if(backend==1 && rep==-2) {
          // Full-grid oracle at the last q, independent of the AABB enumerator.
          std::set<std::array<float,3>> dense;
          for(int ix=0;ix<node.nx_;++ix)for(int iy=0;iy<node.ny_;++iy)for(int iz=0;iz<node.nz_;++iz) {
            const int i=node.linearIndex(ix,iy,iz);
            if(!map.low[i] || (channel==Node::Channel::EXECUTION && map.source_type[i]!=1))continue;
            Eigen::Vector3d point(node.x_min_+ix*node.resolution_,node.y_min_+iy*node.resolution_,node.z_min_+iz*node.resolution_);
            for(const auto& a:anchors) if(a.original_timestep==count-1 &&
                a.primitive->shape.signedDistance(point)<=.015+margin+1e-12) {dense.insert(node.pointForIndex(i));break;}
          }
          auto oldcap=node.max_pairs_per_step_;auto oldtotal=node.max_pairs_;
          node.max_pairs_per_step_=node.grid_size_;node.max_pairs_=node.grid_size_*count;
          const auto all=node.buildPairs(anchors,map,channel,margin,nullptr,nullptr);
          node.max_pairs_per_step_=oldcap;node.max_pairs_=oldtotal;
          std::set<std::array<float,3>> actual;
          for(const auto& p:all)if(p.original_timestep==count-1)actual.insert(p.point);
          check(actual==dense,"actual primitive selector disagrees with full grid oracle");
        }
        if(rep>=0) {
          int shared=0;for(const auto& k:selected)shared+=baseline.count(k);
          out<<"{\"pose\":"<<pose<<",\"knots\":"<<count<<",\"map\":"<<map_mode
             <<",\"rep\":"<<rep<<",\"backend\":"<<backend<<",\"anchors\":"<<anchors.size()
             <<",\"wire_bytes\":"<<cloud.data.size()<<",\"raw_pairs\":"<<raw<<",\"selected_pairs\":"<<pairs.size()
             <<",\"old_only\":"<<(backend?baseline.size()-shared:0)<<",\"new_only\":"<<(backend?selected.size()-shared:0)
             <<",\"selection_ms\":"<<msBetween(selected_begin,done)<<",\"total_ms\":"<<msBetween(begin,done)<<"}\n";
        }
      }
    }
    std::cout<<"PASS: primitive wire, malformed/partial payload, all shapes/margins/AABB, actual selector dense oracle; timing rows saved\n";
  }
};
}
int main(int argc,char** argv) {
  ros::init(argc,argv,"gcdf_primitive_test");
  try {
    if(argc!=4)throw std::runtime_error("root output_dir repetitions required");
    care_collision_cdf::GcdfPrimitiveTest::run(argv[1],argv[2],std::stoi(argv[3]));
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
  return 0;
}
