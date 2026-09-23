// Production map methods with a private TF buffer; no actuator or case runner.
#define main confidence_map_original_main
#include "../src/care_confidence_map/src/confidence_map_node.cpp"
#undef main
#include <chrono>
#include <fstream>
#include <iomanip>
#include <set>

struct ColdStartPrimitiveTest {
  using Node = ConfidenceMapNode;
  static void check(bool value, const std::string& why) {
    if (!value) throw std::runtime_error(why);
  }
  static std::set<int> support(const Node& n) {
    std::set<int> cells;
    for (std::size_t i=0; i<n.grid_points_.size(); ++i)
      if (n.effectiveConfidence(n.grid_points_[i]) > .5f) cells.insert(i);
    return cells;
  }
  static void configure(Node& n, const std::string& backend) {
    n.pnh_.setParam("current_body_prior/geometry_backend", backend);
    check(n.loadConfidenceMapParams(), "map config");
    check(n.loadCurrentBodyPriorParams(), "prior config " + backend);
    n.generateGlobalGrid();
    n.current_body_prior_tf_timeout_ = 0.;
  }
  static void setPose(Node& n, const care_confidence_map::TrajectoryRiskEvaluator& fk,
                      const std::vector<std::string>& frames, const Eigen::VectorXd& q,
                      bool incomplete=false) {
    n.tf_buffer_.clear();
    std::vector<care_confidence_map::FramePoseInBase> poses; std::string error;
    check(fk.computeFramePosesForConfiguration(q,frames,&poses,&error),error);
    for (const auto& p : poses) {
      if (p.frame_name==n.map_frame_ || (incomplete && p.frame_name=="wrist_link3")) continue;
      geometry_msgs::TransformStamped t;
      t.header.frame_id=n.map_frame_; t.header.stamp=ros::Time::now(); t.child_frame_id=p.frame_name;
      t.transform.translation.x=p.translation_base.x(); t.transform.translation.y=p.translation_base.y();
      t.transform.translation.z=p.translation_base.z();
      Eigen::Quaterniond r(p.rotation_base);
      t.transform.rotation.x=r.x(); t.transform.rotation.y=r.y();
      t.transform.rotation.z=r.z(); t.transform.rotation.w=r.w();
      check(n.tf_buffer_.setTransform(t,"offline_test",true),"set valid TF");
    }
  }
  // Independent, straightforward reference formula (not the production scalar SDF).
  static double distance(const care_confidence_map::VbcPrimitive& p, const Eigen::Vector3d& x) {
    using care_confidence_map::PrimitiveKind;
    const Eigen::Vector3d y=p.rotation.transpose()*(x-p.center);
    if (p.kind==PrimitiveKind::Sphere) return y.norm()-p.radius;
    if (p.kind==PrimitiveKind::Cylinder) {
      const Eigen::Vector2d d(y.head<2>().norm()-p.radius,std::abs(y.z())-p.half_length);
      return d.cwiseMax(0.).norm()+std::min(d.maxCoeff(),0.);
    }
    const Eigen::Vector3d d=y.cwiseAbs()-p.half_size;
    return d.cwiseMax(0.).norm()+std::min(d.maxCoeff(),0.);
  }
  static std::set<int> oracle(const Node& n, const std::vector<care_confidence_map::VbcPrimitive>& shapes) {
    std::set<int> expected;
    for (int x=0;x<n.nx_;++x) for(int y=0;y<n.ny_;++y) for(int z=0;z<n.nz_;++z) {
      Eigen::Vector3d point(n.x_min_+x*n.resolution_,n.y_min_+y*n.resolution_,n.z_min_+z*n.resolution_);
      for(const auto& p:shapes) if(distance(p,point)<=n.currentBodyPriorInflationForLink(p.link_name)+1e-12) {
        expected.insert(n.gridLinearIndex(x,y,z));break;
      }
    }
    return expected;
  }
  static void run(const std::string& root,const std::string& output) {
    using namespace care_confidence_map;
    Node old, current;
    configure(old,"samples");
    // A nonexistent YAML must be irrelevant to primitive initialization.
    current.pnh_.setParam("current_body_prior/body_samples_file",output+"/MISSING.yaml");
    configure(current,"primitive");
    check(current.current_body_sample_model_.size()==0,"primitive loaded YAML");
    check(current.current_body_primitive_model_.primitives().size()==26,"prior must include base/link1");
    TrajectoryRiskEvaluator fk; std::string error;
    check(fk.initializeKinematics(root+"/src/arm_description/urdf/Arm_with_self_filter_collision.urdf","base_link",&error),error);
    const auto frames=current.current_body_primitive_model_.frames();
    Eigen::VectorXd q=Eigen::VectorXd::Zero(7);
    for(Node* n:{&old,&current}) {
      n->current_body_prior_lock_after_complete_refresh_=true;
      setPose(*n,fk,frames,q,true);
      std::string msg;
      n->refreshCurrentBodyPrior(ros::Time::now(),"partial",&msg);
      check(!n->current_body_prior_active_ && !n->current_body_prior_locked_ && support(*n).empty(),"partial TF exposed prior");
      check(n->last_body_prior_skipped_samples_>0,"partial TF not reported");
      for(auto& cell:n->grid_points_) {
        cell.confidence=.2f;cell.current_visibility=.3f;cell.occupancy=1.f;cell.last_seen_time=42.;
      }
      setPose(*n,fk,frames,q);
      check(n->refreshCurrentBodyPrior(ros::Time::now(),"complete",&msg),"refresh failed");
      check(n->current_body_prior_active_ && n->current_body_prior_locked_,"complete TF not locked");
      const auto initial=support(*n);check(!initial.empty(),"empty prior fixture");
      for(const auto& cell:n->grid_points_)
        check(cell.confidence==.2f && cell.current_visibility==.3f && cell.occupancy==1.f && cell.last_seen_time==42.,"sensor provenance overwritten");
      setPose(*n,fk,frames,Eigen::VectorXd::Constant(7,.3));
      check(n->refreshCurrentBodyPrior(ros::Time::now(),"moved",&msg),"legacy locked response");
      check(support(*n)==initial,"locked prior moved");
      const auto markers=n->makeCurrentBodyPriorMarkerArray();
      check(markers.markers.size()>1,"missing active markers");
      std_srvs::Trigger::Request req;std_srvs::Trigger::Response res;
      n->handleDeactivateBodyPrior(req,res);
      check(support(*n).empty() && n->current_body_prior_locked_,"deactivate/unlock error");
      for(const auto& cell:n->grid_points_)
        check(cell.confidence==.2f && cell.current_visibility==.3f && cell.occupancy==1.f && cell.last_seen_time==42.,"deactivate erased observations");
      n->refreshCurrentBodyPrior(ros::Time::now(),"after_deactivate",&msg);
      check(support(*n).empty(),"deactivated prior resurrected");
      check(n->makeCurrentBodyPriorMarkerArray().markers.size()==1,"inactive markers retained");
      n->current_body_prior_locked_=false;n->current_body_prior_lock_after_complete_refresh_=false;
      for(auto& cell:n->grid_points_) {
        cell.confidence=cell.current_visibility=cell.occupancy=cell.bootstrap_confidence=0.f;
        cell.last_seen_time=-1.;
      }
    }
    std::ofstream timings(output+"/timings.jsonl"),coverage(output+"/coverage.jsonl");
    timings<<std::setprecision(17);
    // Explicit synthetic configurations: no cases, goals, or trajectory repair.
    for(int pose=0;pose<3;++pose) {
      for(int j=0;j<7;++j) q[j]=pose==0?0.:.35*std::sin((pose+1)*(j+1));
      setPose(old,fk,frames,q);setPose(current,fk,frames,q);
      std::string msg;
      old.refreshCurrentBodyPrior(ros::Time::now(),"oracle",&msg);
      current.refreshCurrentBodyPrior(ros::Time::now(),"oracle",&msg);
      std::vector<VbcPrimitiveFrame> geometry;
      check(current.current_body_primitive_model_.computeTrajectory(fk,{q},&geometry,&error),error);
      const auto expected=oracle(current,geometry[0].primitives),a=support(old),b=support(current);
      check(b==expected,"actual TF prior differs from full-grid FK oracle");
      int old_only=0,new_only=0;for(int i:a)old_only+=!b.count(i);for(int i:b)new_only+=!a.count(i);
      coverage<<"{\"pose\":"<<pose<<",\"samples\":"<<a.size()<<",\"primitive\":"<<b.size()
              <<",\"old_only\":"<<old_only<<",\"new_only\":"<<new_only<<"}\n";
      for(int rep=-3;rep<40;++rep) for(int order=0;order<2;++order) {
        const bool primitive=((rep+order+4)%2)==1;Node& n=primitive?current:old;
        const auto begin=std::chrono::steady_clock::now();
        check(n.refreshCurrentBodyPrior(ros::Time::now(),"benchmark",&msg),"benchmark refresh failed");
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        if(rep>=0)timings<<"{\"pose\":"<<pose<<",\"primitive\":"<<primitive<<",\"ms\":"<<ms<<"}\n";
      }
    }
    // Rotated shapes, spherical dilation, outside-map centres, independent oracle.
    current.current_body_prior_link_inflation_radius_.clear();
    for(int kind=0;kind<3;++kind)for(double padding:{0.,.02,.21})for(double shift:{0.,1.0}) {
      VbcPrimitive p;p.kind=static_cast<PrimitiveKind>(kind);p.center=Eigen::Vector3d(shift,.04,.1);
      p.half_size=Eigen::Vector3d(.1,.06,.08);p.radius=.12;p.half_length=.1;
      p.rotation=Eigen::AngleAxisd(.71,Eigen::Vector3d(1,2,3).normalized()).toRotationMatrix();
      current.current_body_prior_inflation_radius_=padding;
      current.clearBootstrapConfidenceLayer();current.markPrimitiveAsKnownClear(p,padding);current.current_body_prior_active_=true;
      check(support(current)==oracle(current,{p}),"edge/shape/padding oracle mismatch");
    }
    current.current_body_prior_enabled_=false;std::string msg;
    check(current.refreshCurrentBodyPrior(ros::Time::now(),"disabled",&msg) && support(current).empty(),"disabled prior");
    current.pnh_.setParam("current_body_prior/geometry_backend","typo");
    check(!current.loadCurrentBodyPriorParams(),"unknown backend accepted");
    current.pnh_.setParam("current_body_prior/geometry_backend","primitive");
    current.pnh_.setParam("current_body_prior/enabled",true);
    current.pnh_.setParam("current_body_prior/primitive_urdf_file",output+"/missing.urdf");
    check(!current.loadCurrentBodyPriorParams(),"missing URDF accepted");
    current.pnh_.setParam("current_body_prior/primitive_urdf_file",root+"/src/arm_description/urdf/Arm_with_self_filter_collision.urdf");
    current.pnh_.setParam("current_body_prior/risk_samples_only",true);
    check(!current.loadCurrentBodyPriorParams(),"implicit risk filter accepted");
    current.pnh_.setParam("current_body_prior/primitive_risk_excluded_links",std::vector<std::string>{"base_link"});
    check(current.loadCurrentBodyPriorParams() && current.current_body_primitive_model_.primitives().size()==24,"explicit risk filter");
    current.pnh_.setParam("current_body_prior/primitive_risk_excluded_links",std::vector<std::string>{"typo_link"});
    check(!current.loadCurrentBodyPriorParams(),"unknown risk exclusion accepted");
    current.pnh_.setParam("current_body_prior/primitive_risk_excluded_links",std::vector<std::string>{"base_link","base_link"});
    check(!current.loadCurrentBodyPriorParams(),"duplicate risk exclusion accepted");
    current.pnh_.setParam("current_body_prior/inflation_radius",-1.);
    check(!current.loadCurrentBodyPriorParams(),"negative margin accepted");
    current.pnh_.setParam("current_body_prior/inflation_radius",std::numeric_limits<double>::quiet_NaN());
    check(!current.loadCurrentBodyPriorParams(),"NaN margin accepted");
    std::cout<<"PASS prior geometry, TF completeness, lock, deactivate, provenance, markers, config; 18 shape-edge variants; 240 timings\n";
  }
};

int main(int argc,char** argv) {
  ros::init(argc,argv,"coldstart_test");
  try { if(argc<3)throw std::runtime_error("ROOT OUTPUT required");ColdStartPrimitiveTest::run(argv[1],argv[2]);return 0; }
  catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}
}
