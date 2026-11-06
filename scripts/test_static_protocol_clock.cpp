#define main production_selector_main
#include "../src/care_confidence_map/src/trajectory_vbc_selector_temporal_cluster_node.cpp"
#undef main
#include <fstream>
#include <iostream>
struct VbcPrimitiveTest {
  static void check(bool value,const std::string& message) {
    if(!value)throw std::runtime_error(message);
  }
  static void run(const std::string& root,const std::string& input) {
    boost::property_tree::ptree c;boost::property_tree::read_json(input,c);
    const auto stamp=c.get<unsigned long long>("trajectory_stamp_ns");
    const auto deadline=c.get<unsigned long long>("deadline_ns");
    TrajectoryVbcTemporalClusterNode n;std::string error;
    check(n.evaluator_.initializeKinematics(root+"/src/arm_description/urdf/Arm.urdf","base_link",&error),error);
    trajectory_msgs::JointTrajectory t;t.header.stamp.fromNSec(stamp);t.header.frame_id=c.get<std::string>("frame");
    for(const auto& item:c.get_child("joint_names"))t.joint_names.push_back(item.second.get_value<std::string>());
    auto v=c.get_child("v").begin(),ts=c.get_child("times_s").begin();
    for(const auto& row:c.get_child("q")) {
      trajectory_msgs::JointTrajectoryPoint p;
      for(const auto& item:row.second)p.positions.push_back(item.second.get_value<double>());
      for(const auto& item:(v++)->second)p.velocities.push_back(item.second.get_value<double>());
      p.time_from_start=ros::Duration((ts++)->second.get_value<double>());t.points.push_back(p);
    }
    auto put=[&](const boost::property_tree::ptree& tree) {
      std::ostringstream out;boost::property_tree::write_json(out,tree,false);n.recovery_contexts_[stamp]=out.str();
    };
    auto at=[&](unsigned long long now){ros::Time time;time.fromNSec(now);ros::Time::setNow(time);bool eligible;Eigen::VectorXd a;return n.recoveryContext(t,&eligible,&a);};
    put(c);
    for(int i=0;i<3;++i) {
      check(!at(stamp-1),"future request certified");
#if CARE_FIXED
      check(n.current_context_state_=="waiting" && n.current_context_reason_=="context_clock_pending","future request did not wait");
#else
      check(n.current_context_state_=="invalid" && n.current_context_reason_=="context_deadline_invalid","historical failure not reproduced");
#endif
    }
    check(at(stamp),"clock catch-up did not validate exact content");
    check(n.current_context_sequence_==c.get<unsigned long long>("request_sequence"),"identity changed");
    check(!at(deadline+1) && n.current_context_reason_=="context_deadline_invalid","expired request accepted");
    auto invalid=c;invalid.put("deadline_ns",stamp+500000001ULL);put(invalid);
    check(!at(stamp-1) && n.current_context_state_=="invalid","future clock concealed invalid deadline");
    invalid=c;invalid.put("trajectory_stamp_ns",stamp-1);put(invalid);
    check(!at(stamp) && n.current_context_reason_=="audit_stamp_mismatch","foreign audit accepted");
    invalid=c;invalid.get_child("q").front().second.front().second.put_value(1.0);put(invalid);
    check(!at(stamp) && n.current_context_reason_=="context_q_mismatch","changed content accepted");
    put(c);t.header.frame_id="foreign";
    check(!at(stamp) && n.current_context_reason_=="context_frame_mismatch","changed frame accepted");
    t.header.frame_id="base_link";n.recovery_contexts_.erase(stamp);
    check(!at(stamp) && n.current_context_state_=="waiting","missing context accepted");
    n.recovery_contexts_[stamp]="";
    check(!at(stamp) && n.current_context_reason_=="conflicting_context","conflicting context accepted");
    std::cout<<"{\"status\":\"PASS\",\"historical_future_ns\":1,\"future_certified\":false,\"checks\":10,\"fixed\":"<<CARE_FIXED<<"}\n";
  }
};
int main(int argc,char** argv) {
  ros::init(argc,argv,"static_protocol_clock_test",ros::init_options::NoSigintHandler);
  try {VbcPrimitiveTest::run(argv[1],argv[2]);return 0;}
  catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
