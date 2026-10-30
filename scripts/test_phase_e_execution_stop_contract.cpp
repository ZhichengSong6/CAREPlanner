#include "egocentric_arm_planner/execution_stop_state.hpp"
#include "egocentric_arm_planner/fixed_start_clearance.hpp"
#include "egocentric_arm_planner/execution_motion_contract.hpp"
#include <iostream>
#include <stdexcept>
using namespace egocentric_arm_planner;
#define CHECK(x) do{if(!(x))throw std::runtime_error(#x);}while(false)
int main(){try{
  auto zero=Eigen::VectorXd::Zero(7).eval();auto moving=zero;moving[3]=-.16;
  ExecutionStopState s;s.begin(24463000000ULL,35,.654,.446,25.125,"tracking");
  s.observe(25.46,25.46,moving,zero,true,true);
  CHECK(s.token==24463000000ULL && !s.settled && std::string(s.state())=="STOPPING");
  for(int k=0;k<7;++k)s.observe(25.5+k*.01,25.5+k*.01,zero,zero,true,true);
  CHECK(s.settled && s.token==24463000000ULL && std::string(s.state())=="STOPPED_HOLD");
  s.observe(25.57,25.57,zero,zero,false,true);CHECK(!s.settled);
  for(int k=0;k<9;++k)s.observe(25.58+k*.01,25.58,zero,zero,true,true);
  CHECK(!s.settled); // Duplicate state cannot manufacture a settling interval.
  s.observe(25.7,25.7,moving,zero,true,true);CHECK(!s.settled);
  CHECK(fixedStartUnknownBlocked(.009390552626,.02,true));
  CHECK(!fixedStartUnknownBlocked(.03,.02,true));
  CHECK(!fixedStartUnknownBlocked(.015,.02,false)); // legal owned floor is separate
  FixedStartClearanceHold hold;const Eigen::Vector3d p(.3,-.25,.65);
  hold.record(zero,p,.00939,.02,30.);
  CHECK(!hold.observe(30.01,30.01,zero) && hold.armed);
  auto changed=zero;changed[0]=.002;
  CHECK(!hold.observe(29.,30.,changed));CHECK(hold.armed);
  CHECK(hold.observe(30.02,30.02,changed) && !hold.armed);
  hold.record(zero,p,.00939,.02,30.);
  hold.resolvedFree(Eigen::Vector3d(.2,0.,.5));CHECK(hold.armed);
  CHECK(!hold.evidenceRecheck(29.,30.));CHECK(hold.evidenceRecheck(30.1,30.1));
  trajectory_msgs::JointTrajectory t;t.joint_names={"j"};t.points.resize(3);
  for(int k=0;k<3;++k){t.points[k].positions={k==0?0.:k==1?.5:1.};
    t.points[k].velocities={k==1?1.:0.};t.points[k].time_from_start=ros::Duration(k);}
  CHECK(executionMotionViolation(t,{0},{2.},{2.}).empty());
  t.points[1].velocities={2.5};CHECK(executionMotionViolation(t,{0},{2.},{6.})=="reference_velocity_limit");
  std::cout<<"PASS native motion, STOPPING identity/settling, fixed-start hold/requery controls\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
