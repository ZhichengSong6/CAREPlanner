#include <egocentric_arm_planner/task_observation_context.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace egocentric_arm_planner;
using boost::property_tree::ptree;

int main(int argc,char** argv) {
  if(argc!=2)return 2;
  ptree fixture;read_json(argv[1],fixture);
  const std::uint64_t origin=19000000000ULL;
  auto time=[](std::uint64_t ns){ros::Time value;value.fromNSec(ns);return value;};
  const auto first=fixture.get<std::string>("first");
  const auto second=fixture.get<std::string>("second");
  TaskObservationContext context;context.update(first,time(origin));
  const auto binding=context.binding;
  int checks=0;
  auto check=[&](bool condition,const char* reason){if(!condition)throw std::runtime_error(reason);++checks;};
  check(context.current(binding,time(origin)),"fresh context rejected");
  check(!context.current(binding,time(origin+800000001)),"expired context accepted");
  check(!context.current(binding,time(origin+100000000)),"expired native epoch reopened by clock recovery");
  context.update(second,time(origin+1000000000));
  check(context.current(context.binding,time(origin+1000000000)),"new task generation rejected");
  check(!context.current(binding,time(origin+1000000000)),"old binding accepted into new task");

  TaskObservationContext reverse;reverse.update(first,time(origin));
  check(reverse.current(binding,time(origin+1000000)),"forward clock rejected");
  check(!reverse.current(binding,time(origin-1)),"backward clock accepted");
  check(!reverse.current(binding,time(origin+2000000)),"backward clock did not close epoch");

  TaskObservationContext active;active.update(first,time(origin));
  TaskPlanningEventReceiver receiver;
  const auto raw=fixture.get<std::string>("control");
  auto event=receiver.parse(raw,active,time(origin),"regime_control");receiver.remember(event);
  check(event.payload.get<std::string>("mode")=="NORMAL","control payload changed");
  bool rejected=false;
  try {receiver.parse(raw,active,time(origin),"regime_control");}
  catch(const std::exception&){rejected=true;}
  check(rejected,"duplicate control replay accepted");
  rejected=false;
  try {TaskPlanningEventReceiver().parse(raw,context,time(origin+1000000000),"regime_control");}
  catch(const std::exception&){rejected=true;}
  check(rejected,"old control accepted into replacement task");
  const auto status=taskPlanningStatus("summary","event=quoted_\"value\"\\line",active,
      "native-test",1,time(origin),"regime:1:NORMAL");
  check(!status.empty(),"native bound status missing");
  ptree wire;std::istringstream stream(status);read_json(stream,wire);
  check(wire.get<std::string>("payload.task_regime")=="regime:1:NORMAL","status regime lost");
  check(wire.get<std::string>("payload.text")=="event=quoted_\"value\"\\line","status escaping changed");
  std::cout<<"NATIVE_TASK_RECOVERY_PASS "<<checks<<"\n";
  std::cout<<status<<"\n";
  return 0;
}
