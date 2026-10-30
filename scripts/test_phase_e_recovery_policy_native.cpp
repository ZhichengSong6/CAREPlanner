#include <egocentric_arm_planner/task_recovery_policy.hpp>
#include <iostream>
using namespace egocentric_arm_planner;
using boost::property_tree::ptree;
int main(int argc,char** argv) {
  if(argc!=2)return 2;
  ptree f;read_json(argv[1],f);int checks=0;
  const auto binding=f.get<std::string>("binding");const auto& good=f.get_child("good");
  auto check=[&](bool ok,const char* why){if(!ok)throw std::runtime_error(why);++checks;};
  auto p=TaskRecoveryPolicy::parse(good,binding);
  check(p.present,"policy missing");check(p.sha==f.get<std::string>("policy_sha256"),"cross-language digest mismatch");
  check(p.target_sha==f.get<std::string>("target_sha256"),"target digest mismatch");
  Eigen::VectorXd q=Eigen::VectorXd::Zero(7);const uint64_t now=19000000000ULL;
  check(p.current(binding,p.regime,p.target_sha,q,now),"fresh policy rejected");
  check(!p.current(binding,p.regime,p.target_sha,q,p.until),"expired policy accepted");
  check(!p.current(binding,p.regime,p.target_sha,q,now-1),"future measurement accepted");
  check(!p.current("other",p.regime,p.target_sha,q,now),"foreign task accepted");
  check(!p.current(binding,"other:1:REPAIR",p.target_sha,q,now),"foreign regime accepted");
  check(!p.current(binding,p.regime,std::string(64,'a'),q,now),"unacknowledged target accepted");
  q[0]=.020001;check(!p.current(binding,p.regime,p.target_sha,q,now),"stale evaluation pose accepted");
  auto rejected=[&](const ptree& value,const std::string& task) {
    try {TaskRecoveryPolicy::parse(value,task);return false;}catch(const std::exception&){return true;}
  };
  check(rejected(good,"other"),"cross-task content accepted");
  auto bad=good;bad.put("selection.policy_content",good.get<std::string>("selection.policy_content")+" ");
  check(rejected(bad,binding),"mutated bytes accepted");
  bad=good;bad.put("target.sensor_id",3);check(rejected(bad,binding),"mixed target accepted");
  bad=good;bad.put("selection.target_sha256",std::string(64,'0'));check(rejected(bad,binding),"forged target hash accepted");
  auto bare=good;bare.erase("selection");check(!TaskRecoveryPolicy::parse(bare,binding).present,"legacy bootstrap supplied a policy");
  auto control=f.get_child("control");
  check(taskControlRecoveryTarget(control)==p.target_sha,"C4 target acknowledgement lost");
  control.put("recovery_target_sha256","bad");bool invalid=false;
  try {taskControlRecoveryTarget(control);}catch(const std::exception&){invalid=true;}
  check(invalid,"malformed acknowledgement accepted");
  std::cout<<"NATIVE_RECOVERY_POLICY_PASS "<<checks<<"\n";
}
