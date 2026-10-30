#!/usr/bin/env python3
"""Offline regression of production VBC callback -> per-point QP bounds.

Compile the actual callback/parser/row assembly with ROS transport substituted.
No ROS graph, simulation, learned model, or executed trajectory is involved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import test_phase_e_witness_margin as rows

ROOT = Path(__file__).resolve().parents[1]

FIXTURE = r'''
#include <memory>
#include <mutex>
#include <sstream>
#include <cstdlib>
#include <iomanip>
#include "egocentric_arm_planner/final_vbc_no_progress.hpp"
using RepairNoProgress=egocentric_arm_planner::RepairNoProgress;
using FinalVbcNoProgress=egocentric_arm_planner::FinalVbcNoProgress;
#define ROS_WARN_STREAM(x) do { std::ostringstream sink; sink << x; } while(false)
namespace std_msgs {
struct String { std::string data; };
using StringConstPtr=std::shared_ptr<const String>;
}
namespace ros {
struct Duration { explicit Duration(double) {} };
struct WallDuration { explicit WallDuration(double) {} };
struct Time {
  static Time now() { return Time(); }
  Time operator+(const Duration&) const { return Time(); }
  unsigned long long toNSec() const { return 1; }
  double toSec() const { return 1.; }
};
struct WallTime {
  static WallTime now() { return WallTime(); }
  WallTime operator+(const WallDuration&) const { return WallTime(); }
};
}
struct Publisher {
  int count=0;
  std::string last;
  void publish(const std_msgs::String& msg) { ++count; last=msg.data; }
};
struct Recovery {
  bool reject(unsigned long long) { return false; }
  void certify(unsigned long long,unsigned long long) {}
  bool blocked() const { return false; }
};
class LocalSparseSCPPlanner {
public:
  std::mutex mutex_;
  bool repair_mode_=true, probe_mode_=false, safe_frontier_recovery_enabled_=true;
  bool repair_observation_phase_=true, candidate_replacement_enabled_=false;
  bool candidate_replacement_pending_=false, candidate_replacement_trigger_pending_=false;
  bool candidate_replacement_trigger_direction_advanced_=false;
  std::string candidate_replacement_trigger_reason_="final_vbc_repeat";
  bool plan_running_=true, waiting_for_cdf_=true;
  bool repair_witness_overflow_=false, repair_witness_requires_reobserve_=false;
  unsigned long long mode_epoch_=2, plan_sequence_=1;
  unsigned long long candidate_replacement_trigger_raw_=0;
  unsigned long long candidate_replacement_trigger_count_=0;
  unsigned long long candidate_replacement_epoch_=0;
  long long latest_vbc_obligation_id_=-1;
  long long candidate_replacement_trigger_obligation_id_=-1;
  std::shared_ptr<int> pending_batch_;
  Recovery safe_frontier_recovery_;
  FinalVbcNoProgress final_vbc_no_progress_;
  std::vector<Eigen::Vector3d> latest_vbc_obligation_points_;
  std::vector<Eigen::Vector3d> final_vbc_previous_points_;
  std::vector<RepairWitness> repair_unknown_witnesses_;
  double visibility_obligation_cdf_margin_base_=.05;
  std::atomic<double> visibility_obligation_cdf_margin_effective_{.05};
  double vbc_feedback_cdf_margin_step_=.15, vbc_feedback_cdf_margin_max_=.5;
  ros::Time current_query_stamp_, last_query_ros_time_;
  ros::Time candidate_replacement_trigger_deadline_;
  ros::WallTime candidate_replacement_trigger_wall_deadline_;
  std::string latest_observation_token_="care_obs_v1_test";
  Publisher candidate_replacement_trigger_pub_;
  int vbc_feedback_rejection_count_=0, vbc_feedback_matched_point_count_=0;
  int vbc_feedback_repair_witness_count_=0;
  void requestPlanLocked(const std::string&) {}
  void publishSummary(const std::string&) {}
  void finalVerificationCallback(const std_msgs::StringConstPtr&);
};
'''

MAIN = r'''
int checks=0;
void require(bool ok,const char* name) {
  ++checks; if(!ok) throw std::runtime_error(name);
}
void equal(double a,double b,const char* name) { require(std::abs(a-b)<1e-12,name); }
int main() {
 try {
  LocalSparseSCPPlanner p;
  const Eigen::Vector3d A(.15,0,.95), B(.15,0,.90), C(-.15,0,.70);
  p.latest_vbc_obligation_points_={A,B};
  p.final_vbc_no_progress_.progress.select("target_A_B");
  const std::string a="{\"point\":[0.15,0,0.95]}", b="{\"point\":[0.15,0,0.90]}",
                    c="{\"point\":[-0.15,0,0.70]}";
  unsigned long long sequence=100;
  auto message=[&](const std::string& evidence,bool remember=true) {
    auto msg=std::make_shared<std_msgs::String>(); ++sequence;
    if(remember) {
      // Synthetic measured progress permits more than the existing five
      // no-progress failures. The production retry budget is not bypassed.
      p.final_vbc_no_progress_.progress.observe(Eigen::VectorXd::Constant(7,sequence*.02));
      p.final_vbc_no_progress_.remember(sequence,p.mode_epoch_,p.plan_sequence_);
    }
    msg->data="result=unsafe safety_gate=vbc committed=0 raw_candidate_stamp_ns="+
        std::to_string(sequence)+" audited_trajectory_stamp_ns="+std::to_string(sequence+1000)+
        " vbc_evidence={\"violations\":["+evidence+"]}";
    return msg;
  };
  auto margin=[&](const Eigen::Vector3d& point,int k=1) {
    Batch batch; batch.point_flat={point.x(),point.y(),point.z()}; batch.distance={.06};
    batch.gradient_flat={1,0,0,0,0,0,0}; batch.q_linearization_flat={0,0,0,0,0,0,0};
    batch.original_timestep={k};batch.source_type={Batch::SOURCE_UNKNOWN};
    auto answer=select(batch,p.latest_vbc_obligation_points_,p.repair_unknown_witnesses_,
                       p.visibility_obligation_cdf_margin_effective_.load());
    require(answer.margins.size()==1,"point must have a selected guard");
    equal(answer.lower[0],answer.margins[0]-.06,"actual QP lower bound");
    return answer.margins[0];
  };
  equal(margin(A),.05,"A starts at .05");equal(margin(B),.05,"B starts at .05");
  auto first=message(a+","+a);p.finalVerificationCallback(first);
  equal(margin(A),.20,"same point twice in evidence increments once");
  equal(margin(B),.05,"active sibling stays .05");
  p.finalVerificationCallback(first);
  equal(margin(A),.20,"duplicate outcome cannot increment");
  for(int i=0;i<4;++i) p.finalVerificationCallback(message(a));
  equal(margin(A),.50,"three distinct rejections reach .50 and remain capped");
  equal(margin(B),.05,"five rejections never leak to sibling");
  p.finalVerificationCallback(message(c));
  equal(margin(C),.20,"queued point owns independent ladder");
  equal(margin(A),.50,"queued rejection leaves A alone");
  equal(margin(B),.05,"queued rejection leaves B alone");
  for(int i=0;i<10;++i) p.finalVerificationCallback(message(a));
  equal(margin(A,20),.50,"rejections remain .50 over full horizon");
  for(int i=0;i<5;++i) p.finalVerificationCallback(message(a));
  equal(margin(A),.50,"repeated rejections capped .50");
  equal(margin(B),.05,"unrejected sibling remains .05 at A cap");
  p.finalVerificationCallback(message(b+","+c));
  equal(margin(B),.20,"newly rejected sibling gets first increment only");
  equal(margin(C),.35,"queued point increments independently");
  auto unremembered=message(b,false);p.finalVerificationCallback(unremembered);
  equal(margin(B),.20,"unknown raw identity ignored");
  auto stale=message(b);++p.mode_epoch_;p.finalVerificationCallback(stale);
  equal(margin(B),.20,"old mode outcome ignored");
  --p.mode_epoch_;
  auto safe=message(b);safe->data.replace(7,6,"safe");p.finalVerificationCallback(safe);
  equal(margin(B),.20,"safe result does not advance guard");
  p.latest_vbc_obligation_points_={C};
  p.final_vbc_no_progress_.progress.select("target_C");
  equal(margin(A),.50,"previous active point keeps own guard");
  equal(margin(C),.35,"new active point does not inherit A guard");
  equal(p.visibility_obligation_cdf_margin_effective_.load(),.05,"shared base never raised");
  // Legacy duplicate timestep anchors for one point collapse without changing others.
  LocalSparseSCPPlanner q;
  q.final_vbc_no_progress_.progress.select("ordinary");
  q.repair_unknown_witnesses_={{A,1,.01,false},{A,2,.01,false},{B,1,.05,true}};
  q.final_vbc_no_progress_.remember(900,q.mode_epoch_,1);
  auto collapse=std::make_shared<std_msgs::String>();
  collapse->data="result=unsafe safety_gate=vbc committed=0 raw_candidate_stamp_ns=900 audited_trajectory_stamp_ns=901 vbc_evidence={\"violations\":["+a+"]}";
  q.finalVerificationCallback(collapse);
  require(q.repair_unknown_witnesses_.size()==2,"collapse same point only");
  equal(q.repair_unknown_witnesses_[0].safety_margin,.05,"other witness untouched");
  equal(q.repair_unknown_witnesses_[1].safety_margin,.20,"new VBC point starts from base");
  require(q.repair_unknown_witnesses_[1].vbc_feedback,"stored point-scoped");
  // The first exact rejection raises the point only. The next consecutive
  // rejection containing the same point emits exactly one head-switch trigger.
  LocalSparseSCPPlanner trigger;
  trigger.candidate_replacement_enabled_=true;
  trigger.latest_vbc_obligation_id_=7;
  trigger.latest_vbc_obligation_points_={A};
  trigger.final_vbc_no_progress_.progress.select("trigger_target");
  auto trigger_message=[&](unsigned long long raw) {
    trigger.final_vbc_no_progress_.progress.observe(Eigen::VectorXd::Constant(7,raw*.02));
    trigger.final_vbc_no_progress_.remember(raw,trigger.mode_epoch_,trigger.plan_sequence_);
    auto msg=std::make_shared<std_msgs::String>();
    msg->data="result=unsafe safety_gate=vbc committed=0 raw_candidate_stamp_ns="+
        std::to_string(raw)+" audited_trajectory_stamp_ns="+std::to_string(raw+1000)+
        " vbc_evidence={\"violations\":["+a+"]}";
    return msg;
  };
  trigger.finalVerificationCallback(trigger_message(1001));
  require(trigger.candidate_replacement_trigger_pub_.count==0,
          "first rejection must not switch head");
  trigger.finalVerificationCallback(trigger_message(1002));
  require(trigger.candidate_replacement_trigger_pub_.count==1,
          "second same-point rejection must switch head");
  require(trigger.candidate_replacement_trigger_pending_,"trigger waits for reservation");
  require(trigger.candidate_replacement_trigger_pub_.last.find(
      "reason=final_vbc_repeat")!=std::string::npos,"trigger reason");
  // CASE001 exp79 shape: the current target is A, while its trajectory is
  // rejected twice at C, a point owned by another live obligation. Target
  // continuity, rather than active-point ownership, controls the switch.
  LocalSparseSCPPlanner cross;
  cross.candidate_replacement_enabled_=true;
  cross.latest_vbc_obligation_id_=8;
  cross.latest_vbc_obligation_points_={A};
  cross.final_vbc_no_progress_.progress.select("cross_target");
  auto cross_message=[&](unsigned long long raw) {
    cross.final_vbc_no_progress_.progress.observe(Eigen::VectorXd::Constant(7,raw*.02));
    cross.final_vbc_no_progress_.remember(raw,cross.mode_epoch_,cross.plan_sequence_);
    auto msg=std::make_shared<std_msgs::String>();
    msg->data="result=unsafe safety_gate=vbc committed=0 raw_candidate_stamp_ns="+
        std::to_string(raw)+" audited_trajectory_stamp_ns="+std::to_string(raw+1000)+
        " vbc_evidence={\"violations\":["+c+"]}";
    return msg;
  };
  cross.finalVerificationCallback(cross_message(2001));
  require(cross.candidate_replacement_trigger_pub_.count==0,
          "first cross-obligation rejection must not switch head");
  cross.finalVerificationCallback(cross_message(2002));
  require(cross.candidate_replacement_trigger_pub_.count==1,
          "second same cross-obligation point must switch head");
  const std::string marker="repeated_points=";
  const auto offset=cross.candidate_replacement_trigger_pub_.last.find(marker);
  require(offset!=std::string::npos,
          "cross-obligation trigger carries repeated point field");
  std::istringstream point_stream(
      cross.candidate_replacement_trigger_pub_.last.substr(offset+marker.size()));
  double px=0.,py=0.,pz=0.;char comma1=0,comma2=0;
  point_stream>>px>>comma1>>py>>comma2>>pz;
  require(point_stream && comma1==',' && comma2==',' &&
              std::abs(px-C.x())<1e-12 && std::abs(py-C.y())<1e-12 &&
              std::abs(pz-C.z())<1e-12,
          "cross-obligation trigger preserves exact repeated point value");
  equal(margin(A),.50,"trigger fixture cannot alter original point ladder");
  LocalSparseSCPPlanner capped;
  capped.vbc_feedback_cdf_margin_max_=.30;
  capped.final_vbc_no_progress_.progress.select("capped");
  for (int i=0;i<4;++i) {
    const unsigned long long raw=9000+i;
    capped.final_vbc_no_progress_.remember(raw,capped.mode_epoch_,capped.plan_sequence_);
    auto msg=std::make_shared<std_msgs::String>();
    msg->data="result=unsafe safety_gate=vbc committed=0 raw_candidate_stamp_ns="+
        std::to_string(raw)+" audited_trajectory_stamp_ns="+std::to_string(raw+100)+
        " vbc_evidence={\"violations\":["+a+"]}";
    capped.finalVerificationCallback(msg);
    equal(capped.repair_unknown_witnesses_.at(0).safety_margin,i==0?.20:.30,"new .30 cap");
  }
  std::cout<<"PASS "<<checks<<" callback/parser/QP checks\n";
 } catch(const std::exception& e) {std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = rows.SOURCE.read_text()
    method = source[source.index('LocalSparseSCPPlanner::solveSparseSubproblem('):]
    selection = method[method.index('  struct SelectedRow {'):method.index('  out.selected_cdf_rows =')]
    start = method.index('    h_l[row] =\n        row_data.safety_margin')
    assembly = method[start:method.index('    h_u[row] =', start)]
    helpers = source[source.index('bool parseUnsignedToken('):source.index('}  // namespace')]
    callback = source[source.index('void LocalSparseSCPPlanner::finalVerificationCallback('):
                      source.index('void LocalSparseSCPPlanner::gcdfRejectionFeedbackCallback(')]
    generated = (rows.PREFIX + selection + rows.SUFFIX[:rows.SUFFIX.index('void require(')].replace(
        '/*ASSEMBLY*/', assembly) + FIXTURE + helpers + callback + MAIN)
    cpp = out / 'production_point_feedback.cpp'
    cpp.write_text(generated)
    build = subprocess.run(['g++', '-std=c++14', '-O0', '-I/usr/include/eigen3',
                            '-I' + str(ROOT / 'src/egocentric_arm_planner/include'),
                            str(cpp), '-o', str(out / 'production_point_feedback')],
                           capture_output=True, text=True)
    (out / 'build.log').write_text(build.stdout + build.stderr)
    if build.returncode:
        raise RuntimeError(build.stderr)
    test = subprocess.run([str(out / 'production_point_feedback')], capture_output=True, text=True)
    result = dict(status='PASS' if test.returncode == 0 else 'FAIL',
                  output=test.stdout + test.stderr,
                  source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  scope='Production callback, evidence parser and QP row assembly; synthetic messages/progress, no simulation.')
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    print(result['output'], end='')
    raise SystemExit(test.returncode)


if __name__ == '__main__':
    main()
