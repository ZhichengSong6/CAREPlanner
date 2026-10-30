#!/usr/bin/env python3
"""Compile production tracker callback/ledger offline; replay exp84 up to first change.

Only ROS transport, time, and unrelated planner members are fixtures. No ROS graph
or simulation. Historical data after first policy change is NOT a new rollout.
"""
import bisect
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/phase_e_measured_safe_stall'
SOURCE = ROOT / 'src/egocentric_arm_planner/src/local_sparse_scp_planner.cpp'
HEADER = ROOT / 'src/egocentric_arm_planner/include'
RUN = ROOT / 'outputs/c5_5_vbc_gcdf_regime/case_001_r1_point_scoped_witness_20260922_exp84/run'


def extract(text, signature):
    start = text.index(signature)
    brace = text.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


FIXTURE = r'''
#include "egocentric_arm_planner/safe_frontier_recovery.hpp"
#include <memory>
#include <mutex>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#define ROS_INFO_STREAM(x) do {} while(false)
#define ROS_DEBUG_STREAM_THROTTLE(t,x) do {} while(false)
using V=Eigen::VectorXd;
using Recovery=egocentric_arm_planner::SafeFrontierRecovery;
#ifdef BASELINE
struct Ledger : Recovery { void abort(unsigned long long) {} }; // tail replay has no aborts
#else
using Ledger=Recovery;
#endif
namespace ros {
double clock_s=0.;
struct Time {
 double t=0.;
 Time()=default; explicit Time(double v):t(v) {}
 static Time now() { return Time(clock_s); }
 double toSec() const { return t; }
 Time operator-(const Time& rhs) const { return Time(t-rhs.t); }
};
}
namespace std_msgs {
struct String { std::string data; };
using StringConstPtr=std::shared_ptr<const String>;
}
struct JointState { struct { ros::Time stamp; } header; V q=V::Zero(7); };
struct Guard { struct { unsigned long long progress=0; } active; bool blocked() const { return false; } };
struct LocalSparseSCPPlanner {
 std::mutex mutex_;
 bool safe_frontier_recovery_enabled_=true, repair_mode_=true, probe_mode_=false;
 bool repair_observation_phase_=true, has_joint_state_=true;
 struct { bool active=true; } latest_frontier_;
 JointState latest_joint_state_;
 ros::Time latest_joint_state_received_;
 Ledger safe_frontier_recovery_;
 bool plan_running_=true, waiting_for_cdf_=true, plan_requested_=false;
 std::shared_ptr<int> pending_batch_;
 bool candidate_replacement_pending_=false, candidate_replacement_trigger_pending_=false;
 bool candidate_replacement_enabled_=true;
 long long latest_vbc_obligation_id_=4;
 struct { Guard progress; } final_vbc_no_progress_;
 Guard repair_no_progress_, task_no_progress_;
 bool latest_execution_complete_=false, normal_reference_refresh_pending_=false;
 unsigned long long latest_execution_stamp_ns_=0, plan_sequence_=0;
 unsigned long long last_repair_completed_execution_stamp_ns_=0, last_normal_completed_execution_stamp_ns_=0;
 unsigned long long repair_completion_replan_count_=0, normal_completion_replan_count_=0;
 unsigned long long repair_completion_duplicate_count_=0, normal_completion_duplicate_count_=0;
 std::string plan_request_reason_, event, repair_candidate_failed_key_;
 unsigned long long repair_candidate_failed_epoch_=0, repair_candidate_failed_progress_=0, mode_epoch_=1;
 std::string repairTargetKeyLocked() const { return "obligation:4"; }
 LocalSparseSCPPlanner() { safe_frontier_recovery_.select("obligation:4",1); }
 bool extractMeasuredQ(const JointState& state,V& q) { q=state.q; return q.size()==7 && q.allFinite(); }
 void selectRepairTargetLocked() {} // fixed live O4 ticket throughout replay
 void publishSummary(const std::string& value) { event=value; }
 void requestPlanLocked(const std::string& reason);
 void executionSummaryCallback(const std_msgs::StringConstPtr&);
 void message(const std::string& value) {
   auto msg=std::make_shared<std_msgs::String>(); msg->data=value; executionSummaryCallback(msg);
 }
 void fresh(double t,const V& q) {
   ros::clock_s=t; latest_joint_state_.header.stamp=ros::Time(t);
   latest_joint_state_received_=ros::Time(t); latest_joint_state_.q=q;
 }
 void commit(unsigned long long exec,double span=.0013) {
   safe_frontier_recovery_.remember(exec-1,span);
   if(!safe_frontier_recovery_.certify(exec-1,exec)) throw std::runtime_error("certify");
 }
};
std::string tracker(unsigned long long exec,const std::string& suffix="") {
 return "execution_stamp_ns="+std::to_string(exec)+
   " active=1 complete=0 execution_aborted=0 phase_s=0.1"+suffix;
}
int checks=0;
void check(bool ok,const char* name) { ++checks; if(!ok) throw std::runtime_error(name); }
'''
MAIN = r'''
int main(int argc,char**) {
 std::cout<<std::setprecision(17);
 if(argc>1) {
   LocalSparseSCPPlanner p;
   std::string kind;
   int samples=0, commits=0;
   while(std::cin>>kind) {
     if(kind=="C") {
       unsigned long long raw,exec; double span;
       std::cin>>raw>>exec>>span;
       p.safe_frontier_recovery_.remember(raw,span);
       check(p.safe_frontier_recovery_.certify(raw,exec),"replay certificate"); ++commits;
     } else {
       double t,stamp,received; V q(7); std::cin>>t>>stamp>>received;
       for(int j=0;j<7;++j) std::cin>>q[j];
       std::string msg; std::getline(std::cin,msg);
       p.fresh(t,q); p.latest_joint_state_.header.stamp=ros::Time(stamp);
       p.latest_joint_state_received_=ros::Time(received);
       p.message(msg); ++samples;
       if(!p.event.empty()) {
         std::cout<<"{\"first_trigger_ros_s\":"<<t<<",\"event\":\""<<p.event
           <<"\",\"commits_received\":"<<commits<<",\"tracker_samples\":"<<samples;
#ifndef BASELINE
         std::cout<<",\"window_commits\":"<<p.safe_frontier_recovery_.last_stall_commits
           <<",\"window_duration_s\":"<<p.safe_frontier_recovery_.last_stall_duration
           <<",\"measured_motion_inf_rad\":"<<p.safe_frontier_recovery_.last_stall_motion;
#endif
         std::cout<<"}\n"; return 0;
       }
     }
   }
   std::cout<<"{\"first_trigger_ros_s\":null,\"commits_received\":"<<commits
     <<",\"tracker_samples\":"<<samples<<"}\n"; return 0;
 }
#ifndef BASELINE
 V q=V::Zero(7);
 // Healthy small motion, accumulated within and across execution identities.
 {
   LocalSparseSCPPlanner p;
   for(int i=0;i<120;++i) {
     auto exec=1000000000ULL+i*250000000ULL; p.commit(exec,.05);
     q[0]=i*.0003; p.fresh(1.1+i*.25,q); p.message(tracker(exec));
     check(p.safe_frontier_recovery_.stage()==0,"healthy small accumulated movement");
   }
 }
 q.setZero();
 // Six distinct commits AND one second. Same-execution summaries never add counts.
 {
   LocalSparseSCPPlanner p; p.commit(1000000000ULL);
   for(int i=0;i<100;++i) { p.fresh(1.1+i*.01,q); p.message(tracker(1000000000ULL)); }
   check(p.safe_frontier_recovery_.tiny_commits==1,"same execution count once");
   check(p.safe_frontier_recovery_.stage()==0,"one execution cannot trigger");
 }
 // Observe late motion of the same execution instead of looking only at its first sample.
 {
   LocalSparseSCPPlanner p; p.commit(1000000000ULL);
   p.fresh(1.1,q); p.message(tracker(1000000000ULL));
   q[0]=.002; p.fresh(1.2,q); p.message(tracker(1000000000ULL));
   check(p.safe_frontier_recovery_.tiny_commits==0,"within execution motion clears window");
   q.setZero();
 }
 // Late/foreign/aborted/malformed tracker data cannot provide stall evidence.
 for(int mode=0;mode<13;++mode) {
   LocalSparseSCPPlanner p; auto exec=1000000000ULL; p.commit(exec);
   p.fresh(1.1,q); std::string msg=tracker(exec);
   if(mode==0) msg=tracker(exec+1);
   if(mode==1) p.latest_joint_state_.header.stamp=ros::Time(.8);
   if(mode==2) p.latest_joint_state_received_=ros::Time(.8);
   if(mode==3) msg="execution_stamp_ns=1000000000 active=1 complete=0 execution_aborted=0 phase_s=0.049";
   if(mode==4) msg="execution_stamp_ns=1000000000 active=1 complete=0 execution_aborted=1 phase_s=0.1";
   if(mode==5) msg=tracker(exec," execution_stamp_ns=1000000000");
   if(mode==6) { p.safe_frontier_recovery_.invalidateMotion(); }
   if(mode==7) msg="execution_stamp_ns=1000000000junk active=1 complete=0 execution_aborted=0 phase_s=0.1";
   if(mode==8) msg="execution_stamp_ns=1000000000 active=1 complete=0 execution_aborted=0 phase_s=0.1junk";
   if(mode==9) p.probe_mode_=true;
   if(mode==10) p.repair_mode_=false;
   if(mode==11) p.repair_observation_phase_=false;
   if(mode==12) p.latest_frontier_.active=false;
   p.message(msg); check(p.safe_frontier_recovery_.tiny_commits==0,"inadmissible tracker data");
   if(mode==4) { p.fresh(1.2,q); p.message(tracker(exec));
     check(p.safe_frontier_recovery_.tiny_commits==0,"aborted execution cannot revive"); }
 }
 // Out-of-order samples and duplicate certificates never add evidence.
 {
   Recovery s; s.select("A",1); s.remember(10,.05); check(s.certify(10,11),"initial cert");
   s.remember(12,.05); check(!s.certify(12,11),"duplicate execution certificate");
   s.observe(11,10.,q);
   check(!s.observe(11,9.,q)&&s.tiny_commits==1,"backward clock ignored");
   check(!s.observe(11,10.,q)&&s.tiny_commits==1,"duplicate sample time ignored");
   s.abort(11); check(s.tiny_commits==0,"abort clears partial window");
   check(!s.observe(11,11.,q)&&s.tiny_commits==0,"abort certificate removed");
 }
 // Six fast commits cannot trigger before one second.
 {
   LocalSparseSCPPlanner p;
   for(int i=0;i<6;++i) {
     auto exec=1000000000ULL+i*100000000ULL; p.commit(exec,.05);
     p.fresh(1.1+i*.1,q); p.message(tracker(exec));
   }
   check(p.safe_frontier_recovery_.tiny_commits==6&&p.event.empty(),"minimum duration retained");
 }
 // Genuine segment progress accumulates even when individual steps are small.
 {
   Recovery s; s.select("A",1); s.advance();
   for(int i=0;i<61;++i) {
     q[0]=i*.001; s.remember(100+i*2,.05); check(s.certify(100+i*2,101+i*2),"segment cert");
     s.observe(101+i*2,10.+i*.1,q);
   }
   check(s.segments>=5 || s.stage()==2,"small steps reach finite .01 segments");
   check(s.stage()<=2,"healthy moving segments not tiny stalls"); q.setZero();
 }
 // Synthetic repeated failures reach a visible finite hold, and cannot refund on preemption.
 {
   LocalSparseSCPPlanner p;
   for(int stage=1;stage<=6;++stage) {
     for(int i=0;i<6;++i) {
       double start=stage*2.+i*.25; auto exec=static_cast<unsigned long long>(start*1e9);
       p.commit(exec,.05); p.fresh(start+.1,q); p.message(tracker(exec));
     }
     check(p.safe_frontier_recovery_.stage()==stage,"finite stage advancement");
   }
   check(p.safe_frontier_recovery_.blocked()&&!p.plan_requested_,"exhaustion blocks planning");
   check(p.event=="repair_candidate_replacement_exhausted_hold","actual hold event emitted");
   p.safe_frontier_recovery_.select("child",1); p.safe_frontier_recovery_.select("obligation:4",1);
   check(p.safe_frontier_recovery_.blocked(),"owner return cannot refund");
   check(!p.safe_frontier_recovery_.reserveReplacement(),"exhaustion forbids replacement");
 }
 std::cout<<"{\"checks\":"<<checks<<",\"passed\":true}\n";
#endif
}
'''


def text_rows(name):
    with (RUN / name).open() as stream:
        next(stream)
        for line in stream:
            stamp, text = line.rstrip().split(',', 1)  # field.data contains unquoted JSON commas
            yield int(stamp), dict(token.split('=', 1) for token in text.split() if '=' in token), text


def replay_input():
    start, end = 20_369_000_000, 67_484_000_000
    candidates = [(t, f) for t, f, _ in text_rows('local_planner_summary.csv')
                  if start <= t <= end and f.get('event') == 'candidate_published']
    spans = {}
    used = set()
    for t, fields, _ in text_rows('observation_candidate_identity.csv'):
        if not start <= t <= end:
            continue
        raw = int(fields['raw_candidate_stamp_ns'])
        matches = [(ct, cf) for ct, cf in candidates if abs(ct-t) <= 30_000_000
                   and abs(ct-raw) <= 30_000_000
                   and cf['observation_token'] == fields['observation_token']]
        assert len(matches) == 1, (raw, matches)
        ct, cf = matches[0]
        assert cf['plan_seq'] not in used
        used.add(cf['plan_seq'])
        assert cf['init'] == 'repair_hold' and cf['scp_iter'] == '1'
        assert cf['repair'] == '1' and cf['repair_phase'] == 'observation'
        spans[raw] = float(cf['step_inf'])
    events, eligible_execs = [], set()
    for t, f, _ in text_rows('verification_outcome.csv'):
        if not start <= t <= end:
            continue
        assert f['result'] == 'safe' and f['committed'] == '1'
        raw, exec_id = int(f['raw_candidate_stamp_ns']), int(f['execution_stamp_ns'])
        assert raw in spans
        events.append((t, 0, f'C {raw} {exec_id} {spans[raw]:.17g}'))
        eligible_execs.add(exec_id)
    joints = []
    with (RUN / 'joint_states.csv').open() as stream:
        for row in csv.DictReader(stream):
            joints.append((int(row['%time']), int(row['field.header.stamp']),
                           [float(row[f'field.position{j}']) for j in range(7)]))
    times = [row[0] for row in joints]
    for t, f, msg in text_rows('tracker_summary.csv'):
        if not start <= t <= end or int(f.get('execution_stamp_ns', 0)) not in eligible_execs:
            continue
        assert f['execution_aborted'] == '0'
        index = bisect.bisect_left(times, t)-1 # conservative strictly earlier state at equal receipt time
        assert index >= 0
        received, stamp, q = joints[index]
        fields = [t/1e9, stamp/1e9, received/1e9, *q]
        events.append((t, 1, 'T '+' '.join(format(v, '.17g') for v in fields)+' '+msg))
    events.sort()
    path = OUT / 'exp84_replay_input.txt'
    path.write_text('\n'.join(row[2] for row in events)+'\n')
    return path, {'window_ros_s': [start/1e9, end/1e9], 'matched_candidates': len(spans),
                  'safe_commits': len(eligible_execs), 'raw_span_min_rad': min(spans.values()),
                  'raw_span_max_rad': max(spans.values()),
                  'join': 'unique candidate publication within 30ms of identity receipt and raw stamp, same token',
                  'timing_limit': 'CSV receipt time approximates callback arrival; exact thread order unavailable'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    text = SOURCE.read_text()
    production = '\n'.join(extract(text, name) for name in [
        'void LocalSparseSCPPlanner::executionSummaryCallback(',
        'void LocalSparseSCPPlanner::requestPlanLocked('])
    fixture = OUT / 'production_tracker_fixture.cpp'
    fixture.write_text(FIXTURE+'\n'+production+'\n'+MAIN)
    result = {'production_callback_sha256': hashlib.sha256(production.encode()).hexdigest()}
    for label, includes in [('new', HEADER), ('old', OUT / 'baseline')]:
        exe = OUT / f'tracker_fixture_{label}'
        command = ['g++', '-std=c++14', '-O2', '-I/usr/include/eigen3', '-I'+str(includes),
                   str(fixture), '-o', str(exe)]
        if label == 'old':
            command.insert(1, '-DBASELINE')
        subprocess.run(command, check=True)
        if label == 'new':
            result['regression'] = json.loads(subprocess.check_output([str(exe)], text=True))
    path, metadata = replay_input()
    result['exp84'] = metadata
    for label in ['old', 'new']:
        with path.open() as stream:
            result['exp84'][label] = json.loads(subprocess.check_output(
                [str(OUT / f'tracker_fixture_{label}'), '--replay'], stdin=stream, text=True))
    assert result['exp84']['old']['first_trigger_ros_s'] is None
    assert result['exp84']['new']['event'] == 'repair_safe_stall_frontier_retry'
    result['scope'] = 'Offline first-transition replay only; subsequent recovery/robot motion requires a new simulation.'
    (OUT / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
