#!/usr/bin/env python3
"""Frozen before/after liveness replay and actual C++ feedback callback checks."""
import argparse
import bisect
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/c5_5_vbc_gcdf_regime/phase_e_goal_020_r1_executable_view_2_20261011_r5/run'
def strings(path):
    for line in path.read_text().splitlines()[1:]:
        t,_,data=line.partition(',')
        if t.isdigit():yield int(t)/1e9,dict(re.findall(r'(\w+)=([^\s]+)',data))

REPLAY=r'''
#include "egocentric_arm_planner/safe_frontier_recovery.hpp"
#include <iostream>
#include <iomanip>
using namespace egocentric_arm_planner;
int main(){SafeFrontierRecovery s;s.select("obligation:3",1);Eigen::VectorXd goal(7),mask(7),q(7);
  for(int j=0;j<7;++j)std::cin>>goal[j];for(int j=0;j<7;++j)std::cin>>mask[j];
#ifdef NEW_PROGRESS
  s.setObservationGoal("care_obs_v1_3_9f38abecbf5ec2120c9ae92c",goal,mask);
#endif
  int n;std::cin>>n;unsigned long long previous=0;
  for(int i=0;i<n;++i){double t;unsigned long long execution;std::cin>>t>>execution;
    for(int j=0;j<7;++j)std::cin>>q[j];
    if(execution!=previous){s.remember(execution,.05);if(!s.certify(execution,execution))return 3;previous=execution;}
    if(s.observe(execution,t,q)){std::cout<<std::setprecision(17)<<"{\"trigger_ros_s\":"<<t
      <<",\"attempts\":"<<s.attempts<<",\"commits\":"<<s.last_stall_commits<<",\"duration_s\":"<<s.last_stall_duration<<"}\n";return 0;}}
  std::cout<<"{\"trigger_ros_s\":null}\n";}
'''

def build(source,path,includes,definitions=()):
    path.with_suffix('.cpp').write_text(source)
    result=subprocess.run(['g++','-std=c++14','-O0','-I/usr/include/eigen3']+
        ['-I'+str(i) for i in includes]+list(definitions)+[str(path.with_suffix('.cpp')),'-o',str(path)],capture_output=True,text=True)
    path.with_suffix('.build.log').write_text(result.stdout+result.stderr);result.check_returncode()

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--before-root',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    joints=[];joint_times=[]
    for r in csv.DictReader((RUN/'joint_states.csv').open()):
        t=int(r['field.header.stamp'])/1e9
        if 11.930<=t<36.764:
            names={r['field.name'+str(j)]:float(r['field.position'+str(j)]) for j in range(7)}
            joint_times.append(t);joints.append([names[n] for n in ['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']])
    commits={int(r[2]) for r in list(csv.reader((RUN/'committed_trajectory.csv').open()))[1:] if r[0].isdigit()}
    samples=[]
    for t,f in strings(RUN/'tracker_summary.csv'):
        if not 11.930<=t<36.764:continue
        execution=int(f.get('execution_stamp_ns','0'));age=t-execution/1e9
        i=bisect.bisect_right(joint_times,t)-1
        if (execution not in commits or f.get('execution_aborted')!='0' or
            not (f.get('active')=='1' or f.get('complete')=='1') or
            not (float(f.get('phase_s','0'))>=.05 or f.get('complete')=='1') or
            not .05<=age<=2. or i<0 or t-joint_times[i]>.2):continue
        samples.append([t,execution,*joints[i]])
    assert len(samples)>100
    interval=ROOT/'outputs/phase_e_case020_observation_analysis_20261011/full_interval_report.json'
    full=json.loads(interval.read_text());goal=full['active_q_vis'];mask=full['sensor_joint_mask']
    data=' '.join(map(str,goal+mask))+'\n'+str(len(samples))+'\n'+''.join(' '.join(map(str,row))+'\n' for row in samples)
    (out/'replay_input.txt').write_text(data);answers={}
    for name,inc,defs in [('before',args.before_root/'src/egocentric_arm_planner/include',[]),
                          ('after',ROOT/'src/egocentric_arm_planner/include',['-DNEW_PROGRESS'])]:
        binary=out/name;build(REPLAY,binary,[inc],defs)
        answers[name]=json.loads(subprocess.run([str(binary)],input=data,capture_output=True,text=True,check=True).stdout)
    assert answers['after']['trigger_ros_s'] is not None
    assert answers['before']['trigger_ros_s'] is None or answers['after']['trigger_ros_s']<answers['before']['trigger_ros_s']
    cpp=(ROOT/'src/egocentric_arm_planner/src/local_sparse_scp_planner.cpp').read_text()
    start=cpp.index('void LocalSparseSCPPlanner::observationProgressCallback(')
    end=cpp.index('void LocalSparseSCPPlanner::candidateReplacementCallback(',start)
    prefix=r'''
    #include "egocentric_arm_planner/safe_frontier_recovery.hpp"
    #include <map>
    #include <sstream>
    #include <mutex>
    #include <memory>
    #include <stdexcept>
    #include <cassert>
    #include <iostream>
    namespace std_msgs{struct String{std::string data;};using StringConstPtr=std::shared_ptr<const String>;}
    namespace ros{struct Time{static Time now(){return {};}double toSec(){return 80.;}};}
    using namespace egocentric_arm_planner;
    struct LocalSparseSCPPlanner{bool safe_frontier_recovery_enabled_=true,repair_mode_=true,probe_mode_=false,
      repair_observation_phase_=true;long long latest_vbc_obligation_id_=3;std::string latest_observation_token_="care_obs_v1_3";
      std::mutex mutex_;SafeFrontierRecovery safe_frontier_recovery_;
      void observationProgressCallback(const std_msgs::StringConstPtr&);};
    '''
    suffix=r'''
    int main(){LocalSparseSCPPlanner n;auto& s=n.safe_frontier_recovery_;s.select("x",1);
      s.setObservationGoal(n.latest_observation_token_,Eigen::VectorXd::Ones(7),Eigen::VectorXd::Ones(7));
      const std::string base="policy=visibility_acquisition active_query_status=ok active_obligation_id=3 active_point_count=1 active_inside_map_count=1 active_finite_confidence_count=1 active_finite_current_visibility_count=1 active_feedback_token=care_obs_v1_3 active_seen_fraction=0 active_min_confidence=0 active_max_current_visibility=0 ";
      auto call=[&](std::string text){auto p=std::make_shared<std_msgs::String>();p->data=text;n.observationProgressCallback(p);};
      for(auto time:{"79.0","80.1","nan","80junk"}){call(base+"active_query_ros_s="+time);assert(s.feedback_query_time<0.);}
      auto replace=[](std::string a,std::string x,std::string y){a.replace(a.find(x),x.size(),y);return a;};
      for(auto pair:{std::pair<std::string,std::string>{"id=3","id=4"},{"token=care_obs_v1_3","token=care_obs_v1_old"},
          {"status=ok","status=service_exception"},{"inside_map_count=1","inside_map_count=0"},
          {"finite_current_visibility_count=1","finite_current_visibility_count=0"},
          {"fraction=0","fraction=nan"},{"fraction=0","fraction=2"}}){
        call(replace(base,pair.first,pair.second)+"active_query_ros_s=79.9");assert(s.feedback_query_time<0.);}
      call(base+"active_query_ros_s=79.9 active_query_ros_s=79.9");assert(s.feedback_query_time<0.);
      n.probe_mode_=true;call(base+"active_query_ros_s=79.9");assert(s.feedback_query_time<0.);n.probe_mode_=false;
      call(base+"active_query_ros_s=79.9");assert(s.feedback_query_time==79.9);
      call(replace(base,"fraction=0","fraction=.1")+"active_query_ros_s=79.95");assert(s.feedback_progress_pending);
      std::cout<<"PASS actual feedback callback identity, freshness, provenance, malformed and duplicate controls\n";}
    '''
    binary=out/'callback';build(prefix+cpp[start:end]+suffix,binary,[ROOT/'src/egocentric_arm_planner/include'])
    result=subprocess.check_output([str(binary)],text=True);(out/'callback_result.txt').write_text(result)
    report=dict(status='PASS',replay=answers,recorded_tracker_samples_used=len(samples),
      callback_result=result,qualification='Identical recorded measured states and tracker executions up to the first recovery decision. Committed execution stamps are also used as normalized raw ledger keys; certification is established by the original committed CSV. This is a helper-level fixed trace, not a replay of full ROS callback scheduling or a safe alternative trajectory.',
      input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [RUN/'joint_states.csv',RUN/'tracker_summary.csv',RUN/'committed_trajectory.csv',interval]})
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
