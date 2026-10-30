"""Review original score and strict v2 post-completion hold from saved records."""
from pathlib import Path
from collections import Counter
from bisect import bisect_right
import csv,hashlib,importlib.util,json,math,re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'devel/lib/python3/dist-packages')]
from evaluate_phase_d_run import load_fk_helper,fk_T,pose_err,quat_rot
from phase_e_terminal_stop_observation import zero_hold_tail
from urdf_parser_py.urdf import URDF
import argparse
_ap=argparse.ArgumentParser();_ap.add_argument('--batch',required=True);_ap.add_argument('--case',required=True);_ap.add_argument('--tag',required=True);_ap.add_argument('--ros-log-dir');_args=_ap.parse_args()
BATCH=_args.batch;CASE=_args.case;TAG=_args.tag
LOGDIR=Path(_args.ros_log_dir).resolve() if _args.ros_log_dir else OUT/(TAG+'_ros_logs')

RUN=ROOT/'outputs/c5_5_vbc_gcdf_regime'/('phase_e_goal_'+CASE+'_'+BATCH)/'run'
SUMMARY=ROOT/'outputs/phase_e_r1_mainline_30'/BATCH/'case_summaries'/('phase_e_goal_'+CASE+'.json')
s=json.loads(SUMMARY.read_text()); stop=json.loads((RUN/'goal_stop_status.json').read_text()); w=s['benchmark_window']; ob=stop.get('terminal_stop_observation') or {}
inputs={SUMMARY,RUN/'goal_stop_status.json'}
def strings(name):
 p=RUN/(name+'.csv'); inputs.add(p)
 with p.open() as f:return [dict(stamp_ns=int(r[0]),**dict(re.findall(r'(\w+)=([^\s,]+)',','.join(r[1:])))) for r in csv.reader(f) if r and r[0].isdigit()]
outcomes=strings('verification_outcome'); tracker=strings('tracker_summary'); execution=strings('execution_vbc_summary'); regime=strings('regime_summary')
spec=importlib.util.spec_from_file_location('saved_reader',ROOT/'outputs/phase_e_case002_gcdf_motion_fix_20261015/replay.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
audits=m.trajectories(RUN/'audited_candidate_trajectory.csv'); commits=m.trajectories(RUN/'committed_trajectory.csv')
inputs.update([RUN/'audited_candidate_trajectory.csv',RUN/'committed_trajectory.csv',RUN/'joint_states.csv',RUN/'actuator_command.csv'])
def signature(c):return dict(frame=c.header.frame_id,names=list(c.joint_names),points=[dict(q=list(p.positions),v=list(p.velocities),a=list(p.accelerations),ns=p.time_from_start.to_nsec()) for p in c.points])
bindings=[]; issues=[]
for r in outcomes:
 if r.get('committed')!='1':continue
 a=r['audited_trajectory_stamp_ns']; c=r['execution_stamp_ns']; eq=signature(audits[a])==signature(commits[c])
 if not eq:issues.append('audit_commit_payload_mismatch:'+c)
 bindings.append(dict(raw=r['raw_candidate_stamp_ns'],audit=a,execution=c,outcome_ros_ns=r['stamp_ns'],inside_original_score_window=w['start_ros_ns']<=r['stamp_ns']<=w['end_ros_ns'],exact_motion_equal=eq))
assert set(commits)=={b['execution'] for b in bindings}, ('missing_commit_outcomes', sorted(set(commits)-{b['execution'] for b in bindings}))
events=[]; activation={}
for p in sorted(LOGDIR.rglob('rosout.log*'))+[ROOT/'logs/c5_5_vbc_gcdf_regime'/('phase_e_goal_'+CASE+'_'+BATCH)/'run/controlled.log']:
 inputs.add(p)
 for line in p.read_text(errors='replace').splitlines():
  line=re.sub(r'\x1b\[[0-9;]*m','',line)
  if '[NORMAL_TERMINAL]' not in line and 'accepted committed trajectory' not in line:continue
  clock=re.search(r'\[\d+\.\d+,\s*(\d+\.\d+)\]',line)
  leading_clock=re.match(r'^(\d+\.\d+)\s',line)
  ns=round(float(leading_clock[1])*1e9) if leading_clock else (round(float(clock[1])*1e9) if clock else None)
  if '[NORMAL_TERMINAL]' in line:events.append(dict(ros_ns=ns,message=line.split('[NORMAL_TERMINAL]',1)[1].strip()))
  if 'accepted committed trajectory' in line:
   token=re.search(r'execution_stamp_ns=(\d+)',line)
   if token and ns is not None and ns not in activation.setdefault(token[1],[]):activation[token[1]].append(ns)
helper=load_fk_helper(str(ROOT)); robot=URDF.from_xml_file(str(ROOT/'src/arm_description/urdf/Arm.urdf')); chain=helper.find_chain_joints(robot,'base_link','EE_link'); goal=np.array(s['goal_position']); rot=quat_rot(s['goal_orientation_xyzw'])
measured=[]
with (RUN/'joint_states.csv').open() as f:
 for r in csv.DictReader(f):
  t=r.get('field.header.stamp') or ''
  if not t.isdigit() or not w['start_ros_ns']<=int(t):continue
  names=[r['field.name%d'%j] for j in range(7)]; q=[float(r['field.position%d'%j]) for j in range(7)]; v=[float(r['field.velocity%d'%j]) for j in range(7)]
  pe,oe=pose_err(fk_T(helper,chain,dict(zip(names,q))),goal,rot)
  measured.append(dict(stamp_ns=int(t),position_error_m=float(pe),orientation_error_rad=float(oe),speed_inf_rad_s=max(abs(x) for x in v)))
actuator=[]
with (RUN/'actuator_command.csv').open() as f:
 for r in csv.DictReader(f):
  if (r.get('%time') or '').isdigit():actuator.append(dict(stamp_ns=int(r['%time']),speed_inf_rad_s=max(abs(float(r['field.data%d'%j])) for j in range(7))))
physical=None
if ob.get('completed'):
 checks={}
 lo=ob['hold_start_ros_ns']; hi=ob['hold_end_ros_ns']; barrier=ob.get('completion_sample_ros_ns'); token=str(ob['execution_stamp_ns'])
 rr=[x for x in measured if lo<=x['stamp_ns']<=hi]; bb=[b for b in bindings if b['execution']==token]
 checks['v2_protocol']=ob.get('protocol')=='terminal_tail_then_measured_hold_v2'
 checks['post_completion_barrier']=bool(barrier and lo>=barrier)
 checks['continuous_measured_100ms']=bool(rr and rr[0]['stamp_ns']==lo and rr[-1]['stamp_ns']==hi and hi-lo>=100000000 and all(0<b['stamp_ns']-a['stamp_ns']<=50000000 for a,b in zip(rr,rr[1:])))
 checks['pose_and_speed_in_original_limits']=bool(rr and all(x['position_error_m']<=.02 and x['orientation_error_rad']<=.2 and x['speed_inf_rad_s']<=.05 for x in rr))
 checks['exact_certified_commit']=len(bb)==1 and bb[0]['exact_motion_equal']
 checks['static_zero_speed_100ms_tail']=token in commits and zero_hold_tail(commits[token],100000000)
 duration=commits[token].points[-1].time_from_start.to_nsec() if token in commits else 0
 tt=[t for t in tracker if t.get('execution_stamp_ns')==token and t.get('complete')=='1' and t.get('execution_aborted')=='0' and t.get('source') in ('active_trajectory','trajectory_complete_hold') and round(float(t.get('phase_s',0))*1e9)>=duration]
 checks['exact_tracker_completion_sample']=bool(tt and int(tt[0].get('sample_ros_ns',0))==barrier)
 # Coverage uses record clock for topic freshness; producer tracker sample clock is
 # the strict motion-completion boundary and is never replaced by receipt time.
 def coverage(records,predicate):
  ordered=sorted(records,key=lambda x:x['stamp_ns']); ts=[x['stamp_ns'] for x in ordered]
  relevant=[x for x in ordered if lo<=x['stamp_ns']<=hi]
  if any(not predicate(x) for x in relevant):return False
  for x in rr:
   i=bisect_right(ts,x['stamp_ns'])-1
   if i<0 or x['stamp_ns']-ts[i]>100000000 or not predicate(ordered[i]):return False
  return bool(rr)
 checks['execution_vbc_safe_and_fresh']=coverage(execution,lambda x:x.get('success')=='1' and x.get('has_violation')=='0')
 checks['NORMAL_regime_and_fresh']=coverage(regime,lambda x:x.get('state')=='NORMAL')
 checks['tracker_valid_and_fresh']=coverage(tracker,lambda x:x.get('execution_aborted')=='0' and ((x.get('execution_stamp_ns')==token and x.get('source') in ('active_trajectory','trajectory_complete_hold')) or (x.get('execution_stamp_ns')=='0' and x.get('source')=='last_reference_hold')))
 checks['actuator_low_speed_and_fresh']=coverage(actuator,lambda x:math.isfinite(x['speed_inf_rad_s']) and x['speed_inf_rad_s']<=.05)
 checks['frozen_snapshot_evidence_valid']=set(ob.get('evidence',{}))=={'tracker','execution_vbc','regime','actuator'} and all(x['valid'] for x in ob['evidence'].values())
 accepted=activation.get(token,[]); fullend=(accepted[0]+duration) if len(accepted)==1 else None
 checks['activation_log_crosscheck']=bool(fullend and lo>=fullend)
 for name,ok in checks.items():
  if not ok:issues.append(name)
 aa=[x for x in actuator if lo<=x['stamp_ns']<=hi]
 physical=dict(status='PASS' if all(checks.values()) else 'FAIL',execution_stamp_ns=token,binding=bb,checks=checks,completion_sample_ros_ns=barrier,activation_log_ros_ns=accepted,full_curve_end_from_activation_ns=fullend,start_ns=lo,end_ns=hi,samples=len(rr),max_position_error_m=max((x['position_error_m'] for x in rr),default=None),max_orientation_error_rad=max((x['orientation_error_rad'] for x in rr),default=None),max_speed_inf_rad_s=max((x['speed_inf_rad_s'] for x in rr),default=None),max_actuator_speed_inf_rad_s=max((x['speed_inf_rad_s'] for x in aa),default=None),execution_vbc_records_inside=sum(lo<=e['stamp_ns']<=hi for e in execution),tracker_complete_first=tt[0] if tt else None)
else:issues.append('terminal_observation_incomplete:'+str(ob.get('reason')))
end=ob.get('hold_end_ros_ns') or w['end_ros_ns']; full=[e for e in execution if w['start_ros_ns']<=e['stamp_ns']<=end]
unsafe=sum(e.get('has_violation')!='0' or e.get('success')!='1' for e in full)
if unsafe:issues.append('unsafe_execution_vbc_through_hold')
strict=bool(physical and physical['status']=='PASS' and not issues)
report=dict(status='PASS' if strict and s['task_success'] and s['overall_safe'] else 'REVIEW_REQUIRED',new_formal_windows=int(not TAG.startswith('qualification')),task_success=s['task_success'],original_overall_safe=s['overall_safe'],original_score_window=w,original_final_position_error_m=s['final_position_error_m'],original_final_orientation_error_rad=s['final_orientation_error_rad'],goal_watch_elapsed_wall_s=stop['elapsed_wall_s'],goal_watch_success_wall_s=stop.get('success_elapsed_wall_s'),terminal_observation=ob,stopped_hold=physical,strict_formal_stopped_hold_demonstrated=strict,issues=issues,all_observed_certified_commit_bindings=bindings,original_commits=sum(b['inside_original_score_window'] for b in bindings),all_recorded_commits=len(bindings),execution_vbc_through_stop=dict(records=len(full),unsafe=unsafe),terminal_events=events,qualification='Original pose100ms score remains frozen. Strict post-completion hold uses the original remaining 60s budget and is reviewed separately. No success-rate claim.')
(OUT/('formal_review_'+TAG+'.json')).write_text(json.dumps(report,indent=2)+'\n')
(OUT/('formal_input_manifest_'+TAG+'.json')).write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs)},indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('original_score_window','all_observed_certified_commit_bindings','terminal_events')},indent=2))
