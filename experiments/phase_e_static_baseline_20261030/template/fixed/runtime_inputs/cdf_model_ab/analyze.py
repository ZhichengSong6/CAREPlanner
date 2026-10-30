"""Join exact candidate identities and retain direct certificate diagnostics."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,importlib.util,json,math,re,sys
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'devel/lib/python3/dist-packages')]
from analyze_phase_e_observation_events import string_rows
import argparse
_ap=argparse.ArgumentParser();_ap.add_argument('--batch',required=True);_ap.add_argument('--case',required=True);_ap.add_argument('--tag',required=True);_args=_ap.parse_args()
BATCH=_args.batch;CASE=_args.case;TAG=_args.tag

RUN=ROOT/'outputs/c5_5_vbc_gcdf_regime'/('phase_e_goal_'+CASE+'_'+BATCH)/'run'
summary_path=ROOT/'outputs/phase_e_r1_mainline_30'/BATCH/'case_summaries'/('phase_e_goal_'+CASE+'.json')
summary=json.loads(summary_path.read_text());w=summary['benchmark_window']
def rows(name):
    p=RUN/(name+'.csv')
    return string_rows(p,w)
def number(value):
    v=float(value)
    return v if math.isfinite(v) else None
outcomes=rows('verification_outcome');ids=rows('observation_candidate_identity')
obs=rows('visibility_acquisition_summary');planner=rows('local_planner_summary')
by_raw=defaultdict(set)
for r in ids:by_raw[r['raw_candidate_stamp_ns']].add(r['observation_token'])
episodes=[]
for r in obs:
    token=r['active_q_vis_token']
    if not episodes or episodes[-1]['token']!=token:episodes.append(dict(token=token,rows=[]))
    episodes[-1]['rows'].append(r)
episode_report=[]
for e in episodes:
    rr=e['rows'];token=e['token']
    oo=[r for r in outcomes if by_raw.get(r.get('raw_candidate_stamp_ns'))=={token}]
    pp=[r for r in planner if r.get('observation_token')==token]
    episode_report.append(dict(token=token,id=rr[0]['active_obligation_id'],first_ros_s=rr[0]['ros_s'],last_ros_s=rr[-1]['ros_s'],
        first_q_gap_inf=number(rr[0]['active_q_distance_inf']),last_q_gap_inf=number(rr[-1]['active_q_distance_inf']),
        first_seen_fraction=rr[0]['active_seen_fraction'],last_seen_fraction=rr[-1]['active_seen_fraction'],
        token_total_candidates_with_terminal_outcomes=len({r['raw_candidate_stamp_ns'] for r in oo}),token_total_commits=sum(r['committed']=='1' for r in oo),
        dispositions=dict(Counter((r['result']+':'+r['safety_gate']) for r in oo if r['committed']!='1')),
        attempts=sorted({int(r['safe_stall_attempts']) for r in pp})))
def motions(name):
    # rostopic's initial header width does not describe later variable-length
    # JointTrajectory rows. Use the existing full-payload reader, as review.py
    # does, so longer curves are retained and absent padding is not compared.
    spec=importlib.util.spec_from_file_location('saved_motion_reader',ROOT/'outputs/phase_e_case002_gcdf_motion_fix_20261015/replay.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return {stamp:dict(frame=c.header.frame_id,names=list(c.joint_names),points=[
        dict(q=list(p.positions),v=list(p.velocities),a=list(p.accelerations),ns=p.time_from_start.to_nsec())
        for p in c.points]) for stamp,c in module.trajectories(RUN/(name+'.csv')).items()}
audits=motions('audited_candidate_trajectory');commits=motions('committed_trajectory')
bindings=[]
for r in outcomes:
    if r['committed']!='1':continue
    a=r['audited_trajectory_stamp_ns'];c=r['execution_stamp_ns'];assert audits[a]==commits[c],(a,c)
    bindings.append(dict(raw=r['raw_candidate_stamp_ns'],audit=a,execution=c,token=sorted(by_raw[r['raw_candidate_stamp_ns']]),
        repair_nodal_motion=r.get('repair_nodal_motion'),exact_motion_equal=True))
certs=[];handoffs=[];seen=set();log_sources=[];partial_certificates=[]
for p in sorted((OUT/(TAG+'_ros_logs')).rglob('*.log'))+[ROOT/'logs/c5_5_vbc_gcdf_regime'/('phase_e_goal_'+CASE+'_'+BATCH)/'run/controlled.log']:
    if 'rosout' not in p.name and p.name!='controlled.log':continue
    log_sources.append(p)
    for line in p.read_text(errors='replace').splitlines():
        line=re.sub(r'\x1b\[[0-9;]*m','',line)
        if '[VBC_SHELL_CERT] ' in line:
            data=line.split('[VBC_SHELL_CERT] ',1)[1]
            fields=dict(re.findall(r'([A-Za-z0-9_]+)=([^\s]+)',data))
            if not all(k in fields for k in ['trajectory_stamp_ns','request_session','request_sequence','result','stage','proof_reason','begin_ros_ns','end_ros_ns','measured_stamp_ns','fk_before','fk_after','residual_before_m','residual_after_m','reserve_refinement_fk','residual_stage','residual_floor_m','residual_point']):
                partial_certificates.append(dict(source=str(p),fields=fields));continue
            key=('cert',fields['trajectory_stamp_ns'],fields['request_session'],fields['request_sequence'],fields['begin_ros_ns'],fields['end_ros_ns'],fields['result'])
            if key not in seen:certs.append(fields);seen.add(key)
        elif '[SHELL_HANDOFF] ' in line:
            data=line.split('[SHELL_HANDOFF] ',1)[1]
            try:fields=json.loads(data[data.index('{'):])
            except (ValueError,IndexError):continue
            key=('handoff',json.dumps(fields,sort_keys=True))
            if key not in seen:handoffs.append(fields);seen.add(key)
execution=rows('execution_vbc_summary')
assert execution,'missing execution VBC records'
unsafe_execution=[r for r in execution if r.get('has_violation')!='0' or r.get('success')!='1']
requests=rows('candidate_replacement_request');grants=rows('candidate_replacement_grant');joins=[]
for g in grants:
    matches=[r for r in requests if r.get('request_id')==g['request_id'] and r.get('action') in ('reserve','reserve_owner')]
    assert len(matches)==1,(g,matches)
    joins.append(dict(request=matches[0],grant=g))
sources=[summary_path,RUN/'verification_outcome.csv',RUN/'observation_candidate_identity.csv',RUN/'candidate_admission_status.csv',RUN/'local_planner_summary.csv',RUN/'visibility_acquisition_summary.csv',*log_sources]
report=dict(status='PASS' if not unsafe_execution else 'REVIEW_REQUIRED',batch=BATCH,window=w,task_success=summary['task_success'],overall_safe=summary['overall_safe'],
    final_position_error_m=summary.get('final_position_error_m'),final_orientation_error_rad=summary.get('final_orientation_error_rad'),
    total_commits=len(bindings),certified_commit_bindings=bindings,episodes=episode_report,
    execution_vbc_records=len(execution),execution_vbc_unsafe=len(unsafe_execution),unsafe_execution_records=unsafe_execution,last_observation=obs[-1] if obs else None,last_planner=planner[-1] if planner else None,
    outcome_counts=dict(Counter(r['result']+':'+r['safety_gate'] for r in outcomes)),
    terminal_failures=[r for r in outcomes if r['committed']!='1'],
    certificate_diagnostics=certs,partial_certificate_log_records=partial_certificates,shell_handoffs=handoffs,replacement_requests_and_grants=joins,
    admissions=rows('candidate_admission_status'),
    planner_restore_events=[r for r in planner if r.get('event')=='geometric_restore_candidate_published'],
    input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
(OUT/('online_ledger_'+TAG+'.json')).write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(dict(task=report['task_success'],safe=report['overall_safe'],commits=len(bindings),execution_vbc=len(execution),
    episodes=episode_report,certificates=certs,handoffs=handoffs,last_planner=report['last_planner']),indent=2))
