#!/usr/bin/env python3
"""Collect bounded two-case evidence; preserve invalid startup and original ZIPs."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/phase_e_observation_progress_fix_20261011'
BATCH=ROOT/'outputs/phase_e_r1_mainline_30/r1_observation_progress_2_20261011_r2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def strings(p):
    rows=[]
    for line in p.read_text().splitlines()[1:]:
        t,_,s=line.partition(',')
        if t.isdigit():rows.append((int(t)/1e9,dict(re.findall(r'(\w+)=([^\s]+)',s))))
    return rows
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
checks=json.loads((OUT/'verification_checks.json').read_text());assert checks['status']=='PASS'
patch=json.loads((OUT/'patch_manifest.json').read_text())
for name,meta in patch.items():assert sha(ROOT/name)==meta['after_sha256'],name
for name,digest in json.loads((OUT/'preserved_frozen_inputs.json').read_text()).items():assert sha(ROOT/name)==digest,name
summary=json.loads((BATCH/'summary.json').read_text());assert summary['case_count']==2
metadata=json.loads((BATCH/'metadata.json').read_text())
for name,digest in metadata['source_hashes'].items():assert sha(ROOT/name)==digest,name
cases=[];archives=[]
for number in ['020','028']:
    cid='phase_e_goal_'+number;rid=cid+'_r1_observation_progress_2_20261011_r2'
    s=json.loads((BATCH/'case_summaries'/(cid+'.json')).read_text());window=s['benchmark_window']
    assert window['valid'] and s['overall_safe'] and s['execution_vbc_unsafe_records']==0
    status=json.loads((BATCH/'logs'/(cid+'_status.json')).read_text());assert all(v==0 for v in status.values())
    cleanup=json.loads((BATCH/'logs'/(cid+'_cleanup.json')).read_text());assert not cleanup['remaining']
    run=ROOT/'outputs/c5_5_vbc_gcdf_regime'/rid/'run';begin,end=window['start_ros_s'],window['end_ros_s']
    local=[(t,f) for t,f in strings(run/'local_planner_summary.csv') if begin<=t<=end]
    stack=[(t,f) for t,f in strings(run/'blocker_stack_summary.csv') if begin<=t<=end]
    acquisition=[(t,f) for t,f in strings(run/'visibility_acquisition_summary.csv') if begin<=t<=end]
    first_dependency=next((dict(ros_s=t,**{k:f.get(k) for k in ['dependency_reason','dependency_pushes','current_target_id','stack']})
        for t,f in stack if int(f.get('dependency_pushes','0'))>0),None)
    switch=[];last_id=None
    for t,f in acquisition:
        oid=f.get('active_obligation_id')
        if oid!=last_id:switch.append(dict(ros_s=t,obligation_id=oid,token=f.get('active_q_vis_token')));last_id=oid
    last=dict(local[-1][1]);last_time=local[-1][0]
    observed_stall_records=[dict(ros_s=t,event=f.get('event'),attempts=f.get('safe_stall_attempts'))
        for t,f in local if f.get('observation_stall_reason')=='observation_no_progress']
    feedback=sum(f.get('active_feedback_token') not in (None,'none') and f.get('active_query_status')=='ok'
        and t-float(f.get('active_query_ros_s','-1'))<=.5 for t,f in acquisition)
    semantic=dict(l.split('=',1) for l in (run/'runtime_semantics.txt').read_text().splitlines() if '=' in l)
    assert float(semantic['vbc_swept_volume_margin_m'])==.02
    assert float(semantic['execution_vbc_swept_volume_margin_m'])==.01
    cases.append(dict(case_id=cid,run_id=rid,task_success=s['task_success'],overall_safe=s['overall_safe'],
        commit_count=s['commit_count'],execution_vbc_records=s['execution_vbc_records'],execution_vbc_unsafe_records=0,
        window_ros_s=[begin,end],first_dependency=first_dependency,observation_target_switches=switch,
        observed_new_stall_reason_records=observed_stall_records,valid_actual_map_feedback_records=feedback,
        last_planner_ros_s=last_time,last_planner=last,runner_evaluator=status,owned_remaining=cleanup['remaining'],
        source_summary_sha256=sha(BATCH/'case_summaries'/(cid+'.json'))))
    original=ROOT/('CAREPlanner_C5_RESULT_'+rid+'.zip')
    dst=OUT/'archives'/'packed_original'/original.name;dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.move(str(original),str(dst))
    with zipfile.ZipFile(dst) as z:assert z.testzip() is None
    enhanced=OUT/'archives'/dst.name;shutil.copy2(dst,enhanced)
    diagnostic=list(run.parent.glob('qp_repair_conflict_*.json'))
    with zipfile.ZipFile(enhanced,'a',compression=zipfile.ZIP_DEFLATED) as z:
        for p in diagnostic:z.write(p,str(Path(rid)/p.name))
    with zipfile.ZipFile(enhanced) as z:assert z.testzip() is None
    archives.append(dict(run_id=rid,original=str(dst.relative_to(ROOT)),original_sha256=sha(dst),
        enhanced=str(enhanced.relative_to(ROOT)),enhanced_sha256=sha(enhanced),qp_failure_snapshots=len(diagnostic)))
invalid='phase_e_goal_020_r1_observation_progress_2_20261011'
p=ROOT/('CAREPlanner_C5_RESULT_'+invalid+'.zip')
if p.exists():
    dst=OUT/'archives'/'invalid_startup'/p.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(p),str(dst))
    with zipfile.ZipFile(dst) as z:assert z.testzip() is None
    archives.append(dict(run_id=invalid,invalid_startup=True,path=str(dst.relative_to(ROOT)),sha256=sha(dst)))
save(OUT/'archive_manifest.json',archives)
report=dict(status='PASS',implementation_complete=True,cdf_model_fixed=False,global_task_repair_claimed=False,
    production_files_changed=[name for name in patch if name.startswith('src/')],patch_manifest=patch,
    checks=checks,cases=cases,
    fixed_attribution=dict(stage1=str((OUT/'stage1_fixed/report.json').relative_to(ROOT)),
        stage2=str((OUT/'stage2_fixed_r2/report.json').relative_to(ROOT))),
    preserved_contract=dict(candidate_vbc_margin_m=.02,execution_vbc_margin_m=.01,tracking_budget_m=.02,
        cdf_base_margin_model_units=.05,task_wall_budget_s=60,scp_iterations=6,recovery_directions=5,
        recovery_segments=6,continuous_fk_budget=256,continuous_max_depth=16,
        model_checkpoint_sha256=metadata['gcdf_sha256']),
    qualifications=[
      'The two code changes have independent fixed-input mechanism evidence; the combined free closed-loop run cannot attribute task success to either change.',
      'CASE020 still fails its goal. It schedules the original obstacle point observation parent to an UNKNOWN blocker child earlier, but the later CDF hard QP and finite replacement budget remain limiting.',
      'CASE028 succeeds and preserves safety gates. This does not establish repair attribution or eliminate the need for original 30-case regression.',
      'New observation-no-progress reason was not recorded in these two closed-loop traces; its mechanism evidence is the frozen measured/tracker replay and controlled tests.',
      'First attempted ROS run never armed because the original candidate admission round trip was not ready; records and invalid ZIP remain separate. Original startup deadline and gate were retained.',
      'The original 30-case batch has not been rerun in this implementation turn. Existing 30-case statistics are preserved.',
      'Safety means existing GCDF/VBC compliance evaluation; this run adds no physical-contact or self-collision qualification.',
    ],source_and_model_frozen_during_valid_batch=True,archives=archives)
save(OUT/'report.json',report)
print(json.dumps(dict(status='PASS',cases=[{k:c[k] for k in ['case_id','task_success','overall_safe','commit_count','execution_vbc_records','first_dependency']} for c in cases]),indent=2))
