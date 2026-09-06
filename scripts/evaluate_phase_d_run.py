#!/usr/bin/env python3
"""Offline Phase-D evaluator for one CAREPlanner C5.5 run."""
import argparse, csv, importlib.util, json, math, os, re, statistics, sys
from collections import defaultdict
import numpy as np
from urdf_parser_py.urdf import URDF

TOK = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
JOINTS = ["joint1","joint2","joint3","joint4","wrist_joint1","wrist_joint2","wrist_joint3"]

def num(x, d=math.nan):
    try:
        v=float(str(x).replace("ms","")); return v if math.isfinite(v) else d
    except Exception: return d

def integer(x,d=0):
    try: return int(float(x))
    except Exception: return d

def stats(xs):
    a=sorted(x for x in (num(v) for v in xs) if math.isfinite(x))
    if not a: return None
    def q(p):
        z=p*(len(a)-1); i=int(z); j=min(i+1,len(a)-1); w=z-i
        return a[i]*(1-w)+a[j]*w
    return {"min":a[0],"median":statistics.median(a),"mean":statistics.fmean(a),"p95":q(.95),"max":a[-1]}

def hz_from_ms(ms_stats):
    if not isinstance(ms_stats,dict): return None
    out={}
    for k in ("min","median","mean","p95","max"):
        v=num(ms_stats.get(k))
        out[k]=(1000.0/v) if math.isfinite(v) and v>0 else None
    return out

def unique_commit_timing_rows(rows):
    # commit_summary is a high-rate state stream. Timing fields persist after a
    # verdict, so retain only the first row for each completed verification seq.
    by_seq={}
    for r in rows:
        seq=integer(r.get("last_verification_seq"),0)
        if seq>0 and seq not in by_seq:
            by_seq[seq]=r
    return [by_seq[k] for k in sorted(by_seq)]

def tokens(path):
    if not os.path.isfile(path): return []
    out=[]
    with open(path,newline="",errors="replace") as f:
        rd=csv.reader(f); h=next(rd,[])
        if not h: return out
        ti=h.index("%time") if "%time" in h else 0
        di=h.index("field.data") if "field.data" in h else 1
        for r in rd:
            if len(r)<=di: continue
            d=dict(TOK.findall(",".join(r[di:])))
            if d:
                d["_t"]=num(r[ti])/1e9 if len(r)>ti else math.nan; out.append(d)
    return out

def indexed_cols(header,prefix):
    out={}
    for i,h in enumerate(header):
        s=h.replace("field.","").replace("[","").replace("]","")
        m=re.match(r"^([A-Za-z_]+)(\d+)$",s)
        if m and m.group(1)==prefix: out[int(m.group(2))]=i
    return out

def joint_states(path,names):
    if not os.path.isfile(path): return []
    out=[]; need=set(names)
    with open(path,newline="",errors="replace") as f:
        rd=csv.reader(f); h=next(rd,[]); ti=h.index("%time") if "%time" in h else 0
        nc=indexed_cols(h,"name"); pc=indexed_cols(h,"position"); ids=sorted(set(nc)&set(pc))
        if not ids: raise RuntimeError("joint_states.csv has no flattened name*/position* columns")
        for r in rd:
            q={}
            for k in ids:
                if nc[k]>=len(r) or pc[k]>=len(r): continue
                n=r[nc[k]].strip(); v=num(r[pc[k]])
                if n in need and math.isfinite(v): q[n]=v
            if need.issubset(q): out.append((num(r[ti])/1e9,[q[n] for n in names]))
    return out

def load_fk_helper(repo):
    p=os.path.join(repo,"src/egocentric_arm_planner/scripts/compute_ee_workspace_bounds.py")
    spec=importlib.util.spec_from_file_location("care_fk_helper",p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

def quat_rot(q):
    x,y,z,w=map(float,q); n=math.sqrt(x*x+y*y+z*z+w*w); x,y,z,w=x/n,y/n,z/n,w/n
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def fk_T(helper,chain,qmap):
    T=np.eye(4)
    for j in chain: T=T@helper.get_joint_origin_transform(j)@helper.joint_motion_transform(j,qmap.get(j.name,0.0))
    return T

def pose_err(T,pg,Rg):
    pe=float(np.linalg.norm(T[:3,3]-pg)); c=(float(np.trace(Rg.T@T[:3,:3]))-1)/2
    return pe,math.acos(max(-1,min(1,c)))

def goal_eval(samples,helper,chain,names,case,ptol,rtol,hold):
    if not samples: return {"joint_state_samples":0,"task_success":False,"time_to_success_s":None}
    pg=np.array(case["goal_position"]); Rg=quat_rot(case["goal_orientation"]); tr=[]
    for t,q in samples:
        pe,re_=pose_err(fk_T(helper,chain,dict(zip(names,q))),pg,Rg); tr.append((t,pe,re_,pe<=ptol and re_<=rtol))
    start=None; success=None
    for i,(t,_,_,ok) in enumerate(tr):
        if ok:
            if start is None: start=i
            if t-tr[start][0]>=hold: success=tr[start][0]; break
        else: start=None
    return {"joint_state_samples":len(tr),"task_success":success is not None,
            "time_to_success_s":None if success is None else success-tr[0][0],
            "final_goal_within_tolerance":tr[-1][3],"final_position_error_m":tr[-1][1],
            "final_orientation_error_rad":tr[-1][2],"best_position_error_m":min(x[1] for x in tr),
            "best_orientation_error_rad":min(x[2] for x in tr),"trace_start_time_s":tr[0][0],
            "trace_end_time_s":tr[-1][0],"trace_duration_s":tr[-1][0]-tr[0][0]}

def regime_times(rows,end_t):
    d=defaultdict(float)
    for i,r in enumerate(rows):
        t0=num(r.get("_t")); t1=num(rows[i+1].get("_t")) if i+1<len(rows) else end_t
        if math.isfinite(t0) and t1 is not None and math.isfinite(t1) and t1>=t0: d[r.get("state","UNKNOWN")]+=t1-t0
    return {"normal_time_s":d["NORMAL"],"repair_time_s":d["REPAIR"],
            "probe_time_s":d["PROBE_NORMAL"]+d["PROBE"]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run-dir",required=True); ap.add_argument("--case-id",required=True)
    ap.add_argument("--repo",default=os.getcwd()); ap.add_argument("--cases-json",default="src/egocentric_arm_planner/config/phase_c2_vbc_cases.json")
    ap.add_argument("--urdf",default="src/arm_description/urdf/Arm.urdf"); ap.add_argument("--position-tolerance-m",type=float,default=.02)
    ap.add_argument("--orientation-tolerance-rad",type=float,default=.20); ap.add_argument("--success-hold-s",type=float,default=.10)
    ap.add_argument("--method",default="careplanner_full"); ap.add_argument("--trial-id",default="trial_00"); ap.add_argument("--output-json",default="")
    a=ap.parse_args(); repo=os.path.abspath(a.repo); run=os.path.abspath(a.run_dir)
    with open(os.path.join(repo,a.cases_json)) as f: db=json.load(f)
    case=next((c for c in db["cases"] if c["case_id"]==a.case_id),None)
    if case is None: raise KeyError(a.case_id)
    helper=load_fk_helper(repo); robot=URDF.from_xml_file(os.path.join(repo,a.urdf)); chain=helper.find_chain_joints(robot,"base_link","EE_link")
    js=joint_states(os.path.join(run,"joint_states.csv"),JOINTS)
    goal=goal_eval(js,helper,chain,JOINTS,case,a.position_tolerance_m,a.orientation_tolerance_rad,a.success_hold_s)
    reg=tokens(os.path.join(run,"regime_summary.csv")); com=tokens(os.path.join(run,"commit_summary.csv")); exe=tokens(os.path.join(run,"execution_vbc_summary.csv"))
    cand=tokens(os.path.join(run,"candidate_vbc_summary.csv")); trk=tokens(os.path.join(run,"tracker_summary.csv")); loc=tokens(os.path.join(run,"local_planner_summary.csv"))
    acq=tokens(os.path.join(run,"visibility_acquisition_summary.csv")); prog=tokens(os.path.join(run,"nominal_progress_summary.csv"))
    lr=reg[-1] if reg else {}; lc=com[-1] if com else {}; lp=prog[-1] if prog else {}
    commits=integer(lc.get("commit_count")); gs=integer(lc.get("final_gcdf_safe_count")); gu=integer(lc.get("final_gcdf_unsafe_count")); gt=integer(lc.get("final_gcdf_timeout_count"))
    vs=integer(lc.get("verification_safe_count")); vu=integer(lc.get("verification_unsafe_count")); vt=integer(lc.get("verification_timeout_count"))
    er=[r for r in exe if r.get("trajectory_source") in ("predicted","committed")]; eu=sum(r.get("has_violation")=="1" for r in er)
    gcdf_ok=commits>0 and integer(lc.get("final_gcdf_enabled"))==1 and gs>=commits; vbc_ok=commits>0 and vs>=commits; exe_ok=len(er)>0 and eu==0
    rem=[integer(r.get("remaining_obligation_count"),-1) for r in acq]; rem=[x for x in rem if x>=0]

    # Phase-D timing: distinguish optimizer latency from post-plan certification.
    candidate_plans=[r for r in loc if r.get("event")=="candidate_published"]
    timing_commit=unique_commit_timing_rows(com)
    local_plan_ms=stats(r.get("total_plan_ms") for r in candidate_plans)
    local_cdf_roundtrip_mean_ms=stats(
        r.get("cdf_roundtrip_mean_ms") for r in candidate_plans)
    local_cdf_roundtrip_max_ms=stats(
        r.get("cdf_roundtrip_max_ms") for r in candidate_plans)
    piqp_final_ms=stats(r.get("solve_ms") for r in candidate_plans)
    raw_to_safety_ms=stats(r.get("raw_to_safety_dispatch_ms") for r in timing_commit)
    final_gcdf_ms=stats(r.get("final_gcdf_roundtrip_ms") for r in timing_commit)
    exact_vbc_ms=stats(r.get("exact_vbc_roundtrip_ms") for r in timing_commit)
    safety_pipeline_ms=stats(r.get("candidate_total_safety_pipeline_ms") for r in timing_commit)

    predicted_vbc_rows=[
        r for r in cand if r.get("trajectory_source")=="predicted"]
    event_vbc_rows=[
        r for r in predicted_vbc_rows
        if r.get("evaluation_trigger")=="predicted_callback"]
    timer_vbc_rows=[
        r for r in predicted_vbc_rows
        if r.get("evaluation_trigger")=="timer"]
    vbc_stage_keys=(
        "trajectory_convert_ms","body_fk_ms","swept_voxel_build_ms",
        "confidence_query_ms","candidate_filter_ms","sensor_fk_ms",
        "visibility_scan_ms","cluster_ms","total_eval_ms")
    vbc_event_stage_ms={
        k:stats(r.get(k) for r in event_vbc_rows) for k in vbc_stage_keys}
    vbc_all_predicted_stage_ms={
        k:stats(r.get(k) for r in predicted_vbc_rows) for k in vbc_stage_keys}
    predicted_periodic_skip_max=max(
        [integer(r.get("predicted_periodic_skip_count"),0)
         for r in predicted_vbc_rows] or [0])
    predicted_periodic_refresh_max=max(
        [integer(r.get("predicted_periodic_refresh_count"),0)
         for r in predicted_vbc_rows] or [0])

    # This combines independently summarized stages, so label it an estimate
    # rather than pretending that the samples are candidate-by-candidate paired.
    certified_compute_est=None
    if isinstance(local_plan_ms,dict) and isinstance(safety_pipeline_ms,dict):
        certified_compute_est={}
        for k in ("min","median","mean","p95","max"):
            x=num(local_plan_ms.get(k)); y=num(safety_pipeline_ms.get(k))
            certified_compute_est[k]=(x+y) if math.isfinite(x) and math.isfinite(y) else None

    legacy_phase=num(lp.get("phase_s"),None)
    legacy_wall=num(lp.get("wall_elapsed_s"),None)
    legacy_master=num(lp.get("master_duration_s"),None)
    legacy_updates=integer(lp.get("update_count"),0)
    legacy_frozen=integer(lp.get("frozen_count"),0)
    legacy_stale=bool(
        goal.get("task_success") and
        legacy_phase is not None and
        legacy_master is not None and
        legacy_master > 1e-6 and
        legacy_phase < 0.95 * legacy_master)
    # C5.43/Phase-E authoritative task completion is measured-state FK to the
    # requested EE goal. The old nominal-progress gate tracks a legacy master
    # trajectory and may freeze after active-sensing/measured-state replans; do
    # not report that stale projection as the real final task phase.
    authoritative_phase=None if legacy_stale else legacy_phase

    out={"phase":"D.1","method":a.method,"trial_id":a.trial_id,"case_id":a.case_id,"difficulty":case.get("difficulty_bin"),
         "goal_position":case["goal_position"],"goal_orientation_xyzw":case["goal_orientation"],
         "success_thresholds":{"position_tolerance_m":a.position_tolerance_m,"orientation_tolerance_rad":a.orientation_tolerance_rad,"required_hold_s":a.success_hold_s},**goal,
         "task_progress_authority":"measured_fk_goal",
         "final_task_phase_s":authoritative_phase,
         "final_wall_elapsed_s":legacy_wall,
         "legacy_nominal_progress_phase_s":legacy_phase,
         "legacy_nominal_progress_master_duration_s":legacy_master,
         "legacy_nominal_progress_update_count":legacy_updates,
         "legacy_nominal_progress_frozen_count":legacy_frozen,
         "legacy_nominal_progress_stale":legacy_stale,
         "overall_safe":gcdf_ok and vbc_ok and exe_ok,"gcdf_commit_certified":gcdf_ok,"vbc_commit_certified":vbc_ok,"execution_vbc_safe":exe_ok,
         "commit_count":commits,"final_gcdf_safe_count":gs,"final_gcdf_unsafe_rejection_count":gu,"final_gcdf_timeout_count":gt,
         "verification_safe_count":vs,"verification_unsafe_rejection_count":vu,"verification_timeout_count":vt,
         "execution_vbc_records":len(er),"execution_vbc_unsafe_records":eu,
         "candidate_vbc_records":sum(r.get("trajectory_source")=="predicted" for r in cand),
         "candidate_vbc_unsafe_records":sum(r.get("trajectory_source")=="predicted" and r.get("has_violation")=="1" for r in cand),
         "commit_gate_rejection_count":gu+gt+vu+vt,"repair_count":integer(lr.get("repair_entry_count")),"probe_count":integer(lr.get("probe_entry_count")),
         **regime_times(reg,goal.get("trace_end_time_s")),"max_remaining_obligation_count":max(rem) if rem else 0,
         "obligation_clear_events":sum(x>0 and y==0 for x,y in zip(rem,rem[1:])),
         "tracking_error_inf":stats(r.get("tracking_error_inf") for r in trk),
         "timing_sample_counts":{"candidate_plans":len(candidate_plans),"completed_verifications":len(timing_commit)},
         "local_plan_ms":local_plan_ms,
         "local_plan_equivalent_hz":hz_from_ms(local_plan_ms),
         "local_cdf_roundtrip_mean_ms":local_cdf_roundtrip_mean_ms,
         "local_cdf_roundtrip_max_ms":local_cdf_roundtrip_max_ms,
         "sparse_piqp_final_solve_ms":piqp_final_ms,
         "sparse_piqp_final_solve_equivalent_hz":hz_from_ms(piqp_final_ms),
         "raw_to_safety_dispatch_ms":raw_to_safety_ms,
         "final_gcdf_roundtrip_ms":final_gcdf_ms,
         "exact_vbc_roundtrip_ms":exact_vbc_ms,
         "candidate_safety_pipeline_ms":safety_pipeline_ms,
         "candidate_safety_pipeline_equivalent_hz":hz_from_ms(safety_pipeline_ms),
         "candidate_vbc_evaluation_counts":{
             "predicted_total":len(predicted_vbc_rows),
             "predicted_event_driven":len(event_vbc_rows),
             "predicted_timer":len(timer_vbc_rows),
             "predicted_periodic_skip_count_max":predicted_periodic_skip_max,
             "predicted_periodic_refresh_count_max":predicted_periodic_refresh_max},
         "safety_admission_counts":{
             "raw_immediate_dispatch_count":integer(
                 lc.get("raw_immediate_dispatch_count"),0),
             "raw_busy_buffer_count":integer(
                 lc.get("raw_busy_buffer_count"),0),
             "post_cert_pending_dispatch_count":integer(
                 lc.get("post_cert_pending_dispatch_count"),0),
             "selector_fallback_dispatch_count":integer(
                 lc.get("selector_fallback_dispatch_count"),0),
             "pending_replace_count":integer(
                 lc.get("pending_replace_count"),0)},
         "candidate_vbc_event_stage_ms":vbc_event_stage_ms,
         "candidate_vbc_all_predicted_stage_ms":vbc_all_predicted_stage_ms,
         "certified_candidate_compute_ms_estimate":certified_compute_est,
         "certified_candidate_equivalent_hz_estimate":hz_from_ms(certified_compute_est)}
    dst=a.output_json or os.path.join(run,"phase_d_run_summary.json"); os.makedirs(os.path.dirname(os.path.abspath(dst)),exist_ok=True)
    with open(dst,"w") as f: json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2)); print("[PHASE D RUN SUMMARY]",dst)
if __name__=="__main__": main()
