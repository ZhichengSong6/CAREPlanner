#!/usr/bin/env python3
"""Aggregate CAREPlanner Phase-D per-case summaries."""
import argparse, csv, glob, json, math, os, statistics
from collections import defaultdict

def fnum(v):
    try:
        x=float(v); return x if math.isfinite(x) else None
    except Exception: return None

def mean(vals):
    a=[float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return statistics.fmean(a) if a else None

def nested(row,*keys):
    cur=row
    for k in keys:
        if not isinstance(cur,dict): return None
        cur=cur.get(k)
    return cur

def reciprocal_hz(ms):
    x=fnum(ms)
    return (1000.0/x) if x is not None and x>0 else None

def summarize(group):
    ok=[r for r in group if r.get("task_success")]
    n=len(group)
    return {"case_count":n,
            "task_success_count":sum(bool(r.get("task_success")) for r in group),
            "task_success_rate":sum(bool(r.get("task_success")) for r in group)/n if n else None,
            "overall_safe_count":sum(bool(r.get("overall_safe")) for r in group),
            "overall_safe_rate":sum(bool(r.get("overall_safe")) for r in group)/n if n else None,
            "mean_time_to_success_s_successes":mean(fnum(r.get("time_to_success_s")) for r in ok),
            "mean_repair_count":mean(fnum(r.get("repair_count")) for r in group),
            "mean_probe_count":mean(fnum(r.get("probe_count")) for r in group),
            "mean_repair_time_s":mean(fnum(r.get("repair_time_s")) for r in group),
            "mean_probe_time_s":mean(fnum(r.get("probe_time_s")) for r in group),
            "mean_tracking_error_max_rad":mean(fnum(nested(r,"tracking_error_inf","max")) for r in group),
            "mean_local_plan_mean_ms":mean(fnum(nested(r,"local_plan_ms","mean")) for r in group),
            "mean_local_plan_median_ms":mean(fnum(nested(r,"local_plan_ms","median")) for r in group),
            "mean_local_plan_p95_ms":mean(fnum(nested(r,"local_plan_ms","p95")) for r in group),
            "mean_local_plan_equivalent_hz_from_mean_ms":mean(fnum(nested(r,"local_plan_equivalent_hz","mean")) for r in group),
            "mean_local_plan_equivalent_hz_from_median_ms":mean(fnum(nested(r,"local_plan_equivalent_hz","median")) for r in group),
            "mean_local_cdf_roundtrip_mean_ms":mean(fnum(nested(r,"local_cdf_roundtrip_mean_ms","mean")) for r in group),
            "mean_local_cdf_roundtrip_p95_ms":mean(fnum(nested(r,"local_cdf_roundtrip_mean_ms","p95")) for r in group),
            "mean_safety_pipeline_mean_ms":mean(fnum(nested(r,"candidate_safety_pipeline_ms","mean")) for r in group),
            "mean_safety_pipeline_p95_ms":mean(fnum(nested(r,"candidate_safety_pipeline_ms","p95")) for r in group),
            "mean_vbc_event_total_eval_ms":mean(fnum(nested(r,"candidate_vbc_event_stage_ms","total_eval_ms","mean")) for r in group),
            "mean_vbc_event_confidence_query_ms":mean(fnum(nested(r,"candidate_vbc_event_stage_ms","confidence_query_ms","mean")) for r in group),
            "mean_vbc_event_body_fk_ms":mean(fnum(nested(r,"candidate_vbc_event_stage_ms","body_fk_ms","mean")) for r in group),
            "mean_vbc_event_sensor_fk_ms":mean(fnum(nested(r,"candidate_vbc_event_stage_ms","sensor_fk_ms","mean")) for r in group),
            "mean_vbc_event_visibility_scan_ms":mean(fnum(nested(r,"candidate_vbc_event_stage_ms","visibility_scan_ms","mean")) for r in group),
            "mean_certified_candidate_compute_mean_ms_estimate":mean(fnum(nested(r,"certified_candidate_compute_ms_estimate","mean")) for r in group),
            "mean_certified_candidate_equivalent_hz_estimate":mean(fnum(nested(r,"certified_candidate_equivalent_hz_estimate","mean")) for r in group)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input-root",required=True); ap.add_argument("--output-json",default=""); ap.add_argument("--output-csv",default="")
    a=ap.parse_args(); rows=[]
    for p in sorted(glob.glob(os.path.join(a.input_root,"*.json"))):
        try:
            with open(p) as f: r=json.load(f)
            if r.get("case_id"): rows.append(r)
        except Exception: pass
    order={"easy":0,"medium":1,"hard":2}; rows.sort(key=lambda r:(order.get(r.get("difficulty"),9),r.get("case_id","")))
    groups=defaultdict(list)
    for r in rows: groups[r.get("difficulty","unknown")].append(r)
    out={"phase":"D.2","benchmark":"CAREPlanner full 12-case system evaluation","case_count":len(rows),
         "overall":summarize(rows),"by_difficulty":{k:summarize(groups[k]) for k in ("easy","medium","hard","unknown") if groups[k]},"cases":rows}
    oj=os.path.abspath(a.output_json or os.path.join(a.input_root,"..","phase_d_12case_summary.json"))
    oc=os.path.abspath(a.output_csv or os.path.join(a.input_root,"..","phase_d_12case_summary.csv")); os.makedirs(os.path.dirname(oj),exist_ok=True)
    with open(oj,"w") as f: json.dump(out,f,indent=2)
    fields=["case_id","difficulty","task_success","overall_safe","time_to_success_s","final_position_error_m","final_orientation_error_rad",
            "final_task_phase_s","repair_count","probe_count","normal_time_s","repair_time_s","probe_time_s","commit_count",
            "commit_gate_rejection_count","candidate_vbc_unsafe_records","execution_vbc_unsafe_records","tracking_error_max_rad",
            "tracking_error_p95_rad","local_plan_mean_ms","local_plan_median_ms","local_plan_p95_ms",
            "local_plan_equivalent_hz_from_mean_ms","local_plan_equivalent_hz_from_median_ms",
            "local_cdf_roundtrip_mean_ms","local_cdf_roundtrip_p95_ms",
            "piqp_final_solve_mean_ms","final_gcdf_mean_ms","exact_vbc_mean_ms","safety_pipeline_mean_ms",
            "vbc_event_total_eval_mean_ms","vbc_event_confidence_query_mean_ms",
            "vbc_event_body_fk_mean_ms","vbc_event_sensor_fk_mean_ms","vbc_event_visibility_scan_mean_ms",
            "certified_candidate_compute_mean_ms_estimate","certified_candidate_equivalent_hz_estimate"]
    with open(oc,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({"case_id":r.get("case_id"),"difficulty":r.get("difficulty"),"task_success":int(bool(r.get("task_success"))),
                        "overall_safe":int(bool(r.get("overall_safe"))),"time_to_success_s":r.get("time_to_success_s"),
                        "final_position_error_m":r.get("final_position_error_m"),"final_orientation_error_rad":r.get("final_orientation_error_rad"),
                        "final_task_phase_s":r.get("final_task_phase_s"),"repair_count":r.get("repair_count"),"probe_count":r.get("probe_count"),
                        "normal_time_s":r.get("normal_time_s"),"repair_time_s":r.get("repair_time_s"),"probe_time_s":r.get("probe_time_s"),
                        "commit_count":r.get("commit_count"),"commit_gate_rejection_count":r.get("commit_gate_rejection_count"),
                        "candidate_vbc_unsafe_records":r.get("candidate_vbc_unsafe_records"),"execution_vbc_unsafe_records":r.get("execution_vbc_unsafe_records"),
                        "tracking_error_max_rad":nested(r,"tracking_error_inf","max"),"tracking_error_p95_rad":nested(r,"tracking_error_inf","p95"),
                        "local_plan_mean_ms":nested(r,"local_plan_ms","mean"),"local_plan_median_ms":nested(r,"local_plan_ms","median"),
                        "local_plan_p95_ms":nested(r,"local_plan_ms","p95"),
                        "local_plan_equivalent_hz_from_mean_ms":nested(r,"local_plan_equivalent_hz","mean"),
                        "local_plan_equivalent_hz_from_median_ms":nested(r,"local_plan_equivalent_hz","median"),
                        "local_cdf_roundtrip_mean_ms":nested(r,"local_cdf_roundtrip_mean_ms","mean"),
                        "local_cdf_roundtrip_p95_ms":nested(r,"local_cdf_roundtrip_mean_ms","p95"),
                        "piqp_final_solve_mean_ms":nested(r,"sparse_piqp_final_solve_ms","mean"),
                        "final_gcdf_mean_ms":nested(r,"final_gcdf_roundtrip_ms","mean"),
                        "exact_vbc_mean_ms":nested(r,"exact_vbc_roundtrip_ms","mean"),
                        "safety_pipeline_mean_ms":nested(r,"candidate_safety_pipeline_ms","mean"),
                        "vbc_event_total_eval_mean_ms":nested(r,"candidate_vbc_event_stage_ms","total_eval_ms","mean"),
                        "vbc_event_confidence_query_mean_ms":nested(r,"candidate_vbc_event_stage_ms","confidence_query_ms","mean"),
                        "vbc_event_body_fk_mean_ms":nested(r,"candidate_vbc_event_stage_ms","body_fk_ms","mean"),
                        "vbc_event_sensor_fk_mean_ms":nested(r,"candidate_vbc_event_stage_ms","sensor_fk_ms","mean"),
                        "vbc_event_visibility_scan_mean_ms":nested(r,"candidate_vbc_event_stage_ms","visibility_scan_ms","mean"),
                        "certified_candidate_compute_mean_ms_estimate":nested(r,"certified_candidate_compute_ms_estimate","mean"),
                        "certified_candidate_equivalent_hz_estimate":nested(r,"certified_candidate_equivalent_hz_estimate","mean")})
    print(json.dumps(out["overall"],indent=2)); print("[PHASE D BENCHMARK JSON]",oj); print("[PHASE D BENCHMARK CSV]",oc)
if __name__=="__main__": main()
