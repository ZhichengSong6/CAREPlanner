#!/usr/bin/env python3
"""Check the four-case bounded-window regression without altering run data."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch",required=True)
    ap.add_argument("--stamp",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    repo=Path(__file__).resolve().parents[1]
    batch=Path(args.batch)
    summary=json.loads((batch/"qualification_summary.json").read_text())
    metadata=dict(line.split("=",1) for line in (batch/"qualification_metadata.txt").read_text().splitlines() if "=" in line)
    git_short=metadata["git_head"][:8]
    assert {c["case_id"] for c in summary["cases"]} == {
        "phase_e_goal_006","phase_e_goal_024","phase_e_goal_029","phase_e_goal_012"}
    rows=[]
    for c in summary["cases"]:
        cid=c["case_id"]
        run=repo/"outputs/c5_5_vbc_gcdf_regime"/(cid+"_empty_qualification_"+args.stamp+"_"+git_short)/"run"
        status=json.loads((run/"goal_stop_status.json").read_text())
        w=c["benchmark_window"]
        assert w["schema_version"]==2 and w["valid"] and w["start_ros_s"]>0
        assert w["first_sample_ros_s"]==w["start_ros_s"]
        assert c["trace_start_time_s"] >= w["start_ros_s"]
        assert c["trace_end_time_s"] <= w["end_ros_s"]
        assert c["task_success"]==status["goal_reached"]
        assert sum(c[k] for k in ("normal_time_s","repair_time_s","probe_time_s")) <= w["end_ros_s"]-w["start_ros_s"]+1e-6
        assert c["commit_gate_rejection_count"] == c["verification_unsafe_rejection_count"]+c["verification_timeout_count"]
        log=repo/"logs/c5_5_vbc_gcdf_regime"/run.parent.name/"run/controlled.log"
        text=log.read_text(errors="replace")
        events=[]
        for line in text.splitlines():
            if "[TaskReference]" in line:
                events.append(dict(re.findall(r"([a-z_]+)=([^\s]+)",line.split("[TaskReference]",1)[1])))
        dynamic_rejects=text.count("Rejecting dynamically invalid one-shot")
        assert dynamic_rejects==0
        rows.append({"case":cid,"task_success":c["task_success"],"overall_safe":c["overall_safe"],
            "final_position_error_m":c["final_position_error_m"],
            "final_orientation_error_rad":c["final_orientation_error_rad"],
            "watcher_success_elapsed_wall_s":status["success_elapsed_wall_s"],
            "benchmark_wall_elapsed_s":status["elapsed_wall_s"],
            "window_start_ros_s":w["start_ros_s"],"window_end_ros_s":w["end_ros_s"],
            "normal_refresh_requests":text.count("NORMAL ENTRY -> measured-state EE-goal replan requested"),
            "reference_events":events,"dynamics_receiver_rejections":dynamic_rejects,
            "remaining_risks":"task not reached in budget" if not c["task_success"] else "single-run result only"})
    paths=["scripts/evaluate_phase_d_run.py","scripts/wait_for_phase_d_goal.py",
           "scripts/phase_e_evaluation_common.py","scripts/run_phase_e_empty_world_qualification.sh",
           "scripts/run_phase_e_fixes_1_2_regression.sh"]
    base="src/egocentric_arm_planner/"
    paths += [base+"src/"+n+".cpp" for n in ("task_trajectory_generator","receding_horizon_planner","local_sparse_scp_planner")]
    paths += [base+"include/egocentric_arm_planner/"+n+".hpp" for n in
              ("trajectory_dynamics","task_reference_retry","receding_horizon_planner","local_sparse_scp_planner")]
    report={"verification":"PASS", "task_success_count":sum(c["task_success"] for c in rows),
            "overall_safe_count":sum(c["overall_safe"] for c in rows),"cases":rows,
            "source_sha256":{p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths}}
    Path(args.output).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("source_sha256","cases")},indent=2))
    for c in rows:print(c["case"],c["task_success"],c["final_position_error_m"],c["normal_refresh_requests"])


if __name__=="__main__":main()
