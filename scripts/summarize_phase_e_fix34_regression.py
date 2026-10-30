#!/usr/bin/env python3
"""Read-only run analysis; copy owned QP snapshots into a new evidence directory."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess


def bound_conflicts(data):
    """Independent necessary-feasibility check, not inferred from solver status.

    max(G_row*x) over x_l<=x<=x_u must be >= h_l for a lower-bound row.
    Ignores all other constraints, so a strict contradiction already rules out
    this assembled LOCAL problem. It says nothing about global task feasibility.
    """
    terms = {}
    for row,col,value in data["G"]["coo"]:
        terms.setdefault(row,[]).append((col,value))
    conflicts = []
    for row,pair in data["selected_pairs"]:
        upper = math.fsum(v*float(data["x_u" if v > 0 else "x_l"][col])
                          for col,v in terms.get(row,[]) if v != 0)
        lower = float(data["h_l"][row])
        # Numerical diagnostic tolerance only; does not modify any QP bound.
        if math.isfinite(upper) and lower-upper > 1e-8:
            conflicts.append(dict(row=row,pair=pair,max_lhs=upper,required_lower=lower,
                                  deficit=lower-upper,original_timestep=data["original_timestep"][pair],
                                  source_type=data["source_type"][pair] if data["source_type"] else 0))
    return sorted(conflicts,key=lambda x:x["deficit"],reverse=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--replay-binary", default="/tmp/care_replay_qp")
    ap.add_argument("--expected-cases", default="008,004,015,002,026,012")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]; batch = Path(args.batch)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=False)
    summary = json.loads((batch/"qualification_summary.json").read_text())
    metadata = dict(line.split("=",1) for line in (batch/"qualification_metadata.txt").read_text().splitlines() if "=" in line)
    rows = []
    assert {c["case_id"][-3:] for c in summary["cases"]} == set(args.expected_cases.split(','))
    for c in summary["cases"]:
        cid = c["case_id"]
        run_id = cid + "_empty_qualification_" + args.stamp + "_" + metadata["git_head"][:8]
        run = repo/"outputs/c5_5_vbc_gcdf_regime"/run_id/"run"
        w = c["benchmark_window"]
        assert w["schema_version"] == 2 and w["valid"] and w["start_ros_s"] > 0
        assert c["trace_start_time_s"] >= w["start_ros_s"] and c["trace_end_time_s"] <= w["end_ros_s"]
        stop = json.loads((run/"goal_stop_status.json").read_text())
        assert c["task_success"] == stop["goal_reached"]
        log = (repo/"logs/c5_5_vbc_gcdf_regime"/run_id/"run/controlled.log").read_text(errors="replace")
        events = []
        for line in log.splitlines():
            if "[TaskReference]" not in line and "status=stalled" not in line: continue
            stamp = re.search(r"\[\d+\.\d+, (\d+\.\d+)\]", line)
            events.append(dict(text=re.sub(r"\x1b\[[0-9;]*m", "", line),
                               within_window=bool(stamp and w["start_ros_s"] <= float(stamp[1]) <= w["end_ros_s"])))
        regime = []
        for row in csv.DictReader((run/"regime_summary.csv").open()):
            # Recorder CSV stores receipt ROS time, then the complete token string.
            stamp = float(row["%time"]) * 1e-9
            if stamp <= w["end_ros_s"]: regime.append(row)
        snapshots = []
        for path in dict.fromkeys(re.findall(r"QP failure snapshot=(/tmp/care_qp_failure_[A-Za-z0-9]+)", log)):
            source = Path(path); target = out/(cid+"_"+source.name+".json")
            shutil.copyfile(source,target)
            data = json.loads(target.read_text())
            r = subprocess.run([args.replay_binary,str(target)], capture_output=True,text=True)
            result = json.loads(r.stdout) if r.stdout.strip().startswith("{") else {"stderr":r.stderr,"stdout":r.stdout}
            snapshots.append(dict(path=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                                  query_stamp_ns=data["query_stamp_ns"], status=data["status"],
                                  selected_rows=len(data["selected_pairs"]),
                                  source_types=sorted(set(data["source_type"])),
                                  independent_box_bound_conflicts=bound_conflicts(data),
                                  replay_pass=r.returncode == 0, replay=result))
        last = next(reversed(regime), {})
        tokens = dict(re.findall(r"([a-z_]+)=([^\s]+)", " ".join(last.values())))
        rows.append(dict(case=cid,task_success=c["task_success"],overall_safe=c["overall_safe"],
                         final_position_error_m=c["final_position_error_m"],
                         final_orientation_error_rad=c["final_orientation_error_rad"],
                         benchmark_window=w,last_regime=tokens.get("state"),
                         stall_outcome=tokens.get("task_stall_outcome"),
                         events=events,snapshots=snapshots))
    paths = ["src/arm_model/src/robot_model.cpp", "src/arm_model/include/arm_model/ik_result.hpp",
             "src/egocentric_arm_planner/scripts/c4_4_verified_regime_manager_node.py"]
    paths += ["src/egocentric_arm_planner/src/"+n+".cpp" for n in
              ("task_trajectory_generator","receding_horizon_planner","local_sparse_scp_planner")]
    paths += ["src/egocentric_arm_planner/include/egocentric_arm_planner/"+n+".hpp" for n in
              ("qp_no_progress","qp_snapshot","task_reference_retry","trajectory_dynamics",
               "planner_types","task_trajectory_generator","receding_horizon_planner","local_sparse_scp_planner")]
    paths += ["src/arm_model/include/arm_model/robot_model.hpp", "scripts/run_phase_e_fixes_3_4_regression.sh"]
    report = dict(task_success_count=sum(r["task_success"] for r in rows),
                  overall_safe_count=sum(r["overall_safe"] for r in rows), cases=rows,
                  snapshot_replays_pass=all(s["replay_pass"] for r in rows for s in r["snapshots"]),
                  source_sha256={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths})
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    for row in rows:
        print(row["case"],row["task_success"],row["overall_safe"],row["final_position_error_m"],row["stall_outcome"])
    print("Snapshot replays PASS:",report["snapshot_replays_pass"])
    if not report["snapshot_replays_pass"]: raise SystemExit(1)


if __name__ == "__main__": main()
