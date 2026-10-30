#!/usr/bin/env python3
"""Audit the four 20 mm R1 runs and certify saved NORMAL QP box conflicts."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BATCH_ID = "r1_margin20_normal_progress_4_20260923"
BATCH = ROOT / "outputs/phase_e_r1_mainline_30" / BATCH_ID
IDS = (0, 5, 18, 24)


def messages(path, start, end):
    with path.open() as stream:
        for row in csv.DictReader(stream):
            try:
                stamp = int(row["%time"])
            except (ValueError, TypeError):
                continue
            if start <= stamp <= end:
                yield stamp, dict(word.split("=", 1) for word in
                                  (row.get("field.data") or "").split() if "=" in word)


def joint_span(path, end):
    samples = []
    with path.open() as stream:
        for row in csv.DictReader(stream):
            stamp = int(row["%time"])
            if end - 10_000_000_000 <= stamp <= end:
                samples.append([float(row[f"field.position{i}"]) for i in range(7)])
    return max((max(q[j] for q in samples) - min(q[j] for q in samples)
                for j in range(7)), default=None)


def qp_certificate(path):
    data = json.loads(path.read_text())
    assert data["status"] == "primal infeasible"
    rows = defaultdict(list)
    for row, col, value in data["G"]["coo"]:
        rows[row].append((col, value))
    conflicts = []
    for meta in data["cdf_rows"]:
        row = meta["matrix_row"]
        maximum = sum(value * (data["x_u"][col] if value > 0 else
                               data["x_l"][col]) for col, value in rows[row])
        gap = data["h_l"][row] - maximum
        if gap > 1e-8:
            conflicts.append(dict(pair=meta["pair"], timestep=meta["timestep"],
                                  point=meta["point"], source=meta["source"],
                                  gap_native_cdf=gap, distance=meta["distance"],
                                  margin=meta["margin"]))
    assert len(conflicts) == data["box_conflicting_cdf_rows"]
    assert conflicts, path
    conflicts.sort(key=lambda row: row["gap_native_cdf"], reverse=True)
    point_groups = {}
    for row in conflicts:
        key = tuple(row["point"])
        group = point_groups.setdefault(key, dict(point=row["point"], rows=0,
                                                   max_gap_native_cdf=0.0,
                                                   earliest_timestep=row["timestep"]))
        group["rows"] += 1
        group["max_gap_native_cdf"] = max(group["max_gap_native_cdf"],
                                           row["gap_native_cdf"])
        group["earliest_timestep"] = min(group["earliest_timestep"],
                                          row["timestep"])
    return dict(file=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                query_stamp_ns=data["query_stamp_ns"], status=data["status"],
                selected_cdf_rows=len(data["cdf_rows"]),
                box_conflicting_cdf_rows=len(conflicts),
                earliest_conflicting_timestep=min(row["timestep"] for row in conflicts),
                worst=conflicts[0], top_five=conflicts[:5],
                conflict_points=sorted(point_groups.values(),
                                       key=lambda row: row["max_gap_native_cdf"],
                                       reverse=True))


def main():
    summary = json.loads((BATCH / "summary.json").read_text())
    assert summary["case_ids"] == [f"phase_e_goal_{i:03}" for i in IDS]
    output = []
    for i, evaluation in zip(IDS, summary["cases"]):
        run_root = ROOT / "outputs/c5_5_vbc_gcdf_regime" / f"phase_e_goal_{i:03}_{BATCH_ID}"
        run = run_root / "run"
        window = evaluation["benchmark_window"]
        start, end = window["start_ros_ns"], window["end_ros_ns"]
        planner = list(messages(run / "local_planner_summary.csv", start, end))
        regime = list(messages(run / "regime_summary.csv", start, end))
        visibility = list(messages(run / "visibility_acquisition_summary.csv", start, end))
        verification = list(messages(run / "verification_outcome.csv", start, end))
        commits = [stamp for stamp, row in verification if row.get("committed") == "1"]
        outcomes = Counter((row.get("result"), row.get("safety_gate"))
                           for _, row in verification)
        snapshots = [qp_certificate(path) for path in sorted(run_root.glob("qp_normal_*.json"))]
        assert len(snapshots) == 2
        output.append(dict(case_id=evaluation["case_id"],
                           task_success=evaluation["task_success"],
                           overall_safe=evaluation["overall_safe"],
                           final_position_error_m=evaluation["final_position_error_m"],
                           final_orientation_error_rad=evaluation["final_orientation_error_rad"],
                           final_goal_within_tolerance=evaluation.get("final_goal_within_tolerance"),
                           time_to_success_s=evaluation.get("time_to_success_s"),
                           commit_count=evaluation["commit_count"],
                           last_commit_gap_ros_s=(end-max(commits))/1e9 if commits else None,
                           last_ten_ros_seconds_joint_span_inf_rad=joint_span(run / "joint_states.csv", end),
                           final_regime=regime[-1][1].get("state") if regime else None,
                           final_regime_reason=regime[-1][1].get("reason") if regime else None,
                           remaining_obligations=int(visibility[-1][1]["remaining_obligation_count"]),
                           normal_measured_stall_events=sum(row.get("event") ==
                               "normal_certified_no_measured_progress" for _, row in planner),
                           qp_failures=Counter(row.get("status") for _, row in planner
                                               if row.get("event") == "scp_failed"),
                           verification_outcomes=[dict(result=result, gate=gate, count=count)
                                                  for (result, gate), count in sorted(outcomes.items())],
                           qp_snapshots=snapshots))
    destination = BATCH / "normal_progress_qp_audit.json"
    destination.write_text(json.dumps(dict(batch_id=BATCH_ID, cases=output),
                                      indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(output=str(destination.relative_to(ROOT)),
                          task_success=sum(row["task_success"] for row in output),
                          overall_safe=sum(row["overall_safe"] for row in output),
                          qp_snapshots=sum(len(row["qp_snapshots"]) for row in output))))


if __name__ == "__main__":
    main()
