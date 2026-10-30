#!/usr/bin/env python3
"""Reproduce the exp83 S7/S6 result, terminal overflow, and q0 prior geometry."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "case_001_r1_high_witness_s67_priority_20260922_exp83"
RUN = ROOT / "outputs/c5_5_vbc_gcdf_regime" / RUN_ID / "run"
EXP83 = ROOT / "outputs/phase_e_high_witness_sensor_priority/exp83_analysis.json"
EXP81 = ROOT / "outputs/phase_e_final_vbc_candidate_state_machine/exp81_analysis.json"
CONFIG = ROOT / "src/care_confidence_map/config/confidence_map.yaml"
CASE_FILE = ROOT / "src/egocentric_arm_planner/config/phase_c2_vbc_cases.json"
PLANNER = ROOT / "src/egocentric_arm_planner/src/local_sparse_scp_planner.cpp"
POINT = np.array([.10, -.15, .90], dtype=float)
KV = re.compile(r"(\w+)=((?:\[[^\]]*\])|[^ ]+)")

sys.path.insert(0, str(ROOT / "scripts"))
from diagnose_phase_e_case001_exact_witness import Geometry  # noqa: E402


def fields(line):
    return dict(KV.findall(line.split(",", 1)[1]))


def witness_queries():
    grouped = defaultdict(list)
    path = RUN / "local_witness_diagnostics.csv"
    for line in path.read_text().splitlines()[1:]:
        row = fields(line)
        stamp = int(row.get("query_stamp_ns", 0))
        if stamp >= 11700000000 and row.get("event") == "repair_witness_check":
            grouped[(stamp, int(row["plan_seq"]))].append(row)
    result = []
    for (stamp, plan_seq), rows in sorted(grouped.items()):
        counts = Counter(row.get("reason", "missing") for row in rows)
        overflow = [row for row in rows
                    if row.get("reason") == "witness_capacity_exceeded"]
        fresh = [row for row in rows if row.get("reason") == "fresh"]
        result.append({
            "query_stamp_ns": stamp,
            "plan_seq": plan_seq,
            "reason_counts": dict(counts),
            "stored_fresh_count": len(fresh),
            "overflow": [{
                "timestep": int(row["timestep"]),
                "point": [float(x) for x in row["point"].strip("[]").split(",")],
                "batch_pair": int(row["batch_pair"]),
            } for row in overflow],
        })
    return result


def sensor_traces():
    result = []
    for path in sorted((RUN / "projector_traces").glob("c46_obligation_*.json")):
        row = json.loads(path.read_text())
        hybrid = row["per_sensor_hybrid"]
        attempt = hybrid["attempts"][0]
        result.append({
            "obligation_id": int(row["c4_6_obligation_id"]),
            "points": row["active_set_points_xyz"],
            "selected_sensor_id": int(hybrid["selected_sensor_id"]),
            "selected_joint_mask": hybrid["selected_joint_mask"],
            "attempted_sensor_ids": [int(x["sensor_id"])
                                     for x in hybrid["attempts"]],
            "root_found": bool(attempt["root_found"]),
            "final_score": float(attempt["final_score"]),
            "geometry_accepted": bool(attempt["geometry"]["accepted"]),
            "sensor_priority": hybrid["sensor_priority"],
            "q_vis": row["q_vis"],
        })
    return result


def q0_prior_geometry():
    cases = json.loads(CASE_FILE.read_text())["cases"]
    case = next(row for row in cases if row["case_id"] == "case_001")
    q0 = np.asarray(case["initial_q"], dtype=float)
    geometry = Geometry()
    distances = geometry.distances(q0, POINT.reshape(1, 3))[0]
    index = int(np.argmin(distances))
    primitive = geometry.primitives[index]
    config = yaml.safe_load(CONFIG.read_text())
    inflation = float(config["current_body_prior"]["inflation_radius"])
    physical = float(distances[index])
    return {
        "point": POINT.tolist(),
        "q0": q0.tolist(),
        "configuration_file": str(CONFIG.relative_to(ROOT)),
        "configured_uniform_inflation_m": inflation,
        "closest_link": primitive["link"],
        "closest_collision": primitive["name"],
        "physical_signed_distance_m": physical,
        "signed_distance_to_prior_boundary_m": physical - inflation,
        "minimum_link_inflation_to_cover_center_m": physical,
        "candidate_wrist_link3_inflations_m": {
            value: physical - float(value) for value in ("0.13", "0.15", "0.21")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    exp83 = json.loads(EXP83.read_text())
    exp81 = json.loads(EXP81.read_text())
    source = PLANNER.read_text()
    assert "repair_witness_overflow_ = true;" in source
    assert ("repair_witness_requires_reobserve_ = repair_witness_overflow_ || "
            "!valid_batch || !all_seen;") in source

    result = {
        "qualification": "RECORDED_EXP83_POST_RUN_DIAGNOSIS",
        "run_id": RUN_ID,
        "high_witness_sensor_traces": sensor_traces(),
        "outcome": {
            "task_success": exp83["goal"]["task_success"],
            "safe_commits": exp83["verification"]["safe_commits"],
            "exact_vbc_unsafe": exp83["verification"]["unsafe"],
            "execution_vbc_audits": exp83["execution"]["audits"],
            "execution_vbc_unsafe": exp83["execution"]["unsafe"],
            "final_position_error_m": exp83["goal"]["final_position_error_m"],
            "exp81_final_position_error_m": exp81["goal"]["final_position_error_m"],
            "position_error_improvement_m": (
                exp81["goal"]["final_position_error_m"] -
                exp83["goal"]["final_position_error_m"]),
            "joint_range_rad": exp83["motion"]["range_rad"],
            "last_planner_status": exp83["planner_cumulative"]["last_status"],
        },
        "exact_vbc": exp83["vbc_unsafe"],
        "terminal_witness_queries": witness_queries(),
        "terminal_cause": {
            "capacity": 32,
            "first_overflow_query_stamp_ns": 11800000000,
            "all_stored_witnesses_fresh_after_overflow": True,
            "overflow_latched_until_target_or_mode_reset": True,
            "result": "repair_witness_missing_or_invalid_then_5_of_5_exhaustion",
        },
        "q0_prior_geometry": q0_prior_geometry(),
        "recommendation": [
            "repair bounded witness overflow handling and rerun with the same 0.10 m prior",
            "if the same wrist_link3 point still blocks, compare link-specific 0.13/0.15/0.21 m startup priors offline before one isolated rerun",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
