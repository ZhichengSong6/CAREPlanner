#!/usr/bin/env python3
"""Replay exp81 O1/O5 generation with the opt-in S7/S6 high-point order."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
VIS = ROOT / "src/care_visibility_cdf/scripts"
sys.path.insert(0, str(VIS))

from candidate_replacement_runtime import (
    high_witness_sensor_preference,
    sensor_mount_group_id,
)
from per_sensor_visibility_runtime import PerSensorVisibilityRuntime


RUN_ID = "case_001_r1_owner_promotion_group_priority_20260922_exp81"
TRACE_DIR = (ROOT / "outputs/c5_5_vbc_gcdf_regime" / RUN_ID /
             "run/projector_traces")
CHECKPOINT = (ROOT /
    "src/care_visibility_cdf/checkpoints/hierarchical9_r1_scratch50k/final.pt")
REFERENCE_URDF = ROOT / "src/arm_description/urdf/Arm.urdf"
SELF_FILTER_URDF = (
    ROOT / "src/arm_description/urdf/Arm_with_self_filter_collision.urdf")
Q_MIN = np.array([-3.14, -2.30, -3.14, -2.65, -3.14, -3.14, -1.20])
Q_MAX = np.array([3.14, 2.30, 3.14, 2.65, 3.14, 3.14, 1.20])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_from_trace(trace):
    config = trace["config"]
    return PerSensorVisibilityRuntime(
        checkpoint_path=str(CHECKPOINT),
        reference_urdf_path=str(REFERENCE_URDF),
        self_filter_urdf_path=str(SELF_FILTER_URDF),
        device=torch.device("cpu"), q_min=Q_MIN, q_max=Q_MAX,
        projection_iters=config["projection_iters"],
        projection_damping=config["projection_damping"],
        projection_epsilon_f=config["projection_epsilon_f"],
        projection_max_step_norm=config["projection_max_step_norm"],
        root_refine_iters=config["root_refine_iters"],
        root_tolerance_f=config["root_tolerance_f"],
        branch_ascent_steps=config["ascent_steps"],
        branch_step_size=config["ascent_step_size"],
        branch_max_step_norm=config["ascent_max_step_norm"],
        branch_fallback_ascent_steps=config["shared_fallback_ascent_steps"],
        max_branch_attempts=4, min_conservative_g=0.0,
        require_primitive_los=True)


def digest(result):
    return {
        "accepted": result["accepted"],
        "selected_sensor_id": result["selected_sensor_id"],
        "selected_rank": result["selected_rank"],
        "attempted_sensor_ids": [row["sensor_id"] for row in result["attempts"]],
        "attempt_root_found": [row["root_found"] for row in result["attempts"]],
        "attempt_final_scores": [row["final_score"] for row in result["attempts"]],
        "attempt_geometry_accepted": [
            row["geometry"]["accepted"] for row in result["attempts"]],
        "sensor_priority": result["sensor_priority"],
        "replacement_group_priority": result["replacement_group_priority"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    rows = []
    sources = [CHECKPOINT, REFERENCE_URDF, SELF_FILTER_URDF,
               Path(__file__).resolve()]
    for obligation_id in (1, 5):
        trace_path = next(TRACE_DIR.glob(
            "c46_obligation_{:03d}_*.json".format(obligation_id)))
        sources.append(trace_path)
        trace = json.loads(trace_path.read_text())
        points = trace["active_set_points_xyz"]
        preferred = high_witness_sensor_preference(points, True, .85)
        assert preferred == [7, 6]
        runtime = runtime_from_trace(trace)
        baseline = runtime.generate(
            points, trace["q_zero"],
            branch_seed_row=trace["q_deadline_nominal"])
        prioritized = runtime.generate(
            points, trace["q_zero"],
            branch_seed_row=trace["q_deadline_nominal"],
            preferred_sensor_ids=preferred,
            preference_reason="all_points_above_z_threshold")
        replacement = runtime.generate(
            points, trace["q_zero"],
            branch_seed_row=trace["q_deadline_nominal"],
            replacement={
                "excluded": [trace["per_sensor_selected_sensor_id"]],
                "tried_mount_groups": [
                    sensor_mount_group_id(
                        trace["per_sensor_selected_sensor_id"])],
                "previous_q": trace["q_vis"],
                "preferred_sensor_ids": preferred,
                "preference_reason": "repeated_exact_vbc_high_witness",
            },
            preferred_sensor_ids=preferred,
            preference_reason="repeated_exact_vbc_high_witness")
        assert baseline["selected_sensor_id"] == trace["per_sensor_selected_sensor_id"]
        assert prioritized["accepted"] and prioritized["selected_sensor_id"] == 7
        assert prioritized["attempts"][0]["sensor_id"] == 7
        assert replacement["accepted"] and replacement["selected_sensor_id"] == 7
        assert replacement["attempts"][0]["sensor_id"] == 7
        rows.append({
            "obligation_id": obligation_id,
            "points": points,
            "all_points_z_ge_0p85": True,
            "recorded_sensor_id": trace["per_sensor_selected_sensor_id"],
            "baseline_replay": digest(baseline),
            "prioritized_initial_generation": digest(prioritized),
            "prioritized_replacement": digest(replacement),
        })

    result = {
        "qualification": "OFFLINE_REAL_R1_PRIORITY_REGRESSION",
        "source_run_id": RUN_ID,
        "priority": {"minimum_z_m": .85, "sensor_ids": [7, 6]},
        "rows": rows,
        "summary": {
            "obligations_checked": len(rows),
            "baseline_replay_matches_recorded": True,
            "initial_generation_selects_s7": True,
            "replacement_selects_s7": True,
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
