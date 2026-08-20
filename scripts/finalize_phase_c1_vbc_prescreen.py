#!/usr/bin/env python3
"""Finalize Phase-C1 cases from authoritative projector traces.

During Phase C1 the waypoint node intentionally keeps the latest target and
trajectory state. It can therefore briefly regenerate an old target when a new
nominal trajectory arrives, and it continuously republishes q_vis. The raw
prescreen topics are sufficient for target/sweep discovery, but projector
outputs must be paired explicitly before they are used for benchmark selection.

This script matches every VBC case to exactly one projector trace using:
  1) selected target xyz,
  2) nominal sweep time, and
  3) q_nom at the visibility deadline.
It then overwrites q_zero/q_vis and all projector diagnostics with the matched
trace, recomputes d_q, and rebuilds the easy/medium/hard selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def choose_cases(cases: List[Dict[str, object]], n_select: int) -> List[str]:
    if not cases:
        return []
    ordered = sorted(cases, key=lambda c: float(c["distance_qvis_from_nominal_l2"]))
    n_select = min(int(n_select), len(ordered))
    bins = np.array_split(np.asarray(ordered, dtype=object), 3)
    labels = ["easy", "medium", "hard"]
    counts = [n_select // 3] * 3
    for i in range(n_select % 3):
        counts[i] += 1

    selected = []
    for items_np, label, take in zip(bins, labels, counts):
        items = list(items_np)
        for case in items:
            case["difficulty_bin"] = label
        take = min(take, len(items))
        if take:
            indices = np.linspace(0, len(items) - 1, take).round().astype(int)
            selected.extend(items[int(i)] for i in indices)

    if len(selected) < n_select:
        chosen = {id(c) for c in selected}
        selected.extend(c for c in ordered if id(c) not in chosen)
    return [str(c["case_id"]) for c in selected[:n_select]]


def load_traces(trace_dir: Path) -> List[Tuple[Path, Dict[str, object]]]:
    traces = []
    for path in sorted(trace_dir.glob("vbc_visibility_waypoint_*.json")):
        try:
            traces.append((path, json.loads(path.read_text())))
        except Exception as exc:
            print(f"[WARN] skipping unreadable trace {path}: {exc}")
    return traces


def match_errors(case: Dict[str, object], trace: Dict[str, object]):
    case_target = np.asarray(case["selected_target_xyz"], dtype=np.float64)
    trace_target = np.asarray(trace.get("target_xyz", []), dtype=np.float64)
    if trace_target.size != 3:
        return math.inf, math.inf, math.inf, math.inf

    target_err = float(np.max(np.abs(case_target - trace_target)))
    sweep_err = abs(
        float(case["nominal_sweep_time_s"])
        - float(trace.get("nominal_sweep_time_s", math.inf))
    )

    case_q_nom = np.asarray(case["q_nom_deadline"], dtype=np.float64)
    trace_q_nom = np.asarray(trace.get("q_deadline_nominal", []), dtype=np.float64)
    q_nom_err = (
        float(np.max(np.abs(case_q_nom - trace_q_nom)))
        if trace_q_nom.size == 7
        else math.inf
    )

    score = target_err * 1e6 + sweep_err * 1e4 + q_nom_err
    return score, target_err, sweep_err, q_nom_err


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--select-num", type=int, default=12)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    trace_dir = Path(args.trace_dir).expanduser().resolve()
    payload = json.loads(input_path.read_text())
    cases = payload.get("valid_cases", [])
    traces = load_traces(trace_dir)

    raw_backup = input_path.with_name(input_path.stem + "_raw" + input_path.suffix)
    if not raw_backup.exists():
        shutil.copy2(input_path, raw_backup)

    unused = set(range(len(traces)))
    failures = []
    paired = 0

    for case in cases:
        ranked = []
        for trace_index in unused:
            score, target_err, sweep_err, q_nom_err = match_errors(
                case, traces[trace_index][1]
            )
            ranked.append(
                (score, target_err, sweep_err, q_nom_err, trace_index)
            )
        ranked.sort(key=lambda x: x[0])

        if not ranked:
            case["projector_trace_pairing"] = "missing"
            failures.append(
                {"case_id": case.get("case_id"), "reason": "no_trace_left"}
            )
            continue

        _, target_err, sweep_err, q_nom_err, trace_index = ranked[0]
        if target_err > 1e-5 or sweep_err > 1e-5 or q_nom_err > 1e-4:
            case["projector_trace_pairing"] = "mismatch"
            failures.append(
                {
                    "case_id": case.get("case_id"),
                    "reason": "no_consistent_trace",
                    "best_target_error_inf": target_err,
                    "best_sweep_error_s": sweep_err,
                    "best_q_nom_error_inf": q_nom_err,
                }
            )
            continue

        unused.remove(trace_index)
        trace_path, trace = traces[trace_index]
        paired += 1

        case["projector_trace_pairing"] = "matched"
        case["projector_trace_file"] = str(trace_path)
        case["projector_trace_target_error_inf"] = target_err
        case["projector_trace_sweep_error_s"] = sweep_err
        case["projector_trace_q_nom_error_inf"] = q_nom_err

        q_zero = np.asarray(trace["q_zero"], dtype=np.float64)
        q_vis = np.asarray(trace["q_vis"], dtype=np.float64)
        q_nom = np.asarray(case["q_nom_deadline"], dtype=np.float64)
        delta = q_vis - q_nom
        case["q_zero"] = q_zero.tolist()
        case["q_vis"] = q_vis.tolist()
        case["distance_qvis_from_nominal_l2"] = float(np.linalg.norm(delta))
        case["distance_qvis_from_nominal_inf"] = float(np.max(np.abs(delta)))

        for key in (
            "initial_f",
            "projection_zero_f",
            "final_f",
            "projection_root_source",
            "distance_qzero_from_nominal",
            "distance_qvis_from_nominal",
            "initial_oracle_diagnostic",
            "zero_oracle_diagnostic",
            "final_oracle_diagnostic",
            "projection_history",
            "root_refinement_history",
            "ascent_history",
            "config",
        ):
            if key in trace:
                case[key] = copy.deepcopy(trace[key])

    pairing_summary = {
        "num_cases": len(cases),
        "num_trace_files": len(traces),
        "num_paired": paired,
        "num_failures": len(failures),
        "failures": failures,
    }
    payload["projector_trace_pairing_summary"] = pairing_summary

    if failures and args.require_all:
        payload["selected_case_ids"] = []
        payload["selected_cases"] = []
        input_path.write_text(json.dumps(payload, indent=2, allow_nan=True))
        raise SystemExit(
            f"Phase-C1 finalization failed: paired {paired}/{len(cases)} cases; "
            f"see projector_trace_pairing_summary in {input_path}"
        )

    matched_cases = [
        case
        for case in cases
        if case.get("projector_trace_pairing") == "matched"
    ]
    selected_ids = choose_cases(matched_cases, args.select_num)
    selected_set = set(selected_ids)
    payload["selected_case_ids"] = selected_ids
    payload["selected_cases"] = [
        case for case in matched_cases if case["case_id"] in selected_set
    ]
    payload.setdefault("summary", {})["num_projector_traces_paired"] = paired
    payload["summary"]["num_selected_cases"] = len(selected_ids)
    payload["summary"]["finalized_from_projector_traces"] = True
    input_path.write_text(json.dumps(payload, indent=2, allow_nan=True))

    print(json.dumps(pairing_summary, indent=2))
    print("selected_case_ids:", selected_ids)


if __name__ == "__main__":
    main()
