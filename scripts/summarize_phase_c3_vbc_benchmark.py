#!/usr/bin/env python3
"""Aggregate the frozen 12-case x 3 Phase-C3 VBC benchmark."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional


MODES = ("baseline", "deadline_recovery", "predictive_recovery")


def read_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def one_glob(root: Path, pattern: str) -> Optional[Path]:
    xs = sorted(root.glob(pattern))
    return xs[-1] if xs else None


def finite_float(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def bool_or_none(value):
    return value if isinstance(value, bool) else None


def valid_run(row: Dict[str, Any]) -> bool:
    return (
        row.get("run_status") == "ok"
        and row.get("initial_q_match") is True
        and row.get("runtime_case_match") is True
        and row.get("runtime_projector_match") is True
    )


def safe(row: Dict[str, Any]) -> Optional[bool]:
    return row.get("vbc_safe_0p30") if valid_run(row) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-file", required=True)
    ap.add_argument("--output-root", required=True)
    args = ap.parse_args()

    case_file = Path(args.case_file).resolve()
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    frozen = json.loads(case_file.read_text())
    cases = {c["case_id"]: c for c in frozen["cases"]}
    order = list(frozen["selected_case_ids"])
    safety_margin = float(frozen.get("safety_margin_s", 0.30))

    mode_weights = {
        "baseline": float(frozen.get("waypoint_weight_baseline", 0.0)),
        "deadline_recovery": float(frozen.get("waypoint_weight_careplanner", 3000.0)),
        "predictive_recovery": float(frozen.get("waypoint_weight_careplanner", 3000.0)),
    }

    rows = []
    for cid in order:
        c = cases[cid]
        for mode in MODES:
            rd = root / "runs" / cid / mode
            status = read_json(rd / "run_status.json")
            valid = read_json(rd / "runtime_case_validation.json")
            initv = read_json(rd / "initial_q_validation.json")
            projv = read_json(rd / "runtime_projector_validation.json")
            exe = read_json(one_glob(rd, "executed_vbc_*.json"))
            gate = read_json(one_glob(rd, "execution_gate_*.json"))
            wp_path = one_glob(rd, "*_weight_*/summary.json")
            wp = read_json(wp_path)

            control_counts = wp.get("control_mode_counts", {})
            if not isinstance(control_counts, dict):
                control_counts = {}

            row = {
                "case_id": cid,
                "difficulty_bin": c.get("difficulty_bin"),
                "mode": mode,
                "waypoint_weight": mode_weights[mode],
                "run_status": status.get("status", "missing"),
                "initial_q_match": initv.get("passed"),
                "runtime_case_match": valid.get("passed"),
                "runtime_projector_match": projv.get("passed"),
                "expected_sweep_time_s": c.get("nominal_sweep_time_s"),
                "d_q_l2": c.get("distance_qvis_from_nominal_l2"),
                "nominally_visible": c.get("nominally_visible"),
                "nominal_vbc_margin_s": c.get("nominal_vbc_margin_s"),
                "executed_seen_before_sweep": bool_or_none(exe.get("seen_before_sweep")),
                "executed_vbc_margin_s": finite_float(exe.get("executed_vbc_margin_s")),
                "executed_see_delay_s": finite_float(exe.get("see_delay_from_target_s")),
                "executed_sweep_delay_s": finite_float(exe.get("sweep_delay_from_target_s")),
                "min_clearance_all_m": finite_float(exe.get("min_clearance_all_m")),
                "max_tracking_inf": finite_float(wp.get("max_tracking_inf")),
                "max_pred_dev_inf": finite_float(wp.get("max_pred_dev_inf")),
                "max_command_inf": finite_float(wp.get("max_command_inf")),
                "solve_ms_mean": finite_float(wp.get("solve_ms_mean")),
                "solve_ms_max": finite_float(wp.get("solve_ms_max")),
                "min_waypoint_pred_error_inf": finite_float(wp.get("min_waypoint_pred_error_inf")),
                "recovery_entered": int(control_counts.get("recovery", 0)) > 0,
                "recovery_hold_entered": int(control_counts.get("recovery_hold", 0)) > 0,
                "first_recovery_delay_s": finite_float(wp.get("first_recovery_delay_from_target_s")),
                "predictive_trigger_count": int(wp.get("predictive_trigger_count_total", 0) or 0),
                "predictive_trigger_delay_s": finite_float(wp.get("first_predictive_trigger_observed_delay_from_target_s")),
                "predictive_trigger_lead_s": finite_float(wp.get("last_predictive_trigger_lead_s")),
                "predictive_trigger_error_inf": finite_float(wp.get("last_predictive_trigger_error_inf")),
                "gate_released": gate.get("released"),
                "gate_replan_count": int(gate.get("replan_count", 0) or 0),
                "gate_master_duration_s": finite_float(gate.get("master_duration_s")),
            }
            margin = row["executed_vbc_margin_s"]
            row["vbc_safe_0p30"] = None if margin is None else margin >= safety_margin
            rows.append(row)

    fields = list(rows[0]) if rows else []
    with (root / "benchmark_runs.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    paired = []
    for cid in order:
        by_mode = {r["mode"]: r for r in rows if r["case_id"] == cid}
        base = by_mode.get("baseline", {})
        deadline = by_mode.get("deadline_recovery", {})
        pred = by_mode.get("predictive_recovery", {})
        bm = base.get("executed_vbc_margin_s")
        dm = deadline.get("executed_vbc_margin_s")
        pm = pred.get("executed_vbc_margin_s")
        bs, ds, ps = safe(base), safe(deadline), safe(pred)

        paired.append({
            "case_id": cid,
            "difficulty_bin": cases[cid].get("difficulty_bin"),
            "d_q_l2": cases[cid].get("distance_qvis_from_nominal_l2"),
            "baseline_margin_s": bm,
            "deadline_recovery_margin_s": dm,
            "predictive_recovery_margin_s": pm,
            "deadline_vs_baseline_margin_gain_s": None if bm is None or dm is None else dm - bm,
            "predictive_vs_deadline_margin_gain_s": None if dm is None or pm is None else pm - dm,
            "baseline_safe": bs,
            "deadline_recovery_safe": ds,
            "predictive_recovery_safe": ps,
            "deadline_rescue_vs_baseline": bs is False and ds is True,
            "predictive_rescue_vs_deadline": ds is False and ps is True,
            "predictive_regression_vs_deadline": ds is True and ps is False,
            "baseline_safe_to_predictive_unsafe": bs is True and ps is False,
            "deadline_recovery_entered": deadline.get("recovery_entered"),
            "predictive_recovery_entered": pred.get("recovery_entered"),
            "predictive_trigger_count": pred.get("predictive_trigger_count"),
            "predictive_trigger_lead_s": pred.get("predictive_trigger_lead_s"),
            "predictive_trigger_error_inf": pred.get("predictive_trigger_error_inf"),
            "predictive_replan_count": pred.get("gate_replan_count"),
        })

    paired_fields = list(paired[0]) if paired else []
    with (root / "paired_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=paired_fields)
        writer.writeheader()
        writer.writerows(paired)

    def stats(mode: str, difficulty: Optional[str] = None):
        rr = [
            r for r in rows
            if r["mode"] == mode and (difficulty is None or r["difficulty_bin"] == difficulty)
        ]
        ok = [r for r in rr if valid_run(r)]
        margins = [r["executed_vbc_margin_s"] for r in ok if r["executed_vbc_margin_s"] is not None]
        safe_rows = [r for r in ok if r["vbc_safe_0p30"] is True]
        seen = [r for r in ok if r["executed_seen_before_sweep"] is True]
        recovered = [r for r in ok if r["recovery_entered"] is True]
        triggered = [r for r in ok if r["predictive_trigger_count"] > 0]
        return {
            "num_expected_runs": len(rr),
            "num_valid_runs": len(ok),
            "num_seen_before_sweep": len(seen),
            "num_safe_margin_ge_0p30": len(safe_rows),
            "safe_rate_ge_0p30_over_valid": None if not ok else len(safe_rows) / len(ok),
            "safe_rate_ge_0p30_over_expected": None if not rr else len(safe_rows) / len(rr),
            "num_recovery_entered": len(recovered),
            "num_predictive_triggered": len(triggered),
            "margin_mean_s": None if not margins else sum(margins) / len(margins),
            "margin_min_s": None if not margins else min(margins),
            "margin_max_s": None if not margins else max(margins),
        }

    rescue_candidates = [p for p in paired if p["predictive_rescue_vs_deadline"]]
    rescue_candidates.sort(
        key=lambda p: (
            -(p["predictive_vs_deadline_margin_gain_s"] if p["predictive_vs_deadline_margin_gain_s"] is not None else -1e9),
            -(p["predictive_trigger_lead_s"] if p["predictive_trigger_lead_s"] is not None else -1e9),
        )
    )

    stable_candidates = [
        p for p in paired
        if p["deadline_recovery_safe"] is True and p["predictive_recovery_safe"] is True
    ]
    stable_candidates.sort(
        key=lambda p: (
            0 if int(p.get("predictive_trigger_count") or 0) == 0 else 1,
            abs(p["predictive_vs_deadline_margin_gain_s"])
            if p["predictive_vs_deadline_margin_gain_s"] is not None else 1e9,
        )
    )

    representative = {
        "selection_is_provisional": True,
        "note": "Final two cases should be chosen after inspecting the full benchmark. Prefer one predictive rescue and one stable/no-regression case.",
        "predictive_rescue_candidates": rescue_candidates,
        "stable_no_regression_candidates": stable_candidates,
        "suggested_primary_rescue_case": rescue_candidates[0]["case_id"] if rescue_candidates else None,
        "suggested_stability_case": stable_candidates[0]["case_id"] if stable_candidates else None,
    }
    (root / "representative_case_candidates.json").write_text(
        json.dumps(representative, indent=2)
    )

    regressions = [p["case_id"] for p in paired if p["predictive_regression_vs_deadline"]]
    baseline_regressions = [p["case_id"] for p in paired if p["baseline_safe_to_predictive_unsafe"]]
    predictive_rescues = [p["case_id"] for p in paired if p["predictive_rescue_vs_deadline"]]

    payload = {
        "benchmark_name": "phase_c3_predictive_recovery_frozen_12x3",
        "case_file": str(case_file),
        "safety_margin_s": safety_margin,
        "num_cases": len(order),
        "num_expected_runs": len(MODES) * len(order),
        "modes": list(MODES),
        "baseline": stats("baseline"),
        "deadline_recovery": stats("deadline_recovery"),
        "predictive_recovery": stats("predictive_recovery"),
        "by_difficulty": {
            difficulty: {mode: stats(mode, difficulty) for mode in MODES}
            for difficulty in ("easy", "medium", "hard")
        },
        "predictive_rescue_vs_deadline_case_ids": predictive_rescues,
        "predictive_regression_vs_deadline_case_ids": regressions,
        "baseline_safe_to_predictive_unsafe_case_ids": baseline_regressions,
        "num_predictive_rescues_vs_deadline": len(predictive_rescues),
        "num_predictive_regressions_vs_deadline": len(regressions),
        "num_baseline_safe_to_predictive_unsafe": len(baseline_regressions),
        "selected_case_ids": order,
        "paired_cases": paired,
        "representative_case_candidates": representative,
    }
    (root / "benchmark_summary.json").write_text(json.dumps(payload, indent=2))

    print(json.dumps({
        "baseline": payload["baseline"],
        "deadline_recovery": payload["deadline_recovery"],
        "predictive_recovery": payload["predictive_recovery"],
        "predictive_rescues_vs_deadline": predictive_rescues,
        "predictive_regressions_vs_deadline": regressions,
        "baseline_safe_to_predictive_unsafe": baseline_regressions,
        "representative_case_candidates": representative,
    }, indent=2))


if __name__ == "__main__":
    main()
