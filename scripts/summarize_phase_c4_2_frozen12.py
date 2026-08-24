#!/usr/bin/env python3
"""Summarize the Phase-C4.2 frozen-12 closed-loop benchmark.

C4.2 is a control experiment, not a diagnostic classifier.  The primary
questions are therefore:
  1) does the predicted-VBC guard trigger Recovery only for persistent
     predicted violations?
  2) are all executed trials VBC-safe?
  3) how many Recovery interventions are avoided relative to the historical
     legacy Deadline-Recovery runs, when those raw outputs are available?

The optional legacy comparison is explicitly behavioral; legacy Recovery is
not treated as safety ground truth.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

TOKEN_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


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


def finite(v: Any) -> Optional[float]:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def integer(v: Any) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def boolish(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes"):
            return True
        if s in ("0", "false", "no"):
            return False
    return None


def safe_outcome(outcome: Any) -> bool:
    return str(outcome) in ("safe_margin", "safe_by_avoidance")


def parse_guard_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    out: List[Dict[str, str]] = []
    try:
        with path.open(newline="", errors="replace") as fh:
            rd = csv.reader(fh)
            header = next(rd, [])
            if not header:
                return []
            data_idx = header.index("field.data") if "field.data" in header else 1
            for row in rd:
                if len(row) <= data_idx:
                    continue
                tok = dict(TOKEN_RE.findall(row[data_idx]))
                if tok:
                    out.append(tok)
    except Exception:
        return []
    return out


def legacy_recovery_entered(run_dir: Path) -> Optional[bool]:
    wp_path = one_glob(run_dir, "*_weight_*/summary.json")
    wp = read_json(wp_path)
    if not wp:
        return None
    first = finite(wp.get("first_recovery_delay_from_target_s"))
    if first is None:
        first = finite(wp.get("first_recovery_delay_s"))
    if first is not None:
        return True
    counts = wp.get("control_mode_counts", {})
    if isinstance(counts, dict):
        try:
            return int(counts.get("recovery", 0)) > 0 or int(counts.get("recovery_hold", 0)) > 0
        except Exception:
            pass
    return False


def mean_or_none(xs: List[float]) -> Optional[float]:
    return None if not xs else sum(xs) / len(xs)


def median_or_none(xs: List[float]) -> Optional[float]:
    return None if not xs else statistics.median(xs)


def min_or_none(xs: List[float]) -> Optional[float]:
    return None if not xs else min(xs)


def max_or_none(xs: List[float]) -> Optional[float]:
    return None if not xs else max(xs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-file", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--legacy-c4-1-root", default="")
    ap.add_argument("--target-tol-m", type=float, default=0.001)
    args = ap.parse_args()

    case_file = Path(args.case_file).resolve()
    root = Path(args.output_root).resolve()
    legacy_root = Path(args.legacy_c4_1_root).resolve() if args.legacy_c4_1_root else None

    frozen = json.loads(case_file.read_text())
    order = list(frozen["selected_case_ids"])
    cases = {c["case_id"]: c for c in frozen["cases"]}

    rows: List[Dict[str, Any]] = []
    for cid in order:
        c = cases[cid]
        rd = root / "runs" / cid
        status = read_json(rd / "run_status.json")
        c42 = read_json(rd / "c4_2_summary.json")
        exe = read_json(one_glob(rd, "executed_vbc_*.json"))
        wp = read_json(one_glob(rd, "*_weight_*/summary.json"))
        gate = read_json(one_glob(rd, "execution_gate_*.json"))
        guard_records = parse_guard_csv(rd / "predicted_vbc_recovery_guard.csv")

        max_streak = 0
        for g in guard_records:
            s = integer(g.get("max_violation_streak"))
            if s is not None:
                max_streak = max(max_streak, s)

        target_expected = [float(x) for x in c["selected_target_xyz"]]
        target_actual = exe.get("target_xyz") if isinstance(exe.get("target_xyz"), list) else None
        target_err = None
        if target_actual is not None and len(target_actual) == 3:
            target_err = max(abs(float(a) - float(b)) for a, b in zip(target_actual, target_expected))

        run_ok = status.get("status") == "ok"
        summary_ok = bool(c42)
        target_ok = target_err is None or target_err <= args.target_tol_m
        valid = bool(run_ok and summary_ok and target_ok)

        guard_triggered = boolish(c42.get("guard_triggered"))
        recovery_entered = boolish(c42.get("recovery_entered"))
        outcome = c42.get("executed_outcome", "missing")
        audit_violation = boolish(c42.get("audit_violation_observed"))
        verification_hold = boolish(c42.get("verification_hold_observed"))

        legacy_rec = None
        if legacy_root is not None and legacy_root.is_dir():
            legacy_rec = legacy_recovery_entered(legacy_root / "runs" / cid)

        row: Dict[str, Any] = {
            "case_id": cid,
            "difficulty_bin": c.get("difficulty_bin"),
            "d_q_l2": finite(c.get("distance_qvis_from_nominal_l2")),
            "run_status": status.get("status", "missing"),
            "valid": valid,
            "target_error_inf_m": target_err,
            "num_active_audits": integer(c42.get("num_active_audits")),
            "num_violation_audits": integer(c42.get("num_violation_audits")),
            "audit_violation_observed": audit_violation,
            "max_violation_streak": max_streak,
            "guard_triggered": guard_triggered,
            "guard_trigger_count_total": integer(c42.get("guard_trigger_count_total")),
            "guard_trigger_lead_s": finite(c42.get("guard_last_trigger_lead_s")),
            "verification_hold_observed": verification_hold,
            "verification_hold_summary_records": integer(c42.get("verification_hold_summary_records")),
            "recovery_entered": recovery_entered,
            "first_recovery_delay_from_target_s": finite(c42.get("first_recovery_delay_from_target_s", c42.get("first_recovery_delay_s"))),
            "replan_count": integer(c42.get("replan_count", gate.get("replan_count"))),
            "executed_outcome": outcome,
            "executed_safe": safe_outcome(outcome),
            "executed_vbc_margin_s": finite(c42.get("executed_vbc_margin_s")),
            "executed_see_delay_s": finite(c42.get("executed_see_delay_s")),
            "executed_sweep_delay_s": finite(c42.get("executed_sweep_delay_s")),
            "min_clearance_all_m": finite(c42.get("min_clearance_all_m")),
            "audit_ms_mean": finite(c42.get("audit_ms_mean")),
            "audit_ms_max": finite(c42.get("audit_ms_max")),
            "mpc_solve_ms_mean": finite(wp.get("solve_ms_mean")),
            "mpc_solve_ms_max": finite(wp.get("solve_ms_max")),
            "max_tracking_inf": finite(wp.get("max_tracking_inf")),
            "max_pred_dev_inf": finite(wp.get("max_pred_dev_inf")),
            "legacy_deadline_recovery_entered": legacy_rec,
        }
        if legacy_rec is None or recovery_entered is None:
            row["recovery_change_vs_legacy"] = "unavailable"
        elif legacy_rec and not recovery_entered:
            row["recovery_change_vs_legacy"] = "legacy_only"
        elif (not legacy_rec) and recovery_entered:
            row["recovery_change_vs_legacy"] = "c4_2_only"
        elif legacy_rec and recovery_entered:
            row["recovery_change_vs_legacy"] = "both"
        else:
            row["recovery_change_vs_legacy"] = "neither"
        rows.append(row)

    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with (root / "case_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    valid = [r for r in rows if r["valid"]]
    safe = [r for r in valid if r["executed_safe"]]
    triggered = [r for r in valid if r["guard_triggered"]]
    recovery = [r for r in valid if r["recovery_entered"]]
    safe_no_recovery = [r for r in safe if not r["recovery_entered"]]
    isolated_filtered = [
        r for r in valid
        if r["audit_violation_observed"] and not r["guard_triggered"]
    ]
    recovery_without_trigger = [
        r for r in valid if r["recovery_entered"] and not r["guard_triggered"]
    ]
    trigger_without_recovery = [
        r for r in valid if r["guard_triggered"] and not r["recovery_entered"]
    ]
    unsafe = [r for r in valid if not r["executed_safe"]]
    hold_cases = [r for r in valid if r["verification_hold_observed"]]

    leads = [r["guard_trigger_lead_s"] for r in triggered if r["guard_trigger_lead_s"] is not None]
    recovery_delays = [
        r["first_recovery_delay_from_target_s"] for r in recovery
        if r["first_recovery_delay_from_target_s"] is not None
    ]
    margins = [r["executed_vbc_margin_s"] for r in safe if r["executed_vbc_margin_s"] is not None]
    audit_means = [r["audit_ms_mean"] for r in valid if r["audit_ms_mean"] is not None]
    audit_maxes = [r["audit_ms_max"] for r in valid if r["audit_ms_max"] is not None]
    mpc_means = [r["mpc_solve_ms_mean"] for r in valid if r["mpc_solve_ms_mean"] is not None]
    mpc_maxes = [r["mpc_solve_ms_max"] for r in valid if r["mpc_solve_ms_max"] is not None]

    legacy_available = [r for r in valid if r["legacy_deadline_recovery_entered"] is not None]
    legacy_recovery = [r for r in legacy_available if r["legacy_deadline_recovery_entered"]]
    legacy_only = [r for r in legacy_available if r["recovery_change_vs_legacy"] == "legacy_only"]
    c42_only = [r for r in legacy_available if r["recovery_change_vs_legacy"] == "c4_2_only"]

    def ids(xs: List[Dict[str, Any]]) -> List[str]:
        return [r["case_id"] for r in xs]

    summary = {
        "num_cases": len(rows),
        "num_valid": len(valid),
        "valid_case_ids": ids(valid),
        "num_safe": len(safe),
        "safety_rate": None if not valid else len(safe) / len(valid),
        "safe_case_ids": ids(safe),
        "unsafe_case_ids": ids(unsafe),
        "executed_outcome_counts": dict(Counter(str(r["executed_outcome"]) for r in valid)),
        "num_guard_triggered": len(triggered),
        "guard_triggered_case_ids": ids(triggered),
        "num_recovery_entered": len(recovery),
        "recovery_case_ids": ids(recovery),
        "num_safe_without_recovery": len(safe_no_recovery),
        "safe_without_recovery_case_ids": ids(safe_no_recovery),
        "num_isolated_violation_filtered": len(isolated_filtered),
        "isolated_violation_filtered_case_ids": ids(isolated_filtered),
        "recovery_without_guard_trigger_case_ids": ids(recovery_without_trigger),
        "guard_trigger_without_recovery_case_ids": ids(trigger_without_recovery),
        "verification_hold_case_ids": ids(hold_cases),
        "trigger_lead_s": {
            "mean": mean_or_none(leads),
            "median": median_or_none(leads),
            "min": min_or_none(leads),
            "max": max_or_none(leads),
        },
        "first_recovery_delay_from_target_s": {
            "mean": mean_or_none(recovery_delays),
            "median": median_or_none(recovery_delays),
            "min": min_or_none(recovery_delays),
            "max": max_or_none(recovery_delays),
        },
        "executed_margin_s_for_swept_safe_cases": {
            "mean": mean_or_none(margins),
            "median": median_or_none(margins),
            "min": min_or_none(margins),
            "max": max_or_none(margins),
        },
        "audit_runtime_ms": {
            "mean_of_case_means": mean_or_none(audit_means),
            "median_of_case_means": median_or_none(audit_means),
            "worst_single_audit": max_or_none(audit_maxes),
        },
        "mpc_runtime_ms": {
            "mean_of_case_means": mean_or_none(mpc_means),
            "worst_single_solve": max_or_none(mpc_maxes),
        },
        "replan_count_total": sum(r["replan_count"] or 0 for r in valid),
        "replan_case_ids": ids([r for r in valid if (r["replan_count"] or 0) > 0]),
        "legacy_behavior_comparison_available": bool(legacy_available),
        "legacy_behavior_num_cases": len(legacy_available),
        "legacy_deadline_recovery_count": len(legacy_recovery),
        "legacy_deadline_recovery_case_ids": ids(legacy_recovery),
        "c4_2_recovery_count_on_legacy_comparable_cases": sum(bool(r["recovery_entered"]) for r in legacy_available),
        "legacy_recovery_avoided_case_ids": ids(legacy_only),
        "new_c4_2_recovery_case_ids": ids(c42_only),
        "legacy_note": "Historical Deadline-Recovery behavior is a comparison label, not safety ground truth.",
    }
    (root / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    key_cols = [
        "case_id", "guard_triggered", "max_violation_streak", "recovery_entered",
        "guard_trigger_lead_s", "executed_outcome", "executed_vbc_margin_s",
        "replan_count", "audit_ms_mean", "legacy_deadline_recovery_entered",
        "recovery_change_vs_legacy",
    ]
    md = ["| " + " | ".join(key_cols) + " |", "|" + "|".join(["---"] * len(key_cols)) + "|"]
    for r in rows:
        vals = []
        for k in key_cols:
            v = r.get(k)
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            elif v is None:
                vals.append("")
            else:
                vals.append(str(v))
        md.append("| " + " | ".join(vals) + " |")
    (root / "case_results.md").write_text("\n".join(md) + "\n")

    print("[SUMMARY] valid={}/{} safe={}/{} trigger={} recovery={} safe_without_recovery={}".format(
        len(valid), len(rows), len(safe), len(valid), len(triggered), len(recovery), len(safe_no_recovery)))
    print("[CASES] trigger:", ids(triggered))
    print("[CASES] recovery:", ids(recovery))
    print("[CASES] safe without recovery:", ids(safe_no_recovery))
    if legacy_available:
        print("[LEGACY] Deadline Recovery {} -> C4.2 Recovery {} on {} comparable cases".format(
            len(legacy_recovery), sum(bool(r["recovery_entered"]) for r in legacy_available), len(legacy_available)))
        print("[LEGACY] avoided Recovery:", ids(legacy_only))
        print("[LEGACY] new C4.2 Recovery:", ids(c42_only))
    if recovery_without_trigger:
        print("[WARN] Recovery without C4 trigger:", ids(recovery_without_trigger))
    if trigger_without_recovery:
        print("[WARN] C4 trigger without Recovery:", ids(trigger_without_recovery))
    if unsafe:
        print("[WARN] unsafe/inconclusive cases:", [(r["case_id"], r["executed_outcome"]) for r in unsafe])


if __name__ == "__main__":
    main()
