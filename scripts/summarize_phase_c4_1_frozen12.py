#!/usr/bin/env python3
"""Summarize the Phase-C4.1 frozen-12 predicted-VBC diagnostic benchmark.

Ground truth is deliberately behavioral and comes from the same run:
  positive := the validated Deadline-Recovery controller actually entered
              visibility recovery because soft intervention had not solved VBC
              by the physical deadline.

C4.1 remains diagnostic-only.  We compare whether its predicted-VBC violation
signal was available *before* that physical deadline, including 1/2/3-cycle
consecutive-violation variants so the future temporal-consistency choice is
data-driven rather than tuned by hand.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def boolish(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        if v.lower() in ("1", "true", "yes"):
            return True
        if v.lower() in ("0", "false", "no"):
            return False
    return None


def csv_ros_time_to_sec(v: str) -> Optional[float]:
    """ROS1 rostopic -p normally writes %time in integer nanoseconds."""
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    if abs(x) > 1e14:
        return x * 1e-9
    return x


def parse_audit_csv(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    try:
        with path.open(newline="", errors="replace") as fh:
            reader = csv.reader(fh)
            header = next(reader, [])
            data_idx = header.index("field.data") if "field.data" in header else 1
            time_idx = header.index("%time") if "%time" in header else 0
            for row in reader:
                if len(row) <= max(data_idx, time_idx):
                    continue
                tok = dict(TOKEN_RE.findall(row[data_idx]))
                if "audit_ms" not in tok:
                    continue
                out.append({
                    "ros_time_s": csv_ros_time_to_sec(row[time_idx]),
                    "status": tok.get("status"),
                    "violation": int(tok.get("violation", "0")),
                    "predicted_seen": int(tok.get("predicted_seen", "0")) if "predicted_seen" in tok else None,
                    "predicted_sweep": int(tok.get("predicted_sweep", "0")) if "predicted_sweep" in tok else None,
                    "see_time_s": finite(tok.get("see_time_s")),
                    "sweep_time_s": finite(tok.get("sweep_time_s")),
                    "margin_s": finite(tok.get("margin_s")),
                    "min_clearance_m": finite(tok.get("min_clearance_m")),
                    "evaluated_q": int(tok.get("evaluated_q", "0")),
                    "prediction_q": int(tok.get("prediction_q", "0")),
                    "audit_ms": finite(tok.get("audit_ms")),
                })
    except Exception:
        return []
    return out


def parse_initial_gate_timing(log_path: Path) -> Tuple[Optional[float], Optional[float]]:
    if not log_path.is_file():
        return None, None
    text = log_path.read_text(errors="replace")
    # Use the first release/deadline pair. Later recovery replans intentionally
    # reset the gate's execution start and must not be used for C4 trigger lead.
    m0 = re.search(r"\bT0=([+\-0-9.eE]+)\s+master_duration=", text)
    md = re.search(r"synchronized VBC deadline:\s*\+[+\-0-9.eE]+\s+s\s+absolute=([+\-0-9.eE]+)", text)
    return (finite(m0.group(1)) if m0 else None,
            finite(md.group(1)) if md else None)


def waypoint_summary(run_dir: Path) -> Dict[str, Any]:
    xs = sorted(run_dir.glob("*_weight_*/summary.json"))
    return read_json(xs[-1]) if xs else {}


def recovery_entered(wp: Dict[str, Any]) -> Optional[bool]:
    if not wp:
        return None
    first = finite(wp.get("first_recovery_delay_s"))
    if first is not None:
        return True
    counts = wp.get("control_mode_counts", {})
    if isinstance(counts, dict):
        try:
            return int(counts.get("recovery", 0)) > 0 or int(counts.get("recovery_hold", 0)) > 0
        except Exception:
            pass
    # Older logger summaries may expose explicit event counts.
    for key in ("recovery_enter_count", "num_recovery_entered"):
        if key in wp:
            try:
                return int(wp[key]) > 0
            except Exception:
                pass
    return False


def executed_outcome(exe: Dict[str, Any], safety_margin: float) -> str:
    if not exe:
        return "missing"
    margin = finite(exe.get("executed_vbc_margin_s"))
    see = finite(exe.get("see_delay_from_target_s"))
    sweep = finite(exe.get("sweep_delay_from_target_s"))
    if margin is not None:
        return "safe_margin" if margin + 1e-9 >= safety_margin else "unsafe_margin"
    if see is not None and sweep is None:
        return "safe_by_avoidance"
    if sweep is not None and see is None:
        return "unsafe_unseen_sweep"
    return "inconclusive"


def confusion(rows: List[Dict[str, Any]], streak: int) -> Dict[str, Any]:
    valid = [r for r in rows if r.get("valid") and r.get("recovery_entered") is not None]
    key = f"streak{streak}_predicted_positive_in_time"
    tp = sum(bool(r[key]) and bool(r["recovery_entered"]) for r in valid)
    fp = sum(bool(r[key]) and not bool(r["recovery_entered"]) for r in valid)
    fn = sum((not bool(r[key])) and bool(r["recovery_entered"]) for r in valid)
    tn = sum((not bool(r[key])) and (not bool(r["recovery_entered"])) for r in valid)
    def ratio(a: int, b: int) -> Optional[float]:
        return None if b == 0 else a / b
    return {
        "streak_required": streak,
        "num_valid": len(valid),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity_recall": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "precision": ratio(tp, tp + fp),
        "false_positive_rate": ratio(fp, fp + tn),
        "accuracy": ratio(tp + tn, len(valid)),
        "tp_case_ids": [r["case_id"] for r in valid if r[key] and r["recovery_entered"]],
        "fp_case_ids": [r["case_id"] for r in valid if r[key] and not r["recovery_entered"]],
        "fn_case_ids": [r["case_id"] for r in valid if not r[key] and r["recovery_entered"]],
        "tn_case_ids": [r["case_id"] for r in valid if not r[key] and not r["recovery_entered"]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-file", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--log-root", required=True)
    ap.add_argument("--target-tol-m", type=float, default=0.001)
    ap.add_argument("--deadline-offset-tol-s", type=float, default=0.03)
    args = ap.parse_args()

    case_file = Path(args.case_file).resolve()
    root = Path(args.output_root).resolve()
    log_root = Path(args.log_root).resolve()
    frozen = json.loads(case_file.read_text())
    safety_margin = float(frozen.get("safety_margin_s", 0.30))
    cases = {c["case_id"]: c for c in frozen["cases"]}
    order = list(frozen["selected_case_ids"])

    rows: List[Dict[str, Any]] = []
    for cid in order:
        c = cases[cid]
        rd = root / "runs" / cid
        ld = log_root / cid
        status = read_json(rd / "run_status.json")
        exe = read_json(one_glob(rd, "executed_vbc_*.json"))
        gate = read_json(one_glob(rd, "execution_gate_*.json"))
        wp = waypoint_summary(rd)
        audits = parse_audit_csv(rd / "predicted_vbc_audit.csv")
        active = [r for r in audits if r.get("status") not in ("inactive", "waiting_target") and r.get("audit_ms") is not None and r.get("ros_time_s") is not None]

        t0, deadline_abs = parse_initial_gate_timing(ld / "controlled.log")
        expected_deadline_offset = max(0.0, float(c["nominal_sweep_time_s"]) - safety_margin)
        actual_deadline_offset = None if t0 is None or deadline_abs is None else deadline_abs - t0
        deadline_offset_error = None if actual_deadline_offset is None else abs(actual_deadline_offset - expected_deadline_offset)

        target_expected = [float(x) for x in c["selected_target_xyz"]]
        target_actual = exe.get("target_xyz") if isinstance(exe.get("target_xyz"), list) else None
        target_err = None
        if target_actual is not None and len(target_actual) == 3:
            target_err = max(abs(float(a) - float(b)) for a, b in zip(target_actual, target_expected))

        pre = [] if deadline_abs is None else [r for r in active if r["ros_time_s"] <= deadline_abs + 1e-6]
        violations_pre = [r for r in pre if r["violation"] == 1]
        first_v = violations_pre[0] if violations_pre else None

        first_streak_time: Dict[int, Optional[float]] = {1: None, 2: None, 3: None}
        streak = 0
        max_streak = 0
        for r in pre:
            if r["violation"] == 1:
                streak += 1
                max_streak = max(max_streak, streak)
                for n in (1, 2, 3):
                    if streak >= n and first_streak_time[n] is None:
                        first_streak_time[n] = r["ros_time_s"]
            else:
                streak = 0

        rec = recovery_entered(wp)
        recovery_delay = finite(wp.get("first_recovery_delay_s"))
        first_v_delay = None if first_v is None or t0 is None else first_v["ros_time_s"] - t0
        headroom_before_recovery = None
        if recovery_delay is not None and first_v_delay is not None:
            headroom_before_recovery = recovery_delay - first_v_delay

        audit_times = [r["audit_ms"] for r in active if r["audit_ms"] is not None]
        run_ok = status.get("status") == "ok"
        target_ok = target_err is not None and target_err <= args.target_tol_m
        deadline_ok = deadline_offset_error is not None and deadline_offset_error <= args.deadline_offset_tol_s
        valid = bool(run_ok and target_ok and deadline_ok and active and rec is not None)

        row: Dict[str, Any] = {
            "case_id": cid,
            "difficulty_bin": c.get("difficulty_bin"),
            "d_q_l2": finite(c.get("distance_qvis_from_nominal_l2")),
            "run_status": status.get("status", "missing"),
            "valid": valid,
            "target_error_inf_m": target_err,
            "deadline_from_start_expected_s": expected_deadline_offset,
            "deadline_from_start_actual_s": actual_deadline_offset,
            "deadline_offset_error_s": deadline_offset_error,
            "num_active_audits": len(active),
            "num_predeadline_audits": len(pre),
            "num_predeadline_violation_audits": len(violations_pre),
            "max_predeadline_violation_streak": max_streak,
            "first_violation_status": None if first_v is None else first_v.get("status"),
            "first_violation_predicted_seen": None if first_v is None else first_v.get("predicted_seen"),
            "first_violation_predicted_sweep": None if first_v is None else first_v.get("predicted_sweep"),
            "first_violation_predicted_see_time_s": None if first_v is None else first_v.get("see_time_s"),
            "first_violation_predicted_sweep_time_s": None if first_v is None else first_v.get("sweep_time_s"),
            "first_violation_predicted_margin_s": None if first_v is None else first_v.get("margin_s"),
            "first_violation_delay_from_execution_start_s": first_v_delay,
            "first_violation_lead_to_physical_deadline_s": None if first_v is None or deadline_abs is None else deadline_abs - first_v["ros_time_s"],
            "recovery_entered": rec,
            "first_recovery_delay_s": recovery_delay,
            "first_violation_headroom_before_recovery_s": headroom_before_recovery,
            "gate_replan_count": gate.get("replan_count"),
            "audit_ms_mean": None if not audit_times else sum(audit_times) / len(audit_times),
            "audit_ms_max": None if not audit_times else max(audit_times),
            "evaluated_q_mean": None if not active else sum(int(r.get("evaluated_q", 0)) for r in active) / len(active),
            "executed_vbc_margin_s": finite(exe.get("executed_vbc_margin_s")),
            "executed_see_delay_s": finite(exe.get("see_delay_from_target_s")),
            "executed_sweep_delay_s": finite(exe.get("sweep_delay_from_target_s")),
            "executed_outcome": executed_outcome(exe, safety_margin),
        }
        for n in (1, 2, 3):
            t = first_streak_time[n]
            row[f"streak{n}_predicted_positive_in_time"] = t is not None
            row[f"streak{n}_trigger_delay_from_execution_start_s"] = None if t is None or t0 is None else t - t0
            row[f"streak{n}_trigger_lead_to_physical_deadline_s"] = None if t is None or deadline_abs is None else deadline_abs - t
        rows.append(row)

    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with (root / "case_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    by_streak = [confusion(rows, n) for n in (1, 2, 3)]
    with (root / "classification_by_streak.csv").open("w", newline="") as fh:
        flat_fields = [
            "streak_required", "num_valid", "tp", "fp", "fn", "tn",
            "sensitivity_recall", "specificity", "precision", "false_positive_rate", "accuracy",
        ]
        w = csv.DictWriter(fh, fieldnames=flat_fields)
        w.writeheader()
        for d in by_streak:
            w.writerow({k: d.get(k) for k in flat_fields})

    valid_rows = [r for r in rows if r["valid"]]
    positives = [r for r in valid_rows if r["recovery_entered"]]
    negatives = [r for r in valid_rows if not r["recovery_entered"]]
    leads = [r["first_violation_lead_to_physical_deadline_s"] for r in valid_rows if r["first_violation_lead_to_physical_deadline_s"] is not None]
    timing = [r["audit_ms_mean"] for r in valid_rows if r["audit_ms_mean"] is not None]
    timing_max = [r["audit_ms_max"] for r in valid_rows if r["audit_ms_max"] is not None]

    payload = {
        "benchmark": "phase_c4_1_frozen12_predicted_vbc_diagnostic",
        "case_file": str(case_file),
        "safety_margin_s": safety_margin,
        "num_expected_cases": len(order),
        "num_valid_cases": len(valid_rows),
        "ground_truth_definition": "recovery_entered in the same Deadline-Recovery run; C4 auditor is diagnostic-only",
        "ground_truth_positive_case_ids": [r["case_id"] for r in positives],
        "ground_truth_negative_case_ids": [r["case_id"] for r in negatives],
        "classification_by_streak": by_streak,
        "first_violation_lead_s": {
            "count": len(leads),
            "mean": None if not leads else sum(leads) / len(leads),
            "min": None if not leads else min(leads),
            "max": None if not leads else max(leads),
        },
        "audit_timing_ms": {
            "case_mean_of_means": None if not timing else sum(timing) / len(timing),
            "worst_case_max": None if not timing_max else max(timing_max),
        },
        "invalid_case_ids": [r["case_id"] for r in rows if not r["valid"]],
        "cases": rows,
    }
    (root / "benchmark_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False))

    print("\n=== Phase C4.1 frozen-12 classification ===")
    print(f"valid cases: {len(valid_rows)}/{len(order)}; actual recovery positives={len(positives)} negatives={len(negatives)}")
    for d in by_streak:
        print(
            "streak>=%d: TP=%d FP=%d FN=%d TN=%d recall=%s specificity=%s accuracy=%s" % (
                d["streak_required"], d["tp"], d["fp"], d["fn"], d["tn"],
                "nan" if d["sensitivity_recall"] is None else f"{d['sensitivity_recall']:.3f}",
                "nan" if d["specificity"] is None else f"{d['specificity']:.3f}",
                "nan" if d["accuracy"] is None else f"{d['accuracy']:.3f}",
            )
        )
    print("positive cohort:", [r["case_id"] for r in positives])
    print("negative cohort:", [r["case_id"] for r in negatives])
    print("outputs:", root / "case_results.csv", root / "benchmark_summary.json")


if __name__ == "__main__":
    main()
