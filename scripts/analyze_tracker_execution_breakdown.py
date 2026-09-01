#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict


def as_float(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def as_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default


TOK = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


def read_csv(path):
    """Read either flattened CSVs or rostopic-style recorder CSVs.

    CAREPlanner run CSVs are typically recorded as:
        %time,field.data
        <ns>,TRACKER active=1 complete=0 ...
    so the semantic fields live inside field.data as key=value tokens.
    Some derived CSVs are already flattened. Support both formats.
    """
    if not os.path.exists(path):
        return []

    with open(path, newline="", errors="replace") as f:
        rd = csv.reader(f)
        header = next(rd, [])
        if not header:
            return []

        if "field.data" in header:
            ti = header.index("%time") if "%time" in header else 0
            di = header.index("field.data")
            out = []
            for row in rd:
                if len(row) <= di:
                    continue
                payload = ",".join(row[di:])
                rec = dict(TOK.findall(payload))
                try:
                    rec["_t"] = float(row[ti]) / 1e9
                except Exception:
                    rec["_t"] = None
                if rec:
                    out.append(rec)
            return out

        f.seek(0)
        return list(csv.DictReader(f))


def percentile(xs, p):
    xs = sorted(x for x in xs if x is not None and math.isfinite(x))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def stats(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return None
    return {
        "count": len(xs),
        "min": min(xs),
        "median": percentile(xs, 0.5),
        "mean": sum(xs) / len(xs),
        "p95": percentile(xs, 0.95),
        "max": max(xs),
    }


def pearson(xs, ys):
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if x is not None and y is not None
        and math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return None
    xv = [p[0] for p in pairs]
    yv = [p[1] for p in pairs]
    mx = sum(xv) / len(xv)
    my = sum(yv) / len(yv)
    dx = [x - mx for x in xv]
    dy = [y - my for y in yv]
    denx = math.sqrt(sum(x * x for x in dx))
    deny = math.sqrt(sum(y * y for y in dy))
    if denx <= 1e-12 or deny <= 1e-12:
        return None
    return {
        "count": len(pairs),
        "r": sum(x * y for x, y in zip(dx, dy)) / (denx * deny),
    }


def mode_from_view(view):
    if view == "repair_prefix_brake_hold":
        return "REPAIR"
    if view == "probe_prefix_brake_hold":
        return "PROBE"
    if view == "full_horizon":
        return "NORMAL"
    return "UNKNOWN"


def main():
    ap = argparse.ArgumentParser(
        description="Per-execution tracking-error decomposition for CAREPlanner")
    ap.add_argument("--run-dir", required=True,
                    help="Directory containing tracker_summary.csv and verification_outcome.csv")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--csv-out", default=None)
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    tracker = read_csv(os.path.join(run_dir, "tracker_summary.csv"))
    ver = read_csv(os.path.join(run_dir, "verification_outcome.csv"))

    ver_by_stamp = {}
    for r in ver:
        stamp = as_int(r.get("execution_stamp_ns"), 0)
        if stamp <= 0:
            continue
        # Only committed executions own tracker stamps.
        if r.get("committed") != "1":
            continue
        ver_by_stamp[stamp] = {
            "verification_seq": as_int(r.get("seq"), 0),
            "verification_view": r.get("verification_view", "none"),
            "safety_gate": r.get("safety_gate", "none"),
            "result": r.get("result", "none"),
            "mode": mode_from_view(r.get("verification_view", "none")),
            "verification_t": as_float(r.get("_t")),
            "raw_candidate_age_s": as_float(r.get("raw_candidate_age_s")),
            "dispatch_suffix_phase_s": as_float(
                r.get("dispatch_suffix_phase_s")),
            "dispatch_suffix_start_shift_inf": as_float(
                r.get("dispatch_suffix_start_shift_inf")),
            "probe_rebase_enabled": as_int(r.get("probe_rebase_enabled"), 0),
            "probe_rebase_applied": as_int(r.get("probe_rebase_applied"), 0),
            "probe_rebase_clamped": as_int(r.get("probe_rebase_clamped"), 0),
            "probe_rebase_target_shift_inf": as_float(
                r.get("probe_rebase_target_shift_inf")),
            "probe_rebase_shift_inf": as_float(
                r.get("probe_rebase_shift_inf")),
            "probe_rebase_residual_inf": as_float(
                r.get("probe_rebase_residual_inf")),
            "probe_rebase_joint_state_age_ms": as_float(
                r.get("probe_rebase_joint_state_age_ms")),
            "probe_speed_max_before": as_float(
                r.get("probe_speed_max_before")),
            "probe_speed_max_after": as_float(
                r.get("probe_speed_max_after")),
            "probe_time_scale": as_float(r.get("probe_time_scale")),
            "probe_time_scale_clamped": as_int(
                r.get("probe_time_scale_clamped"), 0),
            "probe_effective_prefix_s": as_float(
                r.get("probe_effective_prefix_s")),
            "precommit_suffix_phase_s": as_float(
                r.get("precommit_suffix_phase_s")),
            "precommit_suffix_start_shift_inf": as_float(
                r.get("precommit_suffix_start_shift_inf")),
            "total_start_shift_inf": as_float(r.get("total_start_shift_inf")),
            "raw_start_measured_mismatch_inf_at_dispatch": as_float(
                r.get("raw_start_measured_mismatch_inf_at_dispatch")),
            "dispatch_start_measured_mismatch_inf": as_float(
                r.get("dispatch_start_measured_mismatch_inf")),
            "dispatch_joint_state_age_ms": as_float(
                r.get("dispatch_joint_state_age_ms")),
            "commit_start_measured_mismatch_inf": as_float(
                r.get("commit_start_measured_mismatch_inf")),
            "commit_joint_state_age_ms": as_float(
                r.get("commit_joint_state_age_ms")),
            "raw_candidate_duration_s": as_float(
                r.get("raw_candidate_duration_s")),
            "constructed_executable_duration_s": as_float(
                r.get("constructed_executable_duration_s")),
            "committed_duration_s": as_float(r.get("committed_duration_s")),
            "prefix_endpoint_max_abs_velocity": as_float(
                r.get("prefix_endpoint_max_abs_velocity")),
            "brake_duration_s": as_float(r.get("brake_duration_s")),
            "brake_displacement_inf": as_float(
                r.get("brake_displacement_inf")),
        }

    groups = defaultdict(list)
    for r in tracker:
        stamp = as_int(r.get("execution_stamp_ns"), 0)
        if stamp <= 0:
            continue
        err = as_float(r.get("tracking_error_inf"))
        if err is None:
            continue
        rr = dict(r)
        rr["_err"] = err
        rr["_phase"] = as_float(r.get("phase_s"))
        rr["_t_num"] = as_float(r.get("_t"))
        groups[stamp].append(rr)

    executions = []
    for stamp, rows in groups.items():
        rows.sort(key=lambda r: (
            float("inf") if r["_t_num"] is None else r["_t_num"],
            float("inf") if r["_phase"] is None else r["_phase"],
        ))
        errs = [r["_err"] for r in rows]
        initial = errs[0] if errs else None

        complete_rows = [
            r for r in rows
            if r.get("complete") == "1"
            and r.get("source") == "trajectory_complete_hold"
        ]
        terminal_row = complete_rows[-1] if complete_rows else rows[-1]
        terminal = terminal_row["_err"]
        duration_s = terminal_row["_phase"]
        if duration_s is None:
            phases = [r["_phase"] for r in rows if r["_phase"] is not None]
            duration_s = max(phases) if phases else None

        max_row = max(rows, key=lambda r: r["_err"])
        max_err = max_row["_err"]
        max_phase = max_row["_phase"]
        max_phase_fraction = (
            max_phase / duration_s
            if max_phase is not None and duration_s is not None and duration_s > 1e-12
            else None
        )

        meta = ver_by_stamp.get(stamp, {
            "verification_seq": 0,
            "verification_view": "unknown",
            "safety_gate": "unknown",
            "result": "unknown",
            "mode": "UNKNOWN",
            "verification_t": None,
            "raw_candidate_age_s": None,
            "dispatch_suffix_phase_s": None,
            "dispatch_suffix_start_shift_inf": None,
            "probe_rebase_enabled": 0,
            "probe_rebase_applied": 0,
            "probe_rebase_clamped": 0,
            "probe_rebase_target_shift_inf": None,
            "probe_rebase_shift_inf": None,
            "probe_rebase_residual_inf": None,
            "probe_rebase_joint_state_age_ms": None,
            "probe_speed_max_before": None,
            "probe_speed_max_after": None,
            "probe_time_scale": None,
            "probe_time_scale_clamped": 0,
            "probe_effective_prefix_s": None,
            "precommit_suffix_phase_s": None,
            "precommit_suffix_start_shift_inf": None,
            "total_start_shift_inf": None,
            "raw_start_measured_mismatch_inf_at_dispatch": None,
            "dispatch_start_measured_mismatch_inf": None,
            "dispatch_joint_state_age_ms": None,
            "commit_start_measured_mismatch_inf": None,
            "commit_joint_state_age_ms": None,
            "raw_candidate_duration_s": None,
            "constructed_executable_duration_s": None,
            "committed_duration_s": None,
            "prefix_endpoint_max_abs_velocity": None,
            "brake_duration_s": None,
            "brake_displacement_inf": None,
        })

        over_010 = sum(e > 0.10 for e in errs)
        over_025 = sum(e > 0.25 for e in errs)
        active_rows = [r for r in rows if r.get("source") == "active_trajectory"]
        active_errs = [r["_err"] for r in active_rows]

        executions.append({
            "execution_stamp_ns": stamp,
            "tracker_seq": as_int(terminal_row.get("seq"), 0),
            **meta,
            "sample_count": len(errs),
            "active_sample_count": len(active_errs),
            "complete": bool(complete_rows),
            "duration_s": duration_s,
            "initial_error_inf": initial,
            "mean_error_inf": sum(errs) / len(errs) if errs else None,
            "p95_error_inf": percentile(errs, 0.95),
            "max_error_inf": max_err,
            "terminal_error_inf": terminal,
            "active_mean_error_inf": (
                sum(active_errs) / len(active_errs) if active_errs else None),
            "active_p95_error_inf": percentile(active_errs, 0.95),
            "active_max_error_inf": max(active_errs) if active_errs else None,
            "max_error_phase_s": max_phase,
            "max_error_phase_fraction": max_phase_fraction,
            "fraction_error_gt_0p10": over_010 / len(errs) if errs else None,
            "fraction_error_gt_0p25": over_025 / len(errs) if errs else None,
            "initial_error_joint_index": as_int(
                rows[0].get("tracking_error_joint_index"), -1),
            "initial_error_joint_name": rows[0].get(
                "tracking_error_joint_name", "unknown"),
            "initial_error_q_ref": as_float(
                rows[0].get("tracking_error_q_ref")),
            "initial_error_q_measured": as_float(
                rows[0].get("tracking_error_q_measured")),
            "max_error_joint_index": as_int(
                max_row.get("tracking_error_joint_index"), -1),
            "max_error_joint_name": max_row.get(
                "tracking_error_joint_name", "unknown"),
            "max_error_q_ref": as_float(max_row.get("tracking_error_q_ref")),
            "max_error_q_measured": as_float(
                max_row.get("tracking_error_q_measured")),
            "terminal_error_joint_index": as_int(
                terminal_row.get("tracking_error_joint_index"), -1),
            "terminal_error_joint_name": terminal_row.get(
                "tracking_error_joint_name", "unknown"),
            "terminal_error_q_ref": as_float(
                terminal_row.get("tracking_error_q_ref")),
            "terminal_error_q_measured": as_float(
                terminal_row.get("tracking_error_q_measured")),
            "terminal_source": terminal_row.get("source"),
            "max_error_source": max_row.get("source"),
            "first_t": rows[0]["_t_num"],
            "last_t": rows[-1]["_t_num"],
        })

    executions.sort(key=lambda x: (
        float("inf") if x["first_t"] is None else x["first_t"],
        x["execution_stamp_ns"],
    ))

    by_mode = {}
    for mode in ("REPAIR", "PROBE", "NORMAL", "UNKNOWN"):
        exs = [e for e in executions if e["mode"] == mode]
        if not exs:
            continue
        by_mode[mode] = {
            "execution_count": len(exs),
            "completed_count": sum(bool(e["complete"]) for e in exs),
            "duration_s": stats([e["duration_s"] for e in exs]),
            "initial_error_inf": stats([e["initial_error_inf"] for e in exs]),
            "mean_error_inf": stats([e["mean_error_inf"] for e in exs]),
            "p95_error_inf": stats([e["p95_error_inf"] for e in exs]),
            "max_error_inf": stats([e["max_error_inf"] for e in exs]),
            "terminal_error_inf": stats([e["terminal_error_inf"] for e in exs]),
            "fraction_error_gt_0p10": stats([
                e["fraction_error_gt_0p10"] for e in exs]),
            "fraction_error_gt_0p25": stats([
                e["fraction_error_gt_0p25"] for e in exs]),
            "raw_candidate_age_s": stats([
                e.get("raw_candidate_age_s") for e in exs]),
            "dispatch_suffix_start_shift_inf": stats([
                e.get("dispatch_suffix_start_shift_inf") for e in exs]),
            "probe_rebase_target_shift_inf": stats([
                e.get("probe_rebase_target_shift_inf") for e in exs]),
            "probe_rebase_shift_inf": stats([
                e.get("probe_rebase_shift_inf") for e in exs]),
            "probe_rebase_residual_inf": stats([
                e.get("probe_rebase_residual_inf") for e in exs]),
            "probe_speed_max_before": stats([
                e.get("probe_speed_max_before") for e in exs]),
            "probe_speed_max_after": stats([
                e.get("probe_speed_max_after") for e in exs]),
            "probe_time_scale": stats([
                e.get("probe_time_scale") for e in exs]),
            "probe_effective_prefix_s": stats([
                e.get("probe_effective_prefix_s") for e in exs]),
            "precommit_suffix_start_shift_inf": stats([
                e.get("precommit_suffix_start_shift_inf") for e in exs]),
            "total_start_shift_inf": stats([
                e.get("total_start_shift_inf") for e in exs]),
            "raw_start_measured_mismatch_inf_at_dispatch": stats([
                e.get("raw_start_measured_mismatch_inf_at_dispatch")
                for e in exs]),
            "dispatch_start_measured_mismatch_inf": stats([
                e.get("dispatch_start_measured_mismatch_inf")
                for e in exs]),
            "commit_start_measured_mismatch_inf": stats([
                e.get("commit_start_measured_mismatch_inf")
                for e in exs]),
            "prefix_endpoint_max_abs_velocity": stats([
                e.get("prefix_endpoint_max_abs_velocity") for e in exs]),
            "brake_duration_s": stats([
                e.get("brake_duration_s") for e in exs]),
            "brake_displacement_inf": stats([
                e.get("brake_displacement_inf") for e in exs]),
        }

    probe_executions = [e for e in executions if e["mode"] == "PROBE"]
    probe_correlations = {
        "initial_error_vs_dispatch_suffix_shift": pearson(
            [e.get("initial_error_inf") for e in probe_executions],
            [e.get("dispatch_suffix_start_shift_inf")
             for e in probe_executions]),
        "initial_error_vs_probe_rebase_shift": pearson(
            [e.get("initial_error_inf") for e in probe_executions],
            [e.get("probe_rebase_shift_inf") for e in probe_executions]),
        "initial_error_vs_probe_rebase_residual": pearson(
            [e.get("initial_error_inf") for e in probe_executions],
            [e.get("probe_rebase_residual_inf") for e in probe_executions]),
        "initial_error_vs_precommit_suffix_shift": pearson(
            [e.get("initial_error_inf") for e in probe_executions],
            [e.get("precommit_suffix_start_shift_inf")
             for e in probe_executions]),
        "initial_error_vs_total_start_shift": pearson(
            [e.get("initial_error_inf") for e in probe_executions],
            [e.get("total_start_shift_inf") for e in probe_executions]),
        "initial_error_vs_raw_start_measured_mismatch_at_dispatch": pearson(
            [e.get("initial_error_inf") for e in probe_executions],
            [e.get("raw_start_measured_mismatch_inf_at_dispatch")
             for e in probe_executions]),
        "initial_error_vs_dispatch_start_measured_mismatch": pearson(
            [e.get("initial_error_inf") for e in probe_executions],
            [e.get("dispatch_start_measured_mismatch_inf")
             for e in probe_executions]),
        "initial_error_vs_commit_start_measured_mismatch": pearson(
            [e.get("initial_error_inf") for e in probe_executions],
            [e.get("commit_start_measured_mismatch_inf")
             for e in probe_executions]),
        "max_error_vs_probe_speed_after": pearson(
            [e.get("max_error_inf") for e in probe_executions],
            [e.get("probe_speed_max_after") for e in probe_executions]),
        "max_error_vs_probe_time_scale": pearson(
            [e.get("max_error_inf") for e in probe_executions],
            [e.get("probe_time_scale") for e in probe_executions]),
        "max_error_vs_prefix_endpoint_velocity": pearson(
            [e.get("max_error_inf") for e in probe_executions],
            [e.get("prefix_endpoint_max_abs_velocity")
             for e in probe_executions]),
        "max_error_vs_brake_duration": pearson(
            [e.get("max_error_inf") for e in probe_executions],
            [e.get("brake_duration_s") for e in probe_executions]),
        "max_error_vs_brake_displacement": pearson(
            [e.get("max_error_inf") for e in probe_executions],
            [e.get("brake_displacement_inf") for e in probe_executions]),
        "terminal_error_vs_brake_duration": pearson(
            [e.get("terminal_error_inf") for e in probe_executions],
            [e.get("brake_duration_s") for e in probe_executions]),
    }

    probe_start_continuity = {
        "execution_count": len(probe_executions),
        "all_dispatch_suffix_disabled": bool(probe_executions) and all(
            abs(float(e.get("dispatch_suffix_phase_s") or 0.0)) <= 1e-9
            for e in probe_executions),
        "all_precommit_suffix_disabled": bool(probe_executions) and all(
            abs(float(e.get("precommit_suffix_phase_s") or 0.0)) <= 1e-9
            for e in probe_executions),
        "all_rebase_applied": bool(probe_executions) and all(
            int(e.get("probe_rebase_applied") or 0) == 1
            for e in probe_executions),
        "max_rebase_residual_inf": max(
            [float(e.get("probe_rebase_residual_inf"))
             for e in probe_executions
             if e.get("probe_rebase_residual_inf") is not None],
            default=None),
        "max_commit_start_mismatch_inf": max(
            [float(e.get("commit_start_measured_mismatch_inf"))
             for e in probe_executions
             if e.get("commit_start_measured_mismatch_inf") is not None],
            default=None),
        "max_probe_speed_after": max(
            [float(e.get("probe_speed_max_after"))
             for e in probe_executions
             if e.get("probe_speed_max_after") is not None],
            default=None),
        "max_prefix_endpoint_velocity": max(
            [float(e.get("prefix_endpoint_max_abs_velocity"))
             for e in probe_executions
             if e.get("prefix_endpoint_max_abs_velocity") is not None],
            default=None),
    }
    if probe_start_continuity["max_commit_start_mismatch_inf"] is None:
        probe_start_continuity["pass_0p03"] = False
    else:
        probe_start_continuity["pass_0p03"] = (
            probe_start_continuity["all_dispatch_suffix_disabled"] and
            probe_start_continuity["all_precommit_suffix_disabled"] and
            probe_start_continuity["all_rebase_applied"] and
            probe_start_continuity["max_commit_start_mismatch_inf"] <= 0.03
        )

    out = {
        "run_dir": run_dir,
        "execution_count": len(executions),
        "by_mode": by_mode,
        "probe_correlations": probe_correlations,
        "probe_start_continuity": probe_start_continuity,
        "executions": executions,
    }

    json_out = args.json_out or os.path.join(
        run_dir, "tracker_execution_breakdown.json")
    csv_out = args.csv_out or os.path.join(
        run_dir, "tracker_execution_breakdown.csv")

    with open(json_out, "w") as f:
        json.dump(out, f, indent=2)

    fields = [
        "execution_stamp_ns", "tracker_seq", "mode", "verification_seq",
        "verification_view", "safety_gate", "result", "complete",
        "sample_count", "active_sample_count", "duration_s",
        "initial_error_inf", "mean_error_inf", "p95_error_inf",
        "max_error_inf", "terminal_error_inf", "active_mean_error_inf",
        "active_p95_error_inf", "active_max_error_inf",
        "max_error_phase_s", "max_error_phase_fraction",
        "fraction_error_gt_0p10", "fraction_error_gt_0p25",
        "raw_candidate_age_s", "dispatch_suffix_phase_s",
        "dispatch_suffix_start_shift_inf",
        "probe_rebase_enabled", "probe_rebase_applied",
        "probe_rebase_clamped", "probe_rebase_target_shift_inf",
        "probe_rebase_shift_inf", "probe_rebase_residual_inf",
        "probe_rebase_joint_state_age_ms",
        "probe_speed_max_before", "probe_speed_max_after",
        "probe_time_scale", "probe_time_scale_clamped",
        "probe_effective_prefix_s",
        "precommit_suffix_phase_s",
        "precommit_suffix_start_shift_inf", "total_start_shift_inf",
        "raw_start_measured_mismatch_inf_at_dispatch",
        "dispatch_start_measured_mismatch_inf", "dispatch_joint_state_age_ms",
        "commit_start_measured_mismatch_inf", "commit_joint_state_age_ms",
        "raw_candidate_duration_s", "constructed_executable_duration_s",
        "committed_duration_s", "prefix_endpoint_max_abs_velocity",
        "brake_duration_s", "brake_displacement_inf",
        "initial_error_joint_index", "initial_error_joint_name",
        "initial_error_q_ref", "initial_error_q_measured",
        "max_error_joint_index", "max_error_joint_name",
        "max_error_q_ref", "max_error_q_measured",
        "terminal_error_joint_index", "terminal_error_joint_name",
        "terminal_error_q_ref", "terminal_error_q_measured",
        "terminal_source", "max_error_source", "first_t", "last_t",
    ]
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in executions:
            w.writerow({k: e.get(k) for k in fields})

    print(json.dumps(out, indent=2))
    print(f"[tracker-breakdown] JSON: {json_out}")
    print(f"[tracker-breakdown] CSV:  {csv_out}")


if __name__ == "__main__":
    main()
