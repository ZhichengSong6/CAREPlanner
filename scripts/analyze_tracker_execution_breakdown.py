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


TOK = re.compile(r"([A-Za-z0-9_]+)=([^\\s]+)")


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
        }

    out = {
        "run_dir": run_dir,
        "execution_count": len(executions),
        "by_mode": by_mode,
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
