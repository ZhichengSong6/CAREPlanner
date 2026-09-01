#!/usr/bin/env python3
"""Diagnose PROBE_NORMAL recovery episodes from a CAREPlanner run.

This is an offline diagnostic only.  It does not change planner/controller
semantics.  The main question is why a PROBE episode did or did not reach
NORMAL after real visibility acquisition completed.
"""

import argparse
import csv
import json
import math
import os
import re
from collections import Counter

TOK = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


def as_float(v, default=math.nan):
    try:
        x = float(str(v).replace("ms", ""))
        return x if math.isfinite(x) else default
    except Exception:
        return default


def as_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default


def as_bool(v):
    if v in (True, 1, "1", "true", "True", "TRUE"):
        return True
    if v in (False, 0, "0", "false", "False", "FALSE"):
        return False
    return None


def read_token_csv(path):
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, newline="", errors="replace") as f:
        rd = csv.reader(f)
        header = next(rd, [])
        if not header:
            return rows
        time_i = header.index("%time") if "%time" in header else 0
        data_i = header.index("field.data") if "field.data" in header else 1
        for raw in rd:
            if len(raw) <= data_i:
                continue
            d = dict(TOK.findall(",".join(raw[data_i:])))
            if not d:
                continue
            d["_t"] = as_float(raw[time_i]) / 1e9 if len(raw) > time_i else math.nan
            rows.append(d)
    rows.sort(key=lambda r: as_float(r.get("_t")))
    return rows


def read_bool_csv(path):
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, newline="", errors="replace") as f:
        rd = csv.reader(f)
        header = next(rd, [])
        if not header:
            return rows
        time_i = header.index("%time") if "%time" in header else 0
        data_i = header.index("field.data") if "field.data" in header else 1
        for raw in rd:
            if len(raw) <= data_i:
                continue
            val = as_bool(raw[data_i].strip())
            if val is None:
                continue
            rows.append({
                "_t": as_float(raw[time_i]) / 1e9 if len(raw) > time_i else math.nan,
                "value": val,
            })
    rows.sort(key=lambda r: as_float(r.get("_t")))
    return rows


def subset(rows, t0, t1):
    out = []
    for r in rows:
        t = as_float(r.get("_t"))
        if not math.isfinite(t):
            continue
        if t >= t0 and (t1 is None or t < t1):
            out.append(r)
    return out


def compress_values(rows, key):
    seq = []
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        if not seq or seq[-1] != v:
            seq.append(v)
    return seq


def counter_value_before_or_at(rows, key, t):
    val = None
    for r in rows:
        rt = as_float(r.get("_t"))
        if not math.isfinite(rt):
            continue
        if rt > t:
            break
        if key in r:
            val = as_int(r.get(key))
    return val


def counter_delta(rows, key, t0, t1):
    before = counter_value_before_or_at(rows, key, t0 - 1e-9)
    end_t = t1 if t1 is not None else float("inf")
    after = None
    for r in rows:
        rt = as_float(r.get("_t"))
        if not math.isfinite(rt) or rt < t0:
            continue
        if rt >= end_t:
            break
        if key in r:
            after = as_int(r.get(key))
    if before is None:
        first = next((as_int(r.get(key)) for r in rows
                      if as_float(r.get("_t")) >= t0 and key in r), None)
        before = first if first is not None else 0
    if after is None:
        after = before
    return max(0, after - before)


def true_edge_count(rows, t0, t1):
    vals = subset(rows, t0, t1)
    count = 0
    prev = False
    for r in vals:
        cur = bool(r["value"])
        if cur and not prev:
            count += 1
        prev = cur
    return count


def last_time(*groups):
    vals = []
    for rows in groups:
        for r in rows:
            t = as_float(r.get("_t"))
            if math.isfinite(t):
                vals.append(t)
    return max(vals) if vals else None


def regime_segments(regime, run_end):
    if not regime:
        return []
    segments = []
    start = 0
    for i in range(1, len(regime) + 1):
        changed = i == len(regime) or regime[i].get("state") != regime[start].get("state")
        if not changed:
            continue
        s = regime[start]
        t0 = as_float(s.get("_t"))
        if i < len(regime):
            t1 = as_float(regime[i].get("_t"))
            exit_state = regime[i].get("state")
            exit_reason = regime[i].get("reason")
        else:
            t1 = run_end
            exit_state = None
            exit_reason = None
        prev_state = regime[start - 1].get("state") if start > 0 else None
        segments.append({
            "state": s.get("state"),
            "start_index": start,
            "end_index": i,
            "start_s": t0,
            "end_s": t1,
            "previous_state": prev_state,
            "entry_reason": s.get("reason"),
            "exit_state": exit_state,
            "exit_reason": exit_reason,
        })
        start = i
    return segments


def classify_episode(ep):
    if ep["exit_state"] == "NORMAL":
        return "recovered_to_normal"

    if ep["exit_state"] == "REPAIR":
        reason = (ep.get("exit_reason") or "").lower()
        if "infeasible" in reason:
            return "probe_task_infeasible_to_repair"
        if "uncertified" in reason:
            return "probe_task_uncertified_to_repair"
        if "visibility_obligation_ready" in reason or ep["blocker_rediscovery_delta"] > 0:
            return "probe_blocker_rediscovery_to_repair"
        if "unsafe" in reason:
            return "probe_safety_rejection_to_repair"
        return "probe_to_repair_other"

    if ep["exit_state"] is None:
        if ep["max_completed_prefix_streak"] >= ep["completed_prefixes_required"]:
            return "required_streak_reached_but_no_normal_transition"
        if ep["unmatched_verified_commit_count"] > 0:
            return "verified_commit_missing_matching_tracker_completion"
        if ep["single_flight_final_phase"] == "EXECUTING":
            return "single_flight_stuck_executing"
        if ep["probe_scp_failed_count"] > 0 and ep["probe_candidate_published_count"] == 0:
            return "probe_planner_failed_no_candidate"
        if ep["probe_candidate_published_count"] == 0:
            return "no_probe_candidate_generated"
        return "probe_streak_incomplete_at_run_end"

    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--case-id", default="unknown")
    ap.add_argument("--output-json", default="")
    ap.add_argument("--output-csv", default="")
    args = ap.parse_args()

    run = os.path.abspath(args.run_dir)
    regime = read_token_csv(os.path.join(run, "regime_summary.csv"))
    single = read_token_csv(os.path.join(run, "probe_single_flight_summary.csv"))
    verification = read_token_csv(os.path.join(run, "verification_outcome.csv"))
    tracker = read_token_csv(os.path.join(run, "tracker_summary.csv"))
    local = read_token_csv(os.path.join(run, "local_planner_summary.csv"))
    blocker = read_token_csv(os.path.join(run, "blocker_stack_summary.csv"))
    acquisition = read_token_csv(os.path.join(run, "visibility_acquisition_summary.csv"))
    task_infeasible = read_bool_csv(os.path.join(run, "task_infeasible.csv"))
    task_uncertified = read_bool_csv(os.path.join(run, "task_uncertified.csv"))

    run_end = last_time(
        regime, single, verification, tracker, local, blocker, acquisition,
        task_infeasible, task_uncertified)
    if run_end is None:
        raise RuntimeError("No timestamped diagnostic records found")

    segs = regime_segments(regime, run_end)
    probe_segs = [s for s in segs if s["state"] in ("PROBE_NORMAL", "PROBE")]

    episodes = []
    for idx, seg in enumerate(probe_segs, 1):
        t0, t1 = seg["start_s"], seg["end_s"]
        rr = subset(regime, t0, t1)
        sf = subset(single, t0, t1)
        vv = [
            r for r in subset(verification, t0, t1)
            if r.get("verification_view") == "probe_prefix_brake_hold"
        ]
        tt = subset(tracker, t0, t1)
        ll = [
            r for r in subset(local, t0, t1)
            if r.get("probe") == "1"
        ]
        bb = subset(blocker, t0, t1)
        aa = subset(acquisition, t0, t1)

        required = max(
            [as_int(r.get("probe_completed_prefixes_required"), 0) for r in rr] or [0])
        max_streak = max(
            [as_int(r.get("probe_completed_prefix_streak"), 0) for r in rr] or [0])

        verified_commits = [
            r for r in vv
            if r.get("result") == "safe" and as_bool(r.get("committed")) is True
        ]
        verified_stamps = {
            as_int(r.get("execution_stamp_ns"), 0) for r in verified_commits
            if as_int(r.get("execution_stamp_ns"), 0) > 0
        }
        tracker_completions = [
            r for r in tt
            if as_bool(r.get("complete")) is True
            and r.get("source") == "trajectory_complete_hold"
            and as_int(r.get("execution_stamp_ns"), 0) > 0
        ]
        tracker_stamps = {
            as_int(r.get("execution_stamp_ns"), 0) for r in tracker_completions
        }
        matched = sorted(verified_stamps & tracker_stamps)
        unmatched = sorted(verified_stamps - tracker_stamps)

        sf_phases = compress_values(sf, "phase")
        sf_final = sf[-1].get("phase") if sf else None

        local_events = Counter(r.get("event", "unknown") for r in ll)
        failed_statuses = Counter(
            r.get("status", "unknown") for r in ll
            if r.get("event") == "scp_failed")
        soft_diag = [
            r for r in ll
            if r.get("event") == "task_failure_slack_diagnostic"
        ]
        soft_diag_solved = [
            r for r in soft_diag if r.get("solved") == "1"
        ]
        soft_slacks = [
            as_float(r.get("max_slack")) for r in soft_diag
            if math.isfinite(as_float(r.get("max_slack")))
        ]
        task_ref_horizons = sorted({
            as_int(r.get("task_ref_horizon_steps"), 0) for r in ll
            if as_int(r.get("task_ref_horizon_steps"), 0) > 0
        })
        probe_hold_tail_values = sorted({
            as_int(r.get("probe_hold_tail"), 0) for r in ll
            if "probe_hold_tail" in r
        })

        remaining = [
            as_int(r.get("remaining_obligation_count"), -1) for r in aa
            if as_int(r.get("remaining_obligation_count"), -1) >= 0
        ]
        targets = compress_values(bb, "current_target_id")

        ep = {
            "episode": idx,
            "start_s": t0,
            "end_s": t1,
            "duration_s": (t1 - t0) if t1 is not None else None,
            "previous_state": seg["previous_state"],
            "entry_reason": seg["entry_reason"],
            "exit_state": seg["exit_state"],
            "exit_reason": seg["exit_reason"],
            "completed_prefixes_required": required,
            "max_completed_prefix_streak": max_streak,
            "probe_completed_execution_delta": counter_delta(
                regime, "probe_completed_execution_count", t0, t1),
            "probe_failure_delta": counter_delta(
                regime, "probe_failure_count", t0, t1),
            "blocker_rediscovery_delta": counter_delta(
                regime, "blocker_rediscovery_count", t0, t1),
            "task_infeasible_repair_delta": counter_delta(
                regime, "task_infeasible_repair_entry_count", t0, t1),
            "task_uncertified_repair_delta": counter_delta(
                regime, "task_uncertified_repair_entry_count", t0, t1),
            "task_infeasible_true_edges": true_edge_count(
                task_infeasible, t0, t1),
            "task_uncertified_true_edges": true_edge_count(
                task_uncertified, t0, t1),
            "single_flight_phase_sequence": sf_phases,
            "single_flight_final_phase": sf_final,
            "single_flight_execution_complete_delta": counter_delta(
                single, "execution_complete_count", t0, t1),
            "single_flight_verify_release_delta": counter_delta(
                single, "verify_release_count", t0, t1),
            "single_flight_drop_busy_delta": counter_delta(
                single, "drop_busy_count", t0, t1),
            "probe_verification_count": len(vv),
            "probe_verified_commit_count": len(verified_commits),
            "probe_verification_results": dict(Counter(
                r.get("result", "unknown") for r in vv)),
            "verified_execution_stamps": sorted(verified_stamps),
            "tracker_completion_count": len(tracker_completions),
            "tracker_completion_stamps": sorted(tracker_stamps),
            "matched_verified_tracker_stamps": matched,
            "unmatched_verified_commit_stamps": unmatched,
            "unmatched_verified_commit_count": len(unmatched),
            "probe_plan_started_count": local_events.get("plan_started", 0),
            "probe_scp_solved_count": local_events.get("scp_solved", 0),
            "probe_scp_failed_count": local_events.get("scp_failed", 0),
            "probe_candidate_published_count": local_events.get(
                "candidate_published", 0),
            "probe_planner_event_counts": dict(local_events),
            "probe_scp_failed_statuses": dict(failed_statuses),
            "task_ref_horizon_steps_seen": task_ref_horizons,
            "probe_hold_tail_values_seen": probe_hold_tail_values,
            "soft_slack_diagnostic_count": len(soft_diag),
            "soft_slack_diagnostic_solved_count": len(soft_diag_solved),
            "soft_slack_diagnostic_statuses": dict(Counter(
                r.get("status", "unknown") for r in soft_diag)),
            "soft_slack_required_max": max(soft_slacks) if soft_slacks else None,
            "soft_slack_required_mean": (
                sum(soft_slacks) / len(soft_slacks) if soft_slacks else None),
            "max_remaining_obligation_count": max(remaining) if remaining else 0,
            "last_remaining_obligation_count": remaining[-1] if remaining else None,
            "blocker_target_sequence": targets,
        }
        ep["diagnosis"] = classify_episode(ep)
        episodes.append(ep)

    diag_counts = Counter(ep["diagnosis"] for ep in episodes)
    recovered = sum(ep["exit_state"] == "NORMAL" for ep in episodes)
    to_repair = sum(ep["exit_state"] == "REPAIR" for ep in episodes)
    open_ended = sum(ep["exit_state"] is None for ep in episodes)
    long_open_probe = any(
        ep["exit_state"] is None
        and ep["duration_s"] is not None
        and ep["duration_s"] >= 10.0
        for ep in episodes
    )

    final_regime = regime[-1].get("state") if regime else None
    normal_entries = max(
        [as_int(r.get("normal_entry_count"), 0) for r in regime] or [0])
    repair_entries = max(
        [as_int(r.get("repair_entry_count"), 0) for r in regime] or [0])
    probe_entries = max(
        [as_int(r.get("probe_entry_count"), 0) for r in regime] or [0])

    summary = {
        "phase": "D.2.1",
        "case_id": args.case_id,
        "run_dir": run,
        "final_regime_state": final_regime,
        "normal_entry_count": normal_entries,
        "repair_entry_count": repair_entries,
        "probe_entry_count": probe_entries,
        "probe_episode_count": len(episodes),
        "probe_recovered_to_normal_count": recovered,
        "probe_returned_to_repair_count": to_repair,
        "probe_open_ended_count": open_ended,
        "diagnosis_counts": dict(diag_counts),
        "probe_livelock_signature": bool(
            episodes and final_regime in ("PROBE_NORMAL", "PROBE")
            and long_open_probe
        ),
        "episodes": episodes,
    }

    out_json = args.output_json or os.path.join(
        run, "probe_recovery_diagnostic.json")
    out_csv = args.output_csv or os.path.join(
        run, "probe_recovery_episodes.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)

    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    fields = [
        "episode", "start_s", "end_s", "duration_s", "previous_state",
        "entry_reason", "exit_state", "exit_reason", "diagnosis",
        "completed_prefixes_required", "max_completed_prefix_streak",
        "probe_completed_execution_delta", "probe_failure_delta",
        "blocker_rediscovery_delta", "task_infeasible_repair_delta",
        "task_uncertified_repair_delta", "task_infeasible_true_edges",
        "task_uncertified_true_edges", "single_flight_final_phase",
        "single_flight_execution_complete_delta",
        "single_flight_verify_release_delta", "single_flight_drop_busy_delta",
        "probe_verification_count", "probe_verified_commit_count",
        "tracker_completion_count", "unmatched_verified_commit_count",
        "probe_plan_started_count", "probe_scp_solved_count",
        "probe_scp_failed_count", "probe_candidate_published_count",
        "task_ref_horizon_steps_seen", "probe_hold_tail_values_seen",
        "soft_slack_diagnostic_count", "soft_slack_diagnostic_solved_count",
        "soft_slack_required_max", "soft_slack_required_mean",
        "max_remaining_obligation_count", "last_remaining_obligation_count",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for ep in episodes:
            w.writerow({k: ep.get(k) for k in fields})

    print("========== PROBE RECOVERY DIAGNOSTIC ==========")
    print("case_id:", args.case_id)
    print("final_regime_state:", final_regime)
    print("episodes:", len(episodes),
          "recovered:", recovered,
          "to_repair:", to_repair,
          "open:", open_ended)
    print("probe_livelock_signature:", summary["probe_livelock_signature"])
    for ep in episodes:
        print(
            "episode={episode} duration={duration_s:.3f}s "
            "streak={max_completed_prefix_streak}/{completed_prefixes_required} "
            "verified={probe_verified_commit_count} "
            "tracker_complete={tracker_completion_count} "
            "unmatched={unmatched_verified_commit_count} "
            "scp_failed={probe_scp_failed_count} "
            "ref_steps={task_ref_horizon_steps_seen} "
            "hold_tail={probe_hold_tail_values_seen} "
            "soft_slack_max={soft_slack_required_max} "
            "exit={exit_state} reason={exit_reason} diagnosis={diagnosis}".format(
                **ep))
    print("[JSON]", out_json)
    print("[CSV] ", out_csv)


if __name__ == "__main__":
    main()
