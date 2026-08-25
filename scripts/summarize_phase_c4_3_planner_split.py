#!/usr/bin/env python3
"""Summarize the C4.3 planner/controller-split smoke run."""

import csv
import glob
import json
import os
import re
import sys
from collections import Counter

TOKEN_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


def token_records(path):
    ans = []
    if not os.path.isfile(path):
        return ans
    try:
        with open(path, newline="", errors="replace") as fh:
            rd = csv.reader(fh)
            header = next(rd, [])
            if not header:
                return ans
            data_index = header.index("field.data") if "field.data" in header else 1
            for row in rd:
                if len(row) <= data_index:
                    continue
                message = ",".join(row[data_index:])
                tokens = dict(TOKEN_RE.findall(message))
                if tokens:
                    ans.append(tokens)
    except Exception:
        return []
    return ans


def csv_record_count(path):
    if not os.path.isfile(path):
        return 0
    try:
        with open(path, newline="", errors="replace") as fh:
            rd = csv.reader(fh)
            next(rd, None)
            return sum(1 for _ in rd)
    except Exception:
        return 0


def as_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def main():
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: summarize_phase_c4_3_planner_split.py CASE_ID OUT CONTROLLED_LOG")

    case_id, out, controlled_log = sys.argv[1:]
    selector = token_records(os.path.join(out, "selector_summary.csv"))
    broker = token_records(os.path.join(out, "broker_summary.csv"))
    waypoint = token_records(os.path.join(out, "waypoint_summary.csv"))
    guard = token_records(os.path.join(out, "predicted_vbc_recovery_guard.csv"))
    gate = token_records(os.path.join(out, "gate_summary.csv"))
    mpc = token_records(os.path.join(out, "mpc_summary.csv"))

    selected = [r for r in selector if r.get("has_violation") == "1"]
    source_counts = Counter(r.get("trajectory_source", "unknown") for r in selector)
    reason_counts = Counter(r.get("selection_reason", "none") for r in selected)
    control_mode_counts = Counter(r.get("control_mode", "unknown") for r in mpc)

    global_guard = [r for r in guard if r.get("mode") == "global_set"]
    global_violation_records = [r for r in global_guard if r.get("violation") == "1"]
    max_trigger = max([as_int(r.get("trigger_count_total")) for r in guard] or [0])
    max_clear = max([as_int(r.get("clear_count_total")) for r in guard] or [0])
    max_recovery_switch = max(
        [as_int(r.get("recovery_target_switch_count")) for r in broker] or [0])
    replan_requested = max([as_int(r.get("replan_count")) for r in broker] or [0])
    replan_installed = max([as_int(r.get("replan_count")) for r in gate] or [0])

    text = ""
    if os.path.isfile(controlled_log):
        text = open(controlled_log, errors="replace").read()
    recovery_entries = len(re.findall(
        r"(?:entering VISIBILITY RECOVERY|VBC RECOVERY TRIGGERED WHILE (?:UNSEEN|UNSAFE))",
        text))
    recovery_completes = len(re.findall(
        r"VISIBILITY RECOVERY EPISODE COMPLETE", text))

    payload = {
        "case_id": case_id,
        "architecture": "CAREPlanner trajectory planner -> TrajectoryExecutionManager low-level tracker",
        "planner_controller_split": True,
        "optimized_trajectory_topic": "/care_planner/mpc/predicted_trajectory",
        "actuator_command_topic": "/care_arm/arm_group_velocity_controller/command",
        "mpc_direct_command_topic": "/care_planner/mpc/internal_velocity_command_unused",
        "selector_records": len(selector),
        "selector_source_counts": dict(source_counts),
        "selection_reason_counts": dict(reason_counts),
        "selector_no_violation_records": sum(
            r.get("has_violation") == "0" for r in selector),
        "waypoint_generation_trace_count": len(glob.glob(
            os.path.join(out, "projector_traces", "vbc_visibility_waypoint_*.json"))),
        "waypoint_records": len(waypoint),
        "global_guard_records": len(global_guard),
        "global_guard_violation_records": len(global_violation_records),
        "guard_trigger_count_total": max_trigger,
        "guard_clear_count_total": max_clear,
        "recovery_entry_log_count": recovery_entries,
        "recovery_complete_log_count": recovery_completes,
        "recovery_target_switch_count": max_recovery_switch,
        "replan_requested_count": replan_requested,
        "replan_installed_count": replan_installed,
        "mpc_summary_records": len(mpc),
        "mpc_control_mode_counts": dict(control_mode_counts),
        "low_level_reference_state_records": csv_record_count(
            os.path.join(out, "low_level_reference_state.csv")),
        "gate_released_observed": any(r.get("released") == "1" for r in gate),
        "final_guard_routing": guard[-1].get("routing") if guard else None,
        "final_guard_status": guard[-1].get("status") if guard else None,
    }

    path = os.path.join(out, "c4_3_planner_split_summary.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
