#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_ID="${CASE_ID:-case_003}"
if [[ -z "${RUN_ID:-}" ]]; then
  _RUN_STAMP="$(date +%Y%m%d-%H%M%S-%3N)"
  _GIT_SHORT="$(git -C "${REPO}" rev-parse --short=8 HEAD)"
  RUN_ID="${CASE_ID}_${_RUN_STAMP}_${_GIT_SHORT}"
fi
export CASE_ID RUN_ID
RUN_SECONDS="${RUN_SECONDS:-20.0}"

GPU_ENV="${GPU_ENV:-viscdf}"
GPU_DEVICE="${GPU_DEVICE:-cuda}"
CHECKPOINT="${CHECKPOINT:-${REPO}/src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt}"
GPU_SOCKET="${GPU_SOCKET:-/tmp/care_collision_cdf_gpu_c5_4.sock}"
LOCAL_SCP_PROXIMITY_MARGIN="${LOCAL_SCP_PROXIMITY_MARGIN:-0.025}"

CARE_WEIGHT="${CARE_WEIGHT:-3000.0}"
SAFETY_MARGIN="${SAFETY_MARGIN:-0.30}"
PREDICTION_TIMEOUT="${PREDICTION_TIMEOUT:-0.50}"

# Backward-compatible C4.8 repair-prefix verification controls.
# C5.4/G0 defaults remain full-horizon; C5.5 explicitly opts in.
REPAIR_PREFIX_VERIFY="${REPAIR_PREFIX_VERIFY:-0}"
REPAIR_PREFIX_S="${REPAIR_PREFIX_S:-0.15}"
REPAIR_BRAKE_DT_S="${REPAIR_BRAKE_DT_S:-0.05}"
REPAIR_HOLD_S="${REPAIR_HOLD_S:-0.10}"

ROOT_OUT="${ROOT_OUT:-${REPO}/outputs/phase_c5_4_local_sparse_scp/${RUN_ID}}"
ROOT_LOG="${ROOT_LOG:-${REPO}/logs/phase_c5_4_local_sparse_scp/${RUN_ID}}"
RUN_OUT="${ROOT_OUT}/run"
RUN_LOG="${ROOT_LOG}/run"
SELECTOR_JSONL="${ROOT_OUT}/local_scp_selector.jsonl"
ZIP_PATH="${ZIP_PATH:-${REPO}/CAREPlanner_C5_RESULT_${RUN_ID}.zip}"
SUMMARY_JSON="${ROOT_OUT}/c5_4_local_sparse_scp_summary.json"

cd "${REPO}"

echo "[C5.4] shell syntax preflight..."
bash -n scripts/run_and_pack_phase_c5_4_local_sparse_scp.sh
bash -n scripts/run_phase_c4_4_verified_regime_smoke.sh

echo "[C5.4] branch: $(git branch --show-current)"
echo "[C5.4] head:   $(git rev-parse HEAD)"
echo "[C5.4] case:   ${CASE_ID}"
echo "[C5.4] run:    ${RUN_ID}"
echo "[C5.8] architecture: Sparse SCP -> executable GCDF -> exact VBC -> single commit -> tracker"
echo "[C5.4] planner latency target: diagnostic first; NOT a 50 ms MPC deadline"
echo "[C5.27 PARAM] local+final GCDF proximity margin: ${LOCAL_SCP_PROXIMITY_MARGIN} m"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] signed CDF checkpoint not found: ${CHECKPOINT}"
  exit 2
fi

if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] conda.sh not found"
  exit 3
fi

catkin build care_confidence_map care_collision_cdf egocentric_arm_planner
source devel/setup.bash
python3 -m py_compile \
  src/egocentric_arm_planner/scripts/c4_4_verified_regime_manager_node.py \
  src/egocentric_arm_planner/scripts/optimized_trajectory_continuity_node.py \
  src/egocentric_arm_planner/scripts/probe_single_flight_gate_node.py \
  src/egocentric_arm_planner/scripts/execution_gcdf_measured_state_trajectory_node.py \
  src/egocentric_arm_planner/scripts/execution_gcdf_safety_monitor_node.py \
  scripts/wait_for_phase_d_goal.py \
  scripts/analyze_tracker_execution_breakdown.py \
  src/care_visibility_cdf/scripts/vbc_visibility_acquisition_impl.py \
  src/care_visibility_cdf/scripts/vbc_blocker_aware_acquisition_impl.py
python3 - <<'PY'
import xml.etree.ElementTree as ET
for path in (
    "src/egocentric_arm_planner/launch/phaseC4_4_verified_regime_planner.launch",
    "src/egocentric_arm_planner/launch/phaseC5_4_local_sparse_scp_planner.launch",
):
    ET.parse(path)
print("[C5.8] Python + launch XML preflight passed")
PY

# Fail before launching the persistent GPU worker if a previous ROS/Gazebo
# session is still alive. The common runner performs the same safety check,
# but doing it here makes the failure immediate and preserves a clear reason.
if timeout 2 rosnode list >/dev/null 2>&1; then
  echo "[ERROR] ROS master already running before C5.x launch."
  echo "[ERROR] Stop the previous ROS/Gazebo session before rerunning."
  exit 5
fi

rm -rf "${ROOT_OUT}" "${ROOT_LOG}"
rm -f "${ZIP_PATH}" "${GPU_SOCKET}"
mkdir -p "${ROOT_OUT}" "${ROOT_LOG}"

PIDS=()
kill_group() {
  local pid="${1:-}"
  [[ -z "${pid}" ]] && return 0
  if kill -0 "${pid}" 2>/dev/null; then
    kill -INT -- "-${pid}" 2>/dev/null || true
    sleep 0.20
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM -- "-${pid}" 2>/dev/null || true
    sleep 0.20
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill -KILL -- "-${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
}
cleanup() {
  local pid
  for pid in "${PIDS[@]:-}"; do kill_group "${pid}"; done
  PIDS=()
  rm -f "${GPU_SOCKET}"
}
pack_debug_bundle() {
  cd "${REPO}"

  # One run -> one uniquely named upload bundle.
  local stage
  stage="$(mktemp -d "${TMPDIR:-/tmp}/careplanner_c5_bundle.XXXXXX")"
  mkdir -p "${stage}/run" "${stage}/logs" "${stage}/root"

  if [[ -d "${RUN_OUT}" ]]; then
    cp -a "${RUN_OUT}/." "${stage}/run/"
  fi

  if [[ -f "${SUMMARY_JSON}" ]]; then
    cp -f "${SUMMARY_JSON}" "${stage}/root/"
  fi
  if [[ -f "${SELECTOR_JSONL}" ]]; then
    cp -f "${SELECTOR_JSONL}" "${stage}/root/"
  fi

  # Keep full key logs. Raw high-rate CSVs (joint states, tracker/actuator
  # streams, etc.) are already included via RUN_OUT above.
  local f src
  for f in waypoint_generator.log controlled.log common_runner.log \
           low_level_tracker.log gpu_worker.log; do
    src=""
    if [[ -f "${ROOT_LOG}/${f}" ]]; then
      src="${ROOT_LOG}/${f}"
    elif [[ -f "${RUN_LOG}/${f}" ]]; then
      src="${RUN_LOG}/${f}"
    fi
    if [[ -n "${src}" ]]; then
      cp -f "${src}" "${stage}/logs/${f}"
    fi
  done

  python3 - "${stage}" "${REPO}" "${RUN_ID}" "${CASE_ID}" <<'PY'
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys

stage, repo, run_id, scenario_case_id = sys.argv[1:5]
case_id = run_id
run = os.path.join(stage, "run")
logs = os.path.join(stage, "logs")
TOK = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")

def records(name):
    path = os.path.join(run, name)
    out = []
    if not os.path.isfile(path):
        return out
    try:
        with open(path, newline="", errors="replace") as f:
            rd = csv.reader(f)
            h = next(rd, [])
            if not h:
                return out
            ti = h.index("%time") if "%time" in h else 0
            di = h.index("field.data") if "field.data" in h else 1
            for row in rd:
                if len(row) <= di:
                    continue
                d = dict(TOK.findall(",".join(row[di:])))
                try:
                    d["_t"] = float(row[ti]) / 1e9
                except Exception:
                    d["_t"] = math.nan
                if d:
                    out.append(d)
    except Exception as exc:
        return [{"_parse_error": repr(exc)}]
    return out

def csv_data_row_count(name):
    path = os.path.join(run, name)
    if not os.path.isfile(path):
        return 0
    try:
        with open(path, newline="", errors="replace") as f:
            rd = csv.reader(f)
            next(rd, None)
            return sum(1 for row in rd if row)
    except Exception:
        return 0

def as_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default

def as_float(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None

def transition_sequence(rs, key):
    seq = []
    last = object()
    for r in rs:
        v = r.get(key)
        if v is None or v == last:
            continue
        seq.append({"t": r.get("_t"), key: v, "reason": r.get("reason", r.get("last_transition_reason"))})
        last = v
    return seq

names = [
    "visibility_acquisition_summary.csv",
    "regime_summary.csv",
    "probe_single_flight_summary.csv",
    "verification_outcome.csv",
    "tracker_summary.csv",
    "final_gcdf_selector_summary.csv",
    "final_gcdf_risk_summary.csv",
    "commit_summary.csv",
    "candidate_vbc_summary.csv",
    "execution_vbc_summary.csv",
    "local_planner_summary.csv",
    "local_cdf_selector_summary.csv",
    "task_obstacle_blocked.csv",
    "e3_summary.csv",
    "tof_fusion_summary.csv",
    "final_gcdf_recovery_event.csv",
    "execution_gcdf_selector_summary.csv",
    "execution_gcdf_safety_summary.csv",
    "execution_gcdf_hard_hold.csv",
    "blocker_stack_summary.csv",
    "waypoint_schedule_summary.csv",
]
R = {n: records(n) for n in names}

digest = {
    "case_id": case_id,
    "run_id": run_id,
    "scenario_case_id": scenario_case_id,
    "branch": subprocess.check_output(["git", "-C", repo, "branch", "--show-current"], text=True).strip(),
    "head": subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip(),
    "files": {},
}

for n, rs in R.items():
    digest["files"][n] = {
        "records": len(rs),
        "first": rs[0] if rs else None,
        "last": rs[-1] if rs else None,
    }

acq = R["visibility_acquisition_summary.csv"]
digest["acquisition"] = {
    "timer_exception_count_max": max([as_int(r.get("timer_exception_count")) for r in acq] or [0]),
    "started_any": any(r.get("started") == "1" for r in acq),
    "complete_any": any(r.get("complete") == "1" for r in acq),
    "complete_transitions": transition_sequence(acq, "complete"),
    "remaining_count_transitions": transition_sequence(acq, "remaining_obligation_count"),
    "last": acq[-1] if acq else None,
}

reg = R["regime_summary.csv"]
digest["regime"] = {
    "state_transitions": transition_sequence(reg, "state"),
    "last": reg[-1] if reg else None,
}

# C5.22: NORMAL candidate-unsafe hysteresis must not deadlock after the
# first rejected VBC candidate. The first unsafe should request a fresh plan;
# only a confirmed consecutive unsafe should transition to REPAIR.
normal_retry_rows = [
    r for r in reg
    if str(r.get("reason", "")).startswith("candidate_unsafe_retry_")
]
digest["c5_22_normal_unsafe_retry"] = {
    "normal_unsafe_retry_count": max(
        [as_int(r.get("normal_unsafe_retry_count"), 0) for r in reg] or [0]),
    "normal_unsafe_confirmed_repair_count": max(
        [as_int(r.get("normal_unsafe_confirmed_repair_count"), 0)
         for r in reg] or [0]),
    "max_candidate_unsafe_streak": max(
        [as_int(r.get("candidate_unsafe_streak"), 0) for r in reg] or [0]),
    "retry_events": [
        {
            "t": r.get("_t"),
            "reason": r.get("reason"),
            "last_candidate_seq": r.get("last_candidate_seq"),
            "candidate_unsafe_streak": r.get("candidate_unsafe_streak"),
        }
        for r in normal_retry_rows
    ],
    "final_state": reg[-1].get("state") if reg else None,
    "final_candidate_unsafe_streak": as_int(
        reg[-1].get("candidate_unsafe_streak"), 0) if reg else 0,
}

probe = R["probe_single_flight_summary.csv"]
digest["probe_single_flight"] = {
    "phase_transitions": transition_sequence(probe, "phase"),
    "last": probe[-1] if probe else None,
}

e3 = R["e3_summary.csv"]
local_rows = R["local_planner_summary.csv"]
recovery_events = R["final_gcdf_recovery_event.csv"]
obstacle_signals = R["task_obstacle_blocked.csv"]
commit_rows_e4 = R["commit_summary.csv"]

digest["phase_e4_semantics"] = {
    "e3_last": e3[-1] if e3 else None,
    "max_local_unknown_cdf_rows": max(
        [as_int(r.get("unknown_cdf_rows"), 0) for r in local_rows] or [0]),
    "max_local_occupied_cdf_rows": max(
        [as_int(r.get("occupied_cdf_rows"), 0) for r in local_rows] or [0]),
    "task_obstacle_blocked_signal_count":
        csv_data_row_count("task_obstacle_blocked.csv"),
    "gcdf_occupied_replan_count": max(
        [as_int(r.get("gcdf_occupied_replan_count"), 0) for r in reg] or [0]),
    "final_gcdf_blocker_classes": sorted(set(
        r.get("final_gcdf_blocker_class")
        for r in commit_rows_e4
        if r.get("final_gcdf_blocker_class") not in (None, "none"))),
    "max_final_gcdf_unsafe_unknown_count": max(
        [as_int(r.get("final_gcdf_unsafe_unknown_count"), 0)
         for r in commit_rows_e4] or [0]),
    "max_final_gcdf_unsafe_occupied_count": max(
        [as_int(r.get("final_gcdf_unsafe_occupied_count"), 0)
         for r in commit_rows_e4] or [0]),
    "final_gcdf_recovery_event_records":
        csv_data_row_count("final_gcdf_recovery_event.csv"),
    "last_regime": reg[-1] if reg else None,
}

e5_safety = R["execution_gcdf_safety_summary.csv"]
e5_selector = R["execution_gcdf_selector_summary.csv"]
tracker_e5 = R["tracker_summary.csv"]

e5_d = [
    as_float(r.get("d_min"))
    for r in e5_safety
    if as_float(r.get("d_min")) is not None
]
e5_age = [
    as_float(r.get("batch_age_s"))
    for r in e5_safety
    if as_float(r.get("batch_age_s")) is not None
]
digest["phase_e5_execution_gcdf"] = {
    "safety_records": len(e5_safety),
    "selector_records": len(e5_selector),
    "state_transitions": transition_sequence(e5_safety, "state"),
    "min_measured_gcdf_distance": min(e5_d) if e5_d else None,
    "max_batch_age_s": max(e5_age) if e5_age else None,
    "warning_event_count": max(
        [as_int(r.get("warning_event_count"), 0) for r in e5_safety] or [0]),
    "hard_event_count": max(
        [as_int(r.get("hard_event_count"), 0) for r in e5_safety] or [0]),
    "stale_event_count": max(
        [as_int(r.get("stale_event_count"), 0) for r in e5_safety] or [0]),
    "replan_count": max(
        [as_int(r.get("replan_count"), 0) for r in e5_safety] or [0]),
    "max_unknown_pairs": max(
        [as_int(r.get("unknown_pairs"), 0) for r in e5_safety] or [0]),
    "max_occupied_pairs": max(
        [as_int(r.get("occupied_pairs"), 0) for r in e5_safety] or [0]),
    "tracker_hard_hold_samples": sum(
        1 for r in tracker_e5
        if r.get("source") == "external_gcdf_hard_hold"),
    "hard_hold_topic_records":
        csv_data_row_count("execution_gcdf_hard_hold.csv"),
    "last": e5_safety[-1] if e5_safety else None,
}

reg_last = reg[-1] if reg else {}
commit_rows_for_c520 = R["commit_summary.csv"]
commit_last = commit_rows_for_c520[-1] if commit_rows_for_c520 else {}
digest["c5_20_probe_start_continuity"] = {
    "probe_rebase_count": as_int(
        commit_last.get("probe_rebase_count"), 0),
    "probe_rebase_clamp_count": as_int(
        commit_last.get("probe_rebase_clamp_count"), 0),
    "probe_start_continuity_reject_count": as_int(
        commit_last.get("probe_start_continuity_reject_count"), 0),
    "last_probe_rebase_shift_inf": as_float(
        commit_last.get("probe_rebase_shift_inf")),
    "last_probe_rebase_target_shift_inf": as_float(
        commit_last.get("probe_rebase_target_shift_inf")),
    "last_probe_rebase_residual_inf": as_float(
        commit_last.get("probe_rebase_residual_inf")),
    "last_probe_commit_start_mismatch_inf": as_float(
        commit_last.get("probe_commit_start_mismatch_inf")),
    "probe_speed_scale_count": as_int(
        commit_last.get("probe_speed_scale_count"), 0),
    "probe_speed_scale_clamp_count": as_int(
        commit_last.get("probe_speed_scale_clamp_count"), 0),
    "last_probe_speed_time_scale": as_float(
        commit_last.get("probe_speed_time_scale")),
    "last_probe_speed_max_before": as_float(
        commit_last.get("probe_speed_max_before")),
    "last_probe_speed_max_after": as_float(
        commit_last.get("probe_speed_max_after")),
    "last_probe_effective_prefix_s": as_float(
        commit_last.get("probe_effective_prefix_s")),
}

digest["probe_decision_single_flight"] = {
    "phase": reg_last.get("probe_single_flight_phase"),
    "execution_stamp_ns": reg_last.get(
        "probe_single_flight_execution_stamp_ns"),
    "reason": reg_last.get("probe_single_flight_reason"),
    "suppressed_busy_count": as_int(
        reg_last.get("probe_failure_suppressed_busy_count"), 0),
    "suppressed_infeasible_count": as_int(
        reg_last.get("probe_infeasible_suppressed_busy_count"), 0),
    "suppressed_uncertified_count": as_int(
        reg_last.get("probe_uncertified_suppressed_busy_count"), 0),
    "candidate_drop_busy_count": (
        as_int(probe[-1].get("drop_busy_count"), 0) if probe else 0),
    "execution_complete_count": (
        as_int(probe[-1].get("execution_complete_count"), 0)
        if probe else 0),
}

ver = R["verification_outcome.csv"]
probe_ver = []
for r in ver:
    if r.get("verification_view") == "probe_prefix_brake_hold":
        probe_ver.append({
            "t": r.get("_t"),
            "seq": r.get("seq"),
            "result": r.get("result"),
            "committed": r.get("committed"),
            "safety_gate": r.get("safety_gate"),
            "execution_stamp_ns": r.get("execution_stamp_ns"),
            "raw_candidate_age_s": r.get("raw_candidate_age_s"),
            "dispatch_suffix_phase_s": r.get("dispatch_suffix_phase_s"),
            "dispatch_suffix_start_shift_inf": r.get(
                "dispatch_suffix_start_shift_inf"),
            "probe_rebase_enabled": r.get("probe_rebase_enabled"),
            "probe_rebase_applied": r.get("probe_rebase_applied"),
            "probe_rebase_clamped": r.get("probe_rebase_clamped"),
            "probe_rebase_target_shift_inf": r.get(
                "probe_rebase_target_shift_inf"),
            "probe_rebase_shift_inf": r.get("probe_rebase_shift_inf"),
            "probe_rebase_residual_inf": r.get("probe_rebase_residual_inf"),
            "probe_rebase_joint_state_age_ms": r.get(
                "probe_rebase_joint_state_age_ms"),
            "probe_speed_max_before": r.get("probe_speed_max_before"),
            "probe_speed_max_after": r.get("probe_speed_max_after"),
            "probe_time_scale": r.get("probe_time_scale"),
            "probe_time_scale_clamped": r.get(
                "probe_time_scale_clamped"),
            "probe_effective_prefix_s": r.get(
                "probe_effective_prefix_s"),
            "precommit_suffix_phase_s": r.get("precommit_suffix_phase_s"),
            "precommit_suffix_start_shift_inf": r.get(
                "precommit_suffix_start_shift_inf"),
            "total_start_shift_inf": r.get("total_start_shift_inf"),
            "raw_start_measured_mismatch_inf_at_dispatch": r.get(
                "raw_start_measured_mismatch_inf_at_dispatch"),
            "dispatch_start_measured_mismatch_inf": r.get(
                "dispatch_start_measured_mismatch_inf"),
            "dispatch_joint_state_age_ms": r.get(
                "dispatch_joint_state_age_ms"),
            "commit_start_measured_mismatch_inf": r.get(
                "commit_start_measured_mismatch_inf"),
            "commit_joint_state_age_ms": r.get(
                "commit_joint_state_age_ms"),
            "prefix_endpoint_max_abs_velocity": r.get(
                "prefix_endpoint_max_abs_velocity"),
            "brake_duration_s": r.get("brake_duration_s"),
            "brake_displacement_inf": r.get("brake_displacement_inf"),
            "constructed_executable_duration_s": r.get(
                "constructed_executable_duration_s"),
            "committed_duration_s": r.get("committed_duration_s"),
        })
digest["probe_verifications"] = probe_ver

trk = R["tracker_summary.csv"]
trk_done = []
for r in trk:
    if r.get("complete") == "1" and r.get("source") == "trajectory_complete_hold":
        trk_done.append({
            "t": r.get("_t"),
            "seq": r.get("seq"),
            "execution_stamp_ns": r.get("execution_stamp_ns"),
            "phase_s": r.get("phase_s"),
        })
digest["tracker_completions"] = trk_done

breakdown_path = os.path.join(run, "tracker_execution_breakdown.json")
if os.path.isfile(breakdown_path):
    try:
        with open(breakdown_path) as f:
            tracker_breakdown = json.load(f)
        executions = tracker_breakdown.get("executions", [])
        worst_by_max = sorted(
            executions,
            key=lambda e: float(e.get("max_error_inf") or -1.0),
            reverse=True)[:8]
        worst_by_terminal = sorted(
            executions,
            key=lambda e: float(e.get("terminal_error_inf") or -1.0),
            reverse=True)[:8]
        compact_keys = (
            "execution_stamp_ns", "tracker_seq", "mode", "verification_seq",
            "duration_s", "initial_error_inf", "mean_error_inf",
            "p95_error_inf", "max_error_inf", "terminal_error_inf",
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
            "dispatch_start_measured_mismatch_inf",
            "dispatch_joint_state_age_ms",
            "commit_start_measured_mismatch_inf",
            "commit_joint_state_age_ms",
            "prefix_endpoint_max_abs_velocity", "brake_duration_s",
            "brake_displacement_inf", "constructed_executable_duration_s",
            "committed_duration_s",
            "initial_error_joint_name", "initial_error_q_ref",
            "initial_error_q_measured", "max_error_joint_name",
            "max_error_q_ref", "max_error_q_measured",
            "terminal_error_joint_name", "terminal_error_q_ref",
            "terminal_error_q_measured",
            "max_error_source", "terminal_source")
        digest["tracker_error_decomposition"] = {
            "execution_count": tracker_breakdown.get("execution_count", 0),
            "by_mode": tracker_breakdown.get("by_mode", {}),
            "probe_correlations": tracker_breakdown.get(
                "probe_correlations", {}),
            "probe_start_continuity": tracker_breakdown.get(
                "probe_start_continuity", {}),
            "worst_by_max_error": [
                {k: e.get(k) for k in compact_keys} for e in worst_by_max
            ],
            "worst_by_terminal_error": [
                {k: e.get(k) for k in compact_keys}
                for e in worst_by_terminal
            ],
        }
    except Exception as exc:
        digest["tracker_error_decomposition"] = {
            "error": "failed_to_load_breakdown: {}".format(exc)}
else:
    digest["tracker_error_decomposition"] = {
        "error": "tracker_execution_breakdown.json missing"}

probe_stamps = {
    r["execution_stamp_ns"] for r in probe_ver
    if r.get("result") == "safe" and r.get("committed") == "1"
       and r.get("execution_stamp_ns") not in (None, "0")
}
tracker_stamps = {
    r["execution_stamp_ns"] for r in trk_done
    if r.get("execution_stamp_ns") not in (None, "0")
}
digest["matched_probe_completion_stamps"] = sorted(probe_stamps & tracker_stamps)

exe = R["execution_vbc_summary.csv"]
digest["execution_vbc"] = {
    "records": len(exe),
    "unsafe_records": sum(r.get("has_violation") == "1" for r in exe),
    "last": exe[-1] if exe else None,
}

def timing_stats(values):
    xs = sorted(x for x in values if x is not None and math.isfinite(x))
    if not xs:
        return None
    n = len(xs)
    def pct(p):
        if n == 1:
            return xs[0]
        k = (n - 1) * p
        lo = int(math.floor(k))
        hi = int(math.ceil(k))
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - k) + xs[hi] * (k - lo)
    return {
        "count": n,
        "min": xs[0],
        "median": pct(0.5),
        "mean": sum(xs) / n,
        "p95": pct(0.95),
        "max": xs[-1],
    }

local = R["local_planner_summary.csv"]
digest["task_failure_slack_diagnostics"] = [
    {
        "t": r.get("_t"),
        "plan_seq": r.get("plan_seq"),
        "probe": r.get("probe"),
        "solved": r.get("solved"),
        "status": r.get("status"),
        "cdf_rows": r.get("cdf_rows"),
        "min_d": r.get("min_d"),
        "max_slack": r.get("max_slack"),
        "mean_slack": r.get("mean_slack"),
        "primal": r.get("primal"),
        "dual": r.get("dual"),
    }
    for r in local
    if r.get("event") == "task_failure_slack_diagnostic"
]

commit_rows = R["commit_summary.csv"]
candidate_plans = [r for r in local if r.get("event") == "candidate_published"]

# commit_summary is a high-rate state stream; the timing fields persist after a
# verdict. Collapse it to one record per completed verification seq before
# computing statistics, otherwise one candidate appears hundreds of times.
timing_commit_by_seq = {}
for r in commit_rows:
    seq = as_int(r.get("last_verification_seq"), 0)
    if seq <= 0:
        continue
    timing_commit_by_seq.setdefault(seq, r)
timing_commit_rows = list(timing_commit_by_seq.values())

digest["timing"] = {
    "local_plan_total_ms": timing_stats([
        as_float(r.get("total_plan_ms")) for r in candidate_plans
    ]),
    "piqp_final_solve_ms": timing_stats([
        as_float(r.get("solve_ms")) for r in candidate_plans
    ]),
    "raw_to_safety_dispatch_ms": timing_stats([
        as_float(r.get("raw_to_safety_dispatch_ms")) for r in timing_commit_rows
    ]),
    "final_gcdf_roundtrip_ms": timing_stats([
        as_float(r.get("final_gcdf_roundtrip_ms")) for r in timing_commit_rows
    ]),
    "exact_vbc_roundtrip_ms": timing_stats([
        as_float(r.get("exact_vbc_roundtrip_ms")) for r in timing_commit_rows
    ]),
    "candidate_total_safety_pipeline_ms": timing_stats([
        as_float(r.get("candidate_total_safety_pipeline_ms"))
        for r in timing_commit_rows
    ]),
}

# Keep only unique tracker completions when estimating physical execution time.
tracker_exec_by_stamp = {}
for r in R["tracker_summary.csv"]:
    if r.get("complete") != "1" or r.get("source") != "trajectory_complete_hold":
        continue
    stamp = r.get("execution_stamp_ns")
    if stamp in (None, "0"):
        continue
    phase = as_float(r.get("phase_s"))
    if phase is not None:
        tracker_exec_by_stamp[stamp] = phase * 1000.0
digest["timing"]["tracker_execution_ms"] = timing_stats(
    list(tracker_exec_by_stamp.values()))

schedule_rows = R["waypoint_schedule_summary.csv"]
blocker_rows_for_timing = R["blocker_stack_summary.csv"]
qvis_last_values = [
    as_float(r.get("q_vis_generation_last_ms")) for r in schedule_rows
]
qvis_max_values = [
    as_float(r.get("q_vis_generation_max_ms")) for r in schedule_rows
]
if not any(v is not None for v in qvis_last_values):
    qvis_last_values = [
        as_float(r.get("q_vis_generation_last_ms"))
        for r in blocker_rows_for_timing
    ]
if not any(v is not None for v in qvis_max_values):
    qvis_max_values = [
        as_float(r.get("q_vis_generation_max_ms"))
        for r in blocker_rows_for_timing
    ]
digest["timing"]["q_vis_generation_last_ms"] = timing_stats(
    qvis_last_values)
digest["timing"]["q_vis_generation_max_ms_observed"] = timing_stats(
    qvis_max_values)

# Acquisition elapsed is physical sensing/execution latency, not pure compute.
# Record episode durations separately so they are never confused with planner
# or verifier compute time.
acq_complete_times = [
    r.get("_t") for r in acq if r.get("complete") == "1"
]
acq_start_times = [
    r.get("_t") for r in acq if r.get("started") == "1"
]
if acq_start_times and acq_complete_times:
    t0 = min(x for x in acq_start_times if isinstance(x, (int, float)))
    t1_candidates = [
        x for x in acq_complete_times
        if isinstance(x, (int, float)) and x >= t0
    ]
    digest["timing"]["first_acquisition_episode_elapsed_ms"] = (
        1000.0 * (min(t1_candidates) - t0)
        if t1_candidates else None)
else:
    digest["timing"]["first_acquisition_episode_elapsed_ms"] = None

# Per-latest-candidate end-to-end compute estimate excludes physical tracker
# execution and uses only instrumented online compute/transport stages.
latest_plan_ms = None
if candidate_plans:
    latest_plan_ms = as_float(candidate_plans[-1].get("total_plan_ms"))
latest_safety_ms = None
if commit_rows:
    latest_safety_ms = as_float(
        commit_rows[-1].get("candidate_total_safety_pipeline_ms"))
digest["timing"]["latest_online_compute_end_to_end_ms"] = (
    latest_plan_ms + latest_safety_ms
    if latest_plan_ms is not None and latest_safety_ms is not None
    else None)


blocker_rows = R["blocker_stack_summary.csv"]
digest["c5_12_recovery"] = {
    "final_gcdf_recovery_event_count": (
        as_int(commit_rows[-1].get("final_gcdf_recovery_event_count"))
        if commit_rows else 0),
    "final_gcdf_recovery_event_drop_count": (
        as_int(commit_rows[-1].get("final_gcdf_recovery_event_drop_count"))
        if commit_rows else 0),
    "last_final_gcdf_recovery_seq": (
        as_int(commit_rows[-1].get("last_final_gcdf_recovery_seq"))
        if commit_rows else 0),
    "last_final_gcdf_recovery_timestep": (
        as_int(commit_rows[-1].get("last_final_gcdf_recovery_timestep"), -1)
        if commit_rows else -1),
    "last_final_gcdf_recovery_point_count": (
        as_int(commit_rows[-1].get("last_final_gcdf_recovery_point_count"))
        if commit_rows else 0),
    "gcdf_recovery_generated_count": (
        as_int(blocker_rows[-1].get("gcdf_recovery_generated_count"))
        if blocker_rows else 0),
    "gcdf_recovery_drop_count": (
        as_int(blocker_rows[-1].get("gcdf_recovery_drop_count"))
        if blocker_rows else 0),
    "gcdf_recovery_processed_seq": (
        as_int(blocker_rows[-1].get("gcdf_recovery_processed_seq"))
        if blocker_rows else 0),
    "gcdf_recovery_reason": (
        blocker_rows[-1].get("gcdf_recovery_reason")
        if blocker_rows else None),
    "gcdf_recovery_event_stamp": (
        blocker_rows[-1].get("gcdf_recovery_event_stamp")
        if blocker_rows else None),
    "gcdf_recovery_trajectory_stamp": (
        blocker_rows[-1].get("gcdf_recovery_trajectory_stamp")
        if blocker_rows else None),
    "gcdf_recovery_cache_size": (
        as_int(blocker_rows[-1].get("gcdf_recovery_cache_size"))
        if blocker_rows else 0),
    "gcdf_recovery_cache_hit_count": (
        as_int(blocker_rows[-1].get("gcdf_recovery_cache_hit_count"))
        if blocker_rows else 0),
    "gcdf_recovery_cache_miss_count": (
        as_int(blocker_rows[-1].get("gcdf_recovery_cache_miss_count"))
        if blocker_rows else 0),
}

# The key C5.12 invariant is checked on observed regime records rather than
# inferred from final state: a transition into REPAIR from PROBE must name a
# visibility-obligation-ready reason (direct transition is only allowed when
# waypoint_active was already true).
reg_rows = R["regime_summary.csv"]
probe_repair_entries = []
prev_state = None
for r in reg_rows:
    state = r.get("state")
    if state == "REPAIR" and prev_state == "PROBE_NORMAL":
        probe_repair_entries.append({
            "t": r.get("_t"),
            "reason": r.get("reason"),
            "visibility_waypoint_active": r.get("visibility_waypoint_active"),
            "blocker_rediscovery_pending": r.get("blocker_rediscovery_pending"),
        })
    if state is not None:
        prev_state = state
digest["c5_12_recovery"]["probe_to_repair_entries"] = probe_repair_entries
digest["c5_12_recovery"]["probe_to_repair_invariant_pass"] = all(
    (e.get("visibility_waypoint_active") == "1" or
     "visibility_obligation_ready" in str(e.get("reason")))
    for e in probe_repair_entries
)

# Collect exception/error evidence from the compact log tails.
err_pat = re.compile(r"(Traceback|Exception|ERROR|FATAL|timer callback exception|stamp_miss|timeout)", re.I)
errors = {}
for name in sorted(os.listdir(logs)) if os.path.isdir(logs) else []:
    path = os.path.join(logs, name)
    hits = []
    with open(path, errors="replace") as f:
        for i, line in enumerate(f, 1):
            if err_pat.search(line):
                hits.append({"line": i, "text": line.rstrip()[:1000]})
    if hits:
        errors[name] = hits[-200:]
digest["log_error_evidence"] = errors

with open(os.path.join(stage, "c5_analysis_digest.json"), "w") as f:
    json.dump(digest, f, indent=2, allow_nan=True)

manifest_lines = [
    "CAREPlanner compact C5 analysis bundle",
    f"case_id={case_id}",
    f"branch={digest['branch']}",
    f"head={digest['head']}",
    "",
    "FILES:",
]
for root, _, files in os.walk(stage):
    for name in sorted(files):
        p = os.path.join(root, name)
        if name == "MANIFEST.txt":
            continue
        rel = os.path.relpath(p, stage)
        h = hashlib.sha256()
        with open(p, "rb") as fp:
            for chunk in iter(lambda: fp.read(1024 * 1024), b""):
                h.update(chunk)
        manifest_lines.append(f"{rel}\t{os.path.getsize(p)}\tsha256={h.hexdigest()}")
with open(os.path.join(stage, "MANIFEST.txt"), "w") as f:
    f.write("\n".join(manifest_lines) + "\n")
PY


  # MANIFEST.txt inside the archive contains per-file hashes; no sidecar file.
  rm -f "${ZIP_PATH}"
  python3 - "${stage}" "${ZIP_PATH}" <<'PY'
import os
import sys
import zipfile

stage, dst = sys.argv[1:3]
with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for root, dirs, files in os.walk(stage):
        dirs.sort()
        files.sort()
        for name in files:
            src = os.path.join(root, name)
            arc = os.path.relpath(src, stage).replace(os.sep, "/")
            z.write(src, arc)
print(dst)
PY

  echo "[C5.x SINGLE UPLOAD ZIP] ${ZIP_PATH}"
  ls -lh "${ZIP_PATH}"

  # Chat attachment mounting is outside the runner's control.  Always mirror
  # the machine-readable digest to stdout so a run remains analyzable from the
  # terminal transcript even if the ZIP is not mounted by the chat platform.
  if [[ -f "${stage}/c5_analysis_digest.json" ]]; then
    echo ""
    echo "========== C5 ANALYSIS DIGEST BEGIN =========="
    cat "${stage}/c5_analysis_digest.json"
    echo ""
    echo "========== C5 ANALYSIS DIGEST END =========="
  fi

  rm -rf "${stage}"
}
trap 'rc=$?; cleanup || true; if [[ -d "${ROOT_OUT}" || -d "${ROOT_LOG}" ]]; then pack_debug_bundle || true; fi; exit $rc' EXIT

echo "[C5.4] CUDA preflight..."
bash -lc "
  set -e
  source '${CONDA_SH}'
  conda activate '${GPU_ENV}'
  cd '${REPO}'
  python - <<'PY'
import torch
print('[preflight] torch:', torch.__version__)
print('[preflight] cuda runtime:', torch.version.cuda)
print('[preflight] cuda available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('[ERROR] GPU_ENV has no usable CUDA PyTorch')
print('[preflight] GPU:', torch.cuda.get_device_name(0))
PY
"

# The neural signed-CDF oracle stays as a persistent GPU service.  The local
# planner performs SCP synchronously at the algorithmic level, while transport
# remains decoupled so GPU/C++ geometry code stays reusable.
setsid bash -lc "
  set -e
  source '${CONDA_SH}'
  conda activate '${GPU_ENV}'
  cd '${REPO}'
  exec python -u src/care_collision_cdf/scripts/collision_cdf_gpu_worker.py \
    --checkpoint '${CHECKPOINT}' \
    --activation gelu \
    --device '${GPU_DEVICE}' \
    --socket '${GPU_SOCKET}' \
    --max-pairs 8000 \
    --warmup-pairs 2048
" >"${ROOT_LOG}/gpu_worker.log" 2>&1 &
PIDS+=("$!")

READY=0
for _ in $(seq 1 200); do
  if [[ -S "${GPU_SOCKET}" ]]; then READY=1; break; fi
  if ! kill -0 "${PIDS[0]}" 2>/dev/null; then break; fi
  sleep 0.05
done
if [[ "${READY}" != "1" ]]; then
  echo "[ERROR] GPU worker socket did not become ready"
  tail -n 160 "${ROOT_LOG}/gpu_worker.log" || true
  exit 4
fi

# C5.4-specific ROS diagnostics are recorded by the common runner itself.
# Do not start recorders here: that runner clears RUN_OUT at startup, which
# unlinked the first C5.4 trial's early recorder files.

# C5.4 defaults to full-horizon verification, while later integrations may
# opt into the existing C4.8 safe-prefix verifier without changing the planner.
USE_LOCAL_SPARSE_SCP=true \
FINAL_EXECUTABLE_GCDF_ENABLED=true \
COMMITTED_CONTINUATION_ENABLED=false \
EXECUTION_AUDIT_STREAM_ENABLED=true \
EXECUTION_VBC_TRAJECTORY_TOPIC="/care_planner/execution/audit_trajectory" \
PROBE_SINGLE_FLIGHT_ENABLED=true \
PROBE_SINGLE_FLIGHT_TOPIC="/care_planner/local_planner/candidate_trajectory_single_flight" \
LOCAL_SCP_GPU_SOCKET="${GPU_SOCKET}" \
LOCAL_SCP_SELECTOR_JSONL="${SELECTOR_JSONL}" \
LOCAL_SCP_PROXIMITY_MARGIN="${LOCAL_SCP_PROXIMITY_MARGIN}" \
LOCAL_SCP_CANDIDATE_TOPIC="/care_planner/local_planner/candidate_trajectory" \
LOCAL_SCP_SUMMARY_TOPIC="/care_planner/local_planner/summary" \
LOCAL_SCP_REPLAN_TOPIC="/care_planner/local_planner/replan_request" \
REPAIR_PREFIX_VERIFY="${REPAIR_PREFIX_VERIFY}" \
REPAIR_PREFIX_S="${REPAIR_PREFIX_S}" \
REPAIR_BRAKE_DT_S="${REPAIR_BRAKE_DT_S}" \
REPAIR_HOLD_S="${REPAIR_HOLD_S}" \
TRAJECTORY_RISK_INPUT_TOPIC="/care_planner/local_planner/candidate_trajectory" \
REGION_SCHEDULE_MODE="blocker_aware_acquisition" \
CASE_ID="${CASE_ID}" RUN_SECONDS="${RUN_SECONDS}" CARE_WEIGHT="${CARE_WEIGHT}" \
SAFETY_MARGIN="${SAFETY_MARGIN}" PREDICTION_TIMEOUT="${PREDICTION_TIMEOUT}" \
OUT="${RUN_OUT}" LOG="${RUN_LOG}" \
  bash scripts/run_phase_c4_4_verified_regime_smoke.sh 2>&1 | tee "${ROOT_LOG}/common_runner.log"

cleanup

# C5.18: decompose tracker error by committed execution stamp before packing.
# This is offline analysis only; it never changes planner/controller behavior.
if ! python3 scripts/analyze_tracker_execution_breakdown.py \
    --run-dir "${RUN_OUT}" \
    > "${ROOT_LOG}/tracker_execution_breakdown.log" 2>&1; then
  echo "[WARN] tracker execution breakdown failed; preserving raw CSVs" >&2
fi

python3 - "${RUN_OUT}" "${SUMMARY_JSON}" <<'PY'
import csv
import json
import math
import os
import re
import statistics
import sys

out, dst = sys.argv[1:3]
TOK = re.compile(r'([A-Za-z0-9_]+)=([^\s]+)')

def rows(name):
    path = os.path.join(out, name)
    ans = []
    if not os.path.isfile(path):
        return ans
    with open(path, newline='', errors='replace') as f:
        rd = csv.reader(f)
        h = next(rd, [])
        if not h:
            return ans
        ti = h.index('%time') if '%time' in h else 0
        di = h.index('field.data') if 'field.data' in h else 1
        for row in rd:
            if len(row) <= di:
                continue
            d = dict(TOK.findall(','.join(row[di:])))
            try:
                d['_t'] = float(row[ti]) / 1e9
            except Exception:
                d['_t'] = math.nan
            if d:
                ans.append(d)
    return ans

def fnum(v):
    try:
        return float(str(v).replace('ms',''))
    except Exception:
        return math.nan

def stats(vals):
    a = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not a:
        return None
    def q(frac):
        if len(a) == 1:
            return a[0]
        p = frac * (len(a)-1)
        lo = int(p); hi = min(lo+1, len(a)-1); w = p-lo
        return a[lo]*(1-w)+a[hi]*w
    return {
        'min': min(a),
        'median': statistics.median(a),
        'mean': statistics.fmean(a),
        'p95': q(0.95),
        'max': max(a),
    }

scp = rows('local_planner_summary.csv')
tracker = rows('tracker_summary.csv')
cand = rows('candidate_vbc_summary.csv')
exe = rows('execution_vbc_summary.csv')
commit = rows('commit_summary.csv')
blocker = rows('blocker_stack_summary.csv')
cdfsel = rows('local_cdf_selector_summary.csv')

solved = [r for r in scp if r.get('event') == 'scp_solved' and r.get('solved') == '1']
failed = [r for r in scp if r.get('event') == 'scp_failed']
candidates = [r for r in scp if r.get('event') == 'candidate_published']

def col(source, key):
    return [fnum(r.get(key)) for r in source]

vbc_pred = [r for r in cand if r.get('trajectory_source') == 'predicted']
exe_rows = [r for r in exe if r.get('trajectory_source') in ('predicted','committed')]

max_depth = 0
targets = []
for r in blocker:
    s = r.get('stack','none')
    depth = 0 if s == 'none' else len([x for x in s.split(':') if x])
    max_depth = max(max_depth, depth)
    try:
        t = int(float(r.get('current_target_id','-1')))
    except Exception:
        t = -1
    if t >= 0 and (not targets or targets[-1] != t):
        targets.append(t)

summary = {
    'phase': 'C5.4',
    'architecture': 'event-triggered local Sparse SCP -> exact VBC -> committed full-trajectory tracker',
    'legacy_planning_mpc_present': False,
    'planner_deadline_50ms_required': False,
    'planner_initial_latency_goal_ms': 200.0,
    'scp_summary_records': len(scp),
    'scp_successful_subproblems': len(solved),
    'scp_failed_subproblems': len(failed),
    'candidate_trajectories_published': len(candidates),
    'sparse_piqp_solve_ms': stats(col(solved, 'solve_ms')),
    'cdf_rows': stats(col(solved, 'cdf_rows')),
    'screened_safe_rows': stats(col(solved, 'screened_safe')),
    'max_cdf_slack': stats(col(solved, 'max_slack')),
    'mean_cdf_slack': stats(col(solved, 'mean_slack')),
    'slack_mu_used': stats(col(solved, 'slack_mu_used')),
    'slack_mu_used_sequence': [
        fnum(r.get('slack_mu_used')) for r in solved
        if math.isfinite(fnum(r.get('slack_mu_used')))
    ],
    'max_cdf_slack_sequence': [
        fnum(r.get('max_slack')) for r in solved
        if math.isfinite(fnum(r.get('max_slack')))
    ],
    'min_cdf_distance_sequence': [
        fnum(r.get('min_d')) for r in solved
        if math.isfinite(fnum(r.get('min_d')))
    ],
    'scp_step_inf': stats(col(solved, 'step_inf')),
    'linearization_error_inf': stats(col(solved, 'qlin_error_inf')),
    'candidate_vbc_records': len(vbc_pred),
    'candidate_vbc_safe': sum(r.get('has_violation') == '0' for r in vbc_pred),
    'candidate_vbc_unsafe': sum(r.get('has_violation') == '1' for r in vbc_pred),
    'execution_vbc_records': len(exe_rows),
    'execution_vbc_unsafe': sum(r.get('has_violation') == '1' for r in exe_rows),
    'commit_summary_records': len(commit),
    'commit_count': (
        int(float(commit[-1].get('commit_count', '0')))
        if commit else 0
    ),
    'tracker_records': len(tracker),
    'tracker_error_inf': stats(col(tracker, 'tracking_error_inf')),
    'cdf_selector_records': len(cdfsel),
    'blocker_max_stack_depth': max_depth,
    'target_transition_sequence': targets,
}

plan_total = [
    fnum(r.get('total_plan_ms'))
    for r in candidates
    if math.isfinite(fnum(r.get('total_plan_ms')))
]
summary['total_local_plan_ms'] = stats(plan_total)
summary['any_sparse_scp_solution'] = bool(solved)
summary['any_candidate_published'] = bool(candidates)
summary['execution_safety_pass'] = (
    summary['execution_vbc_records'] > 0 and
    summary['execution_vbc_unsafe'] == 0
)
summary['execution_safety_status'] = (
    'pass' if summary['execution_safety_pass']
    else ('fail' if summary['execution_vbc_records'] > 0 else 'not_evaluated')
)

with open(dst, 'w') as f:
    json.dump(summary, f, indent=2)

print("")
print("========== C5.4 LOCAL SPARSE SCP SUMMARY ==========")
print(json.dumps(summary, indent=2))
print("===================================================")

if not scp:
    raise SystemExit('[ERROR] no local planner summary records')
if not cdfsel:
    raise SystemExit('[ERROR] no local CDF selector records')
PY

cat > "${ROOT_OUT}/c5_4_run_metadata.txt" <<EOF
branch=$(git branch --show-current)
head=$(git rev-parse HEAD)
case_id=${CASE_ID}
run_seconds=${RUN_SECONDS}
architecture=event_triggered_local_sparse_scp
planning_mpc=false
tracker=full_committed_trajectory
committed_publish_semantics=publish_once_tracker_owns_execution
committed_continuation_enabled=false
execution_vbc_stream=elapsed_suffix_on_separate_audit_topic
execution_freshness=tracker_liveness_plus_execution_vbc
final_executable_gcdf_before_vbc=true
final_gcdf_transport=dual_channel_single_gpu_client_final_priority
exact_vbc_before_commit=true
final_commit_gate=final_executable_gcdf_and_exact_vbc
execution_token=committed_header_stamp_ns
probe_success_semantics=matched_tracker_completion_execution_stamp
probe_single_flight=true
runtime_semantics_evidence=run/runtime_semantics.txt
repair_prefix_verification=${REPAIR_PREFIX_VERIFY}
repair_execution_prefix_s=${REPAIR_PREFIX_S}
repair_brake_dt_s=${REPAIR_BRAKE_DT_S}
repair_hold_s=${REPAIR_HOLD_S}
normal_verification_semantics=full_horizon
repair_verification_semantics=prefix_plus_brake_plus_hold_when_enabled
probe_verification_semantics=prefix_plus_brake_plus_hold_when_enabled
probe_mode=task_objective_plus_executable_horizon_gcdf
cdf_constraint_horizon_steps=from_config:${CONFIG_FILE}
normal_cdf_horizon=full_planning_horizon
cdf_variant=signed
cdf_activation=gelu
scp_max_iterations=from_config:${CONFIG_FILE}
solver=piqp_sparse
initial_planner_latency_goal_ms=200
EOF

trap - EXIT
pack_debug_bundle

echo ""
echo "[C5.4 COMPLETE]"
echo "[SUMMARY] ${SUMMARY_JSON}"
echo "[UPLOAD ZIP] ${ZIP_PATH}"
ls -lh "${ZIP_PATH}"
