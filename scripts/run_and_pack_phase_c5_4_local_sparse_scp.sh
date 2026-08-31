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

def as_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default

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

probe = R["probe_single_flight_summary.csv"]
digest["probe_single_flight"] = {
    "phase_transitions": transition_sequence(probe, "phase"),
    "last": probe[-1] if probe else None,
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

def as_float(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None

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

digest["timing"] = {
    "local_plan_total_ms": timing_stats([
        as_float(r.get("total_plan_ms")) for r in candidate_plans
    ]),
    "piqp_final_solve_ms": timing_stats([
        as_float(r.get("solve_ms")) for r in candidate_plans
    ]),
    "raw_to_safety_dispatch_ms": timing_stats([
        as_float(r.get("raw_to_safety_dispatch_ms")) for r in commit_rows
    ]),
    "final_gcdf_roundtrip_ms": timing_stats([
        as_float(r.get("final_gcdf_roundtrip_ms")) for r in commit_rows
    ]),
    "exact_vbc_roundtrip_ms": timing_stats([
        as_float(r.get("exact_vbc_roundtrip_ms")) for r in commit_rows
    ]),
    "candidate_total_safety_pipeline_ms": timing_stats([
        as_float(r.get("candidate_total_safety_pipeline_ms"))
        for r in commit_rows
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
digest["timing"]["q_vis_generation_last_ms"] = timing_stats([
    as_float(r.get("q_vis_generation_last_ms")) for r in schedule_rows
])
digest["timing"]["q_vis_generation_max_ms_observed"] = timing_stats([
    as_float(r.get("q_vis_generation_max_ms")) for r in schedule_rows
])

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
