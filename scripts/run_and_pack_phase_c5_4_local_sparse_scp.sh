#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_ID="${CASE_ID:-case_003}"
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

ROOT_OUT="${ROOT_OUT:-${REPO}/outputs/phase_c5_4_local_sparse_scp/${CASE_ID}}"
ROOT_LOG="${ROOT_LOG:-${REPO}/logs/phase_c5_4_local_sparse_scp/${CASE_ID}}"
RUN_OUT="${ROOT_OUT}/run"
RUN_LOG="${ROOT_LOG}/run"
SELECTOR_JSONL="${ROOT_OUT}/local_scp_selector.jsonl"
ZIP_PATH="${ZIP_PATH:-${REPO}/CAREPlanner_C5_4_local_sparse_scp_${CASE_ID}.zip}"
SUMMARY_JSON="${ROOT_OUT}/c5_4_local_sparse_scp_summary.json"

cd "${REPO}"

echo "[C5.4] shell syntax preflight..."
bash -n scripts/run_and_pack_phase_c5_4_local_sparse_scp.sh
bash -n scripts/run_phase_c4_4_verified_regime_smoke.sh

echo "[C5.4] branch: $(git branch --show-current)"
echo "[C5.4] head:   $(git rev-parse HEAD)"
echo "[C5.4] case:   ${CASE_ID}"
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
  src/egocentric_arm_planner/scripts/probe_single_flight_gate_node.py
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
  rm -f "${ZIP_PATH}"
  zip -r "${ZIP_PATH}" "${ROOT_OUT#${REPO}/}" "${ROOT_LOG#${REPO}/}" >/dev/null
  echo "[C5.4 ZIP] ${ZIP_PATH}"
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
echo "[ZIP] ${ZIP_PATH}"
ls -lh "${ZIP_PATH}"
