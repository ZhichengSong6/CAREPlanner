#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"

# Phase-D launches ROS/Gazebo and also runs ROS-package-backed offline
# evaluators (e.g. urdf_parser_py). Do not inherit an arbitrary research conda
# interpreter from the caller. The C5 child runner has the same guard so direct
# C5 invocation is independently safe.
if [[ "${CONDA_SHLVL:-0}" =~ ^[0-9]+$ ]] && (( CONDA_SHLVL > 0 )); then
  if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  fi
  _phase_d_conda_before="${CONDA_DEFAULT_ENV:-unknown}"
  while [[ "${CONDA_SHLVL:-0}" =~ ^[0-9]+$ ]] && (( CONDA_SHLVL > 0 )); do
    conda deactivate || break
  done
  echo "[PHASE D ENV] sanitized inherited conda env (was ${_phase_d_conda_before}); benchmark uses system ROS/Python"
fi

# Canonical Phase-D benchmark duration is fixed ROS/Gazebo simulation time,
# independent of host load / Gazebo real-time factor. Goal-conditioned early
# termination remains available as an explicit opt-in diagnostic.
RUN_SECONDS="${RUN_SECONDS:-30.0}"
EARLY_STOP_ON_GOAL="${EARLY_STOP_ON_GOAL:-false}"
GOAL_POSITION_TOLERANCE_M="${GOAL_POSITION_TOLERANCE_M:-0.02}"
GOAL_ORIENTATION_TOLERANCE_RAD="${GOAL_ORIENTATION_TOLERANCE_RAD:-0.20}"
GOAL_SUCCESS_HOLD_S="${GOAL_SUCCESS_HOLD_S:-0.10}"
GOAL_SETTLE_VELOCITY_INF_RAD_S="${GOAL_SETTLE_VELOCITY_INF_RAD_S:-0.05}"
GOAL_SETTLE_TIMEOUT_S="${GOAL_SETTLE_TIMEOUT_S:-1.0}"
GOAL_POST_SUCCESS_RECORD_S="${GOAL_POST_SUCCESS_RECORD_S:-0.0}"

TRIAL_ID="${TRIAL_ID:-trial_00}"
METHOD="${METHOD:-careplanner_full}"
KEEP_CASE_ZIPS="${KEEP_CASE_ZIPS:-0}"

cd "${REPO}"

DEFAULT_CASES=(
  case_014 case_018 case_017 case_008
  case_006 case_003 case_000 case_001
  case_009 case_013 case_015 case_007
)

# Optional targeted Phase-D rerun without changing the canonical default.
# Example: PHASE_D_CASES="case_014 case_008 case_007"
if [[ -n "${PHASE_D_CASES:-}" ]]; then
  read -r -a CASES <<< "${PHASE_D_CASES}"
else
  CASES=("${DEFAULT_CASES[@]}")
fi

STAMP="${BATCH_STAMP:-$(date +%Y%m%d-%H%M%S)}"
GIT_SHORT="$(git rev-parse --short=8 HEAD)"
BATCH_ID="${BATCH_ID:-${METHOD}_${TRIAL_ID}_${STAMP}_${GIT_SHORT}}"
BATCH_ROOT="${REPO}/outputs/phase_d_12case/${BATCH_ID}"
CASE_SUMMARY_DIR="${BATCH_ROOT}/case_summaries"
CASE_ARTIFACT_DIR="${BATCH_ROOT}/case_artifacts"
BATCH_LOG_DIR="${BATCH_ROOT}/logs"
FINAL_ZIP="${REPO}/CAREPlanner_PHASE_D_12CASE_${BATCH_ID}.zip"

rm -rf "${BATCH_ROOT}"
rm -f "${FINAL_ZIP}"
mkdir -p "${CASE_SUMMARY_DIR}" "${CASE_ARTIFACT_DIR}" "${BATCH_LOG_DIR}"

cat > "${BATCH_ROOT}/benchmark_metadata.txt" <<EOF
phase=D.2
method=${METHOD}
trial_id=${TRIAL_ID}
run_seconds=${RUN_SECONDS}
run_time_basis=ros_gazebo_sim_time
early_stop_on_goal=${EARLY_STOP_ON_GOAL}
goal_position_tolerance_m=${GOAL_POSITION_TOLERANCE_M}
goal_orientation_tolerance_rad=${GOAL_ORIENTATION_TOLERANCE_RAD}
goal_success_hold_s=${GOAL_SUCCESS_HOLD_S}
goal_settle_velocity_inf_rad_s=${GOAL_SETTLE_VELOCITY_INF_RAD_S}
goal_settle_timeout_s=${GOAL_SETTLE_TIMEOUT_S}
goal_post_success_record_s=${GOAL_POST_SUCCESS_RECORD_S}
git_head=$(git rev-parse HEAD)
git_branch=$(git branch --show-current)
cases=${CASES[*]}
EOF

copy_if_exists() {
  local src="$1" dst="$2"
  if [[ -f "${src}" ]]; then
    cp -f "${src}" "${dst}"
  fi
}

for CASE_ID in "${CASES[@]}"; do
  echo ""
  echo "================================================================"
  echo "[PHASE D] ${CASE_ID}  method=${METHOD}  trial=${TRIAL_ID}"
  echo "================================================================"

  RUN_ID="${CASE_ID}_${BATCH_ID}"
  ROOT_OUT="${REPO}/outputs/c5_5_vbc_gcdf_regime/${RUN_ID}"
  CASE_ZIP="${REPO}/CAREPlanner_C5_RESULT_${RUN_ID}.zip"
  CASE_DIR="${CASE_ARTIFACT_DIR}/${CASE_ID}"
  mkdir -p "${CASE_DIR}"

  set +e
  CASE_ID="${CASE_ID}" RUN_ID="${RUN_ID}" RUN_SECONDS="${RUN_SECONDS}" \
    EARLY_STOP_ON_GOAL="${EARLY_STOP_ON_GOAL}" \
    GOAL_POSITION_TOLERANCE_M="${GOAL_POSITION_TOLERANCE_M}" \
    GOAL_ORIENTATION_TOLERANCE_RAD="${GOAL_ORIENTATION_TOLERANCE_RAD}" \
    GOAL_SUCCESS_HOLD_S="${GOAL_SUCCESS_HOLD_S}" \
    GOAL_SETTLE_VELOCITY_INF_RAD_S="${GOAL_SETTLE_VELOCITY_INF_RAD_S}" \
    GOAL_SETTLE_TIMEOUT_S="${GOAL_SETTLE_TIMEOUT_S}" \
    GOAL_POST_SUCCESS_RECORD_S="${GOAL_POST_SUCCESS_RECORD_S}" \
    bash scripts/run_and_pack_c5_5_vbc_gcdf_regime.sh \
    > >(tee "${BATCH_LOG_DIR}/${CASE_ID}.log") 2>&1
  RUN_RC=$?
  set -e

  EVAL_RC=0
  if [[ -d "${ROOT_OUT}/run" ]]; then
    set +e
    python3 scripts/evaluate_phase_d_run.py \
      --run-dir "${ROOT_OUT}/run" \
      --case-id "${CASE_ID}" \
      --method "${METHOD}" \
      --trial-id "${TRIAL_ID}" \
      --output-json "${CASE_SUMMARY_DIR}/${CASE_ID}.json" \
      >> "${BATCH_LOG_DIR}/${CASE_ID}.log" 2>&1
    EVAL_RC=$?
    set -e

    for name in \
      joint_states.csv \
      task_trajectory.csv \
      committed_trajectory.csv \
      regime_summary.csv \
      visibility_acquisition_summary.csv \
      candidate_vbc_summary.csv \
      execution_vbc_summary.csv \
      commit_summary.csv \
      tracker_summary.csv \
      local_planner_summary.csv \
      nominal_progress_summary.csv \
      tracker_execution_breakdown.json \
      goal_stop_status.json; do
      copy_if_exists "${ROOT_OUT}/run/${name}" "${CASE_DIR}/${name}"
    done
    copy_if_exists "${ROOT_OUT}/c5_4_local_sparse_scp_summary.json" "${CASE_DIR}/c5_4_local_sparse_scp_summary.json"
  else
    EVAL_RC=2
  fi

  if [[ ! -f "${CASE_SUMMARY_DIR}/${CASE_ID}.json" ]]; then
    python3 - "${CASE_SUMMARY_DIR}/${CASE_ID}.json" "${CASE_ID}" "${METHOD}" "${TRIAL_ID}" "${RUN_RC}" "${EVAL_RC}" <<'PY'
import json, sys
path, case_id, method, trial_id, run_rc, eval_rc = sys.argv[1:]
json.dump({
    "phase": "D.1",
    "method": method,
    "trial_id": trial_id,
    "case_id": case_id,
    "difficulty": None,
    "task_success": False,
    "overall_safe": False,
    "benchmark_runner_failure": True,
    "runner_return_code": int(run_rc),
    "evaluator_return_code": int(eval_rc),
}, open(path, "w"), indent=2)
PY
  fi

  python3 - "${CASE_DIR}/runner_status.json" "${RUN_RC}" "${EVAL_RC}" "${ROOT_OUT}" <<'PY'
import json, sys
path, run_rc, eval_rc, root_out = sys.argv[1:]
json.dump({
    "runner_return_code": int(run_rc),
    "evaluator_return_code": int(eval_rc),
    "root_out": root_out,
}, open(path, "w"), indent=2)
PY

  if [[ "${KEEP_CASE_ZIPS}" != "1" ]]; then
    rm -f "${CASE_ZIP}"
  fi

done

python3 scripts/summarize_phase_d_benchmark.py \
  --input-root "${CASE_SUMMARY_DIR}" \
  --output-json "${BATCH_ROOT}/phase_d_12case_summary.json" \
  --output-csv "${BATCH_ROOT}/phase_d_12case_summary.csv" \
  | tee "${BATCH_LOG_DIR}/benchmark_summary.log"

python3 - "${BATCH_ROOT}" "${FINAL_ZIP}" <<'PY'
import os, sys, zipfile
root, dst = sys.argv[1:]
with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for base, _, files in os.walk(root):
        for name in files:
            path = os.path.join(base, name)
            arc = os.path.relpath(path, root)
            z.write(path, arc)
print(dst)
PY

echo ""
echo "=============== PHASE D 12-CASE COMPLETE ==============="
echo "[SUMMARY JSON] ${BATCH_ROOT}/phase_d_12case_summary.json"
echo "[SUMMARY CSV]  ${BATCH_ROOT}/phase_d_12case_summary.csv"
echo "[UPLOAD ZIP]   ${FINAL_ZIP}"
ls -lh "${FINAL_ZIP}"
