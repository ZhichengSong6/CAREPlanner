#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
RUN_SECONDS="${RUN_SECONDS:-60.0}"
TRIAL_ID="${TRIAL_ID:-trial_00}"
METHOD="${METHOD:-careplanner_full}"
KEEP_CASE_ZIPS="${KEEP_CASE_ZIPS:-0}"

cd "${REPO}"

CASES=(
  case_014 case_018 case_017 case_008
  case_006 case_003 case_000 case_001
  case_009 case_013 case_015 case_007
)

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
      regime_summary.csv \
      visibility_acquisition_summary.csv \
      candidate_vbc_summary.csv \
      execution_vbc_summary.csv \
      commit_summary.csv \
      tracker_summary.csv \
      local_planner_summary.csv \
      nominal_progress_summary.csv \
      tracker_execution_breakdown.json; do
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
    "benchmark_runner_failur": True,
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

  if [[ "${KEEP_CASE_ZIPS" != "1" ]]; then
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
