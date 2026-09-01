#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
RUN_SECONDS="${RUN_SECONDS:-60.0}"
TRIAL_ID="${TRIAL_ID:-probe_diag_00}"

cd "${REPO}"

CASES=(case_006 case_007 case_009)
STAMP="${DIAG_STAMP:-$(date +%Y%m%d-%H%M%S)}"
GIT_SHORT="$(git rev-parse --short=8 HEAD)"
DIAG_ID="${DIAG_ID:-${TRIAL_ID}_${STAMP}_${GIT_SHORT}}"
ROOT="${REPO}/outputs/phase_d_probe_recovery_diag/${DIAG_ID}"
CASE_DIR="${ROOT}/cases"
LOG_DIR="${ROOT}/logs"
FINAL_ZIP="${REPO}/CAREPlanner_PHASE_D_PROBE_DIAG_${DIAG_ID}.zip"

rm -rf "${ROOT}"
rm -f "${FINAL_ZIP}"
mkdir -p "${CASE_DIR}" "${LOG_DIR}"

cat > "${ROOT}/diagnostic_metadata.txt" <<EOF
phase=D.2.1
purpose=diagnose_probe_to_normal_recovery_livelock
trial_id=${TRIAL_ID}
run_seconds=${RUN_SECONDS}
git_head=$(git rev-parse HEAD)
git_branch=$(git branch --show-current)
cases=${CASES[*]}
EOF

copy_if_exists() {
  local src="$1" dst="$2"
  if [[ -f "$src" ]]; then
    cp -f "$src" "$dst"
  fi
}

for CASE_ID in "${CASES[@]}"; do
  echo ""
  echo "============================================================"
  echo "[PROBE DIAG] ${CASE_ID}"
  echo "============================================================"

  RUN_ID="${CASE_ID}_${DIAG_ID}"
  ROOT_OUT="${REPO}/outputs/c5_5_vbc_gcdf_regime/${RUN_ID}"
  CASE_ZIP="${REPO}/CAREPlanner_C5_RESULT_${RUN_ID}.zip"
  DEST="${CASE_DIR}/${CASE_ID}"
  mkdir -p "${DEST}"

  set +e
  CASE_ID="${CASE_ID}" RUN_ID="${RUN_ID}" RUN_SECONDS="${RUN_SECONDS}" \
    bash scripts/run_and_pack_c5_5_vbc_gcdf_regime.sh \
    > >(tee "${LOG_DIR}/${CASE_ID}.log") 2>&1
  RUN_RC=$?
  set -e

  EVAL_RC=0
  if [[ -d "${ROOT_OUT}/run" ]]; then
    set +e
    python3 scripts/analyze_probe_recovery_episodes.py \
      --run-dir "${ROOT_OUT}/run" \
      --case-id "${CASE_ID}" \
      --output-json "${DEST}/probe_recovery_diagnostic.json" \
      --output-csv "${DEST}/probe_recovery_episodes.csv" \
      | tee -a "${LOG_DIR}/${CASE_ID}.log"
    EVAL_RC=$?
    set -e

    for name in \
      regime_summary.csv \
      probe_single_flight_summary.csv \
      verification_outcome.csv \
      tracker_summary.csv \
      local_planner_summary.csv \
      task_infeasible.csv \
      task_uncertified.csv \
      blocker_stack_summary.csv \
      visibility_acquisition_summary.csv \
      commit_summary.csv \
      execution_vbc_summary.csv \
      nominal_progress_summary.csv \
      tracker_execution_breakdown.json; do
      copy_if_exists "${ROOT_OUT}/run/${name}" "${DEST}/${name}"
    done

    if [[ -f scripts/evaluate_phase_d_run.py ]]; then
      set +e
      python3 scripts/evaluate_phase_d_run.py \
        --run-dir "${ROOT_OUT}/run" \
        --case-id "${CASE_ID}" \
        --method careplanner_full \
        --trial-id "${TRIAL_ID}" \
        --output-json "${DEST}/phase_d_run_summary.json" \
        >> "${LOG_DIR}/${CASE_ID}.log" 2>&1
      set -e
    fi
  else
    EVAL_RC=2
  fi

  python3 - "${DEST}/runner_status.json" "${RUN_RC}" "${EVAL_RC}" "${ROOT_OUT}" <<'PY'
import json, sys
path, run_rc, eval_rc, root_out = sys.argv[1:]
json.dump({
    "runner_return_code": int(run_rc),
    "diagnostic_return_code": int(eval_rc),
    "root_out": root_out,
}, open(path, "w"), indent=2)
PY

  rm -f "${CASE_ZIP}"
done

python3 - "${CASE_DIR}" "${ROOT}/probe_recovery_diagnostic_index.json" <<'PY'
import glob, json, os, sys
root, out = sys.argv[1:]
cases = []
for path in sorted(glob.glob(os.path.join(root, "case_*", "probe_recovery_diagnostic.json"))):
    with open(path) as f:
        d = json.load(f)
    cases.append({
        "case_id": d.get("case_id"),
        "final_regime_state": d.get("final_regime_state"),
        "probe_episode_count": d.get("probe_episode_count"),
        "probe_recovered_to_normal_count": d.get("probe_recovered_to_normal_count"),
        "probe_returned_to_repair_count": d.get("probe_returned_to_repair_count"),
        "probe_open_ended_count": d.get("probe_open_ended_count"),
        "probe_livelock_signature": d.get("probe_livelock_signature"),
        "diagnosis_counts": d.get("diagnosis_counts"),
    })
json.dump({"phase": "D.2.1", "cases": cases}, open(out, "w"), indent=2)
print(json.dumps(cases, indent=2))
PY

python3 - "${ROOT}" "${FINAL_ZIP}" <<'PY'
import os, sys, zipfile
root, dst = sys.argv[1:]
with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for base, _, files in os.walk(root):
        for name in files:
            path = os.path.join(base, name)
            z.write(path, os.path.relpath(path, root))
print(dst)
PY

echo ""
echo "================ PROBE DIAGNOSTIC COMPLETE ================"
echo "[INDEX]      ${ROOT}/probe_recovery_diagnostic_index.json"
echo "[UPLOAD ZIP] ${FINAL_ZIP}"
ls -lh "${FINAL_ZIP}"
