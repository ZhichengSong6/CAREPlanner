#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_FILE="${CASE_FILE:-${REPO}/outputs/phase_e_goal_sampling/phase_e_obstacle_goal_pool_30.json}"
CASE_ID="${CASE_ID:-phase_e_goal_026}"
RUN_SECONDS="${RUN_SECONDS:-45}"
EARLY_STOP_ON_GOAL="${EARLY_STOP_ON_GOAL:-false}"
GAZEBO_GUI="${GAZEBO_GUI:-true}"
USE_RVIZ="${USE_RVIZ:-false}"

WORLD_FILE="${WORLD_FILE:-${REPO}/src/arm_description/worlds/maixsense_empty.world}"
CONFIDENCE_MAP_CONFIG_FILE="${CONFIDENCE_MAP_CONFIG_FILE:-${REPO}/src/care_confidence_map/config/confidence_map_phase_e_ray.yaml}"

cd "${REPO}"

if [[ ! -f "${CASE_FILE}" ]]; then
  echo "[ERROR] case file missing: ${CASE_FILE}"
  exit 2
fi

python3 - "${CASE_FILE}" "${CASE_ID}" <<'PY'
import json,sys
path,cid=sys.argv[1:]
d=json.load(open(path))
c=next((x for x in d.get("cases",[]) if str(x.get("case_id"))==cid),None)
if c is None:
    raise SystemExit(f"[ERROR] {cid} not found in {path}")
print("[CASE]",cid)
print("[GOAL POSITION]",c.get("goal_position"))
print("[GOAL ORIENTATION]",c.get("goal_orientation"))
print("[INITIAL Q]",c.get("initial_q",[0.0]*7))
PY

if [[ "${CONDA_SHLVL:-0}" =~ ^[0-9]+$ ]] && (( CONDA_SHLVL > 0 )); then
  if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  fi
  while [[ "${CONDA_SHLVL:-0}" =~ ^[0-9]+$ ]] && (( CONDA_SHLVL > 0 )); do
    conda deactivate || break
  done
fi

STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
SHORT="$(git rev-parse --short=8 HEAD)"
RUN_ID="${CASE_ID}_qvis_validity_${STAMP}_${SHORT}"
RUN_ROOT="${REPO}/outputs/c5_5_vbc_gcdf_regime/${RUN_ID}"
DIAG_ROOT="${REPO}/outputs/phase_e_qvis_validity/${RUN_ID}"
LOG_FILE="${DIAG_ROOT}/runner.log"
REPORT_JSON="${DIAG_ROOT}/qvis_visibility_validity.json"
UPLOAD_ZIP="${REPO}/CAREPlanner_PHASE_E_QVIS_VALIDITY_${RUN_ID}.zip"

rm -rf "${DIAG_ROOT}"
mkdir -p "${DIAG_ROOT}/artifacts" "${DIAG_ROOT}/obligation_traces"

cleanup_ros() {
  local n
  for n in gzclient gzserver rviz rosmaster roscore roslaunch; do
    pkill -TERM -x "${n}" 2>/dev/null || true
  done
  sleep 0.6
  for n in gzclient gzserver rviz rosmaster roscore roslaunch; do
    pkill -KILL -x "${n}" 2>/dev/null || true
  done
  rm -f /tmp/care_collision_cdf_gpu_c5_5.sock         /tmp/care_collision_cdf_gpu_c5_4.sock 2>/dev/null || true
}
trap cleanup_ros EXIT INT TERM

echo "================================================================"
echo "PHASE-E CASE 026 Q_VIS VALIDITY / SELF-OCCLUSION DIAGNOSTIC"
echo "case        : ${CASE_ID}"
echo "world       : ${WORLD_FILE}"
echo "runtime     : ${RUN_SECONDS}s"
echo "oracle diag : ON"
echo "GUI         : ${GAZEBO_GUI}"
echo "semantics   : unchanged formal C5.5 / Phase-E"
echo "================================================================"

cleanup_ros
set +e
(
  CASE_FILE="${CASE_FILE}"   CASE_ID="${CASE_ID}"   RUN_ID="${RUN_ID}"   RUN_SECONDS="${RUN_SECONDS}"   WORLD_FILE="${WORLD_FILE}"   CONFIDENCE_MAP_CONFIG_FILE="${CONFIDENCE_MAP_CONFIG_FILE}"   TOF_FUSION_ENABLED=true   EXECUTION_GCDF_AUDIT_ENABLED=true   GCDF_BODY_INFLATION_M=0.015   FORCE_ZERO_INITIAL_Q=true   APPLY_INITIAL_JOINT_OVERRIDES=auto   ENABLE_ORACLE_DIAGNOSTICS=true   EARLY_STOP_ON_GOAL="${EARLY_STOP_ON_GOAL}"   GAZEBO_GUI="${GAZEBO_GUI}"   USE_RVIZ="${USE_RVIZ}"   bash scripts/run_and_pack_phase_e5_execution_gcdf.sh
) > >(tee "${LOG_FILE}") 2>&1
RUN_RC=$?
set -e

RUN_DIR="${RUN_ROOT}/run"
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "[ERROR] run directory missing: ${RUN_DIR}"
  exit 3
fi

for f in   visibility_acquisition_summary.csv   blocker_stack_summary.csv   candidate_vbc_summary.csv   execution_vbc_summary.csv   regime_summary.csv   joint_states.csv   tof_fusion_summary.csv   e3_summary.csv   goal_stop_status.json   startup_gazebo_tf_snapshot.json; do
  [[ -f "${RUN_DIR}/${f}" ]] && cp -f "${RUN_DIR}/${f}" "${DIAG_ROOT}/artifacts/${f}"
done

while IFS= read -r -d '' p; do
  cp -f "${p}" "${DIAG_ROOT}/obligation_traces/$(basename "${p}")"
done < <(find "${RUN_ROOT}" -type f -name 'c46_obligation_*.json' -print0 2>/dev/null)

TRACE_COUNT="$(find "${DIAG_ROOT}/obligation_traces" -type f -name 'c46_obligation_*.json' | wc -l)"
echo "[TRACE] copied ${TRACE_COUNT} obligation traces"
if [[ "${TRACE_COUNT}" -eq 0 ]]; then
  echo "[ERROR] no c46 obligation traces found"
  exit 4
fi

python3 scripts/diagnose_phase_e_qvis_visibility.py   --repo "${REPO}"   --trace-dir "${DIAG_ROOT}/obligation_traces"   --acquisition-csv "${RUN_DIR}/visibility_acquisition_summary.csv"   --output-json "${REPORT_JSON}"   | tee "${DIAG_ROOT}/qvis_visibility_validity.txt"

cat > "${DIAG_ROOT}/metadata.txt" <<EOF
git_head=$(git rev-parse HEAD)
case_file=${CASE_FILE}
case_id=${CASE_ID}
run_seconds=${RUN_SECONDS}
world_file=${WORLD_FILE}
confidence_map_config=${CONFIDENCE_MAP_CONFIG_FILE}
enable_oracle_diagnostics=true
force_zero_initial_q=true
apply_initial_joint_overrides=auto
diagnostic_question=learned_false_positive_vs_self_occlusion_vs_runtime_sensor_visibility
runner_return_code=${RUN_RC}
EOF

rm -f "${UPLOAD_ZIP}"
python3 - "${DIAG_ROOT}" "${UPLOAD_ZIP}" <<'PY'
import os,sys,zipfile
root,dst=sys.argv[1:]
with zipfile.ZipFile(dst,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for base,_,files in os.walk(root):
        for name in files:
            p=os.path.join(base,name)
            z.write(p,os.path.relpath(p,root))
print(dst)
PY

echo ""
echo "================ CASE 026 DIAGNOSTIC COMPLETE ================"
echo "[REPORT] ${REPORT_JSON}"
echo "[UPLOAD] ${UPLOAD_ZIP}"
ls -lh "${UPLOAD_ZIP}"
echo "==============================================================="
