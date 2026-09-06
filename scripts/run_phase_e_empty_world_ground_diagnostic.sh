#!/usr/bin/env bash
set -euo pipefail

# Diagnose the five Phase-E empty-world cases that entered execution-GCDF
# OCCUPIED HARD_HOLD during the 30-case qualification. This wrapper changes no
# planner/mapping semantics. It only captures the exact occupied blocker voxel
# emitted by execution_gcdf_safety_monitor_node.py.
#
# The primary question is whether the first hard blocker lies in the bottom
# confidence-map layer / ground band.

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_FILE="${CASE_FILE:-${REPO}/outputs/phase_e_goal_sampling/phase_e_obstacle_goal_pool_30.json}"
WORLD_FILE="${WORLD_FILE:-${REPO}/src/arm_description/worlds/maixsense_empty.world}"
CONFIDENCE_MAP_CONFIG_FILE="${CONFIDENCE_MAP_CONFIG_FILE:-${REPO}/src/care_confidence_map/config/confidence_map_phase_e_ray.yaml}"
RUN_SECONDS="${RUN_SECONDS:-20}"
GAZEBO_GUI="${GAZEBO_GUI:-false}"
USE_RVIZ="${USE_RVIZ:-false}"
CASES=(phase_e_goal_004 phase_e_goal_015 phase_e_goal_021 phase_e_goal_022 phase_e_goal_027)

cd "${REPO}"

if [[ ! -f "${CASE_FILE}" ]]; then
  echo "[ERROR] case file missing: ${CASE_FILE}" >&2
  exit 2
fi
if [[ ! -f "${WORLD_FILE}" ]]; then
  echo "[ERROR] world missing: ${WORLD_FILE}" >&2
  exit 3
fi

STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
GIT_SHORT="$(git rev-parse --short=8 HEAD)"
DIAG_ID="phase_e_empty_ground_diag_${STAMP}_${GIT_SHORT}"
ROOT="${REPO}/outputs/phase_e_empty_ground_diagnostic/${DIAG_ID}"
LOG_DIR="${ROOT}/logs"
ART_DIR="${ROOT}/artifacts"
SUMMARY="${ROOT}/occupied_blockers.txt"
ZIP="${REPO}/CAREPlanner_PHASE_E_EMPTY_GROUND_DIAGNOSTIC_${DIAG_ID}.zip"

rm -rf "${ROOT}"
rm -f "${ZIP}"
mkdir -p "${LOG_DIR}" "${ART_DIR}"
: > "${SUMMARY}"

cleanup_ros() {
  local n
  for n in gzclient gzserver rviz rosmaster roscore roslaunch; do
    pkill -TERM -x "${n}" 2>/dev/null || true
  done
  sleep 0.5
  for n in gzclient gzserver rviz rosmaster roscore roslaunch; do
    pkill -KILL -x "${n}" 2>/dev/null || true
  done
  rm -f /tmp/care_collision_cdf_gpu_c5_5.sock         /tmp/care_collision_cdf_gpu_c5_4.sock 2>/dev/null || true
}

cat > "${ROOT}/metadata.txt" <<EOF
diagnostic=phase_e_empty_world_ground_blocker
git_head=$(git rev-parse HEAD)
case_file=${CASE_FILE}
world_file=${WORLD_FILE}
confidence_map_config=${CONFIDENCE_MAP_CONFIG_FILE}
run_seconds=${RUN_SECONDS}
cases=${CASES[*]}
question=is execution-GCDF OCCUPIED HARD_HOLD caused by ground/bottom-layer voxels
EOF

for CASE_ID in "${CASES[@]}"; do
  echo ""
  echo "================================================================"
  echo "[GROUND DIAG] ${CASE_ID}"
  echo "================================================================"

  cleanup_ros

  RUN_ID="${CASE_ID}_ground_diag_${STAMP}_${GIT_SHORT}"
  RUN_ROOT="${REPO}/outputs/c5_5_vbc_gcdf_regime/${RUN_ID}"
  CASE_ZIP="${REPO}/CAREPlanner_C5_RESULT_${RUN_ID}.zip"
  LOG="${LOG_DIR}/${CASE_ID}.log"
  CASE_ART="${ART_DIR}/${CASE_ID}"
  mkdir -p "${CASE_ART}"

  set +e
  (
    CASE_FILE="${CASE_FILE}"     CASE_ID="${CASE_ID}"     RUN_ID="${RUN_ID}"     RUN_SECONDS="${RUN_SECONDS}"     WORLD_FILE="${WORLD_FILE}"     CONFIDENCE_MAP_CONFIG_FILE="${CONFIDENCE_MAP_CONFIG_FILE}"     TOF_FUSION_ENABLED=true     EXECUTION_GCDF_AUDIT_ENABLED=true     GCDF_BODY_INFLATION_M=0.015     FORCE_ZERO_INITIAL_Q=true     EARLY_STOP_ON_GOAL=false     GAZEBO_GUI="${GAZEBO_GUI}"     USE_RVIZ="${USE_RVIZ}"     bash scripts/run_and_pack_phase_e5_execution_gcdf.sh
  ) > >(tee "${LOG}") 2>&1
  RC=$?
  set -e

  echo "case=${CASE_ID} runner_rc=${RC}" >> "${SUMMARY}"
  grep -F "[EXECUTION_GCDF_OCCUPIED_BLOCKER]" "${LOG}"     | tee -a "${SUMMARY}" || true
  echo "" >> "${SUMMARY}"

  if [[ -d "${RUN_ROOT}/run" ]]; then
    for name in       execution_gcdf_safety_summary.csv       execution_gcdf_selector_summary.csv       execution_gcdf_hard_hold.csv       e3_summary.csv       tof_fusion_summary.csv       joint_states.csv       regime_summary.csv       local_planner_summary.csv; do
      [[ -f "${RUN_ROOT}/run/${name}" ]] &&         cp -f "${RUN_ROOT}/run/${name}" "${CASE_ART}/${name}"
    done
  fi

  rm -f "${CASE_ZIP}"
done

cleanup_ros

python3 - "${ROOT}" "${ZIP}" <<'PY'
import os, sys, zipfile
root, dst = sys.argv[1:]
with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for base, _, files in os.walk(root):
        for name in files:
            p = os.path.join(base, name)
            z.write(p, os.path.relpath(p, root))
print(dst)
PY

echo ""
echo "================ GROUND DIAGNOSTIC COMPLETE ================"
echo "[BLOCKERS] ${SUMMARY}"
cat "${SUMMARY}"
echo "[UPLOAD ZIP] ${ZIP}"
ls -lh "${ZIP}"
