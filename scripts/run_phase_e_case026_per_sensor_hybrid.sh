#!/usr/bin/env bash
set -euo pipefail

# First end-to-end integration smoke for the mode-preserving 8-head VisCDF.
#
# Only the q_vis generation policy changes:
#   scalar union VisCDF -> scalar q_zero
#   -> rank/optimize per-sensor heads
#   -> conservative per-sensor FOV
#   -> zero-padding primitive self-occlusion
#   -> first accepted sensor branch
#   -> otherwise preserve the original scalar q_vis.
#
# Final GCDF, exact VBC, local Sparse-SCP, tracker, confidence map and ToF remain
# unchanged and retain execution authority.

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_FILE="${CASE_FILE:-${REPO}/outputs/phase_e_goal_sampling/phase_e_obstacle_goal_pool_30.json}"
CASE_ID="${CASE_ID:-phase_e_goal_026}"
RUN_SECONDS="${RUN_SECONDS:-45}"
GAZEBO_GUI="${GAZEBO_GUI:-false}"
USE_RVIZ="${USE_RVIZ:-false}"
EARLY_STOP_ON_GOAL="${EARLY_STOP_ON_GOAL:-true}"
WORLD_FILE="${WORLD_FILE:-${REPO}/src/arm_description/worlds/maixsense_empty.world}"
CONFIDENCE_MAP_CONFIG_FILE="${CONFIDENCE_MAP_CONFIG_FILE:-${REPO}/src/care_confidence_map/config/confidence_map_phase_e_ray.yaml}"

PER_SENSOR_CHECKPOINT="${PER_SENSOR_CHECKPOINT:-${REPO}/src/care_visibility_cdf/checkpoints/per_sensor_e2e_fullbatch_seed0/final.pt}"
PER_SENSOR_BRANCH_ASCENT_STEPS="${PER_SENSOR_BRANCH_ASCENT_STEPS:-1}"
PER_SENSOR_MAX_BRANCH_ATTEMPTS="${PER_SENSOR_MAX_BRANCH_ATTEMPTS:-4}"

cd "${REPO}"

if [[ ! -f "${CASE_FILE}" ]]; then
  echo "[ERROR] case file missing: ${CASE_FILE}" >&2
  exit 2
fi
if [[ ! -f "${PER_SENSOR_CHECKPOINT}" ]]; then
  echo "[ERROR] per-sensor checkpoint missing:" >&2
  echo "        ${PER_SENSOR_CHECKPOINT}" >&2
  echo "[ERROR] copy the trained final.pt into this path before running." >&2
  exit 3
fi

# ROS/Gazebo must run outside research conda environments.  The common runner
# re-enters the NCDF env only for the waypoint-generator subprocess.
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

python3 -m py_compile \
  src/care_visibility_cdf/scripts/per_sensor_visibility_runtime.py \
  src/care_visibility_cdf/scripts/vbc_deadline_waypoint_rolling_impl.py \
  src/care_visibility_cdf/scripts/vbc_deadline_waypoint_online_node.py \
  src/care_visibility_cdf/scripts/vbc_multi_deadline_obligation_impl.py \
  src/care_visibility_cdf/scripts/vbc_blocker_aware_acquisition_impl.py

# Make sure older narrow diagnostics cannot silently own q_vis.
unset CARE_DIAG_QVIS_OVERRIDE_ENABLED || true
unset CARE_DIAG_QVIS_OVERRIDE_OBLIGATION_ID || true
unset CARE_DIAG_QVIS_OVERRIDE_TARGET || true
unset CARE_DIAG_QVIS_OVERRIDE_Q || true
unset CARE_FIXED_VIS_TARGET || true
unset CARE_FIXED_VIS_Q || true
unset CARE_FIXED_VIS_Q_ZERO || true

STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
SHORT="$(git rev-parse --short=8 HEAD)"
RUN_ID="${RUN_ID:-${CASE_ID}_per_sensor_hybrid_${STAMP}_${SHORT}}"
RUN_ROOT="${REPO}/outputs/c5_5_vbc_gcdf_regime/${RUN_ID}"
GEN_LOG="${REPO}/logs/c5_5_vbc_gcdf_regime/${RUN_ID}/run/waypoint_generator.log"
BASE_ZIP="${REPO}/CAREPlanner_C5_RESULT_${RUN_ID}.zip"
UPLOAD_ZIP="${REPO}/CAREPlanner_PHASE_E_CASE026_PER_SENSOR_HYBRID_${RUN_ID}.zip"

echo "================================================================"
echo "PHASE-E CASE026 PER-SENSOR HYBRID INTEGRATION"
echo "case              : ${CASE_ID}"
echo "world             : ${WORLD_FILE}"
echo "runtime           : ${RUN_SECONDS}s"
echo "8-head checkpoint : ${PER_SENSOR_CHECKPOINT}"
echo "branch solver     : 10-step projection + root refinement + ${PER_SENSOR_BRANCH_ASCENT_STEPS} ascent"
echo "branch attempts   : ${PER_SENSOR_MAX_BRANCH_ATTEMPTS}"
echo "acceptance        : conservative per-sensor g >= 0 + primitive LOS clear"
echo "fallback          : original scalar q_vis if all tested branches reject"
echo "unchanged         : Sparse-SCP / final GCDF / exact VBC / tracker / ToF"
echo "================================================================"

set +e
(
  CASE_FILE="${CASE_FILE}" \
  CASE_ID="${CASE_ID}" \
  RUN_ID="${RUN_ID}" \
  RUN_SECONDS="${RUN_SECONDS}" \
  WORLD_FILE="${WORLD_FILE}" \
  CONFIDENCE_MAP_CONFIG_FILE="${CONFIDENCE_MAP_CONFIG_FILE}" \
  REGION_SCHEDULE_MODE="blocker_aware_acquisition" \
  PROGRESSIVE_SHARED_REPAIR_ENABLED=false \
  FRONTIER_STEERING_ENABLED=false \
  VBC_GATED_FRONTIER_STEP_ENABLED=false \
  ADAPTIVE_REFINEMENT_ENABLED=false \
  PER_SENSOR_HYBRID_ENABLED=true \
  PER_SENSOR_CHECKPOINT="${PER_SENSOR_CHECKPOINT}" \
  PER_SENSOR_BRANCH_ASCENT_STEPS="${PER_SENSOR_BRANCH_ASCENT_STEPS}" \
  PER_SENSOR_MAX_BRANCH_ATTEMPTS="${PER_SENSOR_MAX_BRANCH_ATTEMPTS}" \
  PER_SENSOR_REQUIRE_PRIMITIVE_LOS=true \
  PER_SENSOR_MIN_CONSERVATIVE_G=0.0 \
  TOF_FUSION_ENABLED=true \
  EXECUTION_GCDF_AUDIT_ENABLED=true \
  GCDF_BODY_INFLATION_M=0.015 \
  FORCE_ZERO_INITIAL_Q=true \
  APPLY_INITIAL_JOINT_OVERRIDES=auto \
  ENABLE_ORACLE_DIAGNOSTICS=false \
  EARLY_STOP_ON_GOAL="${EARLY_STOP_ON_GOAL}" \
  GAZEBO_GUI="${GAZEBO_GUI}" \
  USE_RVIZ="${USE_RVIZ}" \
  bash scripts/run_and_pack_phase_e5_execution_gcdf.sh
)
RUN_RC=$?
set -e

echo ""
echo "================ HYBRID TRACE HIGHLIGHTS ================"
if [[ -f "${GEN_LOG}" ]]; then
  grep -E \
    "PER-SENSOR HYBRID|PER-SENSOR BRANCH ACCEPTED|per-sensor branches all rejected|ADD obligation=" \
    "${GEN_LOG}" || true
else
  echo "[WARN] waypoint generator log not found: ${GEN_LOG}"
fi
echo "========================================================="

if [[ -f "${BASE_ZIP}" ]]; then
  cp -f "${BASE_ZIP}" "${UPLOAD_ZIP}"
  echo "[UPLOAD] ${UPLOAD_ZIP}"
  ls -lh "${UPLOAD_ZIP}"
else
  echo "[WARN] base result zip not found: ${BASE_ZIP}"
  echo "[RUN ROOT] ${RUN_ROOT}"
fi

exit "${RUN_RC}"
