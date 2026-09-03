#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_ID="${CASE_ID:-case_014}"
RUN_SECONDS="${RUN_SECONDS:-16}"

export WORLD_FILE="${WORLD_FILE:-${REPO}/src/arm_description/worlds/maixsense_obstacles.world}"
export CONFIDENCE_MAP_CONFIG_FILE="${CONFIDENCE_MAP_CONFIG_FILE:-${REPO}/src/care_confidence_map/config/confidence_map_phase_e_ray.yaml}"
export TOF_FUSION_ENABLED="${TOF_FUSION_ENABLED:-true}"
export EXECUTION_GCDF_AUDIT_ENABLED="${EXECUTION_GCDF_AUDIT_ENABLED:-true}"
export EXECUTION_GCDF_WARNING_MARGIN="${EXECUTION_GCDF_WARNING_MARGIN:-0.05}"
export EXECUTION_GCDF_HARD_MARGIN="${EXECUTION_GCDF_HARD_MARGIN:-0.0}"
export EXECUTION_GCDF_STALE_TIMEOUT_S="${EXECUTION_GCDF_STALE_TIMEOUT_S:-0.35}"

export GAZEBO_GUI="${GAZEBO_GUI:-false}"
export USE_RVIZ="${USE_RVIZ:-false}"
export EARLY_STOP_ON_GOAL="${EARLY_STOP_ON_GOAL:-false}"

if [[ -z "${RUN_ID:-}" ]]; then
  STAMP="$(date +%Y%m%d-%H%M%S-%3N)"
  SHORT="$(git -C "${REPO}" rev-parse --short=8 HEAD)"
  export RUN_ID="${CASE_ID}_phase_e5_execution_gcdf_${STAMP}_${SHORT}"
fi

echo "[PHASE E5] case=${CASE_ID}"
echo "[PHASE E5] measured-q execution GCDF audit enabled=${EXECUTION_GCDF_AUDIT_ENABLED}"
echo "[PHASE E5] warning_margin=${EXECUTION_GCDF_WARNING_MARGIN}"
echo "[PHASE E5] hard_margin=${EXECUTION_GCDF_HARD_MARGIN}"
echo "[PHASE E5] stale_timeout=${EXECUTION_GCDF_STALE_TIMEOUT_S}"
echo "[PHASE E5] policy: warning->replan, hard/stale->tracker hold"

CASE_ID="${CASE_ID}" RUN_SECONDS="${RUN_SECONDS}" \
  bash "${REPO}/scripts/run_and_pack_c5_5_vbc_gcdf_regime.sh"
