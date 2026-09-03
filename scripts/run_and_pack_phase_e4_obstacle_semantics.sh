#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_ID="${CASE_ID:-case_014}"
RUN_SECONDS="${RUN_SECONDS:-12.0}"

export WORLD_FILE="${WORLD_FILE:-${REPO}/src/arm_description/worlds/maixsense_obstacles.world}"
export CONFIDENCE_MAP_CONFIG_FILE="${CONFIDENCE_MAP_CONFIG_FILE:-${REPO}/src/care_confidence_map/config/confidence_map_phase_e_ray.yaml}"
export TOF_FUSION_ENABLED="${TOF_FUSION_ENABLED:-true}"

# Phase E4 is a semantic smoke by default. GUI may be enabled explicitly for
# Gazebo sanity checking; RViz is not part of formal visualization.
export GAZEBO_GUI="${GAZEBO_GUI:-false}"
export USE_RVIZ="${USE_RVIZ:-false}"
export EARLY_STOP_ON_GOAL="${EARLY_STOP_ON_GOAL:-false}"

if [[ -z "${RUN_ID:-}" ]]; then
  STAMP="$(date +%Y%m%d-%H%M%S-%3N)"
  SHORT="$(git -C "${REPO}" rev-parse --short=8 HEAD)"
  export RUN_ID="${CASE_ID}_phase_e4_obstacle_semantics_${STAMP}_${SHORT}"
fi

echo "[PHASE E4] case=${CASE_ID}"
echo "[PHASE E4] world=${WORLD_FILE}"
echo "[PHASE E4] confidence=${CONFIDENCE_MAP_CONFIG_FILE}"
echo "[PHASE E4] tof_fusion=${TOF_FUSION_ENABLED}"
echo "[PHASE E4] semantics: hard GCDF = UNKNOWN union OCCUPIED"
echo "[PHASE E4] visibility repair authority = UNKNOWN only"

CASE_ID="${CASE_ID}" RUN_SECONDS="${RUN_SECONDS}" \
  bash "${REPO}/scripts/run_and_pack_c5_5_vbc_gcdf_regime.sh"
