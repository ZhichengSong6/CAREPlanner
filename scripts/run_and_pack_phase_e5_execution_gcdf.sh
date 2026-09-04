#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"

# Gazebo/ROS Noetic must run in the system ROS environment. An inherited
# research conda env (e.g. viscdf) can override libstdc++/Qt/Python paths and
# let gzclient start while spawn_model / gazebo_ros_control never becomes
# healthy. GPU/NCDF subprocesses activate their own conda envs later.
if [[ "${CONDA_SHLVL:-0}" =~ ^[0-9]+$ ]] && (( CONDA_SHLVL > 0 )); then
  if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  fi
  _care_conda_before="${CONDA_DEFAULT_ENV:-unknown}"
  while [[ "${CONDA_SHLVL:-0}" =~ ^[0-9]+$ ]] && (( CONDA_SHLVL > 0 )); do
    conda deactivate || break
  done
  echo "[PHASE E5] sanitized inherited conda env (was ${_care_conda_before}); ROS/Gazebo use system environment"
fi

export CASE_FILE="${CASE_FILE:-${REPO}/src/egocentric_arm_planner/config/phase_e_obstacle_core12_v1.json}"
CASE_ID="${CASE_ID:-phase_e_case_000}"
RUN_SECONDS="${RUN_SECONDS:-16}"

export WORLD_FILE="${WORLD_FILE:-${REPO}/src/arm_description/worlds/maixsense_obstacles.world}"
export CONFIDENCE_MAP_CONFIG_FILE="${CONFIDENCE_MAP_CONFIG_FILE:-${REPO}/src/care_confidence_map/config/confidence_map_phase_e_ray.yaml}"
export TOF_FUSION_ENABLED="${TOF_FUSION_ENABLED:-true}"
export EXECUTION_GCDF_AUDIT_ENABLED="${EXECUTION_GCDF_AUDIT_ENABLED:-true}"
export EXECUTION_GCDF_WARNING_MARGIN="${EXECUTION_GCDF_WARNING_MARGIN:-0.05}"
export EXECUTION_GCDF_HARD_MARGIN="${EXECUTION_GCDF_HARD_MARGIN:-0.0}"
export EXECUTION_GCDF_STALE_TIMEOUT_S="${EXECUTION_GCDF_STALE_TIMEOUT_S:-0.35}"
export GCDF_BODY_INFLATION_M="${GCDF_BODY_INFLATION_M:-0.015}"

export GAZEBO_GUI="${GAZEBO_GUI:-false}"
export USE_RVIZ="${USE_RVIZ:-false}"
export EARLY_STOP_ON_GOAL="${EARLY_STOP_ON_GOAL:-false}"

if [[ -z "${RUN_ID:-}" ]]; then
  STAMP="$(date +%Y%m%d-%H%M%S-%3N)"
  SHORT="$(git -C "${REPO}" rev-parse --short=8 HEAD)"
  export RUN_ID="${CASE_ID}_phase_e5_execution_gcdf_${STAMP}_${SHORT}"
fi

echo "[PHASE E5] case_file=${CASE_FILE}"
echo "[PHASE E5] case=${CASE_ID}"
echo "[PHASE E5] measured-q execution GCDF audit enabled=${EXECUTION_GCDF_AUDIT_ENABLED}"
echo "[PHASE E5] warning_margin=${EXECUTION_GCDF_WARNING_MARGIN}"
echo "[PHASE E5] hard_margin=${EXECUTION_GCDF_HARD_MARGIN}"
echo "[PHASE E5] stale_timeout=${EXECUTION_GCDF_STALE_TIMEOUT_S}"
echo "[PHASE E5] body_inflation=${GCDF_BODY_INFLATION_M} m"
echo "[PHASE E5] policy: warning->replan, hard/stale->tracker hold"

CASE_ID="${CASE_ID}" RUN_SECONDS="${RUN_SECONDS}" \
  bash "${REPO}/scripts/run_and_pack_c5_5_vbc_gcdf_regime.sh"
