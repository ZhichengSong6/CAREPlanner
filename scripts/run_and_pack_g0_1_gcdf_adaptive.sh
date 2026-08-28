#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_ID="${CASE_ID:-case_003}"
RUN_SECONDS="${RUN_SECONDS:-20.0}"

export CONFIG_FILE="${REPO}/src/egocentric_arm_planner/config/planner_g0_1_gcdf_adaptive.yaml"
export CARE_WEIGHT="0.0"
export GPU_SOCKET="${GPU_SOCKET:-/tmp/care_collision_cdf_gpu_g0_1.sock}"
export ROOT_OUT="${ROOT_OUT:-${REPO}/outputs/g0_1_gcdf_adaptive/${CASE_ID}}"
export ROOT_LOG="${ROOT_LOG:-${REPO}/logs/g0_1_gcdf_adaptive/${CASE_ID}}"
export ZIP_PATH="${ZIP_PATH:-${REPO}/CAREPlanner_G0_1_gcdf_adaptive_${CASE_ID}.zip}"

echo "[G0.1] GCDF-aligned baseline + adaptive slack penalty"
echo "[G0.1] low-confidence voxels are the forbidden/obstacle set"
echo "[G0.1] config: ${CONFIG_FILE}"
echo "[G0.1] CARE visibility waypoint weight: ${CARE_WEIGHT}"
echo "[G0.1] mu schedule: 10 -> ... -> 1e8 while max_slack > 0.005"

CASE_ID="${CASE_ID}" RUN_SECONDS="${RUN_SECONDS}"   bash "${REPO}/scripts/run_and_pack_phase_c5_4_local_sparse_scp.sh"
