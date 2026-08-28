#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_ID="${CASE_ID:-case_003}"
RUN_SECONDS="${RUN_SECONDS:-20.0}"

export CONFIG_FILE="${REPO}/src/egocentric_arm_planner/config/planner_g0_2_gcdf_hard_feasibility.yaml"
export CARE_WEIGHT="0.0"
export GPU_SOCKET="${GPU_SOCKET:-/tmp/care_collision_cdf_gpu_g0_2.sock}"
export ROOT_OUT="${ROOT_OUT:-${REPO}/outputs/g0_2_gcdf_hard_feasibility/${CASE_ID}}"
export ROOT_LOG="${ROOT_LOG:-${REPO}/logs/g0_2_gcdf_hard_feasibility/${CASE_ID}}"
export ZIP_PATH="${ZIP_PATH:-${REPO}/CAREPlanner_G0_2_gcdf_hard_feasibility_${CASE_ID}.zip}"

echo "[G0.2] Hard-CDF feasibility diagnostic"
echo "[G0.2] low-confidence voxels are hard forbidden/obstacle space"
echo "[G0.2] user CDF slacks: OFF"
echo "[G0.2] CARE visibility waypoint weight: ${CARE_WEIGHT}"
echo "[G0.2] config: ${CONFIG_FILE}"

CASE_ID="${CASE_ID}" RUN_SECONDS="${RUN_SECONDS}"   bash "${REPO}/scripts/run_and_pack_phase_c5_4_local_sparse_scp.sh"
