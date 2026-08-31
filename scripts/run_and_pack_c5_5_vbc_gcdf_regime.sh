#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_ID="${CASE_ID:-case_003}"
RUN_SECONDS="${RUN_SECONDS:-30.0}"

export CONFIG_FILE="${REPO}/src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml"
export CARE_WEIGHT="${CARE_WEIGHT:-3000.0}"
export GPU_SOCKET="${GPU_SOCKET:-/tmp/care_collision_cdf_gpu_c5_5.sock}"
export ROOT_OUT="${ROOT_OUT:-${REPO}/outputs/c5_5_vbc_gcdf_regime/${CASE_ID}}"
export ROOT_LOG="${ROOT_LOG:-${REPO}/logs/c5_5_vbc_gcdf_regime/${CASE_ID}}"
export ZIP_PATH="${ZIP_PATH:-${REPO}/CAREPlanner_C5_5_vbc_gcdf_regime_${CASE_ID}.zip}"
export INITIAL_GATE_MAX_TRIES="${INITIAL_GATE_MAX_TRIES:-100}"
export INITIAL_GATE_ECHO_TIMEOUT="${INITIAL_GATE_ECHO_TIMEOUT:-0.20}"

# C5.9 execution semantics: REPAIR/PROBE optimize over a 1 s local horizon but
# build a short executable prefix with the actual braking/hold tail. That exact
# executable view must pass final GCDF and exact VBC before a single commit.
# PROBE is single-flight: another task candidate cannot replace the currently
# verified/executing probe prefix. NORMAL remains full-horizon GCDF/VBC.
export REPAIR_PREFIX_VERIFY="${REPAIR_PREFIX_VERIFY:-1}"
export REPAIR_PREFIX_S="${REPAIR_PREFIX_S:-0.15}"
export REPAIR_BRAKE_DT_S="${REPAIR_BRAKE_DT_S:-0.05}"
export REPAIR_HOLD_S="${REPAIR_HOLD_S:-0.10}"

echo "[C5.5] Existing VBC/regime architecture + hard GCDF safety"
echo "[C5.5] REPAIR: q_vis objective + 5-step executable-horizon hard GCDF + hold initialization"
echo "[C5.5] PROBE_NORMAL: task objective + 5-step executable-horizon hard GCDF + safe-prefix commit"
echo "[C5.5] NORMAL: task objective + full-horizon hard GCDF"
echo "[C5.9] PROBE SINGLE-FLIGHT: IDLE -> VERIFYING -> EXECUTING -> tracker-complete"
echo "[C5.9] COMMIT GATE: exact executable prefix+brake+hold -> final GCDF -> exact VBC -> single commit"
echo "[C5.9] GCDF transport: one GPU client, independent local/final channels, final priority"
echo "[C5.9] EXECUTION: committed trajectory published once; tracker owns execution to completion"
echo "[C5.9] EXECUTION VBC: elapsed suffixes use a separate audit-only topic; tracker is never reset"
echo "[C5.9] PROBE success: tracker complete=1 with matching committed header.stamp token"
echo "[C5.5] REPAIR exit: actual visibility/confidence acquisition gate"
echo "[C5.5] REPAIR exact-VBC view: prefix=${REPAIR_PREFIX_S}s + brake=${REPAIR_BRAKE_DT_S}s + hold=${REPAIR_HOLD_S}s"
echo "[C5.5] NORMAL exact-VBC view: full horizon"
echo "[C5.5] config: ${CONFIG_FILE}"
echo "[C5.5] startup gate wait is fail-fast (~25 s worst case)"

CASE_ID="${CASE_ID}" RUN_SECONDS="${RUN_SECONDS}" \
  bash "${REPO}/scripts/run_and_pack_phase_c5_4_local_sparse_scp.sh"
