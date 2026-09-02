#!/usr/bin/env bash
set -euo pipefail

# One-command A/B visualization:
#   task-only nominal baseline vs an existing full CAREPlanner run.
#
# Optional:
#   CASE_ID=case_014
#   RUN_SECONDS=25
#   CARE_RUN_DIR=/.../c5_5.../run
#   OUTPUT_DIR=/.../visualization
#
# If CARE_RUN_DIR is omitted, the newest matching full CAREPlanner run is used.

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_ID="${CASE_ID:-case_014}"
RUN_SECONDS="${RUN_SECONDS:-25.0}"
CARE_RUN_DIR="${CARE_RUN_DIR:-}"
PLAYBACK_SPEED="${PLAYBACK_SPEED:-2.0}"
FPS="${FPS:-20}"

cd "${REPO}"
source devel/setup.bash

if [[ -z "${CARE_RUN_DIR}" ]]; then
  CARE_RUN_DIR="$(find "${REPO}/outputs/c5_5_vbc_gcdf_regime" \
    -mindepth 2 -maxdepth 2 -type d -name run \
    -path "*/${CASE_ID}_careplanner_full_*/*" \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)"
fi

if [[ -z "${CARE_RUN_DIR}" || ! -f "${CARE_RUN_DIR}/joint_states.csv" ]]; then
  echo "[ERROR] Could not find a CAREPlanner run for ${CASE_ID}." >&2
  echo "Set CARE_RUN_DIR explicitly to a .../run directory containing joint_states.csv." >&2
  exit 2
fi

echo "[A/B] CAREPlanner run: ${CARE_RUN_DIR}"
echo "[A/B] Running clean task-only baseline..."

CASE_ID="${CASE_ID}" RUN_SECONDS="${RUN_SECONDS}" \
  bash scripts/run_phase_d_nominal_baseline.sh

LATEST_FILE="${REPO}/outputs/phase_d_nominal_baseline/latest_${CASE_ID}.txt"
if [[ ! -f "${LATEST_FILE}" ]]; then
  echo "[ERROR] baseline runner did not write ${LATEST_FILE}" >&2
  exit 3
fi
BASELINE_RUN_DIR="$(cat "${LATEST_FILE}")"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO}/outputs/phase_d_visualization/${CASE_ID}_ab_${STAMP}}"

python3 scripts/visualize_careplanner_ab.py \
  --case-id "${CASE_ID}" \
  --baseline-run-dir "${BASELINE_RUN_DIR}" \
  --care-run-dir "${CARE_RUN_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --playback-speed "${PLAYBACK_SPEED}" \
  --fps "${FPS}"

echo ""
echo "================ A/B VISUALIZATION COMPLETE ================"
echo "[BASELINE RUN] ${BASELINE_RUN_DIR}"
echo "[CARE RUN]     ${CARE_RUN_DIR}"
echo "[OUTPUT]       ${OUTPUT_DIR}"
echo "[EE PATH]      ${OUTPUT_DIR}/ee_path_3d.png"
echo "[JOINTS]       ${OUTPUT_DIR}/joint_motion_comparison.png"
echo "[DIFFERENCE]   ${OUTPUT_DIR}/motion_difference.png"
if [[ -f "${OUTPUT_DIR}/robot_motion_comparison.mp4" ]]; then
  echo "[VIDEO]        ${OUTPUT_DIR}/robot_motion_comparison.mp4"
elif [[ -f "${OUTPUT_DIR}/robot_motion_comparison.gif" ]]; then
  echo "[VIDEO]        ${OUTPUT_DIR}/robot_motion_comparison.gif"
fi
echo "[SUMMARY]      ${OUTPUT_DIR}/comparison_summary.json"
