#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA="${DATA:-${REPO}/src/care_visibility_cdf/data/visibility_yiming_style_grid30_q20000_k500_fovonly.npz}"
OUT="${OUT:-${REPO}/outputs/phase_e_case026_per_sensor_q0_feasibility/case026_per_sensor_q0_feasibility.json}"

cd "${REPO}"

if [[ ! -f "${DATA}" ]]; then
  echo "[ERROR] dataset missing: ${DATA}"
  echo "[HINT] On the training server the historical absolute path was:"
  echo "       /mnt/slurmfs-3090node3/user_data/zsong469/CAREPlanner/src/care_visibility_cdf/data/visibility_yiming_style_grid30_q20000_k500_fovonly.npz"
  exit 3
fi

mkdir -p "$(dirname "${OUT}")"

echo "================================================================"
echo "CASE 026 PER-SENSOR Q0 FEASIBILITY"
echo "repo  : ${REPO}"
echo "data  : ${DATA}"
echo "out   : ${OUT}"
echo "case  : builtin exact obligation-7 target / measured seed / q_vis"
echo "note  : offline only; no ROS/Gazebo/planner is started"
echo "================================================================"

python3 scripts/evaluate_phase_e_case026_per_sensor_q0_feasibility.py \
  --repo "${REPO}" \
  --data "${DATA}" \
  --blocked-sensor 4 \
  --spatial-neighbors 8 \
  --top-boundary-candidates 16 \
  --target-nominal-margin 0.015 \
  --output-json "${OUT}"
