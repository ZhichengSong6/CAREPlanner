#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
RUN="${RUN:-${REPO}/outputs/c5_5_vbc_gcdf_regime/phase_e_goal_026_qvis_validity_20260908-145717_48c65db9/run}"
DATA="${DATA:-${REPO}/src/care_visibility_cdf/data/visibility_yiming_style_grid30_q20000_k500_fovonly.npz}"

cd "${REPO}"

TRACE="${TRACE:-}"
if [[ -z "${TRACE}" ]]; then
  mapfile -t traces < <(find "${RUN}/projector_traces" -maxdepth 1 -type f \
    -name 'c46_obligation_007_*.json' | sort)
  if (( ${#traces[@]} == 0 )); then
    echo "[ERROR] obligation-7 trace not found under:"
    echo "        ${RUN}/projector_traces"
    echo "Set RUN=... or TRACE=... explicitly."
    exit 2
  fi
  TRACE="${traces[${#traces[@]}-1]}"
fi

if [[ ! -f "${DATA}" ]]; then
  echo "[ERROR] dataset missing: ${DATA}"
  exit 3
fi

OUT="${OUT:-${RUN}/case026_per_sensor_q0_feasibility.json}"

echo "================================================================"
echo "CASE 026 PER-SENSOR Q0 FEASIBILITY"
echo "trace : ${TRACE}"
echo "data  : ${DATA}"
echo "out   : ${OUT}"
echo "note  : offline only; no ROS/Gazebo/planner is started"
echo "================================================================"

python3 scripts/evaluate_phase_e_case026_per_sensor_q0_feasibility.py \
  --repo "${REPO}" \
  --trace "${TRACE}" \
  --data "${DATA}" \
  --blocked-sensor 4 \
  --spatial-neighbors 8 \
  --top-boundary-candidates 16 \
  --target-nominal-margin 0.015 \
  --output-json "${OUT}"
