#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
VISCDF_PYTHON="${VISCDF_PYTHON:-/home/zhicheng/miniconda3/envs/viscdf/bin/python}"
DEVICE="${DEVICE:-cuda}"
PER_SENSOR_CHECKPOINT="${PER_SENSOR_CHECKPOINT:-${REPO}/src/care_visibility_cdf/checkpoints/per_sensor_e2e_fullbatch_seed0/final.pt}"
OUT="${OUT:-${REPO}/outputs/phase_e_case026_targeted_fallback/case026_targeted_per_sensor_fallback_projection_root_ascent.json}"
SUMMARY_OUT="${SUMMARY_OUT:-${OUT%.json}.md}"
EVALUATE_ALL_BRANCHES="${EVALUATE_ALL_BRANCHES:-true}"

cd "${REPO}"

if [[ ! -x "${VISCDF_PYTHON}" ]]; then
  echo "missing viscdf interpreter: ${VISCDF_PYTHON}" >&2
  exit 2
fi

if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi
conda activate viscdf

EXTRA_ARGS=()
if [[ "${EVALUATE_ALL_BRANCHES}" == "true" || "${EVALUATE_ALL_BRANCHES}" == "1" ]]; then
  EXTRA_ARGS+=(--evaluate-all-branches)
fi

"${VISCDF_PYTHON}" -m py_compile \
  src/care_visibility_cdf/scripts/per_sensor_visibility_runtime.py \
  scripts/test_phase_e_case026_targeted_per_sensor_fallback.py

"${VISCDF_PYTHON}" scripts/test_phase_e_case026_targeted_per_sensor_fallback.py \
  --device "${DEVICE}" \
  --scalar-checkpoint \
    src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/final.pt \
  --per-sensor-checkpoint \
    "${PER_SENSOR_CHECKPOINT}" \
  --reference-urdf src/arm_description/urdf/Arm.urdf \
  --self-filter-urdf \
    src/arm_description/urdf/Arm_with_self_filter_collision.urdf \
  --projection-iters 10 \
  --projection-damping 0.5 \
  --projection-epsilon-f 0.03 \
  --projection-max-step-norm 0.25 \
  --root-refine-iters 12 \
  --root-tolerance-f 0.002 \
  --branch-ascent-steps 1 \
  --branch-step-size 0.05 \
  --branch-max-step-norm 0.25 \
  --max-branch-attempts 8 \
  --force-first-sensor 4 \
  --output "${OUT}" \
  --summary-output "${SUMMARY_OUT}" \
  "${EXTRA_ARGS[@]}"

echo ""
echo "[RESULT] ${OUT}"
echo "[SUMMARY] ${SUMMARY_OUT}"
