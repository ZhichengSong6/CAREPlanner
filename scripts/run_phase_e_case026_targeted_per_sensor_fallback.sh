#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
DEVICE="${DEVICE:-cuda}"
OUT="${OUT:-${REPO}/outputs/phase_e_case026_targeted_fallback/case026_targeted_per_sensor_fallback.json}"

cd "${REPO}"

if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi
conda activate viscdf

python -m py_compile \
  src/care_visibility_cdf/scripts/per_sensor_visibility_runtime.py \
  scripts/test_phase_e_case026_targeted_per_sensor_fallback.py

python scripts/test_phase_e_case026_targeted_per_sensor_fallback.py \
  --device "${DEVICE}" \
  --scalar-checkpoint \
    src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/final.pt \
  --per-sensor-checkpoint \
    src/care_visibility_cdf/checkpoints/per_sensor_e2e_fullbatch_seed0/final.pt \
  --reference-urdf src/arm_description/urdf/Arm.urdf \
  --self-filter-urdf \
    src/arm_description/urdf/Arm_with_self_filter_collision.urdf \
  --projection-iters 10 \
  --projection-damping 0.5 \
  --projection-epsilon-f 0.03 \
  --projection-max-step-norm 0.25 \
  --root-refine-iters 12 \
  --root-tolerance-f 0.002 \
  --branch-ascent-steps 12 \
  --branch-step-size 0.05 \
  --branch-max-step-norm 0.25 \
  --max-branch-attempts 8 \
  --force-first-sensor 4 \
  --output "${OUT}"

echo ""
echo "[RESULT] ${OUT}"
