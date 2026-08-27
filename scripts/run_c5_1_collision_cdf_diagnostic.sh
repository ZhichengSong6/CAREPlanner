#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CDF_ENV="${CDF_ENV:-ncdf_l4c}"
DEVICE="${DEVICE:-cpu}"
CHECKPOINT="${CHECKPOINT:-${REPO}/src/care_collision_cdf/checkpoints/yiming_cdf/model_dict.pt}"
OUT="${OUT:-${REPO}/outputs/c5_1_collision_cdf_diagnostic.json}"

cd "${REPO}"

if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] conda.sh not found"
  exit 2
fi

echo "[C5.1] branch: $(git branch --show-current)"
echo "[C5.1] head: $(git rev-parse HEAD)"
echo "[C5.1] env: ${CDF_ENV}"
echo "[C5.1] device: ${DEVICE}"
echo "[C5.1] checkpoint: ${CHECKPOINT}"

bash -lc "
  set -e
  source '${CONDA_SH}'
  conda activate '${CDF_ENV}'
  cd '${REPO}'
  exec python -u src/care_collision_cdf/scripts/evaluate_collision_cdf_gradient.py \
    --checkpoint '${CHECKPOINT}' \
    --device '${DEVICE}' \
    --output '${OUT}'
"

echo "[C5.1 RESULT] ${OUT}"
