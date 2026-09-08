#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:-${REPO}/src/care_visibility_cdf/data/visibility_yiming_style_grid30_q20000_k500_fovonly.npz}"
OUT="${OUT:-${REPO}/src/care_visibility_cdf/checkpoints/exp_per_sensor_yiming_k500_fov_signed}"

STEPS="${STEPS:-50000}"
BATCH_X="${BATCH_X:-512}"
BATCH_Q="${BATCH_Q:-64}"
VAL_BATCH_X="${VAL_BATCH_X:-256}"
VAL_BATCH_Q="${VAL_BATCH_Q:-64}"
DECODE_X_CHUNK="${DECODE_X_CHUNK:-64}"
NEAR_ZERO_RATIO="${NEAR_ZERO_RATIO:-0.0}"
PROFILE="${PROFILE:-0}"
WANDB="${WANDB:-0}"

cd "${REPO}"

if [[ ! -f "${DATA}" ]]; then
  echo "[ERROR] dataset missing: ${DATA}"
  exit 2
fi

python -m py_compile   src/care_visibility_cdf/scripts/train_per_sensor_visibility_cdf.py   src/care_visibility_cdf/scripts/evaluate_per_sensor_visibility_cdf.py

python - <<'PY'
import torch
print("[ENV] torch:", torch.__version__)
print("[ENV] cuda runtime:", torch.version.cuda)
print("[ENV] cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[ENV] gpu:", torch.cuda.get_device_name(0))
    p=torch.cuda.get_device_properties(0)
    print("[ENV] vram_gib:", round(p.total_memory/1024**3,2))
PY

echo "================================================================"
echo "PER-SENSOR VISCDF END-TO-END TRAINING"
echo "repo       : ${REPO}"
echo "data       : ${DATA}"
echo "out        : ${OUT}"
echo "steps      : ${STEPS}"
echo "batch x/q  : ${BATCH_X} / ${BATCH_Q}"
echo "val x/q    : ${VAL_BATCH_X} / ${VAL_BATCH_Q}"
echo "decode     : ${DECODE_X_CHUNK}"
echo "near-zero  : ${NEAR_ZERO_RATIO}"
echo "profile    : ${PROFILE}"
echo "wandb      : ${WANDB}"
echo "semantics  : 8 explicit sensor-specific signed CDF heads"
echo "init       : random, full end-to-end training"
echo "================================================================"

EXTRA=()
if [[ "${PROFILE}" == "1" || "${PROFILE}" == "true" ]]; then
  EXTRA+=(--profile)
fi
if [[ "${WANDB}" == "1" || "${WANDB}" == "true" ]]; then
  EXTRA+=(--wandb)
fi

python -u src/care_visibility_cdf/scripts/train_per_sensor_visibility_cdf.py   --data "${DATA}"   --urdf src/arm_description/urdf/Arm.urdf   --out-dir "${OUT}"   --steps "${STEPS}"   --batch-x "${BATCH_X}"   --batch-q "${BATCH_Q}"   --val-batch-x "${VAL_BATCH_X}"   --val-batch-q "${VAL_BATCH_Q}"   --decode-x-chunk "${DECODE_X_CHUNK}"   --near-zero-ratio "${NEAR_ZERO_RATIO}"   "${EXTRA[@]}"

echo ""
echo "[DONE] checkpoints:"
ls -lh "${OUT}"/best.pt "${OUT}"/final.pt 2>/dev/null || true
