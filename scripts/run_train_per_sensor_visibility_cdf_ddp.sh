#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:-${REPO}/src/care_visibility_cdf/data/visibility_yiming_style_grid30_q20000_k500_fovonly.npz}"
OUT="${OUT:-${REPO}/src/care_visibility_cdf/checkpoints/per_sensor_e2e_fullbatch_seed0}"

STEPS="${STEPS:-50000}"
GLOBAL_BATCH_X="${GLOBAL_BATCH_X:-4000}"
BATCH_Q="${BATCH_Q:-100}"
MICROBATCH_X="${MICROBATCH_X:-250}"
VAL_GLOBAL_BATCH_X="${VAL_GLOBAL_BATCH_X:-512}"
VAL_BATCH_Q="${VAL_BATCH_Q:-100}"
VAL_MICROBATCH_X="${VAL_MICROBATCH_X:-128}"
DECODE_X_CHUNK="${DECODE_X_CHUNK:-64}"
PROFILE="${PROFILE:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

cd "${REPO}"

if [[ ! -f "${DATA}" ]]; then
  echo "[ERROR] dataset missing: ${DATA}"
  exit 2
fi

python -m py_compile   src/care_visibility_cdf/scripts/train_per_sensor_visibility_cdf_ddp.py

echo "================================================================"
echo "EXACT FULL-BATCH 4-GPU PER-SENSOR VISCDF"
echo "repo              : ${REPO}"
echo "data              : ${DATA}"
echo "out               : ${OUT}"
echo "steps             : ${STEPS}"
echo "global x batch    : ${GLOBAL_BATCH_X}"
echo "shared q batch    : ${BATCH_Q}"
echo "pairs / update    : $((GLOBAL_BATCH_X * BATCH_Q))"
echo "world size        : ${NPROC_PER_NODE}"
echo "local x / GPU     : $((GLOBAL_BATCH_X / NPROC_PER_NODE))"
echo "microbatch x/GPU  : ${MICROBATCH_X}"
echo "val global x/q    : ${VAL_GLOBAL_BATCH_X} / ${VAL_BATCH_Q}"
echo "profile           : ${PROFILE}"
echo "================================================================"

EXTRA=()
if [[ "${PROFILE}" == "1" || "${PROFILE}" == "true" ]]; then
  EXTRA+=(--profile)
fi

torchrun   --standalone   --nproc_per_node="${NPROC_PER_NODE}"   src/care_visibility_cdf/scripts/train_per_sensor_visibility_cdf_ddp.py   --data "${DATA}"   --urdf src/arm_description/urdf/Arm.urdf   --out-dir "${OUT}"   --steps "${STEPS}"   --global-batch-x "${GLOBAL_BATCH_X}"   --batch-q "${BATCH_Q}"   --microbatch-x "${MICROBATCH_X}"   --val-global-batch-x "${VAL_GLOBAL_BATCH_X}"   --val-batch-q "${VAL_BATCH_Q}"   --val-microbatch-x "${VAL_MICROBATCH_X}"   --decode-x-chunk "${DECODE_X_CHUNK}"   "${EXTRA[@]}"
