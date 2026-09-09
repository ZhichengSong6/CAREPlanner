#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:-${REPO}/src/care_visibility_cdf/data/visibility_yiming_style_grid30_q20000_k500_fovonly.npz}"
OUT="${OUT:-${REPO}/src/care_visibility_cdf/checkpoints/hierarchical9_e2e_fullbatch_seed0}"

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
RESUME="${RESUME:-}"

SHARED_LAYERS="${SHARED_LAYERS:-1024,512,256}"
BRANCH_LAYERS="${BRANCH_LAYERS:-128,128}"

WEIGHT_SDF="${WEIGHT_SDF:-5.0}"
WEIGHT_GRAD="${WEIGHT_GRAD:-0.1}"
WEIGHT_EIKONAL="${WEIGHT_EIKONAL:-0.01}"
WEIGHT_TENSION="${WEIGHT_TENSION:-0.01}"
WEIGHT_SENSOR_OBJECTIVE="${WEIGHT_SENSOR_OBJECTIVE:-1.0}"
WEIGHT_UNION_OBJECTIVE="${WEIGHT_UNION_OBJECTIVE:-1.0}"
WEIGHT_CONSISTENCY="${WEIGHT_CONSISTENCY:-0.1}"

cd "${REPO}"

if [[ ! -f "${DATA}" ]]; then
  echo "[ERROR] dataset missing: ${DATA}" >&2
  exit 2
fi

python -m py_compile \
  src/care_visibility_cdf/scripts/hierarchical_visibility_cdf_model.py \
  src/care_visibility_cdf/scripts/train_hierarchical_visibility_cdf_ddp.py

echo "================================================================"
echo "HIERARCHICAL 9-OUTPUT VISCDF / EXACT 4-GPU DDP"
echo "repo                 : ${REPO}"
echo "data                 : ${DATA}"
echo "out                  : ${OUT}"
echo "steps                : ${STEPS}"
echo "shared trunk         : 30 -> ${SHARED_LAYERS}"
echo "union/sensor branches: 256 -> ${BRANCH_LAYERS} -> 1"
echo "global x batch       : ${GLOBAL_BATCH_X}"
echo "shared q batch       : ${BATCH_Q}"
echo "pairs / update       : $((GLOBAL_BATCH_X * BATCH_Q))"
echo "world size           : ${NPROC_PER_NODE}"
echo "local x / GPU        : $((GLOBAL_BATCH_X / NPROC_PER_NODE))"
echo "microbatch x / GPU   : ${MICROBATCH_X}"
echo "val global x/q       : ${VAL_GLOBAL_BATCH_X} / ${VAL_BATCH_Q}"
echo "weights              : sdf=${WEIGHT_SDF} grad=${WEIGHT_GRAD} eik=${WEIGHT_EIKONAL} tension=${WEIGHT_TENSION}"
echo "objective mix        : sensor=${WEIGHT_SENSOR_OBJECTIVE} union=${WEIGHT_UNION_OBJECTIVE} consistency=${WEIGHT_CONSISTENCY}"
echo "profile              : ${PROFILE}"
echo "resume               : ${RESUME:-none}"
echo "================================================================"

EXTRA=()
if [[ "${PROFILE}" == "1" || "${PROFILE}" == "true" ]]; then
  EXTRA+=(--profile)
fi
if [[ -n "${RESUME}" ]]; then
  EXTRA+=(--resume "${RESUME}")
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  src/care_visibility_cdf/scripts/train_hierarchical_visibility_cdf_ddp.py \
  --data "${DATA}" \
  --urdf src/arm_description/urdf/Arm.urdf \
  --out-dir "${OUT}" \
  --steps "${STEPS}" \
  --global-batch-x "${GLOBAL_BATCH_X}" \
  --batch-q "${BATCH_Q}" \
  --microbatch-x "${MICROBATCH_X}" \
  --val-global-batch-x "${VAL_GLOBAL_BATCH_X}" \
  --val-batch-q "${VAL_BATCH_Q}" \
  --val-microbatch-x "${VAL_MICROBATCH_X}" \
  --decode-x-chunk "${DECODE_X_CHUNK}" \
  --shared-layers "${SHARED_LAYERS}" \
  --branch-layers "${BRANCH_LAYERS}" \
  --weight-sdf "${WEIGHT_SDF}" \
  --weight-grad "${WEIGHT_GRAD}" \
  --weight-eikonal "${WEIGHT_EIKONAL}" \
  --weight-tension "${WEIGHT_TENSION}" \
  --weight-sensor-objective "${WEIGHT_SENSOR_OBJECTIVE}" \
  --weight-union-objective "${WEIGHT_UNION_OBJECTIVE}" \
  --weight-consistency "${WEIGHT_CONSISTENCY}" \
  "${EXTRA[@]}"
