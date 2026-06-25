#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash src/care_visibility_cdf/scripts/launch_m4_per_sensor_multigpu.sh 8
# or:
#   NGPU=8 bash src/care_visibility_cdf/scripts/launch_m4_per_sensor_multigpu.sh

NGPU="${1:-${NGPU:-8}}"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
OUTDIR="${OUTDIR:-src/care_visibility_cdf/data}"
BASE="${BASE:-visibility_q0_occ_grid30_full_lbfgs_global_sensor_q20000_k100_psk100_gpuocc}"
SCRIPT="${SCRIPT:-src/care_visibility_cdf/scripts/extract_visibility_zero_level_sets_multigpu.py}"

cd "$PROJECT_ROOT"
mkdir -p "$OUTDIR"

echo "Launching ${NGPU} shards from ${PROJECT_ROOT}"
echo "Output base: ${OUTDIR}/${BASE}"

for SID in $(seq 0 $((NGPU - 1))); do
  LOG="${OUTDIR}/m4_${BASE}_shard${SID}of${NGPU}.log"
  OUT="${OUTDIR}/${BASE}_shard${SID}of${NGPU}.npz"
  echo "start shard ${SID}/${NGPU}: GPU=${SID}, log=${LOG}"
  CUDA_VISIBLE_DEVICES=${SID} python -u "$SCRIPT" \
    --urdf src/arm_description/urdf/Arm.urdf \
    --raw src/care_visibility_cdf/data/visibility_raw_grid30x30x24_q100_occ.npz \
    --output "$OUT" \
    --occlusion-urdf src/care_visibility_cdf/data/Arm_self_occlusion_collision_clean.urdf \
    --occlusion-backend torch \
    --p-count 0 \
    --p-num-shards "$NGPU" \
    --p-shard-id "$SID" \
    --q-init-per-p 20000 \
    --optimize-all-initial \
    --target-q0-per-p 100 \
    --extract-per-sensor \
    --per-sensor-target-q0-per-p 100 \
    --epsilon 0.001 \
    --max-iter 50 \
    --zero-level-optimizer lbfgs \
    --lbfgs-lr 1.0 \
    --device cuda \
    --seed $((2 + SID)) \
    --p-seed 2 \
    --progress-every 10 \
    > "$LOG" 2>&1 &
done

wait

echo "All shards finished. Merge with:"
echo "python src/care_visibility_cdf/scripts/merge_visibility_q0_shards.py --pattern '${OUTDIR}/${BASE}_shard*of${NGPU}.npz' --output '${OUTDIR}/${BASE}.npz' --expected-p-count 21600 --require-contiguous"
