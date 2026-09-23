#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO:?}" "${VIS_PYTHON:?}" "${R012_ROOT:?}" "${PAIRED_CACHE:?}" "${OUT:?}"
STEPS="${STEPS:-400}"; BATCH_SIZE="${BATCH_SIZE:-64}"; LR="${LR:-2e-5}"; SEED="${SEED:-20260923}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
arms=(old_value new_value old_value_grad new_value_grad)
ALLOC_VISIBLE="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS="," read -r -a gpus <<< "$ALLOC_VISIBLE"
(( ${#gpus[@]} >= 4 )) || { echo "[STOP] expected four allocated GPUs, got: $ALLOC_VISIBLE"; exit 2; }
pids=()
cleanup(){ for p in "${pids[@]}";do kill -TERM "$p" 2>/dev/null || true;done; }
trap 'cleanup;exit 130' INT TERM
mkdir -p "$OUT/logs"
for i in 0 1 2 3;do
  arm="${arms[$i]}"; gpu="${gpus[$i]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$VIS_PYTHON" -u "$HERE/train_arm.py" \
    --arm "$arm" --r012-root "$R012_ROOT" --cache "$PAIRED_CACHE" --output "$OUT/$arm" \
    --steps "$STEPS" --batch-size "$BATCH_SIZE" --lr "$LR" --seed "$SEED" >"$OUT/logs/$arm.log" 2>&1 &
  pids+=("$!")
done
failed=0
for i in 0 1 2 3;do
  if ! wait "${pids[$i]}";then echo "[ERROR] ${arms[$i]} failed; see log";failed=1;fi
done
((failed==0)) || exit 2
"$VIS_PYTHON" -u "$HERE/compare.py" --root "$OUT"
echo "[done] paired training comparison $OUT"
