#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${OUT:?}" "${VIS_PYTHON:?}" "${STAGE_IMPL:?}"
NODE_INDEX="${SLURM_PROCID:?}"
LOCAL_WORKERS="${LOCAL_WORKERS:-16}"
WORLD="$((LOCAL_WORKERS * SLURM_NTASKS))"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
pids=()
for ((l=0;l<LOCAL_WORKERS;l++));do
  rank="$((NODE_INDEX * LOCAL_WORKERS + l))"
  "$VIS_PYTHON" -u "$HERE/$STAGE_IMPL" --out "$OUT" --rank "$rank" --world "$WORLD" \
    >"$OUT/logs/${STAGE_NAME}_node${NODE_INDEX}_rank${rank}.log" 2>&1 &
  pids+=("$!")
done
bad=0
for p in "${pids[@]}";do wait "$p" || bad=1;done
((bad==0)) || exit 2
echo "[node done] $STAGE_NAME node_index=$NODE_INDEX world=$WORLD"
