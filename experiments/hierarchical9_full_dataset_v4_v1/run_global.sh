#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${OUT:?}" "${REPO:?}" "${VIS_PYTHON:?}"
WORLD="${WORLD:-4}";mkdir -p "$OUT/logs"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
pids=()
for ((r=0;r<WORLD;r++));do "$VIS_PYTHON" -u "$HERE/global_pool.py" --out "$OUT" --device "cuda:$r" --rank "$r" --world "$WORLD" >"$OUT/logs/global_${r}.log" 2>&1 & pids+=("$!");done
bad=0;for p in "${pids[@]}";do wait "$p"||bad=1;done;((bad==0))||exit 2
"$VIS_PYTHON" -u "$HERE/global_pool.py" --out "$OUT" --merge
