#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${VIS_PYTHON:?}" "${R012_ROOT:?}" "${CANDIDATE:?}" "${STARTS:?}" "${OUT:?}"
WORLD=4
ALLOC="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS="," read -r -a gpus <<< "$ALLOC"
(( ${#gpus[@]} >= 4 )) || { echo "[STOP] need 4 allocated GPUs";exit 2; }
pids=()
mkdir -p "$(dirname "$OUT")"
for r in 0 1 2 3;do CUDA_VISIBLE_DEVICES="${gpus[$r]}" "$VIS_PYTHON" -u "$HERE/evaluate.py" --r012-root "$R012_ROOT" --candidate "$CANDIDATE" --starts "$STARTS" --output "$OUT" --rank "$r" --world "$WORLD" >"$(dirname "$OUT")/rank${r}.log" 2>&1 & pids+=("$!");done
bad=0;for p in "${pids[@]}";do wait "$p"||bad=1;done;((bad==0))||exit 2
CUDA_VISIBLE_DEVICES="${gpus[0]}" "$VIS_PYTHON" -u "$HERE/evaluate.py" --r012-root "$R012_ROOT" --candidate "$CANDIDATE" --starts "$STARTS" --output "$OUT" --world "$WORLD" --merge
echo "[done] $OUT"
