#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${VIS_PYTHON:?}" "${R012_ROOT:?}" "${BF_ROOT:?}" "${STARTS:?}" "${OUT:?}"
WORLD=4
ALLOC="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=, read -r -a GPUS <<< "$ALLOC"
(( ${#GPUS[@]} >= WORLD )) || { echo "[STOP] need four allocated GPUs"; exit 2; }
[[ ! -e "$OUT" ]] || { echo "[STOP] output already exists: $OUT"; exit 2; }
mkdir -p "$OUT"
PIDS=()
for RANK in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${GPUS[$RANK]}" "$VIS_PYTHON" -u "$HERE/evaluate.py" \
    --r012-root "$R012_ROOT" --bf-root "$BF_ROOT/formal" --starts "$STARTS" \
    --output "$OUT" --rank "$RANK" --world "$WORLD" >"$OUT/rank${RANK}.log" 2>&1 &
  PIDS+=("$!")
done
FAIL=0
for PID in "${PIDS[@]}"; do wait "$PID" || FAIL=1; done
if (( FAIL != 0 )); then
  echo "[STOP] one or more evaluator ranks failed"
  for RANK in 0 1 2 3; do
    echo "==== rank${RANK}.log ===="
    tail -n 35 "$OUT/rank${RANK}.log" || true
  done
  exit 2
fi
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$VIS_PYTHON" -u "$HERE/evaluate.py" \
  --r012-root "$R012_ROOT" --bf-root "$BF_ROOT/formal" --starts "$STARTS" \
  --output "$OUT" --world "$WORLD" --merge
"$VIS_PYTHON" -u "$HERE/diagnose.py" --input "$OUT" --output "$OUT/diagnosis"
echo "[done] full 1963 paired cases $OUT"
