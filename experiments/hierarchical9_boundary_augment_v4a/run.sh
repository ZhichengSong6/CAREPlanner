#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${VIS_PYTHON:?}" "${SOURCE:?}" "${FRESH_STARTS:?}" "${OUT:?}"
WORLD="${WORLD:-8}"
"$VIS_PYTHON" -u "$HERE/augment.py" --mode prepare --source "$SOURCE" --fresh-starts "$FRESH_STARTS" --output "$OUT" --world "$WORLD" --train-x "${TRAIN_X:-256}" --val-x "${VAL_X:-64}" --anchors-per-sensor "${ANCHORS_PER_SENSOR:-4}"
pids=();mkdir -p "$OUT/logs"
for ((r=0;r<WORLD;r++));do "$VIS_PYTHON" -u "$HERE/augment.py" --mode work --output "$OUT" --rank "$r" --world "$WORLD" >"$OUT/logs/rank${r}.log" 2>&1 & pids+=("$!");done
bad=0;for p in "${pids[@]}";do wait "$p"||bad=1;done;((bad==0))||exit 2
"$VIS_PYTHON" -u "$HERE/augment.py" --mode merge --output "$OUT" --world "$WORLD"
echo "[done] $OUT"
