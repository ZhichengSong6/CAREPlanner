#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
PY="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
SOURCE="${SOURCE:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_offline_labels_v1/smoke_20260921_202241}"
FRESH="${FRESH_STARTS:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_scratch50k_r012_v1/evaluation_fresh_holdout_v1/starts.jsonl}"
BASE="${FULL_V4_BASE:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/full_dataset_v4_v1}"
OUT="${FULL_V4_OUT:-$BASE/run_$(date +%Y%m%d_%H%M%S)_$$}"
[[ -x "$PY" && -f "$SOURCE/manifest.json" && -f "$FRESH" ]] || { echo "[STOP] python/source/fresh starts missing";exit 2; }
mkdir -p "$BASE"
"$PY" -u "$HERE/prepare.py" --source "$SOURCE" --fresh-starts "$FRESH" --out "$OUT" --candidate-q-per-x "${CANDIDATE_Q_PER_X:-128}" --v3-q-per-x "${V3_Q_PER_X:-4}" --anchors-per-sensor "${ANCHORS_PER_SENSOR:-32}" --x-shard-size "${X_SHARD_SIZE:-128}" --seed "${SEED:-260927}"
STATE="$BASE/full_run_$(basename "$OUT").env";printf 'export OUT=%q\nexport BASE=%q\n' "$OUT" "$BASE" >"$STATE";cp "$STATE" "$BASE/latest_full.env"
echo "[prepared] $OUT";echo "[state] $STATE"
