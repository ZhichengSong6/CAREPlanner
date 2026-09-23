#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
SOURCE="${SOURCE:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_offline_labels_v1/smoke_20260921_202241}"
FRESH_STARTS="${FRESH_STARTS:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_scratch50k_r012_v1/evaluation_fresh_holdout_v1/starts.jsonl}"
BASE="${AUGMENT_BASE:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_boundary_augment_v4a}"
OUT="${AUGMENT_OUT:-$BASE/pilot_$(date +%Y%m%d_%H%M%S)_$$}";WORLD="${WORLD:-8}"
[[ -x "$VIS_PYTHON" && -f "$SOURCE/manifest.json" && -f "$FRESH_STARTS" ]]||{ echo "[STOP] python/source/fresh starts missing";exit 2; }
mkdir -p "$BASE/logs";RUN_SCRIPT="$HERE/run.sh";export VIS_PYTHON SOURCE FRESH_STARTS OUT WORLD RUN_SCRIPT TRAIN_X VAL_X ANCHORS_PER_SENSOR
RAW="$(sbatch --parsable --partition=GPU --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 --nodelist="${NODE:-3090node1}" --time="${TIME_LIMIT:-02:00:00}" --output="$BASE/logs/augment_%j.out" --export=ALL "$HERE/worker.sbatch")"
JOB="${RAW%%;*}";STATE="$BASE/augment_job_${JOB}.env";printf 'export JOB_ID=%q\nexport OUT=%q\nexport LOG=%q\n' "$JOB" "$OUT" "$BASE/logs/augment_${JOB}.out" >"$STATE";cp "$STATE" "$BASE/latest_augment.env"
echo "[submitted] job=$JOB";echo "[out] $OUT"
