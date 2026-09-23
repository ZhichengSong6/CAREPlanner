#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
BASE="${FULL_V4_BASE:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/full_dataset_v4_v1}"
STATE="${FULL_STATE:-$BASE/latest_full.env}";[[ -f "$STATE" ]]||{ echo "[STOP] run prepare_full.sh first";exit 2; };source "$STATE"
WORLD="${WORLD:-4}";mkdir -p "$BASE/logs";RUN_SCRIPT="$HERE/run_global.sh";export REPO VIS_PYTHON OUT BASE WORLD RUN_SCRIPT
RAW="$(sbatch --parsable --partition=GPU --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:4 --nodelist="${NODE:-3090node1}" --time="${TIME_LIMIT:-04:00:00}" --output="$BASE/logs/global_%j.out" --export=ALL "$HERE/worker.sbatch")"
JOB="${RAW%%;*}";JSTATE="$BASE/global_job_${JOB}.env";printf 'export JOB_ID=%q\nexport OUT=%q\nexport LOG=%q\nexport STAGE=%q\n' "$JOB" "$OUT" "$BASE/logs/global_${JOB}.out" global >"$JSTATE";cp "$JSTATE" "$BASE/latest_global.env"
echo "[submitted] job=$JOB stage=global";echo "[out] $OUT"
