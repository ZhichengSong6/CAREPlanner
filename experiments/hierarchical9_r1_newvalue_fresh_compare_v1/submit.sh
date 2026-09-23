#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
R012_ROOT="${R012_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_scratch50k_r012_v1}"
PAIR_ROOT="${PAIR_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_paired_training_v1}"
[[ -f "$PAIR_ROOT/latest_train.env" ]] || { echo "[STOP] latest paired-training state missing";exit 2; }
source "$PAIR_ROOT/latest_train.env"
PAIR_RUN="$OUT"
CANDIDATE="${CANDIDATE:-$PAIR_RUN/new_value/final.pt}"
STARTS="${STARTS:-$R012_ROOT/evaluation_fresh_holdout_v1/starts.jsonl}"
BASE="${COMPARE_BASE:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_newvalue_fresh_compare_v1}"
OUT="$BASE/run_$(date +%Y%m%d_%H%M%S)_$$"
[[ -x "$VIS_PYTHON" && -f "$CANDIDATE" && -f "$STARTS" ]] || { echo "[STOP] python/candidate/starts missing";exit 2; }
mkdir -p "$BASE/logs"
RUN_SCRIPT="$HERE/run.sh";export VIS_PYTHON R012_ROOT CANDIDATE STARTS OUT RUN_SCRIPT
RAW="$(sbatch --parsable --partition=GPU --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:4 --nodelist="${NODE:-3090node1}" --time="${TIME_LIMIT:-04:00:00}" --output="$BASE/logs/compare_%j.out" --export=ALL "$HERE/worker.sbatch")"
JOB="${RAW%%;*}";[[ "$JOB" =~ ^[0-9]+$ ]]||exit 2
STATE="$BASE/compare_job_${JOB}.env"
printf 'export JOB_ID=%q\nexport OUT=%q\nexport LOG=%q\n' "$JOB" "$OUT" "$BASE/logs/compare_${JOB}.out" >"$STATE"
cp "$STATE" "$BASE/latest_compare.env"
echo "[submitted] job=$JOB";echo "[out] $OUT";echo "[candidate] $CANDIDATE";echo "[starts] $STARTS"
