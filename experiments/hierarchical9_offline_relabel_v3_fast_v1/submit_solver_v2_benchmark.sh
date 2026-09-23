#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
SOURCE="${SOURCE:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_offline_labels_v3/pilot_20260922_115134}"
BASE="${V3_FAST_BASE:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_v3_fast_v1}";OUT="${BENCH_OUT:-$BASE/solver_v2_$(date +%Y%m%d_%H%M%S)_$$}";WORLD="${WORLD:-16}"
[[ -x "$VIS_PYTHON" && -f "$SOURCE/paired_cache/dataset_index.json" ]]||{ echo "[STOP] python/source missing";exit 2; }
mkdir -p "$BASE/logs";RUN_SCRIPT="$HERE/run_solver_v2_benchmark.sh";export REPO VIS_PYTHON SOURCE BASE OUT WORLD RUN_SCRIPT PER_STRATUM
RAW="$(sbatch --parsable --partition=GPU --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 --nodelist="${NODE:-3090node1}" --time="${TIME_LIMIT:-02:00:00}" --output="$BASE/logs/solver_v2_%j.out" --export=ALL "$HERE/worker.sbatch")"
JOB="${RAW%%;*}";STATE="$BASE/solver_v2_job_${JOB}.env";printf 'export JOB_ID=%q\nexport OUT=%q\nexport LOG=%q\n' "$JOB" "$OUT" "$BASE/logs/solver_v2_${JOB}.out" >"$STATE";cp "$STATE" "$BASE/latest_solver_v2.env";echo "[submitted] job=$JOB";echo "[out] $OUT"
