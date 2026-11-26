#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
export GRID25_ROOT="${GRID25_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1}"
BASE="$GRID25_ROOT/derived/grid25_v3_boundary_pairs_v1"
OUT="$BASE/run_r1_v3_pairs_aug"
[[ -f "$BASE/augmentation_manifest.json" ]] || { echo "[STOP] New true V3 boundary labels not merged; run merge first" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "[STOP] Third H9 run already exists; no overwrites" >&2; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo "[STOP] Local tracked changes" >&2; exit 2; }
"$VIS_PYTHON" -m py_compile "$HERE/train.py"
QUEUE="$(squeue -h -u "$USER")"
[[ -z "$QUEUE" ]] || { echo "[STOP] Existing own job"; printf "%s\n" "$QUEUE"; exit 2; }
mkdir -p "$BASE/logs"
ARGS=(--partition=GPU --nodelist=3090node3 --nodes=1 --ntasks=1
      --gres=gpu:3090:1 --cpus-per-task=4 --time=04:00:00
      --job-name=g25h9_v3pairs
      "--output=$BASE/logs/train_%j.out"
      "--error=$BASE/logs/train_%j.err"
      "--export=ALL,GRID25_ROOT=$GRID25_ROOT")
echo "[THIRD ROUND H9 TRAIN] Existing 124-segment V3 + newly qualified two-sided V3, 20000 steps"
sbatch --test-only "${ARGS[@]}" "$HERE/train_worker.sbatch"
JOB="$(sbatch --parsable "${ARGS[@]}" "$HERE/train_worker.sbatch")"
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "[STOP] Ambiguous Job ID; do not resubmit blindly"; exit 2; }
echo "[THIRD ROUND H9 TRAIN SUBMITTED] job=$JOB log=$BASE/logs/train_${JOB}.out"
