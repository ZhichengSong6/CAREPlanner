#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
export GRID25_ROOT="${GRID25_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1}"
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
BASE="$GRID25_ROOT/derived/grid25_v3_full_training_v1"
OUT="$BASE/run_r1_v3_full"
[[ ! -e "$OUT" ]] || { echo "[STOP] Full training output already exists; never overwrite" >&2; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo "[STOP] Git tracked source changes" >&2; exit 2; }
bash "$HERE/run.sh" verify --root "$GRID25_ROOT"
QUEUE="$(squeue -h -u "$USER")" || { echo "[STOP] Cannot check Slurm queue" >&2; exit 2; }
[[ -z "$QUEUE" ]] || { echo "[STOP] Existing own job; wait until bulk production is complete"; printf "%s\n" "$QUEUE"; exit 2; }
mkdir -p "$BASE/logs"
ARGS=(--partition=GPU --nodelist=3090node3 --nodes=1 --ntasks=1
      --gres=gpu:3090:1 --cpus-per-task=4 --time=04:00:00
      --job-name=g25h9_v3full
      "--output=$BASE/logs/slurm_%j.out"
      "--error=$BASE/logs/slurm_%j.err"
      "--export=ALL,GRID25_ROOT=$GRID25_ROOT")
echo "[Slurm test-only] Full Grid25 V3 H9 R1 20000 steps, 1 RTX3090 on 3090node3"
sbatch --test-only "${ARGS[@]}" "$HERE/worker.sbatch"
JOB="$(sbatch --parsable "${ARGS[@]}" "$HERE/worker.sbatch")" || { echo "[STOP] sbatch rejected" >&2; exit 2; }
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "[UNKNOWN] Ambiguous Slurm job ID, do not retry blindly" >&2; exit 2; }
echo "[FULL GRID25 H9 TRAIN SUBMITTED] job=$JOB log=$BASE/logs/slurm_${JOB}.out"
