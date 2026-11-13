#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
export GRID25_ROOT="${GRID25_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1}"
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
TRAIN_DATA="$GRID25_ROOT/derived/grid25_v3_first_train_v1/dataset_segment_000"
TRAIN_OUT="$GRID25_ROOT/derived/grid25_v3_first_train_v1/run_000"
[[ -f "$TRAIN_DATA/labels.npz" && -f "$TRAIN_DATA/manifest.json" ]] || { echo "[STOP] First run prepare, training snapshot missing" >&2; exit 2; }
[[ ! -e "$TRAIN_OUT" ]] || { echo "[STOP] Training run output exists: $TRAIN_OUT; do not overwrite/retry blindly" >&2; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo "[STOP] Tracked code has local uncommitted edits" >&2; exit 2; }
bash "$HERE/run.sh" verify --root "$GRID25_ROOT"
command -v squeue >/dev/null && command -v sbatch >/dev/null || { echo "[STOP] Slurm unavailable" >&2; exit 2; }
MY_QUEUE="$(squeue -h -u "$USER")" || { echo "[STOP] Cannot query own Slurm jobs" >&2; exit 2; }
[[ -z "$MY_QUEUE" ]] || { echo "[STOP] Finish own existing Slurm job first"; printf "%s\n" "$MY_QUEUE"; exit 2; }
mkdir -p "$GRID25_ROOT/derived/grid25_v3_first_train_v1/logs"
LOGDIR="$GRID25_ROOT/derived/grid25_v3_first_train_v1/logs"
ARGS=(--partition=GPU --nodelist=3090node3 --nodes=1 --ntasks=1
      --gres=gpu:3090:1 --cpus-per-task=4 --time=04:00:00
      --job-name=g25_h9v3
      "--output=$LOGDIR/slurm_%j.out" "--error=$LOGDIR/slurm_%j.err"
      "--export=ALL,TRAIN_DATA=$TRAIN_DATA,TRAIN_OUT=$TRAIN_OUT")
echo "[GPU dry-run] 3090node3 one RTX 3090; 4 CPUs; 4h limit"
sbatch --test-only "${ARGS[@]}" "$HERE/worker.sbatch"
echo "[Submit] One 1000-step H9 R1 V3 signed-distance training run"
JOB="$(sbatch --parsable "${ARGS[@]}" "$HERE/worker.sbatch")" || { echo "[STOP] Slurm submission failed" >&2; exit 2; }
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "[STOP] Ambiguous JobID $JOB; DO NOT RETRY without checking squeue" >&2; exit 2; }
echo "[GRID25 H9 V3 TRAIN SUBMITTED] job=$JOB output=$TRAIN_OUT logs=$LOGDIR/slurm_${JOB}.out"
