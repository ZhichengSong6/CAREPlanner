#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
export GRID25_ROOT="${GRID25_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1}"
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
BASE="$GRID25_ROOT/derived/grid25_v3_8segment_train_v1"
DATA="$BASE/dataset_000_007"
OUT="$BASE/run_r1_v3"
[[ -f "$DATA/manifest.json" && -f "$DATA/labels.npz" ]] || { echo "[STOP] Prepare 8-segment V3 dataset first" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "[STOP] Training output exists; inspect status, do not overwrite" >&2; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo "[STOP] Tracked git changes exist" >&2; exit 2; }
bash "$HERE/run.sh" verify --root "$GRID25_ROOT"
command -v squeue >/dev/null && command -v sbatch >/dev/null || { echo "[STOP] Slurm not available" >&2; exit 2; }
QUEUE="$(squeue -h -u "$USER")" || { echo "[STOP] Cannot check own queue" >&2; exit 2; }
[[ -z "$QUEUE" ]] || { echo "[STOP] Own queue busy; finish first"; printf "%s\n" "$QUEUE"; exit 2; }
mkdir -p "$BASE/logs"
ARGS=(--partition=GPU --nodelist=3090node3 --nodes=1 --ntasks=1
      --gres=gpu:3090:1 --cpus-per-task=4 --time=04:00:00
      --job-name=g25_h9_8seg
      "--output=$BASE/logs/slurm_%j.out" "--error=$BASE/logs/slurm_%j.err"
      "--export=ALL,GRID25_ROOT=$GRID25_ROOT")
echo "[Slurm test-only] H9 V3 8-segment training, 3090node3 1 RTX3090, 4 CPUs"
sbatch --test-only "${ARGS[@]}" "$HERE/worker.sbatch"
echo "[Submit] One single-GPU Grid25 V3-only 8-segment H9 training job"
JOB="$(sbatch --parsable "${ARGS[@]}" "$HERE/worker.sbatch")" || { echo "[STOP] Slurm submission failed" >&2; exit 2; }
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "[UNKNOWN] Slurm response $JOB; DO NOT RETRY blindly" >&2; exit 2; }
echo "[H9 EIGHTSEG TRAIN SUBMITTED] job=$JOB out=$OUT log=$BASE/logs/slurm_${JOB}.out"
