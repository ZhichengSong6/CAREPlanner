#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
export GRID25_ROOT="${GRID25_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1}"
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
BASE="$GRID25_ROOT/derived/grid25_v3_v4_joint_train_v1"
DATA="$BASE/tube_snapshot_000"
OUT="$BASE/run_000"
[[ -f "$DATA/tubes.npz" && -f "$DATA/manifest.json" ]] || { echo "[STOP] Tube data not prepared. Run: bash $HERE/run.sh prepare --root $GRID25_ROOT" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "[STOP] Joint output already exists; inspect logs before any retry" >&2; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo "[STOP] Git contains uncommitted tracked changes" >&2; exit 2; }
bash "$HERE/run.sh" verify --root "$GRID25_ROOT"
command -v squeue >/dev/null && command -v sbatch >/dev/null || { echo "[STOP] Slurm unavailable" >&2; exit 2; }
MY_QUEUE="$(squeue -h -u "$USER")" || { echo "[STOP] Cannot query own Slurm queue" >&2; exit 2; }
[[ -z "$MY_QUEUE" ]] || { echo "[STOP] Own Slurm queue busy; one job only"; printf "%s\n" "$MY_QUEUE"; exit 2; }
mkdir -p "$BASE/logs"
ARGS=(--partition=GPU --nodelist=3090node3 --nodes=1 --ntasks=1
      --gres=gpu:3090:1 --cpus-per-task=4 --time=04:00:00
      --job-name=g25_joint
      "--output=$BASE/logs/slurm_%j.out" "--error=$BASE/logs/slurm_%j.err"
      "--export=ALL,GRID25_ROOT=$GRID25_ROOT")
echo "[GPU dry-run] 3090node3, 1xRTX3090, 4 CPUs"
sbatch --test-only "${ARGS[@]}" "$HERE/worker.sbatch"
echo "[Submit] One 1000-step V3+V4 H9 supervised training comparison"
JOB="$(sbatch --parsable "${ARGS[@]}" "$HERE/worker.sbatch")" || { echo "[STOP] sbatch failed" >&2; exit 2; }
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "[STOP] Ambiguous submission result: $JOB; DO NOT RETRY blindly" >&2; exit 2; }
echo "[GRID25 V3+V4 JOINT TRAIN SUBMITTED] job=$JOB out=$OUT log=$BASE/logs/slurm_${JOB}.out"
