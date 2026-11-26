#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
export GRID25_ROOT="${GRID25_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1}"
BASE="$GRID25_ROOT/derived/grid25_v3_boundary_pairs_v1"
[[ -f "$BASE/plan.json" && -f "$BASE/tasks.npz" ]] || { echo "[STOP] Run prepare first" >&2; exit 2; }
[[ ! -f "$BASE/augmentation_manifest.json" ]] || { echo "[STOP] Boundary labels already frozen" >&2; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo "[STOP] Local tracked changes" >&2; exit 2; }
"$VIS_PYTHON" -m py_compile "$HERE/labels.py"
QUEUE="$(squeue -h -u "$USER")"
[[ -z "$QUEUE" ]] || { echo "[STOP] Another own Slurm job exists"; printf "%s\n" "$QUEUE"; exit 2; }
mkdir -p "$BASE/logs"
ARGS=(--partition=GPU --nodelist=3090node3 --nodes=1 --ntasks=1
      --cpus-per-task=4 --time=04:00:00
      --job-name=g25v3_bndpairs
      "--output=$BASE/logs/labels_%j.out"
      "--error=$BASE/logs/labels_%j.err"
      "--export=ALL,GRID25_ROOT=$GRID25_ROOT")
# ZERO GPUs. This is the original CPU V3 re-projection solver.
echo "[THIRD ROUND V3 BOUNDARY LABELS] 4 CPUs, 0 GPUs, resumable"
sbatch --test-only "${ARGS[@]}" "$HERE/labels_worker.sbatch"
JOB="$(sbatch --parsable "${ARGS[@]}" "$HERE/labels_worker.sbatch")"
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "[STOP] Ambiguous Job ID; do not resubmit blindly"; exit 2; }
echo "[BOUNDARY PAIRS LABEL JOB SUBMITTED] job=$JOB log=$BASE/logs/labels_${JOB}.out"
