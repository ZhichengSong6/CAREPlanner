#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
export GRID25_ROOT="${GRID25_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1}"
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
BASE="$GRID25_ROOT/derived/grid25_v3_boundary_focus_v1"
OUT="$BASE/run_r1_v3_boundary_focus"
SOURCE="$GRID25_ROOT/derived/grid25_v3_full_training_v1/dataset_segments_000_123"
[[ -x "$VIS_PYTHON" ]] || { echo "[STOP] viscdf Python missing: $VIS_PYTHON" >&2; exit 2; }
[[ -f "$SOURCE/manifest.json" && -f "$SOURCE/labels.npz" ]] || { echo "[STOP] Frozen 124-segment V3 training dataset is missing" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "[STOP] Boundary experiment output exists; refuse overwrite" >&2; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo "[STOP] Git has uncommitted tracked changes" >&2; exit 2; }
QUEUE="$(squeue -h -u "$USER")" || { echo "[STOP] Cannot check Slurm queue" >&2; exit 2; }
[[ -z "$QUEUE" ]] || { echo "[STOP] Existing Slurm job; do not submit a second job"; printf "%s\n" "$QUEUE"; exit 2; }
# No extra pilot or repeated 124-segment source verification:
# training already checks frozen snapshot SHA, identity, counts and split.
"$VIS_PYTHON" -m py_compile "$HERE/train.py"
mkdir -p "$BASE/logs"
ARGS=(--partition=GPU --nodelist=3090node3 --nodes=1 --ntasks=1
      --gres=gpu:3090:1 --cpus-per-task=4 --time=04:00:00
      --job-name=g25h9_v3boundary
      "--output=$BASE/logs/slurm_%j.out"
      "--error=$BASE/logs/slurm_%j.err"
      "--export=ALL,GRID25_ROOT=$GRID25_ROOT")
echo "[Slurm test-only] One controlled boundary-weighted H9 20000-step GPU job"
sbatch --test-only "${ARGS[@]}" "$HERE/worker.sbatch"
JOB="$(sbatch --parsable "${ARGS[@]}" "$HERE/worker.sbatch")" || { echo "[STOP] sbatch rejected" >&2; exit 2; }
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "[UNKNOWN] Ambiguous Slurm job ID, do not retry blindly" >&2; exit 2; }
echo "[BOUNDARY FOCUS H9 TRAIN SUBMITTED] job=$JOB log=$BASE/logs/slurm_${JOB}.out"
