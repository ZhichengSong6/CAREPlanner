#!/usr/bin/env bash
set -euo pipefail
SEGMENT_ID="${1:-}"
[[ "$SEGMENT_ID" =~ ^[2-7]$ ]] || { echo "Usage: bash submit.sh 2|3|4|5|6|7" >&2; exit 2; }
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
export GRID25_ROOT="${GRID25_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1}"
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
printf -v SEG "segment_%03d" "$SEGMENT_ID"
OUT="$GRID25_ROOT/derived/grid25_v3only_production_v1/$SEG"
[[ -f "$OUT/manifest.json" && -f "$OUT/x_plan.npz" ]] || { echo "[STOP] Prepare segment first" >&2; exit 2; }
[[ ! -f "$OUT/segment_complete.json" ]] || { echo "[STOP] Already verified COMPLETE" >&2; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo "[STOP] Git has tracked modifications" >&2; exit 2; }
bash "$HERE/run.sh" status --root "$GRID25_ROOT" --segment "$SEGMENT_ID"
command -v squeue >/dev/null && command -v sbatch >/dev/null || { echo "[STOP] Slurm unavailable" >&2; exit 2; }
MY_QUEUE="$(squeue -h -u "$USER")" || { echo "[STOP] Cannot query Slurm queue" >&2; exit 2; }
[[ -z "$MY_QUEUE" ]] || { echo "[STOP] Own queue not empty; one job at a time"; printf "%s\n" "$MY_QUEUE"; exit 2; }
mkdir -p "$OUT/logs"
ARGS=(--partition=GPU --nodes=1 --ntasks=1 --cpus-per-task=4 --time=1-00:00:00
     "--job-name=g25_v3s$SEGMENT_ID"
     "--output=$OUT/logs/slurm_%j.out" "--error=$OUT/logs/slurm_%j.err"
     "--export=ALL,GRID25_ROOT=$GRID25_ROOT,SEGMENT_ID=$SEGMENT_ID")
echo "[Slurm test-only] V3-only, one node, 4 CPUs, NO GPU/memory requests"
sbatch --test-only "${ARGS[@]}" "$HERE/worker.sbatch"
echo "[Submit] A single manually selected V3-only segment; NO dependency/array/automatic continuation"
JOB="$(sbatch --parsable "${ARGS[@]}" "$HERE/worker.sbatch")" || { echo "[STOP] Slurm submit rejected" >&2; exit 2; }
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "[UNKNOWN JOB] $JOB -- inspect squeue before retry" >&2; exit 2; }
echo "[V3-ONLY SEGMENT SUBMITTED] segment=$SEGMENT_ID job=$JOB output=$OUT"
