#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GRID25_ROOT="${GRID25_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1}"
SEGMENT_ID="${SEGMENT_ID:-0}"
[[ "$SEGMENT_ID" =~ ^(0|[1-9][0-9]{0,2})$ ]] || { echo "[STOP] SEGMENT_ID must be 0..123" >&2; exit 2; }
(( 10#$SEGMENT_ID < 124 )) || { echo "[STOP] Segment out of scope" >&2; exit 2; }
printf -v SEGMENT_DIR "segment_%03d" "$SEGMENT_ID"
SEGMENT_OUT="$GRID25_ROOT/derived/grid25_v4_production_v1/$SEGMENT_DIR"
[[ -f "$SEGMENT_OUT/manifest.json" && -f "$SEGMENT_OUT/x_plan.npz" ]] || { echo "[STOP] Prepare this segment first: $SEGMENT_OUT" >&2; exit 2; }
if [[ -f "$SEGMENT_OUT/segment_complete.json" ]]; then echo "[STOP] Segment already completed; do not resubmit" >&2; exit 2; fi
command -v squeue >/dev/null && command -v sbatch >/dev/null || { echo "[STOP] Slurm unavailable" >&2; exit 2; }
MY_QUEUE="$(squeue -h -u "$USER")" || { echo "[STOP] Cannot query Slurm queue" >&2; exit 2; }
[[ -z "$MY_QUEUE" ]] || { echo "[STOP] Own Slurm queue not empty; one job only"; printf "%s\n" "$MY_QUEUE"; exit 2; }
PY="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
"$PY" "$HERE/runner.py" status --out "$SEGMENT_OUT"
mkdir -p "$SEGMENT_OUT/logs"
ARGS=(--partition=GPU --nodes=1 --ntasks=1 --cpus-per-task=4 --time=1-00:00:00
  "--job-name=g25p${SEGMENT_ID}"
  "--output=$SEGMENT_OUT/logs/slurm_%j.out" "--error=$SEGMENT_OUT/logs/slurm_%j.err"
  "--export=ALL,SEGMENT_OUT=$SEGMENT_OUT")
echo "[Slurm test-only] One node, 4 CPUs, zero GPUs; Slurm default memory"
sbatch --test-only "${ARGS[@]}" "$HERE/worker.sbatch"
echo "[Submit] One manually chosen segment; no arrays/retries/dependencies/automatic next"
JOB="$(sbatch --parsable "${ARGS[@]}" "$HERE/worker.sbatch")" || { echo "[STOP] sbatch rejected" >&2; exit 2; }
[[ "$JOB" =~ ^[0-9]+([;].*)?$ ]] || { echo "[UNKNOWN] Ambiguous Slurm job response: $JOB; DO NOT RETRY" >&2; exit 2; }
echo "[PRODUCTION SEGMENT SUBMITTED] job=$JOB segment=$SEGMENT_ID path=$SEGMENT_OUT"
