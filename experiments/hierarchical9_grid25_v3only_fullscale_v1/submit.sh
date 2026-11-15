#!/usr/bin/env bash
# Single user action. The running job schedules exactly one successor each time.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
export GRID25_ROOT="${GRID25_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1}"
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
[[ -x "$VIS_PYTHON" ]] || { echo "[STOP] Python unavailable: $VIS_PYTHON" >&2; exit 2; }
command -v sbatch >/dev/null && command -v squeue >/dev/null || { echo "[STOP] Slurm tools unavailable" >&2; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo "[STOP] Tracked git source changes; freeze checkout before production" >&2; exit 2; }
BASE="$GRID25_ROOT/derived/grid25_v3only_fullscale_v1"
[[ ! -f "$BASE/batch_complete.json" ]] || { echo "[STOP] All 116 segments already complete" >&2; exit 2; }
QUEUE="$(squeue -h -u "$USER")" || { echo "[STOP] Could not read own queue" >&2; exit 2; }
[[ -z "$QUEUE" ]] || { echo "[STOP] Your Slurm queue is not empty. Do not overlap jobs"; printf '%s\n' "$QUEUE"; exit 2; }
bash "$HERE/run.sh" preflight --root "$GRID25_ROOT"
mkdir -p "$BASE/logs"
ARGS=(--partition=GPU --nodes=1 --ntasks=1 --cpus-per-task=4
      --time=1-00:00:00 --requeue --open-mode=append
      --job-name=g25v3full
      "--output=$BASE/logs/slurm_%j.out"
      "--error=$BASE/logs/slurm_%j.err"
      "--export=ALL,GRID25_ROOT=$GRID25_ROOT")
echo "[Slurm test-only] one 24h CPU-only allocation, 4 CPU, zero GPUs, automatic sequential continuation"
sbatch --test-only "${ARGS[@]}" "$HERE/worker.sbatch"
echo "[Submit] Starting Grid25 V3-only fullscale segments 8..123"
JOB="$(sbatch --parsable "${ARGS[@]}" "$HERE/worker.sbatch")" || { echo "[STOP] Slurm rejected initial job" >&2; exit 2; }
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "[UNKNOWN] Ambiguous ID; inspect squeue before retry" >&2; exit 2; }
printf 'initial_job=%s\n' "$JOB" > "$BASE/initial_job.txt"
echo "[FULLSCALE 116-SEGMENT CHAIN STARTED] job=$JOB"
echo "[STATUS] bash $HERE/run.sh status --root $GRID25_ROOT"
echo "[LOG] $BASE/logs/slurm_${JOB}.out"
