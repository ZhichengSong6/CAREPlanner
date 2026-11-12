#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GRID25_ROOT="${GRID25_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1}"
PILOT_OUT="${PILOT_OUT:-$GRID25_ROOT/derived/grid25_v4_pilot_v1/pilot_train2_val2}"
PILOT_WORKERS="${PILOT_WORKERS:-4}"
PILOT_CPUS="${PILOT_CPUS:-4}"
[[ "$PILOT_WORKERS" =~ ^[1-8]$ ]] || { echo "[STOP] PILOT_WORKERS=1..8 required" >&2; exit 2; }
[[ "$PILOT_CPUS" =~ ^[1-8]$ ]] || { echo "[STOP] PILOT_CPUS=1..8 required" >&2; exit 2; }
(( PILOT_CPUS >= PILOT_WORKERS )) || { echo "[STOP] PILOT_CPUS must be >= PILOT_WORKERS" >&2; exit 2; }
[[ -f "$PILOT_OUT/manifest.json" && -f "$PILOT_OUT/x_plan.npz" ]] || { echo "[STOP] Pilot prepare first" >&2; exit 2; }
command -v squeue >/dev/null && command -v sbatch >/dev/null || { echo "[STOP] Slurm commands unavailable" >&2; exit 2; }
USER_QUEUE="$(squeue -h -u "$USER")" || { echo "[STOP] Cannot read own Slurm queue" >&2; exit 2; }
[[ -z "$USER_QUEUE" ]] || { echo "[STOP] Existing own Slurm jobs; single job association limit" >&2; printf "%s\n" "$USER_QUEUE"; exit 2; }
mkdir -p "$PILOT_OUT/logs"
ARGS=(--partition=GPU --nodes=1 --ntasks=1 "--cpus-per-task=$PILOT_CPUS"
  --time=1-00:00:00 --job-name=g25_v4pilot
  "--output=$PILOT_OUT/logs/slurm_%j.out" "--error=$PILOT_OUT/logs/slurm_%j.err"
  "--export=ALL,PILOT_OUT=$PILOT_OUT,PILOT_WORKERS=$PILOT_WORKERS")
echo "[Slurm test-only] No GPUs requested; one node; CPUs=$PILOT_CPUS workers=$PILOT_WORKERS; memory=Slurm partition default"
sbatch --test-only "${ARGS[@]}" "$HERE/worker.sbatch"
echo "[Submit] One job only, no retry/automatic continuation"
JOB="$(sbatch --parsable "${ARGS[@]}" "$HERE/worker.sbatch")" || { echo "[STOP] sbatch rejected" >&2; exit 2; }
[[ "$JOB" =~ ^[0-9]+([;].*)?$ ]] || { echo "[UNKNOWN] Ambiguous sbatch reply: $JOB; DO NOT RETRY" >&2; exit 2; }
echo "[PILOT SUBMITTED] job=$JOB out=$PILOT_OUT"
