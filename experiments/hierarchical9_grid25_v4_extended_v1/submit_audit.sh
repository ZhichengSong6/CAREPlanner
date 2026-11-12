#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GRID25_ROOT="${GRID25_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1}"
EXT_OUT="${EXT_OUT:-$GRID25_ROOT/derived/grid25_v4_extended_v1/pilot_train32_val32}"
FD="$EXT_OUT/extended_fd_audit_v1"
[[ -f "$FD/audit_manifest.json" && -f "$FD/fd_plan.json" ]] || { echo "[STOP] run fd-plan first"; exit 2; }
command -v squeue >/dev/null && command -v sbatch >/dev/null || { echo "[STOP] Slurm unavailable"; exit 2; }
QUEUE="$(squeue -h -u "$USER")" || { echo "[STOP] Cannot inspect user queue"; exit 2; }
[[ -z "$QUEUE" ]] || { echo "[STOP] Own Slurm queue not empty"; printf "%s\n" "$QUEUE"; exit 2; }
mkdir -p "$FD/logs"
ARGS=(--partition=GPU --nodes=1 --ntasks=1 --cpus-per-task=4 --time=1-00:00:00
  --job-name=g25_fd_audit "--output=$FD/logs/slurm_%j.out"
  "--error=$FD/logs/slurm_%j.err" "--export=ALL,EXT_OUT=$EXT_OUT")
echo "[Slurm test-only] CPU only, one node, four CPUs; no explicit mem or GPU"
sbatch --test-only "${ARGS[@]}" "$HERE/audit_worker.sbatch"
echo "[Submit] One job only; no retries/dependencies"
JOB="$(sbatch --parsable "${ARGS[@]}" "$HERE/audit_worker.sbatch")" || { echo "[STOP] sbatch rejected"; exit 2; }
[[ "$JOB" =~ ^[0-9]+([;].*)?$ ]] || { echo "[UNKNOWN] Unexpected sbatch reply: $JOB; do not retry"; exit 2; }
echo "[FD AUDIT SUBMITTED] job=$JOB out=$FD"
