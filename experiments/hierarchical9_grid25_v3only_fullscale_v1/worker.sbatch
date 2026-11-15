#!/usr/bin/env bash
#SBATCH --job-name=g25v3full
set -euo pipefail
: "${GRID25_ROOT:?GRID25_ROOT required}"
: "${SLURM_JOB_ID:?This entrypoint must run in Slurm}"
# sbatch copies this script to /var/lib/slurm/slurmd/job*/slurm_script.
# Resolve Python modules from the source checkout, NEVER from $BASH_SOURCE.
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
HERE="$PWD/experiments/hierarchical9_grid25_v3only_fullscale_v1"
[[ -f "$HERE/batch.py" && -f "$HERE/chain.sh" ]] || {
  echo "[STOP] Fullscale production Python/continuation source missing: $HERE" >&2
  exit 2
}
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "[FULLSCALE ALLOCATION] job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} host=$(hostname)"
if "$VIS_PYTHON" -u "$HERE/batch.py" work --root "$GRID25_ROOT"; then
  echo "[FULLSCALE ALL LABELS COMPLETE] job=$SLURM_JOB_ID"
  exit 0
else
  RC=$?
  if [[ "$RC" -eq 42 ]]; then
    echo "[FULLSCALE CONTINUE] requesting one sequential next allocation"
    bash "$HERE/chain.sh" "$SLURM_JOB_ID"
    exit 0
  fi
  echo "[FULLSCALE STOP] batch returned $RC: inspect data and logs; no next job submitted" >&2
  exit "$RC"
fi
