#!/usr/bin/env bash
#SBATCH --job-name=g25_v3only
set -euo pipefail
: "${GRID25_ROOT:?GRID25_ROOT required}"
: "${SEGMENT_ID:?SEGMENT_ID required}"
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
bash experiments/hierarchical9_grid25_v3only_production_v1/run.sh work --root "$GRID25_ROOT" --segment "$SEGMENT_ID"
