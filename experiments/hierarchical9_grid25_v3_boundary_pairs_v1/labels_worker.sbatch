#!/usr/bin/env bash
#SBATCH --job-name=g25v3_bndpairs
set -euo pipefail
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
: "${GRID25_ROOT:?Set GRID25_ROOT}"
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$VIS_PYTHON" -u experiments/hierarchical9_grid25_v3_boundary_pairs_v1/labels.py work \
  --root "$GRID25_ROOT" --workers 4
