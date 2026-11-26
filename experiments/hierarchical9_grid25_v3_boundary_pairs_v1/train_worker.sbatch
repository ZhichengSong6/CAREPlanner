#!/usr/bin/env bash
#SBATCH --job-name=g25h9_v3pairs
set -euo pipefail
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
: "${GRID25_ROOT:?Set GRID25_ROOT}"
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$VIS_PYTHON" -u experiments/hierarchical9_grid25_v3_boundary_pairs_v1/train.py train \
  --root "$GRID25_ROOT" --steps 20000 --batch 1024 --val-every 500 \
  --lr 0.001 --seed 0 --union-weight 0.2
