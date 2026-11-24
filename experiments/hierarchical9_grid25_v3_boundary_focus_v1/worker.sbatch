#!/usr/bin/env bash
#SBATCH --job-name=g25h9_v3boundary
set -euo pipefail
: "${GRID25_ROOT:?GRID25_ROOT required}"
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "[BOUNDARY H9 CUDA] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"$VIS_PYTHON" -u experiments/hierarchical9_grid25_v3_boundary_focus_v1/train.py train \
  --root "$GRID25_ROOT" --steps 20000 --batch 1024 --val-every 500 \
  --lr 0.001 --seed 0 --union-weight 0.2
