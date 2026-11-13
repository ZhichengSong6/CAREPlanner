#!/usr/bin/env bash
#SBATCH --job-name=g25_joint
set -euo pipefail
: "${GRID25_ROOT:?GRID25_ROOT not set}"
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "[CUDA_VISIBLE_DEVICES] ${CUDA_VISIBLE_DEVICES:-unset}"
"$VIS_PYTHON" -u experiments/hierarchical9_grid25_v3_v4_joint_train_v1/train.py \
  --root "$GRID25_ROOT" --steps 1000 --batch 512 --eval-every 100
