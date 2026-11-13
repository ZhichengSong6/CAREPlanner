#!/usr/bin/env bash
#SBATCH --job-name=g25_h9_8seg
set -euo pipefail
: "${GRID25_ROOT:?Need GRID25_ROOT}"
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "[H9 GPU] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"$VIS_PYTHON" -u experiments/hierarchical9_grid25_v3_8segment_train_v1/train.py train \
  --root "$GRID25_ROOT" --steps 5000 --batch 512 --val-every 250
