#!/usr/bin/env bash
#SBATCH --job-name=g25_h9v3
set -euo pipefail
: "${TRAIN_DATA:?Need TRAIN_DATA}"
: "${TRAIN_OUT:?Need TRAIN_OUT}"
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
echo "[CUDA_VISIBLE_DEVICES] ${CUDA_VISIBLE_DEVICES:-unset}"
"$VIS_PYTHON" -u experiments/hierarchical9_grid25_v3_first_train_v1/train.py \
  --dataset "$TRAIN_DATA" --out "$TRAIN_OUT" --steps 1000 --batch 512 --eval-every 100
