#!/usr/bin/env bash
#SBATCH --job-name=g25_seg01
set -euo pipefail
: "${GRID25_ROOT:?Need GRID25_ROOT}"
: "${TRAIN_MODE:?Need TRAIN_MODE}"
[[ "$TRAIN_MODE" == "v3" || "$TRAIN_MODE" == "joint" ]] || { echo "[STOP] Unknown mode" >&2; exit 2; }
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "[SEG01 MODE] $TRAIN_MODE [CUDA_VISIBLE_DEVICES] ${CUDA_VISIBLE_DEVICES:-unset}"
"$VIS_PYTHON" -u experiments/hierarchical9_grid25_multisegment_train_v1/train.py "$TRAIN_MODE" --root "$GRID25_ROOT"
