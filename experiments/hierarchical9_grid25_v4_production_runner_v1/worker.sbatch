#!/usr/bin/env bash
#SBATCH --job-name=g25_v4prod
set -euo pipefail
: "${SEGMENT_OUT:?SEGMENT_OUT is required}"
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
bash experiments/hierarchical9_grid25_v4_production_runner_v1/run.sh work --out "$SEGMENT_OUT" --workers 4
