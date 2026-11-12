#!/usr/bin/env bash
#SBATCH --job-name=g25_v4pilot
set -euo pipefail
: "${PILOT_OUT:?Pilot output required}"
: "${PILOT_WORKERS:=4}"
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
bash experiments/hierarchical9_grid25_v4_pilot_v1/run.sh work --out "$PILOT_OUT" --workers "$PILOT_WORKERS"
