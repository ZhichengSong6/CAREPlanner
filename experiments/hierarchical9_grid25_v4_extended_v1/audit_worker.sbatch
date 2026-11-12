#!/usr/bin/env bash
#SBATCH --job-name=g25_fd_audit
set -euo pipefail
: "${EXT_OUT:?Expected extended pilot path}"
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$VIS_PYTHON" -u experiments/hierarchical9_grid25_v4_extended_v1/gradient_audit.py work --out "$EXT_OUT" --workers 4
