#!/usr/bin/env bash
#SBATCH --job-name=viscdf8_full
#SBATCH --partition=GPU
#SBATCH --nodelist=3090node1
#SBATCH --gres=gpu:3090:4
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/per_sensor_training_logs/per_sensor_e2e_fullbatch_seed0_%j.out

set -euo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate viscdf

cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner
mkdir -p outputs/per_sensor_training_logs

echo "===== JOB INFO ====="
hostname
nvidia-smi
which python
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("visible_gpus:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
echo "===================="

OUT=/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/src/care_visibility_cdf/checkpoints/per_sensor_e2e_fullbatch_seed0 STEPS=${STEPS:-50000} GLOBAL_BATCH_X=4000 BATCH_Q=100 MICROBATCH_X=${MICROBATCH_X:-250} VAL_GLOBAL_BATCH_X=512 VAL_BATCH_Q=100 VAL_MICROBATCH_X=128 DECODE_X_CHUNK=64 PROFILE=${PROFILE:-0} NPROC_PER_NODE=4 bash scripts/run_train_per_sensor_visibility_cdf_ddp.sh
