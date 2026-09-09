#!/usr/bin/env bash
#SBATCH --job-name=viscdf9_hier
#SBATCH --partition=GPU
#SBATCH --nodelist=3090node3
#SBATCH --gres=gpu:3090:4
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/per_sensor_training_logs/hierarchical9_e2e_fullbatch_seed0_%j.out

set -euo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate viscdf

REPO=/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner
cd "$REPO"
mkdir -p outputs/per_sensor_training_logs

echo "===== JOB INFO ====="
echo "job_id: ${SLURM_JOB_ID:-none}"
echo "node:   $(hostname)"
echo "head:   $(git rev-parse HEAD)"
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

OUT="${OUT:-$REPO/src/care_visibility_cdf/checkpoints/hierarchical9_e2e_fullbatch_seed0}"

OUT="$OUT" \
STEPS="${STEPS:-50000}" \
GLOBAL_BATCH_X="${GLOBAL_BATCH_X:-4000}" \
BATCH_Q="${BATCH_Q:-100}" \
MICROBATCH_X="${MICROBATCH_X:-250}" \
VAL_GLOBAL_BATCH_X="${VAL_GLOBAL_BATCH_X:-512}" \
VAL_BATCH_Q="${VAL_BATCH_Q:-100}" \
VAL_MICROBATCH_X="${VAL_MICROBATCH_X:-128}" \
DECODE_X_CHUNK="${DECODE_X_CHUNK:-64}" \
SHARED_LAYERS="${SHARED_LAYERS:-1024,512,256}" \
BRANCH_LAYERS="${BRANCH_LAYERS:-128,128}" \
WEIGHT_SDF="${WEIGHT_SDF:-5.0}" \
WEIGHT_GRAD="${WEIGHT_GRAD:-0.1}" \
WEIGHT_EIKONAL="${WEIGHT_EIKONAL:-0.01}" \
WEIGHT_TENSION="${WEIGHT_TENSION:-0.01}" \
WEIGHT_SENSOR_OBJECTIVE="${WEIGHT_SENSOR_OBJECTIVE:-1.0}" \
WEIGHT_UNION_OBJECTIVE="${WEIGHT_UNION_OBJECTIVE:-1.0}" \
WEIGHT_CONSISTENCY="${WEIGHT_CONSISTENCY:-0.1}" \
PROFILE="${PROFILE:-0}" \
NPROC_PER_NODE=4 \
RESUME="${RESUME:-}" \
bash scripts/run_train_hierarchical_visibility_cdf_ddp.sh
