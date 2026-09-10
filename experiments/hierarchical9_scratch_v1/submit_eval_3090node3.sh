#!/usr/bin/env bash
#SBATCH --job-name=viscdf9_eval
#SBATCH --partition=GPU
#SBATCH --nodelist=3090node3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/per_sensor_training_logs/hierarchical9_eval_%j.out
# Intentionally NO --mem; offline evaluation uses one GPU, not torchrun/DDP.
set -euo pipefail
REPO="${REPO:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner}"
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-viscdf}"
[[ -f "$CONDA_SH" ]] || { echo "Missing conda activation: $CONDA_SH" >&2; exit 2; }
set +u
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u
cd "$REPO"
PKG="$REPO/experiments/hierarchical9_scratch_v1"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
MODE="${EVAL_MODE:-full}"
case "$MODE" in
  smoke) N=1; BX=8; BQ=8; PN=1; PBX=2; PBQ=8 ;;
  full) N=20; BX=128; BQ=64; PN=10; PBX=8; PBQ=64 ;;
  *) echo 'EVAL_MODE must be smoke or full' >&2; exit 2 ;;
esac
# This independent variable deliberately ignores the training script's OUT.
OUTDIR="${EVAL_OUT:-$REPO/outputs/hierarchical9_eval_${MODE}_${SLURM_JOB_ID:?Run via sbatch}}"
NEW="${H9_CHECKPOINT:-$REPO/src/care_visibility_cdf/checkpoints/hierarchical9_scratch_seed0/final.pt}"
SCALAR="${SCALAR_CHECKPOINT:-$REPO/src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/final.pt}"
EIGHT="${EIGHT_CHECKPOINT:-$REPO/src/care_visibility_cdf/checkpoints/per_sensor_e2e_fullbatch_seed0/final.pt}"
DATA="${DATA:-$REPO/src/care_visibility_cdf/data/visibility_yiming_style_grid30_q20000_k500_fovonly.npz}"
URDF="${URDF:-$REPO/src/arm_description/urdf/Arm.urdf}"
for f in "$NEW" "$SCALAR" "$DATA" "$URDF"; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing $f" >&2; exit 2; }
done
EXTRA=()
case "${SKIP_OLD_EIGHT:-0}" in
  1) EXTRA+=(--skip-old-eight) ;;
  0) [[ -f "$EIGHT" ]] || { echo '[ERROR] Missing old8 final.pt; set EIGHT_CHECKPOINT or explicitly SKIP_OLD_EIGHT=1' >&2; exit 2; } ;;
  *) echo 'SKIP_OLD_EIGHT must be 0 or 1' >&2; exit 2 ;;
esac
sha256sum -c "$PKG/EVAL_SHA256SUMS"
echo "[eval-job] mode=$MODE job=$SLURM_JOB_ID node=$(hostname) repo=$(git rev-parse HEAD)"
echo "[eval-output] $OUTDIR"
echo 'Read-only offline evaluation; no training, no ROS, no runtime/safety changes.'
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.device_count() == 1, 'Expected exactly one Slurm-allocated GPU'
print('torch:', torch.__version__, 'GPU:', torch.cuda.get_device_name(0))
PY
python "$PKG/test_evaluate.py"
exec python "$PKG/evaluate.py" \
  --checkpoint "$NEW" --scalar-checkpoint "$SCALAR" --eight-checkpoint "$EIGHT" \
  --data "$DATA" --urdf "$URDF" --output-dir "$OUTDIR" --device cuda \
  --seed "${EVAL_SEED:-123}" --num-batches "$N" --batch-x "$BX" --batch-q "$BQ" \
  --decode-x-chunk "${EVAL_DECODE_X_CHUNK:-64}" \
  --planning-batches "$PN" --planning-batch-x "$PBX" --planning-batch-q "$PBQ" \
  "${EXTRA[@]}"
