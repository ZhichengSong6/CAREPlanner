#!/usr/bin/env bash
#SBATCH --job-name=viscdf9_scratch
#SBATCH --partition=GPU
#SBATCH --nodelist=3090node3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3090:4
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/per_sensor_training_logs/hierarchical9_scratch_%j.out

# Deliberately NO --mem: keep the cluster-specific constraint from the handoff.
set -euo pipefail
REPO="${REPO:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner}"
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-viscdf}"
if [[ ! -f "$CONDA_SH" ]]; then
  echo "[ERROR] Missing $CONDA_SH; set CONDA_SH to your existing conda.sh" >&2
  exit 2
fi
# Some conda versions access unset shell variables while activating.
set +u
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u
cd "$REPO"
PKG="$REPO/experiments/hierarchical9_scratch_v1"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
# Do not change runtime PER_SENSOR_HYBRID_ENABLED, URDF, or any safety setting.
if [[ -n "${RESUME:-}" || -n "${PRETRAINED:-}" ]]; then
  echo "[ERROR] Unset inherited RESUME/PRETRAINED. This experiment starts from scratch." >&2
  echo "For recovery ONLY, use explicit RESUME_TRAINING=<this experiment's latest.pt>." >&2
  exit 2
fi
MODE="${MODE:-train}"
AMP="${AMP:-fp16}"
case "$MODE" in
  smoke)
    STEPS=2
    LOG_EVERY=1
    VAL_EVERY=1
    SAVE_EVERY=1
    OUT="${OUT:-$REPO/src/care_visibility_cdf/checkpoints/hierarchical9_scratch_smoke_${SLURM_JOB_ID}}"
    if [[ -n "${RESUME_TRAINING:-}" ]]; then
      echo "[ERROR] A smoke test must not resume a training checkpoint" >&2
      exit 2
    fi
    ;;
  train)
    STEPS="${STEPS:-50000}"
    LOG_EVERY="${LOG_EVERY:-100}"
    VAL_EVERY="${VAL_EVERY:-1000}"
    SAVE_EVERY="${SAVE_EVERY:-5000}"
    OUT="${OUT:-$REPO/src/care_visibility_cdf/checkpoints/hierarchical9_scratch_seed0}"
    ;;
  *) echo "[ERROR] MODE must be smoke or train" >&2; exit 2 ;;
esac
DATA="${DATA:-$REPO/src/care_visibility_cdf/data/visibility_yiming_style_grid30_q20000_k500_fovonly.npz}"
URDF="${URDF:-$REPO/src/arm_description/urdf/Arm.urdf}"
for path in "$DATA" "$URDF" "$PKG/model.py" "$PKG/objective.py" "$PKG/train.py"; do
  [[ -f "$path" ]] || { echo "[ERROR] Missing $path" >&2; exit 2; }
done
if [[ -z "${RESUME_TRAINING:-}" && -d "$OUT" ]] && [[ -n "$(ls -A "$OUT")" ]]; then
  echo "[ERROR] Refusing to overwrite nonempty OUT=$OUT; choose a fresh OUT" >&2
  exit 2
fi
python -m py_compile "$PKG/model.py" "$PKG/objective.py" "$PKG/train.py" "$PKG/test_synthetic.py"
echo "===== JOB INFO ====="
echo "mode=$MODE job=${SLURM_JOB_ID:-none} node=$(hostname)"
echo "repo=$REPO commit=$(git rev-parse HEAD)"
echo "out=$OUT amp=$AMP steps=$STEPS"
echo "scratch=yes (unless explicit RESUME_TRAINING is set); no pretrained scalar"
nvidia-smi
python - <<'PY'
import sys, torch
assert sys.version_info >= (3, 10), 'Use Python >=3.10 in the existing viscdf environment'
assert torch.cuda.is_available(), 'CUDA is unavailable'
assert torch.cuda.device_count() == 4, f'Expected 4 visible GPUs; got {torch.cuda.device_count()}'
print('python:', sys.version)
print('torch:', torch.__version__, 'CUDA:', torch.version.cuda)
for i in range(4):
    print(i, torch.cuda.get_device_name(i))
PY
# Synthetic tests fail before loading the large dataset if autograd/DDP is broken.
python "$PKG/test_synthetic.py" --device cuda --amp "$AMP"
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$PKG/test_synthetic.py" --ddp --device cuda
EXTRA=()
if [[ -n "${RESUME_TRAINING:-}" ]]; then
  EXTRA+=(--resume-training "$RESUME_TRAINING")
fi
exec python -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$PKG/train.py" \
  --repo "$REPO" --data "$DATA" --urdf "$URDF" --out-dir "$OUT" \
  --steps "$STEPS" --seed "${SEED:-0}" --lr "${LR:-1e-3}" --amp "$AMP" \
  --global-batch-x "${GLOBAL_BATCH_X:-4000}" --batch-q "${BATCH_Q:-100}" \
  --microbatch-x "${MICROBATCH_X:-250}" \
  --val-global-batch-x "${VAL_GLOBAL_BATCH_X:-512}" --val-batch-q "${VAL_BATCH_Q:-100}" \
  --val-microbatch-x "${VAL_MICROBATCH_X:-128}" --decode-x-chunk "${DECODE_X_CHUNK:-64}" \
  --weight-sdf "${WEIGHT_SDF:-5.0}" --weight-grad "${WEIGHT_GRAD:-0.1}" \
  --weight-eikonal "${WEIGHT_EIKONAL:-0.01}" --weight-tension "${WEIGHT_TENSION:-0.01}" \
  --weight-union-objective "${WEIGHT_UNION_OBJECTIVE:-1.0}" \
  --weight-sensor-objective "${WEIGHT_SENSOR_OBJECTIVE:-1.0}" \
  --weight-consistency "${WEIGHT_CONSISTENCY:-0.1}" \
  --log-every "$LOG_EVERY" --val-every "$VAL_EVERY" --save-every "$SAVE_EVERY" \
  "${EXTRA[@]}"
