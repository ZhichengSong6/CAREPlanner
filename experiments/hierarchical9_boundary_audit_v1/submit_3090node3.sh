#!/usr/bin/env bash
#SBATCH --job-name=h9_boundary_audit
#SBATCH --partition=GPU
#SBATCH --nodelist=3090node3
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=outputs/mainline_b_logs/h9_boundary_%j.out
#SBATCH --error=outputs/mainline_b_logs/h9_boundary_%j.out

set -euo pipefail
# Do NOT add --mem: preserve the cluster workaround. No DDP/training is run.
CODE_REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner}"
AUDIT_MODE="${AUDIT_MODE:-smoke}"
AUDIT_STAGE="${AUDIT_STAGE:-all}"
CONDA_SH="${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-viscdf}"
cd "${CODE_REPO}"
DIR="experiments/hierarchical9_boundary_audit_v1"
test -f "${DIR}/audit.py" || { echo "Submit from the mainline-B worktree root"; exit 2; }
test -f "${CONDA_SH}" || { echo "Set CONDA_SH to the existing conda activation script"; exit 2; }
# Some conda activation scripts access unset shell variables.
set +u
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
case "${AUDIT_MODE}" in
  smoke) points=2; anchors=1; random_starts=1 ;;
  full) points=64; anchors=2; random_starts=2 ;;
  preflight) points=2; anchors=1; random_starts=1 ;;
  *) echo "AUDIT_MODE must be smoke/full/preflight"; exit 2 ;;
esac
case "${AUDIT_STAGE}" in all|boundary) ;; *) echo "AUDIT_STAGE must be all/boundary"; exit 2 ;; esac
OUT="${AUDIT_OUT:-${ARTIFACT_ROOT}/outputs/mainline_b/h9_boundary_${AUDIT_MODE}_${SLURM_JOB_ID:-manual}}"
# Logs must exist BEFORE sbatch. This mkdir is not a substitute for that.
echo "[audit-config] code=${CODE_REPO} artifacts=${ARTIFACT_ROOT} mode=${AUDIT_MODE} stage=${AUDIT_STAGE} out=${OUT}"
git rev-parse HEAD
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
python -m py_compile "${DIR}/core.py" "${DIR}/oracle.py" "${DIR}/runtime_probe.py" "${DIR}/audit.py" "${DIR}/test_audit.py"
# This must include the real repository FK/runtime test: missing deps fail, not skip.
python "${DIR}/test_audit.py" --require-repo
extra=()
if [[ "${AUDIT_MODE}" == preflight ]]; then extra+=(--preflight-only); fi
if [[ "${ANALYTIC_CONTROL:-1}" == 1 ]]; then extra+=(--analytic-control); fi
python "${DIR}/audit.py" \
  --artifact-root "${ARTIFACT_ROOT}" \
  --data "${DATA:-src/care_visibility_cdf/data/visibility_yiming_style_grid30_q20000_k500_fovonly.npz}" \
  --h9 "${H9_CHECKPOINT:-src/care_visibility_cdf/checkpoints/hierarchical9_scratch_seed0/final.pt}" \
  --old8 "${EIGHT_CHECKPOINT:-src/care_visibility_cdf/checkpoints/per_sensor_e2e_fullbatch_seed0/final.pt}" \
  --output-dir "${OUT}" --device cuda --stage "${AUDIT_STAGE}" \
  --points-per-sensor "${points}" --anchors-per-point "${anchors}" \
  --random-starts-per-point "${random_starts}" --seed "${AUDIT_SEED:-271828}" "${extra[@]}"
