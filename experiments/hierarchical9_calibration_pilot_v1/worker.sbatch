#!/usr/bin/env bash
# Resource, chdir and ABSOLUTE log paths are passed explicitly by submit.sh.
set -euo pipefail
: "${CAL_RUN_ROOT:?}" "${CAL_ARTIFACT_ROOT:?}" "${CAL_CODE_REPO:?}" "${CAL_STAGE:?}" "${CAL_MODE:?}" "${CAL_CODE_SHA:?}"
cd "$CAL_CODE_REPO"
[[ "$(git rev-parse HEAD)" == "$CAL_CODE_SHA" ]] || { echo 'Code HEAD changed after submission'; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo 'Tracked code modified after submission'; exit 2; }
DIR=experiments/hierarchical9_calibration_pilot_v1
sha256sum -c "$DIR/SHA256SUMS"
echo "[launch] stage=$CAL_STAGE mode=$CAL_MODE run=$CAL_RUN_ROOT host=$(hostname) code=$CAL_CODE_SHA"
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
[[ -f "$CONDA_SH" ]] || { echo 'Set CONDA_SH to the existing environment'; exit 2; }
set +u
source "$CONDA_SH"
conda activate "${CONDA_ENV:-viscdf}"
set -u
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
case "$CAL_MODE" in
  smoke) tp=8;vp=8;ta=2;va=2;steps=2;bc=512;fb=1;sp=2 ;;
  pilot) tp=512;vp=64;ta=4;va=2;steps=2000;bc=4096;fb=10;sp=32 ;;
  *) echo 'CAL_MODE must be smoke or pilot';exit 2 ;;
esac
case "$CAL_STAGE" in
  prepare)
    python "$DIR/test_pilot.py" --require-repo
    python experiments/hierarchical9_boundary_audit_v1/test_audit.py --require-repo
    python "$DIR/prepare.py" --artifact-root "$CAL_ARTIFACT_ROOT" --output "$CAL_RUN_ROOT/cache" \
      --train-points "$tp" --val-points "$vp" --train-anchors "$ta" --val-anchors "$va"
    ;;
  train)
    arm="P${SLURM_ARRAY_TASK_ID:?}"
    [[ "$arm" == P0 || "$arm" == P1 ]] || exit 2
    python -m torch.distributed.run --standalone --nproc_per_node=4 "$DIR/test_pilot.py" --ddp --backend nccl
    python -m torch.distributed.run --standalone --nproc_per_node=4 "$DIR/train_pilot.py" \
      --artifact-root "$CAL_ARTIFACT_ROOT" --cache "$CAL_RUN_ROOT/cache" --output "$CAL_RUN_ROOT/$arm" \
      --arm "$arm" --steps "$steps" --boundary-count "$bc" \
      --microbatch-x "${CAL_MICROBATCH_X:-250}" --decode-x-chunk "${CAL_DECODE_CHUNK:-64}" \
      --amp "${CAL_AMP:-fp16}"
    ;;
  evaluate)
    python "$DIR/evaluate_pair.py" --artifact-root "$CAL_ARTIFACT_ROOT" --run-root "$CAL_RUN_ROOT" \
      --field-batches "$fb" --solve-points "$sp"
    ;;
  *) echo 'Unknown stage';exit 2 ;;
esac
