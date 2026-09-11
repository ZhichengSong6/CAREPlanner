#!/usr/bin/env bash
set -euo pipefail
: "${P3_REFERENCE_ROOT:?}" "${P3_CODE_REPO:?}" "${P3_CODE_SHA:?}" "${P3_STAGE:?}"
cd "$P3_CODE_REPO"
[[ "$(git rev-parse HEAD)" == "$P3_CODE_SHA" ]] || { echo 'Code HEAD changed after submission';exit 2; }
git diff --quiet && git diff --cached --quiet || { echo 'Tracked sources changed after submission';exit 2; }
DIR=experiments/hierarchical9_p3_neighborhood_sign_v1
sha256sum -c "$DIR/SHA256SUMS"
echo "[launch] P3 stage=$P3_STAGE root=$P3_REFERENCE_ROOT host=$(hostname) code=$P3_CODE_SHA"
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
[[ -f "$CONDA_SH" ]] || { echo 'Set CONDA_SH to your existing environment';exit 2; }
set +u
source "$CONDA_SH"
conda activate "${CONDA_ENV:-viscdf}"
set -u
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
case "$P3_STAGE" in
 smoke|pilot)
   python "$DIR/test_p3.py"
   python "$DIR/test_workflow.py"
   python -m torch.distributed.run --standalone --nproc_per_node=4 "$DIR/test_p3.py" --ddp --backend nccl
   python -m torch.distributed.run --standalone --nproc_per_node=4 "$DIR/train_p3.py" \
     --reference-root "$P3_REFERENCE_ROOT" --mode "$P3_STAGE"
   if [[ "$P3_STAGE" == smoke ]]; then
     python "$DIR/evaluate_p3.py" --reference-root "$P3_REFERENCE_ROOT" --mode smoke
     echo '[done] p3_smoke_workflow_complete'
   fi
   ;;
 eval)
   python "$DIR/evaluate_p3.py" --reference-root "$P3_REFERENCE_ROOT" --mode pilot
   ;;
 *) echo 'Unknown P3 stage';exit 2 ;;
esac
