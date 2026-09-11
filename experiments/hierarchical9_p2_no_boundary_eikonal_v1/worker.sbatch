#!/usr/bin/env bash
set -euo pipefail
: "${P2_REFERENCE_ROOT:?}" "${P2_CODE_REPO:?}" "${P2_CODE_SHA:?}" "${P2_STAGE:?}"
cd "$P2_CODE_REPO"
[[ "$(git rev-parse HEAD)" == "$P2_CODE_SHA" ]] || { echo 'Code changed after submission'; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo 'Tracked code changed';exit 2; }
DIR=experiments/hierarchical9_p2_no_boundary_eikonal_v1
sha256sum -c "$DIR/SHA256SUMS"
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
[[ -f "$CONDA_SH" ]] || { echo 'Use CONDA_SH for the existing environment'; exit 2; }
set +u
source "$CONDA_SH"
conda activate "${CONDA_ENV:-viscdf}"
set -u
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
printf '[launch] stage=%s root=%s node=%s code=%s\n' "$P2_STAGE" "$P2_REFERENCE_ROOT" "$(hostname)" "$P2_CODE_SHA"
case "$P2_STAGE" in
 smoke|pilot)
   python "$DIR/test_p2.py"
   python -m torch.distributed.run --standalone --nproc_per_node=4 "$DIR/test_p2.py" --ddp --backend nccl
   python -m torch.distributed.run --standalone --nproc_per_node=4 "$DIR/train_p2.py" \
     --reference-root "$P2_REFERENCE_ROOT" --mode "$P2_STAGE"
   if [[ "$P2_STAGE" == smoke ]]; then
     # Same allocation; training ranks have exited. Evaluation uses cuda:0 only.
     python "$DIR/evaluate_p2.py" --reference-root "$P2_REFERENCE_ROOT" --mode smoke
     echo '[done] p2_smoke_workflow_complete'
   fi
   ;;
 eval)
   python "$DIR/evaluate_p2.py" --reference-root "$P2_REFERENCE_ROOT" --mode pilot
   ;;
 *) echo 'Unknown stage';exit 2;;
esac
