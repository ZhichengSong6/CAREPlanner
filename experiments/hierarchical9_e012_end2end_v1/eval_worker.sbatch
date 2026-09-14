#!/usr/bin/env bash
set -euo pipefail
: "${E012_REFERENCE_ROOT:?}" "${E012_CODE_REPO:?}" "${E012_CODE_SHA:?}"
cd "$E012_CODE_REPO"
[[ "$(git rev-parse HEAD)" == "$E012_CODE_SHA" ]] || { echo 'Code HEAD changed after eval submission'; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo 'Tracked code changed after eval submission'; exit 2; }
DIR=experiments/hierarchical9_e012_end2end_v1
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
[[ -f "$CONDA_SH" ]] || { echo 'Set CONDA_SH'; exit 2; }
set +u
source "$CONDA_SH"
conda activate "${CONDA_ENV:-viscdf}"
set -u
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "[eval-only] E012 pilot root=$E012_REFERENCE_ROOT code=$E012_CODE_SHA host=$(hostname)"
CUDA_VISIBLE_DEVICES=0 python "$DIR/evaluate_e012.py" --reference-root "$E012_REFERENCE_ROOT" --mode pilot
echo "[done] e012_eval_only_complete mode=pilot"
