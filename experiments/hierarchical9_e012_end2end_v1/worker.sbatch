#!/usr/bin/env bash
set -euo pipefail
: "${E012_REFERENCE_ROOT:?}" "${E012_CODE_REPO:?}" "${E012_CODE_SHA:?}" "${E012_STAGE:?}"
cd "$E012_CODE_REPO"
[[ "$(git rev-parse HEAD)" == "$E012_CODE_SHA" ]] || { echo 'Code HEAD changed after submission'; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo 'Tracked code changed after submission'; exit 2; }
DIR=experiments/hierarchical9_e012_end2end_v1
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
[[ -f "$CONDA_SH" ]] || { echo 'Set CONDA_SH'; exit 2; }
set +u
source "$CONDA_SH"
conda activate "${CONDA_ENV:-viscdf}"
set -u
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python "$DIR/test_e012.py"
MODE="$E012_STAGE"
[[ "$MODE" == smoke || "$MODE" == pilot ]] || { echo 'Unknown E012 stage'; exit 2; }
LOGDIR="$E012_REFERENCE_ROOT/logs/e012_${MODE}_${SLURM_JOB_ID}"
mkdir -p "$LOGDIR"
echo "[launch] E012 mode=$MODE root=$E012_REFERENCE_ROOT code=$E012_CODE_SHA host=$(hostname)"
PIDS=(); ARMS=(E0 E1 E2)
for i in 0 1 2; do
  arm=${ARMS[$i]}
  echo "[launch-arm] $arm gpu=$i log=$LOGDIR/${arm}.out"
  CUDA_VISIBLE_DEVICES=$i python "$DIR/train_arm.py" --reference-root "$E012_REFERENCE_ROOT" --arm "$arm" --mode "$MODE" > "$LOGDIR/${arm}.out" 2>&1 &
  PIDS+=("$!")
done
STATUS=0
for i in 0 1 2; do
  if ! wait "${PIDS[$i]}"; then echo "[arm-failed] ${ARMS[$i]} log=$LOGDIR/${ARMS[$i]}.out"; STATUS=1
  else echo "[arm-complete] ${ARMS[$i]}"; fi
done
if [[ "$STATUS" != 0 ]]; then
  for arm in E0 E1 E2; do echo "===== $arm tail ====="; tail -n 100 "$LOGDIR/${arm}.out" || true; done
  exit 1
fi
for arm in E0 E1 E2; do echo "===== $arm final tail ====="; tail -n 35 "$LOGDIR/${arm}.out"; done
CUDA_VISIBLE_DEVICES=3 python "$DIR/evaluate_e012.py" --reference-root "$E012_REFERENCE_ROOT" --mode "$MODE"
echo "[done] e012_workflow_complete mode=$MODE"
