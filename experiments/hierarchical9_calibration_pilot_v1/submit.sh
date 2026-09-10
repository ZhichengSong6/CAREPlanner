#!/usr/bin/env bash
# Run this on the LOGIN node. It only submits jobs; no CUDA/data preparation here.
set -euo pipefail
MODE="${1:-}"
[[ "$MODE" == smoke || "$MODE" == pilot ]] || { echo 'Usage: bash experiments/hierarchical9_calibration_pilot_v1/submit.sh smoke|pilot'; exit 2; }
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
ARTIFACT_ROOT="${CAL_ARTIFACT_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner}"
[[ "$REPO" != *','* && "$ARTIFACT_ROOT" != *','* ]] || { echo 'Comma in export paths unsupported';exit 2; }
cd "$REPO"
git diff --quiet && git diff --cached --quiet || { echo 'Save tracked changes before submitting reproducible jobs'; exit 2; }
DIR=experiments/hierarchical9_calibration_pilot_v1
sha256sum -c "$DIR/SHA256SUMS"
mkdir -p "$ARTIFACT_ROOT/outputs/mainline_b"
ROOT=$(mktemp -d "$ARTIFACT_ROOT/outputs/mainline_b/h9_cal_${MODE}_$(date +%Y%m%d_%H%M%S)_XXXXXX")
mkdir "$ROOT/logs"
HEAD=$(git rev-parse HEAD)
EXPORT="ALL,CAL_RUN_ROOT=$ROOT,CAL_ARTIFACT_ROOT=$ARTIFACT_ROOT,CAL_CODE_REPO=$REPO,CAL_MODE=$MODE,CAL_CODE_SHA=$HEAD"
WORKER="$REPO/$DIR/worker.sbatch"
COMMON=(--parsable --partition=GPU --nodelist=3090node3 --nodes=1 --ntasks=1 --chdir="$REPO")
# No --mem or --mem-per-* options (cluster RealMemory workaround).
PREP=$(sbatch "${COMMON[@]}" --job-name=h9cal_prep --gres=gpu:3090:1 --cpus-per-task=4 --time=08:00:00 \
  --output="$ROOT/logs/prepare_%j.out" --error="$ROOT/logs/prepare_%j.out" \
  --export="$EXPORT,CAL_STAGE=prepare" "$WORKER")
PREP=${PREP%%;*}
printf 'RUN_ROOT=%q\nPREP_JOB=%q\n' "$ROOT" "$PREP" > "$ROOT/jobs.env"
# %1 permits only one four-GPU arm at a time. Execution order is immaterial.
TRAIN=$(sbatch "${COMMON[@]}" --job-name=h9cal_pair --array=0-1%1 --gres=gpu:3090:4 --cpus-per-task=16 --time=12:00:00 \
  --dependency="afterok:$PREP" --output="$ROOT/logs/train_%A_%a.out" --error="$ROOT/logs/train_%A_%a.out" \
  --export="$EXPORT,CAL_STAGE=train" "$WORKER")
TRAIN=${TRAIN%%;*}
printf 'TRAIN_JOB=%q\n' "$TRAIN" >> "$ROOT/jobs.env"
EVAL=$(sbatch "${COMMON[@]}" --job-name=h9cal_eval --gres=gpu:3090:1 --cpus-per-task=4 --time=08:00:00 \
  --dependency="afterok:$TRAIN" --output="$ROOT/logs/evaluate_%j.out" --error="$ROOT/logs/evaluate_%j.out" \
  --export="$EXPORT,CAL_STAGE=evaluate" "$WORKER")
EVAL=${EVAL%%;*}
printf 'EVAL_JOB=%q\n' "$EVAL" >> "$ROOT/jobs.env"
cp "$DIR/SHA256SUMS" "$ROOT/submitted_SHA256SUMS"
printf '%s\n' "$ROOT" > "$ARTIFACT_ROOT/outputs/mainline_b/last_h9_cal_${MODE}.path"
printf '\n[submitted] mode=%s\nRUN_ROOT=%s\nprepare=%s train_array=%s evaluate=%s\n' "$MODE" "$ROOT" "$PREP" "$TRAIN" "$EVAL"
printf '\nsource %q\nsqueue -r -j %s,%s,%s\n' "$ROOT/jobs.env" "$PREP" "$TRAIN" "$EVAL"
printf 'tail -n 100 -F %q\n' "$ROOT/logs/prepare_$PREP.out"
echo 'P0/P1 start only after preparation succeeds; evaluation starts only after BOTH arms succeed.'
echo 'A failed prerequisite leaves downstream jobs waiting; inspect logs, do not duplicate-submit blindly.'
echo 'No master/runtime changes. Do not switch or edit this code worktree while the jobs are running.'
