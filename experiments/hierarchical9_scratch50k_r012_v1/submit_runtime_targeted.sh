#!/usr/bin/env bash
set -euo pipefail
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
ROOT="${R012_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_scratch50k_r012_v1}"
mkdir -p "$ROOT/logs"
ROOT=$(realpath -e "$ROOT")
cd "$REPO"
HEAD=$(git rev-parse HEAD)
JOB=$(sbatch --parsable \
  --partition=GPU \
  --nodelist=3090node2 \
  --nodes=1 --ntasks=1 --gres=gpu:3090:1 --cpus-per-task=8 --time=01:00:00 \
  --job-name=h9_r1_rt_tgt \
  --chdir="$REPO" \
  --output="$ROOT/logs/r1_runtime_targeted_%j.out" \
  --error="$ROOT/logs/r1_runtime_targeted_%j.out" \
  --export="ALL,R012_CODE_REPO=$REPO,R012_CODE_SHA=$HEAD,R012_ROOT=$ROOT" \
  "$REPO/experiments/hierarchical9_scratch50k_r012_v1/runtime_targeted_worker.sbatch")
JOB=${JOB%%;*}
printf 'R012_RUNTIME_TARGETED_JOB=%s\n' "$JOB" >> "$ROOT/r012_jobs.env"
echo "[submitted] runtime targeted job=$JOB node=3090node2 gpu=1"
echo "tail --retry -n 160 -F $ROOT/logs/r1_runtime_targeted_${JOB}.out"
