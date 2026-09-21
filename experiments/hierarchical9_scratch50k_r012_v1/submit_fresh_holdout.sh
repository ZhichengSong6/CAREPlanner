#!/usr/bin/env bash
set -euo pipefail

REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
DIR=experiments/hierarchical9_scratch50k_r012_v1
ROOT="${R012_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_scratch50k_r012_v1}"
REF="${E012_REFERENCE_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_cal_pilot_20260910_183735_099nu5}"

mkdir -p "$ROOT/logs"
ROOT=$(realpath -e "$ROOT")
REF=$(realpath -e "$REF")
cd "$REPO"

git diff --quiet && git diff --cached --quiet || {
  echo "[ERROR] working tree has tracked changes; commit/stash first" >&2
  exit 2
}

bash "$DIR/submit.sh" verify

python3 - "$ROOT/evaluation_unified_dev/manifest.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
assert m.get("status")=="COMPLETE",m
print("[preflight] development evaluation COMPLETE",flush=True)
PY

OUT="$ROOT/evaluation_fresh_holdout_v1"
if [[ -e "$OUT" ]]; then
  STATUS=""
  [[ -s "$OUT/manifest.json" ]] && STATUS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$OUT/manifest.json" 2>/dev/null || true)
  if [[ "$STATUS" == "COMPLETE" ]]; then
    echo "[ERROR] completed fresh holdout already exists: $OUT" >&2
    exit 2
  fi
  mv "$OUT" "$ROOT/.evaluation_fresh_holdout_v1_failed_$(date +%Y%m%d_%H%M%S)"
fi

HEAD=$(git rev-parse HEAD)
EXPORT="ALL,R012_ROOT=$ROOT,E012_REFERENCE_ROOT=$REF,R012_CODE_REPO=$REPO,R012_CODE_SHA=$HEAD"

JOB=$(sbatch --parsable \
  --partition=GPU \
  --nodelist='3090node[1-3]' \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:3090:4 \
  --cpus-per-task=16 \
  --time=02:00:00 \
  --job-name=h9_r1_fresh \
  --chdir="$REPO" \
  --output="$ROOT/logs/r012_fresh_holdout_%j.out" \
  --error="$ROOT/logs/r012_fresh_holdout_%j.out" \
  --export="$EXPORT" \
  "$REPO/$DIR/fresh_holdout_worker.sbatch")

JOB=${JOB%%;*}
printf 'R012_FRESH_HOLDOUT_JOB=%s\n' "$JOB" >> "$ROOT/r012_jobs.env"

echo "[submitted] fresh holdout job=$JOB nodes=3090node[1-3] gpus=4 code=$HEAD"
echo "tail -n 160 -F $ROOT/logs/r012_fresh_holdout_${JOB}.out"
echo "sacct -j $JOB --format=JobID,State,ExitCode,Elapsed,NodeList"
