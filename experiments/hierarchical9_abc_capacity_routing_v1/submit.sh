#!/usr/bin/env bash
set -euo pipefail
STAGE="${1:-}"
if [[ "$STAGE" != "smoke" && "$STAGE" != "pilot" && "$STAGE" != "pack" ]]; then
  echo 'Usage: submit.sh smoke|pilot|pack'
  exit 2
fi
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
ROOT="${ABC_REFERENCE_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_cal_pilot_20260910_183735_099nu5}"
ROOT=$(realpath -e "$ROOT")
DIR=experiments/hierarchical9_abc_capacity_routing_v1
cd "$REPO"
if [[ "$STAGE" == "pack" ]]; then
  python3 "$DIR/pack_reports.py" --reference-root "$ROOT"
  exit 0
fi
git diff --quiet && git diff --cached --quiet || { echo 'Save tracked changes before submission'; exit 2; }
for f in cache/manifest.json cache/train.npz cache/val.npz P0/final.pt P0/run.json; do
  [[ -s "$ROOT/$f" ]] || { echo "Missing reference: $ROOT/$f"; exit 2; }
done
if [[ "$STAGE" == "pilot" ]]; then
  python3 - "$ROOT/evaluation_abc_capacity_routing_smoke/manifest.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
if m.get('status')!='COMPLETE' or m.get('mode')!='smoke':
    raise SystemExit('Complete ABC smoke first')
PY
fi
if [[ "$STAGE" == "smoke" ]]; then
  TRAIN_DIR=abc_capacity_routing_smoke
  EVAL_DIR=evaluation_abc_capacity_routing_smoke
  LIMIT=01:30:00
  KEY=ABC_SMOKE_JOB
else
  TRAIN_DIR=abc_capacity_routing
  EVAL_DIR=evaluation_abc_capacity_routing
  LIMIT=08:00:00
  KEY=ABC_PILOT_JOB
fi
[[ ! -e "$ROOT/$TRAIN_DIR" && ! -e "$ROOT/$EVAL_DIR" ]] || { echo 'ABC output already exists; refusing overwrite'; exit 2; }
mkdir -p "$ROOT/logs"
LOCK="$ROOT/.abc_${STAGE}_submission"
mkdir "$LOCK" || { echo 'ABC stage already reserved'; exit 2; }
HEAD=$(git rev-parse HEAD)
EXPORT="ALL,ABC_REFERENCE_ROOT=$ROOT,ABC_CODE_REPO=$REPO,ABC_CODE_SHA=$HEAD,ABC_STAGE=$STAGE"
if ! JOB=$(sbatch --parsable --partition=GPU --nodelist=3090node3 --nodes=1 --ntasks=1 --gres=gpu:3090:4 --cpus-per-task=16 --time="$LIMIT" --job-name="h9_abc_$STAGE" --chdir="$REPO" --output="$ROOT/logs/abc_${STAGE}_%j.out" --error="$ROOT/logs/abc_${STAGE}_%j.out" --export="$EXPORT" "$REPO/$DIR/worker.sbatch"); then
  rmdir "$LOCK"
  exit 1
fi
JOB=${JOB%%;*}
printf '%s\n' "$JOB" > "$LOCK/job_id"
printf '%s=%s\n' "$KEY" "$JOB" >> "$ROOT/abc_jobs.env"
echo "[submitted] ABC stage=$STAGE job=$JOB"
echo "tail -n 100 -F $ROOT/logs/abc_${STAGE}_${JOB}.out"
echo "sacct -j $JOB --format=JobID,State,ExitCode,Elapsed"
