#!/usr/bin/env bash
set -euo pipefail
STAGE="${1:-}"
if [[ "$STAGE" != smoke && "$STAGE" != pilot && "$STAGE" != pack ]]; then
  echo 'Usage: submit.sh smoke|pilot|pack'; exit 2
fi
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
ROOT="${E012_REFERENCE_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_cal_pilot_20260910_183735_099nu5}"
ROOT=$(realpath -e "$ROOT")
DIR=experiments/hierarchical9_e012_end2end_v1
cd "$REPO"
if [[ "$STAGE" == pack ]]; then python3 "$DIR/pack_reports.py" --reference-root "$ROOT"; exit 0; fi
git diff --quiet && git diff --cached --quiet || { echo 'Save tracked changes before submission'; exit 2; }
for f in cache/manifest.json cache/train.npz cache/val.npz P0/final.pt P0/run.json; do
  [[ -s "$ROOT/$f" ]] || { echo "Missing reference: $ROOT/$f"; exit 2; }
done
if [[ "$STAGE" == pilot ]]; then
  python3 - "$ROOT/evaluation_e012_end2end_smoke/manifest.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
if m.get('status')!='COMPLETE' or m.get('mode')!='smoke': raise SystemExit('Complete E012 smoke first')
PY
fi
if [[ "$STAGE" == smoke ]]; then
  TRAIN_DIR=e012_end2end_smoke; EVAL_DIR=evaluation_e012_end2end_smoke; LIMIT=02:00:00; KEY=E012_SMOKE_JOB
else
  TRAIN_DIR=e012_end2end; EVAL_DIR=evaluation_e012_end2end; LIMIT=12:00:00; KEY=E012_PILOT_JOB
fi
mkdir -p "$ROOT/logs"; LOCK="$ROOT/.e012_${STAGE}_submission"
if [[ -e "$LOCK" ]]; then
  OLD_JOB=""; [[ -s "$LOCK/job_id" ]] && OLD_JOB=$(cat "$LOCK/job_id")
  STATE=""; [[ "$OLD_JOB" =~ ^[0-9]+$ ]] && STATE=$(sacct -n -X -j "$OLD_JOB" --format=State --noheader 2>/dev/null | awk 'NF{print $1;exit}' | sed 's/+.*//')
  case "$STATE" in
    FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL)
      python3 - "$ROOT/$TRAIN_DIR" "$ROOT/$EVAL_DIR" "$STAGE" <<'PY'
import json,sys
from pathlib import Path
train,ev,stage=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
manifest=ev/'manifest.json'
if manifest.is_file():
    try: m=json.loads(manifest.read_text())
    except Exception: m={}
    if m.get('status')=='COMPLETE': raise SystemExit(f'Refusing reclaim: completed evaluation {manifest}')
if stage=='pilot':
    for arm in ('E0','E1','E2'):
        r=train/arm/'run.json'
        if r.is_file():
            try: d=json.loads(r.read_text())
            except Exception: d={}
            if d.get('status')=='COMPLETE': raise SystemExit(f'Refusing automatic pilot rerun: completed arm {r}')
PY
      ARCHIVE="$ROOT/.e012_${STAGE}_failed_${OLD_JOB}_$(date +%Y%m%d_%H%M%S)"; mkdir "$ARCHIVE"
      mv "$LOCK" "$ARCHIVE/submission_lock"
      [[ -e "$ROOT/$TRAIN_DIR" ]] && mv "$ROOT/$TRAIN_DIR" "$ARCHIVE/$TRAIN_DIR"
      [[ -e "$ROOT/$EVAL_DIR" ]] && mv "$ROOT/$EVAL_DIR" "$ARCHIVE/$EVAL_DIR"
      echo "[reclaim] archived failed E012 job=$OLD_JOB state=$STATE to $ARCHIVE";;
    *) echo "E012 stage already reserved${OLD_JOB:+ by job $OLD_JOB}${STATE:+ state=$STATE}; not reclaiming"; exit 2;;
  esac
fi
[[ ! -e "$ROOT/$TRAIN_DIR" && ! -e "$ROOT/$EVAL_DIR" ]] || { echo 'E012 output exists without reclaimable failed reservation'; exit 2; }
mkdir "$LOCK"; HEAD=$(git rev-parse HEAD)
EXPORT="ALL,E012_REFERENCE_ROOT=$ROOT,E012_CODE_REPO=$REPO,E012_CODE_SHA=$HEAD,E012_STAGE=$STAGE"
if ! JOB=$(sbatch --parsable --partition=GPU --nodelist=3090node3 --nodes=1 --ntasks=1 --gres=gpu:3090:4 --cpus-per-task=16 --time="$LIMIT" --job-name="h9_e012_$STAGE" --chdir="$REPO" --output="$ROOT/logs/e012_${STAGE}_%j.out" --error="$ROOT/logs/e012_${STAGE}_%j.out" --export="$EXPORT" "$REPO/$DIR/worker.sbatch"); then
  rmdir "$LOCK"; exit 1
fi
JOB=${JOB%%;*}; [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "$JOB" > "$LOCK/unparsed_submission.txt"; exit 2; }
echo "$JOB" > "$LOCK/job_id"; printf '%s=%s\n' "$KEY" "$JOB" >> "$ROOT/e012_jobs.env"
echo "[submitted] E012 stage=$STAGE job=$JOB"
echo "tail -n 120 -F $ROOT/logs/e012_${STAGE}_${JOB}.out"
echo "sacct -j $JOB --format=JobID,State,ExitCode,Elapsed"
