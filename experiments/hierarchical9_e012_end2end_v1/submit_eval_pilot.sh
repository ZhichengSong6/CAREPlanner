#!/usr/bin/env bash
set -euo pipefail
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
ROOT="${E012_REFERENCE_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_cal_pilot_20260910_183735_099nu5}"
ROOT=$(realpath -e "$ROOT")
DIR=experiments/hierarchical9_e012_end2end_v1
cd "$REPO"
git diff --quiet && git diff --cached --quiet || { echo 'Save tracked changes before submission'; exit 2; }

python3 - "$ROOT" <<'PY'
import json,sys,hashlib
from pathlib import Path
root=Path(sys.argv[1])
for arm in ('E0','E1','E2'):
    base=root/'e012_end2end'/arm
    run=base/'run.json'; final=base/'final.pt'
    if not run.is_file() or not final.is_file():
        raise SystemExit(f'Missing formal arm artifact: {arm}')
    d=json.loads(run.read_text())
    if d.get('status')!='COMPLETE' or d.get('successful_updates')!=2000:
        raise SystemExit(f'Formal arm not COMPLETE/2000: {arm}: {d.get("status")} {d.get("successful_updates")}')
    h=hashlib.sha256(final.read_bytes()).hexdigest()
    if d.get('final_sha256')!=h:
        raise SystemExit(f'Formal checkpoint hash mismatch: {arm}')
print('[preflight] E0/E1/E2 formal checkpoints COMPLETE @2000 and hashes match')
PY

OUT="$ROOT/evaluation_e012_end2end"
if [[ -e "$OUT" ]]; then
  STATUS=$(python3 - "$OUT/manifest.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file(): print('MISSING')
else:
    try: print(json.loads(p.read_text()).get('status','UNKNOWN'))
    except Exception: print('INVALID')
PY
)
  if [[ "$STATUS" == COMPLETE ]]; then
    echo "Formal E012 evaluation already COMPLETE: $OUT"
    exit 2
  fi
  ARCHIVE="$ROOT/.e012_eval_failed_$(date +%Y%m%d_%H%M%S)"
  mv "$OUT" "$ARCHIVE"
  echo "[archive] moved previous incomplete evaluation status=$STATUS to $ARCHIVE"
fi

LOCK="$ROOT/.e012_eval_pilot_submission"
if [[ -e "$LOCK" ]]; then
  OLD=""; [[ -s "$LOCK/job_id" ]] && OLD=$(cat "$LOCK/job_id")
  STATE=""; [[ "$OLD" =~ ^[0-9]+$ ]] && STATE=$(sacct -n -X -j "$OLD" --format=State --noheader 2>/dev/null | awk 'NF{print $1;exit}' | sed 's/+.*//')
  case "$STATE" in
    FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|COMPLETED)
      mv "$LOCK" "$ROOT/.e012_eval_pilot_submission_${OLD}_${STATE}_$(date +%Y%m%d_%H%M%S)"
      ;;
    *) echo "Eval-only stage already reserved${OLD:+ by job $OLD}${STATE:+ state=$STATE}"; exit 2;;
  esac
fi
mkdir "$LOCK"
HEAD=$(git rev-parse HEAD)
EXPORT="ALL,E012_REFERENCE_ROOT=$ROOT,E012_CODE_REPO=$REPO,E012_CODE_SHA=$HEAD"
ERR=$(mktemp); trap 'rm -f "$ERR"' EXIT
set +e
JOB=$(sbatch --parsable --partition=GPU --nodelist=3090node3 --nodes=1 --ntasks=1 --gres=gpu:3090:1 --cpus-per-task=8 --time=02:00:00 --job-name=h9_e012_eval --chdir="$REPO" --output="$ROOT/logs/e012_eval_%j.out" --error="$ROOT/logs/e012_eval_%j.out" --export="$EXPORT" "$REPO/$DIR/eval_worker.sbatch" 2>"$ERR")
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  rmdir "$LOCK"
  cat "$ERR" >&2
  exit $RC
fi
JOB=${JOB%%;*}
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "$JOB" > "$LOCK/unparsed_submission.txt"; exit 2; }
echo "$JOB" > "$LOCK/job_id"
printf 'E012_EVAL_JOB=%s\n' "$JOB" >> "$ROOT/e012_jobs.env"
echo "[submitted] E012 eval-only pilot job=$JOB"
echo "tail -n 150 -F $ROOT/logs/e012_eval_${JOB}.out"
echo "sacct -j $JOB --format=JobID,State,ExitCode,Elapsed"
