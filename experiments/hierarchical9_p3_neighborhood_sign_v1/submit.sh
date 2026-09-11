#!/usr/bin/env bash
# Single submission per invocation: respects job-count limits, no job arrays/dependencies.
set -euo pipefail
STAGE="${1:-}"
case "$STAGE" in smoke|pilot|eval|pack) ;; *) echo 'Usage: bash experiments/hierarchical9_p3_neighborhood_sign_v1/submit.sh smoke|pilot|eval|pack'; exit 2;; esac
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
ROOT="${P3_REFERENCE_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_cal_pilot_20260910_183735_099nu5}"
ROOT=$(realpath -e "$ROOT")
[[ "$ROOT" != *','* && "$REPO" != *','* ]] || { echo 'Comma in export paths unsupported';exit 2; }
DIR=experiments/hierarchical9_p3_neighborhood_sign_v1
cd "$REPO"
if [[ "$STAGE" == pack ]]; then
  python3 "$DIR/pack_reports.py" --reference-root "$ROOT"
  exit
fi
git diff --quiet && git diff --cached --quiet || { echo 'Save tracked changes before submission';exit 2; }
sha256sum -c "$DIR/SHA256SUMS"
for f in cache/manifest.json cache/train.npz cache/val.npz P0/final.pt P1/final.pt P2/final.pt; do
  [[ -s "$ROOT/$f" ]] || { echo "Missing reference: $ROOT/$f";exit 2; }
done
# Login-node gates are stdlib-only; GPU worker checks hashes/lineage/model/data/geometry.
python3 - "$ROOT" "$STAGE" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); stage=sys.argv[2]
m=json.loads((root/'cache/manifest.json').read_text())
if m.get('status')!='COMPLETE': raise SystemExit('Reference cache not complete')
for k,v in dict(train_points=512,train_anchors=4,val_points=64,val_anchors=2).items():
    if m.get('args',{}).get(k)!=v: raise SystemExit('Not the full pilot cache: '+k)
for arm in ('P0','P1','P2'):
    r=json.loads((root/arm/'run.json').read_text())
    if r.get('status')!='COMPLETE' or r.get('args',{}).get('steps')!=2000:
        raise SystemExit('Incomplete 2000-step reference: '+arm)
if stage in ('pilot','eval'):
    r=json.loads((root/'evaluation_p3_smoke/manifest.json').read_text())
    if r.get('status')!='COMPLETE' or r.get('mode')!='smoke' or r.get('pilot_updates')!=2:
        raise SystemExit('Finish current P3 smoke including evaluation before pilot')
if stage=='eval':
    r=json.loads((root/'P3/run.json').read_text())
    if r.get('status')!='COMPLETE' or r.get('successful_updates')!=2000 or r.get('pair_sample_streams')!='MATCH':
        raise SystemExit('Finish P3 2000 updates with matched original streams')
PY
case "$STAGE" in
 smoke) dest=P3_smoke;gpus=4;cpus=16;limit=01:00:00;key=P3_SMOKE_JOB ;;
 pilot) dest=P3;gpus=4;cpus=16;limit=04:00:00;key=P3_TRAIN_JOB ;;
 eval) dest=evaluation_p3;gpus=1;cpus=4;limit=02:00:00;key=P3_EVAL_JOB ;;
esac
[[ ! -e "$ROOT/$dest" ]] || { echo "Refusing existing output: $ROOT/$dest";exit 2; }
if [[ "$STAGE" == smoke && -e "$ROOT/evaluation_p3_smoke" ]]; then echo 'Existing smoke evaluation';exit 2;fi
mkdir -p "$ROOT/logs"
LOCK="$ROOT/.p3_${STAGE}_submission"
mkdir "$LOCK" || { echo "Stage already submitted/reserved: $LOCK; inspect p3_jobs.env";exit 2; }
HEAD=$(git rev-parse HEAD)
EXPORT="ALL,P3_REFERENCE_ROOT=$ROOT,P3_CODE_REPO=$REPO,P3_CODE_SHA=$HEAD,P3_STAGE=$STAGE"
if ! JOB=$(sbatch --parsable --partition=GPU --nodelist=3090node3 --nodes=1 --ntasks=1 \
  --gres="gpu:3090:$gpus" --cpus-per-task="$cpus" --time="$limit" --job-name="h9_p3_$STAGE" \
  --chdir="$REPO" --output="$ROOT/logs/p3_${STAGE}_%j.out" --error="$ROOT/logs/p3_${STAGE}_%j.out" \
  --export="$EXPORT" "$REPO/$DIR/worker.sbatch"); then
  rmdir "$LOCK"
  echo 'sbatch rejected. References unchanged; no downstream submission. Wait for quota before retrying.'
  exit 1
fi
JOB=${JOB%%;*}
if [[ ! "$JOB" =~ ^[0-9]+$ ]]; then
  printf '%s\n' "$JOB" > "$LOCK/unparsed_submission.txt"
  echo 'Ambiguous sbatch reply; keep reservation and inspect Slurm, do not retry blindly';exit 2
fi
printf '%s\n' "$JOB" > "$LOCK/job_id"
printf '%s=%s\n' "$key" "$JOB" >> "$ROOT/p3_jobs.env"
printf '\n[submitted] stage=%s job=%s\nREFERENCE_ROOT=%s\n' "$STAGE" "$JOB" "$ROOT"
printf 'tail -n 100 -F %q\n' "$ROOT/logs/p3_${STAGE}_${JOB}.out"
printf 'sacct -j %s --format=JobID,State,ExitCode,Elapsed\n' "$JOB"
echo 'One job. No prepare/retraining of controls, arrays, dependencies or runtime switch.'
