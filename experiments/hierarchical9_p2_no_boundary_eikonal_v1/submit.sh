#!/usr/bin/env bash
# Exactly ONE sbatch per invocation. No job array or queued downstream dependencies.
set -euo pipefail
STAGE="${1:-}"
case "$STAGE" in smoke|pilot|eval) ;; *) echo 'Usage: bash experiments/hierarchical9_p2_no_boundary_eikonal_v1/submit.sh smoke|pilot|eval'; exit 2;; esac
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
ROOT="${P2_REFERENCE_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_cal_pilot_20260910_183735_099nu5}"
ROOT=$(realpath -e "$ROOT")
[[ "$ROOT" != *','* && "$REPO" != *','* ]] || { echo 'Comma in export paths unsupported'; exit 2; }
DIR=experiments/hierarchical9_p2_no_boundary_eikonal_v1
cd "$REPO"
git diff --quiet && git diff --cached --quiet || { echo 'Save tracked changes before submitting'; exit 2; }
sha256sum -c "$DIR/SHA256SUMS"
for file in cache/manifest.json cache/train.npz cache/val.npz P0/final.pt P1/final.pt; do
  [[ -s "$ROOT/$file" ]] || { echo "Missing reference: $ROOT/$file"; exit 2; }
done
# Stdlib-only gates on login node. Full model/cache/source checks run on allocated GPUs.
python3 - "$ROOT" "$STAGE" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);stage=sys.argv[2]
m=json.loads((root/'cache/manifest.json').read_text())
if m.get('status')!='COMPLETE': raise SystemExit('Reference cache is not COMPLETE')
for k,v in dict(train_points=512,train_anchors=4,val_points=64,val_anchors=2).items():
    if m.get('args',{}).get(k)!=v: raise SystemExit('Not the full pilot cache: '+k)
for arm in ('P0','P1'):
    r=json.loads((root/arm/'run.json').read_text())
    if r.get('status')!='COMPLETE' or r.get('args',{}).get('steps')!=2000:
        raise SystemExit(arm+' must be the completed 2000-step control')
if stage in ('pilot','eval'):
    smoke=json.loads((root/'evaluation_p2_smoke/manifest.json').read_text())
    if smoke.get('status')!='COMPLETE' or smoke.get('mode')!='smoke' or smoke.get('pilot_updates')!=2:
        raise SystemExit('Complete the P2 smoke including evaluation first')
if stage=='eval':
    r=json.loads((root/'P2/run.json').read_text())
    if r.get('status')!='COMPLETE' or r.get('successful_updates')!=2000 or r.get('pair_sample_streams')!='MATCH':
        raise SystemExit('P2 must complete 2000 updates with matched streams')
PY
case "$STAGE" in
 smoke) dest=P2_smoke;gpus=4;cpus=16;limit=01:00:00;key=P2_SMOKE_JOB ;;
 pilot) dest=P2;gpus=4;cpus=16;limit=04:00:00;key=P2_TRAIN_JOB ;;
 eval) dest=evaluation_p2;gpus=1;cpus=4;limit=02:00:00;key=P2_EVAL_JOB ;;
esac
[[ ! -e "$ROOT/$dest" ]] || { echo "Refusing existing output: $ROOT/$dest";exit 2; }
if [[ "$STAGE" == smoke && -e "$ROOT/evaluation_p2_smoke" ]]; then echo 'Existing P2 smoke evaluation';exit 2;fi
mkdir -p "$ROOT/logs"
LOCK="$ROOT/.p2_${STAGE}_submission"
mkdir "$LOCK" || { echo "Stage already submitted/reserved: $LOCK; inspect p2_jobs.env, do not duplicate";exit 2; }
HEAD=$(git rev-parse HEAD)
EXPORT="ALL,P2_REFERENCE_ROOT=$ROOT,P2_CODE_REPO=$REPO,P2_CODE_SHA=$HEAD,P2_STAGE=$STAGE"
if ! JOB=$(sbatch --parsable --partition=GPU --nodelist=3090node3 --nodes=1 --ntasks=1 \
  --gres="gpu:3090:$gpus" --cpus-per-task="$cpus" --time="$limit" --job-name="h9_p2_$STAGE" \
  --chdir="$REPO" --output="$ROOT/logs/p2_${STAGE}_%j.out" --error="$ROOT/logs/p2_${STAGE}_%j.out" \
  --export="$EXPORT" "$REPO/$DIR/worker.sbatch"); then
  rmdir "$LOCK"
  echo 'sbatch rejected; existing cache/P0/P1 unchanged. No downstream job was submitted.'
  exit 1
fi
JOB=${JOB%%;*}
if [[ ! "$JOB" =~ ^[0-9]+$ ]]; then
  printf '%s\n' "$JOB" > "$LOCK/unparsed_submission.txt"
  echo 'Unexpected sbatch response; retain reservation and inspect Slurm before retrying';exit 2
fi
printf '%s\n' "$JOB" > "$LOCK/job_id"
printf '%s=%s\n' "$key" "$JOB" >> "$ROOT/p2_jobs.env"
printf '\n[submitted] stage=%s job=%s\nREFERENCE_ROOT=%s\n' "$STAGE" "$JOB" "$ROOT"
printf 'tail -n 100 -F %q\n' "$ROOT/logs/p2_${STAGE}_${JOB}.out"
printf 'sacct -j %s --format=JobID,State,ExitCode,Elapsed\n' "$JOB"
echo 'One job only. No prepare, no automatic next-stage submission, no runtime switch.'
