#!/usr/bin/env bash
set -euo pipefail
STAGE="${1:-}"
case "$STAGE" in smoke|R0|R1|R2|resume-R0|resume-R1|resume-R2|verify) ;; *) echo 'Usage: submit.sh smoke|R0|R1|R2|resume-R0|resume-R1|resume-R2|verify'; exit 2;; esac
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
DIR=experiments/hierarchical9_scratch50k_r012_v1
ROOT="${R012_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_scratch50k_r012_v1}"
mkdir -p "$ROOT/logs"; ROOT=$(realpath -e "$ROOT")
cd "$REPO"

git diff --quiet && git diff --cached --quiet || { echo '[ERROR] save tracked changes before submission'; exit 2; }
HEAD=$(git rev-parse HEAD)

verify_arm() {
  # Keep these declarations separate under `set -u`: Bash expands the RHS of a
  # compound `local arm=... p=...$arm...` before `arm` is assigned.
  local arm="$1"
  local steps="$2"
  local p="$ROOT/formal/$arm/run.json"
  python3 - "$p" "$arm" "$steps" <<'PY'
import hashlib,json,sys
from pathlib import Path
p,arm,n=Path(sys.argv[1]),sys.argv[2],int(sys.argv[3])
if not p.is_file(): raise SystemExit(f'Missing {p}')
r=json.loads(p.read_text())
final=p.parent/'final.pt'
if r.get('status')!='COMPLETE' or r.get('arm')!=arm or r.get('successful_updates')!=n or not r.get('final_sha256'):
    raise SystemExit(f'Incomplete {arm}: {r}')
if not final.is_file(): raise SystemExit(f'Missing {final}')
h=hashlib.sha256()
with final.open('rb') as f:
    for b in iter(lambda:f.read(1<<20),b''): h.update(b)
if h.hexdigest()!=r['final_sha256']:
    raise SystemExit(f'Checkpoint SHA mismatch for {arm}')
print(f'[verified] {arm} COMPLETE updates={n} stream={r.get("training_stream_sha256")}')
PY
}

if [[ "$STAGE" == verify ]]; then
  verify_arm R0 50000; verify_arm R1 50000; verify_arm R2 50000
  python3 - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); rows={a:json.loads((root/'formal'/a/'run.json').read_text()) for a in ('R0','R1','R2')}
h={r['training_stream_sha256'] for r in rows.values()}; u={r['training_stream_updates'] for r in rows.values()}
print('unique_stream_hashes=',len(h),'updates=',u)
if len(h)!=1 or u!={50000}: raise SystemExit('R0/R1/R2 streams do not match')
print('[done] R012 FORMAL STREAMS MATCH',next(iter(h)))
PY
  exit 0
fi

if [[ "$STAGE" != smoke ]]; then
  [[ -s "$ROOT/smoke_manifest.json" ]] || { echo '[ERROR] complete smoke first'; exit 2; }
  SMOKE_SHA=$(python3 - "$ROOT/smoke_manifest.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
assert m.get('status')=='COMPLETE' and m.get('training_streams')=='MATCH',m
print(m['code_sha'])
PY
)
  if [[ "$HEAD" != "$SMOKE_SHA" ]]; then
    # Training/scientific code must stay byte-for-byte frozen after smoke.  Permit
    # only this submission wrapper to change so scheduler/preflight bugs can be
    # repaired without invalidating an already-complete 50k arm.
    mapfile -t CHANGED < <(git diff --name-only "$SMOKE_SHA" "$HEAD")
    BAD=()
    for f in "${CHANGED[@]}"; do
      [[ "$f" == "$DIR/submit.sh" ]] || BAD+=("$f")
    done
    if (( ${#BAD[@]} )); then
      echo "[ERROR] scientific code changed since smoke: smoke=$SMOKE_SHA current=$HEAD" >&2
      printf '[ERROR] changed: %s\n' "${BAD[@]}" >&2
      echo '[ERROR] archive the experiment and rerun smoke before continuing.' >&2
      exit 2
    fi
    echo "[preflight] smoke COMPLETE; training code frozen at $SMOKE_SHA; allowing submit.sh-only orchestration patch at $HEAD"
  else
    echo "[preflight] smoke COMPLETE, streams MATCH, frozen code=$SMOKE_SHA"
  fi
fi

RESUME=0
if [[ "$STAGE" == resume-* ]]; then RESUME=1; ARM="${STAGE#resume-}"; MODE=train; KEY="R012_${ARM}_JOB"; LIMIT=3-00:00:00
elif [[ "$STAGE" == smoke ]]; then ARM=ALL; MODE=smoke; KEY=R012_SMOKE_JOB; LIMIT=01:00:00
else ARM="$STAGE"; MODE=train; KEY="R012_${ARM}_JOB"; LIMIT=3-00:00:00
fi

if [[ "$MODE" == train && $RESUME -eq 0 ]]; then
  case "$ARM" in
    R1) verify_arm R0 50000 ;;
    R2) verify_arm R0 50000; verify_arm R1 50000 ;;
  esac
  OUT="$ROOT/formal/$ARM"
  [[ ! -e "$OUT" || -z "$(ls -A "$OUT" 2>/dev/null || true)" ]] || { echo "[ERROR] $OUT exists; if interrupted use: bash $DIR/submit.sh resume-$ARM"; exit 2; }
elif [[ "$MODE" == train && $RESUME -eq 1 ]]; then
  OUT="$ROOT/formal/$ARM"
  [[ -s "$OUT/latest.pt" ]] || { echo "[ERROR] no resumable $OUT/latest.pt"; exit 2; }
  if [[ -s "$OUT/run.json" ]]; then
    STATUS=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("status",""))' "$OUT/run.json")
    [[ "$STATUS" != COMPLETE ]] || { echo "[ERROR] $ARM already COMPLETE"; exit 2; }
  fi
elif [[ "$MODE" == smoke ]]; then
  [[ ! -e "$ROOT/smoke" && ! -e "$ROOT/smoke_manifest.json" ]] || { echo '[ERROR] smoke output exists; inspect/archive it before rerun'; exit 2; }
fi

LOCK="$ROOT/.submit_${STAGE//\//_}"
if [[ -e "$LOCK" ]]; then
  OLD=""; [[ -s "$LOCK/job_id" ]] && OLD=$(cat "$LOCK/job_id")
  STATE=""; [[ "$OLD" =~ ^[0-9]+$ ]] && STATE=$(sacct -n -X -j "$OLD" --format=State --noheader 2>/dev/null | awk 'NF{print $1;exit}' | sed 's/+.*//')
  case "$STATE" in COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL) mv "$LOCK" "$ROOT/.submit_archive_${STAGE}_${OLD}_$(date +%Y%m%d_%H%M%S)";; *) echo "[ERROR] stage reserved${OLD:+ job=$OLD}${STATE:+ state=$STATE}"; exit 2;; esac
fi
mkdir "$LOCK"
EXPORT="ALL,R012_ROOT=$ROOT,R012_CODE_REPO=$REPO,R012_CODE_SHA=$HEAD,R012_MODE=$MODE,R012_ARM=$ARM,R012_RESUME=$RESUME"
ERR=$(mktemp); trap 'rm -f "$ERR"' EXIT
set +e
JOB=$(sbatch --parsable --partition=GPU --nodelist=3090node3 --nodes=1 --ntasks=1 --gres=gpu:3090:4 --cpus-per-task=16 --time="$LIMIT" \
  --job-name="h9_r012_${STAGE}" --chdir="$REPO" --output="$ROOT/logs/r012_${STAGE}_%j.out" --error="$ROOT/logs/r012_${STAGE}_%j.out" \
  --export="$EXPORT" "$REPO/$DIR/worker.sbatch" 2>"$ERR")
RC=$?; set -e
if [[ $RC -ne 0 ]]; then
  rmdir "$LOCK"; cat "$ERR" >&2
  if grep -q 'AssocMaxSubmitJobLimit' "$ERR"; then
    echo '[scheduler] one-job association limit is occupied. Current jobs:' >&2
    squeue -u "$USER" -t RUNNING,PENDING -o '%.18i %.12P %.28j %.10T %.10M %.10l %.4D %R' >&2 || true
  fi
  exit $RC
fi
JOB=${JOB%%;*}; [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "$JOB" > "$LOCK/unparsed"; exit 2; }
echo "$JOB" > "$LOCK/job_id"; printf '%s=%s\n' "$KEY" "$JOB" >> "$ROOT/r012_jobs.env"
echo "[submitted] R012 stage=$STAGE job=$JOB mode=$MODE arm=$ARM node=3090node3 gpus=4 code=$HEAD"
echo "tail -n 160 -F $ROOT/logs/r012_${STAGE}_${JOB}.out"
echo "sacct -j $JOB --format=JobID,State,ExitCode,Elapsed"
