#!/usr/bin/env bash
set -euo pipefail
STAGE="${1:-}"
case "$STAGE" in cache|smoke|train|resume|verify) ;; *) echo "Usage: submit.sh cache|smoke|train|resume|verify"; exit 2;; esac
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
DIR=experiments/hierarchical9_v4_scratch_v1
ROOT="${V4TR_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_v4_scratch_v1}"
mkdir -p "$ROOT/logs";ROOT=$(realpath -e "$ROOT");cd "$REPO"
git diff --quiet && git diff --cached --quiet || { echo "[ERROR] save tracked changes before submission"; exit 2; }
HEAD=$(git rev-parse HEAD)
CACHE="$ROOT/training_cache"
verify_formal(){ python3 - "$ROOT/formal/run.json" "$ROOT/formal/final.pt" <<'PY'
import hashlib,json,sys
from pathlib import Path
rp,fp=map(Path,sys.argv[1:]);
if not rp.is_file() or not fp.is_file():raise SystemExit("formal output missing")
r=json.loads(rp.read_text());
if r.get("status")!="COMPLETE" or r.get("successful_updates")!=50000:raise SystemExit(r)
h=hashlib.sha256();
with fp.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
if h.hexdigest()!=r.get("final_sha256"):raise SystemExit("final sha mismatch")
print("[verified] formal COMPLETE",r["final_sha256"])
PY
}
if [[ "$STAGE" == verify ]]; then verify_formal; exit 0; fi
if [[ "$STAGE" != cache ]]; then [[ -s "$CACHE/manifest.json" ]] || { echo "[ERROR] build cache first"; exit 2; }; fi
if [[ "$STAGE" == train ]]; then
  [[ -s "$ROOT/smoke_manifest.json" ]] || { echo "[ERROR] complete smoke first"; exit 2; }
  SMOKE_SHA=$(python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); assert m["status"]=="COMPLETE"; print(m["code_sha"])' "$ROOT/smoke_manifest.json")
  [[ "$HEAD" == "$SMOKE_SHA" ]] || { echo "[ERROR] code changed since smoke: $SMOKE_SHA -> $HEAD"; exit 2; }
  [[ ! -e "$ROOT/formal" || -z "$(ls -A "$ROOT/formal" 2>/dev/null || true)" ]] || { echo "[ERROR] formal exists; use resume if interrupted"; exit 2; }
fi
if [[ "$STAGE" == smoke ]]; then [[ ! -e "$ROOT/smoke" && ! -e "$ROOT/smoke_manifest.json" ]] || { echo "[ERROR] smoke output exists"; exit 2; }; fi
if [[ "$STAGE" == resume ]]; then [[ -s "$ROOT/formal/latest.pt" ]] || { echo "[ERROR] no resumable latest.pt"; exit 2; }; fi
LOCK="$ROOT/.submit_$STAGE"
if [[ -e "$LOCK" ]]; then
  OLD="";[[ -s "$LOCK/job_id" ]]&&OLD=$(cat "$LOCK/job_id");STATE="";[[ "$OLD" =~ ^[0-9]+$ ]]&&STATE=$(sacct -n -X -j "$OLD" --format=State --noheader 2>/dev/null|awk 'NF{print $1;exit}'|sed 's/+.*//')
  case "$STATE" in COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL) mv "$LOCK" "$ROOT/.submit_archive_${STAGE}_${OLD}_$(date +%Y%m%d_%H%M%S)";; *) echo "[ERROR] stage reserved job=$OLD state=$STATE";exit 2;; esac
fi
mkdir "$LOCK"
MODE="$STAGE";[[ "$STAGE" == resume ]]&&MODE=resume
EXPORT="ALL,V4TR_ROOT=$ROOT,V4TR_REPO=$REPO,V4TR_CODE_SHA=$HEAD,V4TR_MODE=$MODE"
NODE="${NODE:-}"
if [[ "$STAGE" == cache ]]; then GRES="gpu:3090:1";CPUS=8;LIMIT="${TIME_LIMIT:-04:00:00}";else GRES="gpu:3090:4";CPUS=16;LIMIT="${TIME_LIMIT:-3-00:00:00}";fi
NODE_ARGS=()
if [[ -n "$NODE" ]]; then NODE_ARGS=(--nodelist="$NODE"); fi
ERR=$(mktemp);trap 'rm -f "$ERR"' EXIT;set +e
JOB=$(sbatch --parsable --partition=GPU "${NODE_ARGS[@]}" --nodes=1 --ntasks=1 --gres="$GRES" --cpus-per-task="$CPUS" --time="$LIMIT" --job-name="h9_v4tr_$STAGE" --chdir="$REPO" --output="$ROOT/logs/${STAGE}_%j.out" --error="$ROOT/logs/${STAGE}_%j.out" --export="$EXPORT" "$REPO/$DIR/worker.sbatch" 2>"$ERR")
RC=$?;set -e
if [[ $RC -ne 0 ]];then rmdir "$LOCK";cat "$ERR" >&2;exit $RC;fi
JOB="${JOB%%;*}";echo "$JOB" >"$LOCK/job_id";STATEFILE="$ROOT/${STAGE}_job_${JOB}.env";printf 'export JOB_ID=%q\nexport STAGE=%q\nexport LOG=%q\n' "$JOB" "$STAGE" "$ROOT/logs/${STAGE}_${JOB}.out" >"$STATEFILE";cp "$STATEFILE" "$ROOT/latest_${STAGE}.env"
echo "[submitted] stage=$STAGE job=$JOB node=${NODE:-AUTO} gres=$GRES code=$HEAD"
echo "[state] $STATEFILE"
