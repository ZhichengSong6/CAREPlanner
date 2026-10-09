#!/usr/bin/env bash
set -euo pipefail
STAGE="${1:-}"
case "$STAGE" in smoke|pilot|train|resume|verify) ;; *) echo "Usage: submit.sh smoke|pilot|train|resume|verify";exit 2;;esac
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
DIR=experiments/hierarchical9_v4_scratch_v2
ROOT="${V4TR2_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_v4_scratch_v2}"
mkdir -p "$ROOT/logs";ROOT=$(realpath -e "$ROOT");cd "$REPO"
git diff --quiet && git diff --cached --quiet || { echo "[ERROR] save tracked changes before submission";exit 2; }
HEAD=$(git rev-parse HEAD)
verify_formal(){ python3 - "$ROOT/formal/run.json" "$ROOT/formal/final.pt" <<'PY'
import hashlib,json,sys
from pathlib import Path
rp,fp=map(Path,sys.argv[1:]);
if not rp.is_file() or not fp.is_file():raise SystemExit("formal output missing")
r=json.loads(rp.read_text())
if r.get("status")!="COMPLETE" or r.get("successful_updates")!=50000:raise SystemExit(r)
h=hashlib.sha256()
with fp.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
if h.hexdigest()!=r.get("final_sha256"):raise SystemExit("final sha mismatch")
print("[verified] v2 formal COMPLETE",r["final_sha256"])
PY
}
if [[ "$STAGE" == verify ]];then verify_formal;exit 0;fi
check_manifest(){ local p="$1";local expect="$2";python3 - "$p" "$HEAD" "$expect" <<'PY'
import json,sys
p,head,n=sys.argv[1],sys.argv[2],int(sys.argv[3]);m=json.load(open(p))
assert m.get("status")=="COMPLETE" and m.get("code_sha")==head and m.get("successful_updates")==n,m
print("[preflight]",p,"COMPLETE code="+head)
PY
}
if [[ "$STAGE" == pilot || "$STAGE" == train ]];then [[ -s "$ROOT/smoke_manifest.json" ]]||{ echo "[ERROR] complete v2 smoke first";exit 2;};check_manifest "$ROOT/smoke_manifest.json" 2;fi
if [[ "$STAGE" == train ]];then [[ -s "$ROOT/pilot_manifest.json" ]]||{ echo "[ERROR] complete v2 pilot first";exit 2;};check_manifest "$ROOT/pilot_manifest.json" 500;[[ ! -e "$ROOT/formal" || -z "$(ls -A "$ROOT/formal" 2>/dev/null||true)" ]]||{ echo "[ERROR] formal exists; use resume if interrupted";exit 2;};fi
if [[ "$STAGE" == smoke ]];then [[ ! -e "$ROOT/smoke" && ! -e "$ROOT/smoke_manifest.json" ]]||{ echo "[ERROR] smoke output exists";exit 2;};fi
if [[ "$STAGE" == pilot ]];then [[ ! -e "$ROOT/pilot" && ! -e "$ROOT/pilot_manifest.json" ]]||{ echo "[ERROR] pilot output exists";exit 2;};fi
if [[ "$STAGE" == resume ]];then [[ -s "$ROOT/formal/latest.pt" ]]||{ echo "[ERROR] no resumable latest.pt";exit 2;};fi
LOCK="$ROOT/.submit_$STAGE"
if [[ -e "$LOCK" ]];then OLD="";[[ -s "$LOCK/job_id" ]]&&OLD=$(cat "$LOCK/job_id");STATE="";[[ "$OLD" =~ ^[0-9]+$ ]]&&STATE=$(sacct -n -X -j "$OLD" --format=State --noheader 2>/dev/null|awk 'NF{print $1;exit}'|sed 's/+.*//');case "$STATE" in COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL) mv "$LOCK" "$ROOT/.submit_archive_${STAGE}_${OLD}_$(date +%Y%m%d_%H%M%S)";;*) echo "[ERROR] stage reserved job=$OLD state=$STATE";exit 2;;esac;fi
mkdir "$LOCK"
EXPORT="ALL,V4TR2_ROOT=$ROOT,V4TR2_REPO=$REPO,V4TR2_CODE_SHA=$HEAD,V4TR2_MODE=$STAGE"
NODE="${NODE:-}";NODE_ARGS=();if [[ -n "$NODE" ]];then NODE_ARGS=(--nodelist="$NODE");else NODE_ARGS=(--exclude=3090node1);fi
LIMIT="${TIME_LIMIT:-3-00:00:00}"
ERR=$(mktemp);trap 'rm -f "$ERR"' EXIT;set +e
JOB=$(sbatch --parsable --partition=GPU "${NODE_ARGS[@]}" --nodes=1 --ntasks=1 --gres=gpu:3090:4 --cpus-per-task=16 --time="$LIMIT" --job-name="h9_v4v2_$STAGE" --chdir="$REPO" --output="$ROOT/logs/${STAGE}_%j.out" --error="$ROOT/logs/${STAGE}_%j.out" --export="$EXPORT" "$REPO/$DIR/worker.sbatch" 2>"$ERR")
RC=$?;set -e
if [[ $RC -ne 0 ]];then rmdir "$LOCK";cat "$ERR" >&2;exit $RC;fi
JOB="${JOB%%;*}";echo "$JOB" >"$LOCK/job_id";STATEFILE="$ROOT/${STAGE}_job_${JOB}.env";printf 'export JOB_ID=%q\nexport STAGE=%q\nexport LOG=%q\n' "$JOB" "$STAGE" "$ROOT/logs/${STAGE}_${JOB}.out" >"$STATEFILE";cp "$STATEFILE" "$ROOT/latest_${STAGE}.env"
echo "[submitted] v2 stage=$STAGE job=$JOB node=${NODE:-AUTO_EXCLUDING_3090node1} code=$HEAD"
