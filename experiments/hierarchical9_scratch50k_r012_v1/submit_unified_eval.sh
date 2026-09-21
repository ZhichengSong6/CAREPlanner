#!/usr/bin/env bash
set -euo pipefail
STAGE="${1:-eval}"
if [[ "$STAGE" != eval && "$STAGE" != pack ]]; then echo 'Usage: submit_unified_eval.sh eval|pack'; exit 2; fi
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
DIR=experiments/hierarchical9_scratch50k_r012_v1
ROOT="${R012_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_scratch50k_r012_v1}"
REF="${E012_REFERENCE_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_cal_pilot_20260910_183735_099nu5}"
mkdir -p "$ROOT/logs"
ROOT=$(realpath -e "$ROOT")
REF=$(realpath -e "$REF")
cd "$REPO"
git diff --quiet && git diff --cached --quiet || { echo '[ERROR] save tracked changes before evaluation'; exit 2; }

if [[ "$STAGE" == pack ]]; then
  python3 "$DIR/pack_unified_reports.py" --r012-root "$ROOT" --reference-root "$REF"
  exit 0
fi

# Formal R0/R1/R2 integrity + same-stream preflight.
bash "$DIR/submit.sh" verify

# E0/E1 and R checkpoint existence is also revalidated by evaluate_unified.py.
OUT="$ROOT/evaluation_unified_dev"
if [[ -e "$OUT" ]]; then
  STATUS=""
  [[ -s "$OUT/manifest.json" ]] && STATUS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$OUT/manifest.json" 2>/dev/null || true)
  if [[ "$STATUS" == COMPLETE ]]; then
    echo "[ERROR] completed unified evaluation already exists: $OUT"; exit 2
  fi
  ARCHIVE="$ROOT/.evaluation_unified_dev_failed_$(date +%Y%m%d_%H%M%S)"
  mv "$OUT" "$ARCHIVE"
  echo "[archive] previous incomplete evaluation -> $ARCHIVE"
fi

HEAD=$(git rev-parse HEAD)
LOCK="$ROOT/.submit_unified_eval"
if [[ -e "$LOCK" ]]; then
  OLD=""; [[ -s "$LOCK/job_id" ]] && OLD=$(cat "$LOCK/job_id")
  STATE=""; [[ "$OLD" =~ ^[0-9]+$ ]] && STATE=$(sacct -n -X -j "$OLD" --format=State --noheader 2>/dev/null | awk 'NF{print $1;exit}' | sed 's/+.*//')
  case "$STATE" in
    COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL)
      mv "$LOCK" "$ROOT/.submit_unified_eval_archive_${OLD}_$(date +%Y%m%d_%H%M%S)";;
    *) echo "[ERROR] unified eval already reserved${OLD:+ job=$OLD}${STATE:+ state=$STATE}"; exit 2;;
  esac
fi
mkdir "$LOCK"
EXPORT="ALL,R012_ROOT=$ROOT,E012_REFERENCE_ROOT=$REF,R012_CODE_REPO=$REPO,R012_CODE_SHA=$HEAD"
ERR=$(mktemp); trap 'rm -f "$ERR"' EXIT
set +e
JOB=$(sbatch --parsable --partition=GPU --nodelist=3090node3 --nodes=1 --ntasks=1 --gres=gpu:3090:1 --cpus-per-task=8 --time=03:00:00 \
  --job-name=h9_r012_eval --chdir="$REPO" --output="$ROOT/logs/r012_unified_eval_%j.out" --error="$ROOT/logs/r012_unified_eval_%j.out" \
  --export="$EXPORT" "$REPO/$DIR/eval_worker.sbatch" 2>"$ERR")
RC=$?; set -e
if [[ $RC -ne 0 ]]; then
  rmdir "$LOCK"; cat "$ERR" >&2
  if grep -q 'AssocMaxSubmitJobLimit' "$ERR"; then
    echo '[scheduler] one-job association limit is occupied. Current jobs:' >&2
    squeue -u "$USER" -t RUNNING,PENDING -o '%.18i %.12P %.28j %.10T %.10M %.10l %.4D %R' >&2 || true
  fi
  exit $RC
fi
JOB=${JOB%%;*}
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "$JOB" > "$LOCK/unparsed"; exit 2; }
echo "$JOB" > "$LOCK/job_id"
printf 'R012_UNIFIED_EVAL_JOB=%s\n' "$JOB" >> "$ROOT/r012_jobs.env"
echo "[submitted] unified eval job=$JOB node=3090node3 gpu=1 code=$HEAD"
echo "tail -n 160 -F $ROOT/logs/r012_unified_eval_${JOB}.out"
echo "sacct -j $JOB --format=JobID,State,ExitCode,Elapsed"
