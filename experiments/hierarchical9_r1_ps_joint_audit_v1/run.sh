#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${JEA_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_ps_joint_audit_v1}"
MODE="${1:-status}"
PY="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
STATE="${JEA_STATE:-$ROOT/latest.env}"
if [[ "$MODE" != submit ]]; then
  [[ -f "$STATE" ]] || { echo "[STOP] no submitted job state: $STATE"; exit 2; }
  source "$STATE"
  case "$MODE" in
    status)
      squeue -j "$JOB_ID" -o '%.18i %.12T %.12M %.30R' 2>/dev/null || true
      sacct -X -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed,NodeList || true
      echo "[out] $OUT"; echo "[log] $LOG";;
    log) [[ -f "$LOG" ]] && tail -n 120 "$LOG" || echo '[log not created]';;
    follow) echo 'Ctrl+C stops viewing only, not the job.'; tail --retry -n 100 -F "$LOG";;
    ranks)
      for r in 0 1 2 3; do
        echo "========== rank $r =========="
        if [[ -f "$OUT/ranks/rank${r}.log" ]]; then tail -n 25 "$OUT/ranks/rank${r}.log"; else echo '[not started]'; fi
      done;;
    summary)
      if [[ -f "$OUT/summary.md" ]]; then cat "$OUT/summary.md"; else echo '[summary not generated; use log/ranks, do not resubmit]'; fi;;
    pack)
      [[ -f "$OUT/complete.json" && -f "$OUT/report.json" ]] || { echo '[STOP] complete evaluation not available'; exit 2; }
      ARCHIVE="$ROOT/review_${JOB_ID}_$(date +%Y%m%d_%H%M%S).tar.gz"
      [[ ! -e "$ARCHIVE" ]] || { echo '[STOP] archive exists'; exit 2; }
      # OUT contains only audit products and a source snapshot, never model/cache files.
      tar -czf "$ARCHIVE" -C "$OUT" .
      tar -tzf "$ARCHIVE" >/dev/null
      sha256sum "$ARCHIVE"; echo "[review archive] $ARCHIVE";;
    *) echo 'usage: run.sh submit|status|log|follow|ranks|summary|pack'; exit 2;;
  esac
  exit 0
fi
[[ -x "$PY" ]] || { echo "[STOP] Python missing: $PY"; exit 2; }
command -v flock >/dev/null || { echo '[STOP] flock unavailable'; exit 2; }
command -v sbatch >/dev/null || { echo '[STOP] sbatch unavailable'; exit 2; }
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
BRANCH='mainline-b/h9-scratch50k-r012-v1'
[[ "$(git -C "$REPO" branch --show-current)" == "$BRANCH" ]] || { echo '[STOP] wrong branch'; exit 2; }
git -C "$REPO" diff --quiet && git -C "$REPO" diff --cached --quiet || { echo '[STOP] save tracked changes first; no automatic stash/reset'; exit 2; }
mkdir -p "$ROOT/logs"
ROOT="$(cd "$ROOT" && pwd)"
# One submission owner; a second terminal cannot race this block.
exec 9>"$ROOT/.submit.lock"
flock -n 9 || { echo '[STOP] another submitter is active'; exit 2; }
if [[ -f "$STATE" ]]; then
  PREV_ID="$(sed -n 's/^export JOB_ID=//p' "$STATE")"
  if [[ "$PREV_ID" =~ ^[0-9]+$ ]]; then
    ACTIVE="$(squeue -h -j "$PREV_ID" -o '%T' 2>/dev/null || true)"
    [[ -z "$ACTIVE" ]] || { echo "[STOP] job $PREV_ID is still $ACTIVE; view it instead"; exit 2; }
    ACCOUNTING="$(sacct -X -n -j "$PREV_ID" --format=State 2>/dev/null | awk 'NF{print $1;exit}' || true)"
    case "$ACCOUNTING" in
      COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL) ;;
      *) echo "[STOP] previous job $PREV_ID state unresolved ($ACCOUNTING); not risking duplicate submission"; exit 2;;
    esac
  fi
fi
R012="${JEA_R012_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_scratch50k_r012_v1}"
PS="${JEA_PS_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_paired_slope_scratch_v1}"
CACHE="${JEA_CACHE:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_v4_scratch_v1/training_cache}"
STARTS="${JEA_STARTS:-$R012/evaluation_fresh_holdout_v1/starts.jsonl}"
OUT="$ROOT/run_$(date +%Y%m%d_%H%M%S)_${BASHPID}"
mkdir "$OUT"
mkdir "$OUT/source"
cp -- "$HERE"/*.py "$HERE"/*.json "$HERE"/*.sh "$HERE"/*.md "$OUT/source/"
export PYTHONDONTWRITEBYTECODE=1
export JEA_TEST_REPO="$REPO"
# Compile in memory: no cache/bytecode written into protected source directories.
"$PY" - "$OUT/source" <<'PY'
from pathlib import Path
import sys
for p in Path(sys.argv[1]).glob('*.py'):
    compile(p.read_bytes(), str(p), 'exec')
print('[syntax] Python PASS')
PY
for f in "$OUT/source"/*.sh; do bash -n "$f"; done
"$PY" -m unittest discover -s "$OUT/source" -p 'test_*.py' -v >"$OUT/preflight_tests.log" 2>&1 || {
  cat "$OUT/preflight_tests.log"; echo '[STOP] CPU regression failed before GPU submission'; exit 2;
}
cat "$OUT/preflight_tests.log"
"$PY" -u "$OUT/source/cli.py" preflight --repo "$REPO" --r012-root "$R012" --ps-root "$PS" \
  --cache "$CACHE" --starts "$STARTS" --out "$OUT" | tee "$OUT/preflight.log"
# Prevent accidental repetition of a completed identical scientific/source specification.
SIGNATURE="$("$PY" - "$OUT/job.json" <<'PY'
import hashlib,json,sys
j=json.load(open(sys.argv[1]))
print(hashlib.sha256(json.dumps({k:j[k] for k in ('protocol','package_hashes','source_sha256','models')},sort_keys=True).encode()).hexdigest())
PY
)"
CLAIM="$ROOT/claims/$SIGNATURE"
mkdir -p "$ROOT/claims"
if [[ -f "$CLAIM" ]]; then
  OLD_OUT="$(cat "$CLAIM")"
  [[ ! -f "$OLD_OUT/complete.json" ]] || { echo "[STOP] identical complete evaluation exists: $OLD_OUT"; exit 2; }
fi
NODE_ARGS=(--exclude=3090node1)
[[ -z "${NODE:-}" ]] || NODE_ARGS=(--nodelist="$NODE")
JEA_PY="$PY"; JEA_SOURCE="$OUT/source"; JEA_JOB_JSON="$OUT/job.json"
export JEA_PY JEA_SOURCE JEA_JOB_JSON
# Explicit source path avoids the Slurm spool/BASH_SOURCE bug.
RAW="$(sbatch --parsable --partition=GPU --nodes=1 --ntasks=1 --gres=gpu:3090:4 \
  --cpus-per-task=16 --time="${TIME_LIMIT:-04:00:00}" "${NODE_ARGS[@]}" \
  --chdir="$REPO" --job-name=r1_ps_audit --export=ALL \
  --output="$ROOT/logs/audit_%j.out" --error="$ROOT/logs/audit_%j.out" "$OUT/source/worker.sh")"
JOB_ID="${RAW%%;*}"
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo "[STOP] unparsed sbatch result: $RAW; check squeue, do not blindly repeat"; exit 2; }
LOG="$ROOT/logs/audit_${JOB_ID}.out"
JOBSTATE="$ROOT/job_${JOB_ID}.env"
printf 'export JOB_ID=%q\nexport OUT=%q\nexport LOG=%q\n' "$JOB_ID" "$OUT" "$LOG" >"$JOBSTATE"
cp "$JOBSTATE" "$ROOT/latest.env"
printf '%s\n' "$OUT" >"$CLAIM"
echo "[submitted] job=$JOB_ID; four GPUs; full solver + geometry + gradients; optimizer updates=0"
echo "[out] $OUT"; echo "[log] $LOG"; echo "[state] $JOBSTATE"
