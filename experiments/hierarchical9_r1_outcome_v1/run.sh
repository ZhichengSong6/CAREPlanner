#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
BASE=/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b
ROOT="${OUTCOME_ROOT:-$BASE/h9_r1_outcome_v1}"
PY="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
MODE="${1:-status}"
export O_TEST_REPO="$REPO"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
case "$MODE" in
  status|log|follow)
    [[ -f "$ROOT/latest.env" ]] || { echo '[not submitted]'; exit 0; }
    source "$ROOT/latest.env"
    case "$MODE" in
      status) squeue -j "$JOB_ID" -o '%.18i %.12T %.12M %.35R' 2>/dev/null || true
              sacct -X -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed,NodeList || true
              echo "[log] $LOG";;
      log) [[ ! -f "$LOG" ]] || tail -n 120 "$LOG";;
      follow) echo 'Ctrl+C stops viewing only'; tail --retry -n 120 -F "$LOG";;
    esac;exit 0;;
  summary)
    [[ ! -f "$ROOT/formal/run.json" ]] || cat "$ROOT/formal/run.json"
    [[ ! -f "$ROOT/evaluation/summary.md" ]] || cat "$ROOT/evaluation/summary.md"
    [[ ! -f "$ROOT/formal/metrics.jsonl" ]] || tail -n 2 "$ROOT/formal/metrics.jsonl"
    exit 0;;
  test) "$PY" -m unittest discover -s "$HERE" -p 'test_*.py' -v;exit 0;;
  pack)
    [[ -f "$ROOT/formal/run.json" ]] || { echo '[STOP] no results'; exit 2; }
    ARCHIVE="$ROOT/review_$(date +%Y%m%d_%H%M%S)_${BASHPID}.tar.gz"
    FILES=(job.json formal/run.json formal/train_args.json formal/metrics.jsonl logs)
    [[ ! -d "$ROOT/evaluation" ]] || FILES+=(evaluation)
    tar -czf "$ARCHIVE" -C "$ROOT" "${FILES[@]}"
    tar -tzf "$ARCHIVE" >/dev/null;sha256sum "$ARCHIVE";echo "[review] $ARCHIVE";exit 0;;
  train|resume|eval|verify) ;;
  *) echo 'run.sh train|resume|eval|status|log|follow|summary|verify|pack|test';exit 2;;
esac
[[ -x "$PY" ]] || { echo '[STOP] missing viscdf Python'; exit 2; }
[[ "$(git -C "$REPO" branch --show-current)" == mainline-b/h9-scratch50k-r012-v1 ]] || { echo '[STOP] wrong branch';exit 2; }
git -C "$REPO" diff --quiet && git -C "$REPO" diff --cached --quiet || { echo '[STOP] save tracked changes; no reset/stash';exit 2; }
mkdir -p "$ROOT/logs" "$ROOT/sources"
ROOT="$(realpath -e "$ROOT")"
exec 9>"$ROOT/.submit.lock"
flock -n 9 || { echo '[STOP] another submitter'; exit 2; }
ID="$("$PY" - "$HERE" <<'PY'
from pathlib import Path
import sys,hashlib,json
p=Path(sys.argv[1]);h={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir() if f.suffix in ('.py','.json','.sh')}
print(hashlib.sha256(json.dumps(h,sort_keys=True).encode()).hexdigest())
PY
)"
SOURCE="$ROOT/sources/$ID"
if [[ ! -d "$SOURCE" ]]; then
  TMP="$(mktemp -d "$ROOT/sources/.snapshot.XXXXXX")"
  cp "$HERE"/*.py "$HERE"/*.json "$HERE"/*.sh "$TMP/"
  mv "$TMP" "$SOURCE"
fi
COMMON=(--repo "$REPO" --spec "$ROOT/job.json" --out "$ROOT/formal")
if [[ "$MODE" == verify ]]; then "$PY" "$SOURCE/driver.py" verify "${COMMON[@]}";exit 0;fi
if [[ -f "$ROOT/latest.env" ]]; then
  source "$ROOT/latest.env"
  ACTIVE="$(squeue -h -j "$JOB_ID" -o '%T' 2>/dev/null || true)"
  [[ -z "$ACTIVE" ]] || { echo "[STOP] job $JOB_ID still $ACTIVE";exit 2; }
  STATE="$(sacct -X -n -j "$JOB_ID" --format=State 2>/dev/null | awk 'NF{print $1;exit}' || true)"
  case "$STATE" in COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL) ;;
    *) echo '[STOP] previous job state unresolved';exit 2;;esac
fi
if [[ "$MODE" == train ]]; then
  [[ ! -d "$ROOT/formal" || -z "$(ls -A "$ROOT/formal")" ]] || { echo '[STOP] existing training; no overwrite';exit 2; }
elif [[ "$MODE" == resume ]]; then
  [[ -f "$ROOT/formal/latest.pt" && ! -f "$ROOT/formal/final.pt" ]] || { echo '[STOP] no unfinished checkpoint';exit 2; }
else
  [[ ! -e "$ROOT/evaluation" ]] || { echo '[STOP] evaluation already exists';exit 2; }
fi
"$PY" - "$SOURCE" <<'PY'
from pathlib import Path
import sys
for p in Path(sys.argv[1]).glob('*.py'): compile(p.read_bytes(),str(p),'exec')
print('[syntax] Python PASS')
PY
for f in "$SOURCE"/*.sh;do bash -n "$f";done
"$PY" -m unittest discover -s "$SOURCE" -p 'test_*.py' -v
if [[ "$MODE" == train || "$MODE" == resume ]]; then
  "$PY" "$SOURCE/driver.py" preflight "${COMMON[@]}" \
    --r012 "${OUTCOME_R012:-$BASE/h9_scratch50k_r012_v1}" \
    --cache "${OUTCOME_CACHE:-$BASE/h9_v4_scratch_v1/training_cache}" \
    --pairs "${OUTCOME_PAIRS:-$BASE/h9_paired_slope_scratch_v1/formal/pairs}"
  LIMIT="${TIME_LIMIT:-3-00:00:00}"
else
  "$PY" "$SOURCE/driver.py" verify "${COMMON[@]}";LIMIT="${TIME_LIMIT:-04:00:00}"
fi
NODE_ARGS=(--exclude=3090node1);[[ -z "${NODE:-}" ]] || NODE_ARGS=(--nodelist="$NODE")
export O_REPO="$REPO" O_ROOT="$ROOT" O_SOURCE="$SOURCE" O_MODE="$MODE" O_PY="$PY"
RAW="$(sbatch --parsable --partition=GPU --nodes=1 --ntasks=1 --gres=gpu:3090:4 --cpus-per-task=16 \
 --time="$LIMIT" "${NODE_ARGS[@]}" --job-name="r1_outcome_$MODE" --chdir="$REPO" --export=ALL \
 --output="$ROOT/logs/${MODE}_%j.out" --error="$ROOT/logs/${MODE}_%j.out" "$SOURCE/worker.sh")"
JOB_ID="${RAW%%;*}";[[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo "[STOP] unparsed job: $RAW";exit 2; }
LOG="$ROOT/logs/${MODE}_${JOB_ID}.out"
printf 'export JOB_ID=%q\nexport LOG=%q\n' "$JOB_ID" "$LOG" > "$ROOT/job_${JOB_ID}.env"
cp "$ROOT/job_${JOB_ID}.env" "$ROOT/latest.env"
echo "[submitted] mode=$MODE job=$JOB_ID source=$ID gpus=4"
echo "[out] $ROOT/formal";echo "[log] $LOG"
