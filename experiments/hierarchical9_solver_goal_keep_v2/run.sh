#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-status}"
if [[ "$MODE" == _worker ]]; then
  : "${K2_CODE:?}" "${K2_JOB:?}" "${K2_ROOT:?}" "${K2_ACTION:?}" "${VIS_PYTHON:?}"
  export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
  exec 8>"$K2_ROOT/.worker.lock";flock -n 8 || { echo '[STOP] another worker';exit 2; }
  "$VIS_PYTHON" -u "$K2_CODE/run.py" worker --job "$K2_JOB" --action "$K2_ACTION"
  exit 0
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
BASE=/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b
ROOT="${K2_ROOT:-$BASE/h9_solver_goal_keep_v2}"
SGROOT="${SG_ROOT:-$BASE/h9_r1_solver_goal_v1}"
OUT="$ROOT/formal"
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export K2_TEST_REPO="$REPO"
[[ -x "$VIS_PYTHON" ]] || { echo '[STOP] viscdf Python missing';exit 2; }
if [[ "$MODE" == test ]];then "$VIS_PYTHON" -m unittest discover -s "$HERE" -p 'test_*.py' -v;exit;fi
if [[ "$MODE" == status || "$MODE" == log || "$MODE" == ranks || "$MODE" == summary || "$MODE" == pack || "$MODE" == verify ]];then
  [[ -f "$ROOT/source.env" ]] || { echo '[not prepared]';exit 0; };source "$ROOT/source.env"
  if [[ "$MODE" == status || "$MODE" == log ]];then
    [[ -f "$ROOT/latest.env" ]] || { echo '[not submitted]';exit 0; };source "$ROOT/latest.env"
    if [[ "$MODE" == status ]];then
      squeue -j "$JOB" -o '%.18i %.12T %.12M %.30R' 2>/dev/null || true
      sacct -X -j "$JOB" --format=JobID,State,ExitCode,Elapsed,NodeList || true
      echo "[log] $LOG"
    else if [[ -f "$LOG" ]];then tail -n 140 "$LOG";else echo '[log not created yet]';fi;fi
  elif [[ "$MODE" == ranks ]];then
    for kind in starts mine generate evaluate;do
      for rank in 0 1 2 3;do
        f="$(find "$OUT/logs" -maxdepth 1 -type f -name "${kind}_rank${rank}_*.log" 2>/dev/null | sort | tail -n 1 || true)"
        [[ -z "$f" ]] || { echo "=== $f ===";tail -n 12 "$f"; }
      done
    done
  elif [[ "$MODE" == summary ]];then
    for f in "$OUT/A_continue/run.json" "$OUT/B_keep/run.json" "$OUT/training_complete.json" "$OUT/confirmation/report/summary.md";do
      [[ ! -f "$f" ]] || { echo "=== $f ===";cat "$f"; }
    done
  elif [[ "$MODE" == verify ]];then
    "$VIS_PYTHON" "$CODE/run.py" verify --job "$OUT/job.json"
  else
    ARCH="$ROOT/review_$(date +%Y%m%d_%H%M%S)_${BASHPID}.tar.gz"
    "$VIS_PYTHON" - "$ROOT" "$ARCH" <<'PY'
import sys,tarfile
from pathlib import Path
r=Path(sys.argv[1]);dest=Path(sys.argv[2]);assert not dest.exists()
with tarfile.open(dest,'w:gz') as t:
 for f in sorted(r.rglob('*')):
  if f.is_file() and not f.is_symlink() and f.suffix in ('.json','.jsonl','.md','.log','.out','.env','.py','.sh'):
   t.add(f,arcname=str(f.relative_to(r)),recursive=False)
with tarfile.open(dest,'r:gz') as t:
 for f in t:
  if f.isfile():
   x=t.extractfile(f)
   for _ in iter(lambda:x.read(1<<20),b''):pass
print('[pack]',dest)
PY
    sha256sum "$ARCH"
  fi;exit 0
fi
[[ "$MODE" == train || "$MODE" == resume || "$MODE" == eval ]] || { echo 'Usage: run.sh train|resume|eval|status|log|ranks|summary|verify|pack|test';exit 2; }
[[ "$(git -C "$REPO" branch --show-current)" == mainline-b/h9-scratch50k-r012-v1 ]] || { echo '[STOP] wrong branch';exit 2; }
git -C "$REPO" diff --quiet && git -C "$REPO" diff --cached --quiet || { echo '[STOP] tracked modifications; no reset/stash';exit 2; }
mkdir -p "$ROOT/logs" "$ROOT/sources"
exec 9>"$ROOT/.submit.lock";flock -n 9 || { echo '[STOP] another submitter';exit 2; }
if [[ -f "$ROOT/latest.env" ]];then
  source "$ROOT/latest.env"
  [[ -z "$(squeue -h -j "$JOB" -o '%T' 2>/dev/null || true)" ]] || { echo "[STOP] job $JOB active";exit 2; }
  ST="$(sacct -X -n -j "$JOB" --format=State 2>/dev/null | awk 'NF{print $1;exit}' || true)"
  case "$ST" in COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL) ;; *) echo '[STOP] previous Slurm state unresolved';exit 2;;esac
fi
if [[ "$MODE" == train ]];then
  [[ ! -e "$OUT" ]] || { echo '[STOP] existing attempt; use resume, never overwrite';exit 2; }
  "$VIS_PYTHON" -m unittest discover -s "$HERE" -p 'test_*.py' -v
  ID="$( "$VIS_PYTHON" - "$HERE" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);m={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir() if f.suffix in ('.py','.sh','.json')}
print(hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest())
PY
)"
  CODE="$ROOT/sources/$ID"
  if [[ ! -d "$CODE" ]];then
    TMP="$(mktemp -d "$ROOT/sources/.building.XXXXXX")";cp "$HERE"/*.py "$HERE"/*.sh "$HERE"/*.json "$HERE"/*.md "$TMP/";mv "$TMP" "$CODE"
  fi
  "$VIS_PYTHON" - "$HERE" "$CODE" <<'PY'
import hashlib,sys
from pathlib import Path
def digest(root):
 return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(root).iterdir() if p.suffix in ('.py','.sh','.json')}
if digest(sys.argv[1]) != digest(sys.argv[2]):raise SystemExit('[STOP] source snapshot corrupted')
PY
  printf 'export CODE=%q\n' "$CODE" > "$ROOT/source.env"
  "$VIS_PYTHON" "$CODE/run.py" prepare --repo "$REPO" --sgroot "$SGROOT" --out "$OUT"
else
  [[ -f "$OUT/job.json" && -f "$ROOT/source.env" ]] || { echo '[STOP] no resumable experiment';exit 2; };source "$ROOT/source.env"
  if [[ "$MODE" == eval ]];then
    "$VIS_PYTHON" "$CODE/run.py" verify --job "$OUT/job.json"
    [[ ! -f "$OUT/confirmation/report/complete.json" ]] || { echo '[STOP] already evaluated';exit 2; }
  else [[ ! -f "$OUT/training_complete.json" ]] || { echo '[STOP] training already complete; use eval';exit 2; };fi
fi
export K2_CODE="$CODE" K2_JOB="$OUT/job.json" K2_ROOT="$ROOT" K2_ACTION="$MODE"
NODE_ARGS=(--exclude=3090node1);[[ -z "${NODE:-}" ]] || NODE_ARGS=(--nodelist="$NODE")
LIMIT=3-00:00:00;[[ "$MODE" != eval ]] || LIMIT=12:00:00
RAW="$(sbatch --parsable --partition=GPU --nodes=1 --ntasks=1 --gres=gpu:3090:4 --cpus-per-task=16 \
 --time="${TIME_LIMIT:-$LIMIT}" "${NODE_ARGS[@]}" --chdir="$REPO" --export=ALL --job-name="sg_keep_$MODE" \
 --output="$ROOT/logs/${MODE}_%j.out" --error="$ROOT/logs/${MODE}_%j.out" "$CODE/run.sh" _worker)"
JOB="${RAW%%;*}";[[ "$JOB" =~ ^[0-9]+$ ]] || { echo '[STOP] unknown sbatch response';exit 2; }
LOG="$ROOT/logs/${MODE}_${JOB}.out"
printf 'export JOB=%q\nexport LOG=%q\n' "$JOB" "$LOG" > "$ROOT/job_${JOB}.env";cp "$ROOT/job_${JOB}.env" "$ROOT/latest.env"
echo "[submitted] job=$JOB mode=$MODE gpus=4";echo "[out] $OUT";echo "[log] $LOG"
