#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-status}"
if [[ "$MODE" == _worker ]]; then
  : "${IE_CODE:?}" "${IE_JOB:?}" "${IE_ROOT:?}" "${VIS_PYTHON:?}"
  export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
  exec 8>"$IE_ROOT/.worker.lock"
  flock -n 8 || { echo '[STOP] another worker owns this experiment'; exit 2; }
  "$VIS_PYTHON" -u "$IE_CODE/evaluate.py" worker --job "$IE_JOB"
  exit 0
fi
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
BASE=/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b
ROOT="${IE_ROOT:-$BASE/h9_viscdf_iteration_efficiency_v1}"
K2="${K2_ROOT:-$BASE/h9_solver_goal_keep_v2}"
OUT="$ROOT/formal"
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export IE_TEST_REPO="$REPO"
[[ -x "$VIS_PYTHON" ]] || { echo '[STOP] Python not executable'; exit 2; }
if [[ "$MODE" == test ]]; then
  "$VIS_PYTHON" -m unittest discover -s "$HERE" -p 'test_*.py' -v
  exit 0
fi
if [[ "$MODE" == status || "$MODE" == log || "$MODE" == ranks || "$MODE" == summary || "$MODE" == verify || "$MODE" == pack ]]; then
  [[ -f "$ROOT/source.env" ]] || { echo '[not prepared]'; exit 0; }
  source "$ROOT/source.env"
  case "$MODE" in
    status|log)
      [[ -f "$ROOT/latest.env" ]] || { echo '[not submitted]'; exit 0; }
      source "$ROOT/latest.env"
      if [[ "$MODE" == status ]]; then
        squeue -j "$JOB" -o '%.18i %.12T %.12M %.30R' 2>/dev/null || true
        sacct -X -j "$JOB" --format=JobID,State,ExitCode,Elapsed,NodeList || true
        echo "[log] $LOG"
      elif [[ -f "$LOG" ]]; then tail -n 120 "$LOG"; else echo '[log not created]'; fi
      ;;
    ranks)
      for RANK in 0 1 2 3; do
        F="$(find "$OUT/logs" -maxdepth 1 -type f -name "rank${RANK}_*.log" 2>/dev/null | sort | tail -n 1 || true)"
        [[ -z "$F" ]] || { echo "=== $F ==="; tail -n 25 "$F"; }
      done
      ;;
    summary)
      if [[ -f "$OUT/report/complete.json" ]]; then cat "$OUT/report/summary.md"; else echo '[report not complete; check status/log/ranks]'; fi
      ;;
    verify)
      "$VIS_PYTHON" "$CODE/evaluate.py" verify --job "$OUT/job.json"
      ;;
    pack)
      "$VIS_PYTHON" "$CODE/evaluate.py" verify --job "$OUT/job.json"
      ARCH="$ROOT/review_$(date +%Y%m%d_%H%M%S)_${BASHPID}.tar.gz"
      "$VIS_PYTHON" - "$ROOT" "$ARCH" <<'PY'
import sys,tarfile
from pathlib import Path
root,archive=map(Path,sys.argv[1:]);assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
    for f in sorted(root.rglob('*')):
        if f.is_file() and not f.is_symlink() and f.suffix in ('.json','.jsonl','.csv','.svg','.md','.py','.sh','.log','.out','.env'):
            tar.add(f,arcname=str(f.relative_to(root)),recursive=False)
with tarfile.open(archive,'r:gz') as tar:
    for m in tar:
        if m.isfile():
            f=tar.extractfile(m)
            for _ in iter(lambda:f.read(1<<20),b''):pass
print('[pack]',archive)
PY
      sha256sum "$ARCH"
      ;;
  esac
  exit 0
fi
[[ "$MODE" == submit || "$MODE" == resume ]] || { echo 'Usage: run.sh submit|resume|status|log|ranks|summary|verify|pack|test'; exit 2; }
[[ "$(git -C "$REPO" branch --show-current)" == mainline-b/h9-scratch50k-r012-v1 ]] || { echo '[STOP] wrong branch'; exit 2; }
git -C "$REPO" diff --quiet && git -C "$REPO" diff --cached --quiet || { echo '[STOP] tracked modifications; no reset/stash'; exit 2; }
mkdir -p "$ROOT/logs" "$ROOT/sources"
exec 9>"$ROOT/.submit.lock"
flock -n 9 || { echo '[STOP] another submitter'; exit 2; }
if [[ -f "$ROOT/latest.env" ]]; then
  source "$ROOT/latest.env"
  [[ -z "$(squeue -h -j "$JOB" -o '%T' 2>/dev/null || true)" ]] || { echo "[STOP] job $JOB active"; exit 2; }
  ST="$(sacct -X -n -j "$JOB" --format=State 2>/dev/null | awk 'NF{print $1;exit}' || true)"
  case "$ST" in COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL) ;; *) echo '[STOP] unresolved previous Slurm state'; exit 2;; esac
fi
if [[ "$MODE" == submit ]]; then
  [[ ! -e "$OUT" ]] || { echo '[STOP] existing experiment; never overwrite, use resume if unfinished'; exit 2; }
  "$VIS_PYTHON" -m unittest discover -s "$HERE" -p 'test_*.py' -v
  ID="$("$VIS_PYTHON" - "$HERE" <<'PY'
import hashlib,json,sys
from pathlib import Path
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(sys.argv[1]).iterdir() if p.suffix in ('.py','.sh','.json')}
print(hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest())
PY
)"
  CODE="$ROOT/sources/$ID"
  if [[ ! -d "$CODE" ]]; then
    TMP="$(mktemp -d "$ROOT/sources/.building.XXXXXX")"
    cp "$HERE"/*.py "$HERE"/*.sh "$HERE"/*.json "$HERE"/*.md "$TMP/"
    mv "$TMP" "$CODE"
  fi
  "$VIS_PYTHON" - "$HERE" "$CODE" <<'PY'
import hashlib,sys
from pathlib import Path
def digest(p):return {f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(p).iterdir() if f.suffix in ('.py','.sh','.json')}
if digest(sys.argv[1])!=digest(sys.argv[2]):raise SystemExit('[STOP] source snapshot corrupt')
PY
  printf 'export CODE=%q\n' "$CODE" > "$ROOT/source.env"
  "$VIS_PYTHON" "$CODE/evaluate.py" prepare --k2root "$K2" --out "$OUT"
else
  [[ -f "$OUT/job.json" && -f "$ROOT/source.env" ]] || { echo '[STOP] no resumable experiment'; exit 2; }
  source "$ROOT/source.env"
  [[ ! -f "$OUT/report/complete.json" ]] || { echo '[STOP] already complete; use summary'; exit 2; }
fi
"$VIS_PYTHON" "$CODE/evaluate.py" check --job "$OUT/job.json"
export IE_ROOT="$ROOT" IE_CODE="$CODE" IE_JOB="$OUT/job.json"
NODE_ARGS=(--exclude=3090node1); [[ -z "${NODE:-}" ]] || NODE_ARGS=(--nodelist="$NODE")
RAW="$(sbatch --parsable --partition=GPU --nodes=1 --ntasks=1 --gres=gpu:3090:4 --exclusive \
  --cpus-per-task=16 --time="${TIME_LIMIT:-12:00:00}" "${NODE_ARGS[@]}" --chdir="$REPO" --export=ALL \
  --job-name=viscdf_iter --output="$ROOT/logs/eval_%j.out" --error="$ROOT/logs/eval_%j.out" "$CODE/run.sh" _worker)"
JOB="${RAW%%;*}"; [[ "$JOB" =~ ^[0-9]+$ ]] || { echo '[STOP] invalid sbatch response'; exit 2; }
LOG="$ROOT/logs/eval_${JOB}.out"
printf 'export JOB=%q\nexport LOG=%q\n' "$JOB" "$LOG" > "$ROOT/job_${JOB}.env"
cp "$ROOT/job_${JOB}.env" "$ROOT/latest.env"
echo "[submitted] job=$JOB gpus=4 mode=ITERATION_EVALUATION_ONLY"
echo "[out] $OUT"; echo "[log] $LOG"
