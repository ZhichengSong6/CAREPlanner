#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-status}"
if [[ "$MODE" == _worker ]]; then
  : "${OG_CODE:?}" "${OG_JOB:?}" "${OG_ROOT:?}" "${VIS_PYTHON:?}"
  export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  exec 8>"$OG_ROOT/.worker.lock"
  flock -n 8 || { echo '[STOP] another worker owns this evaluation'; exit 2; }
  exec "$VIS_PYTHON" -u "$OG_CODE/run.py" worker --job "$OG_JOB"
fi
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
BASE='/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b'
ROOT="${OG_ROOT:-$BASE/h9_solver_goal_offgrid_v1}"
SGROOT="${SG_ROOT:-$BASE/h9_r1_solver_goal_v1}"
CACHE="$BASE/h9_v4_scratch_v1/training_cache"
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT="$ROOT/formal"
if [[ "$MODE" == status || "$MODE" == log || "$MODE" == ranks || "$MODE" == summary || "$MODE" == pack || "$MODE" == verify ]]; then
  if [[ "$MODE" == summary ]]; then
    [[ ! -f "$OUT/summary.md" ]] || cat "$OUT/summary.md"
    [[ -f "$OUT/complete.json" ]] || echo '[not completed: no final report]'
    exit 0
  fi
  if [[ "$MODE" == pack || "$MODE" == verify ]]; then
    [[ -f "$OUT/complete.json" && -f "$ROOT/latest.env" ]] || { echo '[STOP] complete results not available'; exit 2; }
    source "$ROOT/latest.env"
    "$VIS_PYTHON" "$CODE/run.py" verify --job "$OUT/job.json"
    [[ "$MODE" != verify ]] || exit 0
    ARCHIVE="$ROOT/review_$(date +%Y%m%d_%H%M%S)_${BASHPID}.tar.gz"
    [[ ! -e "$ARCHIVE" ]] || { echo '[STOP] archive exists'; exit 2; }
    # One creation operation. Never append to an already compressed tar archive.
    "$VIS_PYTHON" - "$OUT" "$ROOT" "$CODE" "$ARCHIVE" <<'PY'
from pathlib import Path
import sys,tarfile
out,root,code,dest=map(Path,sys.argv[1:])
with tarfile.open(dest,'x:gz') as archive:
    for folder,prefix in ((out,'results'),(root/'logs','slurm'),(code,'source')):
        for path in sorted(folder.rglob('*')):
            rel=path.relative_to(folder)
            if not any(part.startswith('.') for part in rel.parts) and path.is_file() and not path.is_symlink():
                if path.suffix not in ('.pt','.npy','.npz'):
                    archive.add(path,arcname=str(Path(prefix)/rel),recursive=False)
with tarfile.open(dest,'r:gz') as archive:
    for member in archive:
        if member.isfile():
            file=archive.extractfile(member)
            for _ in iter(lambda:file.read(8<<20),b''):pass
print('[pack] complete archive read/CRC check passed')
PY
    sha256sum "$ARCHIVE";echo "[review] $ARCHIVE";exit 0
  fi
  [[ -f "$ROOT/latest.env" ]] || { echo '[not submitted]'; exit 0; }
  source "$ROOT/latest.env"
  case "$MODE" in
    status) squeue -j "$JOB" -o '%.18i %.12T %.12M %.35R' 2>/dev/null || true
      sacct -X -j "$JOB" --format=JobID,State,ExitCode,Elapsed,NodeList || true
      echo "[out] $OUT [log] $LOG";;
    log) [[ ! -f "$LOG" ]] || tail -n 120 "$LOG";;
    ranks) for stage in generate evaluate; do
      for rank in 0 1 2 3; do
        echo "========== $stage rank$rank =========="
        shopt -s nullglob;files=("$OUT/logs/${stage}_rank${rank}_"*.log)
        if (( ${#files[@]} )); then
          LAST="$(ls -1t -- "${files[@]}" | head -n 1)";tail -n 30 "$LAST"
        else echo '[not started]';fi
      done
    done;;
  esac
  exit 0
fi
[[ "$MODE" == test || "$MODE" == submit || "$MODE" == resume ]] || { echo 'Usage: run.sh submit|resume|test|status|log|ranks|summary|verify|pack';exit 2; }
[[ -x "$VIS_PYTHON" ]] || { echo '[STOP] viscdf Python unavailable';exit 2; }
export OFFGRID_TEST_REPO="$REPO"
"$VIS_PYTHON" -m unittest discover -s "$HERE" -p 'test_*.py' -v
[[ "$MODE" != test ]] || exit 0
[[ "$(git -C "$REPO" branch --show-current)" == mainline-b/h9-scratch50k-r012-v1 ]] || { echo '[STOP] wrong branch';exit 2; }
git -C "$REPO" diff --quiet && git -C "$REPO" diff --cached --quiet || { echo '[STOP] tracked source modified; no reset/stash';exit 2; }
command -v sbatch >/dev/null;command -v flock >/dev/null
mkdir -p "$ROOT/logs" "$ROOT/sources"
exec 9>"$ROOT/.submit.lock";flock -n 9 || { echo '[STOP] concurrent submitter';exit 2; }
if [[ -f "$ROOT/latest.env" ]]; then
  source "$ROOT/latest.env"
  ACTIVE="$(squeue -h -j "$JOB" -o '%T' 2>/dev/null || true)"
  [[ -z "$ACTIVE" ]] || { echo "[STOP] existing job $JOB: $ACTIVE";exit 2; }
  STATE="$(sacct -X -n -j "$JOB" --format=State 2>/dev/null | awk 'NF{print $1;exit}' || true)"
  case "$STATE" in COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL) ;;
    *) echo "[STOP] previous job state unresolved: $STATE";exit 2;;esac
fi
identity() {
  "$VIS_PYTHON" - "$1" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);d={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir() if f.is_file() and f.suffix in ('.py','.sh','.json','.md')}
print(hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest())
PY
}
if [[ "$MODE" == submit ]]; then
  [[ ! -e "$OUT" ]] || { echo '[STOP] existing attempt; use resume only after a failed/interrupted job, never resample';exit 2; }
  ID="$(identity "$HERE")";CODE="$ROOT/sources/$ID"
  if [[ ! -d "$CODE" ]];then
    TMP="$(mktemp -d "$ROOT/sources/.building.XXXXXX")"
    cp "$HERE"/*.py "$HERE"/*.sh "$HERE"/*.json "$HERE"/*.md "$TMP/"
    mv "$TMP" "$CODE"
  fi
  [[ "$(identity "$CODE")" == "$ID" ]] || { echo '[STOP] source snapshot corrupted';exit 2; }
  "$VIS_PYTHON" -u "$CODE/run.py" prepare --repo "$REPO" --sgroot "$SGROOT" --cache "$CACHE" --out "$OUT"
else
  [[ -f "$ROOT/latest.env" && -f "$OUT/preflight_complete.json" && ! -f "$OUT/complete.json" ]] || {
    echo '[STOP] no resumable evaluation or already complete';exit 2; }
  source "$ROOT/latest.env"
fi
export OG_CODE="$CODE" OG_JOB="$OUT/job.json" OG_ROOT="$ROOT"
NODE_ARGS=(--exclude=3090node1)
[[ -z "${NODE:-}" ]] || NODE_ARGS=(--nodelist="$NODE")
RAW="$(sbatch --parsable --partition=GPU --nodes=1 --ntasks=1 --gres=gpu:3090:4 --cpus-per-task=16 \
  --time="${TIME_LIMIT:-12:00:00}" "${NODE_ARGS[@]}" --chdir="$REPO" --export=ALL --job-name=sg_offgrid \
  --output="$ROOT/logs/eval_%j.out" --error="$ROOT/logs/eval_%j.out" "$CODE/run.sh" _worker)"
JOB="${RAW%%;*}";[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "[STOP] unparsed sbatch: $RAW";exit 2; }
LOG="$ROOT/logs/eval_${JOB}.out"
printf 'export JOB=%q\nexport LOG=%q\nexport CODE=%q\n' "$JOB" "$LOG" "$CODE" > "$ROOT/job_${JOB}.env"
cp "$ROOT/job_${JOB}.env" "$ROOT/latest.env"
echo "[submitted] job=$JOB gpus=4 mode=OFFGRID_EVALUATION_ONLY"
echo "[out] $OUT";echo "[log] $LOG"
