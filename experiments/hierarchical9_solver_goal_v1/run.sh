#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-status}"
if [[ "$MODE" == _worker ]]; then
  : "${SG_CODE:?}" "${SG_JOB:?}" "${VIS_PYTHON:?}" "${SG_ACTION:?}" "${SG_ROOT:?}"
  export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
  exec 8>"$SG_ROOT/.worker.lock"; flock -n 8 || { echo '[STOP] concurrent worker'; exit 2; }
  "$VIS_PYTHON" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==4; print("[GPU]", [torch.cuda.get_device_name(i) for i in range(4)])'
  if [[ "$SG_ACTION" == eval ]]; then
    mkdir "$SG_ROOT/eval"
    IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
    PIDS=(); trap 'for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done' EXIT
    for r in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES="${GPUS[$r]}" "$VIS_PYTHON" -u "$SG_CODE/evaluate.py" --job "$SG_JOB" --out "$SG_ROOT/eval" --rank "$r" > "$SG_ROOT/eval/rank$r.log" 2>&1 & PIDS+=("$!")
    done
    for p in "${PIDS[@]}"; do wait "$p"; done
    PIDS=()
    "$VIS_PYTHON" "$SG_CODE/evaluate.py" --job "$SG_JOB" --out "$SG_ROOT/eval" --merge
  else
    EXTRA=(); [[ "$SG_ACTION" != resume ]] || EXTRA=(--resume)
    "$VIS_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 --max_restarts=0 \
      "$SG_CODE/train.py" train --job "$SG_JOB" "${EXTRA[@]}"
  fi
  exit 0
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
BASE="/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b"
ROOT="${SG_ROOT:-$BASE/h9_r1_solver_goal_v1}"
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
if [[ "$MODE" == status || "$MODE" == log || "$MODE" == summary || "$MODE" == pack ]]; then
  [[ -f "$ROOT/latest.env" ]] || { echo '[not submitted]'; exit 0; }
  source "$ROOT/latest.env"
  case "$MODE" in
    status) squeue -j "$JOB" -o '%.18i %.12T %.12M %.30R' 2>/dev/null || true; sacct -X -j "$JOB" --format=JobID,State,ExitCode,Elapsed,NodeList || true; echo "[log] $LOG";;
    log) if [[ -f "$LOG" ]]; then tail -n 120 "$LOG"; else echo '[log not created]'; fi;;
    summary) for f in "$ROOT/formal/solver_goal_complete.json" "$ROOT/eval/summary.md"; do [[ ! -f "$f" ]] || cat "$f"; done;;
    pack)
      ARCH="$ROOT/review_$(date +%Y%m%d_%H%M%S)_${BASHPID}.tar.gz"
      "$VIS_PYTHON" - "$ROOT" "$ARCH" <<'PY'
import sys,tarfile
from pathlib import Path
r=Path(sys.argv[1]); dest=Path(sys.argv[2])
if dest.exists():raise SystemExit('archive exists')
with tarfile.open(dest,'w:gz') as t:
 for f in sorted(r.rglob('*')):
  if f.is_file() and not f.is_symlink() and f.suffix in ('.json','.jsonl','.md','.log','.out','.env'):
   t.add(f,arcname=str(f.relative_to(r)),recursive=False)
print(dest)
PY
      sha256sum "$ARCH";;
  esac
  exit 0
fi
[[ "$MODE" == train || "$MODE" == resume || "$MODE" == eval || "$MODE" == test ]] || { echo 'Usage: run.sh train|resume|eval|test|status|log|summary|pack'; exit 2; }
[[ -x "$VIS_PYTHON" ]] || { echo '[STOP] missing Python'; exit 2; }
export SG_TEST_REPO="$REPO"
if [[ "$MODE" == test ]]; then "$VIS_PYTHON" -m unittest discover -s "$HERE" -p 'test_*.py' -v; exit; fi
[[ "$(git -C "$REPO" branch --show-current)" == mainline-b/h9-scratch50k-r012-v1 ]] || { echo '[STOP] wrong branch'; exit 2; }
git -C "$REPO" diff --quiet && git -C "$REPO" diff --cached --quiet || { echo '[STOP] tracked changes; no reset/stash'; exit 2; }
mkdir -p "$ROOT/logs"
exec 9>"$ROOT/.submit.lock"; flock -n 9 || { echo '[STOP] another submitter'; exit 2; }
if [[ -f "$ROOT/latest.env" ]]; then
  source "$ROOT/latest.env"
  ACTIVE="$(squeue -h -j "$JOB" -o '%T' 2>/dev/null || true)"
  [[ -z "$ACTIVE" ]] || { echo "[STOP] job $JOB still $ACTIVE"; exit 2; }
  ST="$(sacct -X -n -j "$JOB" --format=State 2>/dev/null | awk 'NF{print $1;exit}' || true)"
  case "$ST" in COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL) ;; *) echo '[STOP] previous job state unresolved'; exit 2;; esac
fi
JOBFILE="$ROOT/job.json"; CODE="$ROOT/source"
if [[ "$MODE" == train ]]; then
  [[ ! -e "$ROOT/formal" ]] || { echo '[STOP] existing formal output; never overwrite'; exit 2; }
  if [[ -d "$CODE" ]]; then
    diff -qr "$HERE" "$CODE" >/dev/null || { echo '[STOP] earlier source differs; preserve existing attempt'; exit 2; }
  else cp -a "$HERE" "$CODE"; fi
  "$VIS_PYTHON" -m unittest discover -s "$CODE" -p 'test_*.py' -v
  "$VIS_PYTHON" "$CODE/train.py" prepare --repo "$REPO" --job "$JOBFILE" \
    --r012 "$BASE/h9_scratch50k_r012_v1" --cache "$BASE/h9_v4_scratch_v1/training_cache" \
    --pairs "$BASE/h9_paired_slope_scratch_v1/formal/pairs" --out "$ROOT/formal"
else
  [[ -f "$JOBFILE" && -f "$CODE/train.py" ]] || { echo '[STOP] missing original frozen job'; exit 2; }
  if [[ "$MODE" == resume ]]; then
    [[ -f "$ROOT/formal/latest.pt" && ! -f "$ROOT/formal/final.pt" ]] || { echo '[STOP] cannot resume'; exit 2; }
  else
    [[ -f "$ROOT/formal/solver_goal_complete.json" && ! -e "$ROOT/eval" ]] || { echo '[STOP] no complete model or evaluation exists'; exit 2; }
  fi
fi
export SG_CODE="$CODE" SG_JOB="$JOBFILE" SG_ROOT="$ROOT" SG_ACTION="$MODE"
NODE_ARGS=(--exclude=3090node1); [[ -z "${NODE:-}" ]] || NODE_ARGS=(--nodelist="$NODE")
LIMIT=3-00:00:00; [[ "$MODE" != eval ]] || LIMIT=02:00:00
RAW="$(sbatch --parsable --partition=GPU --nodes=1 --ntasks=1 --gres=gpu:3090:4 --cpus-per-task=16 \
  --time="${TIME_LIMIT:-$LIMIT}" "${NODE_ARGS[@]}" --chdir="$REPO" --export=ALL --job-name="sg_$MODE" \
  --output="$ROOT/logs/${MODE}_%j.out" --error="$ROOT/logs/${MODE}_%j.out" "$CODE/run.sh" _worker)"
JOB="${RAW%%;*}"; [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "[STOP] unparsed submission $RAW; inspect squeue"; exit 2; }
LOG="$ROOT/logs/${MODE}_${JOB}.out"
printf 'export JOB=%q\nexport LOG=%q\n' "$JOB" "$LOG" > "$ROOT/job_${JOB}.env"
cp "$ROOT/job_${JOB}.env" "$ROOT/latest.env"
echo "[submitted] job=$JOB mode=$MODE gpus=4 [out] $ROOT/formal [log] $LOG"
