#!/usr/bin/env bash
# Scheduling-only recovery. Use the original frozen Keep-v2 runner, unmodified.
set -euo pipefail
if [[ "${1:-}" == _worker ]]; then
  : "${SLURM_JOB_ID:?}" "${K2_CODE:?}" "${K2_JOB:?}" "${K2_ROOT:?}" "${K2_RESOURCE_CHECK:?}" "${K2_RESOURCE_SHA:?}" "${VIS_PYTHON:?}"
  export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
  "$VIS_PYTHON" - "$K2_RESOURCE_CHECK" "$K2_RESOURCE_SHA" <<'PY'
import hashlib,sys
from pathlib import Path
if hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()!=sys.argv[2]:
    raise SystemExit('[STOP] resource checker snapshot changed')
PY
  # Both checks run as separate processes. Their CUDA contexts do not survive
  # into the original worker, so the memory probe is not a resident GPU user.
  "$VIS_PYTHON" -u "$K2_RESOURCE_CHECK" validate --root "$K2_ROOT" --code "$K2_CODE"
  "$VIS_PYTHON" -u "$K2_RESOURCE_CHECK" gpu
  export K2_ACTION=resume
  echo "[resume] original frozen source=$K2_CODE; same protocol/batch/weights, no label regeneration"
  exec bash "$K2_CODE/run.sh" _worker
fi
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
MODE="${1:-resume}"
ROOT="${K2_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_solver_goal_keep_v2}"
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
[[ -x "$VIS_PYTHON" ]] || { echo '[STOP] viscdf Python missing'; exit 2; }
if [[ "$MODE" == test ]]; then
  "$VIS_PYTHON" -m unittest discover -s "$HERE" -p 'test_*.py' -v
  exit 0
fi
[[ "$MODE" == resume || "$MODE" == check ]] || { echo 'Usage: bash run.sh check|resume|test'; exit 2; }
[[ "$(git -C "$REPO" branch --show-current)" == mainline-b/h9-scratch50k-r012-v1 ]] || { echo '[STOP] wrong branch'; exit 2; }
git -C "$REPO" diff --quiet && git -C "$REPO" diff --cached --quiet || { echo '[STOP] tracked changes; no reset/stash'; exit 2; }
[[ -d "$ROOT" && -f "$ROOT/source.env" && -f "$ROOT/latest.env" && -f "$ROOT/formal/job.json" ]] || { echo '[STOP] original attempt/state missing; do not create a new training run'; exit 2; }
ROOT="$(realpath -e "$ROOT")"
exec 9>"$ROOT/.submit.lock"
flock -n 9 || { echo '[STOP] another submitter active'; exit 2; }
source "$ROOT/source.env"
source "$ROOT/latest.env"
[[ "$JOB" =~ ^[0-9]+$ && -d "$CODE" ]] || { echo '[STOP] invalid original state'; exit 2; }
PREVIOUS_JOB="$JOB"
QUEUE="$(squeue -h -j "$JOB" -o '%T' 2>/dev/null || true)"
[[ -z "$QUEUE" ]] || { echo "[STOP] job $JOB is still $QUEUE; do not resubmit"; exit 2; }
ACCOUNT="$(sacct -X -n -P -j "$JOB" --format=JobIDRaw,State)"
ST="$(printf '%s\n' "$ACCOUNT" | awk -F'|' -v j="$JOB" '$1==j{print $2;exit}')"
case "$ST" in FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|COMPLETED) ;; *) echo "[STOP] previous job state unresolved: $ST"; exit 2;; esac
exec 8>"$ROOT/.worker.lock"
flock -n 8 || { echo '[STOP] original worker lock is held'; exit 2; }
exec 8>&-
"$VIS_PYTHON" -m unittest discover -s "$HERE" -p 'test_*.py' -v
"$VIS_PYTHON" -u "$HERE/resource_check.py" validate --root "$ROOT" --code "$CODE"
[[ "$MODE" != check ]] || { echo '[CHECK COMPLETE] no GPU job submitted'; exit 0; }
ID="$("$VIS_PYTHON" - "$HERE" <<'PY'
import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]);m={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ('run.sh','resource_check.py')}
print(hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest())
PY
)"
mkdir -p "$ROOT/recovery_sources" "$ROOT/logs"
RECOVERY="$ROOT/recovery_sources/$ID"
if [[ ! -d "$RECOVERY" ]]; then
  TMP="$(mktemp -d "$ROOT/recovery_sources/.building.XXXXXX")"
  cp "$HERE/run.sh" "$HERE/resource_check.py" "$TMP/"
  mv "$TMP" "$RECOVERY"
fi
cmp -s "$HERE/run.sh" "$RECOVERY/run.sh" && cmp -s "$HERE/resource_check.py" "$RECOVERY/resource_check.py" || { echo '[STOP] recovery snapshot corrupted'; exit 2; }
export K2_CODE="$CODE" K2_JOB="$ROOT/formal/job.json" K2_ROOT="$ROOT" K2_ACTION=resume
export K2_RESOURCE_CHECK="$RECOVERY/resource_check.py"
export K2_RESOURCE_SHA="$("$VIS_PYTHON" -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$K2_RESOURCE_CHECK")"
NODE_ARGS=(--exclude=3090node1)
[[ -z "${NODE:-}" ]] || NODE_ARGS=(--nodelist="$NODE")
RAW="$(sbatch --parsable --partition=GPU --nodes=1 --ntasks=1 --exclusive --gres=gpu:3090:4 \
  --cpus-per-task=16 --time="${TIME_LIMIT:-3-00:00:00}" "${NODE_ARGS[@]}" --chdir="$REPO" --export=ALL \
  --job-name=sg_keep_recover --output="$ROOT/logs/resume_%j.out" --error="$ROOT/logs/resume_%j.out" \
  "$RECOVERY/run.sh" _worker)"
JOB="${RAW%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "[STOP] unparsed submission $RAW; inspect queue before retrying"; exit 2; }
LOG="$ROOT/logs/resume_${JOB}.out"
printf 'export JOB=%q\nexport LOG=%q\n' "$JOB" "$LOG" > "$ROOT/job_${JOB}.env"
cp "$ROOT/job_${JOB}.env" "$ROOT/.latest.recovery.$$"
mv "$ROOT/.latest.recovery.$$" "$ROOT/latest.env"
printf 'previous_job=%s\nnew_job=%s\nrecovery_source=%s\noriginal_source=%s\nexclusive_requested=true\n' \
  "$PREVIOUS_JOB" "$JOB" "$RECOVERY" "$CODE" > "$ROOT/recovery_job_${JOB}.txt"
echo "[submitted] job=$JOB action=RESUME_EXISTING exclusive_requested=true gpus=4 previous=$PREVIOUS_JOB"
echo "[source unchanged] $CODE"
echo "[out] $ROOT/formal"
echo "[log] $LOG"
