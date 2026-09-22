#!/usr/bin/env bash
# Explicit single submission. watch.sh never invokes this file.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PHASE="${1:-base}"
[[ "$PHASE" == base || "$PHASE" == audit ]] || { echo 'usage: bash submit.sh base|audit';exit 2; }
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
BATCH_BASE="${BATCH_BASE:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_offline_labels_v3}"
SOURCE_OUT="${SOURCE_OUT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_offline_labels_v1/smoke_20260921_202241}"
# Intentionally ignore old OUT / VERIFY_OUT / LOG_DIR variables from earlier tasks.
BATCH_OUT="${BATCH_OUT:-}"
if [[ "$PHASE" == base && -z "$BATCH_OUT" ]];then BATCH_OUT="$BATCH_BASE/pilot_$(date +%Y%m%d_%H%M%S)_$$";fi
[[ -n "$BATCH_OUT" ]] || { echo '[STOP] audit requires the explicit BATCH_OUT printed by base';exit 2; }
WORKERS="${WORKERS:-4}";DEVICE="${DEVICE:-cuda}";RESUME="${RESUME:-false}"
[[ "$WORKERS" =~ ^[1-9][0-9]*$ && ( "$DEVICE" == cpu || "$DEVICE" == cuda ) ]] || exit 2
[[ -x "$VIS_PYTHON" && -f "$SOURCE_OUT/manifest.json" ]] || { echo '[STOP] Python or source smoke missing';exit 2; }
if [[ "$PHASE" == base && -e "$BATCH_OUT" && "$RESUME" != true ]];then
  echo '[STOP] BATCH_OUT exists; use explicit RESUME=true for an interrupted run, never delete it';exit 2
fi
if [[ "$PHASE" == audit && ! -f "$BATCH_OUT/base_cache/dataset_index.json" ]];then
  echo '[STOP] audit requires a completed base_cache';exit 2
fi
"$VIS_PYTHON" - <<'PY'
import sys,numpy,scipy,torch
from scipy.optimize import minimize,lsq_linear
from urdf_parser_py.urdf import URDF
if sys.version_info<(3,10):raise SystemExit('Python>=3.10 required')
print('[dependencies] PASS',sys.executable,'scipy='+scipy.__version__)
PY
mkdir -p "$BATCH_BASE/logs" "$BATCH_BASE/submission_guards"
KEY="$(printf '%s\n%s' "$PHASE" "$BATCH_OUT" | sha256sum | cut -d' ' -f1)"
exec 8>"$BATCH_BASE/submission_guards/$KEY.lock"
flock -n 8 || { echo '[STOP] concurrent submission';exit 2; }
MARKER="$BATCH_BASE/submission_guards/$KEY.job"
if [[ -f "$MARKER" ]];then
  OLD_JOB="$(cat "$MARKER")"
  ACTIVE="$(squeue -h -j "$OLD_JOB" -o '%A')" || { echo '[STOP] cannot verify previous job state';exit 2; }
  [[ -z "$ACTIVE" ]] || { echo "[STOP] existing job $OLD_JOB is active; use watch.sh, do not resubmit";exit 2; }
  [[ "$RESUME" == true ]] || { echo '[STOP] already submitted; explicit RESUME=true required after confirming old job ended';exit 2; }
fi
LABEL_SCRIPT="$HERE/run.sh"
export PHASE REPO SOURCE_OUT BATCH_BASE BATCH_OUT VIS_PYTHON WORKERS DEVICE RESUME LABEL_SCRIPT
opts=(--parsable --job-name=r1_label_batch --partition=GPU --nodes=1 --ntasks=1 --cpus-per-task=16
      --nodelist="${NODE:-3090node1}" --time="${TIME_LIMIT:-24:00:00}"
      --output="$BATCH_BASE/logs/${PHASE}_%j.out" --export=ALL)
[[ "$DEVICE" == cuda ]] && opts+=(--gres="gpu:$WORKERS")
RAW_JOB="$(sbatch "${opts[@]}" "$HERE/worker.sbatch")"
JOB_ID="${RAW_JOB%%;*}"
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo "[ERROR] sbatch returned $RAW_JOB; inspect squeue before any retry";exit 2; }
printf '%s\n' "$JOB_ID" >"$MARKER"
LOG="$BATCH_BASE/logs/${PHASE}_${JOB_ID}.out"
STATE="$BATCH_BASE/${PHASE}_job_${JOB_ID}.env"
printf 'export JOB_ID=%q\nexport OUT=%q\nexport LOG=%q\nexport WORKERS=%q\nexport PHASE=%q\n' \
  "$JOB_ID" "$BATCH_OUT" "$LOG" "$WORKERS" "$PHASE" >"$STATE"
cp "$STATE" "$BATCH_BASE/latest_${PHASE}.env"
printf '[submitted] job=%s phase=%s\n[out] %s\n[log] %s\n[state] %s\n' "$JOB_ID" "$PHASE" "$BATCH_OUT" "$LOG" "$STATE"
printf '[watch only] bash %q %q main\n' "$HERE/watch.sh" "$STATE"
echo '[notice] finite pilot only; do not resubmit just to view logs'
