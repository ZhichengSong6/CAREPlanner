#!/usr/bin/env bash
# Submit exactly once. watch.sh NEVER calls this script.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
ARTIFACT_REPO="${ARTIFACT_REPO:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner}"
export SOURCE_OUT="${SOURCE_OUT:-$ARTIFACT_REPO/outputs/mainline_b/r1_offline_labels_v1/smoke_20260921_202241}"
BASE="${VERIFY_BASE:-$ARTIFACT_REPO/outputs/mainline_b/r1_offline_labels_v2}"
# Ignore a stale OUT exported by the old smoke. Resume uses VERIFY_OUT explicitly.
export OUT="${VERIFY_OUT:-$BASE/verify_$(date +%Y%m%d_%H%M%S)_$$}"
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
export DEVICE="${DEVICE:-cuda}" WORKERS="${WORKERS:-4}" RESUME="${RESUME:-false}"
export LABEL_VERIFY_SCRIPT="$HERE/run.sh"
[[ -x "$VIS_PYTHON" ]] || { echo "[ERROR] missing Python: $VIS_PYTHON" >&2;exit 2; }
[[ "$WORKERS" =~ ^[1-9][0-9]*$ && ( "$DEVICE" == cpu || "$DEVICE" == cuda ) ]] || exit 2
# No full data extraction or geometry optimization on submission node.
"$VIS_PYTHON" - "$SOURCE_OUT" "$OUT" "$RESUME" <<'PY'
import sys,numpy,scipy,torch
from scipy.optimize import minimize,lsq_linear
from urdf_parser_py.urdf import URDF
from pathlib import Path
s,o=map(lambda p:Path(p).resolve(),sys.argv[1:3])
if sys.version_info<(3,10):raise SystemExit('Python>=3.10 required')
if not (s/'dataset_index.json').is_file():raise SystemExit('Completed source smoke missing')
if s==o or o.is_relative_to(s) or s.is_relative_to(o):raise SystemExit('Output/source overlap forbidden')
if o.exists() and sys.argv[3]!='true':raise SystemExit('Output exists; no duplicate overwrite')
print('[dependencies] PASS',sys.executable,'scipy='+scipy.__version__,flush=True)
PY
mkdir -p "$BASE/logs"
resource=()
if [[ "$DEVICE" == cuda ]]; then resource=(--gres="gpu:$WORKERS"); fi
job=$(sbatch --parsable --job-name=r1_label_verify --partition=GPU --nodes=1 --ntasks=1 \
  --cpus-per-task="${CPUS:-16}" --nodelist="${NODE:-3090node1}" --time="${TIME_LIMIT:-03:00:00}" \
  "${resource[@]}" --output="$BASE/logs/verify_%j.out" --export=ALL "$HERE/worker.sbatch")
JOB_ID="${job%%;*}"; LOG="$BASE/logs/verify_${JOB_ID}.out"
if [[ ! "$JOB_ID" =~ ^[0-9]+$ ]]; then echo "[ERROR] cannot parse sbatch output: $job; DO NOT resubmit blindly" >&2;exit 2;fi
STATE_FILE="$BASE/verify_job_${JOB_ID}.env"
printf 'export JOB_ID=%q\nexport OUT=%q\nexport LOG=%q\nexport WORKERS=%q\nexport SOURCE_OUT=%q\n' \
  "$JOB_ID" "$OUT" "$LOG" "$WORKERS" "$SOURCE_OUT" > "$STATE_FILE"
cp "$STATE_FILE" "$BASE/latest_verify.env"
printf '[submitted] job=%s\n[source] %s\n[out] %s\n[log] %s\n[state] %s\n' "$JOB_ID" "$SOURCE_OUT" "$OUT" "$LOG" "$STATE_FILE"
printf '[watch only] bash %q %q main\n' "$HERE/watch.sh" "$STATE_FILE"
echo '[notice] do not resubmit to view output; Ctrl+C in watch only exits viewing'
