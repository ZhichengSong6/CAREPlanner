#!/usr/bin/env bash
# Executes within an allocation. Never submits/cancels jobs or edits old caches.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
: "${SOURCE_OUT:?Set SOURCE_OUT to the completed v1 smoke}"
: "${OUT:?Set OUT to a separate v2 output directory}"
VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
DEVICE="${DEVICE:-cuda}"; WORKERS="${WORKERS:-4}"
CASES="${CASES:-4:0,1:7,5:6,3:2,2:4,2:3}"
URDF="${URDF:-$REPO/src/arm_description/urdf/Arm.urdf}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ "$WORKERS" =~ ^[1-9][0-9]*$ && ( "$DEVICE" == cpu || "$DEVICE" == cuda ) ]] || exit 2
"$VIS_PYTHON" - "$DEVICE" "$WORKERS" <<'PY'
import sys,numpy,scipy,torch
from scipy.optimize import minimize,lsq_linear
from urdf_parser_py.urdf import URDF
print('[environment]',sys.executable,'numpy='+numpy.__version__,'scipy='+scipy.__version__,'torch='+torch.__version__,flush=True)
if sys.version_info<(3,10):raise SystemExit('Python>=3.10 required')
if sys.argv[1]=='cuda' and torch.cuda.device_count()<int(sys.argv[2]):raise SystemExit('Insufficient allocated CUDA devices')
PY
"$VIS_PYTHON" -m unittest discover -s "$HERE" -p test_verified.py -v
args=()
[[ "${RESUME:-false}" == true ]] && args+=(--resume)
"$VIS_PYTHON" -u "$HERE/review_smoke.py" prepare --source "$SOURCE_OUT" --out "$OUT" --cases "$CASES" "${args[@]}"
mkdir -p "$OUT/logs"
echo "[stage] fixed-query verification; original smoke is read-only"
pids=()
cleanup_own_children() { for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done; for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done; }
trap 'cleanup_own_children; exit 130' INT
trap 'cleanup_own_children; exit 143' TERM
for ((rank=0;rank<WORKERS;rank++)); do
  dev=cpu; [[ "$DEVICE" == cuda ]] && dev="cuda:$rank"
  "$VIS_PYTHON" -u "$HERE/review_smoke.py" worker --source "$SOURCE_OUT" --out "$OUT" --repo "$REPO" --urdf "$URDF" \
    --device "$dev" --rank "$rank" --world-size "$WORKERS" > "$OUT/logs/worker_${rank}.log" 2>&1 &
  pids+=("$!")
done
# Live stage heartbeats in the MAIN log as well as in per-worker logs.
while :; do
  active=0
  for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && active=$((active+1)); done
  [[ "$active" -gt 0 ]] || break
  for ((rank=0;rank<WORKERS;rank++)); do
    printf '[progress worker=%s] ' "$rank"
    tail -n 1 "$OUT/logs/worker_${rank}.log" 2>/dev/null || true
  done
  sleep 15
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
trap - INT TERM
if [[ "$failed" -ne 0 ]]; then echo "[ERROR] read $OUT/logs/worker_*.log; completed cases retained" >&2; exit 2; fi
"$VIS_PYTHON" -u "$HERE/review_smoke.py" merge --out "$OUT"
echo "[done] $OUT/verification_summary.md"
echo '[scope] verification only; no training/promotion and no old-cache overwrite'
