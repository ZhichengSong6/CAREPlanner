#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
[[ -x "$PY" ]] || { echo "[STOP] Python missing: $PY" >&2; exit 2; }
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
MODE="${1:-}"
shift || true
case "$MODE" in
  preflight|status|work-all|verify-all)
    [[ "$MODE" != "work-all" ]] || MODE="work"
    exec "$PY" -u "$HERE/batch.py" "$MODE" "$@" ;;
  prepare|work|verify|segment-status)
    [[ "$MODE" != "segment-status" ]] || MODE="status"
    exec "$PY" -u "$HERE/runner.py" "$MODE" "$@" ;;
  *) echo "Usage: run.sh preflight|status|work-all|verify-all|prepare|work|verify|segment-status [--root ROOT] [--segment ID]" >&2; exit 2 ;;
esac
