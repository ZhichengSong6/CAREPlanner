#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
[[ -x "$PY" ]] || { echo "[STOP] Python not found: $PY" >&2; exit 2; }
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
MODE="${1:-}"
shift || true
case "$MODE" in
  prepare|verify) exec "$PY" -u "$HERE/prepare.py" "$MODE" "$@" ;;
  train) exec "$PY" -u "$HERE/train.py" "$@" ;;
  *) echo "Usage: run.sh prepare|verify|train [args]" >&2; exit 2 ;;
esac
