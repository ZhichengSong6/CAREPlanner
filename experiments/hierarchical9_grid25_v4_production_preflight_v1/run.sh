#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
[[ -x "$PY" ]] || { echo "[STOP] VIS_PYTHON missing: $PY" >&2; exit 2; }
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
MODE="${1:-verify}"
shift || true
if [[ "$MODE" == test ]]; then
  exec "$PY" -m unittest discover -s "$HERE" -p 'test_preflight.py' -v
fi
case "$MODE" in
  prepare|verify) exec "$PY" -u "$HERE/preflight.py" "$MODE" "$@" ;;
  *) echo "Usage: run.sh test|prepare|verify [options]" >&2; exit 2 ;;
esac
