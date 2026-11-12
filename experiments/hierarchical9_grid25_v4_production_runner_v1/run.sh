#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
[[ -x "$PY" ]] || { echo "[STOP] Missing VIS_PYTHON: $PY" >&2; exit 2; }
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
MODE="${1:-status}"
shift || true
if [[ "$MODE" == test ]]; then
  exec "$PY" -m unittest discover -s "$HERE" -p 'test_runner.py' -v
fi
case "$MODE" in
  prepare|status|work|verify) exec "$PY" -u "$HERE/runner.py" "$MODE" "$@" ;;
  *) echo "Usage: run.sh test|prepare|status|work|verify [--root ROOT] [--segment N]" >&2; exit 2 ;;
esac
