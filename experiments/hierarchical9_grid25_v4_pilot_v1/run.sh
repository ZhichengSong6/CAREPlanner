#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
MODE="${1:-status}"
shift || true
[[ -x "$PY" ]] || { echo "[STOP] VIS_PYTHON not executable: $PY" >&2; exit 2; }
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "$MODE" == test ]]; then
  exec "$PY" -m unittest discover -s "$HERE" -p 'test_pilot.py' -v
fi
case "$MODE" in
  prepare|work|verify|status) exec "$PY" -u "$HERE/pilot.py" "$MODE" "$@" ;;
  *) echo "Usage: run.sh test|prepare|work|verify|status [options]" >&2; exit 2 ;;
esac
