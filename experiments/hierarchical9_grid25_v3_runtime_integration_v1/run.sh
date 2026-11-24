#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../.." && pwd)"
cd "$REPO"
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
export GRID25_ROOT="${GRID25_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1}"
[[ -x "$VIS_PYTHON" ]] || { echo "[STOP] viscdf Python missing: $VIS_PYTHON" >&2; exit 2; }
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
MODE="${1:-}"
shift || true
case "$MODE" in
  verify)
    exec "$VIS_PYTHON" -u "$HERE/verify.py" --root "$GRID25_ROOT" "$@"
    ;;
  case026)
    exec bash "$HERE/case026.sh" "$@"
    ;;
  *)
    echo "Usage: bash $HERE/run.sh verify [--device cpu|cuda] OR case026 (CPU, baseline vs V3)" >&2
    exit 2
    ;;
esac
