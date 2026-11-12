#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
BASE="$REPO/experiments/hierarchical9_grid25_v4_pilot_v1"
PY="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
GRID25_ROOT="${GRID25_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1}"
EXT_OUT="${EXT_OUT:-$GRID25_ROOT/derived/grid25_v4_extended_v1/pilot_train32_val32}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
MODE="${1:-status}"
case "$MODE" in
  test)
    exec "$PY" -m unittest discover -s "$HERE" -p 'test_extended.py' -v ;;
  prepare)
    exec bash "$BASE/run.sh" prepare --root "$GRID25_ROOT" --out "$EXT_OUT" --train-x 32 --val-x 32 --seed 260927 ;;
  submit)
    export PILOT_OUT="$EXT_OUT" PILOT_WORKERS=4 PILOT_CPUS=4
    exec bash "$BASE/submit.sh" ;;
  status)
    exec bash "$BASE/run.sh" status --out "$EXT_OUT" ;;
  verify)
    exec "$PY" "$HERE/gradient_audit.py" status --out "$EXT_OUT" ;;
  fd-plan|fd-work|fd-verify|fd-status)
    ACTION="${MODE#fd-}"
    [[ "$ACTION" != verify || -f "$EXT_OUT/extended_fd_audit_v1/audit_manifest.json" ]] || { echo "[STOP] FD audit not prepared" >&2; exit 2; }
    shift
    exec "$PY" -u "$HERE/gradient_audit.py" "$ACTION" --out "$EXT_OUT" "$@" ;;
  *) echo "Usage: run.sh test|prepare|submit|status|verify|fd-plan|fd-work|fd-verify|fd-status" >&2; exit 2 ;;
esac
