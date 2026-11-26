#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export GRID25_ROOT="${GRID25_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1}"
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
[[ -x "$VIS_PYTHON" ]] || { echo "[STOP] viscdf python not found" >&2; exit 2; }
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
MODE="${1:-}"
shift || true
case "$MODE" in
  prepare|merge|label-status)
    if [[ "$MODE" == label-status ]]; then MODE=status; fi
    exec "$VIS_PYTHON" -u "$HERE/labels.py" "$MODE" --root "$GRID25_ROOT" "$@"
    ;;
  submit-labels)
    exec bash "$HERE/submit_labels.sh" "$@"
    ;;
  submit-train)
    exec bash "$HERE/submit_train.sh" "$@"
    ;;
  train-status)
    exec "$VIS_PYTHON" -u "$HERE/train.py" status --root "$GRID25_ROOT" "$@"
    ;;
  fov-compare)
    exec "$VIS_PYTHON" -u "$HERE/fov_compare.py" --root "$GRID25_ROOT" "$@"
    ;;
  *)
    echo "Usage: bash $HERE/run.sh prepare | submit-labels | label-status | merge | submit-train | train-status | fov-compare" >&2
    exit 2
    ;;
esac
