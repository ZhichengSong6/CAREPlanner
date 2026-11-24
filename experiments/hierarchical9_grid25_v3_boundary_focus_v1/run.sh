#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
export GRID25_ROOT="${GRID25_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1}"
MODE="${1:-}"
shift || true
case "$MODE" in
  status)
    exec "$VIS_PYTHON" -u "$HERE/train.py" status --root "$GRID25_ROOT" "$@"
    ;;
  submit)
    exec bash "$HERE/submit.sh" "$@"
    ;;
  *)
    echo "Usage: bash $HERE/run.sh submit | status" >&2
    exit 2
    ;;
esac
