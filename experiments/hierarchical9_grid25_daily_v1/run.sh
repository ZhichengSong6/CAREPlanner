#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-status}"
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
if [[ "$MODE" == _batch ]]; then
  : "${G25D_SOURCE:?}" "${G25D_SPEC:?}" "${SLURM_JOB_ID:?}"
  exec "$VIS_PYTHON" -u "$G25D_SOURCE/launch.py" batch --spec "$G25D_SPEC"
fi
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
ROOT="${GRID25_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1}"
NODE="${NODE:-3090node2}"
[[ -x "$VIS_PYTHON" ]] || { echo '[STOP] viscdf Python missing'; exit 2; }
if [[ "$MODE" == test ]]; then
  "$VIS_PYTHON" -m unittest discover -s "$HERE" -p 'test_daily.py' -v
  exit 0
fi
case "$MODE" in
  next|submit|resume)
    [[ "$(git -C "$REPO" branch --show-current)" == mainline-b/h9-scratch50k-r012-v1 ]] || { echo '[STOP] wrong branch'; exit 2; }
    git -C "$REPO" diff --quiet && git -C "$REPO" diff --cached --quiet || { echo '[STOP] tracked changes; no reset/stash'; exit 2; }
    "$VIS_PYTHON" "$HERE/launch.py" policy --node "$NODE"
    "$VIS_PYTHON" -m unittest discover -s "$HERE" -p 'test_daily.py' -v
    bash -n "$HERE/run.sh"
    ;;
  status|summary|verify|ranks|log|pack) ;;
  *) echo 'Usage: run.sh next|resume|status|summary|ranks|log|verify|pack|test'; exit 2 ;;
esac
exec "$VIS_PYTHON" "$HERE/launch.py" "$MODE" --root "$ROOT" --repo "$REPO" --node "$NODE"
