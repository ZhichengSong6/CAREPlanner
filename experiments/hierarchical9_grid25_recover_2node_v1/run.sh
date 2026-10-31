#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-status}"
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
if [[ "$MODE" == _batch ]]; then
  : "${G25R_SOURCE:?}" "${G25R_SPEC:?}" "${G25R_SPEC_SHA:?}" "${SLURM_JOB_ID:?}"
  exec "$VIS_PYTHON" -u "$G25R_SOURCE/launch.py" batch --spec "$G25R_SPEC"
fi
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
ROOT="${GRID25_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1}"
[[ -x "$VIS_PYTHON" ]] || { echo '[STOP] viscdf Python missing'; exit 2; }
case "$MODE" in
  test) "$VIS_PYTHON" -m unittest discover -s "$HERE" -p 'test_recovery.py' -v; exit ;;
  recover|draft|next|resume)
    [[ "$(git -C "$REPO" branch --show-current)" == mainline-b/h9-scratch50k-r012-v1 ]] || { echo '[STOP] wrong branch'; exit 2; }
    git -C "$REPO" diff --quiet && git -C "$REPO" diff --cached --quiet || { echo '[STOP] tracked edits; no reset/stash'; exit 2; }
    "$VIS_PYTHON" -m unittest discover -s "$HERE" -p 'test_recovery.py' -v
    ;;
  status|summary|verify|log|ranks) ;;
  *) echo 'Usage: run.sh recover|draft|next|resume|status|summary|verify|log|ranks|test'; exit 2 ;;
esac
exec "$VIS_PYTHON" "$HERE/launch.py" "$MODE" --repo "$REPO" --root "$ROOT" --nodes "${NODES:-3090node1,3090node2}"
