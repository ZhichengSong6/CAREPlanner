#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-status}"
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
# sbatch executes a spool copy. Never locate collector code relative to that copy.
if [[ "$MODE" == _batch ]]; then
  : "${G25_SINGLE_CODE:?}" "${G25_CODE:?}" "${G25_ROOT:?}" "${G25_NODES:?}" "${SLURM_JOB_ID:?}"
  exec "$VIS_PYTHON" -u "$G25_SINGLE_CODE/launcher.py" batch \
    --root "$G25_ROOT" --code "$G25_CODE" --nodes "$G25_NODES"
fi
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
BASE="$REPO/experiments/hierarchical9_grid25_data_v1/run.sh"
ROOT="${GRID25_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1}"
export GRID25_ROOT="$ROOT"
NODES="${NODES:-3090node2}"
[[ -x "$VIS_PYTHON" && -f "$BASE" ]] || { echo '[STOP] Python or original collector missing'; exit 2; }
case "$MODE" in
  status|log|ranks|summary|verify|pack|prepare) exec bash "$BASE" "$MODE" ;;
  test) "$VIS_PYTHON" -m unittest discover -s "$HERE" -p 'test_*.py' -v; exit ;;
  submit|resume) ;;
  *) echo 'Usage: run.sh submit|resume|prepare|status|log|ranks|summary|verify|pack|test'; exit 2 ;;
esac
[[ "$(git -C "$REPO" branch --show-current)" == mainline-b/h9-scratch50k-r012-v1 ]] || { echo '[STOP] wrong branch'; exit 2; }
git -C "$REPO" diff --quiet && git -C "$REPO" diff --cached --quiet || { echo '[STOP] tracked changes; no reset/stash'; exit 2; }
mkdir -p "$ROOT/logs"
exec 8>"$ROOT/.single_job_entry.lock"
flock -n 8 || { echo '[STOP] another single-job entry is running'; exit 2; }
"$VIS_PYTHON" "$HERE/launcher.py" policy --nodes "$NODES"
"$VIS_PYTHON" -m unittest discover -s "$HERE" -p 'test_*.py' -v
bash -n "$HERE/run.sh"
if [[ "$MODE" == resume && ! -f "$ROOT/formal/job.json" ]]; then
  echo '[STOP] no prepared collection to resume'; exit 2
fi
SCHEDULER="$("$VIS_PYTHON" "$HERE/launcher.py" freeze --root "$ROOT")"
# prepare never submits a job. Preserve/reuse the collector's original source.env.
NODES=3090node2 bash "$BASE" prepare
source "$ROOT/source.env"
"$VIS_PYTHON" "$SCHEDULER/launcher.py" "$MODE" --root "$ROOT" --repo "$REPO" --code "$CODE" --nodes "$NODES"
