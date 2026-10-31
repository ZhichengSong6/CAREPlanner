#!/bin/bash
# User entry. `next` submits exactly one generated batch.sbatch; do not sbatch this wrapper.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
[[ -x "$VIS_PYTHON" ]] || { echo "[STOP] missing Python: $VIS_PYTHON" >&2; exit 2; }
if [[ $# -eq 0 ]]; then set -- status; fi
if [[ "$1" == test ]]; then
  exec "$VIS_PYTHON" -m unittest discover -s "$HERE" -p 'test_slurm_tasks.py' -v
fi
exec "$VIS_PYTHON" "$HERE/slurm_tasks.py" "$@"
