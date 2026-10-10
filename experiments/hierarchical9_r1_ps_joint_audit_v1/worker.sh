#!/usr/bin/env bash
set -euo pipefail
: "${JEA_PY:?}" "${JEA_SOURCE:?}" "${JEA_JOB_JSON:?}"
# Slurm spools THIS file elsewhere. All Python paths are explicitly frozen at submit.
[[ -x "$JEA_PY" && -f "$JEA_SOURCE/cli.py" && -f "$JEA_JOB_JSON" ]] || {
  echo '[STOP] frozen evaluation source/Python missing'; exit 2;
}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
exec "$JEA_PY" -u "$JEA_SOURCE/cli.py" execute --job "$JEA_JOB_JSON"
