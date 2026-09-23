#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${OUT:?}" "${VIS_PYTHON:?}"
mkdir -p "$OUT/logs"
export LOCAL_WORKERS="${LOCAL_WORKERS:-16}" STAGE_IMPL="v4_tubes.py" STAGE_NAME="v4"
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 bash "$HERE/local_cpu_node.sh"
"$VIS_PYTHON" -u "$HERE/v4_tubes.py" --out "$OUT" --merge
