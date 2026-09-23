#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${OUT:?}" "${VIS_PYTHON:?}"
[[ -f "$OUT/global_pool_summary.json" ]] || { echo "[STOP] global_pool incomplete";exit 2; }
mkdir -p "$OUT/logs"
export LOCAL_WORKERS="${LOCAL_WORKERS:-16}" STAGE_IMPL="v3_labels.py" STAGE_NAME="v3"
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 bash "$HERE/local_cpu_node.sh"
"$VIS_PYTHON" -u "$HERE/v3_labels.py" --out "$OUT" --merge
