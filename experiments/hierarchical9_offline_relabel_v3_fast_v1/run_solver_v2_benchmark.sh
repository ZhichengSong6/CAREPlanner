#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${VIS_PYTHON:?}" "${SOURCE:?}" "${REPO:?}" "${OUT:?}"
WORLD="${WORLD:-16}";URDF="${URDF:-$REPO/src/arm_description/urdf/Arm.urdf}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$(dirname "$OUT")"
pids=();for ((r=0;r<WORLD;r++));do "$VIS_PYTHON" -u "$HERE/benchmark_solver_v2.py" --source "$SOURCE" --repo "$REPO" --urdf "$URDF" --output "$OUT" --rank "$r" --world "$WORLD" --per-stratum "${PER_STRATUM:-4}" >"$(dirname "$OUT")/v2_rank${r}.log" 2>&1 & pids+=("$!");done
bad=0;for p in "${pids[@]}";do wait "$p"||bad=1;done;((bad==0))||exit 2
"$VIS_PYTHON" -u "$HERE/benchmark_solver_v2.py" --source "$SOURCE" --repo "$REPO" --urdf "$URDF" --output "$OUT" --world "$WORLD" --merge
