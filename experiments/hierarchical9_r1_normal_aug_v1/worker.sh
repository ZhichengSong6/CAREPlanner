#!/usr/bin/env bash
# Slurm spools this script: load every Python module from an absolute pinned source.
set -euo pipefail
: "${R1A_REPO:?}" "${R1A_ROOT:?}" "${R1A_SOURCE:?}" "${R1A_MODE:?}" "${R1A_OUT:?}" "${R1A_R012:?}" "${R1A_CACHE:?}" "${R1A_DATA:?}" "${R1A_URDF:?}" "${VIS_PYTHON:?}"
PY="$VIS_PYTHON"
[[ -x "$PY" ]] || { echo "[STOP] missing viscdf Python $PY"; exit 2; }
[[ -f "$R1A_SOURCE/train.py" && -f "$R1A_SOURCE/boundary_aug.py" ]] || { echo '[STOP] immutable source missing'; exit 2; }
cd "$R1A_REPO"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
exec 8>"$R1A_ROOT/.gpu_writer.lock"
flock -n 8 || { echo "[STOP] concurrent writer/evaluator exists"; exit 2; }
trap 'code=$?; printf "job_id=%s\nmode=%s\nexit=%s\n" "${SLURM_JOB_ID:-unknown}" "$R1A_MODE" "$code" > "$R1A_ROOT/last_worker_exit.txt"' EXIT
"$PY" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA required"
assert torch.cuda.device_count()==4, f"Need four GPUs, saw {torch.cuda.device_count()}"
print("[GPU]",[torch.cuda.get_device_name(i) for i in range(4)],flush=True)
PY
echo "[worker] mode=$R1A_MODE source=$R1A_SOURCE host=$(hostname) out=$R1A_OUT"
if [[ "$R1A_MODE" == train || "$R1A_MODE" == resume ]]; then
  EXTRA=()
  if [[ "$R1A_MODE" == resume ]]; then EXTRA=(--resume "$R1A_OUT/latest.pt"); fi
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 --max_restarts=0 \
    "$R1A_SOURCE/train.py" train --r012-root "$R1A_R012" --cache "$R1A_CACHE" \
      --data "$R1A_DATA" --urdf "$R1A_URDF" --out "$R1A_OUT" "${EXTRA[@]}"
  "$PY" -u "$R1A_SOURCE/train.py" verify --r012-root "$R1A_R012" --cache "$R1A_CACHE" \
    --data "$R1A_DATA" --urdf "$R1A_URDF" --out "$R1A_OUT"
  echo '[done] 50k R1-plus-normal training and stream identity verified; solver NOT_RUN'
elif [[ "$R1A_MODE" == eval ]]; then
  [[ -f "$R1A_ROOT/formal/final.pt" ]] || { echo '[STOP] training missing'; exit 2; }
  mkdir -p "$R1A_OUT"
  CUDA_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  IFS=, read -r -a GPUS <<< "$CUDA_LIST"
  [[ "${#GPUS[@]}" -ge 4 ]] || { echo '[STOP] four GPUs not allocated'; exit 2; }
  PIDS=()
  for RANK in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="${GPUS[$RANK]}" "$PY" -u "$R1A_SOURCE/evaluate.py" \
      --r012-root "$R1A_R012" --aug-root "$R1A_ROOT/formal" --starts "$R1A_STARTS" \
      --output "$R1A_OUT" --rank "$RANK" --world 4 > "$R1A_OUT/rank${RANK}.log" 2>&1 &
    PIDS+=("$!")
  done
  FAIL=0
  for PID in "${PIDS[@]}"; do wait "$PID" || FAIL=1; done
  if (( FAIL )); then
    echo '[STOP] solver comparison worker failure'
    for RANK in 0 1 2 3; do tail -n 45 "$R1A_OUT/rank${RANK}.log" || true; done
    exit 2
  fi
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PY" -u "$R1A_SOURCE/evaluate.py" --merge \
    --r012-root "$R1A_R012" --aug-root "$R1A_ROOT/formal" --starts "$R1A_STARTS" \
    --output "$R1A_OUT" --world 4
  echo '[done] exact fixed 1963-case R1 vs R1-plus-boundary comparison'
else
  echo "[STOP] invalid mode=$R1A_MODE"
  exit 2
fi
