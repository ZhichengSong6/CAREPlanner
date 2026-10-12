#!/usr/bin/env bash
set -euo pipefail
: "${O_REPO:?}" "${O_ROOT:?}" "${O_SOURCE:?}" "${O_MODE:?}" "${O_PY:?}"
cd "$O_REPO"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
exec 8>"$O_ROOT/.worker.lock"
flock -n 8 || { echo '[STOP] another worker owns this output'; exit 2; }
COMMON=(--repo "$O_REPO" --spec "$O_ROOT/job.json" --out "$O_ROOT/formal")
"$O_PY" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==4; print("[GPU]",[torch.cuda.get_device_name(i) for i in range(4)],flush=True)'
if [[ "$O_MODE" == train || "$O_MODE" == resume ]]; then
  EXTRA=(); [[ "$O_MODE" != resume ]] || EXTRA=(--resume)
  "$O_PY" -m torch.distributed.run --standalone --nproc_per_node=4 --max_restarts=0 \
    "$O_SOURCE/driver.py" train "${COMMON[@]}" "${EXTRA[@]}"
  "$O_PY" "$O_SOURCE/driver.py" verify "${COMMON[@]}"
  echo '[complete] final.pt/runtime.pt ready for evaluation; do not promote yet'
else
  OUT="$O_ROOT/evaluation"
  [[ ! -e "$OUT" ]] || { echo '[STOP] existing eval, no overwrite'; exit 2; }
  mkdir "$OUT"
  IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  [[ "${#GPUS[@]}" == 4 ]] || { echo '[STOP] GPU assignment mismatch'; exit 2; }
  PIDS=()
  stop_children() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; wait || true; }
  trap stop_children EXIT
  for R in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="${GPUS[$R]}" "$O_PY" -u "$O_SOURCE/evaluate.py" \
       --repo "$O_REPO" --spec "$O_ROOT/job.json" --train "$O_ROOT/formal" --out "$OUT" --rank "$R" > "$OUT/rank${R}.log" 2>&1 &
    PIDS+=("$!")
  done
  for _ in 0 1 2 3; do
    if ! wait -n; then
      echo '[STOP] evaluation worker failed'
      for R in 0 1 2 3; do tail -n 35 "$OUT/rank${R}.log" || true; done
      exit 2
    fi
  done
  trap - EXIT
  "$O_PY" "$O_SOURCE/evaluate.py" --repo "$O_REPO" --spec "$O_ROOT/job.json" \
      --train "$O_ROOT/formal" --out "$OUT" --merge
fi
