#!/usr/bin/env bash
# Offline compute only. No package installs, checkpoint loads or Git writes.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${PHASE:?Use submit.sh base or submit.sh audit}"
: "${BATCH_OUT:?Missing BATCH_OUT}"
: "${REPO:?Missing REPO}"
: "${SOURCE_OUT:?Missing SOURCE_OUT}"
: "${VIS_PYTHON:?Missing VIS_PYTHON}"
WORKERS="${WORKERS:-4}"; DEVICE="${DEVICE:-cuda}"
URDF="${URDF:-$REPO/src/arm_description/urdf/Arm.urdf}"
SAMPLING="${SAMPLING:-$HERE/sampling.json}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ "$PHASE" == base || "$PHASE" == audit ]] || exit 2
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || exit 2
"$VIS_PYTHON" - "$DEVICE" "$WORKERS" <<'PY'
import sys,numpy,scipy,torch
print('[environment]',sys.executable,numpy.__version__,scipy.__version__,torch.__version__,flush=True)
if sys.version_info<(3,10):raise SystemExit('Python>=3.10 required')
if sys.argv[1]=='cuda' and torch.cuda.device_count()<int(sys.argv[2]):raise SystemExit('Insufficient allocated GPUs')
PY
"$VIS_PYTHON" -m unittest discover -s "$HERE" -p test_batch.py -v
args=(prepare --source "$SOURCE_OUT" --out "$BATCH_OUT" --repo "$REPO" --urdf "$URDF" --sampling "$SAMPLING")
if [[ "$PHASE" == audit || "${RESUME:-false}" == true ]];then args+=(--resume);fi
"$VIS_PYTHON" -u "$HERE/cli.py" "${args[@]}"
mkdir -p "$BATCH_OUT/logs"
# One active run per phase. Worker task locks additionally protect shared files.
exec 9>"$BATCH_OUT/.${PHASE}.runner.lock"
flock -n 9 || { echo '[STOP] another runner owns this phase';exit 2; }
if [[ "$PHASE" == audit && ! -f "$BATCH_OUT/base_cache/dataset_index.json" ]];then
  echo '[STOP] base cache is incomplete; do not start audit';exit 2
fi
pids=();alive=()
cleanup(){ for pid in "${pids[@]}";do kill -TERM "$pid" 2>/dev/null || true;done;for pid in "${pids[@]}";do wait "$pid" 2>/dev/null || true;done; }
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
for ((r=0;r<WORKERS;r++));do
  dev=cpu;[[ "$DEVICE" == cuda ]] && dev="cuda:$r"
  "$VIS_PYTHON" -u "$HERE/cli.py" worker --out "$BATCH_OUT" --repo "$REPO" --urdf "$URDF" \
    --phase "$PHASE" --device "$dev" --rank "$r" --world "$WORKERS" \
    >"$BATCH_OUT/logs/${PHASE}_worker_${r}.log" 2>&1 &
  pids+=("$!");alive+=(1)
done
while :;do
  remaining=0
  for ((r=0;r<WORKERS;r++));do
    [[ "${alive[$r]}" == 1 ]] || continue
    if kill -0 "${pids[$r]}" 2>/dev/null;then
      remaining=$((remaining+1))
      printf '[progress %s worker=%s] ' "$PHASE" "$r"
      tail -n 1 "$BATCH_OUT/logs/${PHASE}_worker_${r}.log" || true
    else
      if ! wait "${pids[$r]}";then
        echo "[ERROR] worker $r failed; completed records retained; see its log"
        cleanup;exit 2
      fi
      alive[$r]=0
    fi
  done
  ((remaining>0)) || break
  sleep "${HEARTBEAT_SECONDS:-20}"
done
trap - INT TERM
"$VIS_PYTHON" -u "$HERE/cli.py" merge --out "$BATCH_OUT" --phase "$PHASE"
echo "[done] $PHASE $BATCH_OUT"
echo '[scope] finite paired labeling; no training, no promotion, no overwrite of v1/v2'
