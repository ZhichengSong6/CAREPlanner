#!/usr/bin/env bash
# Run on a compute node / terminal with the already-installed viscdf environment.
# No git operations, package installation, training, ROS launch or global cleanup.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-smoke}"
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
ARTIFACT_REPO="${ARTIFACT_REPO:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner}"
DATA="${DATA:-$ARTIFACT_REPO/src/care_visibility_cdf/data/visibility_yiming_style_grid30_q20000_k500_fovonly.npz}"
BANK_CACHE="${BANK_CACHE:-$ARTIFACT_REPO/outputs/mainline_b/r1_offline_labels_v1/bank_numeric_cache}"
OUT="${OUT:-$ARTIFACT_REPO/outputs/mainline_b/r1_offline_labels_v1/${MODE}_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
DEVICE="${DEVICE:-cuda}"
WORKERS="${WORKERS:-4}"
RESUME="${RESUME:-false}"
URDF="${URDF:-$REPO/src/arm_description/urdf/Arm.urdf}"
SOLVER_CONFIG="${SOLVER_CONFIG:-$HERE/solver_config.json}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
[[ -x "$VIS_PYTHON" ]] || { echo "[ERROR] set VIS_PYTHON to an existing environment" >&2; exit 2; }
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo '[ERROR] WORKERS must be positive'; exit 2; }
case "$MODE" in
  smoke) TRAIN_X="${TRAIN_X:-2}"; VAL_X="${VAL_X:-2}"; UNIFORM_PER_X="${UNIFORM_PER_X:-1}"; NEAR_PER_X="${NEAR_PER_X:-1}"; SHARD_SIZE="${SHARD_SIZE:-2}" ;;
  pilot) TRAIN_X="${TRAIN_X:-32}"; VAL_X="${VAL_X:-8}"; UNIFORM_PER_X="${UNIFORM_PER_X:-4}"; NEAR_PER_X="${NEAR_PER_X:-2}"; SHARD_SIZE="${SHARD_SIZE:-8}" ;;
  *) echo '[ERROR] mode must be smoke or pilot' >&2; exit 2 ;;
esac
[[ "$DEVICE" == cpu || "$DEVICE" == cuda ]] || { echo '[ERROR] DEVICE=cpu or cuda';exit 2; }
[[ -f "$DATA" && -f "$URDF" ]] || { echo '[ERROR] DATA or URDF missing; set the explicit paths' >&2; exit 2; }
"$VIS_PYTHON" - "$DEVICE" "$WORKERS" <<'PY'
import sys,numpy,scipy,torch
print('[environment]',sys.version.split()[0], 'numpy='+numpy.__version__, 'scipy='+scipy.__version__, 'torch='+torch.__version__)
if sys.version_info<(3,10): raise SystemExit('Python >=3.10 required by the existing repository')
if sys.argv[1]=='cuda' and torch.cuda.device_count()<int(sys.argv[2]):
    raise SystemExit(f'Requested {sys.argv[2]} CUDA workers but only {torch.cuda.device_count()} visible GPUs')
PY
"$VIS_PYTHON" -m unittest discover -s "$HERE" -p test_labels.py -v
if [[ -e "$OUT" ]]; then
  [[ "$RESUME" == true && -f "$OUT/manifest.json" ]] || { echo "[ERROR] output exists; use a NEW OUT or explicitly RESUME=true" >&2; exit 2; }
  "$VIS_PYTHON" - "$OUT/manifest.json" "$TRAIN_X" "$VAL_X" "$UNIFORM_PER_X" "$NEAR_PER_X" "$SHARD_SIZE" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]));a=m['sampling']
expected=dict(zip(('train_x','val_x','uniform_per_x','near_per_x'),map(int,sys.argv[2:6])))
if any(a[k]!=v for k,v in expected.items()) or m['shard_size']!=int(sys.argv[6]):
    raise SystemExit('Resume sampling differs; repeat the original mode/options')
PY
else
  "$VIS_PYTHON" "$HERE/cli.py" prepare --data "$DATA" --bank-cache "$BANK_CACHE" --out "$OUT" \
    --train-x "$TRAIN_X" --val-x "$VAL_X" --uniform-per-x "$UNIFORM_PER_X" --near-per-x "$NEAR_PER_X" --shard-size "$SHARD_SIZE"
fi
mkdir -p "$OUT/logs"
echo "[output] $OUT"
"$VIS_PYTHON" "$HERE/cli.py" preflight --repo "$REPO" --urdf "$URDF" --out "$OUT" --device cpu --solver-config "$SOLVER_CONFIG"
pids=()
stop_own_workers() {
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap 'stop_own_workers; exit 130' INT
trap 'stop_own_workers; exit 143' TERM
for ((rank=0; rank<WORKERS; rank++)); do
  dev=cpu
  [[ "$DEVICE" == cuda ]] && dev="cuda:$rank"
  "$VIS_PYTHON" "$HERE/cli.py" worker --repo "$REPO" --urdf "$URDF" --out "$OUT" \
    --device "$dev" --rank "$rank" --world-size "$WORKERS" \
    > "$OUT/logs/worker_${rank}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "[ERROR] a worker failed. Read $OUT/logs/worker_*.log; completed shards are retained." >&2
    stop_own_workers; exit 2
  fi
done
trap - INT TERM
"$VIS_PYTHON" "$HERE/cli.py" merge --out "$OUT"
cat "$OUT/label_summary.md"
echo "[done] labeling complete; review numerical quality before any training"
