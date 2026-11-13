#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-}"
[[ "$MODE" == "v3" || "$MODE" == "joint" ]] || { echo "Usage: submit.sh v3|joint" >&2; exit 2; }
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
export GRID25_ROOT="${GRID25_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1}"
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
BASE="$GRID25_ROOT/derived/grid25_multisegment_train_v1"
DATA="$BASE/dataset_segments_000_001"
OUT="$BASE/run_$MODE"
[[ -f "$DATA/manifest.json" && -f "$DATA/v3.npz" && -f "$DATA/v4.npz" ]] || { echo "[STOP] Prepare multi-segment snapshot first" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "[STOP] Run output already exists: $OUT" >&2; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo "[STOP] Uncommitted tracked code edits" >&2; exit 2; }
bash "$HERE/run.sh" verify --root "$GRID25_ROOT"
if [[ "$MODE" == "joint" ]]; then
  "$VIS_PYTHON" - "$BASE/run_v3/run.json" "$BASE/run_v3/best.pt" <<'PY'
import json,hashlib,sys
from pathlib import Path
p=Path(sys.argv[1]); c=Path(sys.argv[2])
if not p.is_file() or not c.is_file(): raise SystemExit("[STOP] Complete two-segment V3 training first")
r=json.loads(p.read_text())
if r.get("status")!="COMPLETE" or r.get("successful_updates")!=1000 or r.get("config",{}).get("mode")!="v3":
  raise SystemExit("[STOP] V3 baseline not complete")
h=hashlib.sha256()
with c.open("rb") as f:
  for block in iter(lambda:f.read(8<<20),b""):h.update(block)
if h.hexdigest()!=r["best_sha256"]: raise SystemExit("[STOP] V3 checkpoint hash mismatch")
print("[VERIFIED] Segment0+1 V3 baseline complete")
PY
fi
command -v squeue >/dev/null && command -v sbatch >/dev/null || { echo "[STOP] Slurm unavailable" >&2; exit 2; }
MY_QUEUE="$(squeue -h -u "$USER")" || { echo "[STOP] Cannot query own Slurm jobs" >&2; exit 2; }
[[ -z "$MY_QUEUE" ]] || { echo "[STOP] Own queue occupied; one job only"; printf "%s\n" "$MY_QUEUE"; exit 2; }
mkdir -p "$BASE/logs"
ARGS=(--partition=GPU --nodelist=3090node3 --nodes=1 --ntasks=1
    --gres=gpu:3090:1 --cpus-per-task=4 --time=04:00:00
    "--job-name=h9_01_$MODE"
    "--output=$BASE/logs/${MODE}_%j.out" "--error=$BASE/logs/${MODE}_%j.err"
    "--export=ALL,GRID25_ROOT=$GRID25_ROOT,TRAIN_MODE=$MODE")
echo "[Slurm test-only] $MODE on 3090node3: 1x3090 GPU, 4 CPU"
sbatch --test-only "${ARGS[@]}" "$HERE/worker.sbatch"
echo "[Submit] Exactly one manual Grid25 segment0+1 H9 $MODE training job"
JOB="$(sbatch --parsable "${ARGS[@]}" "$HERE/worker.sbatch")" || { echo "[STOP] sbatch rejected" >&2; exit 2; }
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "[UNKNOWN] Ambiguous job result $JOB; do not retry" >&2; exit 2; }
echo "[SEG01 H9 TRAIN SUBMITTED] mode=$MODE job=$JOB output=$OUT log=$BASE/logs/${MODE}_${JOB}.out"
