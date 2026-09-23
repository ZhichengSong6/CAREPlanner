#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "\${BASH_SOURCE[0]}")" && pwd)"
REPO="\${REPO:-$(cd "$HERE/../.." && pwd)}"
VIS_PYTHON="\${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
R012_ROOT="\${R012_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_scratch50k_r012_v1}"
PAIRED_CACHE="\${PAIRED_CACHE:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_offline_labels_v3/pilot_20260922_115134/paired_cache}"
BASE="\${TRAIN_BASE:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_paired_training_v1}"
OUT="\${TRAIN_OUT:-$BASE/run_$(date +%Y%m%d_%H%M%S)_$$}"
STEPS="\${STEPS:-400}";BATCH_SIZE="\${BATCH_SIZE:-64}";LR="\${LR:-2e-5}";SEED="\${SEED:-20260923}"
[[ -x "$VIS_PYTHON" && -f "$R012_ROOT/formal/R1/final.pt" && -f "$PAIRED_CACHE/dataset_index.json" ]] || { echo "[STOP] python/R1/cache missing";exit 2; }
[[ ! -e "$OUT" ]] || { echo "[STOP] output exists: $OUT";exit 2; }
"$VIS_PYTHON" - "$R012_ROOT/formal/R1/final.pt" "$PAIRED_CACHE/dataset_index.json" <<'PY'
import hashlib,json,pathlib,sys
def h(p):
 x=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):x.update(b)
 return x.hexdigest()
if h(sys.argv[1])!="4f395926fa79c29474be8748cef4733ec400d155cd8fadb76c632c2838864002":raise SystemExit("wrong R1 SHA")
d=json.loads(pathlib.Path(sys.argv[2]).read_text())
if not d.get("complete") or not d.get("audit_complete") or d.get("gradient_policy")!="FD_PASS_ONLY":raise SystemExit("paired cache incomplete")
print("[precheck] R1 SHA and paired cache PASS")
PY
mkdir -p "$BASE/logs"
TRAIN_SCRIPT="$HERE/run.sh";export REPO VIS_PYTHON R012_ROOT PAIRED_CACHE BASE OUT TRAIN_SCRIPT STEPS BATCH_SIZE LR SEED
RAW="$(sbatch --parsable --partition=GPU --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:4 --nodelist="\${NODE:-3090node1}" --time="\${TIME_LIMIT:-06:00:00}" --output="$BASE/logs/train_%j.out" --export=ALL "$HERE/worker.sbatch")"
JOB="\${RAW%%;*}";[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "[ERROR] sbatch=$RAW";exit 2; }
STATE="$BASE/train_job_\${JOB}.env"
printf 'export JOB_ID=%q\nexport OUT=%q\nexport LOG=%q\n' "$JOB" "$OUT" "$BASE/logs/train_\${JOB}.out" >"$STATE"
cp "$STATE" "$BASE/latest_train.env"
echo "[submitted] job=$JOB";echo "[out] $OUT";echo "[state] $STATE"
