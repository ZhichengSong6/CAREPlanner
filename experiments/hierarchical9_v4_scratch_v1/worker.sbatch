#!/usr/bin/env bash
set -euo pipefail
: "${V4TR_ROOT:?}" "${V4TR_REPO:?}" "${V4TR_CODE_SHA:?}" "${V4TR_MODE:?}"
REPO=$(realpath -e "$V4TR_REPO")
ROOT=$(mkdir -p "$V4TR_ROOT" && realpath -e "$V4TR_ROOT")
DIR="$REPO/experiments/hierarchical9_v4_scratch_v1"
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$V4TR_CODE_SHA" ]] || { echo "[ERROR] code SHA changed after submission"; exit 3; }
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"; CONDA_ENV="${CONDA_ENV:-viscdf}"
set +u; source "$CONDA_SH"; conda activate "$CONDA_ENV"; set -u
cd "$REPO"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" PYTHONUNBUFFERED=1
DATA_ROOT="${V4TR_DATA_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/full_dataset_v4_v1/run_20260923_234733_2755825}"
CACHE_ROOT="${V4TR_CACHE_ROOT:-$ROOT/training_cache}"
for f in "$DIR/cache.py" "$DIR/objective.py" "$DIR/model.py" "$DIR/train.py" "$DIR/test_v4_train.py"; do [[ -s "$f" ]] || { echo "[ERROR] missing $f"; exit 2; }; done
python -m py_compile "$DIR/cache.py" "$DIR/objective.py" "$DIR/model.py" "$DIR/train.py" "$DIR/test_v4_train.py"
python "$DIR/test_v4_train.py"
if [[ "$V4TR_MODE" == cache ]]; then
  [[ -f "$DATA_ROOT/global_pool_summary.json" && -f "$DATA_ROOT/v3_summary.json" && -f "$DATA_ROOT/v4_summary.json" ]] || { echo "[ERROR] full V4 data incomplete"; exit 2; }
  [[ ! -e "$CACHE_ROOT" ]] || { echo "[ERROR] cache output exists: $CACHE_ROOT"; exit 2; }
  python -u "$DIR/cache.py" --source "$DATA_ROOT" --out "$CACHE_ROOT"
  python - "$CACHE_ROOT" "$DIR/cache.py" <<'PY'
import importlib.util,sys
root,path=sys.argv[1:]
spec=importlib.util.spec_from_file_location("v4cache_verify",path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
c=m.TrainingCache(root,verify_hashes=True)
print("[verified-cache]",c.identity,c.manifest["counts"],flush=True)
PY
  exit 0
fi
[[ -f "$CACHE_ROOT/manifest.json" ]] || { echo "[ERROR] build training cache first"; exit 2; }
python - "$CACHE_ROOT" "$DIR/cache.py" <<'PY'
import importlib.util,sys
root,path=sys.argv[1:]
spec=importlib.util.spec_from_file_location("v4cache_verify",path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
c=m.TrainingCache(root,verify_hashes=True);print("[cache]",c.identity,c.manifest["counts"],flush=True)
PY
python - <<'PY'
import torch
assert torch.cuda.is_available()
assert torch.cuda.device_count()==4,torch.cuda.device_count()
print("[gpu]",[torch.cuda.get_device_name(i) for i in range(4)],flush=True)
PY
if [[ "$V4TR_MODE" == smoke ]]; then
  OUT="$ROOT/smoke"; STEPS=2; LOG=1; VAL=1; SAVE=1; EXTRA=()
elif [[ "$V4TR_MODE" == train ]]; then
  OUT="$ROOT/formal"; STEPS=50000; LOG=100; VAL=1000; SAVE=5000; EXTRA=()
elif [[ "$V4TR_MODE" == resume ]]; then
  OUT="$ROOT/formal"; STEPS=50000; LOG=100; VAL=1000; SAVE=5000; [[ -s "$OUT/latest.pt" ]] || { echo "[ERROR] missing $OUT/latest.pt"; exit 2; }; EXTRA=(--resume "$OUT/latest.pt")
else
  echo "[ERROR] V4TR_MODE cache|smoke|train|resume"; exit 2
fi
if [[ "$V4TR_MODE" != resume ]]; then [[ ! -e "$OUT" || -z "$(ls -A "$OUT" 2>/dev/null || true)" ]] || { echo "[ERROR] refusing overwrite $OUT"; exit 2; }; fi
echo "[launch] mode=$V4TR_MODE steps=$STEPS out=$OUT code=$V4TR_CODE_SHA host=$(hostname)"
nvidia-smi
python -m torch.distributed.run --standalone --nproc_per_node=4 "$DIR/train.py" \
  --cache "$CACHE_ROOT" --out "$OUT" --code-sha "$V4TR_CODE_SHA" --steps "$STEPS" \
  --seed 0 --stream-seed 261009 --lr 1e-3 \
  --batch-global 8192 --batch-v3 512 --batch-v4 8192 --batch-boundary 2048 \
  --val-batch-global 8192 --val-batch-v3 2048 --val-batch-v4 8192 --val-batch-boundary 2048 \
  --log-every "$LOG" --val-every "$VAL" --save-every "$SAVE" "${EXTRA[@]}"
python - "$OUT/run.json" "$OUT/final.pt" "$STEPS" "$V4TR_CODE_SHA" <<'PY'
import hashlib,json,sys
runp,finalp,n,sha=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4]
r=json.load(open(runp));assert r["status"]=="COMPLETE" and r["successful_updates"]==n and r["code_sha"]==sha,r
h=hashlib.sha256()
with open(finalp,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
assert h.hexdigest()==r["final_sha256"],(h.hexdigest(),r["final_sha256"])
print("[verified-train]",n,r["final_sha256"],flush=True)
PY
if [[ "$V4TR_MODE" == smoke ]]; then
  python - "$ROOT/smoke/run.json" "$ROOT/smoke_manifest.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));json.dump({"status":"COMPLETE","code_sha":r["code_sha"],"final_sha256":r["final_sha256"],"successful_updates":r["successful_updates"]},open(sys.argv[2],"w"),indent=2);open(sys.argv[2],"a").write("\n")
PY
fi
