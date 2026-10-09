#!/usr/bin/env bash
set -euo pipefail
: "${V4TR2_ROOT:?}" "${V4TR2_REPO:?}" "${V4TR2_CODE_SHA:?}" "${V4TR2_MODE:?}"
REPO=$(realpath -e "$V4TR2_REPO")
ROOT=$(mkdir -p "$V4TR2_ROOT" && realpath -e "$V4TR2_ROOT")
DIR="$REPO/experiments/hierarchical9_v4_scratch_v2"
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$V4TR2_CODE_SHA" ]] || { echo "[ERROR] code SHA changed after submission"; exit 3; }
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}";CONDA_ENV="${CONDA_ENV:-viscdf}"
set +u;source "$CONDA_SH";conda activate "$CONDA_ENV";set -u
cd "$REPO"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" PYTHONUNBUFFERED=1
CACHE_ROOT="${V4TR2_CACHE_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_v4_scratch_v1/training_cache}"
EXPECTED_CACHE="060eef93a0030800c8ef2265258c7dff850944ab9c5df52382adb98cb506f7cc"
for f in "$DIR/cache.py" "$DIR/objective.py" "$DIR/model.py" "$DIR/train.py" "$DIR/test_v4_train.py";do [[ -s "$f" ]]||{ echo "[ERROR] missing $f";exit 2;};done
python -m py_compile "$DIR/cache.py" "$DIR/objective.py" "$DIR/model.py" "$DIR/train.py" "$DIR/test_v4_train.py"
python "$DIR/test_v4_train.py"
python - "$CACHE_ROOT" "$DIR/cache.py" "$EXPECTED_CACHE" <<'PY'
import importlib.util,sys
root,path,expected=sys.argv[1:]
spec=importlib.util.spec_from_file_location("v4cache_verify",path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
c=m.TrainingCache(root,verify_hashes=True)
assert c.identity==expected,(c.identity,expected)
print("[cache]",c.identity,c.manifest["counts"],flush=True)
PY
python - <<'PY'
import torch
assert torch.cuda.is_available()
assert torch.cuda.device_count()==4,torch.cuda.device_count()
print("[gpu]",[torch.cuda.get_device_name(i) for i in range(4)],flush=True)
PY
case "$V4TR2_MODE" in
 smoke) OUT="$ROOT/smoke";STEPS=2;LOG=1;VAL=1;SAVE=1;EXTRA=();;
 pilot) OUT="$ROOT/pilot";STEPS=500;LOG=25;VAL=100;SAVE=500;EXTRA=();;
 train) OUT="$ROOT/formal";STEPS=50000;LOG=100;VAL=1000;SAVE=5000;EXTRA=();;
 resume) OUT="$ROOT/formal";STEPS=50000;LOG=100;VAL=1000;SAVE=5000;[[ -s "$OUT/latest.pt" ]]||{ echo "[ERROR] missing $OUT/latest.pt";exit 2;};EXTRA=(--resume "$OUT/latest.pt");;
 *) echo "[ERROR] V4TR2_MODE smoke|pilot|train|resume";exit 2;;
esac
if [[ "$V4TR2_MODE" != resume ]];then [[ ! -e "$OUT" || -z "$(ls -A "$OUT" 2>/dev/null||true)" ]]||{ echo "[ERROR] refusing overwrite $OUT";exit 2;};fi
echo "[launch] v2 mode=$V4TR2_MODE steps=$STEPS out=$OUT code=$V4TR2_CODE_SHA host=$(hostname)"
nvidia-smi
python -m torch.distributed.run --standalone --nproc_per_node=4 "$DIR/train.py" \
  --cache "$CACHE_ROOT" --out "$OUT" --code-sha "$V4TR2_CODE_SHA" --steps "$STEPS" \
  --seed 0 --stream-seed 261009 --lr 1e-3 \
  --batch-global 8192 --batch-v3 512 --batch-v4 8192 --batch-boundary 2048 \
  --val-batch-global 8192 --val-batch-v3 2048 --val-batch-v4 8192 --val-batch-boundary 2048 \
  --log-every "$LOG" --val-every "$VAL" --save-every "$SAVE" "${EXTRA[@]}"
python - "$OUT/run.json" "$OUT/final.pt" "$STEPS" "$V4TR2_CODE_SHA" <<'PY'
import hashlib,json,sys
runp,finalp,n,sha=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4]
r=json.load(open(runp));assert r["status"]=="COMPLETE" and r["successful_updates"]==n and r["code_sha"]==sha,r
h=hashlib.sha256()
with open(finalp,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
assert h.hexdigest()==r["final_sha256"],(h.hexdigest(),r["final_sha256"])
print("[verified-train]",n,r["final_sha256"],flush=True)
PY
if [[ "$V4TR2_MODE" == smoke || "$V4TR2_MODE" == pilot ]];then
  NAME="$V4TR2_MODE";python - "$OUT/run.json" "$ROOT/${NAME}_manifest.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));m={"status":"COMPLETE","code_sha":r["code_sha"],"final_sha256":r["final_sha256"],"successful_updates":r["successful_updates"],"cache_identity":r["cache_identity"]}
open(sys.argv[2],"w").write(json.dumps(m,indent=2)+"\n")
PY
fi
