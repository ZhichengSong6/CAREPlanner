#!/usr/bin/env bash
set -euo pipefail
: "${R012_ROOT:?}" "${R012_CODE_REPO:?}" "${R012_CODE_SHA:?}" "${R012_MODE:?}" "${R012_ARM:?}"
REPO=$(realpath -e "$R012_CODE_REPO")
ROOT=$(mkdir -p "$R012_ROOT" && realpath -e "$R012_ROOT")
DIR="$REPO/experiments/hierarchical9_scratch50k_r012_v1"
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$R012_CODE_SHA" ]] || { echo '[ERROR] code SHA changed after submission'; exit 3; }
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"; CONDA_ENV="${CONDA_ENV:-viscdf}"
set +u; source "$CONDA_SH"; conda activate "$CONDA_ENV"; set -u
cd "$REPO"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" PYTHONUNBUFFERED=1
DATA="${R012_DATA:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/src/care_visibility_cdf/data/visibility_yiming_style_grid30_q20000_k500_fovonly.npz}"
URDF="${R012_URDF:-$REPO/src/arm_description/urdf/Arm.urdf}"
for f in "$DATA" "$URDF" "$DIR/model.py" "$DIR/train.py" "$DIR/test_r012.py"; do [[ -s "$f" ]] || { echo "[ERROR] missing $f"; exit 2; }; done
python -m py_compile "$DIR/model.py" "$DIR/train.py" "$DIR/test_r012.py"
python "$DIR/test_r012.py"
python - <<'PY'
import torch
assert torch.cuda.is_available()
assert torch.cuda.device_count()==4, torch.cuda.device_count()
print('[gpu]', [torch.cuda.get_device_name(i) for i in range(4)], flush=True)
PY

echo "[launch] R012 mode=$R012_MODE arm=$R012_ARM root=$ROOT code=$R012_CODE_SHA host=$(hostname)"
nvidia-smi

if [[ "$R012_MODE" == smoke ]]; then
  STEPS=2; LOG_EVERY=1; VAL_EVERY=1; SAVE_EVERY=1; ARMS=(R0 R1 R2); BASE="$ROOT/smoke"
elif [[ "$R012_MODE" == train ]]; then
  STEPS=50000; LOG_EVERY=100; VAL_EVERY=1000; SAVE_EVERY=5000; ARMS=("$R012_ARM"); BASE="$ROOT/formal"
else
  echo '[ERROR] R012_MODE must be smoke or train'; exit 2
fi
mkdir -p "$BASE"

for ARM in "${ARMS[@]}"; do
  OUT="$BASE/$ARM"
  EXTRA=()
  if [[ "${R012_RESUME:-0}" == 1 ]]; then
    [[ "$R012_MODE" == train ]] || { echo '[ERROR] resume only valid for formal train'; exit 2; }
    [[ -s "$OUT/latest.pt" ]] || { echo "[ERROR] no $OUT/latest.pt to resume"; exit 2; }
    EXTRA+=(--resume-training "$OUT/latest.pt")
  else
    [[ ! -e "$OUT" || -z "$(ls -A "$OUT" 2>/dev/null || true)" ]] || { echo "[ERROR] refusing overwrite $OUT"; exit 2; }
  fi
  echo "[arm-start] $ARM steps=$STEPS out=$OUT resume=${R012_RESUME:-0}"
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$DIR/train.py" --arm "$ARM" --repo "$REPO" --data "$DATA" --urdf "$URDF" --out-dir "$OUT" \
    --steps "$STEPS" --seed 0 --stream-seed 190915 --lr 1e-3 --amp fp16 \
    --global-batch-x 4000 --batch-q 100 --microbatch-x 250 \
    --val-global-batch-x 512 --val-batch-q 100 --val-microbatch-x 128 --decode-x-chunk 64 \
    --weight-sdf 5.0 --weight-grad 0.1 --weight-eikonal 0.01 --weight-tension 0.01 \
    --weight-union-objective 1.0 --weight-sensor-objective 1.0 --weight-consistency 0.1 \
    --log-every "$LOG_EVERY" --val-every "$VAL_EVERY" --save-every "$SAVE_EVERY" \
    "${EXTRA[@]}"
  python - "$OUT/run.json" "$STEPS" <<'PY'
import json,sys
p,n=sys.argv[1],int(sys.argv[2]);r=json.load(open(p))
assert r['status']=='COMPLETE',r
assert r['successful_updates']==n,r
assert r['training_stream_updates']==n,r
print('[arm-complete]',r['arm'],'updates=',n,'stream=',r['training_stream_sha256'],flush=True)
PY
done

if [[ "$R012_MODE" == smoke ]]; then
python - "$BASE" "$ROOT/smoke_manifest.json" <<'PY'
import json,sys
from pathlib import Path
base=Path(sys.argv[1]);out=Path(sys.argv[2]); rows={a:json.load(open(base/a/'run.json')) for a in ('R0','R1','R2')}
h={r['training_stream_sha256'] for r in rows.values()}
if len(h)!=1: raise SystemExit(f'smoke streams differ: {h}')
m={'status':'COMPLETE','mode':'smoke','arms':list(rows),'training_streams':'MATCH','training_stream_sha256':next(iter(h))}
out.write_text(json.dumps(m,indent=2)+'\n')
print('[done] r012_smoke_complete stream='+next(iter(h)),flush=True)
PY
else
  echo "[done] r012_formal_arm_complete arm=$R012_ARM"
fi
