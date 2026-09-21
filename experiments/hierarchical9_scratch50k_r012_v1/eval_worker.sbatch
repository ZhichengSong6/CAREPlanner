#!/usr/bin/env bash
set -euo pipefail
: "${R012_ROOT:?}" "${E012_REFERENCE_ROOT:?}" "${R012_CODE_REPO:?}" "${R012_CODE_SHA:?}"
REPO=$(realpath -e "$R012_CODE_REPO")
ROOT=$(realpath -e "$R012_ROOT")
REF=$(realpath -e "$E012_REFERENCE_ROOT")
DIR="$REPO/experiments/hierarchical9_scratch50k_r012_v1"
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$R012_CODE_SHA" ]] || { echo '[ERROR] code SHA changed after submission'; exit 3; }

CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-viscdf}"
set +u
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u
cd "$REPO"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

python -m py_compile "$DIR/evaluate_unified.py"

# Fail fast on every dynamic import used by the evaluator.  old.module() requires
# the import name to equal the actual filename, so verify both importability and
# resolved __file__ before loading any checkpoint or running any solve.
python - <<'PY'
from pathlib import Path
import os, sys
repo=Path(os.environ['R012_CODE_REPO']).resolve()
rdir=repo/'experiments/hierarchical9_scratch50k_r012_v1'
e012=repo/'experiments/hierarchical9_e012_end2end_v1'
abc=repo/'experiments/hierarchical9_abc_capacity_routing_v1'
for p in (str(rdir),str(abc),str(e012)):
    if p not in sys.path: sys.path.insert(0,p)
import e012_protocol as proto
local=proto._load('r012_eval_preflight_local_model',rdir/'model.py')
assert hasattr(local,'build_model')
compat=proto._load('r012_eval_preflight_compat',abc/'eval_compat.py')
assert hasattr(compat,'legacy_compatible')
checks=[
    ('evaluate_pair',proto.abc.p2.LEGACY),
    ('evaluate_p2',proto.abc.P2_DIR),
    ('neighborhood',proto.P3_DIR),
    ('train_signed_visibility_cdf_pairwise_replace',proto.old.SCRIPTS),
    ('oracle',proto.old.AUDIT),
    ('core',proto.old.AUDIT),
    ('runtime_probe',proto.old.AUDIT),
    ('audit',proto.old.AUDIT),
]
for name,directory in checks:
    mod=proto.old.module(name,directory)
    expected=(Path(directory)/(name+'.py')).resolve()
    got=Path(mod.__file__).resolve()
    assert got==expected,(name,got,expected)
    print('[import-ok]',name,got,flush=True)
print('[done] unified-eval import preflight OK',flush=True)
PY

python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.device_count() >= 1
print('[gpu]', torch.cuda.get_device_name(0), flush=True)
PY
echo "[eval-launch] unified R012 root=$ROOT ref=$REF code=$R012_CODE_SHA host=$(hostname)"
nvidia-smi

python "$DIR/evaluate_unified.py" --r012-root "$ROOT" --reference-root "$REF"

python - "$ROOT/evaluation_unified_dev/manifest.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
assert m.get('status')=='COMPLETE',m
assert m.get('r012_training_streams')=='MATCH',m
assert m.get('r012_training_stream_updates')==50000,m
assert m.get('evaluated_models')==['V1','P0','E0','E1','R0','R1','R2'],m
print('[done] r012_unified_eval_workflow_complete',flush=True)
PY
