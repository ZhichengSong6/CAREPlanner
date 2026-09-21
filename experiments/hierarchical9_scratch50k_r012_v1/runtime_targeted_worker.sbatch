#!/usr/bin/env bash
set -euo pipefail
: "${R012_CODE_REPO:?}" "${R012_CODE_SHA:?}" "${R012_ROOT:?}" "${V1_CHECKPOINT:?}" "${SCALAR_CHECKPOINT:?}"
REPO=$(realpath -e "$R012_CODE_REPO")
ROOT=$(realpath -e "$R012_ROOT")
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$R012_CODE_SHA" ]] || { echo '[ERROR] code SHA changed'; exit 3; }
VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
[[ -x "$VIS_PYTHON" ]] || { echo "[ERROR] missing VIS_PYTHON=$VIS_PYTHON"; exit 2; }
"$VIS_PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() == 4, torch.cuda.device_count()
print("[gpus]", [torch.cuda.get_device_name(i) for i in range(4)], flush=True)
PY
R1="$ROOT/formal/R1/final.pt"
V1=$(realpath -e "$V1_CHECKPOINT")
SCALAR=$(realpath -e "$SCALAR_CHECKPOINT")
OUT="$ROOT/runtime_targeted_v1_r1"
if [[ -e "$OUT" ]]; then
  if [[ -s "$OUT/targeted_compare.json" ]] && python3 - "$OUT/targeted_compare.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
raise SystemExit(0 if r.get("valid_diagnostic") is True else 1)
PY
  then
    echo "[ERROR] completed targeted qualification already exists: $OUT" >&2
    exit 2
  fi
  ARCHIVE="$ROOT/.runtime_targeted_v1_r1_failed_$(date +%Y%m%d_%H%M%S)"
  mv "$OUT" "$ARCHIVE"
  echo "[archive] previous incomplete targeted output -> $ARCHIVE"
fi
cd "$REPO"
OUT="$OUT" REPO="$REPO" V1_CHECKPOINT="$V1" R1_CHECKPOINT="$R1" SCALAR_CHECKPOINT="$SCALAR" VIS_PYTHON="$VIS_PYTHON" DEVICE=cuda TARGETED_PARALLEL_4GPU=true \
  bash scripts/run_phase_e_v1_r1_targeted_qualification.sh
python3 - "$OUT/targeted_compare.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); assert r.get('valid_diagnostic') is True,r
print('[done] server targeted qualification VALID')
PY
