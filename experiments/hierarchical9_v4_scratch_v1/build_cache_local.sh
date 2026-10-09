#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ROOT="${V4TR_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_v4_scratch_v1}"
DATA_ROOT="${V4TR_DATA_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/full_dataset_v4_v1/run_20260923_234733_2755825}"
CACHE_ROOT="${V4TR_CACHE_ROOT:-$ROOT/training_cache}"
PY="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
cd "$REPO"
git diff --quiet && git diff --cached --quiet || { echo "[ERROR] save tracked changes before cache build"; exit 2; }
HEAD="$(git rev-parse HEAD)"
[[ -x "$PY" ]] || { echo "[ERROR] python missing: $PY"; exit 2; }
for f in "$DATA_ROOT/manifest.json" "$DATA_ROOT/global_pool_summary.json" "$DATA_ROOT/v3_summary.json" "$DATA_ROOT/v4_summary.json"; do [[ -s "$f" ]] || { echo "[ERROR] missing $f"; exit 2; }; done
[[ ! -e "$CACHE_ROOT" ]] || { echo "[ERROR] cache output exists: $CACHE_ROOT"; exit 2; }
mkdir -p "$ROOT/logs"
LOCK="$ROOT/.cache_local.lock"
exec 9>"$LOCK"
flock -n 9 || { echo "[ERROR] another local cache build is active"; exit 2; }
echo "[cache-local] host=$(hostname) code=$HEAD"
echo "[cache-local] source=$DATA_ROOT"
echo "[cache-local] out=$CACHE_ROOT"
"$PY" -m py_compile "$HERE/cache.py" "$HERE/objective.py" "$HERE/model.py" "$HERE/train.py" "$HERE/test_v4_train.py"
"$PY" "$HERE/test_v4_train.py"
"$PY" -u "$HERE/cache.py" --source "$DATA_ROOT" --out "$CACHE_ROOT" | tee "$ROOT/logs/cache_local.log"
"$PY" - "$CACHE_ROOT" "$HERE/cache.py" <<'PY'
import importlib.util,sys
root,path=sys.argv[1:]
spec=importlib.util.spec_from_file_location("v4cache_verify",path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
c=m.TrainingCache(root,verify_hashes=True)
print("[verified-cache]",c.identity)
print(c.manifest["counts"])
PY
echo "[done] local training cache complete: $CACHE_ROOT"
