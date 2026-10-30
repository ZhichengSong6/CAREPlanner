#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-status}"
export VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
if [[ "$MODE" == _node ]];then
  : "${G25_CODE:?}" "${G25_JOB:?}" "${G25_ROOT:?}" "${G25_SLOT_BASE:?}" "${SLURM_JOB_ID:?}"
  exec "$VIS_PYTHON" -u "$G25_CODE/run.py" node --job "$G25_JOB" --slot "$G25_SLOT_BASE"
fi
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
BASE=/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner
ROOT="${GRID25_ROOT:-$BASE/outputs/mainline_b/h9_grid25_data_v1}"
REFERENCE="${GRID25_REFERENCE_DATA:-$BASE/src/care_visibility_cdf/data/visibility_yiming_style_grid30_q20000_k500_fovonly.npz}"
export G25_TEST_REPO="$REPO"
[[ -x "$VIS_PYTHON" ]] || { echo '[STOP] viscdf Python missing';exit 2; }
if [[ "$MODE" == test ]];then
  "$VIS_PYTHON" -m unittest discover -s "$HERE" -p 'test_*.py' -v;exit 0
fi
if [[ "$MODE" == submit || "$MODE" == prepare || "$MODE" == resume ]];then
  [[ "$(git -C "$REPO" branch --show-current)" == mainline-b/h9-scratch50k-r012-v1 ]] || { echo '[STOP] wrong branch';exit 2; }
  git -C "$REPO" diff --quiet && git -C "$REPO" diff --cached --quiet || { echo '[STOP] tracked modifications; no reset/stash';exit 2; }
  mkdir -p "$ROOT/logs" "$ROOT/sources"
  exec 9>"$ROOT/.submit.lock";flock -n 9 || { echo '[STOP] concurrent submit';exit 2; }
  "$VIS_PYTHON" -m unittest discover -s "$HERE" -p 'test_*.py' -v
  for f in "$HERE"/*.sh;do bash -n "$f";done
  ID="$("$VIS_PYTHON" - "$HERE" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);m={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir() if f.suffix in ('.py','.sh','.json')}
for f in p.glob('*.py'):compile(f.read_bytes(),str(f),'exec')
print(hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest())
PY
)"
  CODE="$ROOT/sources/$ID"
  if [[ ! -d "$CODE" ]];then
    TMP="$(mktemp -d "$ROOT/sources/.building.XXXXXX")"
    cp "$HERE"/*.py "$HERE"/*.sh "$HERE"/*.json "$HERE"/*.md "$TMP/"
    mv "$TMP" "$CODE"
  fi
  "$VIS_PYTHON" - "$HERE" "$CODE" <<'PY'
from pathlib import Path
import hashlib,sys
def h(p):return {x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in Path(p).iterdir() if x.suffix in ('.py','.sh','.json')}
if h(sys.argv[1])!=h(sys.argv[2]):raise SystemExit('[STOP] code snapshot changed')
PY
  [[ "$MODE" != resume || -f "$ROOT/formal/job.json" ]] || { echo '[STOP] no prepared run to resume';exit 2; }
  "$VIS_PYTHON" "$CODE/run.py" prepare --repo "$REPO" --root "$ROOT" --reference "$REFERENCE"
  printf 'export CODE=%q\n' "$CODE" > "$ROOT/source.env"
  if [[ "$MODE" != prepare ]];then
    "$VIS_PYTHON" "$CODE/run.py" "$MODE" --repo "$REPO" --root "$ROOT" --nodes "${NODES:-3090node2,3090node3}"
  fi
  exit 0
fi
[[ -f "$ROOT/source.env" ]] || { echo '[not prepared]';exit 2; }
source "$ROOT/source.env"
case "$MODE" in
  status|log|ranks|pack) "$VIS_PYTHON" "$CODE/run.py" "$MODE" --root "$ROOT" ;;
  summary|verify) "$VIS_PYTHON" "$CODE/run.py" "$MODE" --job "$ROOT/formal/job.json" ;;
  *) echo 'Usage: run.sh submit|prepare|resume|status|log|ranks|summary|verify|pack|test';exit 2 ;;
esac
