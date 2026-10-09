#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
R012_ROOT="${R012_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_scratch50k_r012_v1}"
CANDIDATE="${CANDIDATE:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_v4_scratch_v2/formal/final.pt}"
STARTS="${STARTS:-$R012_ROOT/evaluation_fresh_holdout_v1/starts.jsonl}"
BASE="${COMPARE_BASE:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_v4v2_fresh_compare_v1}"
OUT="$BASE/run_$(date +%Y%m%d_%H%M%S)_$$"
[[ -x "$VIS_PYTHON" && -f "$CANDIDATE" && -f "$CANDIDATE/../run.json" && -f "$STARTS" ]] || { echo "[STOP] python/candidate/starts missing";exit 2; }
cd "$REPO"
git diff --quiet && git diff --cached --quiet || { echo "[STOP] tracked working tree changes";exit 2; }
HEAD="$(git rev-parse HEAD)"
"$VIS_PYTHON" - "$CANDIDATE" "$STARTS" <<'PY'
import hashlib,json,sys,torch
from pathlib import Path
cp=Path(sys.argv[1]);starts=Path(sys.argv[2])
r=json.load(open(cp.parent/"run.json"))
c=torch.load(cp,map_location="cpu",weights_only=False)
h=hashlib.sha256()
with cp.open("rb") as f:
    for b in iter(lambda:f.read(1<<20),b""):h.update(b)
assert r["status"]=="COMPLETE" and r["successful_updates"]==50000
assert r["final_sha256"]==h.hexdigest()
assert c["format"]=="care_h9_v4_scratch_v2" and c["completed"] and c["step"]==50000
text=starts.read_text();m=json.load(open(starts.parent/"manifest.json"))
assert hashlib.sha256(text.encode()).hexdigest()==m["starts_sha256"]
print("[preflight] formal V2-50K and frozen starts verified")
PY
mkdir -p "$BASE/logs"
RUN_SCRIPT="$HERE/run.sh"
export VIS_PYTHON R012_ROOT CANDIDATE STARTS OUT RUN_SCRIPT
NODE="${NODE:-}"
NODE_ARGS=()
if [[ -n "$NODE" ]]; then NODE_ARGS=(--nodelist="$NODE"); else NODE_ARGS=(--exclude=3090node1); fi
RAW="$(sbatch --parsable --partition=GPU "${NODE_ARGS[@]}" --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:3090:4 --time="${TIME_LIMIT:-04:00:00}" --output="$BASE/logs/compare_%j.out" --export=ALL "$HERE/worker.sbatch")"
JOB="${RAW%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || exit 2
STATE="$BASE/compare_job_${JOB}.env"
printf 'export JOB_ID=%q\nexport OUT=%q\nexport LOG=%q\n' "$JOB" "$OUT" "$BASE/logs/compare_${JOB}.out" >"$STATE"
cp "$STATE" "$BASE/latest_compare.env"
echo "[submitted] full fresh comparison job=$JOB node=${NODE:-AUTO_EXCLUDING_3090node1}"
echo "[out] $OUT"
