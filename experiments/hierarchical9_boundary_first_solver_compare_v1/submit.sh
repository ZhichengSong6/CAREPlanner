#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
R012_ROOT="${R012_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_scratch50k_r012_v1}"
BF_ROOT="${BF_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_boundary_first_scratch_v1}"
STARTS="${STARTS:-$R012_ROOT/evaluation_fresh_holdout_v1/starts.jsonl}"
BASE="${COMPARE_BASE:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/bf_r1_fov_compare_v1}"
OUT="$BASE/run_$(date +%Y%m%d_%H%M%S)_$$"
[[ -x "$VIS_PYTHON" ]] || { echo "[STOP] Python missing: $VIS_PYTHON"; exit 2; }
[[ -f "$BF_ROOT/formal/run.json" ]] || { echo "[STOP] BF run.json missing"; exit 2; }
[[ -f "$BF_ROOT/formal/final.pt" && -f "$BF_ROOT/formal/best_val.pt" ]] || { echo "[STOP] candidate checkpoints missing"; exit 2; }
[[ -f "$STARTS" ]] || { echo "[STOP] frozen starts missing"; exit 2; }
[[ -f "$R012_ROOT/formal/R1/final.pt" ]] || { echo "[STOP] old R1 baseline checkpoint missing"; exit 2; }
cd "$REPO"
git diff --quiet && git diff --cached --quiet || { echo "[STOP] tracked tree modified"; exit 2; }
"$VIS_PYTHON" -m py_compile "$HERE/evaluate.py" "$HERE/diagnose.py"
bash -n "$HERE/run.sh" "$HERE/worker.sbatch" "$HERE/watch.sh" "$HERE/submit.sh"
"$VIS_PYTHON" - "$BF_ROOT" "$STARTS" <<'PY'
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1])/"formal";starts=Path(sys.argv[2])
run=json.loads((root/"run.json").read_text())
assert run["status"]=="COMPLETE" and run["successful_updates"]==50000
assert run["initialization"]=="random_from_scratch"
assert 1<=run["best_step"]<=50000
for fn,key in (("final.pt","final_sha256"),("best_val.pt","best_val_sha256")):
    p=root/fn
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    guard=json.loads(p.with_suffix(p.suffix+".sha256.json").read_text())
    assert h.hexdigest()==run[key]==guard["sha256"],fn
txt=starts.read_text()
manifest=json.loads((starts.parent/"manifest.json").read_text())
assert hashlib.sha256(txt.encode()).hexdigest()==manifest["starts_sha256"]=="0e4711fc21628fdbb341191ec634415781696800a2386a4ad48e592c34f3198c"
assert sum(bool(x.strip()) for x in txt.splitlines())==1963
print("[preflight] formal BF best/final and unchanged 1963-case starts verified",flush=True)
PY
mkdir -p "$BASE/logs"
[[ ! -e "$OUT" ]] || { echo "[STOP] duplicate output $OUT"; exit 2; }
NODE_ARGS=()
if [[ -n "${NODE:-}" ]]; then NODE_ARGS=(--nodelist="$NODE"); else NODE_ARGS=(--exclude=3090node1); fi
RUN_SCRIPT="$HERE/run.sh"
export VIS_PYTHON R012_ROOT BF_ROOT STARTS OUT RUN_SCRIPT
RAW="$(sbatch --parsable --partition=GPU "${NODE_ARGS[@]}" --nodes=1 --ntasks=1 \
  --gres=gpu:3090:4 --cpus-per-task=16 --time="${TIME_LIMIT:-04:00:00}" \
  --chdir="$REPO" --job-name=bf_r1_fov --export=ALL \
  --output="$BASE/logs/compare_%j.out" --error="$BASE/logs/compare_%j.out" \
  "$HERE/worker.sbatch")"
JOB_ID="${RAW%%;*}"
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo "[STOP] invalid sbatch output: $RAW"; exit 2; }
STATE="$BASE/compare_job_${JOB_ID}.env"
printf 'export JOB_ID=%q\nexport OUT=%q\nexport LOG=%q\n' "$JOB_ID" "$OUT" "$BASE/logs/compare_${JOB_ID}.out" > "$STATE"
cp "$STATE" "$BASE/latest_compare.env"
echo "[submitted] formal paired FOV comparison job=$JOB_ID; four GPUs, fixed 1963 cases"
echo "[out] $OUT"
echo "[state] $STATE"
