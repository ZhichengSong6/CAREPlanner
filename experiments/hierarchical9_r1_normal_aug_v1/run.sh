#!/usr/bin/env bash
# One immutable scratch-50k augmented run, then optional frozen fixed-solver comparison.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
ROOT="${R1A_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_r1_normal_aug_v1}"
R012="${R1A_R012:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_scratch50k_r012_v1}"
CACHE="${R1A_CACHE:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_v4_scratch_v1/training_cache}"
DATA="${R1A_DATA:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/src/care_visibility_cdf/data/visibility_yiming_style_grid30_q20000_k500_fovonly.npz}"
URDF="${R1A_URDF:-$REPO/src/arm_description/urdf/Arm.urdf}"
STARTS="${R1A_STARTS:-$R012/evaluation_fresh_holdout_v1/starts.jsonl}"
VIS_PYTHON="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
MODE="${1:-status}"
OUT="$ROOT/formal"
mkdir -p "$ROOT/logs" "$ROOT/sources"
ROOT="$(realpath -e "$ROOT")"
OUT="$ROOT/formal"
export R1A_REPO="$REPO" R1A_ROOT="$ROOT" R1A_R012="$R012" R1A_CACHE="$CACHE"
export R1A_DATA="$DATA" R1A_URDF="$URDF" R1A_STARTS="$STARTS" VIS_PYTHON
export PYTHONDONTWRITEBYTECODE=1

usage() { echo 'Usage: bash run.sh train|resume|eval|verify|status|log|follow|summary'; }

if [[ "$MODE" == status || "$MODE" == log || "$MODE" == follow || "$MODE" == summary ]]; then
  STATE="$ROOT/latest.env"
  [[ -f "$STATE" ]] || { echo '[not submitted]'; exit 0; }
  source "$STATE"
  case "$MODE" in
    status)
      squeue -j "$JOB_ID" -o '%.18i %.12T %.12M %.35R' 2>/dev/null || true
      sacct -X -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed,NodeList || true
      echo "[mode] $JOB_MODE [out] $JOB_OUT [log] $LOG";;
    log)
      [[ -f "$LOG" ]] && tail -n 120 "$LOG" || echo '[no Slurm log yet]';;
    follow)
      echo 'Ctrl+C stops tail only, does not cancel Slurm job.'
      tail --retry -n 120 -F "$LOG";;
    summary)
      if [[ -f "$ROOT/formal/run.json" ]]; then
        "$VIS_PYTHON" - "$ROOT/formal/run.json" <<'PY'
import json,sys
a=json.load(open(sys.argv[1]))
for k in ("status","successful_updates","training_stream_sha256","baseline_stream_exact_match","final_sha256","best_val","variant","boundary_groups"):
    print(k,":",a.get(k))
PY
      fi
      if [[ -f "$ROOT/evaluation_fixed_1963/summary.md" ]]; then
        cat "$ROOT/evaluation_fixed_1963/summary.md"
      else
        echo '[solver evaluation not yet completed]'
      fi;;
  esac
  exit 0
fi

[[ "$MODE" == train || "$MODE" == resume || "$MODE" == eval || "$MODE" == verify ]] || { usage; exit 2; }
[[ -x "$VIS_PYTHON" ]] || { echo "[STOP] Python unavailable: $VIS_PYTHON"; exit 2; }
command -v flock >/dev/null || { echo '[STOP] flock unavailable'; exit 2; }
exec 9>"$ROOT/.submit.lock"
flock -n 9 || { echo '[STOP] simultaneous submission refused'; exit 2; }

code_identity() {
  "$VIS_PYTHON" - "$1" <<'PY'
from pathlib import Path
import hashlib,json,sys
p=Path(sys.argv[1])
items={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in p.iterdir()
       if x.is_file() and x.suffix in (".py",".sh",".json")}
print(hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest())
PY
}
ID="$(code_identity "$HERE")"
SOURCE="$ROOT/sources/$ID"
if [[ ! -d "$SOURCE" ]]; then
  TMP="$ROOT/sources/.tmp.${BASHPID}"
  [[ ! -e "$TMP" ]] || { echo '[STOP] temporary source already exists'; exit 2; }
  mkdir "$TMP"
  cp "$HERE"/*.py "$HERE"/*.sh "$HERE"/*.json "$TMP/"
  mv "$TMP" "$SOURCE"
fi
[[ "$(code_identity "$SOURCE")" == "$ID" ]] || { echo '[STOP] frozen package corrupted'; exit 2; }
export R1A_SOURCE="$SOURCE" R1A_CODE_ID="$ID"
ARGS=(--r012-root "$R012" --cache "$CACHE" --data "$DATA" --urdf "$URDF" --out "$OUT")

if [[ "$MODE" == verify ]]; then
  "$VIS_PYTHON" "$SOURCE/train.py" verify "${ARGS[@]}"
  exit 0
fi

[[ "$(git -C "$REPO" branch --show-current)" == 'mainline-b/h9-scratch50k-r012-v1' ]] || { echo '[STOP] wrong branch'; exit 2; }
git -C "$REPO" diff --quiet && git -C "$REPO" diff --cached --quiet || {
  echo '[STOP] tracked worktree modified; no automatic reset'; exit 2;
}
if [[ -f "$ROOT/latest.env" ]]; then
  source "$ROOT/latest.env"
  ACTIVE="$(squeue -h -j "$JOB_ID" -o '%T' 2>/dev/null || true)"
  [[ -z "$ACTIVE" ]] || { echo "[STOP] job $JOB_ID still $ACTIVE"; exit 2; }
  STATE="$(sacct -X -n -j "$JOB_ID" --format=State 2>/dev/null | awk 'NF{print $1;exit}' || true)"
  case "$STATE" in
    COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL) ;;
    *) echo "[STOP] previous job state unresolved: $STATE"; exit 2;;
  esac
fi

if [[ "$MODE" == train || "$MODE" == resume ]]; then
  if [[ "$MODE" == train ]]; then
    [[ ! -e "$OUT" || -z "$(ls -A "$OUT" 2>/dev/null || true)" ]] || {
      echo '[STOP] prior run exists; never overwrite, use resume only if unfinished'; exit 2; }
  else
    [[ -f "$OUT/latest.pt" ]] || { echo '[STOP] no resumable latest.pt'; exit 2; }
    [[ ! -f "$OUT/final.pt" ]] || { echo '[STOP] already finished'; exit 2; }
  fi
  # CPU checks are NOT a GPU pilot. Full 50k is the only submitted train.
  "$VIS_PYTHON" -m unittest discover -s "$SOURCE" -p 'test_*.py' -v
  "$VIS_PYTHON" "$SOURCE/train.py" preflight "${ARGS[@]}"
  TARGET="$OUT"
  LIMIT="${TIME_LIMIT:-3-00:00:00}"
else
  "$VIS_PYTHON" "$SOURCE/train.py" verify "${ARGS[@]}"
  [[ -f "$STARTS" ]] || { echo '[STOP] frozen solver starts unavailable'; exit 2; }
  TARGET="$ROOT/evaluation_fixed_1963"
  [[ ! -e "$TARGET" ]] || { echo "[STOP] existing evaluation: $TARGET"; exit 2; }
  LIMIT="${TIME_LIMIT:-04:00:00}"
fi

[[ "$MODE" != train && "$MODE" != resume ]] || command -v sbatch >/dev/null
command -v sbatch >/dev/null || { echo '[STOP] sbatch unavailable'; exit 2; }
NODE_ARGS=(--exclude=3090node1)
[[ -z "${NODE:-}" ]] || NODE_ARGS=(--nodelist="$NODE")
export R1A_MODE="$MODE" R1A_OUT="$TARGET"
RAW="$(sbatch --parsable --partition=GPU --nodes=1 --ntasks=1 --gres=gpu:3090:4 \
  --cpus-per-task=16 --time="$LIMIT" "${NODE_ARGS[@]}" --chdir="$REPO" \
  --job-name="r1_normal_${MODE}" --export=ALL \
  --output="$ROOT/logs/${MODE}_%j.out" --error="$ROOT/logs/${MODE}_%j.out" \
  "$SOURCE/worker.sh")"
JOB_ID="${RAW%%;*}"
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo "[STOP] unparsed submission: $RAW"; exit 2; }
LOG="$ROOT/logs/${MODE}_${JOB_ID}.out"
printf 'export JOB_ID=%q\nexport JOB_MODE=%q\nexport JOB_OUT=%q\nexport LOG=%q\n' \
  "$JOB_ID" "$MODE" "$TARGET" "$LOG" > "$ROOT/job_${JOB_ID}.env"
cp "$ROOT/job_${JOB_ID}.env" "$ROOT/latest.env"
echo "[submitted] mode=$MODE job=$JOB_ID source=$ID gpus=4"
echo "[out] $TARGET"
echo "[log] $LOG"
