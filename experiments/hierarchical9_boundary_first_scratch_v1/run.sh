#!/usr/bin/env bash
# One formal submission; inspection commands never submit jobs.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${BF_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_boundary_first_scratch_v1}"
CACHE="${BF_CACHE:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_v4_scratch_v1/training_cache}"
PY="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
MODE="${1:-status}"
OUT="$ROOT/formal"
STATE="$ROOT/latest.env"

if [[ "$MODE" == _worker ]]; then
  : "${BF_CODE_ID:?}" "${BF_MODE:?}"
  [[ -x "$PY" ]] || { echo "[STOP] Python unavailable: $PY"; exit 2; }
  trap 'rc=$?; printf "exit_code=%s\njob_id=%s\n" "$rc" "${SLURM_JOB_ID:-unknown}" > "$ROOT/worker_exit.txt"' EXIT
  exec 8>"$ROOT/writer.lock"
  flock -n 8 || { echo '[STOP] another writer holds this experiment'; exit 2; }
  [[ "$("$PY" "$HERE/train.py" identity)" == "$BF_CODE_ID" ]] || { echo '[STOP] queued source changed'; exit 2; }
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
  "$PY" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count()==4, torch.cuda.device_count(); print("[GPU]",[torch.cuda.get_device_name(i) for i in range(4)],flush=True)'
  EXTRA=()
  if [[ "$BF_MODE" == resume ]]; then EXTRA=(--resume "$OUT/latest.pt"); fi
  echo "[worker] host=$(hostname) source=$HERE out=$OUT mode=$BF_MODE"
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 --max_restarts=0 \
    "$HERE/train.py" train --cache "$CACHE" --out "$OUT" "${EXTRA[@]}"
  "$PY" "$HERE/train.py" verify --out "$OUT"
  exit 0
fi

case "$MODE" in
 status|log|follow|summary|verify)
  if [[ "$MODE" == verify ]]; then "$PY" "$HERE/train.py" verify --out "$OUT"; exit 0; fi
  if [[ "$MODE" == summary ]]; then
    "$PY" - "$OUT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
for name in ('run.json','validation_latest.json','validation_best.json'):
    f=p/name
    print('\n========== '+name+' ==========')
    if f.is_file():
        x=json.loads(f.read_text())
        if name=='run.json':
            x={k:x.get(k) for k in ('status','initialization','successful_updates','best_step','best_score','cumulative_rows','final_sha256','best_val_sha256','solver_evaluation')}
        print(json.dumps(x,indent=2))
    else: print('[not generated yet]',f)
PY
    exit 0
  fi
  [[ -f "$STATE" ]] || { echo "[not submitted] $STATE"; exit 0; }
  source "$STATE"
  if [[ "$MODE" == status ]]; then
    squeue -j "$JOB_ID" -o '%.18i %.12T %.12M %.40R' 2>&1 || true
    sacct -X -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed,NodeList || true
    echo "[log] $LOG"
  elif [[ "$MODE" == follow ]]; then
    echo 'Ctrl+C stops viewing only, not the job.'
    tail --retry -n 100 -F "$LOG"
  else
    if [[ -f "$LOG" ]]; then tail -n 120 "$LOG"; else echo "[log not created yet] $LOG"; fi
  fi
  exit 0;;
 train|resume) ;;
 *) echo 'Usage: bash run.sh train|resume|status|log|follow|summary|verify'; exit 2;;
esac

[[ -x "$PY" ]] || { echo "[STOP] Python missing: $PY"; exit 2; }
[[ -f "$CACHE/manifest.json" ]] || { echo "[STOP] existing training cache missing: $CACHE"; exit 2; }
mkdir -p "$ROOT/logs" "$ROOT/sources"
ROOT="$(realpath -e "$ROOT")"; OUT="$ROOT/formal"; STATE="$ROOT/latest.env"
exec 9>"$ROOT/submit.lock"
flock -n 9 || { echo '[STOP] concurrent submission'; exit 2; }
if [[ -f "$STATE" ]]; then
  OLD_JOB="$(bash -c 'source "$1"; printf "%s" "$JOB_ID"' _ "$STATE")"
  [[ "$OLD_JOB" =~ ^[0-9]+$ ]] || { echo '[STOP] invalid existing job state'; exit 2; }
  ACTIVE="$(squeue -h -j "$OLD_JOB" -o '%T' 2>/dev/null || true)"
  [[ -z "$ACTIVE" ]] || { echo "[STOP] job $OLD_JOB active ($ACTIVE); use run.sh log"; exit 2; }
  OLD_STATUS="$(sacct -X -n -j "$OLD_JOB" --format=State --noheader | awk 'NF{print $1;exit}')"
  case "$OLD_STATUS" in
    COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL) ;;
    *) echo "[STOP] cannot confirm previous job ended: $OLD_STATUS"; exit 2;;
  esac
fi
if [[ "$MODE" == train ]]; then
  [[ ! -f "$OUT/run.json" && ! -f "$OUT/latest.pt" && ! -f "$OUT/final.pt" ]] || {
    echo '[STOP] existing training artifacts preserved; use resume only for an interrupted run'; exit 2;
  }
  # Preflight is automatic, not a separate GPU smoke/pilot experiment.
  for f in "$HERE"/*.py; do "$PY" -m py_compile "$f"; done
  bash -n "$HERE/run.sh"
  "$PY" -m unittest discover -s "$HERE" -p test_training.py -v
  "$PY" "$HERE/train.py" preflight --cache "$CACHE"
  CODE_ID="$("$PY" "$HERE/train.py" identity)"
  CODE="$ROOT/sources/$CODE_ID"
  if [[ ! -d "$CODE" ]]; then
    TMP_CODE="$(mktemp -d "$ROOT/sources/.copy.XXXXXX")"
    cp -- "$HERE"/*.py "$HERE"/*.sh "$HERE"/*.json "$HERE"/*.md "$TMP_CODE/"
    mv -- "$TMP_CODE" "$CODE"
  fi
else
  [[ -f "$STATE" && -s "$OUT/latest.pt" && ! -f "$OUT/final.pt" ]] || {
    echo '[STOP] no interrupted checkpoint to resume'; exit 2;
  }
  CODE="$(bash -c 'source "$1"; printf "%s" "$CODE"' _ "$STATE")"
  CODE_ID="$(bash -c 'source "$1"; printf "%s" "$CODE_ID"' _ "$STATE")"
fi
[[ "$("$PY" "$CODE/train.py" identity)" == "$CODE_ID" ]] || { echo '[STOP] source snapshot checksum mismatch'; exit 2; }
NODE_ARGS=()
if [[ -n "${NODE:-}" ]]; then NODE_ARGS=(--nodelist="$NODE"); else NODE_ARGS=(--exclude=3090node1); fi
export BF_ROOT="$ROOT" BF_CACHE="$CACHE" BF_CODE_ID="$CODE_ID" BF_MODE="$MODE" VIS_PYTHON="$PY"
RAW="$(sbatch --parsable --partition=GPU "${NODE_ARGS[@]}" --nodes=1 --ntasks=1 \
  --gres=gpu:3090:4 --cpus-per-task=16 --time="${TIME_LIMIT:-1-00:00:00}" \
  --job-name=h9_boundary_first --chdir="$CODE" --export=ALL \
  --output="$ROOT/logs/${MODE}_%j.out" --error="$ROOT/logs/${MODE}_%j.out" \
  "$CODE/run.sh" _worker)"
JOB_ID="${RAW%%;*}"
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo "[STOP] unrecognized sbatch result: $RAW; do not resubmit blindly"; exit 2; }
LOG="$ROOT/logs/${MODE}_${JOB_ID}.out"
SAVED="$ROOT/${MODE}_job_${JOB_ID}.env"
printf 'export JOB_ID=%q\nexport LOG=%q\nexport CODE=%q\nexport CODE_ID=%q\n' "$JOB_ID" "$LOG" "$CODE" "$CODE_ID" > "$SAVED"
cp "$SAVED" "$STATE.tmp"; mv "$STATE.tmp" "$STATE"
echo "[submitted] job=$JOB_ID mode=$MODE nodes=1 gpus=4 source=$CODE_ID"
echo "[out] $OUT"; echo "[log] $LOG"; echo "[state] $SAVED"
