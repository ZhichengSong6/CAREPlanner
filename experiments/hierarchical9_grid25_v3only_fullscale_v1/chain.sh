#!/usr/bin/env bash
# Run *inside* the current Slurm job after 4 verified V3-only segments.
# Try one afterok follow-up; if account submit quota prevents it, requeue self.
set -euo pipefail
PARENT="${1:-}"
[[ "$PARENT" =~ ^[0-9]+$ && "${SLURM_JOB_ID:-}" == "$PARENT" ]] || {
  echo "[STOP] Wrong Slurm parent ID for chain" >&2; exit 2;
}
: "${GRID25_ROOT:?GRID25_ROOT required}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="$GRID25_ROOT/derived/grid25_v3only_fullscale_v1"
mkdir -p "$BASE/logs" "$BASE/chain"
RECORD="$BASE/chain/after_job_${PARENT}.txt"
if [[ -f "$RECORD" ]]; then
  echo "[FULLSCALE CHAIN ALREADY RECORDED] $(cat "$RECORD")"
  exit 0
fi
ARGS=(--partition=GPU --nodes=1 --ntasks=1 --cpus-per-task=4
      --time=1-00:00:00 --requeue --open-mode=append
      --job-name=g25v3full
      "--dependency=afterok:$PARENT"
      "--output=$BASE/logs/slurm_%j.out"
      "--error=$BASE/logs/slurm_%j.err"
      "--export=ALL,GRID25_ROOT=$GRID25_ROOT")
echo "[FULLSCALE CONTINUATION] submitting ONE afterok:$PARENT job (4 CPUs, zero GPUs)"
if OUTPUT="$(sbatch --parsable "${ARGS[@]}" "$HERE/worker.sbatch" 2>"$BASE/chain/submit_${PARENT}.err")"; then
  CHILD="${OUTPUT%%;*}"
  if [[ "$CHILD" =~ ^[0-9]+$ ]]; then
    printf 'parent=%s child=%s mode=afterok\n' "$PARENT" "$CHILD" > "$RECORD"
    echo "[FULLSCALE NEXT JOB SUBMITTED] parent=$PARENT child=$CHILD"
    exit 0
  fi
  echo "[STOP] sbatch returned ambiguous child ID: $OUTPUT. Do NOT auto requeue." >&2
  exit 4
fi
echo "[CHAIN SUBMIT DENIED] $(cat "$BASE/chain/submit_${PARENT}.err")" >&2
echo "[FULLSCALE FALLBACK] trying one-job scontrol requeue for quota-limited clusters"
if scontrol requeue "$PARENT"; then
  # Requeued jobs keep the same Slurm ID: NEVER write a once-only
  # afterok-child sentinel here, or the second restart would stop early.
  echo "[FULLSCALE REQUEUED] job=$PARENT (same ID, new allocation)"
  exit 0
fi
echo "[STOP] Neither afterok submission nor self-requeue permitted. Source labels remain intact; manually resume after reviewing Slurm limits." >&2
exit 5
