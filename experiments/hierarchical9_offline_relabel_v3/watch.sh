#!/usr/bin/env bash
# READ ONLY. Never submit/cancel jobs, create logs, or change label data.
set -euo pipefail
BASE="${BATCH_BASE:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_offline_labels_v3}"
STATE="${1:-$BASE/latest_base.env}"; MODE="${2:-main}"
[[ -f "$STATE" ]] || { echo "[STOP] state not found: $STATE; use the path printed by submit" >&2; exit 2; }
source "$STATE" # Only a locally generated, trusted state file.
[[ "${JOB_ID:-}" =~ ^[0-9]+$ && "${WORKERS:-}" =~ ^[1-9][0-9]*$ ]] || { echo '[STOP] invalid job/worker state' >&2; exit 2; }
[[ "${PHASE:-}" == base || "${PHASE:-}" == audit ]] || { echo '[STOP] invalid phase' >&2; exit 2; }
: "${OUT:?missing OUT}" "${LOG:?missing LOG}"
printf '[job] %s\n[phase] %s\n[out] %s\n[log] %s\n' "$JOB_ID" "$PHASE" "$OUT" "$LOG"

read_slurm() {
  if command -v timeout >/dev/null 2>&1; then timeout 10 "$@"; else "$@"; fi
}
job_status() {
  JOB_STATE=UNKNOWN; JOB_REASON='No scheduler/accounting record available'; JOB_EXIT=''
  local queue='' accounting='' row='' queue_error=''
  if queue=$(read_slurm squeue -h -j "$JOB_ID" -o '%T|%R' 2>&1); then
    if [[ -n "$queue" ]]; then
      IFS='|' read -r JOB_STATE JOB_REASON <<< "${queue%%$'\n'*}"
      JOB_STATE="${JOB_STATE%% *}"; JOB_STATE="${JOB_STATE%+}"
      return
    fi
  else
    queue_error="$queue"
  fi
  if accounting=$(read_slurm sacct -n -P -X -j "$JOB_ID" --format=JobIDRaw,State,ExitCode 2>&1); then
    row=$(awk -F'|' -v id="$JOB_ID" '$1==id {print;exit}' <<< "$accounting")
    if [[ -n "$row" ]]; then
      IFS='|' read -r _ JOB_STATE JOB_EXIT _ <<< "$row"
      JOB_STATE="${JOB_STATE%% *}"; JOB_STATE="${JOB_STATE%+}"
      JOB_REASON="accounting ExitCode=$JOB_EXIT"
    fi
  else
    JOB_REASON="Scheduler/accounting query failed: ${queue_error:-squeue returned no row}; $accounting"
  fi
}
terminal_state() {
  case "$JOB_STATE" in
    COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED|SPECIAL_EXIT) return 0 ;;
    *) return 1 ;;
  esac
}
show_location() {
  echo '[diagnostic] Scheduler job details (read only):' >&2
  read_slurm scontrol show job "$JOB_ID" >&2 || true
  printf '[STOP] Expected log: %s\n[STOP] No job was submitted/cancelled. Do not resubmit just to view output.\n' "$LOG" >&2
}
follow_main() {
  local interval="${WATCH_INTERVAL_SECONDS:-20}" unknown_count=0 running_count=0
  [[ "$interval" =~ ^[0-9]+([.][0-9]+)?$ && "$interval" =~ [1-9] ]] || { echo '[STOP] invalid watch interval' >&2; return 2; }
  echo '[read-only] Ctrl+C exits viewing, not the job'
  # No tail until the log exists and is readable. Query status instead of
  # misreporting an uncreated Slurm log as a labeling failure.
  while :; do
    job_status
    printf '[status] job=%s state=%s reason=%s\n' "$JOB_ID" "$JOB_STATE" "$JOB_REASON"
    if [[ -e "$LOG" && ( ! -f "$LOG" || ! -r "$LOG" ) ]]; then
      echo '[STOP] Log exists but is not a readable regular file; check permissions/path.' >&2
      show_location; return 2
    fi
    if [[ -f "$LOG" && -r "$LOG" ]]; then
      if terminal_state; then
        tail -n 160 -- "$LOG"
        echo "[finished] job=$JOB_ID state=$JOB_STATE ExitCode=${JOB_EXIT:-not-reported}; viewer only"
        [[ "$JOB_STATE" == COMPLETED ]] && return 0
        return 2
      fi
      echo '[log-ready] Following the existing log. Ctrl+C only exits viewing.'
      exec tail -n 160 -F -- "$LOG"
    fi
    if terminal_state; then
      echo '[STOP] Job has ended but its log is absent; this is not a pending-log wait.' >&2
      show_location; return 2
    fi
    case "$JOB_STATE" in
      PENDING|CONFIGURING|REQUEUED|REQUEUE_FED|REQUEUE_HOLD|RESIZING|SUSPENDED)
        unknown_count=0; running_count=0
        echo '[waiting] Slurm has not produced the log yet; displaying job status until it appears.' ;;
      RUNNING|COMPLETING|STAGE_OUT|SIGNALING)
        unknown_count=0; running_count=$((running_count+1))
        echo '[waiting] Job is active, but the expected log is not visible yet.'
        if (( running_count >= 6 )); then show_location; return 2; fi ;;
      *)
        running_count=0; unknown_count=$((unknown_count+1))
        echo '[waiting] Job state could not be confirmed; not assuming success or failure.'
        if (( unknown_count >= 6 )); then show_location; return 2; fi ;;
    esac
    sleep "$interval"
  done
}
case "$MODE" in
 main) follow_main ;;
 workers) for ((r=0;r<WORKERS;r++)); do echo "=== $PHASE worker $r ==="; f="$OUT/logs/${PHASE}_worker_${r}.log"; if [[ -f "$f" && -r "$f" ]]; then tail -n 40 -- "$f"; else echo '[not created or not readable; check main/status]'; fi; done ;;
 status) squeue -j "$JOB_ID" -o '%.18i %.24j %.10T %.10M %.10l %.20R' || true; sacct -X -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed || true ;;
 summary) sub=base_cache; [[ "$PHASE" == audit ]] && sub=paired_cache; for name in summary.md summary.json; do echo "=== $sub/$name ==="; if [[ -f "$OUT/$sub/$name" ]]; then cat "$OUT/$sub/$name"; else echo '[not generated yet]'; fi; done ;;
 plan) for name in sampling_report.json audit_plan.json; do if [[ -f "$OUT/$name" ]]; then cat "$OUT/$name"; else echo "[not generated yet] $name"; fi; done ;;
 *) echo 'usage: bash watch.sh [state.env] main|workers|status|summary|plan'; exit 2 ;;
esac
