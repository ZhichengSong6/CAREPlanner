#!/usr/bin/env bash
# READ ONLY. No submission, cancellation, deletion, merge or label generation.
set -euo pipefail
BASE="${BATCH_BASE:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_offline_labels_v3}"
STATE="${1:-$BASE/latest_base.env}";MODE="${2:-main}"
[[ -f "$STATE" ]] || { echo "[STOP] state not found: $STATE; use the path printed by submit";exit 2; }
source "$STATE" # Only source a locally generated, trusted state file.
printf '[job] %s\n[phase] %s\n[out] %s\n[log] %s\n' "$JOB_ID" "$PHASE" "$OUT" "$LOG"
case "$MODE" in
 main) echo '[read-only] Ctrl+C exits viewing, not the job';tail --retry -n 160 -F "$LOG" ;;
 workers) for ((r=0;r<WORKERS;r++));do echo "=== $PHASE worker $r ===";f="$OUT/logs/${PHASE}_worker_${r}.log";if [[ -f "$f" ]];then tail -n 40 "$f";else echo '[not created yet]';fi;done ;;
 status) squeue -j "$JOB_ID" -o '%.18i %.24j %.10T %.10M %.10l %.20R' || true;sacct -X -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed || true ;;
 summary) sub=base_cache;[[ "$PHASE" == audit ]] && sub=paired_cache;for name in summary.md summary.json;do echo "=== $sub/$name ===";if [[ -f "$OUT/$sub/$name" ]];then cat "$OUT/$sub/$name";else echo '[not generated yet]';fi;done ;;
 plan) for name in sampling_report.json audit_plan.json;do if [[ -f "$OUT/$name" ]];then cat "$OUT/$name";else echo "[not generated yet] $name";fi;done ;;
 *) echo 'usage: bash watch.sh [state.env] main|workers|status|summary|plan';exit 2 ;;
esac
