#!/usr/bin/env bash
set -euo pipefail
STAGE="${1:-smoke}";MODE="${2:-status}"
ROOT="${V4TR_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_v4_scratch_v1}"
STATE="${V4TR_STATE:-$ROOT/latest_${STAGE}.env}";[[ -f "$STATE" ]]||{ echo "[STOP] missing $STATE";exit 2; };source "$STATE"
case "$MODE" in
 status) squeue -j "$JOB_ID" -o '%.18i %.10T %.10M %.20R'||true;sacct -X -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed||true;;
 log) [[ -f "$LOG" ]]&&tail -n 120 "$LOG"||echo "[log not created yet]";;
 summary)
   if [[ "$STAGE" == cache ]];then f="$ROOT/training_cache/manifest.json";elif [[ "$STAGE" == smoke ]];then f="$ROOT/smoke/run.json";else f="$ROOT/formal/run.json";fi
   [[ -f "$f" ]]&&cat "$f"||echo "[summary not created yet]";;
 *) echo "usage: watch.sh <cache|smoke|train|resume> <status|log|summary>";exit 2;;
esac
