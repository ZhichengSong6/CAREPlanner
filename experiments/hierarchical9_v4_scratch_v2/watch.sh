#!/usr/bin/env bash
set -euo pipefail
STAGE="${1:-smoke}";MODE="${2:-status}"
ROOT="${V4TR2_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_v4_scratch_v2}"
STATE="${V4TR2_STATE:-$ROOT/latest_${STAGE}.env}";[[ -f "$STATE" ]]||{ echo "[STOP] missing $STATE";exit 2;};source "$STATE"
case "$MODE" in
 status) squeue -j "$JOB_ID" -o '%.18i %.10T %.10M %.20R'||true;sacct -X -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed||true;;
 log) [[ -f "$LOG" ]]&&tail -n 160 "$LOG"||echo "[log not created yet]";;
 summary) case "$STAGE" in smoke) f="$ROOT/smoke/run.json";;pilot) f="$ROOT/pilot/run.json";;train|resume) f="$ROOT/formal/run.json";;*)exit 2;;esac;[[ -f "$f" ]]&&cat "$f"||echo "[summary not created yet]";;
 *) echo "usage: watch.sh <smoke|pilot|train|resume> <status|log|summary>";exit 2;;
esac
