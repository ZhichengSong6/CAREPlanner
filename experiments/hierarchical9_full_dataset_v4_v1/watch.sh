#!/usr/bin/env bash
set -euo pipefail
STAGE="${1:-global}";MODE="${2:-status}"
BASE="${FULL_V4_BASE:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/full_dataset_v4_v1}"
STATE="${FULL_JOB_STATE:-$BASE/latest_${STAGE}.env}";[[ -f "$STATE" ]]||{ echo "[STOP] no state for $STAGE";exit 2; };source "$STATE"
case "$MODE" in
 status) squeue -j "$JOB_ID" -o '%.18i %.10T %.10M %.20R'||true;sacct -X -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed||true;;
 logs) for f in "$OUT/logs/${STAGE}_"*.log;do [[ -e "$f" ]]||continue;echo "===== $f =====";tail -n 2 "$f";done;;
 summary) case "$STAGE" in global) f="$OUT/global_pool_summary.json";;v3) f="$OUT/v3_summary.json";;v4) f="$OUT/v4_summary.json";;*) exit 2;;esac;[[ -f "$f" ]]&&cat "$f"||echo "[not generated yet]";;
 *) echo "usage: watch.sh global|v3|v4 status|logs|summary";exit 2;;
esac
