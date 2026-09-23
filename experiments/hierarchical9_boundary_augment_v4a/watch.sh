#!/usr/bin/env bash
set -euo pipefail
BASE="${AUGMENT_BASE:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_boundary_augment_v4a}"
STATE="${1:-$BASE/latest_augment.env}";MODE="${2:-status}";[[ -f "$STATE" ]]||{ echo "[STOP] state missing";exit 2; };source "$STATE"
case "$MODE" in
 status) squeue -j "$JOB_ID" -o '%.18i %.10T %.10M %.20R'||true;sacct -X -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed||true;;
 summary) [[ -f "$OUT/summary.md" ]]&&cat "$OUT/summary.md"||echo "[not generated yet]";;
 logs) for f in "$OUT"/logs/rank*.log;do [[ -e "$f" ]]||continue;echo "===== $f =====";tail -n 4 "$f";done;;
 *) echo 'usage: watch.sh [state] status|summary|logs';exit 2;;
esac
