#!/usr/bin/env bash
set -euo pipefail
BASE="${V3_FAST_BASE:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_v3_fast_v1}";STATE="${1:-$BASE/latest_solver_v2.env}";MODE="${2:-status}";[[ -f "$STATE" ]]||exit 2;source "$STATE"
case "$MODE" in
 status) squeue -j "$JOB_ID" -o '%.18i %.10T %.10M %.20R'||true;sacct -X -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed||true;;
 summary) [[ -f "$OUT/summary.md" ]]&&cat "$OUT/summary.md"||echo "[not generated yet]";;
 logs) for f in "$(dirname "$OUT")"/v2_rank*.log;do [[ -e "$f" ]]||continue;echo "===== $f =====";tail -n 2 "$f";done;;
 *) exit 2;;
esac
