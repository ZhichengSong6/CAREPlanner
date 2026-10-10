#!/usr/bin/env bash
set -euo pipefail
BASE="${COMPARE_BASE:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/bf_r1_fov_compare_v1}"
STATE="${1:-$BASE/latest_compare.env}"
MODE="${2:-status}"
[[ -f "$STATE" ]] || { echo "[STOP] state missing: $STATE"; exit 2; }
source "$STATE"
case "$MODE" in
 status)
  squeue -j "$JOB_ID" -o '%.18i %.12T %.12M %.40R' 2>&1 || true
  sacct -X -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed,NodeList || true
  ;;
 summary) [[ -f "$OUT/summary.md" ]] && cat "$OUT/summary.md" || echo "[not generated yet]";;
 diagnosis) [[ -f "$OUT/diagnosis/diagnosis.md" ]] && cat "$OUT/diagnosis/diagnosis.md" || echo "[not generated yet]";;
 log) [[ -f "$LOG" ]] && tail -n 120 "$LOG" || echo "[log not generated yet]";;
 ranks)
  for rank in 0 1 2 3; do echo "==== rank $rank ===="; [[ -f "$OUT/rank${rank}.log" ]] && tail -n 15 "$OUT/rank${rank}.log" || true; done;;
 *) echo "Usage: watch.sh [state] status|summary|diagnosis|log|ranks"; exit 2;;
esac
