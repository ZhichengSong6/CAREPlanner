#!/usr/bin/env bash
set -euo pipefail
BASE="${COMPARE_BASE:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_v4v2_fresh_compare_v1}"
STATE="${1:-$BASE/latest_compare.env}"
MODE="${2:-status}"
[[ -f "$STATE" ]] || { echo "[STOP] state missing";exit 2; }
source "$STATE"
case "$MODE" in
  status) squeue -j "$JOB_ID" -o '%.18i %.10T %.10M %.20R' || true; sacct -X -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed || true;;
  summary) [[ -f "$OUT/summary.md" ]] && cat "$OUT/summary.md" || echo "[not generated yet]";;
  diagnosis) [[ -f "$OUT/diagnosis/diagnosis.md" ]] && cat "$OUT/diagnosis/diagnosis.md" || echo "[not generated yet]";;
  log) [[ -f "$LOG" ]] && tail -n 120 "$LOG" || echo "[log not created yet]";;
  *) echo "usage: watch.sh [state] status|summary|diagnosis|log"; exit 2;;
esac
