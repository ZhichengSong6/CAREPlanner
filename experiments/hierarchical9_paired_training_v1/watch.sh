#!/usr/bin/env bash
set -euo pipefail
BASE="\${TRAIN_BASE:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_paired_training_v1}"
STATE="\${1:-$BASE/latest_train.env}";MODE="\${2:-status}"
[[ -f "$STATE" ]] || { echo "[STOP] state missing: $STATE";exit 2; }
source "$STATE"
case "$MODE" in
 status) squeue -j "$JOB_ID" -o '%.18i %.10T %.10M %.20R' || true;sacct -X -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed || true;;
 logs) for a in old_value new_value old_value_grad new_value_grad;do echo "===== $a =====";f="$OUT/logs/$a.log";[[ -f "$f" ]]&&tail -n 20 "$f"||echo '[not created yet]';done;;
 summary) [[ -f "$OUT/comparison.md" ]]&&cat "$OUT/comparison.md"||echo '[not generated yet]';;
 *) echo 'usage: watch.sh [state] status|logs|summary';exit 2;;
esac
