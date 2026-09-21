#!/usr/bin/env bash
# Submission only; resource availability is determined by Slurm.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LABEL_MODE="${1:-smoke}"
[[ "$LABEL_MODE" == smoke || "$LABEL_MODE" == pilot ]] || { echo 'usage: bash submit.sh smoke|pilot';exit 2; }
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
ARTIFACT_REPO="${ARTIFACT_REPO:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner}"
BASE="${LABEL_BASE:-$ARTIFACT_REPO/outputs/mainline_b/r1_offline_labels_v1}"
OUT="${OUT:-$BASE/${LABEL_MODE}_$(date +%Y%m%d_%H%M%S)}"
BANK_CACHE="${BANK_CACHE:-$BASE/bank_numeric_cache}"
LOG_DIR="${LOG_DIR:-$BASE/logs}"
mkdir -p "$LOG_DIR"
LABEL_SCRIPT="$HERE/run.sh"
export LABEL_MODE LABEL_SCRIPT REPO ARTIFACT_REPO OUT BANK_CACHE
# Do not request --mem: preserve the cluster's existing submission convention.
job=$(sbatch --parsable --nodelist="${NODE:-3090node1}" --time="${TIME_LIMIT:-01:00:00}" \
  --output="$LOG_DIR/relabel_%j.out" --export=ALL "$HERE/worker.sbatch")
echo "[submitted] job=$job node=${NODE:-3090node1} workers=4 (GPU allocation requested, not guaranteed immediately)"
echo "[out] $OUT"
echo "[log] $LOG_DIR/relabel_${job%%;*}.out"
echo "[resume] use the same MODE and OUT with RESUME=true; do not edit code/config for an existing cache"
