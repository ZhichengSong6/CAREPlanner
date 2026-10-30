#!/usr/bin/env bash
set -euo pipefail
cd /home/zhicheng/Project/CAREPlanner
# Same code, CUDA device, cases, safety gates, scalar projector and 45s budget.
# Only the per-sensor hybrid opt-in changes. Never launches all 30 cases.
STAMP="${AB_STAMP:-$(date +%Y%m%d-%H%M%S)-observation-ab}"
for model in scalar h9; do
  export BATCH_STAMP="${STAMP}-${model}"
  export BATCH_ID="phase_e_${BATCH_STAMP}"
  export DIAGNOSTIC_CASES=007,026
  if [[ "$model" == h9 ]]; then
    export PER_SENSOR_HYBRID_ENABLED=true
  else
    export PER_SENSOR_HYBRID_ENABLED=false
  fi
  bash scripts/run_phase_e_qp_events_regression.sh
done
