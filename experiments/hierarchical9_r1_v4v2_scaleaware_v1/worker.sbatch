#!/usr/bin/env bash
#SBATCH --job-name=r1_v2scale
set -euo pipefail
: "${RUN_SCRIPT:?}"
exec bash "$RUN_SCRIPT"
