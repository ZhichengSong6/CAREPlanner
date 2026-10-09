#!/usr/bin/env bash
#SBATCH --job-name=r1_v2fresh
set -euo pipefail
: "${RUN_SCRIPT:?}"
exec bash "$RUN_SCRIPT"
