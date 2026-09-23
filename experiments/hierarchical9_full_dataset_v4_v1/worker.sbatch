#!/usr/bin/env bash
#SBATCH --job-name=full_v4
set -euo pipefail
: "${RUN_SCRIPT:?}"
exec bash "$RUN_SCRIPT"
