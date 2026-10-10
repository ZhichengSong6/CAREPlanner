#!/usr/bin/env bash
#SBATCH --job-name=bf_r1_fov
set -euo pipefail
: "${RUN_SCRIPT:?}"
exec bash "$RUN_SCRIPT"
