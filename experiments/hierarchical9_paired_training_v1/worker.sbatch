#!/usr/bin/env bash
#SBATCH --job-name=r1_pairtrain
set -euo pipefail
exec bash "$TRAIN_SCRIPT"
