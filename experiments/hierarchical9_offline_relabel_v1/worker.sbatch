#!/usr/bin/env bash
#SBATCH --job-name=r1_relabel
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
set -euo pipefail
: "${LABEL_SCRIPT:?submit with submit.sh}"
: "${LABEL_MODE:?submit with submit.sh}"
bash "$LABEL_SCRIPT" "$LABEL_MODE"
