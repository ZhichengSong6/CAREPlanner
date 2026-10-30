#!/usr/bin/env bash
set -euo pipefail
care_prefix_objective_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec /usr/bin/python3 -B \
  "$care_prefix_objective_repo/outputs/phase_e_prefix_objective_20261029/run.py" "$@"
