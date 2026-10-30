#!/usr/bin/env bash
set -euo pipefail
care_recovery_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec /usr/bin/python3 -B \
  "$care_recovery_repo/outputs/phase_e_recovery_feasible_20261027/run.py" "$@"
