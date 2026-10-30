#!/usr/bin/env bash
set -euo pipefail
care_action_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec /usr/bin/python3 -B \
  "$care_action_repo/outputs/phase_e_observation_progress_20261028/run.py" "$@"
