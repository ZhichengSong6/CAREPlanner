#!/usr/bin/env bash
set -euo pipefail
care_replacement_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec /usr/bin/python3 -B \
  "$care_replacement_repo/outputs/phase_e_replacement_execution_20261028/run.py" "$@"
