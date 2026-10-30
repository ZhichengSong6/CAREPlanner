#!/usr/bin/env bash
set -euo pipefail
care_occlusion_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$care_occlusion_repo/scripts/run_phase_e_research_static.sh" \
  "${1:-detour_box}" "${2:-occlusion_$(date -u +%Y%m%d_%H%M%S)}" occlusion
