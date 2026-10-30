#!/usr/bin/env bash
set -euo pipefail
care_planner_research_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
care_planner_research_scene="${1:-baseline}"
case "$care_planner_research_scene" in
  baseline|nonblocking_box|detour_box|blocked_goal) ;;
  *) echo "Usage: $0 [baseline|nonblocking_box|detour_box|blocked_goal] [unique_tag] [feasible|original|replacement|progress|task_first|executable_task|continuation|prefix_objective]" >&2; exit 2 ;;
esac
care_planner_research_tag="${2:-manual_$(date -u +%Y%m%d_%H%M%S)}"
care_planner_research_variant="${3:-feasible}"
case "$care_planner_research_variant" in
  feasible|original)
    care_planner_research_policy=repair_soft
    if [[ "$care_planner_research_variant" == original ]]; then
      care_planner_research_policy=original
    fi
    exec /usr/bin/python3 -B \
      "$care_planner_research_repo/outputs/phase_e_recovery_feasible_20261027/run.py" \
      --configuration "$care_planner_research_scene" --tag "$care_planner_research_tag" \
      --policy "$care_planner_research_policy"
    ;;
  replacement)
    exec /usr/bin/python3 -B \
      "$care_planner_research_repo/outputs/phase_e_replacement_execution_20261028/run.py" \
      --configuration "$care_planner_research_scene" --tag "$care_planner_research_tag" \
      --policy replacement
    ;;
  progress)
    exec /usr/bin/python3 -B \
      "$care_planner_research_repo/outputs/phase_e_observation_progress_20261028/run.py" \
      --configuration "$care_planner_research_scene" --tag "$care_planner_research_tag" \
      --policy progress
    ;;
  task_first)
    exec /usr/bin/python3 -B \
      "$care_planner_research_repo/outputs/phase_e_task_first_20261028/run.py" \
      --configuration "$care_planner_research_scene" --tag "$care_planner_research_tag" \
      --policy task_first
    ;;
  executable_task)
    exec /usr/bin/python3 -B \
      "$care_planner_research_repo/outputs/phase_e_executable_task_20261029/run.py" \
      --configuration "$care_planner_research_scene" --tag "$care_planner_research_tag" \
      --policy executable_task
    ;;
  continuation)
    exec /usr/bin/python3 -B \
      "$care_planner_research_repo/outputs/phase_e_continuation_20261029/run.py" \
      --configuration "$care_planner_research_scene" --tag "$care_planner_research_tag" \
      --policy continuation
    ;;
  prefix_objective)
    exec /usr/bin/python3 -B \
      "$care_planner_research_repo/outputs/phase_e_prefix_objective_20261029/run.py" \
      --configuration "$care_planner_research_scene" --tag "$care_planner_research_tag" \
      --policy prefix_objective
    ;;
  restore|primary|recovery|occlusion)
    echo "The obsolete '$care_planner_research_variant' study was removed during the authorized outputs cleanup. Use feasible or original (the paired control in the current runtime)." >&2
    exit 2
    ;;
  *) echo "Unknown research variant: $care_planner_research_variant" >&2; exit 2 ;;
esac
