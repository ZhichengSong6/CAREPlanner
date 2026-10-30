#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"
export RUN_ID="${RUN_ID:-case_001_r1_motion_prior_$(date +%Y%m%d_%H%M%S)}"
case "${RUN_ID}" in
  ''|*[!A-Za-z0-9_-]*) echo "Invalid RUN_ID: ${RUN_ID}" >&2; exit 2 ;;
esac
export CASE_ID="${CASE_ID:-case_001}" RUN_SECONDS="${RUN_SECONDS:-60}"
# Use separate ROS/Gazebo masters; override the URIs if those ports are busy.
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11497}"
export GAZEBO_MASTER_URI="${GAZEBO_MASTER_URI:-http://127.0.0.1:11498}"
export CONFIDENCE_MAP_CONFIG_FILE="${REPO}/src/care_confidence_map/config/confidence_map_static_box_ray.yaml"
export CANDIDATE_REPLACEMENT_ENABLED=true
export CASE_FILE="${CASE_FILE:-${REPO}/src/egocentric_arm_planner/config/phase_c2_vbc_cases.json}"
export CHECKPOINT="${CHECKPOINT:?explicit model checkpoint required}"
export CDF_ACTIVATION="${CDF_ACTIVATION:?explicit activation required}"
export CDF_EXPECTED_SHA256="${CDF_EXPECTED_SHA256:?explicit checkpoint SHA256 required}"
export CDF_CHECKPOINT_KEY="${CDF_CHECKPOINT_KEY:-latest}"
export GPU_ENV=viscdf GPU_DEVICE=cuda NCDF_ENV=ncdf_l4c NCDF_DEVICE=cpu
export GPU_SOCKET="/tmp/care_collision_cdf_gpu_${RUN_ID}.sock"
export GAZEBO_GUI=false USE_RVIZ=false EARLY_STOP_ON_GOAL="${EARLY_STOP_ON_GOAL:-false}"
export FORCE_ZERO_INITIAL_Q=false TOF_FUSION_ENABLED=true REQUIRE_REAL_TOF_READINESS=true
export APPLY_INITIAL_JOINT_OVERRIDES=false
export GEOMETRY_BACKEND=primitive GCDF_GEOMETRY_BACKEND=primitive
export VBC_GEOMETRY_BACKEND=primitive BODY_PRIOR_GEOMETRY_BACKEND=primitive
export DIAGNOSTIC_GEOMETRY_BACKEND=primitive GCDF_BODY_INFLATION_M=0.0
# Candidate and execution margins are independent. The current R1 experiment
# uses candidate 20 mm / execution 10 mm; tracking keeps its existing budget.
export VBC_SWEPT_VOLUME_MARGIN_M="${VBC_SWEPT_VOLUME_MARGIN_M:-0.020}"
export EXECUTION_VBC_SWEPT_VOLUME_MARGIN_M="${EXECUTION_VBC_SWEPT_VOLUME_MARGIN_M:-0.010}"
export RECORD_AUDITED_CANDIDATES="${RECORD_AUDITED_CANDIDATES:-true}"
# Experimental downstream expiry/braking plugin; retain the historical default
# until full task and command-release safety qualification is complete.
export ACTUATOR_COMMAND_WATCHDOG="${ACTUATOR_COMMAND_WATCHDOG:-false}"
# Every primitive verifier must enclose the full piecewise-linear trajectory.
export VBC_CONTINUOUS_MOTION_BOUND_ENABLED=true
export TRACKER_CERTIFIED_MARGIN_M=0.020 LOCAL_SCP_PROXIMITY_MARGIN=0.025
export EXECUTION_GCDF_AUDIT_ENABLED=false FINAL_VBC_RECOVERY_ENABLED=false
export PER_SENSOR_HYBRID_ENABLED=true ADAPTIVE_REFINEMENT_ENABLED=false
# Research proposals use the nominal ideal-FOV model plus primitive LOS.
# Observation completion and every executable motion retain their own audits.
export PER_SENSOR_ALLOW_NOMINAL_PROPOSALS=true
export PER_SENSOR_CHECKPOINT="${REPO}/src/care_visibility_cdf/checkpoints/hierarchical9_r1_scratch50k/final.pt"
export PER_SENSOR_HIGH_WITNESS_PRIORITY_ENABLED=true
export PER_SENSOR_HIGH_WITNESS_Z_MIN=0.85
export FRONTIER_STEERING_ENABLED=true VBC_GATED_FRONTIER_STEP_ENABLED=true
export ROLLING_VISIBILITY_GUIDANCE_MODE="${ROLLING_VISIBILITY_GUIDANCE_MODE:-drive}"
export FRONTIER_VBC_ESCALATION_ENABLED=true FRONTIER_VBC_ESCALATION_AFTER=3
export FRONTIER_ESCALATED_STEP_INF=0.10 FRONTIER_STEP_INF=0.05
export RECORDING_PROFILE=full LOCAL_CDF_PAIR_AUDIT_ENABLED=false
export VERIFICATION_TCP_NODELAY=true VBC_CONFIDENCE_QUERY_PERSISTENT=true
for target in \
  "outputs/c5_5_vbc_gcdf_regime/${RUN_ID}" \
  "logs/c5_5_vbc_gcdf_regime/${RUN_ID}" \
  "CAREPlanner_C5_RESULT_${RUN_ID}.zip" \
  "${GPU_SOCKET}"; do
  if [[ -e "$target" ]]; then
    echo "Refusing to overwrite existing run target: $target" >&2
    exit 2
  fi
done
source /opt/ros/noetic/setup.bash
bash scripts/run_and_pack_c5_5_vbc_gcdf_regime.sh
