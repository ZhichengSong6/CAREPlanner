#!/usr/bin/env bash
# CPU-only offline diagnostics. No ROS node, CUDA, simulator or execution.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
care_diagnosis_root="$PWD"
care_diagnosis_python="${CARE_VISIBILITY_PYTHON:-python3}"
care_piqp_include="$(python3 scripts/prepare_piqp.py)"
g++ -std=c++14 -O1 -DPINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION \
  -DPINOCCHIO_WITH_HPP_FCL -DPINOCCHIO_WITH_URDFDOM \
  -DBOOST_MPL_LIMIT_LIST_SIZE=30 -DBOOST_MPL_LIMIT_VECTOR_SIZE=30 \
  -I/usr/include/eigen3 -I/opt/ros/noetic/include -Isrc/care_confidence_map/include \
  scripts/query_phase_e_body_clearance.cpp -Ldevel/lib -L/opt/ros/noetic/lib \
  -Wl,-rpath,"${care_diagnosis_root}/devel/lib:/opt/ros/noetic/lib" \
  -ltrajectory_risk_evaluator -lpinocchio_default -o /tmp/care_repair_body_clearance
g++ -std=c++14 -O1 -I/usr/include/eigen3 -I"${care_piqp_include}" \
  scripts/replay_phase_e_qp.cpp -o /tmp/care_replay_qp
"${care_diagnosis_python}" scripts/test_phase_e_repair_conflict_diagnosis.py
"${care_diagnosis_python}" scripts/diagnose_phase_e_repair_conflicts.py \
  --output-dir "${1:?new output directory required}"
