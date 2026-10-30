#!/usr/bin/env bash
# CPU-only offline diagnostics. No ROS node, CUDA, simulator or execution.
set -euo pipefail
cd /home/zhicheng/Project/CAREPlanner
g++ -std=c++14 -O1 -DPINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION \
  -DPINOCCHIO_WITH_HPP_FCL -DPINOCCHIO_WITH_URDFDOM \
  -DBOOST_MPL_LIMIT_LIST_SIZE=30 -DBOOST_MPL_LIMIT_VECTOR_SIZE=30 \
  -I/usr/include/eigen3 -I/opt/ros/noetic/include -Isrc/care_confidence_map/include \
  scripts/query_phase_e_body_clearance.cpp -Ldevel/lib -L/opt/ros/noetic/lib \
  -Wl,-rpath,/home/zhicheng/Project/CAREPlanner/devel/lib:/opt/ros/noetic/lib \
  -ltrajectory_risk_evaluator -lpinocchio_default -o /tmp/care_repair_body_clearance
g++ -std=c++14 -O1 -I/usr/include/eigen3 -Isrc/third_party/piqp/include \
  scripts/replay_phase_e_qp.cpp -o /tmp/care_replay_qp
/home/zhicheng/miniconda3/envs/viscdf/bin/python scripts/test_phase_e_repair_conflict_diagnosis.py
/home/zhicheng/miniconda3/envs/viscdf/bin/python scripts/diagnose_phase_e_repair_conflicts.py \
  --output-dir "${1:?new output directory required}"
