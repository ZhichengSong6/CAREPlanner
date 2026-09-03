cmake_minimum_required(VERSION 3.0.2)
project(egocentric_arm_planner)

add_compile_options(-std=c++14 -Wall -Wextra -Wpedantic)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  trajectory_msgs
  visualization_msgs
  interactive_markers
  tf2_ros
  tf2_geometry_msgs
  arm_model
  arm_trajectory
  care_collision_cdf
)

find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)

set(PIQP_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/piqp/include")
if(NOT EXISTS "${PIQP_INCLUDE_DIR}/piqp/piqp.hpp")
  message(FATAL_ERROR
    "PIQP headers are missing. Run: git submodule update --init --recursive")
endif()

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES egocentric_arm_planner
  CATKIN_DEPENDS
    roscpp
    std_msgs
    sensor_msgs
    geometry_msgs
    trajectory_msgs
    visualization_msgs
    interactive_markers
    tf2_ros
    tf2_geometry_msgs
    arm_model
    arm_trajectory
    care_collision_cdf
  DEPENDS EIGEN3
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_library(egocentric_arm_planner
  src/task_trajectory_generator.cpp
  src/dummy_trajectory_evaluator.cpp
  src/intervention_manager.cpp
  src/receding_horizon_planner.cpp
  src/trajectory_visualizer.cpp
  src/trajectory_execution_manager.cpp
)

add_dependencies(egocentric_arm_planner
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(egocentric_arm_planner
  ${catkin_LIBRARIES}
)

add_executable(receding_horizon_planner_node
  src/receding_horizon_planner_node.cpp
)
add_dependencies(receding_horizon_planner_node
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(receding_horizon_planner_node
  egocentric_arm_planner
  ${catkin_LIBRARIES}
)

add_executable(trajectory_visualizer_node
  src/trajectory_visualizer_node.cpp
)
add_dependencies(trajectory_visualizer_node
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(trajectory_visualizer_node
  egocentric_arm_planner
  ${catkin_LIBRARIES}
)

add_executable(trajectory_execution_manager_node
  src/trajectory_execution_manager_node.cpp
)
add_dependencies(trajectory_execution_manager_node
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(trajectory_execution_manager_node
  egocentric_arm_planner
  ${catkin_LIBRARIES}
)

add_executable(velocity_qp_mpc_node
  src/velocity_qp_mpc_node.cpp
  src/velocity_qp_mpc.cpp
)
add_dependencies(velocity_qp_mpc_node
  ${catkin_EXPORTED_TARGETS}
)
target_include_directories(velocity_qp_mpc_node PRIVATE
  ${PIQP_INCLUDE_DIR}
)
target_compile_options(velocity_qp_mpc_node PRIVATE
  -O3
)
target_compile_definitions(velocity_qp_mpc_node PRIVATE
  NDEBUG
)
target_link_libraries(velocity_qp_mpc_node
  ${catkin_LIBRARIES}
)

# Experimental Visibility-Before-Contact deadline-waypoint MPC. In the
# planner/controller split the executable also listens to the final actuator
# command so the first-step acceleration and smoothness boundary is anchored to
# what was ACTUALLY executed, not to a possibly rejected raw candidate.
add_executable(velocity_qp_mpc_waypoint_node
  src/velocity_qp_mpc_waypoint_execution_anchored_node.cpp
  src/velocity_qp_mpc_waypoint.cpp
)
add_dependencies(velocity_qp_mpc_waypoint_node
  ${catkin_EXPORTED_TARGETS}
)
target_include_directories(velocity_qp_mpc_waypoint_node PRIVATE
  ${PIQP_INCLUDE_DIR}
)
target_compile_options(velocity_qp_mpc_waypoint_node PRIVATE
  -O3
)
target_compile_definitions(velocity_qp_mpc_waypoint_node PRIVATE
  NDEBUG
)
target_link_libraries(velocity_qp_mpc_waypoint_node
  ${catkin_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
)

# C5.4 event-triggered local trajectory optimizer. This is the new planning
# backend: explicit q/u multiple shooting, sparse PIQP, slackened CDF constraints,
# and a short sequential-convex inner loop. It never publishes actuator commands.
add_executable(local_sparse_scp_planner_node
  src/local_sparse_scp_planner_node.cpp
  src/local_sparse_scp_planner.cpp
)
add_dependencies(local_sparse_scp_planner_node
  ${catkin_EXPORTED_TARGETS}
)
target_include_directories(local_sparse_scp_planner_node PRIVATE
  ${PIQP_INCLUDE_DIR}
)
target_compile_options(local_sparse_scp_planner_node PRIVATE
  -O3
)
target_compile_definitions(local_sparse_scp_planner_node PRIVATE
  NDEBUG
)
target_link_libraries(local_sparse_scp_planner_node
  ${catkin_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
)

# One-shot C4.4 diagnostic: freeze the first actual REPAIR snapshot and solve
# the exact repair-QP form at several horizons. The generated trajectories are
# published only on a diagnostic topic and never enter the commit/execution path.
add_executable(frozen_snapshot_repair_diagnostic_node
  src/frozen_snapshot_repair_diagnostic_node.cpp
)
add_dependencies(frozen_snapshot_repair_diagnostic_node
  ${catkin_EXPORTED_TARGETS}
)
target_include_directories(frozen_snapshot_repair_diagnostic_node PRIVATE
  ${PIQP_INCLUDE_DIR}
)
target_compile_options(frozen_snapshot_repair_diagnostic_node PRIVATE
  -O3
)
target_compile_definitions(frozen_snapshot_repair_diagnostic_node PRIVATE
  NDEBUG
)
target_link_libraries(frozen_snapshot_repair_diagnostic_node
  ${catkin_LIBRARIES}
)

add_executable(ee_interactive_marker_node
  src/ee_interactive_marker_node.cpp
)
add_dependencies(ee_interactive_marker_node
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(ee_interactive_marker_node
  egocentric_arm_planner
  ${catkin_LIBRARIES}
)

catkin_install_python(PROGRAMS
  scripts/phase_b2_controlled_trial_node.py
  scripts/phase_c4_3_continuous_trial_node.py
  scripts/c4_3_planner_regime_handoff_node.py
  scripts/c4_4_verified_regime_manager_node.py
  scripts/optimized_trajectory_continuity_node.py
  scripts/probe_single_flight_gate_node.py
  scripts/joint_velocity_rate_limiter_node.py
  scripts/execution_gcdf_measured_state_trajectory_node.py
  scripts/execution_gcdf_safety_monitor_node.py
  scripts/phase_b2_visibility_acquisition_adapter_node.py
  scripts/phase_b2_ab_logger_node.py
  scripts/phase_b2_steering_logger_node.py
  scripts/phase_b2_vbc_waypoint_logger_node.py
  scripts/vbc_execution_reference_gate_node.py
  scripts/vbc_progress_reference_gate_node.py
  scripts/vbc_robustness_prescreen_node.py
  scripts/predictive_recovery_guard_node.py
  scripts/predicted_vbc_recovery_guard_node.py
  scripts/compare_phase_b2_ab_runs.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
