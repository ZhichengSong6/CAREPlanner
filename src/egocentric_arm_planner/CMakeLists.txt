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
)

find_package(Eigen3 REQUIRED)

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

# Experimental Visibility-Before-Contact deadline-waypoint MPC.  Kept as a
# separate executable so the previously validated Phase-A/B.2 controller remains
# untouched for ablations and regression checks.
add_executable(velocity_qp_mpc_waypoint_node
  src/velocity_qp_mpc_waypoint_node.cpp
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
)

add_executable(ee_interactive_marker_node
  src/ee_interactive_marker_node.cpp
)
add_dependencies(ee_interactive_marker_node
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(ee_interactive_marker_node
  ${catkin_LIBRARIES}
)

catkin_install_python(PROGRAMS
  scripts/phase_b2_controlled_trial_node.py
  scripts/phase_b2_visibility_acquisition_adapter_node.py
  scripts/phase_b2_ab_logger_node.py
  scripts/phase_b2_steering_logger_node.py
  scripts/compare_phase_b2_ab_runs.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)