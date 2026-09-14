cmake_minimum_required(VERSION 3.0.2)
project(care_confidence_map)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  std_srvs
  sensor_msgs
  geometry_msgs
  gazebo_msgs
  trajectory_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  urdf
  pcl_ros
  pcl_conversions
  message_generation
)

find_package(yaml-cpp REQUIRED)
find_package(pinocchio REQUIRED)

add_service_files(
  FILES
  QueryConfidence.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES
    body_sample_model
    trajectory_risk_evaluator
  CATKIN_DEPENDS
    roscpp
    rospy
    std_msgs
    std_srvs
    sensor_msgs
    geometry_msgs
    gazebo_msgs
    trajectory_msgs
    visualization_msgs
    tf2
    tf2_ros
    tf2_geometry_msgs
    urdf
    pcl_ros
    pcl_conversions
    message_runtime
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
)

add_library(body_sample_model
  legacy/src/body_sample_model.cpp
)

add_dependencies(body_sample_model
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(body_sample_model
  ${catkin_LIBRARIES}
  yaml-cpp
)

add_library(trajectory_risk_evaluator
  src/trajectory_risk_evaluator.cpp
  src/vbc_primitive_model.cpp
  src/legacy_body_backend.cpp
)

add_dependencies(trajectory_risk_evaluator
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(trajectory_risk_evaluator
  ${CMAKE_DL_LIBS}
  ${catkin_LIBRARIES}
  yaml-cpp
  pinocchio::pinocchio
)

if(CATKIN_ENABLE_TESTING
   AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/test_phase_e_diagnostic_primitives.cpp"
   AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/test_phase_e_cold_start_primitives.cpp"
   AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/test_phase_e_vbc_primitives.cpp"
   AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/benchmark_phase_e_vbc_real_trajectories.cpp")
  add_executable(test_diagnostic_primitives ../../scripts/test_phase_e_diagnostic_primitives.cpp)
  add_dependencies(test_diagnostic_primitives ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  target_link_libraries(test_diagnostic_primitives trajectory_risk_evaluator ${catkin_LIBRARIES} pinocchio::pinocchio)
  add_executable(test_cold_start_primitives ../../scripts/test_phase_e_cold_start_primitives.cpp)
  add_dependencies(test_cold_start_primitives ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  target_link_libraries(test_cold_start_primitives trajectory_risk_evaluator ${catkin_LIBRARIES} pinocchio::pinocchio yaml-cpp)
  add_executable(test_vbc_primitives ../../scripts/test_phase_e_vbc_primitives.cpp)
  add_dependencies(test_vbc_primitives ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  target_link_libraries(test_vbc_primitives trajectory_risk_evaluator ${catkin_LIBRARIES} pinocchio::pinocchio)
  add_executable(benchmark_vbc_real_trajectories ../../scripts/benchmark_phase_e_vbc_real_trajectories.cpp)
  add_dependencies(benchmark_vbc_real_trajectories ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  target_link_libraries(benchmark_vbc_real_trajectories trajectory_risk_evaluator ${catkin_LIBRARIES} pinocchio::pinocchio)
endif()

add_executable(confidence_map_node
  src/confidence_map_node.cpp
)

add_executable(gcdf_primitive_anchor_node src/gcdf_primitive_anchor_node.cpp)
add_dependencies(gcdf_primitive_anchor_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(gcdf_primitive_anchor_node trajectory_risk_evaluator ${catkin_LIBRARIES} pinocchio::pinocchio)
install(TARGETS gcdf_primitive_anchor_node RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

add_dependencies(confidence_map_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(confidence_map_node
  trajectory_risk_evaluator
  pinocchio::pinocchio
  ${catkin_LIBRARIES}
  yaml-cpp
)

add_executable(tof_fusion_self_filter_node
  src/tof_fusion_self_filter_node.cpp
)

add_dependencies(tof_fusion_self_filter_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(tof_fusion_self_filter_node
  ${catkin_LIBRARIES}
)

add_executable(body_sample_model_node
  src/body_sample_model_node.cpp
)

add_dependencies(body_sample_model_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(body_sample_model_node
  trajectory_risk_evaluator
  pinocchio::pinocchio
  ${catkin_LIBRARIES}
  yaml-cpp
)

add_executable(body_observability_node
  src/body_observability_node.cpp
)

add_dependencies(body_observability_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(body_observability_node
  trajectory_risk_evaluator
  pinocchio::pinocchio
  ${catkin_LIBRARIES}
  yaml-cpp
)

add_executable(trajectory_risk_node
  src/trajectory_risk_node.cpp
)

add_dependencies(trajectory_risk_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(trajectory_risk_node
  trajectory_risk_evaluator
  ${catkin_LIBRARIES}
  yaml-cpp
  pinocchio::pinocchio
)

# C4.3 exact-timestep temporal-layer selector retained for ablation.
add_executable(trajectory_vbc_selector_node
  src/trajectory_vbc_selector_node.cpp
)

add_dependencies(trajectory_vbc_selector_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(trajectory_vbc_selector_node
  trajectory_risk_evaluator
  ${catkin_LIBRARIES}
  yaml-cpp
  pinocchio::pinocchio
)

# C4.3 bounded temporal-cluster selector.  The default experiment groups
# violations over a maximum span of two original MPC prediction steps, then
# spatially clusters each temporal layer while steering all points in the
# earliest unsafe layer together.
add_executable(trajectory_vbc_selector_temporal_cluster_node
  src/trajectory_vbc_selector_temporal_cluster_node.cpp
)

add_dependencies(trajectory_vbc_selector_temporal_cluster_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(trajectory_vbc_selector_temporal_cluster_node
  trajectory_risk_evaluator
  ${catkin_LIBRARIES}
  yaml-cpp
  pinocchio::pinocchio
)

# Exact point-level selector source frozen before the C4.3 risk-region work.
# Kept as a separate executable so C4.2 can be reproduced without routing
# through any of the temporal-layer or clustering semantics.
add_executable(trajectory_vbc_selector_legacy_node
  src/trajectory_vbc_selector_legacy_node.cpp
)

add_dependencies(trajectory_vbc_selector_legacy_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(trajectory_vbc_selector_legacy_node
  trajectory_risk_evaluator
  ${catkin_LIBRARIES}
  yaml-cpp
  pinocchio::pinocchio
)

add_executable(predicted_vbc_auditor_node
  src/predicted_vbc_auditor_node.cpp
)

add_dependencies(predicted_vbc_auditor_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(predicted_vbc_auditor_node
  trajectory_risk_evaluator
  ${catkin_LIBRARIES}
  yaml-cpp
  pinocchio::pinocchio
)

catkin_install_python(PROGRAMS
  scripts/test_query_confidence.py
  scripts/self_filter_coverage_marker_publisher.py
  scripts/runtime_self_hit_rviz_diagnostic.py
  scripts/phase_e_case026_exact_self_occlusion_rviz.py
  scripts/diagnose_gazebo_tf_frame_delta.py
  scripts/initialize_body_prior_once.py
  scripts/executed_vbc_logger_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Legacy-only implementations, with unchanged rosrun executable basenames.
# Install the real files, not the source compatibility symlinks: those links
# are relative to the source package layout, not lib/<package>.
catkin_install_python(PROGRAMS
  legacy/scripts/generate_body_samples.py
  legacy/scripts/body_sample_marker_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# config/body_samples.yaml is a relative alias to this canonical asset.
install(DIRECTORY legacy/config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/legacy)

# Importable helper, not a catkin executable relay (relays do not export module symbols).
install(FILES scripts/primitive_point_geometry.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(TARGETS
  body_sample_model
  trajectory_risk_evaluator
  confidence_map_node
  tof_fusion_self_filter_node
  body_sample_model_node
  body_observability_node
  trajectory_risk_node
  trajectory_vbc_selector_node
  trajectory_vbc_selector_temporal_cluster_node
  trajectory_vbc_selector_legacy_node
  predicted_vbc_auditor_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  launch
  config
  srv
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
