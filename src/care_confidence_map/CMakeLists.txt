cmake_minimum_required(VERSION 3.0.2)
project(care_confidence_map)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  trajectory_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
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
    sensor_msgs
    geometry_msgs
    trajectory_msgs
    visualization_msgs
    tf2
    tf2_ros
    tf2_geometry_msgs
    pcl_ros
    pcl_conversions
    message_runtime
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
)

# ============================================================
# Libraries
# ============================================================

add_library(body_sample_model
  src/body_sample_model.cpp
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
)

add_dependencies(trajectory_risk_evaluator
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(trajectory_risk_evaluator
  body_sample_model
  ${catkin_LIBRARIES}
  yaml-cpp
  pinocchio::pinocchio
)

# ============================================================
# Nodes
# ============================================================

add_executable(confidence_map_node
  src/confidence_map_node.cpp
)

add_dependencies(confidence_map_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(confidence_map_node
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
  body_sample_model
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
  body_sample_model
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
  body_sample_model
  ${catkin_LIBRARIES}
  yaml-cpp
  pinocchio::pinocchio
)

# ============================================================
# Python tools
# ============================================================

catkin_install_python(PROGRAMS
  scripts/test_query_confidence.py
  scripts/generate_body_samples.py
  scripts/body_sample_marker_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# ============================================================
# Install
# ============================================================

install(TARGETS
  body_sample_model
  trajectory_risk_evaluator
  confidence_map_node
  body_sample_model_node
  body_observability_node
  trajectory_risk_node
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