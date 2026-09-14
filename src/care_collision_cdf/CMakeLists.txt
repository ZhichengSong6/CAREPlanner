cmake_minimum_required(VERSION 3.0.2)
project(care_collision_cdf)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  message_generation
  rospy
  std_msgs
  sensor_msgs
  care_confidence_map
)

add_message_files(
  FILES
  CollisionCDFConstraintBatch.msg
  CollisionCDFWitnessRequest.msg
  CollisionCDFWitnessResponse.msg
  CollisionCDFRejectionFeedback.msg
)

add_service_files(
  FILES
  QueryCollisionCDF.srv
  QueryCollisionCDFPairs.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs message_runtime roscpp rospy std_msgs sensor_msgs care_confidence_map
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/collision_cdf_model.py
  scripts/collision_cdf_server_node.py
  scripts/inspect_collision_cdf_checkpoint.py
  scripts/evaluate_collision_cdf_gradient.py
  scripts/query_collision_cdf_smoke.py
  scripts/multi_step_forbidden_space_cdf_shadow_node.py
  scripts/benchmark_collision_cdf_pairs_gpu.py
  scripts/collision_cdf_gpu_worker.py
  scripts/multi_step_forbidden_space_cdf_gpu_shadow_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
# Preserve historical packaging unless the deployment build explicitly opts out.
# Release acceptance supplies an external, SHA-bound checkpoint instead.
option(CARE_INSTALL_CHECKPOINTS "Bundle collision checkpoints in install space" ON)
if(CARE_INSTALL_CHECKPOINTS)
  install(DIRECTORY checkpoints DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
endif()


add_executable(cpp_forbidden_voxel_gpu_shadow_node
  src/cpp_forbidden_voxel_gpu_shadow_node.cpp
)

add_dependencies(cpp_forbidden_voxel_gpu_shadow_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(cpp_forbidden_voxel_gpu_shadow_node
  ${catkin_LIBRARIES}
)

if(CATKIN_ENABLE_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/test_phase_e_gcdf_primitives.cpp")
  find_package(pinocchio REQUIRED)
  add_executable(test_gcdf_primitives ../../scripts/test_phase_e_gcdf_primitives.cpp)
  add_dependencies(test_gcdf_primitives ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  target_link_libraries(test_gcdf_primitives ${catkin_LIBRARIES} pinocchio::pinocchio)
endif()

install(TARGETS
  cpp_forbidden_voxel_gpu_shadow_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
