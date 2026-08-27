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
  CATKIN_DEPENDS geometry_msgs message_runtime roscpp rospy std_msgs sensor_msgs
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

install(DIRECTORY config launch checkpoints
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)


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

install(TARGETS
  cpp_forbidden_voxel_gpu_shadow_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
