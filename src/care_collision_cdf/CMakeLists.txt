cmake_minimum_required(VERSION 3.0.2)
project(care_collision_cdf)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  rospy
  std_msgs
)

add_service_files(
  FILES
  QueryCollisionCDF.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs message_runtime rospy std_msgs
)

catkin_install_python(PROGRAMS
  scripts/collision_cdf_model.py
  scripts/collision_cdf_server_node.py
  scripts/inspect_collision_cdf_checkpoint.py
  scripts/evaluate_collision_cdf_gradient.py
  scripts/query_collision_cdf_smoke.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch checkpoints
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
