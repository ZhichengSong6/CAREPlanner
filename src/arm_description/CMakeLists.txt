cmake_minimum_required(VERSION 3.0.2)
project(arm_description)

find_package(catkin REQUIRED)

catkin_package()

install(DIRECTORY urdf meshes launch config rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)