cmake_minimum_required(VERSION 3.0.2)
project(arm_trajectory)

add_compile_options(-std=c++14 -Wall -Wextra -Wpedantic)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
)

find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES arm_trajectory
  CATKIN_DEPENDS roscpp std_msgs
  DEPENDS EIGEN3
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_library(arm_trajectory
  src/joint_trajectory.cpp
)

add_dependencies(arm_trajectory
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(arm_trajectory
  ${catkin_LIBRARIES}
)