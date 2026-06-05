cmake_minimum_required(VERSION 3.0.2)
project(arm_model)

add_compile_options(-std=c++14 -Wall -Wextra -Wpedantic)

add_definitions(
  -DBOOST_MPL_LIMIT_LIST_SIZE=30
  -DBOOST_MPL_LIMIT_VECTOR_SIZE=30
)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  urdf
)

find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES arm_model
  CATKIN_DEPENDS
    roscpp
    sensor_msgs
    geometry_msgs
    tf2
    tf2_ros
    urdf
  DEPENDS EIGEN3 pinocchio
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${pinocchio_INCLUDE_DIRS}
)

add_library(arm_model
  src/robot_model.cpp
  src/sensor_frame_manager.cpp
)

add_dependencies(arm_model
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(arm_model
  ${catkin_LIBRARIES}
  ${pinocchio_LIBRARIES}
)