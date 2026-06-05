cmake_minimum_required(VERSION 3.0.2)
project(egocentric_arm_planner)

add_compile_options(-std=c++14 -Wall -Wextra -Wpedantic)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  trajectory_msgs
  arm_model
  arm_trajectory
)

find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES egocentric_arm_planner
  CATKIN_DEPENDS
    roscpp
    std_msgs
    sensor_msgs
    geometry_msgs
    trajectory_msgs
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