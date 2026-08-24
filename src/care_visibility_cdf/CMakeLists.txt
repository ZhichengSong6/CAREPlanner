cmake_minimum_required(VERSION 3.0.2)
project(care_visibility_cdf)

find_package(catkin REQUIRED)

catkin_package()

# Runtime ROS entry points. Offline training/evaluation helpers remain in the
# source tree and are imported by these nodes through the package source path.
catkin_install_python(PROGRAMS
  scripts/ncdf_active_sensing_dryrun_node.py
  scripts/ncdf_horizon_observer_node.py
  scripts/ncdf_action_qp_dryrun_node.py
  scripts/ncdf_action_qp_logger_node.py
  scripts/ncdf_stage3_trajectory_qp_node.py
  scripts/vbc_deadline_projection_smoke_node.py
  scripts/vbc_deadline_waypoint_node.py
  scripts/vbc_deadline_waypoint_rolling_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
