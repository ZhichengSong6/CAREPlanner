cmake_minimum_required(VERSION 3.0.2)
project(care_visibility_cdf)

find_package(catkin REQUIRED)

catkin_package()

# Runtime ROS entry point.  Offline training/evaluation helpers remain in the
# source tree and are imported by this node through the package source path.
catkin_install_python(PROGRAMS
  scripts/ncdf_active_sensing_dryrun_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
