cmake_minimum_required(VERSION 3.0.2)
project(care_visibility_cdf)

# The visibility-CDF package currently contains offline data-generation,
# training, and evaluation tools.  These scripts are run directly from the
# source tree and are not ROS runtime executables, so they do not need to be
# installed into the catkin devel space.
#
# Keep this CMakeLists intentionally minimal.  When runtime nodes or scripts
# (for example the planned L4CasADi NCDF interface) are added, install only
# those explicit entry points here.
find_package(catkin REQUIRED)

catkin_package()
