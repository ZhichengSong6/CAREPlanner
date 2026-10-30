# Publication build integration for sources previously compiled independently.
find_package(OpenSSL REQUIRED)
find_package(hpp-fcl REQUIRED)
target_link_libraries(local_sparse_scp_planner_node OpenSSL::Crypto hpp-fcl::hpp-fcl)
add_library(known_ground SHARED src/known_ground_kernel.cpp)
set_target_properties(known_ground PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION}")
install(TARGETS known_ground LIBRARY DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
