cmake_minimum_required(VERSION 3.0.2)
project(care_visibility_cdf)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/validate_visibility_oracle.py
  scripts/generate_visibility_raw_samples.py
  scripts/check_visibility_raw_samples.py
  scripts/extract_visibility_zero_level_sets.py
  scripts/check_visibility_oracle_consistency.py
  scripts/check_visibility_gradient.py
  scripts/visualize_visibility_q0_rviz.py
  scripts/check_visibility_self_occlusion.py
  scripts/annotate_visibility_raw_self_occlusion.py
  scripts/generate_visibility_cdf_labels.py
  scripts/train_visibility_cdf.py
  scripts/eval_visibility_cdf.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
