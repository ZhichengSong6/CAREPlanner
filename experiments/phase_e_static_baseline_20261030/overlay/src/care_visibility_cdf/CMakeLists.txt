cmake_minimum_required(VERSION 3.0.2)
project(care_visibility_cdf)

find_package(catkin REQUIRED)

catkin_package()

# Runtime ROS entry points. Install their import closure below as plain modules.
catkin_install_python(PROGRAMS
  scripts/ncdf_active_sensing_dryrun_node.py
  scripts/ncdf_horizon_observer_node.py
  scripts/ncdf_action_qp_dryrun_node.py
  scripts/ncdf_action_qp_logger_node.py
  scripts/ncdf_stage3_trajectory_qp_node.py
  scripts/vbc_deadline_projection_smoke_node.py
  scripts/vbc_deadline_waypoint_node.py
  scripts/vbc_deadline_waypoint_rolling_node.py
  scripts/vbc_deadline_waypoint_rolling_impl.py
  scripts/vbc_deadline_waypoint_rolling_persistent_impl.py
  scripts/vbc_deadline_waypoint_online_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

# Runtime code resolves helpers through rospkg's share/<package>/scripts.
# Two historical evaluation/export modules are runtime imports, not training jobs.
# No checkpoints, datasets, generated libraries or output logs are bundled here.
install(FILES
  scripts/bounded_tcpros.py
  scripts/pending_vbc_blockers.py
  scripts/pending_observation_relevance.py
  scripts/bounded_visibility_recovery.py
  scripts/check_visibility_self_occlusion.py
  scripts/evaluate_direct_vs_projection_ascent.py
  scripts/export_ncdf_l4casadi.py
  scripts/hierarchical_visibility_cdf_model.py
  scripts/ncdf_action_qp_core.py
  scripts/ncdf_action_qp_dryrun_node.py
  scripts/ncdf_action_qp_logger_node.py
  scripts/ncdf_active_sensing_dryrun_node.py
  scripts/ncdf_horizon_observer_node.py
  scripts/ncdf_stage3_trajectory_qp_core.py
  scripts/ncdf_stage3_trajectory_qp_node.py
  scripts/ncdf_stage3_trajectory_qp_runtime.py
  scripts/observation_identity.py
  scripts/observation_task_rejoin.py
  scripts/observation_corridor_cost.py
  scripts/rolling_task_observation.py
  scripts/rolling_visibility_guidance.py
  scripts/observation_activation_refresh.py
  scripts/observation_task_corridor.py
  scripts/native_view_selection.py
  scripts/per_sensor_visibility_runtime.py
  scripts/r1_visibility_model.py
  scripts/candidate_replacement_runtime.py
  scripts/test_ncdf_local_optimizer.py
  scripts/validate_visibility_oracle.py
  scripts/vbc_blocker_aware_acquisition_impl.py
  scripts/vbc_deadline_projection_smoke_node.py
  scripts/vbc_deadline_waypoint_node.py
  scripts/vbc_deadline_waypoint_online_node.py
  scripts/vbc_deadline_waypoint_rolling_impl.py
  scripts/vbc_deadline_waypoint_rolling_node.py
  scripts/vbc_deadline_waypoint_rolling_persistent_impl.py
  scripts/vbc_deadline_waypoint_sequential_impl.py
  scripts/vbc_fixed_visibility_diagnostic_impl.py
  scripts/vbc_multi_deadline_obligation_impl.py
  scripts/vbc_visibility_acquisition_impl.py
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts)
# Entry points also import siblings before consulting rospkg.
install(FILES
  scripts/bounded_tcpros.py
  scripts/evaluate_direct_vs_projection_ascent.py
  scripts/ncdf_action_qp_core.py
  scripts/ncdf_stage3_trajectory_qp_core.py
  scripts/ncdf_stage3_trajectory_qp_runtime.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(FILES scripts/pending_vbc_blockers.py scripts/pending_observation_relevance.py
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts)
