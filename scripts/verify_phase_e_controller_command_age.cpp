// Exercise the installed production controller update with fake joint handles.
// No hardware, Gazebo, ROS master, or physical deceleration claim.
#include <velocity_controllers/joint_group_velocity_controller.h>
#include <cmath>
#include <iostream>

int main() {
  velocity_controllers::JointGroupVelocityController controller;
  double position[7] = {}, velocity[7] = {}, effort[7] = {}, command[7] = {};
  controller.n_joints_ = 7;
  for (int i = 0; i < 7; ++i) {
    hardware_interface::JointStateHandle state("j" + std::to_string(i),
                                               &position[i], &velocity[i], &effort[i]);
    controller.joints_.emplace_back(state, &command[i]);
  }
  controller.commands_buffer_.writeFromNonRT(std::vector<double>(7, .6));
  for (int tick = 0; tick <= 2000; ++tick) {
    controller.update(ros::Time(100. + .001 * tick), ros::Duration(.001));
    for (double c : command) if (std::abs(c - .6) > 1e-12) return 1;
  }
  controller.commands_buffer_.writeFromNonRT(std::vector<double>(7, 0.));
  controller.update(ros::Time(103.), ros::Duration(.001));
  for (double c : command) if (c != 0.) return 2;
  std::cout << "{\"status\":\"PASS\",\"stale_command_retained_for_simulated_s\":2,"
               "\"updates_without_new_input\":2001,\"scope\":\"Installed production controller update, fake joint handles\"}\n";
}
