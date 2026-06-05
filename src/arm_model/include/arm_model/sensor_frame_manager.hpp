#pragma once

#include <ros/ros.h>

#include <string>
#include <vector>

namespace arm_model {

struct SensorFrame {
  int id = -1;
  std::string name;
  std::string frame;
};

class SensorFrameManager {
public:
  SensorFrameManager() = default;

  bool loadFromRosParam(const ros::NodeHandle& nh,
                        const std::string& param_name = "sensors");

  const std::vector<SensorFrame>& sensors() const;
  std::size_t size() const;
  bool empty() const;

  bool getSensorById(int id, SensorFrame& sensor) const;
  bool getSensorByName(const std::string& name, SensorFrame& sensor) const;

private:
  std::vector<SensorFrame> sensors_;
};

}  // namespace arm_model