#include "arm_model/sensor_frame_manager.hpp"

#include <XmlRpcValue.h>

namespace arm_model {

bool SensorFrameManager::loadFromRosParam(const ros::NodeHandle& nh,
                                          const std::string& param_name) {
  sensors_.clear();

  XmlRpc::XmlRpcValue sensor_list;
  if (!nh.getParam(param_name, sensor_list)) {
    ROS_WARN_STREAM("[SensorFrameManager] No sensor list found at param: " << nh.resolveName(param_name));
    return false;
  }

  if (sensor_list.getType() != XmlRpc::XmlRpcValue::TypeArray) {
    ROS_ERROR_STREAM("[SensorFrameManager] Param '" << param_name << "' must be a list.");
    return false;
  }

  for (int i = 0; i < sensor_list.size(); ++i) {
    if (sensor_list[i].getType() != XmlRpc::XmlRpcValue::TypeStruct) {
      ROS_ERROR_STREAM("[SensorFrameManager] Sensor entry " << i << " is not a struct.");
      return false;
    }

    SensorFrame sensor;
    sensor.id = i;

    if (sensor_list[i].hasMember("id")) {
      sensor.id = static_cast<int>(sensor_list[i]["id"]);
    }

    if (!sensor_list[i].hasMember("name") || !sensor_list[i].hasMember("frame")) {
      ROS_ERROR_STREAM("[SensorFrameManager] Sensor entry " << i
                       << " must contain 'name' and 'frame'.");
      return false;
    }

    sensor.name = static_cast<std::string>(sensor_list[i]["name"]);
    sensor.frame = static_cast<std::string>(sensor_list[i]["frame"]);

    sensors_.push_back(sensor);
  }

  ROS_INFO_STREAM("[SensorFrameManager] Loaded " << sensors_.size() << " sensor frames.");
  return true;
}

const std::vector<SensorFrame>& SensorFrameManager::sensors() const {
  return sensors_;
}

std::size_t SensorFrameManager::size() const {
  return sensors_.size();
}

bool SensorFrameManager::empty() const {
  return sensors_.empty();
}

bool SensorFrameManager::getSensorById(int id, SensorFrame& sensor) const {
  for (const auto& s : sensors_) {
    if (s.id == id) {
      sensor = s;
      return true;
    }
  }
  return false;
}

bool SensorFrameManager::getSensorByName(const std::string& name, SensorFrame& sensor) const {
  for (const auto& s : sensors_) {
    if (s.name == name) {
      sensor = s;
      return true;
    }
  }
  return false;
}

}  // namespace arm_model