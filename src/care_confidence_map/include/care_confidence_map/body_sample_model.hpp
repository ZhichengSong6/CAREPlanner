#pragma once

#include <string>
#include <vector>

#include <tf2/LinearMath/Vector3.h>

namespace care_confidence_map
{

struct BodySample
{
  std::string link_name;
  std::string frame_name;

  tf2::Vector3 center_link;
  double radius = 0.0;

  std::string source_type;
  int source_collision_index = -1;
  int sample_index_in_link = -1;

  bool include_for_risk = true;
};

struct WorldBodySample
{
  std::string link_name;
  std::string frame_name;

  tf2::Vector3 center_base;
  double radius = 0.0;

  std::string source_type;
  int source_collision_index = -1;
  int sample_index_in_link = -1;

  bool include_for_risk = true;
};

class BodySampleModel
{
public:
  BodySampleModel() = default;

  bool loadFromYaml(const std::string& yaml_file, std::string* error_msg = nullptr);

  const std::vector<BodySample>& samples() const
  {
    return samples_;
  }

  const std::vector<std::string>& frames() const
  {
    return frames_;
  }

  std::size_t size() const
  {
    return samples_.size();
  }

  std::size_t riskSampleCount() const;

  void clear();

private:
  void rebuildFrameList();

private:
  std::vector<BodySample> samples_;
  std::vector<std::string> frames_;
};

}  // namespace care_confidence_map