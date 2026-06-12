#include <care_confidence_map/body_sample_model.hpp>

#include <algorithm>
#include <sstream>
#include <unordered_set>

#include <yaml-cpp/yaml.h>

namespace care_confidence_map
{

void BodySampleModel::clear()
{
  samples_.clear();
  frames_.clear();
}

std::size_t BodySampleModel::riskSampleCount() const
{
  std::size_t count = 0;
  for (const auto& s : samples_)
  {
    if (s.include_for_risk)
    {
      ++count;
    }
  }
  return count;
}

void BodySampleModel::rebuildFrameList()
{
  frames_.clear();

  std::unordered_set<std::string> seen;
  for (const auto& s : samples_)
  {
    if (s.frame_name.empty())
    {
      continue;
    }

    if (seen.insert(s.frame_name).second)
    {
      frames_.push_back(s.frame_name);
    }
  }

  std::sort(frames_.begin(), frames_.end());
}

bool BodySampleModel::loadFromYaml(const std::string& yaml_file,
                                   std::string* error_msg)
{
  clear();

  try
  {
    YAML::Node root = YAML::LoadFile(yaml_file);

    if (!root["body_sampling"])
    {
      if (error_msg)
      {
        *error_msg = "Missing root key: body_sampling";
      }
      return false;
    }

    YAML::Node body_sampling = root["body_sampling"];

    if (!body_sampling["links"] || !body_sampling["links"].IsSequence())
    {
      if (error_msg)
      {
        *error_msg = "Missing or invalid key: body_sampling.links";
      }
      return false;
    }

    YAML::Node links = body_sampling["links"];

    for (std::size_t li = 0; li < links.size(); ++li)
    {
      YAML::Node link_node = links[li];

      if (!link_node["link_name"])
      {
        continue;
      }

      const std::string link_name =
          link_node["link_name"].as<std::string>();

      std::string frame_name = link_name;
      if (link_node["frame"])
      {
        frame_name = link_node["frame"].as<std::string>();
      }

      bool include_for_risk = true;
      if (link_node["include_for_risk"])
      {
        include_for_risk = link_node["include_for_risk"].as<bool>();
      }

      if (!link_node["samples"] || !link_node["samples"].IsSequence())
      {
        continue;
      }

      YAML::Node samples = link_node["samples"];

      for (std::size_t si = 0; si < samples.size(); ++si)
      {
        YAML::Node sample_node = samples[si];

        if (!sample_node["center"] || !sample_node["center"].IsSequence())
        {
          continue;
        }

        YAML::Node center = sample_node["center"];
        if (center.size() != 3)
        {
          continue;
        }

        if (!sample_node["radius"])
        {
          continue;
        }

        BodySample sample;

        sample.link_name = link_name;
        sample.frame_name = frame_name;

        sample.center_link = tf2::Vector3(
            center[0].as<double>(),
            center[1].as<double>(),
            center[2].as<double>());

        sample.radius = sample_node["radius"].as<double>();

        sample.source_type = "unknown";
        if (sample_node["source_type"])
        {
          sample.source_type = sample_node["source_type"].as<std::string>();
        }

        sample.source_collision_index = -1;
        if (sample_node["source_collision_index"])
        {
          sample.source_collision_index =
              sample_node["source_collision_index"].as<int>();
        }

        sample.sample_index_in_link = static_cast<int>(si);
        sample.include_for_risk = include_for_risk;

        samples_.push_back(sample);
      }
    }

    rebuildFrameList();

    if (samples_.empty())
    {
      if (error_msg)
      {
        *error_msg = "No valid body samples loaded.";
      }
      return false;
    }

    return true;
  }
  catch (const YAML::Exception& e)
  {
    if (error_msg)
    {
      std::ostringstream oss;
      oss << "YAML exception while loading " << yaml_file << ": " << e.what();
      *error_msg = oss.str();
    }
    return false;
  }
  catch (const std::exception& e)
  {
    if (error_msg)
    {
      std::ostringstream oss;
      oss << "Exception while loading " << yaml_file << ": " << e.what();
      *error_msg = oss.str();
    }
    return false;
  }
}

}  // namespace care_confidence_map