#include <care_confidence_map/trajectory_risk_evaluator.hpp>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <sstream>
#include <unordered_set>

namespace care_confidence_map
{

bool TrajectoryRiskEvaluator::initialize(
    const std::string& robot_urdf_file,
    const std::string& body_samples_file,
    const std::string& base_frame,
    std::string* error_msg)
{
  initialized_ = false;

  robot_urdf_file_ = robot_urdf_file;
  body_samples_file_ = body_samples_file;
  base_frame_ = base_frame;

  if (!buildPinocchioModel(robot_urdf_file_, error_msg))
  {
    return false;
  }

  std::string body_error;
  if (!body_sample_model_.loadFromYaml(body_samples_file_, &body_error))
  {
    if (error_msg)
    {
      *error_msg = "Failed to load body samples: " + body_error;
    }
    return false;
  }

  if (!validateBodySampleFrames(error_msg))
  {
    return false;
  }

  extractActiveJointNames();

  initialized_ = true;
  return true;
}

bool TrajectoryRiskEvaluator::buildPinocchioModel(
    const std::string& robot_urdf_file,
    std::string* error_msg)
{
  try
  {
    pinocchio::urdf::buildModel(robot_urdf_file, model_);
    data_ = pinocchio::Data(model_);
    return true;
  }
  catch (const std::exception& e)
  {
    if (error_msg)
    {
      std::ostringstream oss;
      oss << "Pinocchio failed to build model from URDF: "
          << robot_urdf_file
          << ". Exception: "
          << e.what();
      *error_msg = oss.str();
    }
    return false;
  }
}

void TrajectoryRiskEvaluator::extractActiveJointNames()
{
  active_joint_names_.clear();

  // Pinocchio joint 0 is universe.
  for (pinocchio::JointIndex jid = 1; jid < model_.njoints; ++jid)
  {
    active_joint_names_.push_back(model_.names[jid]);
  }
}

bool TrajectoryRiskEvaluator::validateBodySampleFrames(
    std::string* error_msg) const
{
  std::unordered_set<std::string> missing_frames;

  for (const auto& frame : body_sample_model_.frames())
  {
    if (!model_.existFrame(frame))
    {
      missing_frames.insert(frame);
    }
  }

  if (!model_.existFrame(base_frame_))
  {
    missing_frames.insert(base_frame_);
  }

  if (!missing_frames.empty())
  {
    if (error_msg)
    {
      std::ostringstream oss;
      oss << "Some body sample frames do not exist in Pinocchio model:";
      for (const auto& name : missing_frames)
      {
        oss << " " << name;
      }
      *error_msg = oss.str();
    }
    return false;
  }

  return true;
}

bool TrajectoryRiskEvaluator::checkConfigurationSize(
    const Eigen::VectorXd& q,
    std::string* error_msg) const
{
  if (q.size() != model_.nq)
  {
    if (error_msg)
    {
      std::ostringstream oss;
      oss << "Invalid q size. Expected model.nq="
          << model_.nq
          << ", got "
          << q.size();
      *error_msg = oss.str();
    }
    return false;
  }

  return true;
}

bool TrajectoryRiskEvaluator::computeSamplesForConfiguration(
    const Eigen::VectorXd& q,
    int timestep_index,
    TrajectoryFrameSamples* out,
    std::string* error_msg) const
{
  if (!initialized_)
  {
    if (error_msg)
    {
      *error_msg = "TrajectoryRiskEvaluator is not initialized.";
    }
    return false;
  }

  if (!out)
  {
    if (error_msg)
    {
      *error_msg = "Output pointer is null.";
    }
    return false;
  }

  if (!checkConfigurationSize(q, error_msg))
  {
    return false;
  }

  out->timestep_index = timestep_index;
  out->q = q;
  out->samples.clear();
  out->samples.reserve(body_sample_model_.riskSampleCount());

  pinocchio::forwardKinematics(model_, data_, q);
  pinocchio::updateFramePlacements(model_, data_);

  const pinocchio::FrameIndex base_fid = model_.getFrameId(base_frame_);
  const pinocchio::SE3& T_world_base = data_.oMf[base_fid];
  const pinocchio::SE3 T_base_world = T_world_base.inverse();

  for (const auto& sample : body_sample_model_.samples())
  {
    if (!sample.include_for_risk)
    {
      continue;
    }

    if (!model_.existFrame(sample.frame_name))
    {
      continue;
    }

    const pinocchio::FrameIndex fid = model_.getFrameId(sample.frame_name);
    const pinocchio::SE3& T_world_link = data_.oMf[fid];

    const Eigen::Vector3d center_link(
        sample.center_link.x(),
        sample.center_link.y(),
        sample.center_link.z());

    const Eigen::Vector3d center_world =
        T_world_link.act(center_link);

    const Eigen::Vector3d center_base =
        T_base_world.act(center_world);

    TrajectoryBodySample out_sample;
    out_sample.timestep_index = timestep_index;

    out_sample.link_name = sample.link_name;
    out_sample.frame_name = sample.frame_name;

    out_sample.center_base = center_base;
    out_sample.radius = sample.radius;

    out_sample.source_type = sample.source_type;
    out_sample.source_collision_index = sample.source_collision_index;
    out_sample.sample_index_in_link = sample.sample_index_in_link;

    out_sample.include_for_risk = sample.include_for_risk;

    out->samples.push_back(out_sample);
  }

  return true;
}

TrajectorySampleResult TrajectoryRiskEvaluator::computeTrajectorySamples(
    const std::vector<Eigen::VectorXd>& q_traj) const
{
  TrajectorySampleResult result;

  if (!initialized_)
  {
    result.success = false;
    result.message = "TrajectoryRiskEvaluator is not initialized.";
    return result;
  }

  if (q_traj.empty())
  {
    result.success = false;
    result.message = "Input q_traj is empty.";
    return result;
  }

  result.frames.clear();
  result.frames.reserve(q_traj.size());

  for (std::size_t k = 0; k < q_traj.size(); ++k)
  {
    TrajectoryFrameSamples frame_samples;
    std::string error_msg;

    if (!computeSamplesForConfiguration(
            q_traj[k],
            static_cast<int>(k),
            &frame_samples,
            &error_msg))
    {
      result.success = false;

      std::ostringstream oss;
      oss << "Failed at timestep "
          << k
          << ": "
          << error_msg;
      result.message = oss.str();

      return result;
    }

    result.total_samples +=
        static_cast<int>(frame_samples.samples.size());

    result.frames.push_back(frame_samples);
  }

  result.num_timesteps =
      static_cast<int>(result.frames.size());

  if (!result.frames.empty())
  {
    result.num_samples_per_timestep =
        static_cast<int>(result.frames.front().samples.size());
  }

  result.success = true;
  result.message = "OK";

  return result;
}

}  // namespace care_confidence_map