#include <care_confidence_map/trajectory_risk_evaluator.hpp>
#include <care_confidence_map/legacy_body_backend.hpp>
#include <care_confidence_map/vbc_primitive_model.hpp>
#include <care_confidence_map/primitive_probe_grid.hpp>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <cmath>
#include <limits>
#include <sstream>
#include <unordered_set>

namespace care_confidence_map
{

bool TrajectoryRiskEvaluator::initializeKinematics(
    const std::string& robot_urdf_file,
    const std::string& base_frame,
    std::string* error_msg)
{
  initialized_ = false;
  primitive_backend_ = false;
  primitive_local_.clear();
  fast_audit_prepared_ = false;
  fast_audit_sensor_frame_ids_.clear();
  fast_audit_body_frames_.clear();
  fast_audit_body_sample_count_ = 0;

  robot_urdf_file_ = robot_urdf_file;
  body_samples_file_.clear();
  legacy_body_samples_.reset();
  base_frame_ = base_frame;

  if (!buildPinocchioModel(robot_urdf_file_, error_msg))
  {
    return false;
  }

  if (!model_.existFrame(base_frame_))
  {
    if (error_msg) *error_msg = "Missing base frame: " + base_frame_;
    return false;
  }
  extractActiveJointNames();
  initialized_ = true;
  return true;
}

bool TrajectoryRiskEvaluator::initialize(
    const std::string& robot_urdf_file,
    const std::string& body_samples_file,
    const std::string& base_frame,
    std::string* error_msg)
{
  if (!initializeKinematics(robot_urdf_file, base_frame, error_msg)) return false;
  initialized_ = false;
  body_samples_file_ = body_samples_file;

  std::string body_error;
  auto legacy = std::make_shared<BodySampleModel>();
  if (!loadLegacyBodySamples(legacy.get(), body_samples_file_, &body_error))
  {
    if (error_msg)
    {
      *error_msg = "Failed to load body samples: " + body_error;
    }
    return false;
  }

  legacy_body_samples_ = std::move(legacy);
  if (!validateBodySampleFrames(error_msg))
  {
    return false;
  }

  extractActiveJointNames();

  initialized_ = true;
  return true;
}

bool TrajectoryRiskEvaluator::initializePrimitives(const std::string& urdf,
    const std::string& collision_urdf,const std::string& base,double resolution,
    const Eigen::Vector3d& origin,std::string* error) {
  if(!initializeKinematics(urdf,base,error))return false;
  initialized_=false;
  if(!std::isfinite(resolution) || resolution<=0 || !origin.allFinite()) {
    if(error)*error="Invalid primitive diagnostic grid";return false;
  }
  VbcPrimitiveModel geometry;
  // Match existing YAML risk semantics: base excluded, link1 included. Consumers
  // still apply their original ignored_risk_links (normally base + link1).
  if(!geometry.load(collision_urdf,{"base_link"},error))return false;
  for(const auto& frame:geometry.frames()) if(!model_.existFrame(frame)) {
    if(error)*error="Missing primitive frame: "+frame;return false;
  }
  primitive_local_=geometry.primitives();primitive_probe_resolution_=resolution;
  primitive_probe_origin_=origin;primitive_backend_=true;initialized_=true;return true;
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

  for (const auto& frame : bodySampleModel().frames())
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
  if (q.size() != model_.nq || !q.allFinite())
  {
    if (error_msg)
    {
      std::ostringstream oss;
      oss << "Invalid q (size or nonfinite). Expected model.nq="
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
  if (!initialized_ || (!primitive_backend_ && legacyRiskSampleCount(bodySampleModel()) == 0))
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
  if (!primitive_backend_) out->samples.reserve(legacyRiskSampleCount(bodySampleModel()));

  pinocchio::forwardKinematics(model_, data_, q);
  pinocchio::updateFramePlacements(model_, data_);

  const pinocchio::FrameIndex base_fid = model_.getFrameId(base_frame_);
  const pinocchio::SE3& T_world_base = data_.oMf[base_fid];
  const pinocchio::SE3 T_base_world = T_world_base.inverse();

  if(primitive_backend_) {
    try {
      std::vector<VbcPrimitive> world;world.reserve(primitive_local_.size());
      pinocchio::SE3 pose;
      std::size_t previous_frame=std::numeric_limits<std::size_t>::max();
      for(const auto& local:primitive_local_) {
        if(local.frame_index!=previous_frame) {
          pose=T_base_world*data_.oMf[model_.getFrameId(local.link_name)];previous_frame=local.frame_index;
        }
        auto p=local;p.center=pose.act(local.center);
        const double* a=pose.rotation().data();const double* b=local.rotation.data();double* r=p.rotation.data();
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)
          r[3*j+i]=a[i]*b[3*j]+a[3+i]*b[3*j+1]+a[6+i]*b[3*j+2];
        world.push_back(std::move(p));
      }
      for(const auto& probe:primitiveProbeGrid(world,primitive_probe_resolution_,primitive_probe_origin_)) {
        TrajectoryBodySample p;p.timestep_index=timestep_index;p.link_name=p.frame_name=probe.link_name;
        p.center_base=probe.point;p.radius=0.;p.source_type=probe.source_type;
        p.source_collision_index=probe.collision_index;p.sample_index_in_link=-1;
        out->samples.push_back(std::move(p));
      }
      return true;
    } catch(const std::exception& e) {out->samples.clear();if(error_msg)*error_msg=e.what();return false;}
  }

  for (const auto& sample : bodySampleModel().samples())
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

bool TrajectoryRiskEvaluator::computeFramePosesForConfiguration(
    const Eigen::VectorXd& q,
    const std::vector<std::string>& frame_names,
    std::vector<FramePoseInBase>* out,
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
      *error_msg = "Output frame-pose pointer is null.";
    }
    return false;
  }

  if (!checkConfigurationSize(q, error_msg))
  {
    return false;
  }

  for (const auto& frame_name : frame_names)
  {
    if (!model_.existFrame(frame_name))
    {
      if (error_msg)
      {
        *error_msg = "Requested frame does not exist in Pinocchio model: " + frame_name;
      }
      return false;
    }
  }

  pinocchio::forwardKinematics(model_, data_, q);
  pinocchio::updateFramePlacements(model_, data_);

  const pinocchio::FrameIndex base_fid = model_.getFrameId(base_frame_);
  const pinocchio::SE3 T_base_world = data_.oMf[base_fid].inverse();

  out->clear();
  out->reserve(frame_names.size());

  for (const auto& frame_name : frame_names)
  {
    const pinocchio::FrameIndex fid = model_.getFrameId(frame_name);
    const pinocchio::SE3 T_base_frame = T_base_world * data_.oMf[fid];

    FramePoseInBase pose;
    pose.frame_name = frame_name;
    pose.translation_base = T_base_frame.translation();
    pose.rotation_base = T_base_frame.rotation();
    out->push_back(pose);
  }

  return true;
}

bool TrajectoryRiskEvaluator::computeAuditGeometryForConfiguration(
    const Eigen::VectorXd& q,
    int timestep_index,
    const std::vector<std::string>& frame_names,
    ConfigurationAuditGeometry* out,
    std::string* error_msg) const
{
  if(primitive_backend_) {
    if(!out){if(error_msg)*error_msg="Null primitive audit output";return false;}
    out->body_samples.clear();out->frame_poses.clear();TrajectoryFrameSamples probes;
    if(!computeSamplesForConfiguration(q,timestep_index,&probes,error_msg) ||
       !computeFramePosesForConfiguration(q,frame_names,&out->frame_poses,error_msg))return false;
    out->timestep_index=timestep_index;out->body_samples=std::move(probes.samples);return true;
  }
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
      *error_msg = "Output audit-geometry pointer is null.";
    }
    return false;
  }

  if (!checkConfigurationSize(q, error_msg))
  {
    return false;
  }

  for (const auto& frame_name : frame_names)
  {
    if (!model_.existFrame(frame_name))
    {
      if (error_msg)
      {
        *error_msg = "Requested frame does not exist in Pinocchio model: " + frame_name;
      }
      return false;
    }
  }

  pinocchio::forwardKinematics(model_, data_, q);
  pinocchio::updateFramePlacements(model_, data_);

  const pinocchio::FrameIndex base_fid = model_.getFrameId(base_frame_);
  const pinocchio::SE3 T_base_world = data_.oMf[base_fid].inverse();

  out->timestep_index = timestep_index;
  out->body_samples.clear();
  out->body_samples.reserve(legacyRiskSampleCount(bodySampleModel()));
  out->frame_poses.clear();
  out->frame_poses.reserve(frame_names.size());

  for (const auto& sample : bodySampleModel().samples())
  {
    if (!sample.include_for_risk || !model_.existFrame(sample.frame_name))
    {
      continue;
    }

    const pinocchio::FrameIndex fid = model_.getFrameId(sample.frame_name);
    const Eigen::Vector3d center_link(
        sample.center_link.x(),
        sample.center_link.y(),
        sample.center_link.z());
    const Eigen::Vector3d center_world = data_.oMf[fid].act(center_link);

    TrajectoryBodySample out_sample;
    out_sample.timestep_index = timestep_index;
    out_sample.link_name = sample.link_name;
    out_sample.frame_name = sample.frame_name;
    out_sample.center_base = T_base_world.act(center_world);
    out_sample.radius = sample.radius;
    out_sample.source_type = sample.source_type;
    out_sample.source_collision_index = sample.source_collision_index;
    out_sample.sample_index_in_link = sample.sample_index_in_link;
    out_sample.include_for_risk = sample.include_for_risk;
    out->body_samples.push_back(out_sample);
  }

  for (const auto& frame_name : frame_names)
  {
    const pinocchio::FrameIndex fid = model_.getFrameId(frame_name);
    const pinocchio::SE3 T_base_frame = T_base_world * data_.oMf[fid];

    FramePoseInBase pose;
    pose.frame_name = frame_name;
    pose.translation_base = T_base_frame.translation();
    pose.rotation_base = T_base_frame.rotation();
    out->frame_poses.push_back(pose);
  }

  return true;
}

bool TrajectoryRiskEvaluator::prepareFastAudit(
    const std::vector<std::string>& sensor_frame_names,
    const std::vector<std::string>& ignored_risk_links,
    std::string* error_msg)
{
  if (!initialized_)
  {
    if (error_msg)
    {
      *error_msg = "TrajectoryRiskEvaluator is not initialized.";
    }
    return false;
  }

  fast_audit_prepared_ = false;
  fast_audit_sensor_frame_ids_.clear();
  fast_audit_body_frames_.clear();
  fast_audit_body_sample_count_ = 0;

  if (!model_.existFrame(base_frame_))
  {
    if (error_msg)
    {
      *error_msg = "Base frame does not exist: " + base_frame_;
    }
    return false;
  }
  fast_audit_base_frame_id_ = model_.getFrameId(base_frame_);

  fast_audit_sensor_frame_ids_.reserve(sensor_frame_names.size());
  for (const auto& frame_name : sensor_frame_names)
  {
    if (!model_.existFrame(frame_name))
    {
      if (error_msg)
      {
        *error_msg = "Requested sensor frame does not exist: " + frame_name;
      }
      return false;
    }
    fast_audit_sensor_frame_ids_.push_back(model_.getFrameId(frame_name));
  }

  const std::unordered_set<std::string> ignored(
      ignored_risk_links.begin(), ignored_risk_links.end());

  if(primitive_backend_) for(const auto& p:primitive_local_) {
    if(ignored.count(p.link_name))continue;
    const auto fid=model_.getFrameId(p.link_name);
    auto it=std::find_if(fast_audit_body_frames_.begin(),fast_audit_body_frames_.end(),
        [&](const auto& f){return f.frame_id==fid;});
    if(it==fast_audit_body_frames_.end()) {
      CachedAuditBodyFrame f;f.frame_id=fid;fast_audit_body_frames_.push_back(std::move(f));
      it=fast_audit_body_frames_.end()-1;
    }
    it->primitives.push_back(p);++fast_audit_body_sample_count_;
  }

  for (const auto& sample : bodySampleModel().samples())
  {
    if (!sample.include_for_risk || ignored.count(sample.link_name) != 0)
    {
      continue;
    }
    if (!model_.existFrame(sample.frame_name))
    {
      if (error_msg)
      {
        *error_msg = "Body sample frame does not exist: " + sample.frame_name;
      }
      return false;
    }

    const pinocchio::FrameIndex fid = model_.getFrameId(sample.frame_name);
    CachedAuditBodyFrame* group = nullptr;
    for (auto& existing : fast_audit_body_frames_)
    {
      if (existing.frame_id == fid)
      {
        group = &existing;
        break;
      }
    }
    if (!group)
    {
      CachedAuditBodyFrame created;
      created.frame_id = fid;
      fast_audit_body_frames_.push_back(created);
      group = &fast_audit_body_frames_.back();
    }

    CachedAuditBodySample cached;
    cached.center_link = Eigen::Vector3d(
        sample.center_link.x(),
        sample.center_link.y(),
        sample.center_link.z());
    cached.radius = sample.radius;
    group->samples.push_back(cached);
    ++fast_audit_body_sample_count_;
  }

  if (fast_audit_sensor_frame_ids_.empty() || fast_audit_body_sample_count_ == 0)
  {
    if (error_msg)
    {
      *error_msg = "Fast audit cache is empty.";
    }
    return false;
  }

  fast_audit_prepared_ = true;
  return true;
}

bool TrajectoryRiskEvaluator::evaluateFastAuditForConfiguration(
    const Eigen::VectorXd& q,
    const Eigen::Vector3d& target_base,
    std::vector<FastAuditSensorPose>* sensor_poses,
    double* min_clearance_m,
    std::string* error_msg) const
{
  if (!fast_audit_prepared_)
  {
    if (error_msg)
    {
      *error_msg = "Fast audit cache has not been prepared.";
    }
    return false;
  }
  if (!sensor_poses || !min_clearance_m)
  {
    if (error_msg)
    {
      *error_msg = "Fast audit output pointer is null.";
    }
    return false;
  }
  if (!target_base.allFinite() || !checkConfigurationSize(q, error_msg))
  {
    return false;
  }

  pinocchio::forwardKinematics(model_, data_, q);
  pinocchio::updateFramePlacements(model_, data_);

  const pinocchio::SE3 T_base_world =
      data_.oMf[fast_audit_base_frame_id_].inverse();

  sensor_poses->clear();
  if (sensor_poses->capacity() < fast_audit_sensor_frame_ids_.size())
  {
    sensor_poses->reserve(fast_audit_sensor_frame_ids_.size());
  }
  for (const auto fid : fast_audit_sensor_frame_ids_)
  {
    const pinocchio::SE3 T_base_sensor = T_base_world * data_.oMf[fid];
    FastAuditSensorPose pose;
    pose.translation_base = T_base_sensor.translation();
    pose.rotation_base = T_base_sensor.rotation();
    sensor_poses->push_back(pose);
  }

  double best = std::numeric_limits<double>::infinity();
  for (const auto& frame : fast_audit_body_frames_)
  {
    // One SE3 composition per body frame, then all local sample centers for
    // that link share the transform.  This replaces one frame lookup and one
    // world/base composition per body sample in the previous implementation.
    const pinocchio::SE3 T_base_link = T_base_world * data_.oMf[frame.frame_id];
    if(!frame.primitives.empty()) {
      const Eigen::Vector3d target_link=T_base_link.actInv(target_base);
      for(const auto& p:frame.primitives)best=std::min(best,p.signedDistance(target_link));
    }
    for (const auto& sample : frame.samples)
    {
      const Eigen::Vector3d center_base = T_base_link.act(sample.center_link);
      const double clearance =
          (center_base - target_base).norm() - sample.radius;
      if (clearance < best)
      {
        best = clearance;
      }
    }
  }

  if (!std::isfinite(best))
  {
    if (error_msg)
    {
      *error_msg = "Fast audit produced no finite body clearance.";
    }
    return false;
  }

  *min_clearance_m = best;
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
