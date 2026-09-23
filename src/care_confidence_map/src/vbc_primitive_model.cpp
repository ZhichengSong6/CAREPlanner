#include <care_confidence_map/vbc_primitive_model.hpp>
#include <urdf/model.h>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <sstream>

namespace care_confidence_map {
namespace {

bool finiteVector3(const Eigen::Vector3d& value) {
  return std::isfinite(value.x()) && std::isfinite(value.y()) &&
         std::isfinite(value.z());
}

double vector3Norm(const Eigen::Vector3d& value) {
  return std::sqrt(value.x() * value.x() + value.y() * value.y() +
                   value.z() * value.z());
}

bool finiteQuaternionCoefficients(double x, double y, double z, double w) {
  return std::isfinite(x) && std::isfinite(y) && std::isfinite(z) &&
         std::isfinite(w);
}

bool finiteVector(const Eigen::VectorXd& value) {
  for (Eigen::Index i = 0; i < value.size(); ++i) {
    if (!std::isfinite(value[i])) return false;
  }
  return true;
}

Eigen::Vector3d rotateVector(const Eigen::Matrix3d& rotation,
                             const Eigen::Vector3d& value) {
  return Eigen::Vector3d(
      rotation(0, 0) * value.x() + rotation(0, 1) * value.y() +
          rotation(0, 2) * value.z(),
      rotation(1, 0) * value.x() + rotation(1, 1) * value.y() +
          rotation(1, 2) * value.z(),
      rotation(2, 0) * value.x() + rotation(2, 1) * value.y() +
          rotation(2, 2) * value.z());
}

double relativeRotationTrace(const Eigen::Matrix3d& measured,
                             const Eigen::Matrix3d& reference) {
  double trace = 0.0;
  for (int column = 0; column < 3; ++column) {
    for (int row = 0; row < 3; ++row) {
      trace += measured(row, column) * reference(row, column);
    }
  }
  return trace;
}

}  // namespace

bool VbcPrimitiveModel::load(const std::string& path,
                            const std::vector<std::string>& ignored, std::string* error) {
  frames_.clear(); primitives_.clear(); joint_radii_.clear();
  try {
    tinyxml2::XMLDocument xml;
    if (xml.LoadFile(path.c_str()) != tinyxml2::XML_SUCCESS || !xml.FirstChildElement("robot"))
      throw std::runtime_error("Cannot read primitive URDF XML: " + path);
    urdf::Model robot;
    if (!robot.initXml(&xml)) throw std::runtime_error("Cannot parse primitive URDF: " + path);
    // urdfdom can log an invalid collision and still return a valid link/model.
    // Compare the raw XML so a dropped solid cannot make VBC appear clear.
    for (auto* link = xml.FirstChildElement("robot")->FirstChildElement("link"); link;
         link = link->NextSiblingElement("link")) {
      const char* name = link->Attribute("name");
      if (!name || !robot.getLink(name)) throw std::runtime_error("Invalid URDF link");
      std::size_t count = 0;
      for (auto* c = link->FirstChildElement("collision"); c; c = c->NextSiblingElement("collision")) {
        ++count;
        const auto* g = c->FirstChildElement("geometry");
        if (!g || g->NextSiblingElement("geometry") || !g->FirstChildElement() ||
            g->FirstChildElement()->NextSiblingElement())
          throw std::runtime_error("Missing/ambiguous collision geometry: " + std::string(name));
      }
      if (robot.getLink(name)->collision_array.size() != count)
        throw std::runtime_error("URDF parser dropped collision geometry: " + std::string(name));
    }
    std::vector<VbcPrimitive> parsed;
    std::vector<std::string> frames;
    for (const auto& item : robot.links_) {
      const auto& link = *item.second;
      if (std::find(ignored.begin(), ignored.end(), link.name) != ignored.end()) continue;
      if (link.collision_array.empty()) continue;
      const std::size_t frame_index = frames.size();
      frames.push_back(link.name);
      for (std::size_t i = 0; i < link.collision_array.size(); ++i) {
        const auto& c = link.collision_array[i];
        if (!c || !c->geometry) throw std::runtime_error("Missing collision geometry: " + link.name);
        VbcPrimitive p;
        p.link_name = link.name; p.collision_name = c->name;
        p.collision_index = static_cast<int>(i); p.frame_index = frame_index;
        p.center = Eigen::Vector3d(c->origin.position.x, c->origin.position.y, c->origin.position.z);
        double x, y, z, w; c->origin.rotation.getQuaternion(x, y, z, w);
        if (!finiteVector3(p.center) ||
            !finiteQuaternionCoefficients(x, y, z, w))
          throw std::runtime_error("Invalid collision origin: " + link.name);
        const double quat_norm = std::sqrt(x * x + y * y + z * z + w * w);
        if (!std::isfinite(quat_norm) || quat_norm < 1e-12)
          throw std::runtime_error("Invalid collision rotation: " + link.name);
        x /= quat_norm; y /= quat_norm; z /= quat_norm; w /= quat_norm;
        p.rotation <<
            1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
            2.0 * (x * z + y * w), 2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
            2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y);
        switch (c->geometry->type) {
          case urdf::Geometry::BOX: {
            const auto& b = static_cast<const urdf::Box&>(*c->geometry);
            p.kind = PrimitiveKind::Box;
            p.half_size = 0.5 * Eigen::Vector3d(b.dim.x, b.dim.y, b.dim.z);
            if (!finiteVector3(p.half_size) || p.half_size.x() <= 0.0 ||
                p.half_size.y() <= 0.0 || p.half_size.z() <= 0.0)
              throw std::runtime_error("Invalid box dimensions: " + link.name);
            break;
          }
          case urdf::Geometry::CYLINDER: {
            const auto& b = static_cast<const urdf::Cylinder&>(*c->geometry);
            p.kind = PrimitiveKind::Cylinder; p.radius = b.radius; p.half_length = 0.5 * b.length;
            if (!std::isfinite(p.radius) || !std::isfinite(p.half_length) || p.radius <= 0 || p.half_length <= 0)
              throw std::runtime_error("Invalid cylinder dimensions: " + link.name);
            break;
          }
          case urdf::Geometry::SPHERE: {
            p.kind = PrimitiveKind::Sphere;
            p.radius = static_cast<const urdf::Sphere&>(*c->geometry).radius;
            if (!std::isfinite(p.radius) || p.radius <= 0)
              throw std::runtime_error("Invalid sphere radius: " + link.name);
            break;
          }
          default: throw std::runtime_error("Unsupported collision geometry (no fallback): " + link.name);
        }
        parsed.push_back(std::move(p));
      }
    }
    if (parsed.empty()) throw std::runtime_error("No active URDF collision primitives");
    std::vector<std::map<std::string, double>> radii;
    for (const auto& p : parsed) {
      // Triangle-inequality radius from each ancestor joint axis to ANY point
      // of the solid, for ALL configurations. Fixed-origin rotations preserve
      // norms. This bounds revolute arc travel even for simultaneous joints.
      double radius = vector3Norm(p.center) + (p.kind == PrimitiveKind::Box ? vector3Norm(p.half_size) :
          p.kind == PrimitiveKind::Cylinder ? std::hypot(p.radius, p.half_length) : p.radius);
      std::map<std::string, double> bounds;
      auto link = robot.getLink(p.link_name);
      while (link && link->parent_joint) {
        const auto& joint = *link->parent_joint;
        if (joint.mimic || (joint.type != urdf::Joint::FIXED &&
            joint.type != urdf::Joint::REVOLUTE && joint.type != urdf::Joint::CONTINUOUS))
          throw std::runtime_error("Continuous sweep requires independent revolute/fixed joints: " + joint.name);
        if (joint.type != urdf::Joint::FIXED) bounds[joint.name] = radius;
        const auto& pos = joint.parent_to_joint_origin_transform.position;
        radius += std::hypot(std::hypot(pos.x, pos.y), pos.z);
        link = robot.getLink(joint.parent_link_name);
      }
      radii.push_back(std::move(bounds));
    }
    joint_radii_ = std::move(radii);
    frames_ = std::move(frames); primitives_ = std::move(parsed);
    return true;
  } catch (const std::exception& e) {
    if (error) *error = e.what();
    return false;
  }
}

std::vector<double> VbcPrimitiveModel::displacementBounds(
    const std::vector<std::string>& names, const Eigen::VectorXd& delta) const {
  if (names.size() != static_cast<std::size_t>(delta.size()) || !finiteVector(delta) ||
      joint_radii_.size() != primitives_.size()) throw std::invalid_argument("Invalid motion-bound query");
  std::map<std::string, double> travel;
  for (std::size_t j=0; j<names.size(); ++j)
    if (!travel.emplace(names[j], std::abs(delta[j])).second) throw std::invalid_argument("Duplicate motion joint");
  std::vector<double> result;
  for (const auto& solid : joint_radii_) {
    double d=0.;
    for (const auto& joint : solid) {
      const auto it=travel.find(joint.first);
      if (it==travel.end()) throw std::invalid_argument("Missing motion joint: " + joint.first);
      d += joint.second * it->second;
    }
    result.push_back(d);
  }
  return result;
}

bool VbcPrimitiveModel::initializeRelativeFk(
    const std::string& urdf_file,
    const std::vector<std::string>& joint_names,
    std::string* error) {
  relative_fk_initialized_ = false;
  relative_fk_q_indices_.clear();
  relative_fk_frame_ids_.clear();
  relative_fk_primitive_radii_.clear();
  try {
    if (urdf_file.empty() || joint_names.empty() || primitives_.empty()) {
      throw std::invalid_argument("Relative FK requires URDF, joints and primitives");
    }
    pinocchio::urdf::buildModel(urdf_file, relative_fk_model_);
    relative_fk_measured_data_ = pinocchio::Data(relative_fk_model_);
    relative_fk_reference_data_ = pinocchio::Data(relative_fk_model_);

    relative_fk_q_indices_.reserve(joint_names.size());
    for (const auto& name : joint_names) {
      if (!relative_fk_model_.existJointName(name)) {
        throw std::invalid_argument("Relative FK joint missing from URDF: " + name);
      }
      const pinocchio::JointIndex jid = relative_fk_model_.getJointId(name);
      if (jid == 0 || jid >= relative_fk_model_.joints.size() ||
          relative_fk_model_.nqs[jid] != 1 ||
          relative_fk_model_.idx_qs[jid] < 0 ||
          relative_fk_model_.idx_qs[jid] >= relative_fk_model_.nq) {
        throw std::invalid_argument("Relative FK expects independent 1-DoF joint: " + name);
      }
      relative_fk_q_indices_.push_back(relative_fk_model_.idx_qs[jid]);
    }

    relative_fk_frame_ids_.reserve(primitives_.size());
    relative_fk_primitive_radii_.reserve(primitives_.size());
    for (const auto& primitive : primitives_) {
      if (!relative_fk_model_.existFrame(primitive.link_name)) {
        throw std::invalid_argument("Relative FK frame missing from URDF: " +
                                    primitive.link_name);
      }
      relative_fk_frame_ids_.push_back(
          relative_fk_model_.getFrameId(primitive.link_name));
      double radius = 0.0;
      if (primitive.kind == PrimitiveKind::Box) {
        radius = vector3Norm(primitive.half_size);
      } else if (primitive.kind == PrimitiveKind::Cylinder) {
        radius = std::hypot(primitive.radius, primitive.half_length);
      } else if (primitive.kind == PrimitiveKind::Sphere) {
        radius = primitive.radius;
      }
      if (!std::isfinite(radius) || radius < 0.0) {
        throw std::invalid_argument("Invalid relative FK primitive radius");
      }
      relative_fk_primitive_radii_.push_back(radius);
    }
    relative_fk_initialized_ = true;
    return true;
  } catch (const std::exception& ex) {
    if (error) {
      std::ostringstream oss;
      oss << "Failed to initialize relative FK: " << ex.what();
      *error = oss.str();
    }
    return false;
  }
}

std::vector<double> VbcPrimitiveModel::relativeFkDisplacementBounds(
    const Eigen::VectorXd& q_measured,
    const Eigen::VectorXd& q_reference) const {
  if (!relative_fk_initialized_ ||
      q_measured.size() != static_cast<int>(relative_fk_q_indices_.size()) ||
      q_reference.size() != q_measured.size() ||
      !finiteVector(q_measured) || !finiteVector(q_reference)) {
    throw std::invalid_argument("Invalid or uninitialized relative FK query");
  }

  Eigen::VectorXd q_measured_model = Eigen::VectorXd::Zero(relative_fk_model_.nq);
  Eigen::VectorXd q_reference_model = Eigen::VectorXd::Zero(relative_fk_model_.nq);
  for (std::size_t i = 0; i < relative_fk_q_indices_.size(); ++i) {
    q_measured_model[relative_fk_q_indices_[i]] = q_measured[static_cast<int>(i)];
    q_reference_model[relative_fk_q_indices_[i]] = q_reference[static_cast<int>(i)];
  }

  pinocchio::forwardKinematics(relative_fk_model_, relative_fk_measured_data_,
                               q_measured_model);
  pinocchio::updateFramePlacements(relative_fk_model_, relative_fk_measured_data_);
  pinocchio::forwardKinematics(relative_fk_model_, relative_fk_reference_data_,
                               q_reference_model);
  pinocchio::updateFramePlacements(relative_fk_model_, relative_fk_reference_data_);

  std::vector<double> result;
  result.reserve(relative_fk_frame_ids_.size());
  for (std::size_t i = 0; i < relative_fk_frame_ids_.size(); ++i) {
    const auto frame_id = relative_fk_frame_ids_[i];
    const auto& measured = relative_fk_measured_data_.oMf[frame_id];
    const auto& reference = relative_fk_reference_data_.oMf[frame_id];
    const auto& primitive = primitives_[i];
    const Eigen::Vector3d measured_center =
        measured.translation() + rotateVector(measured.rotation(), primitive.center);
    const Eigen::Vector3d reference_center =
        reference.translation() + rotateVector(reference.rotation(), primitive.center);
    const double relative_trace =
        relativeRotationTrace(measured.rotation(), reference.rotation());
    const double cos_theta = std::max(
        -1.0, std::min(1.0, 0.5 * (relative_trace - 1.0)));
    // For two rotations, ||R_m - R_r||_2 = 2 sin(theta/2) = sqrt(2 - 2 cos(theta)).
    const double rotation_scale = std::sqrt(std::max(0.0, 2.0 - 2.0 * cos_theta));
    const double bound = vector3Norm(measured_center - reference_center) +
                         rotation_scale * relative_fk_primitive_radii_[i];
    if (!std::isfinite(bound)) {
      throw std::runtime_error("Nonfinite relative FK displacement bound");
    }
    result.push_back(bound);
  }
  return result;
}

bool VbcPrimitiveModel::computeContinuousTrajectory(const TrajectoryRiskEvaluator& fk,
    const std::vector<Eigen::VectorXd>& q, const std::vector<int>& indices,
    const std::vector<double>& times, std::vector<VbcPrimitiveFrame>* out,
    std::vector<int>* out_indices, std::vector<double>* out_times,
    std::string* error, bool use_motion_bounds) const {
  out->clear(); out_indices->clear(); out_times->clear();
  try {
    if (q.empty() || q.size()!=times.size() || q.size()!=indices.size())
      throw std::invalid_argument("Incomplete continuous trajectory");
    // Numerical enclosure tightness, NOT a safety margin or time step. Every
    // subinterval is covered by its full analytic displacement bound.
    constexpr double max_enclosure_m=0.005;
    constexpr std::size_t max_frames=4096;
    for (std::size_t k=0; k<q.size(); ++k) {
      if (!finiteVector(q[k]) || q[k].size()!=fk.nq() || !std::isfinite(times[k]) ||
          times[k]<0 || indices[k]<0 || (k && times[k]<=times[k-1]))
        throw std::invalid_argument("Invalid continuous trajectory knot");
    }
    if (q.size()==1) {
      if (!computeTrajectory(fk,q,out,error)) return false;
      *out_indices=indices; *out_times=times; return true;
    }
    for (std::size_t k=0; k+1<q.size(); ++k) {
      const Eigen::VectorXd delta=q[k+1]-q[k];
      const auto full=displacementBounds(fk.activeJointNames(),delta);
      const double peak=*std::max_element(full.begin(),full.end());
      const double count=std::max(1.,std::ceil(peak/(2.*max_enclosure_m)));
      if (!std::isfinite(count) || count>max_frames || out->size()+count>max_frames)
        throw std::runtime_error("Continuous sweep capacity exceeded; no partial certificate");
      const int n=static_cast<int>(count);
      std::vector<Eigen::VectorXd> midpoints;
      for (int i=0;i<n;++i) midpoints.push_back(q[k]+((i+.5)/n)*delta);
      std::vector<VbcPrimitiveFrame> frames;
      if (!computeTrajectory(fk,midpoints,&frames,error)) return false;
      for (int i=0;i<n;++i) {
        frames[i].original_eval_timestep=static_cast<int>(k);
        // Keep adaptive midpoint sampling when motion-bound inflation is
        // disabled. This isolates the conservative enclosure from knot-only
        // sampling.
        for (double bound:full) {
          frames[i].motion_bounds.push_back(
              use_motion_bounds ? bound/(2.*n) : 0.0);
        }
        out->push_back(std::move(frames[i]));
        out_indices->push_back(indices[k]);
        // Earliest possible entry, not midpoint/end time: never grant extra
        // visibility lead time just because the representative pose is later.
        out_times->push_back(times[k]+(double(i)/n)*(times[k+1]-times[k]));
      }
    }
    return true;
  } catch (const std::exception& e) {
    out->clear(); out_indices->clear(); out_times->clear();
    if(error) *error=e.what();
    return false;
  }
}

bool VbcPrimitiveModel::computeTrajectory(const TrajectoryRiskEvaluator& fk,
    const std::vector<Eigen::VectorXd>& trajectory, std::vector<VbcPrimitiveFrame>* out,
    std::string* error) const {
  if (out) out->clear();
  if (!out || trajectory.empty() || primitives_.empty()) {
    if (error) *error = "Empty/uninitialized primitive trajectory";
    return false;
  }
  std::vector<FramePoseInBase> poses;
  out->reserve(trajectory.size());
  for (const auto& q : trajectory) {
    if (!fk.computeFramePosesForConfiguration(q, frames_, &poses, error)) {
      out->clear(); return false;
    }
    VbcPrimitiveFrame frame;
    frame.primitives.reserve(primitives_.size());
    for (const auto& local : primitives_) {
      const auto& pose = poses.at(local.frame_index);
      VbcPrimitive world = local;
      world.center = pose.translation_base + pose.rotation_base * local.center;
      world.rotation = pose.rotation_base * local.rotation;
      frame.primitives.push_back(std::move(world));
    }
    out->push_back(std::move(frame));
  }
  return true;
}
}  // namespace care_confidence_map
