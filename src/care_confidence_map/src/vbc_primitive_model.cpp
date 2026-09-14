#include <care_confidence_map/vbc_primitive_model.hpp>
#include <urdf/model.h>

namespace care_confidence_map {
bool VbcPrimitiveModel::load(const std::string& path,
                            const std::vector<std::string>& ignored, std::string* error) {
  frames_.clear(); primitives_.clear();
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
        const Eigen::Quaterniond quat(w, x, y, z);
        if (!p.center.allFinite() || !quat.coeffs().allFinite() || quat.norm() < 1e-12)
          throw std::runtime_error("Invalid collision origin: " + link.name);
        p.rotation = quat.normalized().toRotationMatrix();
        switch (c->geometry->type) {
          case urdf::Geometry::BOX: {
            const auto& b = static_cast<const urdf::Box&>(*c->geometry);
            p.kind = PrimitiveKind::Box;
            p.half_size = 0.5 * Eigen::Vector3d(b.dim.x, b.dim.y, b.dim.z);
            if (!p.half_size.allFinite() || (p.half_size.array() <= 0).any())
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
    frames_ = std::move(frames); primitives_ = std::move(parsed);
    return true;
  } catch (const std::exception& e) {
    if (error) *error = e.what();
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
