#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <string>

namespace care_confidence_map {
enum class PrimitiveKind { Box, Cylinder, Sphere };

// Exact signed Euclidean distance to an oriented URDF primitive. Margin is
// applied as a Minkowski dilation, not as independent box/cylinder axis growth.
struct VbcPrimitive {
  PrimitiveKind kind = PrimitiveKind::Box;
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  Eigen::Vector3d half_size = Eigen::Vector3d::Zero();
  double radius = 0.0, half_length = 0.0;
  std::string link_name, collision_name;
  int collision_index = -1;
  std::size_t frame_index = 0;

  const char* sourceType() const {
    return kind == PrimitiveKind::Box ? "primitive_box" :
        kind == PrimitiveKind::Cylinder ? "primitive_cylinder" : "primitive_sphere";
  }
  double signedDistance(const Eigen::Vector3d& point) const {
    return signedDistanceXYZ(point.x(),point.y(),point.z());
  }
  double signedDistanceXYZ(double px,double py,double pz) const {
    // Scalar hot path also stays fast in the workspace's unoptimized builds.
    // Eigen matrices remain the pose representation; no approximate geometry.
    const double* c = center.data();
    const double dx = px - c[0], dy = py - c[1], dz = pz - c[2];
    const double* r = rotation.data();  // Eigen Matrix3d is column-major.
    const double x = r[0]*dx + r[1]*dy + r[2]*dz;
    const double y = r[3]*dx + r[4]*dy + r[5]*dz;
    const double z = r[6]*dx + r[7]*dy + r[8]*dz;
    if (kind == PrimitiveKind::Sphere) return std::sqrt(x*x + y*y + z*z) - radius;
    if (kind == PrimitiveKind::Cylinder) {
      const double a = std::sqrt(x*x + y*y) - radius, b = std::abs(z) - half_length;
      const double ap = std::max(a, 0.), bp = std::max(b, 0.);
      return std::sqrt(ap*ap + bp*bp) + std::min(std::max(a, b), 0.);
    }
    const double* h = half_size.data();
    const double a = std::abs(x) - h[0], b = std::abs(y) - h[1], z_gap = std::abs(z) - h[2];
    const double ap = std::max(a, 0.), bp = std::max(b, 0.), cp = std::max(z_gap, 0.);
    return std::sqrt(ap*ap + bp*bp + cp*cp) + std::min(std::max(a, std::max(b, z_gap)), 0.);
  }
  Eigen::Vector3d aabbHalfExtent(double margin) const {
    Eigen::Vector3d extent;
    if (kind == PrimitiveKind::Box) extent = rotation.cwiseAbs() * half_size;
    else if (kind == PrimitiveKind::Sphere) extent.setConstant(radius);
    else for (int i = 0; i < 3; ++i)
      extent[i] = radius * std::hypot(rotation(i, 0), rotation(i, 1)) +
                  half_length * std::abs(rotation(i, 2));
    return extent.array() + margin;
  }
};

}  // namespace care_confidence_map
