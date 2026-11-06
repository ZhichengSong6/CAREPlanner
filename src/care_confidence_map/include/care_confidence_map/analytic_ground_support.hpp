#pragma once
// Plane evidence and residual volume are independent. A compatible endpoint
// never establishes that the rest of its voxel is empty or ground-only.
#include <Eigen/Core>
#include <boost/property_tree/json_parser.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace care_confidence_map {
// A pinhole frustum is the intersection of six half spaces. Checking each
// support maximum covers the complete AABB, including its faces and edges.
inline bool supportAabbInsideFrustum(const Eigen::Vector3d& center,const Eigen::Vector3d& half,
    const Eigen::Matrix3d& rotation,const Eigen::Vector3d& translation,
    const Eigen::Vector3d& forward,const Eigen::Vector3d& right,const Eigen::Vector3d& up,
    double near,double far,double tan_h,double tan_v) {
  if(!center.allFinite() || !half.allFinite() || (half.array()<0.).any() || !rotation.allFinite() || !translation.allFinite() ||
      !std::isfinite(near) || !std::isfinite(far) || near<0. || far<=near ||
      !std::isfinite(tan_h) || !std::isfinite(tan_v) || tan_h<=0. || tan_v<=0.)return false;
  const Eigen::Vector3d delta=center-translation;
  const auto maximum=[&](const Eigen::Vector3d& direction) {
    const Eigen::Vector3d world=rotation*direction;return world.dot(delta)+world.cwiseAbs().dot(half);
  };
  return maximum(-forward)<=-near && maximum(forward)<=far &&
      maximum(right-tan_h*forward)<=0. && maximum(-right-tan_h*forward)<=0. &&
      maximum(up-tan_v*forward)<=0. && maximum(-up-tan_v*forward)<=0.;
}
struct AnalyticGroundSupport {
  Eigen::Vector3d normal=Eigen::Vector3d::UnitZ();
  double offset=0.,uncertainty=0.,association_band=.002;
  double registration_bound=std::numeric_limits<double>::quiet_NaN();
  std::string frame,calibration_id,support_id;
  bool association_enabled=false;
  void validate() const {
    if(frame!="base_link" || support_id!="known_ground" || calibration_id.empty() ||
        !normal.allFinite() || std::abs(normal.squaredNorm()-1.)>1e-10 || !std::isfinite(offset) ||
        !std::isfinite(uncertainty) || uncertainty<0. || !std::isfinite(association_band) || association_band<0. ||
        (association_enabled && (!std::isfinite(registration_bound) || registration_bound<0.)))
      throw std::invalid_argument("Invalid analytic ground support contract");
  }
  static AnalyticGroundSupport load(const std::string& filename) {
    boost::property_tree::ptree tree;boost::property_tree::read_json(filename,tree);
    const int schema=tree.get<int>("schema");
    if(schema!=3 && schema!=4)throw std::invalid_argument("Plane support requires schema3 or research schema4");
    if(schema==4 && tree.get<std::string>("research_profile","")!="nominal_plane_point_v1")throw std::invalid_argument("Unknown research plane profile");
    AnalyticGroundSupport result;const auto& plane=tree.get_child("plane");int i=0;
    for(const auto& x:plane.get_child("normal")) {
      if(i>=3)throw std::invalid_argument("Invalid support normal count");result.normal[i++]=x.second.get_value<double>();
    }
    if(i!=3)throw std::invalid_argument("Invalid support normal count");
    result.offset=plane.get<double>("offset_m");result.uncertainty=tree.get<double>("plane_uncertainty_m");
    result.frame=plane.get<std::string>("frame");result.support_id=plane.get<std::string>("support_id");
    result.calibration_id=plane.get<std::string>("calibration_id");
    result.association_enabled=plane.get<bool>("association_enabled");
    result.association_band=plane.get<double>("association_band_m",.002);
    const auto bound=plane.get_optional<double>("registration_bound_m");if(bound)result.registration_bound=*bound;
    result.validate();return result;
  }
  double height(const Eigen::Vector3d& point) const {return normal.dot(point)+offset;}
  // Only a surface-support envelope, never a whole-cell FREE or GROUND label.
  bool compatibleEndpoint(const Eigen::Vector3d& point,bool exact_source_valid) const {
    return association_enabled && exact_source_valid && point.allFinite() &&
        std::isfinite(registration_bound) && std::abs(height(point))<=association_band;
  }
  double supportUpper() const {
    if(!association_enabled)throw std::logic_error("Ground association is not enabled");
    return uncertainty+association_band+registration_bound;
  }
  struct ResidualVolume {
    std::vector<Eigen::Vector3d> vertices;
    Eigen::Vector3d minimum=Eigen::Vector3d::Zero(),maximum=Eigen::Vector3d::Zero();
    bool empty=true;
  };
  // Exact vertices of cube intersect {n.p+d >= upper}; its AABB is a
  // conservative cover. Empty residual means the plane envelope covers it,
  // not that a sensor observed free space. All other classes remain separate.
  ResidualVolume residual(const Eigen::Vector3d& center,double resolution,double upper) const {
    if(!center.allFinite() || !std::isfinite(resolution) || resolution<=0. || !std::isfinite(upper))
      throw std::invalid_argument("Invalid residual support volume");
    ResidualVolume result;Eigen::Vector3d corners[8];double h[8];
    for(int i=0;i<8;++i) {
      for(int j=0;j<3;++j)corners[i][j]=center[j]+((i&(1<<j))?.5:-.5)*resolution;
      h[i]=height(corners[i])-upper;if(h[i]>=0.)result.vertices.push_back(corners[i]);
    }
    for(int i=0;i<8;++i)for(int axis=0;axis<3;++axis) {
      const int j=i^(1<<axis);if(j<=i || (h[i]<0.)==(h[j]<0.))continue;
      const double t=h[i]/(h[i]-h[j]);result.vertices.push_back(corners[i]+t*(corners[j]-corners[i]));
    }
    if(result.vertices.empty())return result;
    result.empty=false;result.minimum=result.maximum=result.vertices.front();
    for(const auto& point:result.vertices){result.minimum=result.minimum.cwiseMin(point);result.maximum=result.maximum.cwiseMax(point);}
    return result;
  }
};
}
