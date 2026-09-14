#pragma once
#include <care_confidence_map/gcdf_primitive_anchors.hpp>
#include <tuple>

namespace care_confidence_map {
struct PrimitiveProbe {
  Eigen::Vector3d point;
  std::string link_name, source_type;
  int collision_index = -1;
};

// Exact discrete grid-centre support, NOT spheres or a continuous certificate.
// Do not clip to the confidence map: outside-map body points remain UNKNOWN.
inline std::vector<PrimitiveProbe> primitiveProbeGrid(const std::vector<VbcPrimitive>& shapes,
    double resolution, const Eigen::Vector3d& origin) {
  if (shapes.empty() || !std::isfinite(resolution) || resolution<=0 || !origin.allFinite())
    throw std::invalid_argument("Invalid primitive diagnostic grid");
  using Key=std::array<int,4>;
  std::map<Key,PrimitiveProbe> points;
  std::map<std::string,int> links;
  for(const auto& p:shapes)links.emplace(p.link_name,0);
  int link_index=0;for(auto& item:links)item.second=link_index++;
  const double* grid_origin=origin.data();
  for(const auto& p:shapes) {
    GcdfPrimitiveAnchor a;a.shape=p;a.eval_timestep=a.original_timestep=0;
    validatePrimitiveAnchor(a);
    Eigen::Vector3i start,dims;Eigen::Vector3d grid_min;
    const double* r=p.rotation.data();const double* h=p.half_size.data();const double* center=p.center.data();
    double cells=1.;
    for(int axis=0;axis<3;++axis) {
      double extent=p.radius;
      if(p.kind==PrimitiveKind::Box)extent=std::abs(r[axis])*h[0]+std::abs(r[3+axis])*h[1]+std::abs(r[6+axis])*h[2];
      else if(p.kind==PrimitiveKind::Cylinder)extent=p.radius*std::hypot(r[axis],r[3+axis])+p.half_length*std::abs(r[6+axis]);
      extent+=1e-12;
      const double lo=std::floor((center[axis]-extent-grid_origin[axis])/resolution);
      const double hi=std::ceil((center[axis]+extent-grid_origin[axis])/resolution);
      cells*=hi-lo+1.;
      if(!std::isfinite(lo) || !std::isfinite(hi) || std::abs(lo)>1e7 || std::abs(hi)>1e7 || cells>1000000)
        throw std::invalid_argument("Primitive diagnostic grid capacity exceeded (no truncation)");
      start.data()[axis]=static_cast<int>(lo);dims.data()[axis]=static_cast<int>(hi-lo+1);
      grid_min.data()[axis]=grid_origin[axis]+resolution*lo;
    }
    const int sx=start.x(),sy=start.y(),sz=start.z(),link=links.at(p.link_name);
    visitPrimitiveProximity(p,0.,0.,grid_min,dims,resolution,[&](int x,int y,int z,float) {
      const int ix=sx+x,iy=sy+y,iz=sz+z;
      PrimitiveProbe probe;probe.point=Eigen::Vector3d(grid_origin[0]+resolution*ix,grid_origin[1]+resolution*iy,grid_origin[2]+resolution*iz);
      probe.link_name=p.link_name;probe.source_type=p.sourceType();probe.collision_index=p.collision_index;
      points.emplace(Key{{link,ix,iy,iz}},std::move(probe));
      if(points.size()>1000000) throw std::invalid_argument("Primitive diagnostic output capacity exceeded");
    });
  }
  if(points.empty()) throw std::invalid_argument("No primitive grid centres; cannot certify empty geometry");
  std::vector<PrimitiveProbe> out;out.reserve(points.size());
  for(auto& item:points)out.push_back(std::move(item.second));
  return out;
}
}  // namespace care_confidence_map
