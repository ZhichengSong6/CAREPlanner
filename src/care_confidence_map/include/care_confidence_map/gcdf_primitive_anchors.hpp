#pragma once
#include <care_confidence_map/primitive_geometry.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace care_confidence_map {
// Versioned wire geometry: float64 poses/shapes, original float32 GPU q,
// explicit inflation, no sphere radius field which a legacy reader can accept.
struct GcdfPrimitiveAnchor {
  VbcPrimitive shape;
  std::array<float, 7> q{};
  int eval_timestep = -1, original_timestep = -1;
  double inflation = 0.;
};

inline sensor_msgs::PointCloud2 primitiveAnchorSchema() {
  sensor_msgs::PointCloud2 c;
  c.height = 1; c.is_dense = true;
  auto field = [&](const std::string& name, int type, int bytes) {
    sensor_msgs::PointField f; f.name=name; f.offset=c.point_step;
    f.datatype=type; f.count=1; c.fields.push_back(f); c.point_step+=bytes;
  };
  for (const auto& name : {"primitive_schema", "primitive_kind", "eval_timestep", "original_timestep", "shape_count", "shape_index", "knot_count"})
    field(name, sensor_msgs::PointField::INT32, 4);
  for (int j=0;j<7;++j) field("q"+std::to_string(j),sensor_msgs::PointField::FLOAT32,4);
  for (int j=0;j<18;++j) field("shape"+std::to_string(j),sensor_msgs::PointField::FLOAT64,8);
  return c;
}

inline void validatePrimitiveAnchor(const GcdfPrimitiveAnchor& a) {
  const auto& p=a.shape;
  const double* r=p.rotation.data();
  bool rotation_ok=std::all_of(r,r+9,[](double x){return std::isfinite(x);});
  for(int i=0;i<3;++i) for(int j=0;j<3;++j)
    rotation_ok=rotation_ok && std::abs(r[3*i]*r[3*j]+r[3*i+1]*r[3*j+1]+r[3*i+2]*r[3*j+2]-(i==j?1.:0.))<=1e-9;
  const double det=r[0]*(r[4]*r[8]-r[7]*r[5])-r[3]*(r[1]*r[8]-r[7]*r[2])+r[6]*(r[1]*r[5]-r[4]*r[2]);
  if (a.original_timestep<0 || a.eval_timestep<0 ||
      !std::isfinite(a.inflation) || a.inflation<0 || !p.center.allFinite() ||
      !rotation_ok || !p.half_size.allFinite() ||
      !std::isfinite(p.radius) || !std::isfinite(p.half_length) ||
      !std::all_of(a.q.begin(),a.q.end(),[](float q){return std::isfinite(q);}) ||
      std::abs(det-1.)>1e-9)
    throw std::invalid_argument("Invalid primitive anchor pose/q/inflation/index");
  if ((p.kind==PrimitiveKind::Box && (p.half_size.array()<=0).any()) ||
      (p.kind==PrimitiveKind::Cylinder && (p.radius<=0 || p.half_length<=0)) ||
      (p.kind==PrimitiveKind::Sphere && p.radius<=0) ||
      (p.kind!=PrimitiveKind::Box && p.kind!=PrimitiveKind::Cylinder && p.kind!=PrimitiveKind::Sphere))
    throw std::invalid_argument("Invalid primitive anchor shape");
}

inline sensor_msgs::PointCloud2 encodePrimitiveAnchors(const std_msgs::Header& header,
    const std::vector<GcdfPrimitiveAnchor>& anchors) {
  if (anchors.empty() || anchors.size()>50000) throw std::invalid_argument("Invalid primitive anchor count");
  auto c=primitiveAnchorSchema(); c.header=header; c.width=anchors.size();
  c.row_step=c.width*c.point_step; c.data.resize(c.row_step);
  std::map<int,int> counts,seen;
  for (const auto& a:anchors) ++counts[a.original_timestep];
  for (std::size_t i=0;i<anchors.size();++i) {
    const auto& a=anchors[i]; validatePrimitiveAnchor(a);
    const auto& p=a.shape;
    int32_t ints[]={1,static_cast<int32_t>(p.kind),a.eval_timestep,a.original_timestep,
                    counts[a.original_timestep],seen[a.original_timestep]++,static_cast<int32_t>(counts.size())};
    double shape[18];
    for (int j=0;j<3;++j) {shape[j]=p.center[j];shape[12+j]=p.half_size[j];}
    for (int j=0;j<9;++j) shape[3+j]=p.rotation.data()[j];
    shape[15]=p.radius;shape[16]=p.half_length;shape[17]=a.inflation;
    auto* dst=c.data.data()+i*c.point_step;
    std::memcpy(dst,ints,28);std::memcpy(dst+28,a.q.data(),28);std::memcpy(dst+56,shape,144);
  }
  return c;
}

inline std::vector<GcdfPrimitiveAnchor> decodePrimitiveAnchors(const sensor_msgs::PointCloud2& c) {
  auto schema=primitiveAnchorSchema();
  if (c.is_bigendian || c.height!=1 || c.width==0 || c.width>50000 ||
      c.point_step!=schema.point_step || c.row_step!=c.width*c.point_step ||
      c.data.size()!=c.row_step || c.fields.size()!=schema.fields.size())
    throw std::invalid_argument("Invalid primitive cloud dimensions/schema");
  for (std::size_t i=0;i<c.fields.size();++i) {
    const auto& a=c.fields[i];const auto& b=schema.fields[i];
    if (a.name!=b.name || a.offset!=b.offset || a.datatype!=b.datatype || a.count!=1)
      throw std::invalid_argument("Invalid primitive cloud field");
  }
  std::vector<GcdfPrimitiveAnchor> out;out.reserve(c.width);
  std::map<int,std::pair<int,std::set<int>>> coverage;
  int knot_count=0;std::map<int,int> eval_to_original;
  for (std::size_t i=0;i<c.width;++i) {
    const auto* src=c.data.data()+i*c.point_step;
    int32_t ints[7];double shape[18];GcdfPrimitiveAnchor a;
    std::memcpy(ints,src,28);std::memcpy(a.q.data(),src+28,28);std::memcpy(shape,src+56,144);
    if (ints[0]!=1) throw std::invalid_argument("Unknown primitive cloud version");
    if (ints[6]<1 || ints[6]>50 || (knot_count && knot_count!=ints[6]) || ints[2]<0 || ints[2]>=ints[6])
      throw std::invalid_argument("Invalid primitive knot coverage");
    knot_count=ints[6];
    auto eval=eval_to_original.emplace(ints[2],ints[3]);
    if (!eval.second && eval.first->second!=ints[3])
      throw std::invalid_argument("Inconsistent primitive timestep identity");
    if (ints[4]<1 || ints[4]>50000 || ints[5]<0 || ints[5]>=ints[4])
      throw std::invalid_argument("Invalid primitive shape count/index");
    auto& seen=coverage[ints[3]];
    if ((seen.first && seen.first!=ints[4]) || !seen.second.insert(ints[5]).second)
      throw std::invalid_argument("Inconsistent/duplicate primitive coverage");
    seen.first=ints[4];
    auto& p=a.shape;p.kind=static_cast<PrimitiveKind>(ints[1]);a.eval_timestep=ints[2];a.original_timestep=ints[3];
    for (int j=0;j<3;++j) {p.center[j]=shape[j];p.half_size[j]=shape[12+j];}
    for (int j=0;j<9;++j) p.rotation.data()[j]=shape[3+j];
    p.radius=shape[15];p.half_length=shape[16];a.inflation=shape[17];
    validatePrimitiveAnchor(a);out.push_back(std::move(a));
  }
  for (const auto& step:coverage)
    if (step.second.second.size()!=static_cast<std::size_t>(step.second.first))
      throw std::invalid_argument("Incomplete primitive coverage");
  if (coverage.size()!=static_cast<std::size_t>(knot_count) || eval_to_original.size()!=coverage.size())
    throw std::invalid_argument("Incomplete primitive trajectory knot coverage");
  return out;
}

// Grid-centre proximity, not full voxel cubes or a continuous-motion proof.
// Rounded-box/cylinder Minkowski dilation uses exact SDF, never independent
// axis inflation. Clip before integer conversion, including outside centres.
struct AllPrimitiveCells { bool operator()(int,int,int) const {return true;} };
template<class Visit,class Eligible=AllPrimitiveCells>
void visitPrimitiveProximity(const VbcPrimitive& p, double inflation, double margin,
    const Eigen::Vector3d& grid_min, const Eigen::Vector3i& dims, double resolution, Visit visit,
    Eligible eligible=Eligible()) {
  const double band=inflation+margin;
  if (!std::isfinite(band) || inflation<0 || margin<0 || resolution<=0 ||
      !std::isfinite(resolution) || !grid_min.allFinite() || (dims.array()<=0).any())
    throw std::invalid_argument("Invalid primitive proximity grid/margin");
  const double* rotation=p.rotation.data();const double* half=p.half_size.data();
  const double* center=p.center.data();const double* grid=grid_min.data();const int* size=dims.data();
  int begin[3],end[3];
  for(int axis=0;axis<3;++axis) {
    double extent=p.radius;
    if(p.kind==PrimitiveKind::Box)
      extent=std::abs(rotation[axis])*half[0]+std::abs(rotation[3+axis])*half[1]+std::abs(rotation[6+axis])*half[2];
    else if(p.kind==PrimitiveKind::Cylinder)
      extent=p.radius*std::hypot(rotation[axis],rotation[3+axis])+p.half_length*std::abs(rotation[6+axis]);
    extent+=band+1e-12;
    const double lo=(center[axis]-extent-grid[axis])/resolution;
    const double hi=(center[axis]+extent-grid[axis])/resolution;
    if(!std::isfinite(lo) || !std::isfinite(hi)) throw std::invalid_argument("Nonfinite primitive AABB");
    if(hi<0 || lo>size[axis]-1) return;
    begin[axis]=static_cast<int>(std::max(0.,std::ceil(lo)));
    end[axis]=static_cast<int>(std::min(double(size[axis]-1),std::floor(hi)));
  }
  const int bx=begin[0],by=begin[1],bz=begin[2],ex=end[0],ey=end[1],ez=end[2];
  const double gx=grid[0],gy=grid[1],gz=grid[2];
  for (int x=bx;x<=ex;++x) {
    const double px=gx+resolution*x;
    for (int y=by;y<=ey;++y) {
      const double py=gy+resolution*y;
      for (int z=bz;z<=ez;++z) {
        if (!eligible(x,y,z)) continue;
        const double sdf=p.signedDistanceXYZ(px,py,gz+resolution*z);
        if (sdf<=band+1e-12) visit(x,y,z,static_cast<float>(sdf-inflation));
      }
    }
  }
}
}  // namespace care_confidence_map
