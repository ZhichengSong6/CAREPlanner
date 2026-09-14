#pragma once
#include <care_confidence_map/vbc_primitive_model.hpp>
#include <care_confidence_map/primitive_probe_grid.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

namespace care_confidence_map {
inline std::vector<VbcPrimitive> transformPrimitiveGeometry(const VbcPrimitiveModel& model,
    tf2_ros::Buffer& buffer,const std::string& map_frame) {
  std::vector<tf2::Transform> poses;
  for(const auto& frame:model.frames()) {
    const auto msg=buffer.lookupTransform(map_frame,frame,ros::Time(0),ros::Duration(.005));
    tf2::Transform pose;tf2::fromMsg(msg.transform,pose);poses.push_back(pose);
  }
  std::vector<VbcPrimitive> world;world.reserve(model.primitives().size());
  for(const auto& local:model.primitives()) {
    const auto& pose=poses.at(local.frame_index);auto p=local;
    const auto c=pose*tf2::Vector3(local.center.x(),local.center.y(),local.center.z());
    p.center=Eigen::Vector3d(c.x(),c.y(),c.z());
    double* r=p.rotation.data();const double* l=local.rotation.data();
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)
      r[3*j+i]=pose.getBasis()[i][0]*l[3*j]+pose.getBasis()[i][1]*l[3*j+1]+pose.getBasis()[i][2]*l[3*j+2];
    GcdfPrimitiveAnchor a;a.shape=p;a.eval_timestep=a.original_timestep=0;validatePrimitiveAnchor(a);
    world.push_back(std::move(p));
  }
  if(world.empty())throw std::invalid_argument("Empty TF primitive geometry");
  return world;
}
}  // namespace care_confidence_map
