#pragma once
#include <care_confidence_map/trajectory_risk_evaluator.hpp>
#include <ros/ros.h>

namespace care_confidence_map {
inline bool initializeBodyGeometry(TrajectoryRiskEvaluator& evaluator,ros::NodeHandle& nh,
    const std::string& ns,const std::string& robot,const std::string& samples,const std::string& base,
    std::string* error) {
  std::string backend,primitive;
  nh.param<std::string>(ns+"/geometry_backend",backend,"samples");
  if(backend=="samples")return evaluator.initialize(robot,samples,base,error);
  if(backend!="primitive") {if(error)*error="Unknown geometry backend: "+backend;return false;}
  nh.param<std::string>(ns+"/primitive_urdf_file",primitive,"");
  double resolution;nh.param(ns+"/probe_resolution",resolution,.05);
  std::vector<double> origin;
  if(!nh.getParam(ns+"/probe_origin",origin))origin={-.95,-.95,0.};
  if(origin.size()!=3){if(error)*error="probe_origin must have 3 entries";return false;}
  return evaluator.initializePrimitives(robot,primitive,base,resolution,
      Eigen::Vector3d(origin[0],origin[1],origin[2]),error);
}
}  // namespace care_confidence_map
