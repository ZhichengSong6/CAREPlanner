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
  ++geometry_revision_;
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

bool VbcPrimitiveModel::provePointOutsideContinuousTrajectory(
    const TrajectoryRiskEvaluator& fk, const std::vector<Eigen::VectorXd>& q,
    const Eigen::Vector3d& point, double margin, int* remaining_fk,
    const std::vector<Eigen::VectorXd>& curvature, double* proved_clearance) const {
  return provePointOutsideContinuousTrajectory(fk,q,point,margin,remaining_fk,
      curvature,proved_clearance,nullptr);
}

bool VbcPrimitiveModel::provePointOutsideContinuousTrajectory(
    const TrajectoryRiskEvaluator& fk, const std::vector<Eigen::VectorXd>& q,
    const Eigen::Vector3d& point, double margin, int* remaining_fk,
    const std::vector<Eigen::VectorXd>& curvature, double* proved_clearance,
    std::string* diagnostic) const {
  return provePointOutsideContinuousTrajectory(fk,q,point,margin,remaining_fk,
      curvature,proved_clearance,diagnostic,nullptr);
}

bool VbcPrimitiveModel::provePointOutsideContinuousTrajectory(
    const TrajectoryRiskEvaluator& fk, const std::vector<Eigen::VectorXd>& q,
    const Eigen::Vector3d& point, double margin, int* remaining_fk,
    const std::vector<Eigen::VectorXd>& curvature, double* proved_clearance,
    std::string* diagnostic, ProofFkCache* cache, double desired_clearance,
    const std::function<bool()>& may_refine) const {
  std::string last_failure;
  const auto reject=[diagnostic,&last_failure](const char* reason) {
    last_failure=reason;
    if(diagnostic)*diagnostic=reason;
    return false;
  };
  if(proved_clearance)*proved_clearance=0.;
  double certificate_slack=std::numeric_limits<double>::infinity();
  if (!remaining_fk)return reject("missing_fk_budget");
  if(cache && (cache->model!=this || cache->evaluator!=&fk || cache->geometry_revision!=geometry_revision_)) {
    *cache=ProofFkCache{};cache->model=this;cache->evaluator=&fk;cache->geometry_revision=geometry_revision_;
  }
  if (*remaining_fk<0)return reject("fk_budget_exhausted");
  if (q.empty() || q.size()>4096 ||
      primitives_.empty() || !point.allFinite() || !std::isfinite(margin) || margin<0 ||
      !std::isfinite(desired_clearance) || desired_clearance<0)
    return reject("invalid_proof_input");
  for (const auto& pose:q)
    if (pose.size()!=fk.nq() || !pose.allFinite()) return reject("invalid_configuration");
  if (!curvature.empty() && curvature.size()+1!=q.size()) return reject("invalid_curvature");
  for(const auto& c:curvature) if(c.size()!=fk.nq() || !c.allFinite()) return reject("invalid_curvature");
  struct Interval { Eigen::VectorXd a,b,c; int depth, root; };
  struct Leaf { Interval interval; double slack; };
  std::vector<Leaf> leaves;
  double sampled_upper=std::numeric_limits<double>::infinity();
  std::vector<Interval> pending;
  if (q.size()==1) pending.push_back({q.front(),q.front(),Eigen::VectorXd::Zero(fk.nq()),0,0});
  else for (std::size_t k=q.size()-1;k>0;--k)
    pending.push_back({q[k-1],q[k],curvature.empty()?Eigen::VectorXd::Zero(fk.nq()):curvature[k-1],0,int(k-1)});
  std::vector<std::vector<double>> root_keys(pending.size());
  std::vector<double> root_upper(pending.size(),std::numeric_limits<double>::infinity());
  for(auto it=pending.begin();it!=pending.end();) {
    auto& key=root_keys[it->root];key={point.x(),point.y(),point.z(),margin};
    for(const auto* pose:{&it->a,&it->b,&it->c})key.insert(key.end(),pose->data(),pose->data()+pose->size());
    const auto found=cache?cache->covers.find(key):std::map<std::vector<double>,ProofCover>::iterator{};
    if(cache && found!=cache->covers.end()) {
      for(const auto& leaf:found->second.leaves)
        leaves.push_back({{leaf.a,leaf.b,leaf.c,leaf.depth,it->root},leaf.slack});
      root_upper[it->root]=found->second.sampled_upper;
      sampled_upper=std::min(sampled_upper,found->second.sampled_upper);
      ++cache->cover_hits;it=pending.erase(it);
    } else ++it;
  }
  try {
    const auto evaluate=[&](const Interval& interval,double* slack) {
      const Eigen::VectorXd half_delta=(interval.b-interval.a)*.5;
      const Eigen::VectorXd midpoint=interval.a+half_delta+interval.c;
      const Eigen::VectorXd radius=half_delta.cwiseAbs()+interval.c.cwiseAbs();
      const auto bounds=displacementBounds(fk.activeJointNames(),radius);
      std::vector<VbcPrimitiveFrame> frames;
      std::string error;
      const std::vector<double> key(midpoint.data(),midpoint.data()+midpoint.size());
      const auto cached=cache ? cache->frames.find(key) : std::map<std::vector<double>,VbcPrimitiveFrame>::iterator{};
      if(cache && cached!=cache->frames.end()) {
        frames.push_back(cached->second);++cache->hits;
      } else {
        if(*remaining_fk<=0)return reject("fk_budget_exhausted");
        --*remaining_fk;
        if (!computeTrajectory(fk,{midpoint},&frames,&error) || frames.size()!=1)return reject("proof_fk_failed");
        if(cache) {
          ++cache->misses;
          if(cache->frames.size()<256)cache->frames.emplace(key,frames.front());
        }
      }
      if(bounds.size()!=frames.front().primitives.size())return reject("proof_fk_failed");
      double interval_slack=std::numeric_limits<double>::infinity();
      for (std::size_t pi=0;pi<bounds.size();++pi) {
        const double distance=frames.front().primitives[pi].signedDistance(point);
        if (!std::isfinite(distance) || !std::isfinite(bounds[pi]) || bounds[pi]<0)
          return reject("nonfinite_proof_geometry");
        if(distance<=margin+1e-12)return reject("distance_within_margin");
        // Signed distance is 1-Lipschitz in rigid solid displacement. This
        // same analytic radius covers the FULL half-interval on either side.
        interval_slack=std::min(interval_slack,distance-margin-bounds[pi]);
        sampled_upper=std::min(sampled_upper,distance-margin);
        root_upper[interval.root]=std::min(root_upper[interval.root],distance-margin);
      }
      *slack=interval_slack;
      return true;
    };
    const auto split=[](const Interval& interval) {
      const Eigen::VectorXd midpoint=interval.a+(interval.b-interval.a)*.5+interval.c;
      return std::make_pair(Interval{interval.a,midpoint,interval.c*.25,interval.depth+1,interval.root},
                            Interval{midpoint,interval.b,interval.c*.25,interval.depth+1,interval.root});
    };
    while (!pending.empty()) {
      auto interval=std::move(pending.back()); pending.pop_back();
      double interval_slack=0.;
      if(!evaluate(interval,&interval_slack))return false;
      if (interval_slack>1e-12) {
        certificate_slack=std::min(certificate_slack,interval_slack);
        leaves.push_back({interval,interval_slack});
        continue;
      }
      const Eigen::VectorXd midpoint=interval.a+(interval.b-interval.a)*.5+interval.c;
      if (interval.depth>=16 || (midpoint.array()==interval.a.array()).all() ||
          (midpoint.array()==interval.b.array()).all()) return reject("proof_subdivision_exhausted");
      pending.push_back({midpoint,interval.b,interval.c*.25,interval.depth+1,interval.root});
      pending.push_back({interval.a,midpoint,interval.c*.25,interval.depth+1,interval.root});
    }
    // All leaves already prove the original floor over the entire motion.
    // Improve the weakest enclosure first. Replace a parent only after BOTH
    // children have valid bounds; budget exhaustion never loses coverage.
    // An exact sampled distance is an upper bound on achievable clearance,
    // never permission to accept a motion. Stop within 5% of the optional
    // quality target when geometry makes that target unattainable.
    while(desired_clearance>0. && !leaves.empty()) {
      const auto weakest=std::min_element(leaves.begin(),leaves.end(),
          [](const Leaf& a,const Leaf& b){return a.slack<b.slack;});
      certificate_slack=weakest->slack;
      if(certificate_slack>=desired_clearance ||
          sampled_upper-certificate_slack<=.05*desired_clearance || *remaining_fk<=0 ||
          (may_refine && !may_refine()))break;
      const auto& interval=weakest->interval;
      const Eigen::VectorXd midpoint=interval.a+(interval.b-interval.a)*.5+interval.c;
      if(interval.depth>=16 || (midpoint.array()==interval.a.array()).all() ||
          (midpoint.array()==interval.b.array()).all())break;
      const auto children=split(interval);double left=0.,right=0.;
      if(!evaluate(children.first,&left)) {
        if(last_failure!="fk_budget_exhausted")return false;
        break;
      }
      if(may_refine && !may_refine())break;
      if(!evaluate(children.second,&right)) {
        if(last_failure!="fk_budget_exhausted")return false;
        break;
      }
      // The parent's enclosure also covers each child, so subdivision cannot
      // weaken a previously established bound through numerical looseness.
      left=std::max(left,weakest->slack);right=std::max(right,weakest->slack);
      *weakest={children.first,left};leaves.push_back({children.second,right});
    }
    certificate_slack=std::numeric_limits<double>::infinity();
    std::vector<ProofCover> covers(root_keys.size());
    for(const auto& leaf:leaves) {
      if(leaf.slack<certificate_slack) {
        certificate_slack=leaf.slack;
        if(cache)cache->last_weak_interval=leaf.interval.root;
      }
      if(cache)covers[leaf.interval.root].leaves.push_back({leaf.interval.a,leaf.interval.b,
          leaf.interval.c,leaf.interval.depth,leaf.slack});
    }
    if(cache)for(std::size_t i=0;i<covers.size();++i) {
      covers[i].sampled_upper=root_upper[i];
      auto old=cache->covers.find(root_keys[i]);
      const auto old_size=old==cache->covers.end()?0:old->second.leaves.size();
      // Bounded scratch. Dropping a cache entry never refunds FK or grants a proof.
      if(cache->covers.size()>=1024 && old==cache->covers.end())continue;
      if(cache->cover_leaves-old_size+covers[i].leaves.size()>4096)continue;
      cache->cover_leaves=cache->cover_leaves-old_size+covers[i].leaves.size();
      cache->covers[root_keys[i]]=std::move(covers[i]);
    }
    if(proved_clearance)*proved_clearance=certificate_slack;
    if(diagnostic)*diagnostic="proved";
    return true;
  } catch (const std::exception&) {
    return reject("proof_exception");
  }
}

bool VbcPrimitiveModel::computeContinuousTrajectory(const TrajectoryRiskEvaluator& fk,
    const std::vector<Eigen::VectorXd>& q, const std::vector<int>& indices,
    const std::vector<double>& times, std::vector<VbcPrimitiveFrame>* out,
    std::vector<int>* out_indices, std::vector<double>* out_times,
    std::string* error, bool use_motion_bounds,
    const std::vector<Eigen::VectorXd>& curvature) const {
  out->clear(); out_indices->clear(); out_times->clear();
  try {
    if (q.empty() || q.size()!=times.size() || q.size()!=indices.size())
      throw std::invalid_argument("Incomplete continuous trajectory");
    if(!curvature.empty() && curvature.size()+1!=q.size())
      throw std::invalid_argument("Incomplete trajectory curvature");
    for(const auto& c:curvature) if(c.size()!=fk.nq() || !c.allFinite())
      throw std::invalid_argument("Invalid trajectory curvature");
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
      const Eigen::VectorXd c=curvature.empty()?Eigen::VectorXd::Zero(fk.nq()):curvature[k];
      const Eigen::VectorXd derivative_bound=delta.cwiseAbs()+4.*c.cwiseAbs();
      const auto full=displacementBounds(fk.activeJointNames(),derivative_bound);
      const double peak=*std::max_element(full.begin(),full.end());
      const double count=std::max(1.,std::ceil(peak/(2.*max_enclosure_m)));
      if (!std::isfinite(count) || count>max_frames || out->size()+count>max_frames)
        throw std::runtime_error("Continuous sweep capacity exceeded; no partial certificate");
      const int n=static_cast<int>(count);
      std::vector<Eigen::VectorXd> midpoints;
      for (int i=0;i<n;++i) {
        const double s=(i+.5)/n;
        midpoints.push_back(q[k]+s*delta+4.*s*(1.-s)*c);
      }
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
