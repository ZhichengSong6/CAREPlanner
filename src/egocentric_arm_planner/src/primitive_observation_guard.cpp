#include "egocentric_arm_planner/primitive_observation_guard.hpp"
#include <care_confidence_map/vbc_primitive_model.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <sstream>
#include <set>

namespace egocentric_arm_planner {
bool parseHandoffResidualFailure(const std::string& evidence,
    std::uint64_t raw_stamp,std::uint64_t audited_stamp,std::uint64_t mode_epoch,
    const std::string& observation_token,double ordinary_floor,HandoffResidualFailure* failure) {
  if(!failure || !raw_stamp || !audited_stamp || !mode_epoch || observation_token.empty() ||
      evidence.size()>2048 || !std::isfinite(ordinary_floor))return false;
  try {
    boost::property_tree::ptree t;std::istringstream input(evidence);boost::property_tree::read_json(input,t);
    for(const auto* k:{"schema","kind","trajectory_stamp_ns","raw_candidate_stamp_ns","mode_epoch",
        "observation_token","stage","floor_m","point","ordinary_unknown","interval_index",
        "required_residual_m","certified_residual_m","requires_reconstruction"})if(t.count(k)!=1)return false;
    const auto identity=[&](const char* k,std::uint64_t expected) {
      const auto s=t.get<std::string>(k);return !s.empty() && s.find_first_not_of("0123456789")==std::string::npos && std::stoull(s)==expected;
    };
    const auto ordinary=t.get<std::string>("ordinary_unknown"),rebuild=t.get<std::string>("requires_reconstruction");
    const auto stage=t.get<std::string>("stage");double floor=t.get<double>("floor_m");
    HandoffResidualFailure f;f.required_residual_m=t.get<double>("required_residual_m");
    f.certified_residual_m=t.get<double>("certified_residual_m");
    if(t.get<int>("schema")!=1 || t.get<std::string>("kind")!="handoff_residual_insufficient" ||
        !identity("trajectory_stamp_ns",audited_stamp) || !identity("raw_candidate_stamp_ns",raw_stamp) ||
        !identity("mode_epoch",mode_epoch) || t.get<std::string>("observation_token")!=observation_token ||
        (ordinary!="true" && ordinary!="false") || (rebuild!="true" && rebuild!="false") ||
        !std::isfinite(floor) || floor<0. || floor>1. ||
        !std::isfinite(f.required_residual_m) || !std::isfinite(f.certified_residual_m) ||
        f.certified_residual_m<=0. || f.required_residual_m<f.certified_residual_m-1e-12 ||
        (stage!="map_path_proof" && stage!="map_brake_proof" && stage!="support_path_proof" &&
         stage!="support_onset_proof" && stage!="support_brake_proof" && stage!="onset_budget") ||
        t.get<int>("interval_index") < -1 || t.get<int>("interval_index")>=4096 ||
        (ordinary=="true" && (stage!="map_path_proof" || floor!=ordinary_floor)) ||
        t.get_child("point").size()!=3)return false;
    int j=0;for(const auto& x:t.get_child("point")){if(!x.first.empty())return false;f.point.point[j++]=x.second.get_value<double>();}
    if(!f.point.point.allFinite() || f.point.point.cwiseAbs().maxCoeff()>1e6)return false;
    f.point.audited_stamp=audited_stamp;f.point.ordinary_path_guard=ordinary=="true";
    if(t.count("initial_state_changed")) {
      const auto changed=t.get<std::string>("initial_state_changed");
      if(t.count("initial_state_changed")!=1 || (changed!="true" && changed!="false"))return false;
      f.initial_state_changed=changed=="true";
    }
    f.requires_reconstruction=rebuild=="true";*failure=f;return true;
  } catch(...){return false;}
}
bool parseProofResourceFailure(const std::string& evidence,
    std::uint64_t audited_stamp, std::uint64_t mode_epoch,
    const std::string& observation_token, double ordinary_floor,
    PrimitiveObservationPoint* point) {
  if(!point || !audited_stamp || !mode_epoch || observation_token.empty() ||
      evidence.size()>2048 || !std::isfinite(ordinary_floor))return false;
  try {
    boost::property_tree::ptree t;std::istringstream input(evidence);
    boost::property_tree::read_json(input,t);
    for(const auto* key:{"schema","kind","trajectory_stamp_ns","mode_epoch",
        "observation_token","stage","floor_m","point","fk_before","fk_spent",
        "fk_remaining","ordinary_unknown"})if(t.count(key)!=1)return false;
    const auto stamp=t.get<std::string>("trajectory_stamp_ns");
    const auto epoch=t.get<std::string>("mode_epoch");
    const auto stage=t.get<std::string>("stage");
    const auto ordinary=t.get<std::string>("ordinary_unknown");
    const double floor=t.get<double>("floor_m");
    if(t.get<int>("schema")!=1 || t.get<std::string>("kind")!="mandatory_proof_resource_exhausted" ||
        stamp.empty() || stamp.find_first_not_of("0123456789")!=std::string::npos ||
        epoch.empty() || epoch.find_first_not_of("0123456789")!=std::string::npos ||
        std::stoull(stamp)!=audited_stamp || std::stoull(epoch)!=mode_epoch ||
        t.get<std::string>("observation_token")!=observation_token ||
        (stage!="map_path_proof" && stage!="map_brake_proof" && stage!="support_path_proof" &&
         stage!="support_onset_proof" && stage!="support_brake_proof") ||
        t.get<int>("fk_before")!=256 || t.get<int>("fk_spent")!=256 ||
        t.get<int>("fk_remaining")!=0 || !std::isfinite(floor) || floor<0. ||
        (ordinary!="true" && ordinary!="false") ||
        (ordinary=="true" && (stage!="map_path_proof" || floor!=ordinary_floor)) ||
        t.get_child("point").size()!=3)return false;
    PrimitiveObservationPoint p;int j=0;
    for(const auto& x:t.get_child("point")){if(!x.first.empty())return false;p.point[j++]=x.second.get_value<double>();}
    if(!p.point.allFinite() || p.point.cwiseAbs().maxCoeff()>1e6)return false;
    p.audited_stamp=audited_stamp;p.ordinary_path_guard=ordinary=="true";*point=p;return true;
  } catch(...){return false;}
}
bool parseOrdinaryPathCounterexample(const std::string& evidence,
    std::uint64_t audited_stamp, double ordinary_floor, PrimitiveObservationPoint* point) {
  if(!point || !audited_stamp || evidence.size()>2048 || !std::isfinite(ordinary_floor))return false;
  try {
    boost::property_tree::ptree t;std::istringstream input(evidence);
    boost::property_tree::read_json(input,t);
    for(const auto* key:{"schema","kind","trajectory_stamp_ns","floor_m","point"})
      if(t.count(key)!=1)return false;
    const auto stamp=t.get<std::string>("trajectory_stamp_ns");
    if(t.get<int>("schema")!=1 || t.get<std::string>("kind")!="unknown_ordinary_path_floor" ||
        stamp.empty() || stamp.find_first_not_of("0123456789")!=std::string::npos ||
        std::stoull(stamp)!=audited_stamp || t.get<double>("floor_m")!=ordinary_floor ||
        t.get_child("point").size()!=3)return false;
    PrimitiveObservationPoint p;int j=0;
    for(const auto& x:t.get_child("point")){if(!x.first.empty())return false;p.point[j++]=x.second.get_value<double>();}
    if(!p.point.allFinite() || p.point.cwiseAbs().maxCoeff()>1e6)return false;
    p.audited_stamp=audited_stamp;p.ordinary_path_guard=true;*point=p;return true;
  } catch(...){return false;}
}
bool parsePrimitiveObservationEvidence(const std::string& evidence,
    std::uint64_t audited_stamp, std::vector<PrimitiveObservationPoint>* points) {
  if (!points || !audited_stamp || evidence.size() > 256*1024) return false;
  points->clear();
  try {
    boost::property_tree::ptree tree;
    std::istringstream input(evidence);
    boost::property_tree::read_json(input, tree);
    if (tree.count("schema") != 1 || tree.get<int>("schema") != 1 ||
        tree.count("trajectory_stamp_ns") != 1 ||
        tree.count("violations") != 1 ||
        tree.get<std::string>("geometry_backend") != "primitive" ||
        tree.get<std::string>("spatial_semantics") != "voxel_centers_piecewise_linear_enclosure")
      return false;
    const auto stamp = tree.get<std::string>("trajectory_stamp_ns");
    if (stamp.empty() || stamp.find_first_not_of("0123456789") != std::string::npos ||
        std::stoull(stamp) != audited_stamp) return false;
    std::vector<PrimitiveObservationPoint> parsed;
    for (const auto& entry : tree.get_child("violations")) {
      const auto& t = entry.second;
      for (const auto& key : {"point", "sweeping_link", "source_collision_name",
                              "source_collision_index", "sweep_s"})
        if (t.count(key) != 1) return false;
      if (!entry.first.empty() || t.count("point") != 1 ||
          t.get_child("point").size() != 3) return false;
      PrimitiveObservationPoint p;
      int j = 0;
      for (const auto& v : t.get_child("point")) {
        if (!v.first.empty()) return false;
        p.point[j++] = v.second.get_value<double>();
      }
      p.sweeping_link = t.get<std::string>("sweeping_link");
      p.collision_name = t.get<std::string>("source_collision_name");
      p.collision_index = t.get<int>("source_collision_index");
      p.sweep_time_s = t.get<double>("sweep_s");
      p.audited_stamp = audited_stamp;
      if (!p.point.allFinite() || p.sweeping_link.empty() || p.collision_index < 0 ||
          !std::isfinite(p.sweep_time_s) || p.sweep_time_s < 0.) return false;
      if (std::none_of(parsed.begin(), parsed.end(), [&](const PrimitiveObservationPoint& old) {
            return (old.point-p.point).lpNorm<Eigen::Infinity>() <= 1e-5;
          })) parsed.push_back(p);
      if (parsed.size() > 32) return false;
    }
    if (parsed.empty()) return false;
    *points = std::move(parsed);
    return true;
  } catch (...) { return false; }
}

struct PrimitiveObservationGuard::Impl {
  pinocchio::Model model;
  mutable pinocchio::Data data;
  care_confidence_map::VbcPrimitiveModel primitives;
  std::vector<int> q_indices, v_indices;
  std::vector<pinocchio::FrameIndex> frame_ids;
  pinocchio::FrameIndex base_id = 0;
  bool ready = false;
};
PrimitiveObservationGuard::PrimitiveObservationGuard() : impl_(new Impl) {}
PrimitiveObservationGuard::~PrimitiveObservationGuard() = default;

bool PrimitiveObservationGuard::initialize(const std::string& fk_urdf,
    const std::string& primitive_urdf, const std::string& base_frame,
    const std::vector<std::string>& joints, const std::vector<std::string>& ignored,
    std::string* error) {
  impl_.reset(new Impl);
  try {
    auto& p = *impl_;
    pinocchio::urdf::buildModel(fk_urdf, p.model);
    p.data = pinocchio::Data(p.model);
    if (!p.model.existFrame(base_frame)) throw std::runtime_error("Missing primitive guard base frame");
    p.base_id = p.model.getFrameId(base_frame);
    if (p.model.frames[p.base_id].parent != 0)
      throw std::runtime_error("Primitive observation guard requires a fixed base");
    std::set<std::string> unique_joints;
    for (const auto& name : joints) {
      if (!unique_joints.insert(name).second) throw std::runtime_error("Duplicate primitive guard joint");
      if (!p.model.existJointName(name)) throw std::runtime_error("Missing primitive guard joint: " + name);
      const auto id = p.model.getJointId(name);
      if (p.model.nqs[id] != 1 || p.model.nvs[id] != 1)
        throw std::runtime_error("Primitive guard requires independent scalar joints");
      p.q_indices.push_back(p.model.idx_qs[id]);
      p.v_indices.push_back(p.model.idx_vs[id]);
    }
    if (joints.empty() || joints.size() != static_cast<std::size_t>(p.model.nq))
      throw std::runtime_error("Incomplete primitive guard joint configuration");
    if (!p.primitives.load(primitive_urdf, ignored, error)) return false;
    for (const auto& body : p.primitives.primitives()) {
      if (!p.model.existFrame(body.link_name)) throw std::runtime_error("Missing primitive guard link: " + body.link_name);
      p.frame_ids.push_back(p.model.getFrameId(body.link_name));
    }
    p.ready = true;
    return true;
  } catch (const std::exception& e) {
    if (error) *error = e.what();
    return false;
  }
}

bool PrimitiveObservationGuard::framePose(const Eigen::VectorXd& q,
    const std::string& frame,Eigen::Isometry3d* pose) const {
  const auto& p=*impl_;
  if(!pose || !p.ready || q.size()!=static_cast<int>(p.q_indices.size()) ||
      !q.allFinite() || !p.model.existFrame(frame))return false;
  try {
    pinocchio::Data scratch(p.model);
    Eigen::VectorXd configuration=Eigen::VectorXd::Zero(p.model.nq);
    for(int j=0;j<q.size();++j)configuration[p.q_indices[j]]=q[j];
    pinocchio::framesForwardKinematics(p.model,scratch,configuration);
    const auto relative=scratch.oMf[p.base_id].inverse()*scratch.oMf[p.model.getFrameId(frame)];
    pose->matrix()=relative.toHomogeneousMatrix();return pose->matrix().allFinite();
  } catch(...) {return false;}
}

bool PrimitiveObservationGuard::linearize(const Eigen::MatrixXd& q,
    const std::vector<PrimitiveObservationPoint>& points, std::vector<Row>* rows,
    std::string* error, bool nearest_only_proposal) const {
  if (!rows) return false;
  rows->clear();
  try {
    auto& p = *impl_;
    if (!p.ready || q.rows() != static_cast<int>(p.q_indices.size()) ||
        q.cols() < 1 || !q.allFinite() || points.size() > 32)
      throw std::runtime_error("Invalid primitive observation guard input");
    const auto& local = p.primitives.primitives();
    std::vector<care_confidence_map::VbcPrimitive> world(local.size());
    std::vector<Eigen::MatrixXd> jacobians(local.size());
    std::vector<Eigen::Vector3d> origins(local.size());
    for (int k = 0; k < q.cols(); ++k) {
      Eigen::VectorXd full = pinocchio::neutral(p.model);
      for (int j = 0; j < q.rows(); ++j) full[p.q_indices[j]] = q(j,k);
      // One FK/Jacobian pass per knot, shared across all points and bodies.
      pinocchio::computeJointJacobians(p.model, p.data, full);
      pinocchio::updateFramePlacements(p.model, p.data);
      const auto base_inverse = p.data.oMf[p.base_id].inverse();
      for (std::size_t i = 0; i < local.size(); ++i) {
        const auto pose = base_inverse * p.data.oMf[p.frame_ids[i]];
        world[i] = local[i];
        world[i].center = pose.translation() + pose.rotation()*local[i].center;
        world[i].rotation = pose.rotation()*local[i].rotation;
        origins[i] = pose.translation();
        Eigen::Matrix<double,6,Eigen::Dynamic> J(6,p.model.nv);
        J.setZero();
        pinocchio::getFrameJacobian(p.model,p.data,p.frame_ids[i],pinocchio::LOCAL_WORLD_ALIGNED,J);
        jacobians[i].resize(6,q.rows());
        for (int j = 0; j < q.rows(); ++j) {
          jacobians[i].col(j).head<3>() = base_inverse.rotation()*J.col(p.v_indices[j]).head<3>();
          jacobians[i].col(j).tail<3>() = base_inverse.rotation()*J.col(p.v_indices[j]).tail<3>();
        }
      }
      for (const auto& point : points) {
        if (!point.point.allFinite()) throw std::runtime_error("Nonfinite primitive observation point");
        std::size_t nearest = 0, swept = local.size();
        double distance = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < local.size(); ++i) {
          const double d = world[i].signedDistance(point.point);
          if (d < distance) { distance = d; nearest = i; }
          if (local[i].link_name == point.sweeping_link &&
              local[i].collision_index == point.collision_index &&
              local[i].collision_name == point.collision_name) swept = i;
        }
        if (!nearest_only_proposal && !point.ordinary_path_guard && swept == local.size()) throw std::runtime_error("VBC primitive identity does not match guard geometry");
        std::vector<std::size_t> selected{nearest};
        if (!nearest_only_proposal && !point.ordinary_path_guard && swept != nearest) selected.push_back(swept);
        for (const auto i : selected) {
          Row row;
          row.timestep = k; row.point = point.point;
          row.q_linearization = q.col(k);
          row.distance_m = world[i].signedDistance(point.point);
          row.link = local[i].link_name; row.collision_name = local[i].collision_name;
          // Differentiate the primitive SDF in workspace, then apply the rigid
          // frame Jacobian. No perturbed joint FK or learned model is involved.
          Eigen::Vector3d normal;
          constexpr double h = 1e-6;
          for (int axis = 0; axis < 3; ++axis) {
            Eigen::Vector3d plus = point.point, minus = point.point;
            plus[axis] += h; minus[axis] -= h;
            normal[axis] = (world[i].signedDistance(plus)-world[i].signedDistance(minus))/(2*h);
          }
          row.gradient.resize(q.rows());
          for (int j = 0; j < q.rows(); ++j) {
            const Eigen::Vector3d velocity = jacobians[i].col(j).head<3>() +
                jacobians[i].col(j).tail<3>().cross(point.point-origins[i]);
            row.gradient[j] = -normal.dot(velocity);
          }
          if (!std::isfinite(row.distance_m) || !row.gradient.allFinite())
            throw std::runtime_error("Nonfinite primitive observation linearization");
          rows->push_back(std::move(row));
        }
      }
    }
    return true;
  } catch (const std::exception& e) {
    rows->clear();
    if (error) *error = e.what();
    return false;
  }
}
}  // namespace egocentric_arm_planner
