#include "egocentric_arm_planner/normal_executable_limits.hpp"
#include "egocentric_arm_planner/normal_executable_curve.hpp"
#include "egocentric_arm_planner/observation_local_decision.hpp"
#include "egocentric_arm_planner/executable_task_objective.hpp"
#include "egocentric_arm_planner/local_sparse_scp_planner.hpp"
#include "egocentric_arm_planner/geometric_restore_proposal.hpp"
#include "egocentric_arm_planner/visibility_objective_step.hpp"
#include "egocentric_arm_planner/measured_braking_seed.hpp"
#include "egocentric_arm_planner/qp_box_conflict.hpp"
#include "egocentric_arm_planner/qp_reachable_conflict.hpp"
#include "egocentric_arm_planner/primitive_reachable_conflict.hpp"
#include "egocentric_arm_planner/qp_snapshot.hpp"
#include <cstdio>
#include <cfenv>
#include <iostream>
#include "egocentric_arm_planner/candidate_audit_context.hpp"
#include <iomanip>

#include <arm_trajectory/joint_trajectory.hpp>
#include <care_collision_cdf/QueryCollisionCDFPairs.h>
#include <piqp/piqp.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace egocentric_arm_planner {

// Opt-in diagnostics share the exact setup/solve entry with the running planner.
// Input snapshots are serialized before setup; failed solves cannot alter them.
static std::string sparseQpInputJson(
    piqp::SparseSolver<double>& solver,
    const Eigen::SparseMatrix<double>& P, const Eigen::VectorXd& c,
    const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b,
    const Eigen::SparseMatrix<double>& G, const Eigen::VectorXd& hl,
    const Eigen::VectorXd& hu, const Eigen::VectorXd& xl, const Eigen::VectorXd& xu) {
  std::ostringstream out;
  const auto& s=solver.settings();
  out << "{\"schema\":2,\"settings\":{\"schema\":1";
  out << ",\"rho_init\":"; jsonNumber(out,static_cast<double>(s.rho_init));
  out << ",\"delta_init\":"; jsonNumber(out,static_cast<double>(s.delta_init));
  out << ",\"eps_abs\":"; jsonNumber(out,static_cast<double>(s.eps_abs));
  out << ",\"eps_rel\":"; jsonNumber(out,static_cast<double>(s.eps_rel));
  out << ",\"check_duality_gap\":"; jsonNumber(out,static_cast<double>(s.check_duality_gap));
  out << ",\"eps_duality_gap_abs\":"; jsonNumber(out,static_cast<double>(s.eps_duality_gap_abs));
  out << ",\"eps_duality_gap_rel\":"; jsonNumber(out,static_cast<double>(s.eps_duality_gap_rel));
  out << ",\"infeasibility_threshold\":"; jsonNumber(out,static_cast<double>(s.infeasibility_threshold));
  out << ",\"reg_lower_limit\":"; jsonNumber(out,static_cast<double>(s.reg_lower_limit));
  out << ",\"reg_finetune_lower_limit\":"; jsonNumber(out,static_cast<double>(s.reg_finetune_lower_limit));
  out << ",\"reg_finetune_primal_update_threshold\":"; jsonNumber(out,static_cast<double>(s.reg_finetune_primal_update_threshold));
  out << ",\"reg_finetune_dual_update_threshold\":"; jsonNumber(out,static_cast<double>(s.reg_finetune_dual_update_threshold));
  out << ",\"max_iter\":"; jsonNumber(out,static_cast<double>(s.max_iter));
  out << ",\"max_factor_retires\":"; jsonNumber(out,static_cast<double>(s.max_factor_retires));
  out << ",\"preconditioner_scale_cost\":"; jsonNumber(out,static_cast<double>(s.preconditioner_scale_cost));
  out << ",\"preconditioner_reuse_on_update\":"; jsonNumber(out,static_cast<double>(s.preconditioner_reuse_on_update));
  out << ",\"preconditioner_iter\":"; jsonNumber(out,static_cast<double>(s.preconditioner_iter));
  out << ",\"tau\":"; jsonNumber(out,static_cast<double>(s.tau));
  out << ",\"kkt_solver\":"; jsonNumber(out,static_cast<double>(s.kkt_solver));
  out << ",\"iterative_refinement_always_enabled\":"; jsonNumber(out,static_cast<double>(s.iterative_refinement_always_enabled));
  out << ",\"iterative_refinement_eps_abs\":"; jsonNumber(out,static_cast<double>(s.iterative_refinement_eps_abs));
  out << ",\"iterative_refinement_eps_rel\":"; jsonNumber(out,static_cast<double>(s.iterative_refinement_eps_rel));
  out << ",\"iterative_refinement_max_iter\":"; jsonNumber(out,static_cast<double>(s.iterative_refinement_max_iter));
  out << ",\"iterative_refinement_min_improvement_rate\":"; jsonNumber(out,static_cast<double>(s.iterative_refinement_min_improvement_rate));
  out << ",\"iterative_refinement_static_regularization_eps\":"; jsonNumber(out,static_cast<double>(s.iterative_refinement_static_regularization_eps));
  out << ",\"iterative_refinement_static_regularization_rel\":"; jsonNumber(out,static_cast<double>(s.iterative_refinement_static_regularization_rel));
  out << ",\"verbose\":"; jsonNumber(out,static_cast<double>(s.verbose));
  out << ",\"compute_timings\":"; jsonNumber(out,static_cast<double>(s.compute_timings));
  out << "},\"rounding_mode\":" << std::fegetround()
      << ",\"eigen_version\":[" << EIGEN_WORLD_VERSION << ',' << EIGEN_MAJOR_VERSION << ',' << EIGEN_MINOR_VERSION
      << "],\"max_iter\":" << s.max_iter << ",\"eps_abs\":"; jsonNumber(out,s.eps_abs);
  out << ",\"eps_rel\":";jsonNumber(out,s.eps_rel);
  out << ",\"P\":";jsonSparse(out,P);out << ",\"c\":";jsonVector(out,c);
  out << ",\"A\":";jsonSparse(out,A);out << ",\"b\":";jsonVector(out,b);
  out << ",\"G\":";jsonSparse(out,G);out << ",\"h_l\":";jsonVector(out,hl);
  out << ",\"h_u\":";jsonVector(out,hu);out << ",\"x_l\":";jsonVector(out,xl);
  out << ",\"x_u\":";jsonVector(out,xu);out << '}';return out.str();
}

static piqp::Status runSparseQpEntry(
    piqp::SparseSolver<double>& solver,
    const Eigen::SparseMatrix<double>& P, const Eigen::VectorXd& c,
    const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b,
    const Eigen::SparseMatrix<double>& G, const Eigen::VectorXd& hl,
    const Eigen::VectorXd& hu, const Eigen::VectorXd& xl, const Eigen::VectorXd& xu,
    unsigned long long stamp) {
  const char* folder=std::getenv("CARE_QP_CAPTURE_DIR");
  const bool capture=folder && *folder;
  const auto before=capture ? sparseQpInputJson(solver,P,c,A,b,G,hl,hu,xl,xu) : std::string{};
  solver.setup(P,c,A,b,G,hl,hu,xl,xu);
  const auto status=solver.solve();
  if(capture) {
    static std::atomic<unsigned> serial{0}; const auto id=serial++;
    if(status!=piqp::PIQP_SOLVED || id==0) {
      const auto after=sparseQpInputJson(solver,P,c,A,b,G,hl,hu,xl,xu);
      std::ofstream out(std::string(folder)+"/qp_"+std::to_string(stamp)+"_"+std::to_string(id)+".json");
      out << "{\"query_stamp_ns\":" << stamp << ",\"input\":" << before
          << ",\"input_unchanged\":" << (before==after ? "true" : "false")
          << ",\"status\":";jsonString(out,piqp::status_to_string(status));
      out << ",\"iterations\":" << solver.result().info.iter
          << ",\"primal_residual\":";jsonNumber(out,solver.result().info.primal_res);
      out << ",\"dual_residual\":";jsonNumber(out,solver.result().info.dual_res);out << "}\n";
    }
  }
  return status;
}

// Offline mode of the production binary: no ROS master, publisher or actuator.
int replaySparseQpFile(const char* path) {
  try {
    boost::property_tree::ptree document;boost::property_tree::read_json(path,document);
    const auto& tree=document.get_child_optional("input") ? document.get_child("input") :
        document.get_child_optional("subproblem") ? document.get_child("subproblem") : document;
    auto vec=[](const boost::property_tree::ptree& t) {
      Eigen::VectorXd v(t.size());int i=0;for(const auto& e:t)v[i++]=std::stod(e.second.data());return v;
    };
    auto mat=[&](const boost::property_tree::ptree& t) {
      const auto shape=vec(t.get_child("shape"));Eigen::SparseMatrix<double> m(shape[0],shape[1]);
      std::vector<Eigen::Triplet<double>> triplets;
      for(const auto& e:t.get_child("coo")){const auto a=vec(e.second);triplets.emplace_back(a[0],a[1],a[2]);}
      m.setFromTriplets(triplets.begin(),triplets.end());m.makeCompressed();
      if(t.get_child_optional("outer")) {
        const auto outer=vec(t.get_child("outer")),inner=vec(t.get_child("inner")),values=vec(t.get_child("values"));
        if(outer.size()!=m.outerSize()+1 || inner.size()!=m.nonZeros() || values.size()!=m.nonZeros())throw std::runtime_error("sparse storage size differs");
        for(int i=0;i<outer.size();++i)if(outer[i]!=m.outerIndexPtr()[i])throw std::runtime_error("sparse outer differs");
        for(int i=0;i<inner.size();++i)if(inner[i]!=m.innerIndexPtr()[i] || values[i]!=m.valuePtr()[i])throw std::runtime_error("sparse values/inner differ");
      }
      return m;
    };
    piqp::SparseSolver<double> solver;auto& settings=solver.settings();
    settings.max_iter=tree.get<int>("max_iter");settings.eps_abs=tree.get<double>("eps_abs");
    settings.eps_rel=tree.get<double>("eps_rel");settings.compute_timings=true;settings.kkt_solver=piqp::KKTSolver::sparse_ldlt;
    settings.rho_init = static_cast<decltype(settings.rho_init)>(tree.get<double>("settings.rho_init", static_cast<double>(settings.rho_init)));
    settings.delta_init = static_cast<decltype(settings.delta_init)>(tree.get<double>("settings.delta_init", static_cast<double>(settings.delta_init)));
    settings.eps_abs = static_cast<decltype(settings.eps_abs)>(tree.get<double>("settings.eps_abs", static_cast<double>(settings.eps_abs)));
    settings.eps_rel = static_cast<decltype(settings.eps_rel)>(tree.get<double>("settings.eps_rel", static_cast<double>(settings.eps_rel)));
    settings.check_duality_gap = static_cast<decltype(settings.check_duality_gap)>(tree.get<double>("settings.check_duality_gap", static_cast<double>(settings.check_duality_gap)));
    settings.eps_duality_gap_abs = static_cast<decltype(settings.eps_duality_gap_abs)>(tree.get<double>("settings.eps_duality_gap_abs", static_cast<double>(settings.eps_duality_gap_abs)));
    settings.eps_duality_gap_rel = static_cast<decltype(settings.eps_duality_gap_rel)>(tree.get<double>("settings.eps_duality_gap_rel", static_cast<double>(settings.eps_duality_gap_rel)));
    settings.infeasibility_threshold = static_cast<decltype(settings.infeasibility_threshold)>(tree.get<double>("settings.infeasibility_threshold", static_cast<double>(settings.infeasibility_threshold)));
    settings.reg_lower_limit = static_cast<decltype(settings.reg_lower_limit)>(tree.get<double>("settings.reg_lower_limit", static_cast<double>(settings.reg_lower_limit)));
    settings.reg_finetune_lower_limit = static_cast<decltype(settings.reg_finetune_lower_limit)>(tree.get<double>("settings.reg_finetune_lower_limit", static_cast<double>(settings.reg_finetune_lower_limit)));
    settings.reg_finetune_primal_update_threshold = static_cast<decltype(settings.reg_finetune_primal_update_threshold)>(tree.get<double>("settings.reg_finetune_primal_update_threshold", static_cast<double>(settings.reg_finetune_primal_update_threshold)));
    settings.reg_finetune_dual_update_threshold = static_cast<decltype(settings.reg_finetune_dual_update_threshold)>(tree.get<double>("settings.reg_finetune_dual_update_threshold", static_cast<double>(settings.reg_finetune_dual_update_threshold)));
    settings.max_iter = static_cast<decltype(settings.max_iter)>(tree.get<double>("settings.max_iter", static_cast<double>(settings.max_iter)));
    settings.max_factor_retires = static_cast<decltype(settings.max_factor_retires)>(tree.get<double>("settings.max_factor_retires", static_cast<double>(settings.max_factor_retires)));
    settings.preconditioner_scale_cost = static_cast<decltype(settings.preconditioner_scale_cost)>(tree.get<double>("settings.preconditioner_scale_cost", static_cast<double>(settings.preconditioner_scale_cost)));
    settings.preconditioner_reuse_on_update = static_cast<decltype(settings.preconditioner_reuse_on_update)>(tree.get<double>("settings.preconditioner_reuse_on_update", static_cast<double>(settings.preconditioner_reuse_on_update)));
    settings.preconditioner_iter = static_cast<decltype(settings.preconditioner_iter)>(tree.get<double>("settings.preconditioner_iter", static_cast<double>(settings.preconditioner_iter)));
    settings.tau = static_cast<decltype(settings.tau)>(tree.get<double>("settings.tau", static_cast<double>(settings.tau)));
    settings.kkt_solver = static_cast<decltype(settings.kkt_solver)>(tree.get<double>("settings.kkt_solver", static_cast<double>(settings.kkt_solver)));
    settings.iterative_refinement_always_enabled = static_cast<decltype(settings.iterative_refinement_always_enabled)>(tree.get<double>("settings.iterative_refinement_always_enabled", static_cast<double>(settings.iterative_refinement_always_enabled)));
    settings.iterative_refinement_eps_abs = static_cast<decltype(settings.iterative_refinement_eps_abs)>(tree.get<double>("settings.iterative_refinement_eps_abs", static_cast<double>(settings.iterative_refinement_eps_abs)));
    settings.iterative_refinement_eps_rel = static_cast<decltype(settings.iterative_refinement_eps_rel)>(tree.get<double>("settings.iterative_refinement_eps_rel", static_cast<double>(settings.iterative_refinement_eps_rel)));
    settings.iterative_refinement_max_iter = static_cast<decltype(settings.iterative_refinement_max_iter)>(tree.get<double>("settings.iterative_refinement_max_iter", static_cast<double>(settings.iterative_refinement_max_iter)));
    settings.iterative_refinement_min_improvement_rate = static_cast<decltype(settings.iterative_refinement_min_improvement_rate)>(tree.get<double>("settings.iterative_refinement_min_improvement_rate", static_cast<double>(settings.iterative_refinement_min_improvement_rate)));
    settings.iterative_refinement_static_regularization_eps = static_cast<decltype(settings.iterative_refinement_static_regularization_eps)>(tree.get<double>("settings.iterative_refinement_static_regularization_eps", static_cast<double>(settings.iterative_refinement_static_regularization_eps)));
    settings.iterative_refinement_static_regularization_rel = static_cast<decltype(settings.iterative_refinement_static_regularization_rel)>(tree.get<double>("settings.iterative_refinement_static_regularization_rel", static_cast<double>(settings.iterative_refinement_static_regularization_rel)));
    settings.verbose = static_cast<decltype(settings.verbose)>(tree.get<double>("settings.verbose", static_cast<double>(settings.verbose)));
    settings.compute_timings = static_cast<decltype(settings.compute_timings)>(tree.get<double>("settings.compute_timings", static_cast<double>(settings.compute_timings)));
    const auto P=mat(tree.get_child("P")),A=mat(tree.get_child("A")),G=mat(tree.get_child("G"));
    const auto c=vec(tree.get_child("c")),b=vec(tree.get_child("b")),hl=vec(tree.get_child("h_l")),hu=vec(tree.get_child("h_u")),xl=vec(tree.get_child("x_l")),xu=vec(tree.get_child("x_u"));
    const auto status=runSparseQpEntry(solver,P,c,A,b,G,hl,hu,xl,xu,0);
    std::cout << "{\"status\":";jsonString(std::cout,piqp::status_to_string(status));
    std::cout << ",\"iterations\":" << solver.result().info.iter << ",\"primal_residual\":";
    jsonNumber(std::cout,solver.result().info.primal_res);std::cout << ",\"dual_residual\":";
    jsonNumber(std::cout,solver.result().info.dual_res);std::cout << "}\n";return 0;
  } catch(const std::exception& e) {std::cerr << e.what() << '\n';return 2;}
}

namespace {

bool finiteVector(const Eigen::VectorXd& x) {
  for (int i = 0; i < x.size(); ++i) {
    if (!std::isfinite(x[i])) return false;
  }
  return true;
}

bool finiteMatrix(const Eigen::MatrixXd& x) {
  for (int r = 0; r < x.rows(); ++r) {
    for (int c = 0; c < x.cols(); ++c) {
      if (!std::isfinite(x(r, c))) return false;
    }
  }
  return true;
}

Eigen::VectorXd jointMaskFromLayout(
    const std_msgs::MultiArrayLayout& layout, int dof) {
  Eigen::VectorXd mask = Eigen::VectorXd::Ones(dof);
  const std::string prefix = "care_qmask_v1_";
  for (const auto& dim : layout.dim) {
    if (dim.label.compare(0, prefix.size(), prefix) != 0) continue;
    const std::string bits = dim.label.substr(prefix.size());
    if (bits.size() != static_cast<std::size_t>(dof)) return mask;
    for (int j = 0; j < dof; ++j) {
      if (bits[static_cast<std::size_t>(j)] != '0' &&
          bits[static_cast<std::size_t>(j)] != '1') return mask;
      mask[j] = bits[static_cast<std::size_t>(j)] == '1' ? 1.0 : 0.0;
    }
    return mask;
  }
  return mask;
}

double clampValue(double x, double lo, double hi) {
  return std::max(lo, std::min(hi, x));
}

bool parseUnsignedToken(
    const std::string& text,
    const std::string& key,
    unsigned long long* value) {
  if (value == nullptr) return false;
  const std::string needle = key + "=";
  const std::size_t pos = text.find(needle);
  if (pos == std::string::npos) return false;
  const std::size_t begin = pos + needle.size();
  std::size_t end = begin;
  while (end < text.size() &&
         text[end] >= '0' && text[end] <= '9') {
    ++end;
  }
  if (end == begin) return false;
  try {
    *value = std::stoull(text.substr(begin, end - begin));
    return true;
  } catch (...) {
    return false;
  }
}

bool parseVbcEvidencePoints(
    const std::string& evidence, std::vector<Eigen::Vector3d>* points) {
  if (points == nullptr) return false;
  points->clear();
  const std::string marker = "\"point\":[";
  std::size_t search = 0;
  while (true) {
    const std::size_t begin = evidence.find(marker, search);
    if (begin == std::string::npos) break;
    const char* cursor = evidence.c_str() + begin + marker.size();
    Eigen::Vector3d parsed = Eigen::Vector3d::Zero();
    bool valid = true;
    for (int i = 0; i < 3; ++i) {
      char* end = nullptr;
      const double value = std::strtod(cursor, &end);
      if (end == cursor || !std::isfinite(value)) {
        valid = false;
        break;
      }
      parsed[i] = value;
      cursor = end;
      if (i < 2) {
        if (*cursor != ',') {
          valid = false;
          break;
        }
        ++cursor;
      }
    }
    if (valid && std::none_of(
            points->begin(), points->end(), [&](const Eigen::Vector3d& old) {
              return (old - parsed).lpNorm<Eigen::Infinity>() <= 1e-5;
            })) {
      points->push_back(parsed);
    }
    search = begin + marker.size();
  }
  return !points->empty();
}

// Stricter parser for cross-identity scheduling memory. Legacy point feedback
// keeps its existing parser; malformed/foreign evidence cannot enable inheritance.
bool failureMemoryEvidence(const std::string& text, std::uint64_t* stamp,
                           VbcFailureDirectionMemory::Points* points) {
  if (text.size()>256*1024) return false;
  try {
    boost::property_tree::ptree tree;
    std::istringstream stream(text);
    boost::property_tree::read_json(stream,tree);
    if (tree.count("trajectory_stamp_ns")!=1 || tree.count("violations")!=1) return false;
    const auto id=tree.get<std::string>("trajectory_stamp_ns");
    if (id.empty() || id.find_first_not_of("0123456789")!=std::string::npos) return false;
    *stamp=std::stoull(id); points->clear();
    for (const auto& row:tree.get_child("violations")) {
      if (!row.first.empty() || row.second.count("point")!=1) return false;
      const auto& values=row.second.get_child("point");
      if (values.size()!=3 || points->size()>=VbcFailureDirectionMemory::max_points) return false;
      Eigen::Vector3d p; int j=0;
      for (const auto& v:values) {
        if (!v.first.empty()) return false;
        p[j++]=v.second.get_value<double>();
      }
      if (!p.allFinite()) return false;
      points->push_back(p);
    }
    return *stamp && !points->empty();
  } catch (...) { return false; }
}

}  // namespace

LocalSparseSCPPlanner::~LocalSparseSCPPlanner() {
  stopWorker();
}

bool LocalSparseSCPPlanner::initialize(
    const ros::NodeHandle& nh,
    const ros::NodeHandle& pnh) {
  nh_ = nh;
  pnh_ = pnh;

  if (!loadConfig()) return false;
  if (!loadJointLimits()) return false;
  pnh_.param("local_planner/primitive_observation_guard/enabled",
             primitive_observation_guard_enabled_, false);
  pnh_.param("local_planner/unknown_corridor_guard_enabled", unknown_corridor_guard_enabled_, true);
  if (primitive_observation_guard_enabled_) {
    std::string fk_urdf, primitive_urdf, base_frame, error;
    std::vector<std::string> ignored;
    if (!pnh_.getParam("local_planner/primitive_observation_guard/fk_urdf", fk_urdf) ||
        !pnh_.getParam("local_planner/primitive_observation_guard/primitive_urdf", primitive_urdf) ||
        !pnh_.getParam("local_planner/primitive_observation_guard/base_frame", base_frame) ||
        !pnh_.getParam("local_planner/primitive_observation_guard/ignored_links", ignored) ||
        !pnh_.getParam("local_planner/primitive_observation_guard/margin_m", primitive_observation_margin_m_) ||
        !std::isfinite(primitive_observation_margin_m_) || primitive_observation_margin_m_ < 0. ||
        !primitive_observation_guard_.initialize(fk_urdf, primitive_urdf, base_frame,
             joint_names_, ignored, &error)) {
      ROS_ERROR_STREAM("Cannot initialize primitive observation guard: " << error);
      return false;
    }
  }

  pnh_.param("local_planner/repair_executable_view/enabled", repair_executable_view_enabled_, false);
  pnh_.param("local_planner/repair_prefix_binding/enabled", repair_prefix_binding_enabled_, false);
  if (repair_prefix_binding_enabled_ && !repair_executable_view_enabled_) {
    ROS_ERROR("Selected-prefix binding requires the executable view"); return false;
  }
  pnh_.param("local_planner/fixed_start_recovery_enabled", fixed_start_recovery_enabled_, false);
  pnh_.param("local_planner/initial_shell_recovery/enabled", initial_shell_recovery_enabled_, false);
  pnh_.param("local_planner/initial_shell_recovery/execution_margin_m", initial_shell_execution_margin_m_, 0.);
  if (initial_shell_recovery_enabled_ && (!primitive_observation_guard_enabled_ ||
      !repair_executable_view_enabled_ || !std::isfinite(initial_shell_execution_margin_m_) ||
      initial_shell_execution_margin_m_<0. || initial_shell_execution_margin_m_>=primitive_observation_margin_m_)) {
    ROS_ERROR("Invalid initial-shell recovery geometry contract"); return false;
  }
  if (repair_executable_view_enabled_) {
    double prefix = 0.;
    bool cycle = false;
    std::vector<double> fallback, acceleration;
    if (!primitive_observation_guard_enabled_ ||
        !pnh_.getParam("local_planner/repair_executable_view/prefix_s", prefix) ||
        !pnh_.getParam("local_planner/repair_executable_view/cycle_enabled", cycle) ||
        !pnh_.getParam("local_planner/repair_executable_view/cycle_prefixes_s", fallback) ||
        !pnh_.getParam("local_planner/repair_executable_view/brake_dt_s", repair_executable_settings_.brake_dt) ||
        !pnh_.getParam("local_planner/repair_executable_view/max_brake_steps", repair_executable_settings_.max_brake_steps) ||
        !pnh_.getParam("local_planner/repair_executable_view/acceleration_limits", acceleration)) {
      ROS_ERROR("Missing executable view contract"); return false;
    }
    repair_executable_settings_.prefixes = {prefix};
    if (cycle) for (double p : fallback) if (p <= prefix+1e-12) repair_executable_settings_.prefixes.push_back(p);
    auto& prefixes = repair_executable_settings_.prefixes;
    std::sort(prefixes.begin(), prefixes.end(), std::greater<double>());
    prefixes.erase(std::unique(prefixes.begin(), prefixes.end(),
        [](double a, double b) { return std::abs(a-b) <= 1e-9; }), prefixes.end());
    repair_executable_settings_.acceleration = Eigen::Map<Eigen::VectorXd>(acceleration.data(), acceleration.size());
    std::vector<RepairExecutableSample> samples;
    std::string error;
    if (!repairExecutableSamples(Eigen::MatrixXd::Zero(dof_,num_intervals_+1),
        Eigen::MatrixXd::Zero(dof_,num_intervals_),dt_,repair_executable_settings_,&samples,&error)) {
      ROS_ERROR_STREAM(error); return false;
    }
  }

  const char* probe_task_policy=std::getenv("CARE_PROBE_EXECUTABLE_TASK");
  probe_task_objective_enabled_=probe_task_policy && std::string(probe_task_policy)=="true";
  probe_task_settings_.acceleration=repair_executable_settings_.acceleration;
  nh_.param("/optimized_trajectory_continuity/probe_execution_prefix_s",probe_task_settings_.prefix,.25);
  nh_.param("/optimized_trajectory_continuity/probe_velocity_cap",probe_task_settings_.velocity_cap,.5);
  nh_.param("/optimized_trajectory_continuity/probe_time_scale_max",probe_task_settings_.time_scale_max,5.);
  nh_.param("/optimized_trajectory_continuity/repair_brake_dt_s",probe_task_settings_.brake_dt,.05);
  nh_.param("/optimized_trajectory_continuity/repair_brake_max_steps",probe_task_settings_.brake_steps,20);
  nh_.param("/optimized_trajectory_continuity/repair_hold_s",probe_task_settings_.hold,.1);
  latest_executed_command_ = Eigen::VectorXd::Zero(dof_);
  pnh_.param("local_planner/terminal_retention/enabled",terminal_retention_enabled_,false);
  const char* repair_goal_policy=std::getenv("CARE_REPAIR_GOAL_STOPPING");
  repair_goal_stopping_enabled_=repair_goal_policy && std::string(repair_goal_policy)=="true";
  pnh_.param<std::string>("base_frame",terminal_base_frame_,"base_link");
  pnh_.param<std::string>("ee_frame",terminal_ee_frame_,"EE_link");
  if(terminal_retention_enabled_ && !primitive_observation_guard_enabled_) {
    ROS_ERROR("Terminal retention requires the initialized objective FK model");return false;
  }
  if(terminal_retention_enabled_ || primitive_observation_guard_.knownGroundEnabled()) {
    terminal_goal_sub_=nh_.subscribe("/care_planner/ee_target_pose",1,&LocalSparseSCPPlanner::terminalGoalCallback,this);
  }
  if(terminal_retention_enabled_) {
    terminal_audit_sub_=nh_.subscribe("/care_planner/optimized_trajectory",16,&LocalSparseSCPPlanner::terminalAuditCallback,this);
    terminal_commit_sub_=nh_.subscribe("/care_planner/committed_trajectory",16,&LocalSparseSCPPlanner::terminalCommitCallback,this);
  }
  if (repair_prefix_binding_enabled_) {
    repair_prefix_selection_sub_ = nh_.subscribe("/care_planner/repair_prefix_selection",8,
        &LocalSparseSCPPlanner::repairPrefixSelectionCallback,this);
    planned_candidate_pub_ = nh_.advertise<egocentric_arm_planner::CandidateTransfer>(
        candidate_trajectory_topic_+"/planned",1,false);
  }
  latest_single_waypoint_q_ = Eigen::VectorXd::Zero(dof_);
  latest_single_waypoint_joint_mask_ = Eigen::VectorXd::Ones(dof_);
  latest_frontier_.q = Eigen::VectorXd::Zero(dof_);
  latest_frontier_.joint_mask = Eigen::VectorXd::Ones(dof_);
  plan_frontier_.q = Eigen::VectorXd::Zero(dof_);
  plan_frontier_.joint_mask = Eigen::VectorXd::Ones(dof_);

  joint_state_sub_ = nh_.subscribe(
      joint_state_topic_, 1,
      &LocalSparseSCPPlanner::jointStateCallback, this);
  reference_sub_ = nh_.subscribe(
      reference_topic_, 1,
      &LocalSparseSCPPlanner::referenceCallback, this);
  task_reference_status_sub_ = nh_.subscribe(
      "/care_planner/task_reference_status", 10,
      &LocalSparseSCPPlanner::taskReferenceStatusCallback, this);
  waypoint_schedule_sub_ = nh_.subscribe(
      waypoint_schedule_topic_, 1,
      &LocalSparseSCPPlanner::waypointScheduleCallback, this);
  vbc_obligation_points_sub_ = nh_.subscribe(
      vbc_obligation_points_topic_, 1,
      &LocalSparseSCPPlanner::vbcObligationPointsCallback, this);
  visibility_frontier_sub_ = nh_.subscribe(
      visibility_frontier_topic_, 1,
      &LocalSparseSCPPlanner::visibilityFrontierCallback, this);
  visibility_guidance_sub_ = nh_.subscribe(
      "/care_planner/active_sensing/visibility_guidance", 1,
      &LocalSparseSCPPlanner::visibilityGuidanceCallback, this);
  single_waypoint_active_sub_ = nh_.subscribe(
      single_waypoint_active_topic_, 1,
      &LocalSparseSCPPlanner::singleWaypointActiveCallback, this);
  if (std::getenv("CARE_NATIVE_VIEW_SELECTION") &&
      std::string(std::getenv("CARE_NATIVE_VIEW_SELECTION")) == "true") {
    const char* preview_mode=std::getenv("CARE_NATIVE_VIEW_MODE");
    view_preview_aligned_enabled_=preview_mode && std::string(preview_mode)!="legacy";
    view_preview_pub_=nh_.advertise<std_msgs::String>("/care_planner/view_preview/response",2,false);
    view_preview_sub_=nh_.subscribe("/care_planner/view_preview/request",1,
        &LocalSparseSCPPlanner::viewPreviewCallback,this);
  }
  single_waypoint_q_sub_ = nh_.subscribe(
      single_waypoint_q_topic_, 1,
      &LocalSparseSCPPlanner::singleWaypointQCallback, this);
  recovery_sub_ = nh_.subscribe(
      recovery_topic_, 1,
      &LocalSparseSCPPlanner::recoveryCallback, this);
  probe_active_sub_ = nh_.subscribe(
      probe_active_topic_, 1,
      &LocalSparseSCPPlanner::probeActiveCallback, this);
  replan_request_sub_ = nh_.subscribe(
      replan_request_topic_, 10,
      &LocalSparseSCPPlanner::replanRequestCallback, this);
  qp_retry_request_sub_ = nh_.subscribe(
      replan_request_topic_ + "/qp_retry", 10,
      &LocalSparseSCPPlanner::qpRetryRequestCallback, this);
  qp_retry_ack_pub_ = nh_.advertise<std_msgs::String>(
      replan_request_topic_ + "/qp_retry_ack", 10, false);
  smooth_replan_request_sub_ = nh_.subscribe(
      smooth_replan_request_topic_, 10,
      &LocalSparseSCPPlanner::smoothReplanRequestCallback, this);
  executed_command_sub_ = nh_.subscribe(
      executed_command_topic_, 2,
      &LocalSparseSCPPlanner::executedCommandCallback, this);
  execution_summary_sub_ = nh_.subscribe(
      execution_summary_topic_, 10,
      &LocalSparseSCPPlanner::executionSummaryCallback, this);
  std::string desired_command_topic, limiter_summary_topic;
  pnh_.param<std::string>("local_planner/desired_command_topic", desired_command_topic,
                         "/care_planner/execution/tracker_velocity_desired");
  pnh_.param<std::string>("local_planner/limiter_summary_topic", limiter_summary_topic,
                         "/care_planner/execution/rate_limiter_summary");
  desired_command_sub_ = nh_.subscribe(
      desired_command_topic, 2,
      &LocalSparseSCPPlanner::desiredCommandCallback, this);
  limiter_summary_sub_ = nh_.subscribe(
      limiter_summary_topic, 2,
      &LocalSparseSCPPlanner::limiterSummaryCallback, this);
  std::string vbc_hold_topic,gcdf_hold_topic;
  pnh_.param<std::string>("execution/vbc_safety_hold_topic",vbc_hold_topic,
                         "/care_planner/execution/predicted_vbc_verification_hold");
  pnh_.param<std::string>("execution/external_safety_hold_topic",gcdf_hold_topic,
                         "/care_planner/execution_gcdf/hard_hold");
  vbc_execution_hold_sub_=nh_.subscribe(vbc_hold_topic,2,
      &LocalSparseSCPPlanner::vbcExecutionHoldCallback,this);
  gcdf_execution_hold_sub_=nh_.subscribe(gcdf_hold_topic,2,
      &LocalSparseSCPPlanner::gcdfExecutionHoldCallback,this);
  observation_progress_sub_ = nh_.subscribe(
      "/care_planner/active_sensing/visibility_acquisition_summary", 2,
      &LocalSparseSCPPlanner::observationProgressCallback, this);
  cdf_batch_sub_ = nh_.subscribe(
      cdf_batch_topic_, 2,
      &LocalSparseSCPPlanner::cdfConstraintBatchCallback, this);
  witness_request_pub_ = nh_.advertise<care_collision_cdf::CollisionCDFWitnessRequest>(
      cdf_batch_topic_ + "/witness_request", 2);
  witness_response_sub_ = nh_.subscribe(cdf_batch_topic_ + "/witness_response", 2,
      &LocalSparseSCPPlanner::witnessResponseCallback, this);

  query_trajectory_pub_ =
      nh_.advertise<trajectory_msgs::JointTrajectory>(
          query_trajectory_topic_, 1);
  candidate_trajectory_pub_ =
      nh_.advertise<trajectory_msgs::JointTrajectory>(
          candidate_trajectory_topic_, 1);
  observation_identity_pub_ = nh_.advertise<std_msgs::String>(
      "/care_planner/local_planner/observation_candidate_identity", 100);
  observation_dependency_pub_ = nh_.advertise<std_msgs::String>(
      "/care_planner/local_planner/observation_dependency", 8);
  candidate_replacement_grant_pub_ = nh_.advertise<std_msgs::String>(
      "/care_planner/local_planner/candidate_replacement_grant", 8);
  candidate_replacement_trigger_pub_ = nh_.advertise<std_msgs::String>(
      "/care_planner/local_planner/candidate_replacement_trigger", 8);
  candidate_replacement_sub_ = nh_.subscribe(
      "/care_planner/local_planner/candidate_replacement_request", 8,
      &LocalSparseSCPPlanner::candidateReplacementCallback, this);
  std::string final_outcome_topic;
  pnh_.param<std::string>("local_planner/verification_outcome_topic", final_outcome_topic,
                         "/care_planner/verification_outcome");
  final_verification_sub_ = nh_.subscribe(final_outcome_topic, 100,
      &LocalSparseSCPPlanner::finalVerificationCallback, this);
  admission_terminal_sub_ = nh_.subscribe(candidate_trajectory_topic_ + "/admission_terminal", 8,
      &LocalSparseSCPPlanner::admissionTerminalCallback, this);
  gcdf_rejection_feedback_sub_ = nh_.subscribe(
      candidate_trajectory_topic_ + "/gcdf_rejection_feedback", 4,
      &LocalSparseSCPPlanner::gcdfRejectionFeedbackCallback, this);
  pnh_.param("local_planner/rejection_snapshots_enabled", rejection_snapshots_enabled_, false);
  candidate_audit_context_pub_ = nh_.advertise<std_msgs::String>(
      candidate_trajectory_topic_ + "/audit_context", 4);
  summary_pub_ =
      nh_.advertise<std_msgs::String>(summary_topic_, 20, true);
  witness_diagnostic_pub_ = nh_.advertise<std_msgs::String>(summary_topic_ + "/witness", 100);
  task_infeasible_pub_ =
      nh_.advertise<std_msgs::Bool>(task_infeasible_topic_, 10, false);
  task_obstacle_blocked_pub_ =
      nh_.advertise<std_msgs::Bool>(task_obstacle_blocked_topic_, 10, false);
  task_uncertified_pub_ =
      nh_.advertise<std_msgs::Bool>(task_uncertified_topic_, 10, false);
  task_stall_pub_ = nh_.advertise<std_msgs::String>(
      "/care_planner/local_planner/task_stall", 10, true);
  force_vbc_bootstrap_pub_ =
      nh_.advertise<std_msgs::Bool>(force_vbc_bootstrap_topic_, 1, true);
  gcdf_recovery_trajectory_pub_ =
      nh_.advertise<trajectory_msgs::JointTrajectory>(
          gcdf_recovery_trajectory_topic_, 2, false);
  gcdf_recovery_event_pub_ =
      nh_.advertise<std_msgs::Float64MultiArray>(
          gcdf_recovery_event_topic_, 2, false);

  std_msgs::Bool bootstrap_init;
  bootstrap_init.data = false;
  force_vbc_bootstrap_pub_.publish(bootstrap_init);

  timer_ = nh_.createTimer(
      ros::Duration(1.0 / planner_poll_rate_),
      &LocalSparseSCPPlanner::timerCallback, this);

  startWorker();

  ROS_WARN_STREAM(
      "[LocalSparseSCPPlanner] C5.4 EVENT-TRIGGERED LOCAL PLANNER ENABLED: "
      "explicit q/u multiple shooting + Sparse PIQP + CDF slacks + SCP. "
      "This node does NOT publish actuator commands.");
  ROS_INFO_STREAM(
      "[LocalSparseSCPPlanner] K=" << num_intervals_
      << " dt=" << dt_
      << " SCP=" << max_scp_iterations_
      << " query=" << query_trajectory_topic_
      << " batch=" << cdf_batch_topic_
      << " candidate=" << candidate_trajectory_topic_);

  return true;
}

bool LocalSparseSCPPlanner::loadConfig() {
  if (!pnh_.getParam("joint_names", joint_names_) ||
      joint_names_.empty()) {
    ROS_ERROR("[LocalSparseSCPPlanner] missing joint_names");
    return false;
  }
  dof_ = static_cast<int>(joint_names_.size());

  pnh_.param<int>("local_planner/num_intervals",
                  num_intervals_, num_intervals_);
  pnh_.param<double>("local_planner/horizon_duration",
                     horizon_duration_, horizon_duration_);
  if (num_intervals_ <= 0 || horizon_duration_ <= 0.0) {
    ROS_ERROR("[LocalSparseSCPPlanner] invalid horizon");
    return false;
  }
  dt_ = horizon_duration_ / static_cast<double>(num_intervals_);

  pnh_.param<double>("local_planner/poll_rate",
                     planner_poll_rate_, planner_poll_rate_);
  pnh_.param<double>("local_planner/min_replan_interval",
                     min_replan_interval_s_, min_replan_interval_s_);
  pnh_.param<double>("local_planner/cdf_wait_timeout",
                     cdf_wait_timeout_s_, cdf_wait_timeout_s_);
  pnh_.param<double>("local_planner/cdf_stamp_tolerance",
                     cdf_stamp_tolerance_s_, cdf_stamp_tolerance_s_);

  pnh_.param<int>("local_planner/scp/max_iterations",
                  max_scp_iterations_, max_scp_iterations_);
  pnh_.param<double>("local_planner/scp/step_tolerance_inf",
                     scp_step_tolerance_inf_,
                     scp_step_tolerance_inf_);
  pnh_.param<double>("local_planner/scp/trust_region_initial",
                     trust_region_initial_, trust_region_initial_);
  pnh_.param<double>("local_planner/scp/trust_region_min",
                     trust_region_min_, trust_region_min_);
  pnh_.param<double>("local_planner/scp/trust_region_max",
                     trust_region_max_, trust_region_max_);
  pnh_.param<double>("local_planner/scp/trust_region_grow",
                     trust_region_grow_, trust_region_grow_);
  pnh_.param<double>("local_planner/scp/trust_region_shrink",
                     trust_region_shrink_, trust_region_shrink_);
  pnh_.param<double>("local_planner/scp/improvement_tolerance",
                     trust_region_improvement_tol_,
                     trust_region_improvement_tol_);

  pnh_.param<double>("local_planner/q_tracking_weight",
                     q_tracking_weight_, q_tracking_weight_);
  pnh_.param<double>("local_planner/terminal_q_tracking_weight",
                     terminal_q_tracking_weight_,
                     terminal_q_tracking_weight_);
  pnh_.param<double>("local_planner/u_tracking_weight",
                     u_tracking_weight_, u_tracking_weight_);
  pnh_.param<bool>("local_planner/u_reference_tracking_enabled",
                   u_reference_tracking_enabled_,
                   u_reference_tracking_enabled_);
  pnh_.param<double>("local_planner/u_smooth_weight",
                     u_smooth_weight_, u_smooth_weight_);
  pnh_.param<double>("local_planner/handoff_velocity_weight",
                     handoff_velocity_weight_, handoff_velocity_weight_);
  pnh_.param<bool>("local_planner/enforce_acceleration_constraints",
                   enforce_acceleration_constraints_,
                   enforce_acceleration_constraints_);
  pnh_.param<bool>("local_planner/repair_hold_initialization_enabled",
                   repair_hold_initialization_enabled_,
                   repair_hold_initialization_enabled_);
  pnh_.param<double>("local_planner/repair_task_tracking_scale",
                     repair_task_tracking_scale_,
                     repair_task_tracking_scale_);
  pnh_.param<double>("local_planner/repair_free_joint_task_tracking_scale",
                     repair_free_joint_task_tracking_scale_,
                     repair_free_joint_task_tracking_scale_);
  pnh_.param<int>("local_planner/probe_task_horizon_steps",
                  probe_task_horizon_steps_,
                  probe_task_horizon_steps_);
  pnh_.param<double>("local_planner/probe_velocity_cap", probe_velocity_cap_, probe_velocity_cap_);
  pnh_.param<double>("local_planner/visibility_waypoint_weight",
                     visibility_waypoint_weight_,
                     visibility_waypoint_weight_);
  pnh_.param<int>("local_planner/visibility_frontier_horizon_step",
                  visibility_frontier_horizon_step_,
                  visibility_frontier_horizon_step_);

  pnh_.param<double>("local_planner/cdf/safety_margin",
                     cdf_safety_margin_, cdf_safety_margin_);
  pnh_.param<bool>("local_planner/safe_frontier_recovery_enabled",
                   safe_frontier_recovery_enabled_, false);
  const char* guidance_mode=std::getenv("ROLLING_VISIBILITY_GUIDANCE_MODE");
  pnh_.param<bool>("local_planner/rolling_visibility_guidance_enabled",
      rolling_visibility_guidance_enabled_, guidance_mode && std::string(guidance_mode)!="off");
  pnh_.param<bool>("local_planner/candidate_replacement_enabled",
                   candidate_replacement_enabled_, false);
  const char* replacement_v2=std::getenv("CARE_REPLACEMENT_EXECUTION_V2");
  candidate_replacement_execution_v2_=replacement_v2 && std::string(replacement_v2)=="true";
  const char* task_first=std::getenv("CARE_TASK_FIRST_RESEARCH");
  task_first_research_=task_first && std::string(task_first)=="true";
  const char* executable_task=std::getenv("CARE_EXECUTABLE_TASK_RESEARCH");
  executable_task_research_=executable_task && std::string(executable_task)=="true";
  // These margins live in the learned CDF output units. Keep the old
  // parameter names as a compatibility fallback; they were incorrectly
  // suffixed/documented as meters in earlier C5.5 configs.
  if (!pnh_.getParam(
          "local_planner/cdf/visibility_obligation_cdf_margin",
          visibility_obligation_cdf_margin_)) {
    pnh_.param<double>(
        "local_planner/cdf/visibility_obligation_safety_margin",
        visibility_obligation_cdf_margin_, visibility_obligation_cdf_margin_);
  }
  visibility_obligation_cdf_margin_base_ =
      visibility_obligation_cdf_margin_;
  visibility_obligation_cdf_margin_effective_.store(
      visibility_obligation_cdf_margin_);
  // These are native CDF/model-unit feedback parameters. Keep the legacy
  // keys as read-only compatibility fallbacks; their old `_m` suffix never
  // caused a meter-to-CDF conversion and is no longer used in the config.
  if (!pnh_.getParam(
          "local_planner/cdf/vbc_feedback_cdf_margin_step",
          vbc_feedback_cdf_margin_step_)) {
    pnh_.param<double>(
        "local_planner/cdf/vbc_feedback_margin_step_m",
        vbc_feedback_cdf_margin_step_, vbc_feedback_cdf_margin_step_);
  }
  if (!pnh_.getParam(
          "local_planner/cdf/vbc_feedback_cdf_margin_max",
          vbc_feedback_cdf_margin_max_)) {
    pnh_.param<double>(
        "local_planner/cdf/vbc_feedback_margin_max_m",
        vbc_feedback_cdf_margin_max_, vbc_feedback_cdf_margin_max_);
  }
  pnh_.param<double>("local_planner/cdf/slack_linear_weight",
                     cdf_slack_linear_weight_,
                     cdf_slack_linear_weight_);
  pnh_.param<double>("local_planner/cdf/slack_quadratic_weight",
                     cdf_slack_quadratic_weight_,
                     cdf_slack_quadratic_weight_);
  pnh_.param<double>("local_planner/cdf/slack_upper_bound",
                     cdf_slack_upper_bound_,
                     cdf_slack_upper_bound_);
  pnh_.param<bool>("local_planner/cdf/per_constraint_slack",
                   cdf_per_constraint_slack_,
                   cdf_per_constraint_slack_);
  pnh_.param<bool>("local_planner/cdf/slack_use_upper_bound",
                   cdf_slack_use_upper_bound_,
                   cdf_slack_use_upper_bound_);
  pnh_.param<bool>("local_planner/cdf/slack_enabled",
                   cdf_slack_enabled_,
                   cdf_slack_enabled_);
  pnh_.param<bool>("local_planner/cdf/task_failure_slack_diagnostic_enabled",
                   task_failure_slack_diagnostic_enabled_,
                   task_failure_slack_diagnostic_enabled_);
  pnh_.param<double>("local_planner/normal_no_progress_target_gap_inf",
                     normal_no_progress_target_gap_inf_,
                     normal_no_progress_target_gap_inf_);
  pnh_.param<double>("local_planner/normal_active_no_progress_duration_s",
                     normal_active_no_progress_.min_duration_s,
                     normal_active_no_progress_.min_duration_s);
  pnh_.param<int>("local_planner/normal_active_no_progress_min_executions",
                  normal_active_no_progress_.min_execution_stamps,
                  normal_active_no_progress_.min_execution_stamps);
  if (!std::isfinite(normal_no_progress_target_gap_inf_) ||
      normal_no_progress_target_gap_inf_ <= 0.0 ||
      !std::isfinite(normal_active_no_progress_.min_duration_s) ||
      normal_active_no_progress_.min_duration_s <= 0.0 ||
      normal_active_no_progress_.min_execution_stamps < 2) return false;
  pnh_.param<double>("local_planner/cdf/task_failure_slack_diagnostic_weight",
                     task_failure_slack_diagnostic_weight_,
                     task_failure_slack_diagnostic_weight_);
  pnh_.param<bool>("local_planner/cdf/probe_feasibility_restoration_enabled",
                   probe_feasibility_restoration_enabled_,
                   probe_feasibility_restoration_enabled_);
  pnh_.param<int>("local_planner/cdf/probe_feasibility_restoration_max_attempts",
                  probe_feasibility_restoration_max_attempts_,
                  probe_feasibility_restoration_max_attempts_);
  pnh_.param<bool>("local_planner/cdf/adaptive_slack_penalty",
                   cdf_adaptive_slack_penalty_,
                   cdf_adaptive_slack_penalty_);
  pnh_.param<double>("local_planner/cdf/slack_penalty_multiplier",
                     cdf_slack_penalty_multiplier_,
                     cdf_slack_penalty_multiplier_);
  pnh_.param<double>("local_planner/cdf/slack_penalty_max",
                     cdf_slack_penalty_max_,
                     cdf_slack_penalty_max_);
  pnh_.param<double>("local_planner/cdf/slack_tolerance",
                     cdf_slack_tolerance_,
                     cdf_slack_tolerance_);
  pnh_.param<bool>("local_planner/cdf/safe_row_screening",
                   cdf_safe_row_screening_,
                   cdf_safe_row_screening_);
  pnh_.param<double>("local_planner/cdf/linearization_tolerance_inf",
                     cdf_linearization_tolerance_inf_,
                     cdf_linearization_tolerance_inf_);
  pnh_.param<int>("local_planner/cdf/constraint_horizon_steps",
                  cdf_constraint_horizon_steps_,
                  cdf_constraint_horizon_steps_);
  pnh_.param<int>("local_planner/cdf/visibility_obligation_horizon_steps",
                  visibility_obligation_cdf_horizon_steps_,
                  visibility_obligation_cdf_horizon_steps_);

  pnh_.param<int>("local_planner/piqp/max_iterations",
                  piqp_max_iterations_, piqp_max_iterations_);
  pnh_.param<double>("local_planner/piqp/eps_abs",
                     piqp_eps_abs_, piqp_eps_abs_);
  pnh_.param<double>("local_planner/piqp/eps_rel",
                     piqp_eps_rel_, piqp_eps_rel_);
  pnh_.param<bool>("local_planner/piqp/verbose",
                   piqp_verbose_, piqp_verbose_);

  pnh_.param<double>("mpc/joint_position_margin",
                     joint_position_margin_,
                     joint_position_margin_);

  pnh_.param<std::string>("local_planner/joint_states",
                          joint_state_topic_, joint_state_topic_);
  pnh_.param<std::string>("local_planner/reference_trajectory",
                          reference_topic_, reference_topic_);
  pnh_.param<std::string>("local_planner/waypoint_schedule_topic",
                          waypoint_schedule_topic_,
                          waypoint_schedule_topic_);
  pnh_.param<std::string>("local_planner/vbc_obligation_points_topic",
                          vbc_obligation_points_topic_,
                          vbc_obligation_points_topic_);
  pnh_.param<std::string>("local_planner/visibility_frontier_topic",
                          visibility_frontier_topic_,
                          visibility_frontier_topic_);
  pnh_.param<std::string>("local_planner/single_waypoint_active_topic",
                          single_waypoint_active_topic_,
                          single_waypoint_active_topic_);
  pnh_.param<std::string>("local_planner/single_waypoint_q_topic",
                          single_waypoint_q_topic_,
                          single_waypoint_q_topic_);
  pnh_.param<std::string>("local_planner/recovery_topic",
                          recovery_topic_, recovery_topic_);
  pnh_.param<std::string>("local_planner/probe_active_topic",
                          probe_active_topic_, probe_active_topic_);
  pnh_.param<std::string>("local_planner/replan_request_topic",
                          replan_request_topic_, replan_request_topic_);
  pnh_.param<std::string>("local_planner/smooth_replan_request_topic",
                          smooth_replan_request_topic_,
                          smooth_replan_request_topic_);
  pnh_.param<std::string>("local_planner/executed_command_topic",
                          executed_command_topic_,
                          executed_command_topic_);
  pnh_.param<std::string>("local_planner/execution_summary_topic",
                          execution_summary_topic_,
                          execution_summary_topic_);
  pnh_.param<std::string>("local_planner/cdf_batch_topic",
                          cdf_batch_topic_, cdf_batch_topic_);
  pnh_.param<std::string>("local_planner/query_trajectory_topic",
                          query_trajectory_topic_,
                          query_trajectory_topic_);
  pnh_.param<std::string>("local_planner/candidate_trajectory_topic",
                          candidate_trajectory_topic_,
                          candidate_trajectory_topic_);
  pnh_.param<std::string>("local_planner/summary_topic",
                          summary_topic_, summary_topic_);
  pnh_.param<std::string>("local_planner/task_infeasible_topic",
                          task_infeasible_topic_,
                          task_infeasible_topic_);
  pnh_.param<std::string>("local_planner/task_obstacle_blocked_topic",
                          task_obstacle_blocked_topic_,
                          task_obstacle_blocked_topic_);
  pnh_.param<std::string>("local_planner/task_uncertified_topic",
                          task_uncertified_topic_,
                          task_uncertified_topic_);
  pnh_.param<std::string>("local_planner/force_vbc_bootstrap_topic",
                          force_vbc_bootstrap_topic_,
                          force_vbc_bootstrap_topic_);
  pnh_.param<std::string>("local_planner/gcdf_recovery_trajectory_topic",
                          gcdf_recovery_trajectory_topic_,
                          gcdf_recovery_trajectory_topic_);
  pnh_.param<std::string>("local_planner/gcdf_recovery_event_topic",
                          gcdf_recovery_event_topic_,
                          gcdf_recovery_event_topic_);

  if (planner_poll_rate_ <= 0.0 ||
      max_scp_iterations_ < 1 ||
      trust_region_initial_ <= 0.0 ||
      trust_region_min_ <= 0.0 ||
      trust_region_max_ < trust_region_min_ ||
      cdf_slack_linear_weight_ < 0.0 ||
      cdf_slack_quadratic_weight_ < 0.0 ||
      (cdf_slack_use_upper_bound_ && cdf_slack_upper_bound_ <= 0.0) ||
      cdf_slack_penalty_multiplier_ < 1.0 ||
      cdf_slack_penalty_max_ < cdf_slack_linear_weight_ ||
      cdf_slack_tolerance_ < 0.0 ||
      !std::isfinite(cdf_safety_margin_) || cdf_safety_margin_ < 0.0 ||
      !std::isfinite(visibility_obligation_cdf_margin_) ||
      visibility_obligation_cdf_margin_ < 0.0 ||
      !std::isfinite(vbc_feedback_cdf_margin_step_) ||
      vbc_feedback_cdf_margin_step_ <= 0.0 ||
      !std::isfinite(vbc_feedback_cdf_margin_max_) ||
      vbc_feedback_cdf_margin_max_ < visibility_obligation_cdf_margin_base_ ||
      probe_feasibility_restoration_max_attempts_ < 0 ||
      probe_task_horizon_steps_ < 1 ||
      probe_task_horizon_steps_ > num_intervals_ ||
      !std::isfinite(probe_velocity_cap_) || probe_velocity_cap_ <= 0.0 ||
      visibility_frontier_horizon_step_ < 1 ||
      visibility_frontier_horizon_step_ > num_intervals_ ||
      cdf_constraint_horizon_steps_ < 1 ||
      cdf_constraint_horizon_steps_ > num_intervals_ ||
      visibility_obligation_cdf_horizon_steps_ < 1 ||
      visibility_obligation_cdf_horizon_steps_ > num_intervals_) {
    ROS_ERROR("[LocalSparseSCPPlanner] invalid local_planner parameters");
    return false;
  }

  return true;
}

bool LocalSparseSCPPlanner::loadJointLimits() {
  velocity_limits_ = Eigen::VectorXd::Zero(dof_);
  acceleration_limits_ = Eigen::VectorXd::Zero(dof_);
  q_min_ = Eigen::VectorXd::Zero(dof_);
  q_max_ = Eigen::VectorXd::Zero(dof_);

  for (int j = 0; j < dof_; ++j) {
    const std::string& name = joint_names_[static_cast<std::size_t>(j)];
    double v = 0.0, a = 0.0, lo = 0.0, hi = 0.0;
    if (!pnh_.getParam("mpc/joint_velocity_limits/" + name, v) ||
        !pnh_.getParam("mpc/joint_acceleration_limits/" + name, a) ||
        !pnh_.getParam("mpc/joint_position_limits/" + name + "/lower", lo) ||
        !pnh_.getParam("mpc/joint_position_limits/" + name + "/upper", hi)) {
      ROS_ERROR_STREAM(
          "[LocalSparseSCPPlanner] missing limits for " << name);
      return false;
    }
    if (!(v > 0.0) || !(a > 0.0) || !(lo < hi)) {
      ROS_ERROR_STREAM(
          "[LocalSparseSCPPlanner] invalid limits for " << name);
      return false;
    }
    velocity_limits_[j] = v;
    acceleration_limits_[j] = a;
    q_min_[j] = lo;
    q_max_[j] = hi;
  }
  return true;
}

void LocalSparseSCPPlanner::jointStateCallback(
    const sensor_msgs::JointStateConstPtr& msg) {
  if (!msg) return;
  std::unique_lock<std::mutex> lock(mutex_);
  std::string evidence_event;
  latest_joint_state_ = *msg;
  latest_joint_state_received_ = ros::Time::now();
  has_joint_state_ = true;
  Eigen::VectorXd measured;
  if (extractMeasuredQ(*msg, measured)) {
    if(fixed_start_clearance_hold_.observe(msg->header.stamp.toSec(),ros::Time::now().toSec(),measured))
      requestPlanLocked("fixed_start_changed_state_requery");
    const std::string evidence_key=std::to_string(mode_epoch_)+":"+safe_frontier_recovery_.key;
    const double age=(ros::Time::now()-msg->header.stamp).toSec();
    if(!state_evidence_retry_pending_.empty()) {
      if(state_evidence_retry_pending_!=evidence_key || !repair_mode_ || probe_mode_ ||
          state_residual_retry_epoch_!=mode_epoch_ || state_residual_retry_token_!=latest_observation_token_) {
        if(state_residual_retry_raw_)safe_frontier_recovery_.pending.erase(state_residual_retry_raw_);
        state_residual_retry_raw_=0;
        state_evidence_retry_pending_.clear();state_evidence_retry_wait_.reset();
        evidence_event="state_evidence_context_changed";
      } else if(ros::WallTime::now()<state_evidence_retry_deadline_ && !msg->header.stamp.isZero() &&
          age>=0. && age<=.05 && msg->velocity.size()==msg->name.size() &&
          (ros::Time::now()-latest_executed_command_received_).toSec()>=0. &&
          (ros::Time::now()-latest_executed_command_received_).toSec()<=.05) {
        Eigen::VectorXd velocity(dof_);
        for(int j=0;j<dof_;++j) {
          const auto it=std::find(msg->name.begin(),msg->name.end(),joint_names_[j]);
          velocity[j]=msg->velocity[std::distance(msg->name.begin(),it)];
        }
        if(state_evidence_retry_wait_.observe(msg->header.stamp.toSec(),measured,velocity,latest_executed_command_)) {
          if(state_residual_retry_raw_)safe_frontier_recovery_.pending.erase(state_residual_retry_raw_);
          state_residual_retry_raw_=0;
          state_evidence_retry_pending_.clear();state_evidence_retry_wait_.reset();
          requestPlanLocked("state_evidence_fresh_state_retry");
          evidence_event=plan_request_reason_;
        }
      } else state_evidence_retry_wait_.reset();
    }
    vbc_failure_direction_memory_.observeMeasured(measured,msg->header.stamp.toSec());
    const bool vbc_was_blocked = final_vbc_no_progress_.progress.blocked();
    if (final_vbc_no_progress_.progress.observe(measured) && vbc_was_blocked && repair_mode_)
      requestPlanLocked("measured_progress_after_final_vbc_stall");
    const bool repair_was_blocked = repair_no_progress_.blocked();
    const bool failed_candidate_was_held = candidate_replacement_enabled_ &&
        repair_mode_ && repair_candidate_failure_.blocks(
            repairTargetKeyLocked(), latest_observation_token_, mode_epoch_,
            repair_no_progress_.active.progress);
    const bool repair_progress = repair_no_progress_.observe(measured);
    if (repair_progress && repair_was_blocked && repair_mode_)
      requestPlanLocked("measured_progress_after_repair_qp_stall");
    if (repair_progress && failed_candidate_was_held)
      requestPlanLocked("measured_progress_after_failed_candidate_hold");
    const bool was_blocked = task_no_progress_.blocked();
    if (!normal_task_stall_pending_ && task_no_progress_.observe(measured) && was_blocked) {
      std_msgs::String status; status.data = "status=reset reason=measured_progress";
      task_stall_pub_.publish(status);
      requestPlanLocked("measured_progress_after_qp_stall");
    }
  } else vbc_failure_direction_memory_.reset();
  lock.unlock();
  if(!evidence_event.empty())publishSummary(evidence_event);
}

void LocalSparseSCPPlanner::taskReferenceStatusCallback(
    const std_msgs::StringConstPtr& msg) {
  if (!msg) return;
  unsigned long long id = 0, stamp = 0;
  if (!parseUnsignedToken(msg->data, "request_id", &id) ||
      !parseUnsignedToken(msg->data, "request_ros_ns", &stamp)) return;
  std::unique_lock<std::mutex> lock(mutex_);
  const bool exhausted = msg->data.find("status=exhausted") != std::string::npos;
  if (!task_reference_receipt_.status(id, stamp, exhausted)) return;
  // Different ROS topics may be delivered out of order: do not invalidate a
  // fresh reference just because its earlier 'pending' status arrives later.
  if (!task_reference_receipt_.pending && !task_reference_receipt_.failed &&
      latest_reference_.header.stamp.toNSec() >= stamp && !latest_reference_.points.empty()) {
    const bool reference_was_unavailable = !has_reference_;
    has_reference_ = true;
    if (reference_was_unavailable ||
        (normal_reference_refresh_pending_ && !probe_mode_ && !repair_mode_)) {
      normal_reference_refresh_pending_ = false;
      last_normal_completed_execution_stamp_ns_ = latest_execution_stamp_ns_;
      requestPlanLocked("fresh_reference_status_after_reference");
    }
    return;
  }
  normal_reference_refresh_pending_ = !exhausted;
  has_reference_ = false;
  plan_requested_ = false;
  ++mode_epoch_;  // in-flight candidates based on the old reference are stale
  vbc_failure_direction_memory_.reset();
  plan_request_reason_ = exhausted ? "normal_task_reference_retry_exhausted"
                                   : "waiting_fresh_normal_task_reference";
  ROS_WARN_STREAM("[LocalSparseSCPPlanner] " << plan_request_reason_ << " " << msg->data);
  const std::string event=plan_request_reason_;
  lock.unlock();
  publishSummary(event);
}

void LocalSparseSCPPlanner::referenceCallback(
    const trajectory_msgs::JointTrajectoryConstPtr& msg) {
  if (!msg || msg->points.empty()) return;
  std::lock_guard<std::mutex> lock(mutex_);
  if (!task_reference_receipt_.reference(msg->header.stamp.toNSec())) {
    if (msg->header.stamp.toNSec() >= task_reference_receipt_.min_stamp) {
      // Buffer only; an exhausted request cannot authorize planning. A newer
      // request's status may arrive after its reference on the other topic.
      latest_reference_ = *msg;
      latest_reference_received_ = ros::Time::now();
    }
    return;
  }
  latest_reference_ = *msg;
  latest_reference_received_ = ros::Time::now();
  has_reference_ = true;
  task_no_progress_.reset();
  normal_active_no_progress_.reset();
  normal_unknown_no_progress_.reset();
  resetNormalTaskProgressLocked("reference_changed");
  vbc_failure_direction_memory_.reset();
  std_msgs::String reset; reset.data = "status=reset reason=fresh_reference";
  task_stall_pub_.publish(reset);

  if (normal_reference_refresh_pending_) {
    // C5.33: the first fresh /task_trajectory after PROBE->NORMAL is the
    // measured-state task rebase. Baseline the just-completed PROBE execution
    // stamp so a delayed duplicate tracker summary cannot be interpreted as a
    // NORMAL completion event.
    normal_reference_refresh_pending_ = false;
    last_normal_completed_execution_stamp_ns_ =
        latest_execution_stamp_ns_;
    ++normal_reference_refresh_count_;
    requestPlanLocked("fresh_normal_task_reference");
  } else {
    requestPlanLocked("new_nominal_reference");
  }
}

void LocalSparseSCPPlanner::waypointScheduleCallback(
    const std_msgs::Float64MultiArrayConstPtr& msg) {
  if (!msg) return;
  const bool has_joint_masks =
      !msg->layout.dim.empty() &&
      msg->layout.dim.front().label == "care_visibility_schedule_v2_qmask";
  const std::size_t record_size = has_joint_masks ? 16u : 9u;
  if (msg->data.size() % record_size != 0) return;

  std::vector<DeadlineWaypoint> incoming;
  incoming.reserve(msg->data.size() / record_size);
  for (std::size_t r = 0; r < msg->data.size() / record_size; ++r) {
    const std::size_t off = record_size * r;
    DeadlineWaypoint wp;
    wp.id = static_cast<long long>(std::llround(msg->data[off]));
    wp.deadline_abs_s = msg->data[off + 1];
    wp.q = Eigen::VectorXd::Zero(dof_);
    wp.joint_mask = Eigen::VectorXd::Ones(dof_);
    if (!std::isfinite(wp.deadline_abs_s) || wp.deadline_abs_s <= 0.0)
      return;
    for (int j = 0; j < dof_; ++j) {
      wp.q[j] = msg->data[off + 2 + static_cast<std::size_t>(j)];
      if (has_joint_masks) {
        wp.joint_mask[j] =
            msg->data[off + 9 + static_cast<std::size_t>(j)];
      }
    }
    if (!finiteVector(wp.q) || !finiteVector(wp.joint_mask) ||
        (wp.joint_mask.array() < 0.0).any() ||
        (wp.joint_mask.array() > 1.0).any()) return;
    incoming.push_back(wp);
  }

  std::lock_guard<std::mutex> lock(mutex_);
  bool changed = incoming.size() != latest_schedule_.size();
  if (!changed) {
    for (std::size_t i = 0; i < incoming.size(); ++i) {
      if (incoming[i].id != latest_schedule_[i].id ||
          std::fabs(incoming[i].deadline_abs_s -
                    latest_schedule_[i].deadline_abs_s) > 1e-5 ||
          (incoming[i].q - latest_schedule_[i].q)
                  .lpNorm<Eigen::Infinity>() > 1e-5 ||
          incoming[i].joint_mask.size() != latest_schedule_[i].joint_mask.size() ||
          (incoming[i].joint_mask - latest_schedule_[i].joint_mask)
                  .lpNorm<Eigen::Infinity>() > 1e-5) {
        changed = true;
        break;
      }
    }
  }
  latest_schedule_ = incoming;
  if (changed) selectRepairTargetLocked();
  if (changed && repair_mode_ && visibility_waypoint_weight_ > 0.0)
    requestPlanLocked("visibility_schedule_changed");
}

void LocalSparseSCPPlanner::vbcObligationPointsCallback(
    const std_msgs::Float64MultiArrayConstPtr& msg) {
  // Message layout:
  // [publication_seq, active_obligation_id, point_count,
  //  x0, y0, z0, ...].  An id of -1 and count 0 clears the active target.
  if (!msg || msg->data.size() < 3) return;
  const double seq_value = msg->data[0];
  const double id_value = msg->data[1];
  const double count_value = msg->data[2];
  if (!std::isfinite(seq_value) || !std::isfinite(id_value) ||
      !std::isfinite(count_value)) return;
  const auto seq = static_cast<unsigned long long>(std::llround(seq_value));
  const auto obligation_id = static_cast<long long>(std::llround(id_value));
  const auto count = static_cast<std::size_t>(std::llround(count_value));
  if (seq == 0 || std::fabs(seq_value - static_cast<double>(seq)) > 1e-6 ||
      std::fabs(id_value - static_cast<double>(obligation_id)) > 1e-6 ||
      count_value < 0.0 ||
      std::fabs(count_value - static_cast<double>(count)) > 1e-6 ||
      msg->data.size() != 3 + 3 * count ||
      (count == 0 && obligation_id != -1) ||
      (count > 0 && obligation_id < 0)) {
    return;
  }

  std::vector<Eigen::Vector3d> points;
  points.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    Eigen::Vector3d point(
        msg->data[3 + 3 * i],
        msg->data[3 + 3 * i + 1],
        msg->data[3 + 3 * i + 2]);
    if (!point.allFinite()) return;
    if (std::none_of(points.begin(), points.end(),
                     [&](const Eigen::Vector3d& old) {
                       return (old - point).lpNorm<Eigen::Infinity>() <= 1e-5;
                     })) {
      points.push_back(point);
    }
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (seq <= latest_vbc_obligation_points_seq_) return;
  const bool changed =
      obligation_id != latest_vbc_obligation_id_ ||
      points.size() != latest_vbc_obligation_points_.size() ||
      std::any_of(points.begin(), points.end(), [&](const Eigen::Vector3d& p) {
        return std::none_of(
            latest_vbc_obligation_points_.begin(),
            latest_vbc_obligation_points_.end(),
            [&](const Eigen::Vector3d& old) {
              return (old - p).lpNorm<Eigen::Infinity>() <= 1e-5;
            });
      });
  latest_vbc_obligation_points_seq_ = seq;
  latest_vbc_obligation_id_ = obligation_id;
  latest_vbc_obligation_points_ = std::move(points);
  vbc_failure_direction_memory_.observeContext(vbcFailureContextLocked());
  if (!changed) return;
  normal_unknown_no_progress_.reset();
  resetNormalTaskProgressLocked("obligation_points_changed");

  ROS_WARN_STREAM("[LocalSparseSCPPlanner] vbc_obligation_points_update seq="
      << seq << " obligation_id=" << obligation_id
      << " points=" << latest_vbc_obligation_points_.size());
  // This is the explicit urgent handoff: the blocker-aware scheduler only
  // publishes the stack-top target, so a changed point set is allowed to
  // invalidate an in-flight *uncommitted* local solve. It never cancels a
  // trajectory already owned by the tracker.
  if (repair_mode_ && !probe_mode_ && visibility_waypoint_weight_ > 0.0) {
    ++plan_sequence_;
    plan_running_ = false;
    waiting_for_cdf_ = false;
    pending_batch_.reset();
    requestPlanLocked("vbc_obligation_points_changed");
  }
}

void LocalSparseSCPPlanner::visibilityFrontierCallback(
    const std_msgs::Float64MultiArrayConstPtr& msg) {
  // Message layout:
  // Legacy: [active, frontier_weight_scale, qvis_weight_scale, q_frontier(7)]
  // Masked: legacy fields followed by joint_mask(7).
  if (!msg || (msg->data.size() != 10 && msg->data.size() != 17)) return;
  const bool has_joint_mask = msg->data.size() == 17;

  const bool active = msg->data[0] > 0.5;
  const double frontier_scale = msg->data[1];
  const double qvis_scale = msg->data[2];
  if (!std::isfinite(frontier_scale) ||
      !std::isfinite(qvis_scale) ||
      frontier_scale < 0.0 ||
      qvis_scale < 0.0) {
    return;
  }

  Eigen::VectorXd q = Eigen::VectorXd::Zero(dof_);
  Eigen::VectorXd joint_mask = Eigen::VectorXd::Ones(dof_);
  for (int j = 0; j < dof_; ++j) {
    q[j] = msg->data[3 + static_cast<std::size_t>(j)];
    if (has_joint_mask)
      joint_mask[j] = msg->data[10 + static_cast<std::size_t>(j)];
  }
  if (!finiteVector(joint_mask) ||
      (joint_mask.array() < 0.0).any() ||
      (joint_mask.array() > 1.0).any()) return;
  if (active && !finiteVector(q)) return;
  if (!active) {
    q.setZero();
    joint_mask.setOnes();
  }

  std::lock_guard<std::mutex> lock(mutex_);
  const bool changed =
      active != latest_frontier_.active ||
      std::fabs(frontier_scale -
                latest_frontier_.frontier_weight_scale) > 1e-5 ||
      std::fabs(qvis_scale -
                latest_frontier_.qvis_weight_scale) > 1e-5 ||
      latest_frontier_.q.size() != dof_ ||
      (q - latest_frontier_.q).lpNorm<Eigen::Infinity>() > 1e-4 ||
      latest_frontier_.joint_mask.size() != dof_ ||
      (joint_mask - latest_frontier_.joint_mask)
              .lpNorm<Eigen::Infinity>() > 1e-5;

  if (rolling_visibility_guidance_enabled_ && active &&
      (msg->layout.dim.empty() || msg->layout.dim.front().label!=latest_observation_token_)) return;

  latest_frontier_.active = active;
  latest_frontier_.frontier_weight_scale = frontier_scale;
  latest_frontier_.qvis_weight_scale = qvis_scale;
  latest_frontier_.q = q;
  latest_frontier_.joint_mask = joint_mask;

  if (changed && repair_mode_ && visibility_waypoint_weight_ > 0.0 &&
      (!rolling_visibility_guidance_enabled_ || (!plan_running_ && !waiting_for_cdf_))) {
    requestPlanLocked(
        active
            ? "visibility_frontier_changed"
            : "visibility_frontier_cleared");
  }
}

void LocalSparseSCPPlanner::visibilityGuidanceCallback(
    const egocentric_arm_planner::VisibilityGuidanceConstPtr& msg) {
  if (!msg || !rolling_visibility_guidance_enabled_) return;
  VisibilityGuidanceState g;
  g.token=msg->observation_token; g.region=msg->region_token;
  g.sequence=msg->sequence; g.sensor=msg->sensor_id; g.stamp=msg->measured_stamp.toSec();
  g.value=msg->value; g.drive=msg->drive_enabled;
  g.geometry_valid=msg->geometry_valid;g.geometry_signature=msg->geometry_signature;
  g.conservative_fov_margin=msg->conservative_fov_margin;
  g.q_eval=Eigen::Map<const Eigen::VectorXd>(msg->q_eval.data(),7);
  g.target=Eigen::Map<const Eigen::VectorXd>(msg->q_target.data(),7);
  g.mask=Eigen::Map<const Eigen::VectorXd>(msg->joint_mask.data(),7);
  g.gradient=Eigen::Map<const Eigen::VectorXd>(msg->gradient.data(),7);
  std::lock_guard<std::mutex> lock(mutex_);
  g.epoch=mode_epoch_;
  Eigen::VectorXd measured;
  if (!has_joint_state_ || !extractMeasuredQ(latest_joint_state_, measured) ||
      !g.compatible(latest_observation_token_, latest_observation_region_,
        latest_observation_sensor_, latest_single_waypoint_joint_mask_, mode_epoch_,
        measured, ros::Time::now().toSec()) ||
      (latest_visibility_guidance_.epoch==g.epoch && latest_visibility_guidance_.sequence>=g.sequence)) return;
  const bool changed=latest_visibility_guidance_.token!=g.token ||
      latest_visibility_guidance_.target.size()!=7 ||
      (latest_visibility_guidance_.target-g.target).lpNorm<Eigen::Infinity>()>1e-4;
  latest_visibility_guidance_=g;
  if(g.geometry_valid && g.drive)
    safe_frontier_recovery_.noteGeometryFeedback(g.token,g.geometry_signature,g.stamp,
        ros::Time::now().toSec(),g.conservative_fov_margin,g.q_eval);
  // Buffer while solving. New soft guidance cannot cancel its in-flight CDF query.
  if (changed && repair_mode_ && !plan_running_ && !waiting_for_cdf_)
    requestPlanLocked("visibility_guidance_ready");
}

void LocalSparseSCPPlanner::singleWaypointActiveCallback(
    const std_msgs::BoolConstPtr& msg) {
  if (!msg) return;

  std::lock_guard<std::mutex> lock(mutex_);
  if (latest_single_waypoint_active_ == msg->data) return;

  latest_single_waypoint_active_ = msg->data;
  normal_unknown_no_progress_.reset();
  resetNormalTaskProgressLocked("waypoint_active_changed");
  selectRepairTargetLocked();
  if (repair_mode_ && visibility_waypoint_weight_ > 0.0) {
    requestPlanLocked(
        msg->data
            ? "single_visibility_waypoint_activated"
            : "single_visibility_waypoint_deactivated");
  }
}

void LocalSparseSCPPlanner::singleWaypointQCallback(
    const std_msgs::Float64MultiArrayConstPtr& msg) {
  if (!msg ||
      msg->data.size() != static_cast<std::size_t>(dof_)) {
    return;
  }

  Eigen::VectorXd q(dof_);
  for (int j = 0; j < dof_; ++j) {
    q[j] = msg->data[static_cast<std::size_t>(j)];
  }
  if (!finiteVector(q)) return;
  const Eigen::VectorXd joint_mask = jointMaskFromLayout(msg->layout, dof_);

  std::lock_guard<std::mutex> lock(mutex_);
  const bool changed =
      !has_single_waypoint_q_ ||
      latest_single_waypoint_q_.size() != dof_ ||
      (q - latest_single_waypoint_q_)
              .lpNorm<Eigen::Infinity>() > 1e-5 ||
      latest_single_waypoint_joint_mask_.size() != dof_ ||
      (joint_mask - latest_single_waypoint_joint_mask_)
              .lpNorm<Eigen::Infinity>() > 1e-5;

  const std::string previous_token = latest_observation_token_;
  latest_single_waypoint_q_ = q;
  latest_single_waypoint_joint_mask_ = joint_mask;
  latest_observation_region_.clear(); latest_observation_sensor_=-1;
  for (const auto& dim : msg->layout.dim) {
    if (dim.label.compare(0,15,"care_region_v1_")==0) latest_observation_region_=dim.label;
    if (dim.label.compare(0,15,"care_sensor_v1_")==0 && dim.label.size()==16 &&
        dim.label.back()>='0' && dim.label.back()<='7')
      latest_observation_sensor_=dim.label.back()-'0';
  }
  latest_observation_token_ = "none";
  if (!msg->layout.dim.empty() &&
      msg->layout.dim.front().label.find("care_obs_v1_") == 0 &&
      msg->layout.dim.front().label.find_first_not_of(
          "abcdefghijklmnopqrstuvwxyz0123456789_") == std::string::npos)
    latest_observation_token_ = msg->layout.dim.front().label;
  has_single_waypoint_q_ = true;
  vbc_failure_direction_memory_.observeContext(vbcFailureContextLocked());
  if (changed || previous_token != latest_observation_token_) {
    normal_unknown_no_progress_.reset();
    resetNormalTaskProgressLocked("waypoint_target_changed");
    selectRepairTargetLocked();
  }

  if ((changed || previous_token != latest_observation_token_) &&
      repair_mode_ && visibility_waypoint_weight_ > 0.0) {
    requestPlanLocked("single_visibility_waypoint_q_changed");
  }
}

void LocalSparseSCPPlanner::recoveryCallback(
    const std_msgs::BoolConstPtr& msg) {
  if (!msg) return;
  bool clear_bootstrap = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (repair_mode_ != msg->data) {
      const bool goal_handoff=repair_goal_stopping_enabled_ && repair_mode_ && !msg->data &&
          terminal_retention_.armed && retainTerminalLocked();
      repair_mode_ = msg->data;
      ++mode_epoch_;
      if(goal_handoff) {
        // Same audited motion and goal; only its scheduling owner changes.
        // This does not retire any observation obligation.
        terminal_retention_.armed_context=terminalContextLocked();
        repair_goal_handoff_execution_=terminal_retention_.armed;
      }
      vbc_failure_direction_memory_.reset();
      normal_active_no_progress_.reset();
      normal_unknown_no_progress_.reset();
      resetNormalTaskProgressLocked("repair_mode_changed");
      if (repair_mode_) {
        task_no_progress_.reset();
        std_msgs::String reset; reset.data = "status=reset reason=real_repair_transition";
        task_stall_pub_.publish(reset);
        // C5.32: the execution that was already active/completed before REPAIR
        // is not a REPAIR completion event. Baseline the stamp here so a
        // repeated latched complete=1 from the previous mode cannot trigger a
        // spurious REPAIR replan.
        last_repair_completed_execution_stamp_ns_ =
            latest_execution_stamp_ns_;
        // PROBE->REPAIR owns its next plan through the REPAIR trigger, so any
        // pending wait for a fresh NORMAL task reference is cancelled.
        normal_reference_refresh_pending_ = false;
      }
      if(!goal_handoff)requestPlanLocked(
          repair_mode_ ? "enter_repair" : "leave_repair");
      clear_bootstrap = !repair_mode_;
    }
  }
  if (clear_bootstrap) {
    std_msgs::Bool bootstrap_msg;
    bootstrap_msg.data = false;
    force_vbc_bootstrap_pub_.publish(bootstrap_msg);
  }
}

void LocalSparseSCPPlanner::probeActiveCallback(
    const std_msgs::BoolConstPtr& msg) {
  if (!msg) return;
  std::lock_guard<std::mutex> lock(mutex_);
  if (probe_mode_ == msg->data) return;

  const bool was_probe = probe_mode_;
  probe_mode_ = msg->data;
  ++mode_epoch_;
  vbc_failure_direction_memory_.reset();
  normal_active_no_progress_.reset();
  normal_unknown_no_progress_.reset();
  resetNormalTaskProgressLocked("probe_mode_changed");

  if (probe_mode_) {
    probe_reference_request_id_ = task_reference_receipt_.id;
    normal_reference_refresh_pending_ = false;
    requestPlanLocked("enter_probe_normal");
    return;
  }

  if (!was_probe) return;

  if (repair_mode_) {
    // PROBE->REPAIR is already replanned by recoveryCallback().
    normal_reference_refresh_pending_ = false;
    return;
  }

  // C5.33 PROBE->NORMAL: do NOT immediately plan against the stale one-shot
  // task trajectory. Wait for the regime-manager pulse -> repeated EE goal ->
  // fresh measured-state /task_trajectory callback.
  if (has_reference_ && task_reference_receipt_.id > probe_reference_request_id_ &&
      !task_reference_receipt_.pending && !task_reference_receipt_.failed) {
    normal_reference_refresh_pending_ = false;
    requestPlanLocked("fresh_reference_before_normal_transition");
    return;
  }
  normal_reference_refresh_pending_ = !task_reference_receipt_.failed;
  plan_requested_ = false;
  plan_request_reason_ = "waiting_fresh_normal_task_reference";
}

void LocalSparseSCPPlanner::replanRequestCallback(
    const std_msgs::BoolConstPtr& msg) {
  if (!msg || !msg->data) return;
  std::lock_guard<std::mutex> lock(mutex_);
  if (normal_reference_refresh_pending_ && !repair_mode_ && !probe_mode_) {
    ++normal_refresh_blocked_replan_count_;
    return;
  }
  requestPlanLocked("external_replan_request");
  ROS_INFO_STREAM("[REPLAN_REQUEST_DISPOSITION] requested=" << plan_requested_
      << " running=" << plan_running_ << " reason=" << plan_request_reason_
      << " plan_seq=" << plan_sequence_ << " mode_epoch=" << mode_epoch_
      << " recovery_attempts=" << safe_frontier_recovery_.attempts
      << " recovery_rejected=" << safe_frontier_recovery_.rejected
      << " recovery_generation=" << safe_frontier_recovery_.generation);
}

void LocalSparseSCPPlanner::qpRetryRequestCallback(
    const std_msgs::StringConstPtr& msg) {
  if (!msg || msg->data.empty()) return;
  std::lock_guard<std::mutex> lock(mutex_);
  // A transport retransmission must never buy another solve. The ticket is
  // minted by one terminal failed QP, not by the requesting node or ROS time.
  std::string status = "stale";
  if (msg->data.compare(0, 6, "ready_") == 0) {
    // Benchmark-only round trip through the actual request/ack callbacks.
    // Readiness probes carry no planning or execution authority.
    status = "ready";
  } else if (msg->data == qp_retry_consumed_token_) {
    status = "duplicate";
  } else if (msg->data == qp_retry_token_ &&
             qp_retry_plan_sequence_ == plan_sequence_ &&
             qp_retry_mode_epoch_ == mode_epoch_ && !repair_mode_ &&
             !plan_running_ && !plan_requested_ &&
             !normal_reference_refresh_pending_) {
    qp_retry_consumed_token_ = msg->data;
    requestPlanLocked("acknowledged_qp_retry");
    status = plan_requested_ ? "accepted" : "blocked";
  }
  std_msgs::String ack;
  ack.data = "token=" + msg->data + " status=" + status;
  qp_retry_ack_pub_.publish(ack);
  ROS_INFO_STREAM("[LocalSparseSCPPlanner] qp_retry " << ack.data);
}

void LocalSparseSCPPlanner::smoothReplanRequestCallback(
    const std_msgs::BoolConstPtr& msg) {
  if (!msg || !msg->data) return;
  std::lock_guard<std::mutex> lock(mutex_);

  if (probe_mode_ || normal_reference_refresh_pending_) {
    ++smooth_handoff_replan_suppressed_probe_count_;
    return;
  }
  if (plan_running_ || plan_requested_) {
    ++smooth_handoff_replan_suppressed_busy_count_;
    return;
  }

  ++smooth_handoff_replan_count_;
  requestPlanLocked(
      repair_mode_ ? "smooth_handoff_repair" : "smooth_handoff_normal");
}

std::string LocalSparseSCPPlanner::terminalContextLocked() const {
  if(!terminal_retention_enabled_ || !terminal_goal_valid_ || probe_mode_ ||
      (repair_mode_ && !repair_goal_stopping_enabled_) ||
      !has_reference_ || (!repair_mode_ && normal_reference_refresh_pending_))return {};
  return std::to_string(terminal_goal_revision_)+":"+std::to_string(mode_epoch_)+":"+
      (repair_mode_ ? "repair_goal_stop" :
       std::to_string(latest_reference_.header.stamp.toNSec()));
}

bool LocalSparseSCPPlanner::terminalPoseInsideLocked(const Eigen::VectorXd& q) const {
  Eigen::Isometry3d pose;
  if(!terminal_goal_valid_ || !primitive_observation_guard_.framePose(q,terminal_ee_frame_,&pose))return false;
  const double angle=Eigen::AngleAxisd(terminal_goal_.linear().transpose()*pose.linear()).angle();
  return std::isfinite(angle) && (pose.translation()-terminal_goal_.translation()).norm()<=.02 && angle<=.2;
}

void LocalSparseSCPPlanner::terminalGoalCallback(const geometry_msgs::PoseStampedConstPtr& msg) {
  if(!msg)return;
  std::lock_guard<std::mutex> lock(mutex_);
  const auto& p=msg->pose.position;const auto& r=msg->pose.orientation;
  Eigen::Quaterniond orientation(r.w,r.x,r.y,r.z);
  Eigen::Vector3d position(p.x,p.y,p.z);
  if(msg->header.frame_id!=terminal_base_frame_ || !position.allFinite() ||
      !orientation.coeffs().allFinite() || orientation.norm()<1e-12) {
    terminal_goal_valid_=false;terminal_retention_.invalidate("invalid_goal");
    resetNormalTaskProgressLocked("invalid_goal");return;
  }
  orientation.normalize();Eigen::Isometry3d goal=Eigen::Isometry3d::Identity();
  goal.translation()=position;goal.linear()=orientation.toRotationMatrix();
  if(terminal_goal_valid_ && (goal.matrix()-terminal_goal_.matrix()).cwiseAbs().maxCoeff()<=1e-9)return;
  terminal_goal_=goal;terminal_goal_valid_=true;++terminal_goal_revision_;
  known_ground_hold_.bind(terminal_goal_revision_);
  ground_retry_goal_=terminal_goal_revision_;ground_resource_retries_=0;ground_resource_blocked_=false;ground_admission_.clear();
  geometry_resource_blocked_reason_="ground_proof_unavailable_hold";
  resetNormalTaskProgressLocked("goal_changed");
  terminal_retention_.invalidate("goal_changed");
}

void LocalSparseSCPPlanner::armTerminalLocked() {
  // A REPAIR goal-stop lease begins only after its tracker owns the actual
  // committed motion. Earlier audit/outcome callbacks may precede that sample.
  if(repair_goal_stopping_enabled_ && repair_mode_ &&
      (!terminal_retention_.newest_commit || terminal_tracker_token_!=terminal_retention_.newest_commit ||
       terminal_tracker_aborted_ || (terminal_tracker_source_!="active_trajectory" &&
                                     terminal_tracker_source_!="trajectory_complete_hold")))return;
  const auto before=terminal_retention_.armed;
  terminal_retention_.tryArm(terminalContextLocked(),[&](const trajectory_msgs::JointTrajectory& t) {
    std::vector<int> mapping;Eigen::VectorXd q;
    return t.header.frame_id==terminal_base_frame_ && buildTrajectoryMapping(t,mapping) &&
        sampleReferencePosition(t,mapping,t.points.back().time_from_start.toSec(),q) && terminalPoseInsideLocked(q);
  });
  if(terminal_retention_.armed && terminal_retention_.armed!=before) {
    terminal_tracker_completed_=false;
    if(repair_mode_ && repair_goal_stopping_enabled_) {
      // A newer in-flight proposal must not replace this certified goal stop.
      // Observation obligations and recovery budgets are retained.
      ++plan_sequence_;plan_running_=false;waiting_for_cdf_=false;
      pending_batch_.reset();plan_requested_=false;
    }
    ROS_INFO_STREAM("[NORMAL_TERMINAL] armed execution_stamp_ns="<<terminal_retention_.armed
        <<" context="<<terminal_retention_.armed_context);
  }
}

void LocalSparseSCPPlanner::terminalAuditCallback(const trajectory_msgs::JointTrajectoryConstPtr& msg) {
  if(!msg)return;
  std::lock_guard<std::mutex> lock(mutex_);
  terminal_retention_.audit(*msg);armTerminalLocked();
}
void LocalSparseSCPPlanner::terminalCommitCallback(const trajectory_msgs::JointTrajectoryConstPtr& msg) {
  if(!msg)return;
  std::lock_guard<std::mutex> lock(mutex_);
  terminal_retention_.commit(*msg);armTerminalLocked();
}

bool LocalSparseSCPPlanner::retainTerminalLocked() {
  auto& t=terminal_retention_;
  if(!t.armed)return false;
  std::string invalid;
  const auto now=ros::Time::now();
  const auto fresh=[&](const ros::Time& stamp) {const double age=(now-stamp).toSec();return !stamp.isZero() && age>=0. && age<=.05;};
  if(t.armed_context!=terminalContextLocked())invalid="context_changed";
  else if(vbc_execution_hold_ || gcdf_execution_hold_ || execution_response_monitor_.blocked)invalid="safety_or_response_hold";
  else if(!has_joint_state_ || !fresh(latest_joint_state_.header.stamp) || !fresh(latest_joint_state_received_) ||
          !fresh(terminal_tracker_received_) || !fresh(latest_executed_command_received_))invalid="stale_state_or_tracker";
  else if(terminal_tracker_aborted_)invalid="tracker_aborted";
  else if(!((terminal_tracker_token_==t.armed && (terminal_tracker_source_=="active_trajectory" ||
             terminal_tracker_source_=="trajectory_complete_hold")) ||
            (!terminal_tracker_token_ && terminal_tracker_completed_ && terminal_tracker_source_=="last_reference_hold")))invalid="tracker_identity_or_source";
  else if(terminal_tracker_completed_) {
    Eigen::VectorXd measured;
    if(!extractMeasuredQ(latest_joint_state_,measured) || !terminalPoseInsideLocked(measured))invalid="completed_outside_goal";
  }
  if(invalid.empty())return true;
  const auto token=t.armed;t.invalidate(invalid);
  ROS_INFO_STREAM("[NORMAL_TERMINAL] revoked execution_stamp_ns="<<token<<" reason="<<invalid);
  return false;
}

void LocalSparseSCPPlanner::executedCommandCallback(
    const std_msgs::Float64MultiArrayConstPtr& msg) {
  if (!msg || msg->data.size() != static_cast<std::size_t>(dof_)) return;
  Eigen::VectorXd u(dof_);
  for (int j = 0; j < dof_; ++j)
    u[j] = msg->data[static_cast<std::size_t>(j)];
  if (!finiteVector(u)) return;

  std::lock_guard<std::mutex> lock(mutex_);
  latest_executed_command_ = u;
  latest_executed_command_received_ = ros::Time::now();
}

void LocalSparseSCPPlanner::desiredCommandCallback(
    const std_msgs::Float64MultiArrayConstPtr& msg) {
  if (!msg || msg->data.size()!=static_cast<std::size_t>(dof_)) return;
  Eigen::VectorXd u(dof_);
  for (int j=0;j<dof_;++j) u[j]=msg->data[j];
  if (!u.allFinite()) return;
  std::lock_guard<std::mutex> lock(mutex_);
  latest_desired_command_=u; latest_desired_command_received_=ros::Time::now();
}

void LocalSparseSCPPlanner::limiterSummaryCallback(const std_msgs::StringConstPtr& msg) {
  if (!msg) return;
  std::istringstream stream(msg->data); std::string word;
  std::map<std::string,std::string> fields;
  while (stream>>word) {
    const auto eq=word.find('=');
    if (eq!=std::string::npos && !fields.emplace(word.substr(0,eq),word.substr(eq+1)).second) return;
  }
  if ((fields["vbc_hold"]!="0" && fields["vbc_hold"]!="1") ||
      (fields["gcdf_hold"]!="0" && fields["gcdf_hold"]!="1")) return;
  const auto& status=fields["status"];
  const bool waiting=status=="waiting_input" || status=="waiting_input_limited" ||
      status=="stale_ramp_to_zero" || status=="stale_ramp_to_zero_limited";
  const bool recognized=waiting || status=="tracking" || status=="tracking_limited" ||
      status=="vbc_hold_ramp_to_zero" || status=="vbc_hold_ramp_to_zero_limited" ||
      status=="gcdf_hold_ramp_to_zero" || status=="gcdf_hold_ramp_to_zero_limited";
  if (!recognized) return;
  std::lock_guard<std::mutex> lock(mutex_);
  latest_limiter_intentional_hold_=waiting || fields["vbc_hold"]=="1" || fields["gcdf_hold"]=="1";
  latest_limiter_summary_received_=ros::Time::now();
}

void LocalSparseSCPPlanner::vbcExecutionHoldCallback(const std_msgs::BoolConstPtr& msg) {
  if (!msg) return;
  std::lock_guard<std::mutex> lock(mutex_); vbc_execution_hold_=msg->data;
  if(msg->data)terminal_retention_.invalidate("vbc_hold");
}
void LocalSparseSCPPlanner::gcdfExecutionHoldCallback(const std_msgs::BoolConstPtr& msg) {
  if (!msg) return;
  std::lock_guard<std::mutex> lock(mutex_); gcdf_execution_hold_=msg->data;
  if(msg->data)terminal_retention_.invalidate("gcdf_hold");
}

void LocalSparseSCPPlanner::executionSummaryCallback(
    const std_msgs::StringConstPtr& msg) {
  if (!msg) return;
  std::map<std::string, std::string> tracker_fields;
  std::istringstream tracker_input(msg->data);
  std::string tracker_word;
  while (tracker_input >> tracker_word) {
    const auto eq = tracker_word.find('=');
    if (eq != std::string::npos &&
        !tracker_fields.emplace(tracker_word.substr(0,eq),tracker_word.substr(eq+1)).second) return;
  }
  double tracker_phase = -1.;
  try {
    const auto& phase = tracker_fields.at("phase_s");
    std::size_t consumed = 0;
    tracker_phase = std::stod(phase, &consumed);
    if (consumed != phase.size()) tracker_phase = -1.;
  } catch (...) {}
  bool safe_recovery_changed = false;
  bool execution_response_changed = false;
  bool execution_response_blocked = false;
  bool intentional_execution_hold = false;
  std::string safe_recovery_event;

  const bool complete = tracker_fields["complete"] == "1";

  unsigned long long execution_stamp_ns = 0;
  try {
    const auto& stamp = tracker_fields.at("execution_stamp_ns");
    if (!stamp.empty() && std::all_of(stamp.begin(), stamp.end(),
        [](char c) { return c >= '0' && c <= '9'; }))
      execution_stamp_ns = std::stoull(stamp);
  } catch (...) {}
  const bool has_execution_stamp = execution_stamp_ns > 0;

  bool request_repair_replan = false;
  bool request_normal_replan = false;
  bool normal_measured_stall = false;
  bool normal_active_stall = false;
  bool normal_unknown_stall = false;
  bool normal_task_stall = false;
  double normal_task_position_error = -1., normal_task_orientation_error = -1.;
  unsigned long long normal_unknown_query = 0;
  std::string normal_unknown_context;
  unsigned long long normal_stall_query_stamp_ns = 0;
  int normal_stall_failures = 0;
  double normal_stall_target_gap = 0.0;
  double normal_active_stall_duration_s = 0.0;
  int normal_active_stall_executions = 0;
  bool duplicate_repair_completion = false;
  bool duplicate_normal_completion = false;
  unsigned long long repair_completion_replan_count_snapshot = 0;
  unsigned long long normal_completion_replan_count_snapshot = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if(complete && has_execution_stamp && tracker_fields["execution_aborted"]=="0")
      known_ground_hold_.completed(execution_stamp_ns);
    if(terminal_retention_enabled_ && execution_stamp_ns>=terminal_retention_.newest_commit && has_execution_stamp) {
      terminal_tracker_token_=execution_stamp_ns;terminal_tracker_received_=ros::Time::now();
      terminal_tracker_source_=tracker_fields["source"];
      terminal_tracker_aborted_=tracker_fields["execution_aborted"]!="0";
      if(terminal_tracker_aborted_ || (terminal_tracker_source_!="active_trajectory" &&
          terminal_tracker_source_!="trajectory_complete_hold"))
        terminal_retention_.invalidate("tracker_abort_or_hold");
      if(complete && execution_stamp_ns==terminal_retention_.armed) {
        const auto committed=terminal_retention_.commits.find(execution_stamp_ns);
        terminal_tracker_completed_=committed!=terminal_retention_.commits.end() &&
            std::isfinite(tracker_phase) && tracker_phase+1e-9>=committed->second.points.back().time_from_start.toSec();
      }
    } else if(terminal_retention_enabled_ && !has_execution_stamp && terminal_tracker_completed_) {
      terminal_tracker_token_=0;terminal_tracker_received_=ros::Time::now();
      terminal_tracker_source_=tracker_fields["source"];terminal_tracker_aborted_=tracker_fields["execution_aborted"]!="0";
      if(terminal_tracker_source_!="last_reference_hold" || terminal_tracker_aborted_)
        terminal_retention_.invalidate("tracker_hold_changed");
    }
    if(repair_goal_stopping_enabled_ && repair_mode_)armTerminalLocked();
    if (has_execution_stamp) {
      latest_execution_stamp_ns_ = execution_stamp_ns;
    }
    const auto now = ros::Time::now();
    const double measured_age = (now-latest_joint_state_.header.stamp).toSec();
    const double execution_age = has_execution_stamp
        ? now.toSec()-static_cast<double>(execution_stamp_ns)*1e-9 : -1.;
    const double desired_age=(now-latest_desired_command_received_).toSec();
    const double limiter_age=(now-latest_limiter_summary_received_).toSec();
    const bool intentional_hold=vbc_execution_hold_ || gcdf_execution_hold_ ||
        (!latest_limiter_summary_received_.isZero() && limiter_age>=0. &&
         limiter_age<=.2 && latest_limiter_intentional_hold_);
    intentional_execution_hold=intentional_hold;
    Eigen::VectorXd response_q;
    bool response_state_fresh=false;
    if (has_joint_state_ && measured_age>=0. && measured_age<=.2 &&
        (now-latest_joint_state_received_).toSec()>=0. &&
        (now-latest_joint_state_received_).toSec()<=.2 &&
        extractMeasuredQ(latest_joint_state_,response_q)) {
      response_state_fresh=true;
      const bool eligible=has_execution_stamp && execution_age>=0. && execution_age<=2. &&
          tracker_fields["execution_aborted"]=="0" &&
          (tracker_fields["active"]=="1" || complete) && !intentional_hold &&
          !latest_desired_command_received_.isZero() && desired_age>=0. && desired_age<=.1;
      execution_response_changed=execution_response_monitor_.observe(
          now.toSec(),response_q,latest_desired_command_,eligible,intentional_hold);
    }
    execution_response_blocked=execution_response_monitor_.blocked;
    if (execution_response_blocked) {
      // Preserve candidate/direction budgets and identities. Discard only
      // progress windows accumulated while actuation was unresponsive.
      safe_frontier_recovery_.clearWindow(); safe_frontier_recovery_.clearObservationWindow();
      normal_active_no_progress_.reset(); normal_unknown_no_progress_.progress.reset();
      normal_unknown_task_progress_.resetWindow("execution_response_blocked");
      if (execution_response_changed && execution_response_blocked) {
        ++plan_sequence_; plan_running_=false; waiting_for_cdf_=false;
        pending_batch_.reset(); plan_requested_=false;
        plan_request_reason_="execution_command_no_response_hold";
      }
    } else if (intentional_hold) {
      // A short safety wait pauses elapsed progress time; it must not erase
      // the already-observed commits every time certification briefly holds.
      safe_frontier_recovery_.pauseProgress(now.toSec());
      normal_active_no_progress_.reset(); normal_unknown_no_progress_.progress.reset();
      normal_unknown_task_progress_.resetWindow("intentional_safety_hold");
    } else safe_frontier_recovery_.resumeProgress(now.toSec());
    if (execution_response_changed && !execution_response_blocked)
      requestPlanLocked(intentional_hold ? "execution_response_safety_wait" : "execution_response_restored");
    if (has_execution_stamp && tracker_fields["execution_aborted"] == "1")
      safe_frontier_recovery_.abort(execution_stamp_ns);
    // A failed execution is not a completed task. Once the tracker confirms
    // its measured stop, request one fresh plan for this execution identity.
    // Do not resurrect the old command or consume a lifetime retry allowance.
    if (!execution_response_blocked && has_reference_ && terminal_goal_valid_ &&
        has_execution_stamp && tracker_fields["execution_aborted"] == "1" &&
        tracker_fields["execution_state"] == "STOPPED_HOLD" &&
        tracker_fields["stop_execution_stamp_ns"] == std::to_string(execution_stamp_ns) &&
        execution_stamp_ns > last_stopped_replan_stamp_ns_ &&
        execution_stamp_ns >= terminal_retention_.newest_commit &&
        !plan_running_ && !waiting_for_cdf_ && !plan_requested_ &&
        response_state_fresh && response_q.size() == dof_ && !terminalPoseInsideLocked(response_q)) {
      last_stopped_replan_stamp_ns_ = execution_stamp_ns;
      ++stopped_execution_replan_count_;
      requestPlanLocked("tracking_stop_replan_measured");
      ROS_INFO_STREAM("[TASK_RECOVERY] event=stopped_execution execution_stamp_ns="
          << execution_stamp_ns << " count=" << stopped_execution_replan_count_
          << " measured_stamp_ns=" << latest_joint_state_.header.stamp.toNSec()
          << " request=" << plan_request_reason_);
    }
    if (!execution_response_blocked && !intentional_hold && safe_frontier_recovery_enabled_ && repair_mode_ && !probe_mode_ &&
        repair_observation_phase_ && latest_frontier_.active && has_execution_stamp &&
        tracker_fields["execution_aborted"] == "0" &&
        (tracker_fields["active"] == "1" || complete) &&
        std::isfinite(tracker_phase) && (tracker_phase >= .05 || complete) &&
        has_joint_state_ && measured_age >= 0. && measured_age <= .2 &&
        (now-latest_joint_state_received_).toSec() >= 0. &&
        (now-latest_joint_state_received_).toSec() <= .2 &&
        execution_age >= .05 && execution_age <= 2.) {
      Eigen::VectorXd measured;
      const int previous_attempts = safe_frontier_recovery_.attempts;
      const auto certified = safe_frontier_recovery_.certified.find(execution_stamp_ns);
      const bool monitoring_observation = certified!=safe_frontier_recovery_.certified.end() &&
          (!safe_frontier_recovery_.active || certified->second.following_frontier ||
            safe_frontier_recovery_.monitor_detour_observation ||
            safe_frontier_recovery_.persistent_observation_monitor);
      if (extractMeasuredQ(latest_joint_state_, measured))
        safe_recovery_changed = safe_frontier_recovery_.observe(
            execution_stamp_ns, now.toSec(), measured);
      if (safe_recovery_changed) {
        ++plan_sequence_; plan_running_ = false; waiting_for_cdf_ = false;
        pending_batch_.reset();
        const auto& recovery = safe_frontier_recovery_;
        // Six certified executions with no observation progress can reach
        // q_vis while a live point remains unseen. Repeating this same pose
        // cannot test a new sensor direction. Use the existing reservation
        // ledger, only after fresh, token-matched map feedback and measured
        // arrival; ordinary detours and motion toward q_vis keep their path.
        const double feedback_age = now.toSec()-recovery.feedback_query_time;
        const double query_age = (now-last_query_ros_time_).toSec();
        const bool reached_unseen = candidate_replacement_enabled_ &&
            !candidate_replacement_pending_ && !candidate_replacement_trigger_pending_ &&
            monitoring_observation &&
            recovery.observation_replacement_due &&
            recovery.last_stall_reason == "observation_no_progress" &&
            latest_observation_feedback_unseen_ &&
            recovery.observation_token == latest_observation_token_ &&
            recovery.observation_goal.size()==dof_ && measured.size()==dof_ &&
             (recovery.monitor_detour_observation || recovery.persistent_observation_monitor ||
             ((measured-recovery.observation_goal).cwiseProduct(recovery.observation_mask)).norm()<=.001) &&
            recovery.feedback_query_time>=0. && feedback_age>=0. && feedback_age<=.5 &&
            !current_query_stamp_.isZero() && query_age>=0. && query_age<=1.;
        if (reached_unseen && requestQpReplacementLocked("observation_no_progress")) {
          // At first activation direction one is the failed original q_vis;
          // the reservation charges direction two. An already-active stall
          // has advanced to the next direction, so do not charge it twice.
          candidate_replacement_trigger_direction_advanced_ = previous_attempts>0;
          plan_request_reason_ = "repair_observation_candidate_replacement_wait";
        } else {
          requestPlanLocked("repair_safe_stall_frontier_retry");
        }
        safe_recovery_event = plan_request_reason_;
      }
    }

    const bool rising = complete && !latest_execution_complete_;
    latest_execution_complete_ = complete;

    if (!execution_response_blocked && complete && repair_mode_) {
      if (has_execution_stamp) {
        // C5.32: completion identity is the immutable committed execution
        // stamp, not a lossy Boolean edge. Faster C5.31 planning can replace
        // trajectories quickly enough that an intermediate complete=0 tracker
        // sample is dropped.
        if (execution_stamp_ns !=
            last_repair_completed_execution_stamp_ns_) {
          last_repair_completed_execution_stamp_ns_ =
              execution_stamp_ns;
          ++repair_completion_replan_count_;
          repair_completion_replan_count_snapshot =
              repair_completion_replan_count_;
          request_repair_replan = true;
        } else {
          ++repair_completion_duplicate_count_;
          duplicate_repair_completion = true;
        }
      } else if (rising) {
        ++repair_completion_replan_count_;
        repair_completion_replan_count_snapshot =
            repair_completion_replan_count_;
        request_repair_replan = true;
      }
    } else if (!execution_response_blocked && complete && !probe_mode_ &&
               !normal_reference_refresh_pending_) {
      // C5.33: NORMAL is also a receding-horizon execution mode. Once one
      // certified local trajectory finishes, request the next plan from the
      // latest measured state/reference. Use execution_stamp_ns exactly once,
      // mirroring C5.32, so dropped complete=0 samples cannot deadlock NORMAL.
      if (has_execution_stamp) {
        if (execution_stamp_ns !=
            last_normal_completed_execution_stamp_ns_) {
          last_normal_completed_execution_stamp_ns_ =
              execution_stamp_ns;
          ++normal_completion_replan_count_;
          normal_completion_replan_count_snapshot =
              normal_completion_replan_count_;
          request_normal_replan = true;
        } else {
          ++normal_completion_duplicate_count_;
          duplicate_normal_completion = true;
        }
      } else if (rising) {
        ++normal_completion_replan_count_;
        normal_completion_replan_count_snapshot =
            normal_completion_replan_count_;
        request_normal_replan = true;
      }
    }

    // A receding-horizon tracker may replace every safe hold before complete=1.
    // Count only distinct active, non-aborted certified execution stamps and
    // fresh measured joints. A long stationary episode still needs the same
    // task-stall exact-VBC/obligation route before entering REPAIR.
    if (!execution_response_blocked && !intentional_hold && !repair_mode_ && !probe_mode_ && !normal_reference_refresh_pending_ &&
        has_reference_ && tracker_fields["active"] == "1" && !complete &&
        tracker_fields["execution_aborted"] == "0" && has_execution_stamp &&
        std::isfinite(tracker_phase) && tracker_phase >= .05 &&
        has_joint_state_ && measured_age >= 0.0 && measured_age <= .2 &&
        (now-latest_joint_state_received_).toSec() >= 0.0 &&
        (now-latest_joint_state_received_).toSec() <= .2 &&
        execution_age >= .05 && execution_age <= 2.0 &&
        plan_q_ref_.rows() == dof_ && plan_q_ref_.cols() > 0) {
      Eigen::VectorXd measured;
      if (extractMeasuredQ(latest_joint_state_, measured)) {
        normal_stall_target_gap =
            (plan_q_ref_.col(plan_q_ref_.cols()-1)-measured)
                .lpNorm<Eigen::Infinity>();
        normal_unknown_context = normalUnknownContextLocked();
        normal_unknown_stall = normal_unknown_no_progress_.observe(
            now.toSec(), execution_stamp_ns, measured, normal_unknown_context,
            normal_stall_target_gap > normal_no_progress_target_gap_inf_);
        const bool general_stall = normal_active_no_progress_.observe(
                now.toSec(), execution_stamp_ns, measured,
                normal_stall_target_gap > normal_no_progress_target_gap_inf_);
        const auto task_context = normalTaskProgressContextLocked();
        Eigen::Isometry3d task_pose;
        const bool task_pose_valid = (normal_unknown_task_progress_.qualified(
            now.toSec(),execution_stamp_ns,task_context) ||
            (normal_task_stall_pending_ && normal_task_stall_context_==task_context)) &&
            primitive_observation_guard_.framePose(measured,terminal_ee_frame_,&task_pose);
        if (task_pose_valid) {
          normal_task_position_error=(task_pose.translation()-terminal_goal_.translation()).norm();
          normal_task_orientation_error=Eigen::AngleAxisd(
              terminal_goal_.linear().transpose()*task_pose.linear()).angle();
        }
        if (normal_task_stall_pending_ && task_pose_valid &&
            std::max(normal_task_position_error/normal_unknown_task_progress_.position_tolerance_m,
                     normal_task_orientation_error/normal_unknown_task_progress_.orientation_tolerance_rad)
                <=normal_task_stall_credited_error_-normal_unknown_task_progress_.progress_epsilon) {
          normal_task_stall_pending_=false;normal_task_stall_context_.clear();
          task_no_progress_.reset();
          normal_unknown_task_progress_.resetWindow("task_progress_after_stall");
          std_msgs::String reset;reset.data="status=reset reason=task_space_progress";
          task_stall_pub_.publish(reset);
          requestPlanLocked("task_space_progress_after_stall");
        }
        normal_task_stall=normal_unknown_task_progress_.observe(now.toSec(),execution_stamp_ns,
            normal_task_position_error,normal_task_orientation_error,task_context,true);
        if ((normal_task_stall || normal_unknown_stall || general_stall) &&
            !task_no_progress_.blocked() && !current_query_stamp_.isZero()) {
          normal_active_stall = normal_measured_stall = true;
          normal_active_stall_duration_s =
              normal_active_no_progress_.duration(now.toSec());
          normal_active_stall_executions =
              normal_active_no_progress_.execution_stamps;
          if (normal_unknown_stall) {
            normal_active_stall_duration_s = normal_unknown_no_progress_.progress.duration(now.toSec());
            normal_active_stall_executions = normal_unknown_no_progress_.progress.execution_stamps;
            normal_unknown_query = normal_unknown_no_progress_.last.query;
          }
          if (normal_task_stall) {
            normal_active_stall_duration_s=normal_unknown_task_progress_.duration(now.toSec());
            normal_active_stall_executions=normal_unknown_task_progress_.execution_stamps;
            normal_unknown_query=normal_unknown_task_progress_.last.query;
            normal_unknown_context=task_context;
            normal_task_stall_pending_=true;
            normal_task_stall_context_=task_context;
            normal_task_stall_credited_error_=normal_unknown_task_progress_.credited_error;
          }
          normal_stall_query_stamp_ns = current_query_stamp_.toNSec();
          task_no_progress_.block(measured);
          normal_stall_failures = task_no_progress_.failures;
          ++normal_measured_stall_count_;
          ++plan_sequence_;
          plan_running_ = false;
          waiting_for_cdf_ = false;
          pending_batch_.reset();
          plan_requested_ = false;
        }
      }
    } else if (repair_mode_ || probe_mode_ || !has_reference_ || normal_reference_refresh_pending_ ||
               tracker_fields["active"] != "1" || tracker_fields["execution_aborted"] != "0" ||
               measured_age < 0. || measured_age > .2 ||
               (now-latest_joint_state_received_).toSec() > .2) {
      normal_unknown_no_progress_.progress.reset();
      normal_unknown_task_progress_.resetWindow("tracker_or_state_ineligible");
    }

    logNormalTaskProgressLocked();
    if (request_repair_replan) {
      requestPlanLocked("repair_trajectory_complete");
    } else if (request_normal_replan) {
      // Count completed, distinct certified executions using fresh measured
      // joints. A successful hard-QP hold can still make no task progress.
      // The target-gap guard avoids diagnosing a legitimate goal hold.
      if (!intentional_hold && !normal_task_stall_pending_ && has_execution_stamp && has_joint_state_ && measured_age >= 0.0 &&
          measured_age <= 0.2 &&
          (now-latest_joint_state_received_).toSec() >= 0.0 &&
          (now-latest_joint_state_received_).toSec() <= 0.2 &&
          plan_q_ref_.rows() == dof_ && plan_q_ref_.cols() > 0) {
        Eigen::VectorXd measured;
        if (extractMeasuredQ(latest_joint_state_, measured)) {
          normal_stall_target_gap =
              (plan_q_ref_.col(plan_q_ref_.cols()-1)-measured)
                  .lpNorm<Eigen::Infinity>();
          if (normal_stall_target_gap > normal_no_progress_target_gap_inf_ &&
              task_no_progress_.fail(measured)) {
            normal_measured_stall = true;
            normal_stall_query_stamp_ns = current_query_stamp_.toNSec();
            normal_stall_failures = task_no_progress_.failures;
            ++normal_measured_stall_count_;
          } else if (normal_stall_target_gap <=
                     normal_no_progress_target_gap_inf_) {
            task_no_progress_.reset();
          }
        }
      }
      requestPlanLocked("normal_trajectory_complete");
    }
  }

  if (execution_response_changed) publishSummary(execution_response_blocked
      ? "execution_command_no_response_hold" : intentional_execution_hold
      ? "execution_response_safety_wait" : "execution_response_restored");
  if (safe_recovery_changed) publishSummary(safe_recovery_event);
  if (normal_measured_stall && normal_stall_query_stamp_ns > 0) {
    std_msgs::String msg;
    std::ostringstream s;
    s << "status=stalled classification="
      << (normal_task_stall ? "NORMAL_UNKNOWN_TASK_NO_PROGRESS"
          : normal_unknown_stall ? "NORMAL_UNKNOWN_CONSTRAINED_NO_MEASURED_PROGRESS"
          : normal_active_stall ? "NORMAL_ACTIVE_CERTIFIED_NO_MEASURED_PROGRESS"
                              : "NORMAL_CERTIFIED_NO_MEASURED_PROGRESS")
      << " failures=" << normal_stall_failures
      << " query_stamp_ns=" << normal_stall_query_stamp_ns
      << " execution_stamp_ns=" << execution_stamp_ns
      << " target_gap_inf=" << normal_stall_target_gap
      << " snapshot=normal_hard_qp";
    if (normal_active_stall)
      s << " active_execution_stamps=" << normal_active_stall_executions
        << " duration_s=" << normal_active_stall_duration_s;
    if (normal_unknown_stall || normal_task_stall)
      s << " unknown_evidence_query_stamp_ns=" << normal_unknown_query
        << " unknown_context=" << normal_unknown_context;
    if (normal_task_stall)
      s << " task_position_error_m=" << normal_task_position_error
        << " task_orientation_error_rad=" << normal_task_orientation_error;
    msg.data = s.str();
    task_stall_pub_.publish(msg);
    publishSummary("normal_certified_no_measured_progress");
    ROS_WARN_STREAM("[LocalSparseSCPPlanner] " << msg.data);
  }
  if (request_repair_replan) {
    ROS_INFO_STREAM(
        "[LocalSparseSCPPlanner] REPAIR execution completion -> replan"
        << " stamp_ns="
        << (has_execution_stamp ? execution_stamp_ns : 0ULL)
        << " count=" << repair_completion_replan_count_snapshot);
  } else if (request_normal_replan) {
    ROS_INFO_STREAM(
        "[LocalSparseSCPPlanner] NORMAL execution completion -> replan"
        << " stamp_ns="
        << (has_execution_stamp ? execution_stamp_ns : 0ULL)
        << " count=" << normal_completion_replan_count_snapshot);
  } else if (duplicate_repair_completion) {
    ROS_DEBUG_STREAM_THROTTLE(
        1.0,
        "[LocalSparseSCPPlanner] duplicate REPAIR completion ignored stamp_ns="
            << execution_stamp_ns);
  } else if (duplicate_normal_completion) {
    ROS_DEBUG_STREAM_THROTTLE(
        1.0,
        "[LocalSparseSCPPlanner] duplicate NORMAL completion ignored stamp_ns="
            << execution_stamp_ns);
  }

  // PROBE_NORMAL replanning remains owned by the regime manager and is
  // released only after tracker complete=1 for the exact committed token.
}

void LocalSparseSCPPlanner::cdfConstraintBatchCallback(
    const care_collision_cdf::CollisionCDFConstraintBatchConstPtr& msg) {
  acceptCdfBatch(msg, nullptr);
}

void LocalSparseSCPPlanner::witnessResponseCallback(
    const care_collision_cdf::CollisionCDFWitnessResponseConstPtr& msg) {
  if (!msg) return;
  acceptCdfBatch(boost::make_shared<care_collision_cdf::CollisionCDFConstraintBatch>(msg->batch), msg.get());
}

void LocalSparseSCPPlanner::acceptCdfBatch(
    const care_collision_cdf::CollisionCDFConstraintBatchConstPtr& msg,
    const care_collision_cdf::CollisionCDFWitnessResponse* response) {
  if (!msg) return;

  bool accepted = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // A legacy batch cannot stand in for an explicit witness acknowledgement.
    // Empty requests keep the legacy batch path (NORMAL/PROBE included).
    if (witness_response_required_ != static_cast<bool>(response)) return;
    ++cdf_batch_received_;

    if (!plan_running_ || !waiting_for_cdf_) return;

    // Stamp is a request identity, not a noisy measurement timestamp. In
    // simulation adjacent requests may differ by only 1 ns; a tolerance
    // would allow a delayed previous (possibly empty) batch into a new solve.
    if (msg->header.stamp != current_query_stamp_) {
      ++cdf_stamp_miss_;
      return;
    }

    const double roundtrip_ms =
        current_query_wall_.toSec() > 0.0
            ? (ros::WallTime::now() - current_query_wall_).toSec() * 1000.0
            : 0.0;
    last_cdf_roundtrip_ms_ = roundtrip_ms;
    plan_cdf_roundtrip_sum_ms_ += roundtrip_ms;
    plan_cdf_roundtrip_max_ms_ =
        std::max(plan_cdf_roundtrip_max_ms_, roundtrip_ms);
    ++plan_cdf_roundtrip_count_;

    bool response_error = false;
    if (response) {
      const auto& a = response->request;
      const auto& b = current_witness_request_;
      if (a.header.stamp != b.header.stamp || a.header.frame_id != b.header.frame_id ||
          a.plan_sequence != b.plan_sequence || a.mode_epoch != b.mode_epoch ||
          a.target_revision != b.target_revision || a.progress_epoch != b.progress_epoch ||
          a.dof != b.dof || a.original_timestep != b.original_timestep ||
          a.point_flat != b.point_flat || a.q_flat != b.q_flat) {
        publishWitnessDiagnosticLocked(-1, -1, Eigen::Vector3d::Zero(), "response_identity_mismatch");
        return;
      }
      response_error = response->status.size() != b.original_timestep.size();
      if (response_error)
        publishWitnessDiagnosticLocked(-1, -1, Eigen::Vector3d::Zero(), "response_dimension_error");
      // Only a fresh map disposition in this exact query may retire a point.
      // A missing pair in a plain batch is never a free-space certificate.
      if (!response_error && plan_mode_epoch_ == mode_epoch_ &&
          repair_no_progress_.current(plan_repair_ticket_)) {
        // Validate the entire disposition before retiring anything. Mixed
        // free/unknown/occupied replies for one point cannot clear its guard.
        response_error = std::any_of(response->status.begin(), response->status.end(),
            [](const std::string& s) { return s != "resolved_free" &&
                s != "evaluated_unknown" && s != "evaluated_occupied"; });
        for (std::size_t i = 0; i < response->status.size(); ++i) {
          const Eigen::Vector3d p(b.point_flat[3*i], b.point_flat[3*i+1], b.point_flat[3*i+2]);
          const auto& status = response->status[i];
          publishWitnessDiagnosticLocked(static_cast<int>(i), b.original_timestep[i], p, "upstream_"+status);
          const bool point_free = !response_error && status == "resolved_free" &&
              resolvedFreePoint(response->status, b.point_flat, p);
          if (point_free) {
            initial_shell_recovery_.observedFree(p);
            fixed_start_clearance_hold_.resolvedFree(p);
            repair_unknown_witnesses_.erase(std::remove_if(repair_unknown_witnesses_.begin(),
                repair_unknown_witnesses_.end(), [&](const RepairWitness& w) {
                  if ((w.point-p).lpNorm<Eigen::Infinity>() > 1e-5) return false;
                  // resolvedFreePoint has already required unanimous fresh
                  // disposition for every requested knot of this point.
                  // Keep the existing recovery gate for elevated exact-VBC
                  // guards; ordinary point witnesses retire as one identity.
                  if (w.vbc_feedback) return safe_frontier_recovery_enabled_;
                  return w.safety_margin <= cdf_safety_margin_ + 1e-9;
                }), repair_unknown_witnesses_.end());
          } else if (status != "resolved_free" && status != "evaluated_unknown" && status != "evaluated_occupied") {
            response_error = true;
          }
        }
      }
    }
    view_preview_batch_=msg; view_preview_batch_received_=ros::Time::now();
    pending_batch_ = msg;
    if (plan_repair_mode_ && repair_mode_ && plan_mode_epoch_ == mode_epoch_ &&
        repair_no_progress_.current(plan_repair_ticket_))
      updateRepairWitnessesLocked(*msg);
    if (response_error) repair_witness_requires_reobserve_ = true;
    waiting_for_cdf_ = false;
    accepted = true;
  }

  if (accepted) {
    // Latched diagnostic: if PIQP later spends too long in setup/solve, the
    // benchmark still tells us that timestamp matching and ROS transport worked.
    publishSummary("cdf_batch_accepted");
    worker_cv_.notify_one();
  }
}

void LocalSparseSCPPlanner::publishWitnessDiagnosticLocked(
    int index, int timestep, const Eigen::Vector3d& point,
    const std::string& reason, int pair, double q_error) {
  std::ostringstream s;
  s << std::setprecision(17) << "C5_4_WITNESS event=repair_witness_check"
    << " query_stamp_ns=" << current_query_stamp_.toNSec()
    << " plan_seq=" << plan_sequence_ << " mode_epoch=" << plan_mode_epoch_
    << " target_revision=" << plan_repair_ticket_.revision
    << " progress_epoch=" << plan_repair_ticket_.progress
    << " phase=" << (plan_repair_observation_phase_ ? "observation" : "retreat")
    << " witness_index=" << index << " timestep=" << timestep
    << " point=[" << point[0] << ',' << point[1] << ',' << point[2] << ']'
    << " reason=" << reason << " batch_pair=" << pair << " q_error_inf=" << q_error;
  if (timestep >= 1 && timestep < plan_q_bar_.cols()) {
    s << " q_expected=[";
    for (int j = 0; j < dof_; ++j) { if (j) s << ','; s << plan_q_bar_(j,timestep); }
    s << ']';
  }
  std_msgs::String msg; msg.data = s.str(); witness_diagnostic_pub_.publish(msg);
  // Unthrottled evidence with exact identities also survives CSV packet loss.
  ROS_INFO_STREAM(msg.data);
}

void LocalSparseSCPPlanner::updateRepairWitnessesLocked(
    const care_collision_cdf::CollisionCDFConstraintBatch& batch) {
  // Identity only is retained. Every accepted query must supply its own finite
  // distance and full 7D gradient at the current (point, timestep, q_bar).
  // This runs in BOTH phases, and freshness is recomputed, never latched.
  repair_witness_requires_reobserve_ = true;
  const std::size_t n = batch.distance.size();
  if (batch.num_pairs < 0 || n != static_cast<std::size_t>(batch.num_pairs) ||
      batch.dof != dof_ || batch.original_timestep.size() != n ||
      (!batch.source_type.empty() && batch.source_type.size() != n) ||
      batch.point_flat.size() != n * 3 ||
      batch.gradient_flat.size() != n * dof_ ||
      batch.q_linearization_flat.size() != n * dof_) {
    publishWitnessDiagnosticLocked(-1, -1, Eigen::Vector3d::Zero(), "batch_dimension_error");
    for (std::size_t i = 0; i < repair_unknown_witnesses_.size(); ++i)
      publishWitnessDiagnosticLocked(i, repair_unknown_witnesses_[i].timestep,
          repair_unknown_witnesses_[i].point, "batch_dimension_error");
    return;
  }

  std::vector<RepairWitness> fresh;
  std::vector<std::string> reasons(n, "outside_safety_horizon");
  std::vector<double> q_errors(n, 0.0);
  bool valid_batch = true;
  auto same_point = [](const RepairWitness& a, const RepairWitness& b) {
    return (a.point - b.point).lpNorm<Eigen::Infinity>() <= 1e-5;
  };
  auto is_persistent_witness_point = [&](const Eigen::Vector3d& point) {
    return std::any_of(
        repair_unknown_witnesses_.begin(), repair_unknown_witnesses_.end(),
        [&](const RepairWitness& witness) {
          return (witness.point - point).lpNorm<Eigen::Infinity>() <= 1e-5;
        });
  };
  auto is_vbc_feedback_point = [&](const Eigen::Vector3d& point) {
    return std::any_of(
        repair_unknown_witnesses_.begin(), repair_unknown_witnesses_.end(),
        [&](const RepairWitness& witness) {
          return witness.vbc_feedback &&
              (witness.point - point).lpNorm<Eigen::Infinity>() <= 1e-5;
        });
  };
  // The active VBC obligation is a target to approach, not a confirmed
  // collision witness.  Its points already enter the QP through the short
  // visibility-obligation horizon.  Keep them out of the persistent
  // repair-witness set even when the ordinary local CDF query reports them as
  // UNKNOWN near the current linearization; otherwise the same point is
  // promoted to a full-horizon hard wall and the q_vis frontier can never
  // reach it.
  auto is_active_obligation_point = [&](const Eigen::Vector3d& point) {
    return std::any_of(
        latest_vbc_obligation_points_.begin(),
        latest_vbc_obligation_points_.end(),
        [&](const Eigen::Vector3d& obligation_point) {
          return (point - obligation_point).lpNorm<Eigen::Infinity>() <= 1e-5;
        });
  };
  repair_unknown_witnesses_.erase(
      std::remove_if(
          repair_unknown_witnesses_.begin(),
          repair_unknown_witnesses_.end(),
          [&](const RepairWitness& witness) {
            return is_active_obligation_point(witness.point) &&
                !witness.vbc_feedback;
          }),
      repair_unknown_witnesses_.end());
  for (std::size_t i = 0; i < n; ++i) {
    const int k = batch.original_timestep[i];
    const Eigen::Vector3d point(batch.point_flat[3*i],
        batch.point_flat[3*i+1], batch.point_flat[3*i+2]);
    const bool persistent_witness_row = is_persistent_witness_point(point);
    const bool vbc_feedback_row = is_vbc_feedback_point(point);
    if (k < 1 || k > num_intervals_ ||
        (k > std::min(num_intervals_, cdf_constraint_horizon_steps_) &&
         !persistent_witness_row)) continue;
    RepairWitness w{Eigen::Vector3d(batch.point_flat[3*i],
        batch.point_flat[3*i+1], batch.point_flat[3*i+2]),
        k, cdf_safety_margin_, vbc_feedback_row};
    std::string reason = "fresh";
    if (!w.point.allFinite()) reason = "nonfinite_point";
    else if (!std::isfinite(batch.distance[i])) reason = "nonfinite_distance";
    double gradient_l1 = 0.0;
    for (int j = 0; j < dof_; ++j) {
      const double q = batch.q_linearization_flat[i*dof_+j];
      const double g = batch.gradient_flat[i*dof_+j];
      if (!std::isfinite(q)) reason = "nonfinite_q";
      else if (!std::isfinite(g)) reason = "nonfinite_gradient";
      else {
        q_errors[i] = std::max(q_errors[i], std::fabs(q-plan_q_bar_(j,k)));
        if (q_errors[i] > cdf_linearization_tolerance_inf_ && reason == "fresh")
          reason = "q_linearization_mismatch";
      }
      gradient_l1 += std::fabs(g);
    }
    const uint8_t source = batch.source_type.empty()
        ? static_cast<uint8_t>(care_collision_cdf::CollisionCDFConstraintBatch::SOURCE_UNKNOWN) : batch.source_type[i];
    if (source != care_collision_cdf::CollisionCDFConstraintBatch::SOURCE_UNKNOWN &&
        source != care_collision_cdf::CollisionCDFConstraintBatch::SOURCE_OCCUPIED)
      reason = "invalid_source";
    reasons[i] = reason;
    if (reason != "fresh") {
      valid_batch = false;
      publishWitnessDiagnosticLocked(-1, k, w.point, reason, i, q_errors[i]);
      continue;
    }
    // Preserve a local CDF margin already assigned to the same witness when
    // it is re-queried by GCDF. A fresh ordinary GCDF witness starts at the
    // configured CDF margin. VBC feedback may assign a larger native CDF
    // margin only to the matching rejected point.
    for (const auto& old : repair_unknown_witnesses_) {
      if (same_point(old, w) && std::isfinite(old.safety_margin)) {
        w.timestep = old.timestep;
        w.vbc_feedback = old.vbc_feedback;
        w.safety_margin = std::max(cdf_safety_margin_, old.safety_margin);
        break;
      }
    }
    fresh.push_back(w);
    if (source == care_collision_cdf::CollisionCDFConstraintBatch::SOURCE_UNKNOWN &&
        batch.distance[i] - trust_radius_ * gradient_l1 < cdf_safety_margin_ &&
        !is_active_obligation_point(w.point) &&
        std::none_of(repair_unknown_witnesses_.begin(), repair_unknown_witnesses_.end(),
                     [&](const RepairWitness& old) {
                       return same_point(old, w);
                     })) {
      if (repair_unknown_witnesses_.size() < 32) repair_unknown_witnesses_.push_back(w);
      else {
        repair_witness_overflow_ = true;
        publishWitnessDiagnosticLocked(-1, k, w.point, "witness_capacity_exceeded", i);
      }
    }
  }
  const bool all_seen = std::all_of(repair_unknown_witnesses_.begin(), repair_unknown_witnesses_.end(),
      [&](const RepairWitness& old) {
        return std::any_of(fresh.begin(), fresh.end(),
                          [&](const RepairWitness& w) {
                            return same_point(old, w);
                          });
      });
  repair_witness_requires_reobserve_ = repair_witness_overflow_ || !valid_batch || !all_seen;
  for (std::size_t wi = 0; wi < repair_unknown_witnesses_.size(); ++wi) {
    const auto& w = repair_unknown_witnesses_[wi];
    int matched = -1;
    std::string reason = "missing_point";
    for (std::size_t i = 0; i < n; ++i) {
      const Eigen::Vector3d p(batch.point_flat[3*i], batch.point_flat[3*i+1], batch.point_flat[3*i+2]);
      if (!p.allFinite() || (p-w.point).lpNorm<Eigen::Infinity>() > 1e-5) continue;
      matched = static_cast<int>(i); reason = reasons[i];
      if (reason == "fresh") break;
    }
    publishWitnessDiagnosticLocked(wi, w.timestep, w.point, reason, matched,
        matched >= 0 ? q_errors[matched] : 0.0);
  }
}

std::string LocalSparseSCPPlanner::groundContextLocked() const {
  if(!terminal_goal_valid_)return {};
  return std::to_string(terminal_goal_revision_)+":"+std::to_string(mode_epoch_)+":"+
      std::to_string(latest_reference_.header.stamp.toNSec());
}

void LocalSparseSCPPlanner::checkKnownGroundHoldLocked() {
  if(!primitive_observation_guard_.knownGroundEnabled() || !known_ground_hold_.due(ros::WallTime::now().toSec()))return;
  const double now=ros::Time::now().toSec(),wall=ros::WallTime::now().toSec();
  Eigen::VectorXd q;double minimum=std::numeric_limits<double>::infinity();
  const double stamp=latest_joint_state_.header.stamp.toSec();
  bool valid=has_joint_state_ && stamp>0. && now>=stamp && now-stamp<=.05 &&
      extractMeasuredQ(latest_joint_state_,q);
  if(valid) {
    Eigen::MatrixXd points=q;std::vector<PrimitiveObservationGuard::Row> rows;std::string error;
    valid=primitive_observation_guard_.linearizeKnownGround(points,&rows,&error) && !rows.empty();
    for(const auto& row:rows)minimum=std::min(minimum,row.distance_m-row.required_margin_m);
  }
  const auto before=known_ground_hold_.state;
  const bool wake=known_ground_hold_.observe(groundContextLocked(),now,wall,stamp,valid,minimum,0.);
  if(wake)requestPlanLocked("ground_clearance_restored");
  if(wake || before!=known_ground_hold_.state)
    ROS_INFO_STREAM("[GROUND_HOLD] event="<<known_ground_hold_.reason<<" goal="<<known_ground_hold_.goal
        <<" minimum_m="<<minimum<<" wakeups="<<known_ground_hold_.wakeups<<" requested="<<plan_requested_);
}

void LocalSparseSCPPlanner::timerCallback(const ros::TimerEvent&) {
  bool goal_stop_heartbeat=false;
  bool should_start = false;
  bool should_abort = false;
  std::string replacement_event;

  {
    std::unique_lock<std::mutex> lock(mutex_);
    checkKnownGroundHoldLocked();
    if(terminal_retention_.armed && (terminal_tracker_token_==terminal_retention_.armed || terminal_tracker_completed_) &&
        !retainTerminalLocked())requestPlanLocked("terminal_retention_released");
    goal_stop_heartbeat=repair_goal_stopping_enabled_ && repair_goal_handoff_execution_;
    if(repair_goal_stopping_enabled_ && terminal_retention_.armed &&
        (terminal_tracker_token_==terminal_retention_.armed || terminal_tracker_completed_) &&
        retainTerminalLocked()) {
      // Keep the receiver's certified stop while it is executing or holding.
      // Replacement reservations remain recorded; their ordinary timer work
      // must not schedule a competing motion or suppress this heartbeat.
      lock.unlock();publishSummary("goal_stop_heartbeat");return;
    }
    if(!state_evidence_retry_pending_.empty() && ros::WallTime::now()>=state_evidence_retry_deadline_) {
      const bool residual=state_residual_retry_raw_ && state_residual_retry_epoch_==mode_epoch_ &&
          state_residual_retry_token_==latest_observation_token_ && repair_mode_ && !probe_mode_ &&
          state_evidence_retry_pending_==std::to_string(mode_epoch_)+":"+safe_frontier_recovery_.key;
      state_evidence_retry_pending_.clear();
      plan_requested_=false;
      // A state failure is not a collision witness. Its saved, authenticated
      // residual can still schedule a new proposal from the SAME finite
      // steering ledger. Never renew the state wait or reuse the old audit.
      const auto event=residual ? handleHandoffResidualLocked(state_residual_retry_raw_,
          state_residual_retry_audit_,state_residual_retry_failure_) :
          std::string("state_evidence_retry_exhausted_hold");
      state_residual_retry_raw_=0;
      lock.unlock();publishSummary(event);return;
    }

    if (candidate_replacement_pending_) {
      const bool applied = !candidate_replacement_new_token_.empty() &&
          latest_observation_token_ == candidate_replacement_new_token_;
      if (!applied && repair_mode_ && !probe_mode_ &&
          mode_epoch_ == candidate_replacement_epoch_ &&
          (candidate_replacement_execution_v2_ || ros::Time::now() < candidate_replacement_deadline_) &&
          ros::WallTime::now() < candidate_replacement_wall_deadline_) return;
      if(applied) {
        candidate_replacement_pending_ = false;
        safe_frontier_recovery_.finishViewReplacement(true);
        requestPlanLocked("candidate_replacement_applied");
      } else {
        // Expiry retires this reservation, not its still-live observation.
        // A new measured query must authenticate the next charged attempt.
        replacement_event=finishReplacementLocked(true,"candidate_replacement_expired",true);
      }
    }

    if (candidate_replacement_trigger_pending_) {
      if (repair_mode_ && !probe_mode_ &&
          mode_epoch_ == candidate_replacement_epoch_ &&
          latest_observation_token_ == candidate_replacement_trigger_observation_token_ &&
          ros::Time::now() < candidate_replacement_trigger_deadline_ &&
          ros::WallTime::now() < candidate_replacement_trigger_wall_deadline_) return;
      candidate_replacement_trigger_pending_ = false;
      requestPlanLocked("candidate_replacement_trigger_invalid_or_expired");
      ROS_WARN_STREAM("[candidate_replacement_trigger_retired] mode_epoch="<<mode_epoch_
          <<" query_stamp_ns="<<current_query_stamp_.toNSec()
          <<" observation_token="<<candidate_replacement_trigger_observation_token_
          <<" trigger_reason="<<candidate_replacement_trigger_reason_
          <<" plan_request_reason="<<plan_request_reason_
          <<" plan_requested="<<plan_requested_
          <<" attempts="<<safe_frontier_recovery_.attempts
          <<" pending="<<candidate_replacement_pending_
          <<" wall_s="<<ros::WallTime::now().toSec());
      if(replacement_event.empty())replacement_event="candidate_replacement_trigger_retired";
    }

    if (plan_running_ && waiting_for_cdf_ &&
        current_query_wall_.toSec() > 0.0 &&
        (ros::WallTime::now() - current_query_wall_).toSec() >
            cdf_wait_timeout_s_) {
      should_abort = true;
    }

    if (!plan_running_ && plan_requested_ && state_evidence_retry_pending_.empty()) {
      const bool interval_ok =
          last_plan_finish_time_.isZero() ||
          (ros::Time::now() - last_plan_finish_time_).toSec() >=
              min_replan_interval_s_;
      should_start = interval_ok;
    }
  }

  if(goal_stop_heartbeat)publishSummary("goal_stop_heartbeat");
  if(!replacement_event.empty())publishSummary(replacement_event);
  if (should_abort) {
    abortPlan("cdf_wait_timeout");
    return;
  }
  if (should_start) startPlan();
}

bool LocalSparseSCPPlanner::extractMeasuredQ(
    const sensor_msgs::JointState& msg,
    Eigen::VectorXd& q) const {
  std::unordered_map<std::string, std::size_t> index;
  for (std::size_t i = 0; i < msg.name.size(); ++i)
    index[msg.name[i]] = i;

  q = Eigen::VectorXd::Zero(dof_);
  for (int j = 0; j < dof_; ++j) {
    const auto it = index.find(joint_names_[static_cast<std::size_t>(j)]);
    if (it == index.end() || it->second >= msg.position.size())
      return false;
    q[j] = msg.position[it->second];
  }
  return finiteVector(q);
}

bool LocalSparseSCPPlanner::buildTrajectoryMapping(
    const trajectory_msgs::JointTrajectory& msg,
    std::vector<int>& mapping) const {
  std::unordered_map<std::string, int> index;
  for (std::size_t i = 0; i < msg.joint_names.size(); ++i)
    index[msg.joint_names[i]] = static_cast<int>(i);

  mapping.resize(static_cast<std::size_t>(dof_));
  for (int j = 0; j < dof_; ++j) {
    const auto it = index.find(joint_names_[static_cast<std::size_t>(j)]);
    if (it == index.end()) return false;
    mapping[static_cast<std::size_t>(j)] = it->second;
  }
  return true;
}

bool LocalSparseSCPPlanner::sampleReferencePosition(
    const trajectory_msgs::JointTrajectory& msg,
    const std::vector<int>& mapping,
    double t,
    Eigen::VectorXd& q) const {
  if (msg.points.empty()) return false;

  auto copy =
      [&](const trajectory_msgs::JointTrajectoryPoint& p) -> bool {
        if (p.positions.size() < msg.joint_names.size()) return false;
        q = Eigen::VectorXd::Zero(dof_);
        for (int j = 0; j < dof_; ++j) {
          const int idx = mapping[static_cast<std::size_t>(j)];
          q[j] = p.positions[static_cast<std::size_t>(idx)];
        }
        return finiteVector(q);
      };

  const double first = msg.points.front().time_from_start.toSec();
  const double last = msg.points.back().time_from_start.toSec();
  if (t <= first) return copy(msg.points.front());
  if (t >= last) return copy(msg.points.back());

  std::size_t hi = 1;
  while (hi < msg.points.size() &&
         msg.points[hi].time_from_start.toSec() < t) ++hi;
  if (hi >= msg.points.size()) return copy(msg.points.back());

  Eigen::VectorXd q0, q1;
  const auto& p0 = msg.points[hi - 1];
  const auto& p1 = msg.points[hi];
  if (!copy(p0)) return false;
  q0 = q;
  if (!copy(p1)) return false;
  q1 = q;

  const double t0 = p0.time_from_start.toSec();
  const double t1 = p1.time_from_start.toSec();
  const double h = t1 - t0;
  if (h <= 1e-12) return false;

  const double alpha = (t - t0) / h;
  q = (1.0 - alpha) * q0 + alpha * q1;
  return finiteVector(q);
}

bool LocalSparseSCPPlanner::buildReferenceHorizon(
    const Eigen::VectorXd& q_current,
    const trajectory_msgs::JointTrajectory& reference,
    const ros::Time& now,
    bool probe_mode,
    Eigen::MatrixXd& q_ref,
    Eigen::MatrixXd& u_ref,
    Eigen::MatrixXd& q_init,
    Eigen::MatrixXd& u_init) const {
  std::vector<int> mapping;
  if (!buildTrajectoryMapping(reference, mapping)) return false;

  q_ref = Eigen::MatrixXd::Zero(dof_, num_intervals_ + 1);
  u_ref = Eigen::MatrixXd::Zero(dof_, num_intervals_);
  q_init = Eigen::MatrixXd::Zero(dof_, num_intervals_ + 1);
  u_init = Eigen::MatrixXd::Zero(dof_, num_intervals_);

  q_ref.col(0) = q_current;
  q_init.col(0) = q_current;

  double reference_age = 0.0;
  if (!reference.header.stamp.isZero()) {
    reference_age =
        std::max(0.0, (now - reference.header.stamp).toSec());
  }

  for (int k = 1; k <= num_intervals_; ++k) {
    Eigen::VectorXd qk;
    if (!sampleReferencePosition(
            reference, mapping, reference_age + k * dt_, qk)) {
      return false;
    }
    if (probe_mode) {
      // Build a local reference in the same physical time and speed domain
      // as the executable PROBE, rather than slowing a 2-rad/s path later.
      for (int j = 0; j < dof_; ++j) {
        const double step = dt_*std::min(velocity_limits_[j], probe_velocity_cap_);
        qk[j] = q_ref(j,k-1) + clampValue(qk[j]-q_ref(j,k-1), -step, step);
      }
    }
    q_ref.col(k) = qk;
  }

  // C5.24: a PROBE is a local task-continuation test, not a full NORMAL
  // horizon solve whose suffix happens to be discarded later. Advance the
  // nominal task reference only through probe_task_horizon_steps_, then hold
  // that local target for the rest of the optimization horizon. This keeps the
  // fixed QP dimension while making terminal/state tracking consistent with
  // the short executable PROBE semantics.
  if (probe_mode) {
    const int hold_k =
        std::max(1, std::min(probe_task_horizon_steps_, num_intervals_));
    const Eigen::VectorXd q_hold = q_ref.col(hold_k);
    for (int k = hold_k + 1; k <= num_intervals_; ++k) {
      q_ref.col(k) = q_hold;
    }
  }

  for (int k = 0; k < num_intervals_; ++k) {
    Eigen::VectorXd ur =
        (q_ref.col(k + 1) - q_ref.col(k)) / dt_;
    for (int j = 0; j < dof_; ++j) {
      ur[j] = clampValue(
          ur[j], -std::min(velocity_limits_[j], probe_mode ? probe_velocity_cap_ : velocity_limits_[j]),
                  std::min(velocity_limits_[j], probe_mode ? probe_velocity_cap_ : velocity_limits_[j]));
    }
    u_ref.col(k) = ur;

    Eigen::VectorXd ui =
        (q_ref.col(k + 1) - q_init.col(k)) / dt_;
    for (int j = 0; j < dof_; ++j) {
      ui[j] = clampValue(
          ui[j], -std::min(velocity_limits_[j], probe_mode ? probe_velocity_cap_ : velocity_limits_[j]),
                  std::min(velocity_limits_[j], probe_mode ? probe_velocity_cap_ : velocity_limits_[j]));
    }
    u_init.col(k) = ui;
    q_init.col(k + 1) = q_init.col(k) + dt_ * ui;
  }

  return finiteMatrix(q_ref) && finiteMatrix(q_init) &&
         finiteMatrix(u_ref) && finiteMatrix(u_init);
}

std::string LocalSparseSCPPlanner::normalUnknownContextLocked() const {
  if (repair_mode_ || probe_mode_ || !has_reference_ || normal_reference_refresh_pending_ ||
      latest_vbc_obligation_id_ < 0 || latest_vbc_obligation_points_.empty() ||
      !latest_single_waypoint_active_ || !has_single_waypoint_q_ ||
      latest_observation_token_ == "none") return "";
  std::vector<std::string> points;
  for (const auto& p : latest_vbc_obligation_points_) {
    std::ostringstream s;
    for (int j=0;j<3;++j) s << ':' << std::llround(p[j]*1e5);
    points.push_back(s.str());
  }
  std::sort(points.begin(), points.end());
  std::ostringstream s;
  s << mode_epoch_ << ':' << latest_reference_.header.stamp.toNSec() << ':'
    << latest_vbc_obligation_id_ << ':' << latest_observation_token_;
  for (const auto& p : points) s << p;
  return s.str();
}

std::string LocalSparseSCPPlanner::normalTaskProgressContextLocked() const {
  if (!terminal_retention_enabled_ || !terminal_goal_valid_) return {};
  const auto context=normalUnknownContextLocked();
  return context.empty() ? context : context+":goal:"+std::to_string(terminal_goal_revision_);
}

void LocalSparseSCPPlanner::resetNormalTaskProgressLocked(const std::string& reason) {
  normal_unknown_task_progress_.reset(reason);
  logNormalTaskProgressLocked();
  if (!normal_task_stall_pending_) return;
  normal_task_stall_pending_=false;normal_task_stall_context_.clear();
  task_no_progress_.reset();
  std_msgs::String reset;reset.data="status=reset reason=normal_task_"+reason;
  task_stall_pub_.publish(reset);
  if (!repair_mode_ && !probe_mode_ && has_reference_ && !normal_reference_refresh_pending_)
    requestPlanLocked("normal_task_"+reason);
}

void LocalSparseSCPPlanner::logNormalTaskProgressLocked() {
  const auto& g=normal_unknown_task_progress_;
  if (normal_task_progress_logged_sequence_==g.event_sequence) return;
  normal_task_progress_logged_sequence_=g.event_sequence;
  const bool reset=g.event=="reset";
  const auto& evidence=reset ? g.reset_evidence : g.last;
  ROS_INFO_STREAM("[NORMAL_TASK_PROGRESS] event=" << g.event << " reason=" << g.reason
      << " context=" << g.event_context << " raw_candidate_stamp_ns=" << evidence.raw
      << " query_stamp_ns=" << evidence.query
      << " execution_stamp_ns=" << (reset ? g.reset_execution : g.last_execution_stamp_ns)
      << " duration_s=" << (reset ? g.reset_duration_s : g.duration(ros::Time::now().toSec()))
      << " executions=" << (reset ? g.reset_execution_stamps : g.execution_stamps)
      << " credited_error=" << (reset ? g.reset_credited_error : g.credited_error)
      << " current_error=" << (reset ? g.reset_current_error : g.current_error));
}

std::string LocalSparseSCPPlanner::repairTargetKeyLocked() const {
  // Deadlines may be refreshed without changing the actual observation goal.
  std::ostringstream s;
  s << std::setprecision(17);
  if (!latest_schedule_.empty()) {
    s << "schedule";
    for (const auto& wp : latest_schedule_) {
      s << ':' << wp.id;
      for (int j = 0; j < wp.q.size(); ++j) s << ':' << wp.q[j];
      for (int j = 0; j < wp.joint_mask.size(); ++j)
        s << ":m" << wp.joint_mask[j];
    }
  } else if (latest_single_waypoint_active_ && has_single_waypoint_q_) {
    s << latest_observation_token_;
    for (int j = 0; j < latest_single_waypoint_q_.size(); ++j)
      s << ':' << latest_single_waypoint_q_[j];
    for (int j = 0; j < latest_single_waypoint_joint_mask_.size(); ++j)
      s << ":m" << latest_single_waypoint_joint_mask_[j];
  }
  return s.str();
}

bool LocalSparseSCPPlanner::prefixSelectionCompatible(const RepairPrefixSelection& c) const {
  if (c.session.size()!=32 || !c.revision || c.acceleration.size()!=static_cast<std::size_t>(dof_) ||
      !std::all_of(c.session.begin(),c.session.end(),[](char x){return (x>='0'&&x<='9')||(x>='a'&&x<='f');}) ||
      !std::isfinite(c.prefix_s) || !std::isfinite(c.brake_dt_s) ||
      c.brake_dt_s!=repair_executable_settings_.brake_dt ||
      c.brake_steps!=static_cast<unsigned>(repair_executable_settings_.max_brake_steps) ||
      std::none_of(repair_executable_settings_.prefixes.begin(),repair_executable_settings_.prefixes.end(),
          [&](double p){return p==c.prefix_s;})) return false;
  for (int j=0;j<dof_;++j)
    if (!std::isfinite(c.acceleration[j]) || c.acceleration[j]!=repair_executable_settings_.acceleration[j])return false;
  return true;
}

bool LocalSparseSCPPlanner::prefixSelectionCurrentLocked(const RepairPrefixSelection& c) const {
  return prefixSelectionCompatible(c) && c.session==latest_prefix_selection_.session &&
      c.revision==latest_prefix_selection_.revision && c.prefix_s==latest_prefix_selection_.prefix_s;
}

void LocalSparseSCPPlanner::repairPrefixSelectionCallback(const RepairPrefixSelectionConstPtr& msg) {
  if (!repair_prefix_binding_enabled_ || !msg || !prefixSelectionCompatible(*msg) ||
      msg->header.stamp.isZero() || msg->header.stamp>ros::Time::now()) return;
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_prefix_sessions_.count(msg->session))return;
  if (msg->session==latest_prefix_selection_.session) {
    if (msg->revision<latest_prefix_selection_.revision)return;
    if (msg->revision==latest_prefix_selection_.revision)return; // Never alter one identity.
  } else if (!latest_prefix_selection_.session.empty()) {
    if (closed_prefix_sessions_.size()>=64)return;
    closed_prefix_sessions_.insert(latest_prefix_selection_.session);
  }
  latest_prefix_selection_=*msg;
  if (repair_mode_ && !probe_mode_) {
    if (plan_running_) {
      ++plan_sequence_; // Invalidate an in-flight worker/start snapshot.
      plan_running_=false;waiting_for_cdf_=false;pending_batch_.reset();
      plan_executable_iterates_=RepairExecutableIterates{};
    }
    requestPlanLocked("repair_prefix_selection_changed");
  }
  ROS_INFO_STREAM("[REPAIR_PREFIX_SELECTION] session="<<msg->session<<" revision="<<msg->revision
      <<" prefix_s="<<msg->prefix_s);
}

std::string LocalSparseSCPPlanner::repairExecutableContextLocked() const {
  std::ostringstream s;
  s << std::setprecision(17) << plan_sequence_ << ':' << mode_epoch_ << ':'
    << plan_repair_ticket_.revision << ':' << plan_repair_ticket_.progress << ':'
    << std::quoted(plan_repair_ticket_.target) << ':'
    << std::quoted(plan_observation_token_) << ':'
    << std::quoted(latest_observation_token_) << ':'
    << std::quoted(repairTargetKeyLocked()) << ':' << latest_vbc_obligation_id_;
  if (repair_prefix_binding_enabled_) s<<":prefix:"<<latest_prefix_selection_.session<<':'
      <<latest_prefix_selection_.revision<<':'<<latest_prefix_selection_.prefix_s;
  if (!latest_reference_.points.empty()) {
    s << ":goal:" << std::quoted(latest_reference_.header.frame_id);
    for (const auto& name : latest_reference_.joint_names) s << ':' << std::quoted(name);
    for (double value : latest_reference_.points.back().positions) s << ':' << value;
  }
  std::vector<std::string> rows;
  for (const auto& w : repair_unknown_witnesses_) {
    std::ostringstream row;
    row << std::setprecision(17) << w.point.transpose() << ':'
        << w.safety_margin << ':' << w.has_primitive_guard;
    if (w.has_primitive_guard)
      row << ':' << w.primitive_guard.point.transpose() << ':'
          << std::quoted(w.primitive_guard.sweeping_link) << ':'
          << std::quoted(w.primitive_guard.collision_name) << ':'
          << w.primitive_guard.collision_index;
    rows.push_back(row.str());
  }
  if(restore_path_guard_scope_==std::to_string(mode_epoch_)+":"+latest_observation_token_)
    for(const auto& p:restore_path_guard_points_) {
      std::ostringstream row;row<<std::setprecision(17)<<"ordinary_path:"<<p.point.transpose();
      rows.push_back(row.str());
    }
  for (const auto& p : latest_vbc_obligation_points_) {
    std::ostringstream row; row << std::setprecision(17) << "obligation:" << p.transpose();
    rows.push_back(row.str());
  }
  std::sort(rows.begin(),rows.end());
  for (const auto& row : rows) s << ':' << std::quoted(row);
  return s.str();
}

VbcFailureDirectionMemory::Context LocalSparseSCPPlanner::vbcFailureContextLocked() const {
  VbcFailureDirectionMemory::Context c;
  c.enabled=candidate_replacement_enabled_ && safe_frontier_recovery_enabled_ &&
      repair_mode_ && !probe_mode_ && repair_observation_phase_ &&
      latest_schedule_.empty() && latest_single_waypoint_active_ && has_single_waypoint_q_;
  c.mode=mode_epoch_; c.reference=latest_reference_.header.stamp.toNSec();
  c.owner=latest_vbc_obligation_id_; c.token=latest_observation_token_;
  c.target=latest_single_waypoint_q_; c.mask=latest_single_waypoint_joint_mask_;
  c.points=latest_vbc_obligation_points_;
  return c;
}

void LocalSparseSCPPlanner::selectRepairTargetLocked() {
  vbc_failure_direction_memory_.observeContext(vbcFailureContextLocked());
  const auto key = repairTargetKeyLocked();
  // A different q_vis for the same region is not evidence for relaxing the
  // region's active-target guard. Quantization mirrors witness point matching.
  std::vector<std::string> geometry_points;
  for (const auto& point : latest_vbc_obligation_points_) {
    std::ostringstream p;
    for (int j=0;j<3;++j) p << ':' << std::llround(point[j]*1e5);
    geometry_points.push_back(p.str());
  }
  std::sort(geometry_points.begin(), geometry_points.end());
  std::ostringstream geometry;
  geometry << mode_epoch_; // Same region/new token or id cannot buy a fresh budget.
  for (const auto& p : geometry_points) geometry << p;
  const bool same_guard_region = !geometry_points.empty() &&
      geometry.str() == safe_frontier_margin_geometry_key_;
  safe_frontier_margin_geometry_key_ = geometry.str();
  if (key != vbc_feedback_target_key_) {
    // Same-region target revisions invalidate in-flight steering evidence,
    // but do not replenish attempts or forget the failed directions.
    if (safe_frontier_recovery_enabled_ && same_guard_region)
      safe_frontier_recovery_.invalidateMotion();
    // A new observation target starts a fresh feedback accounting epoch. Do
    // not carry rejection counters for a previous target into an unrelated
    // queued target. Point-scoped witness ladders are preserved separately across target changes.
    vbc_feedback_target_key_ = key;
    vbc_feedback_rejection_count_ = 0;
    vbc_feedback_matched_point_count_ = 0;
    if (!(safe_frontier_recovery_enabled_ && same_guard_region))
      visibility_obligation_cdf_margin_effective_.store(
          visibility_obligation_cdf_margin_base_);
  }
  if (key != final_vbc_repeat_target_key_) {
    final_vbc_repeat_target_key_ = key;
    final_vbc_previous_points_.clear();
    final_gcdf_previous_points_.clear();
  }
  repair_no_progress_.select(key);
  final_vbc_no_progress_.progress.select(key);
  safe_frontier_recovery_.select(
      candidate_replacement_enabled_ && latest_vbc_obligation_id_ >= 0 ?
          "obligation:" + std::to_string(latest_vbc_obligation_id_) :
          (safe_frontier_recovery_enabled_ && !geometry_points.empty() ? geometry.str() : key),
      mode_epoch_);
  // Use the stable published observation goal; rolling frontier updates must
  // not reset its progress window or credit visibility-free wrist motion.
  safe_frontier_recovery_.setObservationGoal(
      latest_single_waypoint_active_ && has_single_waypoint_q_ && latest_schedule_.empty()
          ? latest_observation_token_ : "none",
      latest_single_waypoint_q_, latest_single_waypoint_joint_mask_);
  const char* monitor=std::getenv("CARE_OBSERVATION_PROGRESS_ENABLED");
  safe_frontier_recovery_.persistent_observation_monitor=monitor && std::string(monitor)=="true";
  const char* action_progress=std::getenv("CARE_OBSERVATION_ACTION_PROGRESS");
  safe_frontier_recovery_.action_progress_enabled=action_progress && std::string(action_progress)=="true";
  safe_frontier_recovery_.setObservationRegion(latest_observation_region_);
}

void LocalSparseSCPPlanner::observationProgressCallback(const std_msgs::StringConstPtr& msg) {
  if (!safe_frontier_recovery_enabled_ || !msg) return;
  std::map<std::string,std::string> fields;
  std::istringstream input(msg->data); std::string word;
  while (input>>word) {
    const auto eq=word.find('=');
    if (eq==std::string::npos || !fields.emplace(word.substr(0,eq),word.substr(eq+1)).second) return;
  }
  if (fields["policy"]!="visibility_acquisition" || fields["active_query_status"]!="ok") return;
  double query_time=0.; Eigen::Vector3d metrics; long long owner=-1;
  long long count=0,inside=0,finite_conf=0,finite_vis=0;
  try {
    auto number=[&](const char* name) {
      const auto& value=fields.at(name); std::size_t used=0;
      const double n=std::stod(value,&used);
      if (used!=value.size() || !std::isfinite(n)) throw std::invalid_argument("number");
      return n;
    };
    query_time=number("active_query_ros_s");
    metrics<<number("active_seen_fraction"),number("active_min_confidence"),
        number("active_max_current_visibility");
    auto integer=[&](const char* name) {
      const auto& value=fields.at(name); std::size_t used=0; const auto n=std::stoll(value,&used);
      if (used!=value.size()) throw std::invalid_argument("integer"); return n;
    };
    owner=integer("active_obligation_id"); count=integer("active_point_count");
    inside=integer("active_inside_map_count"); finite_conf=integer("active_finite_confidence_count");
    finite_vis=integer("active_finite_current_visibility_count");
  } catch (...) { return; }
  if (count<=0 || inside!=count || finite_conf!=count || finite_vis!=count) return;
  std::lock_guard<std::mutex> lock(mutex_);
  if (!repair_mode_ || probe_mode_ || !repair_observation_phase_ ||
      owner!=latest_vbc_obligation_id_ || fields["active_feedback_token"]!=latest_observation_token_) return;
  if (safe_frontier_recovery_.noteObservationFeedback(fields["active_feedback_token"],query_time,
      ros::Time::now().toSec(),metrics)) {
    latest_observation_feedback_unseen_ = fields["active_seen"]=="0";
    // This fresh, complete seen report schedules a requery only. The new hard
    // geometry/map query still decides whether the fixed start is admissible.
    if(fields["active_seen"]=="1" && metrics[0]>=1. &&
        std::any_of(latest_vbc_obligation_points_.begin(),latest_vbc_obligation_points_.end(),
          [&](const Eigen::Vector3d& p){return (p-fixed_start_clearance_hold_.point).lpNorm<Eigen::Infinity>()<=1e-5;}) &&
        fixed_start_clearance_hold_.evidenceRecheck(query_time,ros::Time::now().toSec()))
      requestPlanLocked("fixed_start_new_seen_evidence_requery");
  }
}

std::string LocalSparseSCPPlanner::finishReplacementLocked(
    bool refresh_failed_target, const std::string& reason, bool unapplied_resource) {
  candidate_replacement_pending_=false;
  candidate_replacement_retry_pending_=false;
  safe_frontier_recovery_.finishViewReplacement(false);
  selectRepairTargetLocked();
  if(candidate_replacement_execution_v2_ && unapplied_resource) {
    const bool refunded=safe_frontier_recovery_.refundUnappliedReplacement(candidate_replacement_charge_);
    ROS_WARN_STREAM("[replacement_unapplied] reason="<<reason<<" refunded="<<refunded
        <<" attempts="<<safe_frontier_recovery_.attempts
        <<" resource_failures="<<safe_frontier_recovery_.unapplied_replacements);
  }
  candidate_replacement_retry_context_=candidate_replacement_reserved_failure_;
  const bool current_failure=candidate_replacement_reserved_qp_ && refresh_failed_target &&
      mode_epoch_==candidate_replacement_epoch_ &&
      candidate_replacement_retry_context_.blocks(repairTargetKeyLocked(),
          latest_observation_token_,mode_epoch_,repair_no_progress_.active.progress);
  std::string event=reason;
  if(current_failure && candidate_replacement_execution_v2_ &&
      !safe_frontier_recovery_.canRetryUnappliedReplacement()) {
    event="replacement_generation_resource_exhausted_hold";
  } else if(current_failure && qpReplacementRetryCurrentLocked()) {
    candidate_replacement_retry_pending_=true;
    event="repair_qp_replacement_refresh_scheduled";
  } else if(current_failure && !safe_frontier_recovery_.canReserveReplacement()) {
    event="repair_qp_candidate_exhausted_hold";
  }
  requestPlanLocked(reason);
  return event;
}

void LocalSparseSCPPlanner::candidateReplacementCallback(const std_msgs::StringConstPtr& msg) {
  if (!candidate_replacement_enabled_ || !msg) return;
  std::map<std::string, std::string> fields;
  std::istringstream input(msg->data);
  std::string word;
  while (input >> word) {
    const auto eq = word.find('=');
    if (eq == std::string::npos || !fields.emplace(word.substr(0,eq),word.substr(eq+1)).second) return;
  }
  unsigned long long id=0, epoch=0, query=0;
  if (!parseUnsignedToken(msg->data,"request_id",&id) || !id) return;
  std::unique_lock<std::mutex> lock(mutex_);
  if (fields["action"] == "finish") {
    if (!candidate_replacement_pending_ || id != candidate_replacement_id_) return;
    if((!candidate_replacement_execution_v2_ && ros::Time::now()>=candidate_replacement_deadline_) ||
        ros::WallTime::now()>=candidate_replacement_wall_deadline_) {
      const auto event=finishReplacementLocked(true,"candidate_replacement_expired",true);
      lock.unlock();publishSummary(event);return;
    }
    if (fields["new_token"].find("care_obs_v1_") == 0)
      candidate_replacement_new_token_ = fields["new_token"];
    else { // Generation rejection still costs its reserved attempt.
      const bool geometric_failure = fields["new_token"] == "none" &&
          (fields["generation_result"] == "no_geometric_candidate" ||
           fields["generation_result"] == "candidate_not_distinct" ||
           fields["generation_result"] == "generation_expired");
      const std::string outcome=fields["generation_result"];
      const bool unapplied=outcome=="generation_expired" || outcome=="context_changed" ||
          outcome=="stale_measured_state" || outcome=="invalid_measured_state" ||
          outcome=="generation_error" || outcome=="no_trajectory" ||
          outcome=="observation_already_seen" || outcome=="application_geometry_changed";
      const auto event=finishReplacementLocked(geometric_failure ||
          (candidate_replacement_execution_v2_ && unapplied),
          "candidate_replacement_rejected",unapplied);
      lock.unlock();
      publishSummary(event);
    }
    return;
  }
  const bool owner_promotion_request = fields["action"] == "reserve_owner";
  const bool decline_request = fields["action"] == "decline";
  if ((!owner_promotion_request && !decline_request && fields["action"] != "reserve") ||
      id <= candidate_replacement_last_id_) return;
  candidate_replacement_last_id_ = id;
  if (!parseUnsignedToken(msg->data,"mode_epoch",&epoch) ||
      !parseUnsignedToken(msg->data,"query_stamp_ns",&query)) return;
  selectRepairTargetLocked();
  const double age = (ros::Time::now()-last_query_ros_time_).toSec();
  const bool owner_promotion_reason =
      fields["reason"] == "final_vbc_owner_promotion" ||
      fields["reason"] == "observation_dependency_owner";
  const bool qp_request = fields["reason"] == "repair_qp_failure";
  const bool observation_request = fields["reason"] == "observation_no_progress";
  const bool path_request = fields["reason"] == "geometric_restore_path_counterexample" ||
      fields["reason"] == "geometric_restore_proof_resource_exhausted" ||
      fields["reason"] == "geometric_restore_handoff_residual_insufficient";
  const bool target_request = qp_request || observation_request || path_request;
  const bool final_vbc_request = target_request || fields["reason"] == "final_vbc_repeat" ||
      fields["reason"] == "final_gcdf_repeat" ||
      fields["reason"] == "final_vbc_owner_promotion";
  unsigned long long trigger_raw = 0;
  const bool trigger_identity = !final_vbc_request ||
      (parseUnsignedToken(msg->data, "trigger_raw_candidate_stamp_ns", &trigger_raw) &&
       trigger_raw == candidate_replacement_trigger_raw_ &&
       fields["observation_token"] == candidate_replacement_trigger_observation_token_ &&
       (owner_promotion_reason || fields["reason"] == candidate_replacement_trigger_reason_) &&
       fields["obligation_id"] ==
           std::to_string(candidate_replacement_trigger_obligation_id_));
  const bool valid = !candidate_replacement_pending_ && safe_frontier_recovery_enabled_ &&
      repair_mode_ && !probe_mode_ && repair_observation_phase_ &&
      epoch == mode_epoch_ && query == current_query_stamp_.toNSec() &&
      fields["observation_token"] == latest_observation_token_ &&
      age >= 0. && age <= (target_request ? 1.0 : .5) && trigger_identity &&
      owner_promotion_request == owner_promotion_reason &&
      (!final_vbc_request || candidate_replacement_trigger_pending_);
  // A denial for a delayed/foreign request must not retire a newer trigger.
  // Freshness and budget still govern grants; this only binds the rejection
  // transition to the exact outstanding query/owner/reason.
  const bool matches_pending_trigger = final_vbc_request &&
      candidate_replacement_trigger_pending_ && trigger_identity &&
      epoch == candidate_replacement_epoch_ && epoch == mode_epoch_ &&
      query == current_query_stamp_.toNSec() &&
      owner_promotion_request == owner_promotion_reason;
  if (decline_request) {
    // A terminal Python pool/capacity response acknowledges only the exact
    // outstanding final-VBC trigger. It grants no motion, spends no attempt,
    // and cannot reset failure memory or bypass the existing hold policies.
    if (!valid || (fields["reason"] != "final_vbc_repeat" && fields["reason"] != "final_gcdf_repeat") ||
        fields["decline_reason"] != "no_untried_sensor_or_family_capacity") return;
    candidate_replacement_trigger_pending_ = false;
    requestPlanLocked("candidate_replacement_unavailable");
    lock.unlock();
    publishSummary("candidate_replacement_unavailable");
    return;
  }
  // Owner promotion changes only which existing obligation is observed first.
  // It does not synthesize a q_vis candidate, so it must not consume one of
  // the finite candidate-replacement attempts. The same authenticated
  // trigger/pending handshake still freezes planning until the new token is
  // published or the request expires.
  auto replacement_charge=safe_frontier_recovery_.replacementCharge();
  const bool resource_available=!candidate_replacement_execution_v2_ || owner_promotion_request ||
      safe_frontier_recovery_.canRetryUnappliedReplacement();
  const bool granted = valid && resource_available && (owner_promotion_request
      ? owner_promotion_reason && safe_frontier_recovery_.authorizeOwnerPromotion()
      : final_vbc_request
      ? safe_frontier_recovery_.reserveReplacementAfterVbc(
            candidate_replacement_trigger_direction_advanced_)
      : safe_frontier_recovery_.reserveReplacement());
  const std::string denial = granted ? "none" : !valid ? "stale_context" :
      !resource_available ? "generation_resource_exhausted" :
      owner_promotion_request ? "owner_unavailable" : "recovery_budget_exhausted";
  std_msgs::String reply;
  std::ostringstream s;
  s << "request_id=" << id << " granted=" << granted << " mode_epoch=" << mode_epoch_
    << " query_stamp_ns=" << query << " attempts=" << safe_frontier_recovery_.attempts
    << " segments=" << safe_frontier_recovery_.segments
    << " owner_promotion=" << static_cast<int>(owner_promotion_request)
    << " denial_reason=" << denial;
  if (granted) {
    if(!owner_promotion_request)safe_frontier_recovery_.beginViewReplacement();
    if(!owner_promotion_request)safe_frontier_recovery_.sealReplacementCharge(replacement_charge);
    candidate_replacement_charge_=replacement_charge;
    candidate_replacement_reserved_qp_ = qp_request;
    candidate_replacement_reserved_failure_ = repair_candidate_failure_;
    candidate_replacement_trigger_pending_ = false;
    candidate_replacement_pending_=true; candidate_replacement_id_=id;
    candidate_replacement_epoch_=mode_epoch_; candidate_replacement_new_token_.clear();
    candidate_replacement_deadline_=ros::Time::now()+ros::Duration(candidate_replacement_execution_v2_?10.:2.);
    candidate_replacement_wall_deadline_=ros::WallTime::now()+ros::WallDuration(candidate_replacement_execution_v2_?10.:3.);
    ++plan_sequence_; plan_running_=false; waiting_for_cdf_=false;
    pending_batch_.reset(); plan_requested_=false;
    s << " deadline_ns=" << candidate_replacement_deadline_.toNSec();
    if(candidate_replacement_execution_v2_)
      s << " lease_timing=wall_v2 wait_budget_s=3 generation_budget_s=6 deadline_wall_ns="
        << candidate_replacement_wall_deadline_.toNSec();
  } else if (matches_pending_trigger) {
    candidate_replacement_trigger_pending_ = false;
    requestPlanLocked("candidate_replacement_denied");
  }
  reply.data=s.str(); candidate_replacement_grant_pub_.publish(reply);
  ROS_WARN_STREAM("[candidate_replacement] " << reply.data);
  lock.unlock();
  if (!granted) publishSummary(denial == "recovery_budget_exhausted"
      ? "repair_qp_candidate_exhausted_hold" : "candidate_replacement_denied");
}

void LocalSparseSCPPlanner::admissionTerminalCallback(const CandidateTransferConstPtr& msg) {
  if (!msg || (msg->action!="preaudit_expired" && msg->action!="preaudit_ground_rejected" && msg->action!="preaudit_ground_resource" && msg->action!="preaudit_metric_rejected" && msg->action!="preaudit_metric_resource") ||
      msg->remaining_budget_s!=0. || !msg->sequence) return;
  const bool metric=msg->action=="preaudit_metric_rejected" || msg->action=="preaudit_metric_resource";
  const bool resource=msg->action=="preaudit_ground_resource" || msg->action=="preaudit_metric_resource";
  const bool ground=msg->action=="preaudit_ground_rejected" || msg->action=="preaudit_ground_resource";
  if((ground || metric) && !primitive_observation_guard_.knownGroundEnabled())return;
  const auto hex=[](const std::string& s,std::size_t n) {
    return s.size()==n && s.find_first_not_of("0123456789abcdef")==std::string::npos;
  };
  if(!hex(msg->session,32) || !hex(msg->receiver_boot_id,32) || !hex(msg->payload_sha256,64))return;
  std::string event;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto raw=msg->trajectory.header.stamp.toNSec();
    if(metric || (ground && (resource || !repair_mode_ || probe_mode_))) {
      const auto it=ground_admission_.find(raw);
      if(it==ground_admission_.end())return;
      const auto original=it->second;
      if(original.goal!=terminal_goal_revision_ || original.epoch!=mode_epoch_ ||
          original.repair!=repair_mode_ || original.probe!=probe_mode_ ||
          original.context!=groundContextLocked() || ros::WallTime::now()>=original.deadline ||
          msg->planned_prefix_context!=original.planned.planned_prefix_context)return;
      auto received=msg->trajectory,published=original.planned.trajectory;
      received.header.seq=published.header.seq=0;if(received!=published)return;
      ground_admission_.erase(it);admission_pending_.erase(raw);
      final_vbc_no_progress_.pending.erase(raw);gcdf_feedback_pending_.erase(raw);
      safe_frontier_recovery_.pending.erase(raw); // unknown proof is not a direction counterexample
      if(ground_retry_goal_!=terminal_goal_revision_)return;
      ++plan_sequence_;plan_running_=false;waiting_for_cdf_=false;pending_batch_.reset();
      if(ground_resource_retries_>=2 && !task_first_research_) {
        geometry_resource_blocked_reason_=metric?"metric_proof_unavailable_hold":"ground_proof_unavailable_hold";
        ground_resource_blocked_=true;plan_requested_=false;plan_request_reason_=geometry_resource_blocked_reason_;
      } else {
        ++ground_resource_retries_;
        requestPlanLocked(metric ? (resource ? "metric_resource_retry" : "metric_candidate_retry") :
            (resource ? "ground_resource_retry" : "ground_candidate_retry"));
      }
      ROS_INFO_STREAM("[GEOMETRY_RESOURCE] raw="<<raw<<" action="<<msg->action
          <<" goal="<<terminal_goal_revision_<<" retries="<<ground_resource_retries_
          <<" requested="<<plan_requested_<<" reason="<<plan_request_reason_);
      return;
    }
    const auto sent=admission_pending_.find(raw);
    const auto failure=final_vbc_no_progress_.pending.find(raw);
    const auto proposal=safe_frontier_recovery_.pending.find(raw);
    if(!repair_prefix_binding_enabled_ || !safe_frontier_recovery_enabled_ || !repair_mode_ || probe_mode_ ||
        sent==admission_pending_.end() || failure==final_vbc_no_progress_.pending.end() ||
        proposal==safe_frontier_recovery_.pending.end())return;
    const auto& original=sent->second;
    if(original.epoch!=mode_epoch_ || original.observation_token!=latest_observation_token_ ||
        original.recovery_key!=safe_frontier_recovery_.key ||
        !prefixSelectionCurrentLocked(original.prefix) ||
        failure->second.mode_epoch!=mode_epoch_ || !final_vbc_no_progress_.progress.current(failure->second.ticket) ||
        proposal->second.generation!=safe_frontier_recovery_.generation ||
        msg->planned_prefix_context!=original.planned.planned_prefix_context)return;
    auto received=msg->trajectory, published=original.planned.trajectory;
    received.header.seq=published.header.seq=0; // publisher-local, not identity
    if(received!=published)return;
    admission_pending_.erase(sent);final_vbc_no_progress_.pending.erase(failure);gcdf_feedback_pending_.erase(raw);
    // An unexecuted scheduling timeout contains no geometric counterexample.
    // The unchanged task watchdog bounds retries; retain all direction evidence.
    const bool scheduling_only=task_first_research_ && msg->action=="preaudit_expired";
    bool advanced=false;
    if(scheduling_only) safe_frontier_recovery_.pending.erase(raw);
    else advanced=safe_frontier_recovery_.reject(raw,true);
    ++plan_sequence_;plan_running_=false;waiting_for_cdf_=false;pending_batch_.reset();
    requestPlanLocked(ground ? "known_ground_preaudit_retry" : "preaudit_expired_proposal_retry");event=plan_request_reason_;
    ROS_INFO_STREAM("[ADMISSION_DISPOSITION] raw_candidate_stamp_ns="<<raw<<" session="<<msg->session
        <<" sequence="<<msg->sequence<<" action="<<msg->action<<" audit_started=0 direction_advanced="<<advanced
        <<" attempts="<<safe_frontier_recovery_.attempts<<" segments="<<safe_frontier_recovery_.segments
        <<" requested="<<plan_requested_<<" reason="<<event);
  }
  publishSummary(event);
}

void LocalSparseSCPPlanner::finalVerificationCallback(const std_msgs::StringConstPtr& msg) {
  if (!msg) return;
  // Exact field tokens, not substring/nearest-time matching. No malformed or
  // positive outcome can become a final VBC failure.
  std::map<std::string, std::string> fields;
  std::istringstream input(msg->data);
  std::string word;
  while (input >> word) {
    const auto eq = word.find('=');
    if (eq != std::string::npos && !fields.emplace(word.substr(0,eq),word.substr(eq+1)).second) return;
  }
  if (fields["result"] == "safe" && fields["safety_gate"] == "vbc" && fields["committed"] == "1") {
    unsigned long long raw_safe = 0, execution = 0, audited = 0;
    if (!parseUnsignedToken(msg->data, "raw_candidate_stamp_ns", &raw_safe) ||
        !parseUnsignedToken(msg->data, "execution_stamp_ns", &execution)) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if((fields["verification_view"]=="full_horizon" ||
        (repair_goal_stopping_enabled_ && repair_mode_ && !probe_mode_ &&
         fields["verification_view"]=="repair_prefix_brake_hold")) &&
        parseUnsignedToken(msg->data,"audited_trajectory_stamp_ns",&audited)) {
      terminal_retention_.certify(raw_safe,audited,execution,terminalContextLocked());armTerminalLocked();
    }
    vbc_failure_direction_memory_.accepted(raw_safe,vbcFailureContextLocked());
    if (!repair_mode_ && !probe_mode_ && fields["verification_view"] == "full_horizon")
      normal_unknown_no_progress_.certify(raw_safe, execution, normalUnknownContextLocked());
    if (!repair_mode_ && !probe_mode_ && fields["verification_view"] == "full_horizon" &&
        parseUnsignedToken(msg->data,"audited_trajectory_stamp_ns",&audited) && audited &&
        normal_unknown_task_progress_.certify(raw_safe,execution,normalTaskProgressContextLocked()))
      ROS_INFO_STREAM("[NORMAL_TASK_CERTIFIED] raw_candidate_stamp_ns=" << raw_safe
          << " audited_trajectory_stamp_ns=" << audited << " execution_stamp_ns=" << execution
          << " context=" << normalTaskProgressContextLocked());
    if (safe_frontier_recovery_enabled_ && repair_mode_ && !probe_mode_)
      safe_frontier_recovery_.certify(raw_safe, execution);
    return;
  }
  const bool state_failure=fields["result"]=="timeout" && fields["safety_gate"]=="state_evidence";
  const bool restore_counterexample=fields["result"]=="protocol_invalid" &&
      fields["vbc_protocol_reason"]=="geometric_restore_path_counterexample";
  const bool restore_resource=fields["result"]=="protocol_invalid" &&
      fields["vbc_protocol_reason"]=="geometric_restore_proof_resource_exhausted";
  const bool restore_uncertified=fields["result"]=="protocol_invalid" &&
      (fields["vbc_protocol_reason"]=="geometric_restore_certificate_unavailable" || restore_counterexample || restore_resource);
  if((state_failure || restore_uncertified) && fields["committed"]=="0") {
    unsigned long long raw=0,audited=0;
    if(!parseUnsignedToken(msg->data,"raw_candidate_stamp_ns",&raw) || !raw ||
        !parseUnsignedToken(msg->data,"audited_trajectory_stamp_ns",&audited) || !audited)return;
    std::string event;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto pending=final_vbc_no_progress_.pending.find(raw);
      if(!repair_mode_ || probe_mode_ || pending==final_vbc_no_progress_.pending.end() ||
          pending->second.mode_epoch!=mode_epoch_ ||
          !final_vbc_no_progress_.progress.current(pending->second.ticket))return;
      PrimitiveObservationPoint resource_point;
      HandoffResidualFailure residual_failure;
      const bool residual=state_failure && fields["failure_kind"]=="geometric_certificate_state_changed" &&
          fields["handoff_residual_evidence"]!="none" && !fields["handoff_residual_evidence"].empty();
      if(residual && !parseHandoffResidualFailure(fields["handoff_residual_evidence"],raw,audited,
          mode_epoch_,latest_observation_token_,primitive_observation_margin_m_,&residual_failure))return;
      if(restore_resource && !parseProofResourceFailure(fields["proof_resource_evidence"],audited,
          mode_epoch_,latest_observation_token_,primitive_observation_margin_m_,&resource_point))return;
      final_vbc_no_progress_.pending.erase(pending);gcdf_feedback_pending_.erase(raw);
      ++plan_sequence_;plan_running_=false;waiting_for_cdf_=false;pending_batch_.reset();plan_requested_=false;
      const std::string evidence_key=std::to_string(mode_epoch_)+":"+safe_frontier_recovery_.key;
      const bool boundary_changed=residual &&
          (residual_failure.requires_reconstruction || residual_failure.initial_state_changed);
      if(residual && primitive_observation_guard_enabled_ && residual_failure.point.ordinary_path_guard) {
        const auto scope=std::to_string(mode_epoch_)+":"+latest_observation_token_;
        if(restore_path_guard_scope_!=scope){restore_path_guard_points_.clear();restore_path_guard_scope_=scope;}
        auto point=residual_failure.point;
        point.proposal_reserve_m=std::min(residual_failure.required_residual_m,
            primitive_observation_margin_m_-initial_shell_execution_margin_m_);
        auto old=std::find_if(restore_path_guard_points_.begin(),restore_path_guard_points_.end(),
            [&](const PrimitiveObservationPoint& p){return (p.point-point.point).lpNorm<Eigen::Infinity>()<=1e-5;});
        if(old!=restore_path_guard_points_.end())old->proposal_reserve_m=std::max(old->proposal_reserve_m,point.proposal_reserve_m);
        else if(restore_path_guard_points_.size()<32)restore_path_guard_points_.push_back(point);
        // Preserve point identity and proposal cost before the one original
        // state retry; waiting for a second rejection wastes that retry.
      }
      if(residual && !boundary_changed) {
        event=handleHandoffResidualLocked(raw,audited,residual_failure);
      } else if(restore_uncertified) {
        PrimitiveObservationPoint point;
        const bool ordinary_resource=restore_resource && resource_point.ordinary_path_guard;
        if(primitive_observation_guard_enabled_ && ((restore_counterexample &&
            parseOrdinaryPathCounterexample(fields["path_counterexample_evidence"],audited,
                primitive_observation_margin_m_,&point)) || ordinary_resource)) {
          if(ordinary_resource)point=resource_point;
          const auto scope=std::to_string(mode_epoch_)+":"+latest_observation_token_;
          if(restore_path_guard_scope_!=scope){restore_path_guard_points_.clear();restore_path_guard_scope_=scope;}
          const bool duplicate=std::any_of(restore_path_guard_points_.begin(),restore_path_guard_points_.end(),
              [&](const PrimitiveObservationPoint& p){return (p.point-point.point).lpNorm<Eigen::Infinity>()<=1e-5;});
          if(!duplicate && restore_path_guard_points_.size()<32) {
            restore_path_guard_points_.push_back(point);
            ROS_INFO_STREAM("[RESTORE_PATH_GUARD] raw_candidate_stamp_ns="<<raw
                <<" audited_trajectory_stamp_ns="<<audited<<" point="<<point.point.transpose()
                <<" floor_m="<<primitive_observation_margin_m_<<" guards="<<restore_path_guard_points_.size());
          }
        }
        // Spend the existing finite direction ledger without inventing a
        // collision witness or escalating a point's learned safety margin.
        const int attempts_before=safe_frontier_recovery_.attempts;
        const bool active_before=safe_frontier_recovery_.active;
        const bool advanced=safe_frontier_recovery_.reject(raw,restore_counterexample || restore_resource);
        ROS_INFO_STREAM("[RESTORE_REJECTION] raw_candidate_stamp_ns="<<raw
            <<" path_counterexample="<<restore_counterexample<<" proof_resource="<<restore_resource
            <<" active_before="<<active_before
            <<" attempts_before="<<attempts_before<<" active_after="<<safe_frontier_recovery_.active
            <<" attempts_after="<<safe_frontier_recovery_.attempts<<" direction_advanced="<<advanced);
        // Two current, distinct direct counterexamples or full-budget mandatory
        // proof failures change this steering proposal. Resource failure is a
        // scheduling reason, never a collision verdict or motion permission.
        // The first advance accounts for the rejected original direction;
        // an already-active advance has charged the next direction already.
        if ((restore_counterexample || restore_resource) && advanced && candidate_replacement_enabled_ &&
            requestQpReplacementLocked(restore_resource ? "geometric_restore_proof_resource_exhausted" :
                "geometric_restore_path_counterexample",raw,audited)) {
          candidate_replacement_trigger_direction_advanced_=active_before;
          event="restore_path_candidate_replacement_waiting";
        } else {
          requestPlanLocked("geometric_restore_uncertified_retry");
          event=plan_request_reason_;
        }
      } else {
        const std::string key=std::to_string(mode_epoch_)+":"+safe_frontier_recovery_.key;
        if(!state_evidence_retry_used_.count(key) && state_evidence_retry_used_.size()<64) {
          state_evidence_retry_used_[key]=true;
          state_evidence_retry_pending_=key;
          state_evidence_retry_wait_.reset();
          state_evidence_retry_deadline_=ros::WallTime::now()+ros::WallDuration(.5);
          state_residual_retry_epoch_=mode_epoch_;state_residual_retry_token_=latest_observation_token_;
          if(residual) {
            state_residual_retry_raw_=raw;state_residual_retry_audit_=audited;
            state_residual_retry_failure_=residual_failure;
          } else {
            safe_frontier_recovery_.pending.erase(raw);state_residual_retry_raw_=0;
          }
          event="state_evidence_waiting_fresh_state";
        } else if(residual) {
          event=handleHandoffResidualLocked(raw,audited,residual_failure);
        } else {
          safe_frontier_recovery_.pending.erase(raw);
          event="state_evidence_retry_exhausted_hold";
        }
      }
    }
    publishSummary(event);return;
  }
  if (fields["result"] != "unsafe" || fields["safety_gate"] != "vbc" || fields["committed"] != "0") return;
  unsigned long long raw = 0, audited = 0;
  try {
    for (const auto& key : {"raw_candidate_stamp_ns", "audited_trajectory_stamp_ns"}) {
      const auto& value = fields[key];
      if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) return;
    }
    raw = std::stoull(fields["raw_candidate_stamp_ns"]);
    audited = std::stoull(fields["audited_trajectory_stamp_ns"]);
  } catch (...) { return; }
  if (!raw || !audited) return;
  std::string event;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!repair_mode_ || probe_mode_) return;

    // Only an outstanding candidate from this mode can advance a point's
    // ladder. reject() below consumes this identity, so duplicate outcomes
    // cannot increment margins again (even after a target revision).
    const auto pending = final_vbc_no_progress_.pending.find(raw);
    if (pending == final_vbc_no_progress_.pending.end() ||
        pending->second.mode_epoch != mode_epoch_) return;
    const auto failure_ticket = pending->second.ticket;
    const auto live_ticket = final_vbc_no_progress_.progress.active;
    const int recovery_attempts_before = safe_frontier_recovery_.attempts;
    const int recovery_rejected_before = safe_frontier_recovery_.rejected;

    const bool recovery_direction_advanced = safe_frontier_recovery_enabled_ &&
        safe_frontier_recovery_.reject(raw);
    if (recovery_direction_advanced) {
      ++plan_sequence_; plan_running_ = false; waiting_for_cdf_ = false;
      pending_batch_.reset();
    }

    // Every rejected point owns its native CDF ladder, including points in
    // the active obligation. Never raise the shared observation base: another
    // point in the same obligation has not thereby been rejected by VBC.
    std::vector<Eigen::Vector3d> vbc_points;
    parseVbcEvidencePoints(fields["vbc_evidence"], &vbc_points);
    std::vector<PrimitiveObservationPoint> primitive_points;
    const bool primitive_evidence = primitive_observation_guard_enabled_ &&
        parsePrimitiveObservationEvidence(fields["vbc_evidence"], audited, &primitive_points);
    std::vector<Eigen::Vector3d> repeated_points;
    for (const auto& point : vbc_points) {
      const bool repeated = std::any_of(
          final_vbc_previous_points_.begin(), final_vbc_previous_points_.end(),
          [&](const Eigen::Vector3d& previous) {
            return (previous-point).lpNorm<Eigen::Infinity>() <= 1e-5;
          });
      // The exact collision point can belong to another still-live
      // obligation crossed by the trajectory for the current target. Target
      // continuity is already enforced by final_vbc_repeat_target_key_; do
      // not require the blocker point to belong to the target being steered.
      if (repeated) {
        repeated_points.push_back(point);
      }
    }
    final_vbc_previous_points_ = vbc_points;
    if (!vbc_points.empty()) {
      ++vbc_feedback_rejection_count_;
      bool updated = false;
      for (const auto& point : vbc_points) {
        const bool active = std::any_of(
            latest_vbc_obligation_points_.begin(), latest_vbc_obligation_points_.end(),
            [&](const Eigen::Vector3d& p) {
              return (p-point).lpNorm<Eigen::Infinity>() <= 1e-5;
            });
        if (active) ++vbc_feedback_matched_point_count_;
        double previous = visibility_obligation_cdf_margin_base_;
        double rollback = -1., ceiling = vbc_feedback_cdf_margin_max_;
        bool has_primitive_guard = false;
        PrimitiveObservationPoint primitive_guard;
        // Collapse ordinary timestep witnesses into one point-scoped VBC
        // witness. The query builder expands it over the complete horizon.
        for (auto it = repair_unknown_witnesses_.begin();
             it != repair_unknown_witnesses_.end();) {
          if ((it->point-point).lpNorm<Eigen::Infinity>() <= 1e-5) {
            if (std::isfinite(it->safety_margin))
              previous = std::max(previous, it->safety_margin);
            rollback = it->previous_margin;
            ceiling = std::min(ceiling, it->margin_ceiling);
            if (it->has_primitive_guard) {
              has_primitive_guard = true;
              primitive_guard = it->primitive_guard;
            }
            it = repair_unknown_witnesses_.erase(it);
          } else {
            ++it;
          }
        }
        if (repair_unknown_witnesses_.size() >= 32) {
          repair_witness_overflow_ = true;
          continue;
        }
        const double next = std::min(ceiling,
            std::max(visibility_obligation_cdf_margin_base_,
                     previous + vbc_feedback_cdf_margin_step_));
        if (next > previous && rollback < 0.) rollback = previous;
        if (primitive_evidence) {
          const auto found = std::find_if(primitive_points.begin(), primitive_points.end(),
              [&](const PrimitiveObservationPoint& p) {
                return (p.point-point).lpNorm<Eigen::Infinity>() <= 1e-5;
              });
          if (found != primitive_points.end()) {
            has_primitive_guard = true;
            primitive_guard = *found;
          }
        }
        RepairWitness witness{point, 1, next, true, rollback, ceiling};
        witness.has_primitive_guard = has_primitive_guard;
        witness.primitive_guard = primitive_guard;
        repair_unknown_witnesses_.push_back(std::move(witness));
        updated = true;
        ROS_WARN_STREAM("[LocalSparseSCPPlanner] exact VBC point feedback point=["
            << point.transpose() << "] active=" << active
            << " previous_margin=" << previous << " cdf_margin=" << next
            << " storage=point_scoped raw_candidate_stamp_ns=" << raw
            << " cdf_margin_units=configuration_space_model"
            << " primitive_guard=" << has_primitive_guard
            << " primitive_margin_m=" << primitive_observation_margin_m_
            << " audited_trajectory_stamp_ns=" << audited
            << " sweeping_link=" << primitive_guard.sweeping_link
            << " source_collision_name=" << primitive_guard.collision_name
            << " rejected_entry_s=" << primitive_guard.sweep_time_s);
      }
      if (updated || repair_witness_overflow_) {
        repair_witness_requires_reobserve_ = true;
        if (updated) ++vbc_feedback_repair_witness_count_;
        ++plan_sequence_;
        plan_running_ = false;
        waiting_for_cdf_ = false;
        pending_batch_.reset();
      }
    }
    const auto outcome = final_vbc_no_progress_.reject(raw, mode_epoch_);
    // Failure accounting belongs to the candidate's target/progress episode.
    // Its expiry must not strand valid safety feedback after that feedback
    // has invalidated an outstanding solve. The current gates decide whether
    // a new query is allowed; no failure or direction budget is reset here.
    const bool failure_counted = outcome != RepairNoProgress::Failure::Stale;
    std::uint64_t evidence_stamp=0;
    VbcFailureDirectionMemory::Points strict_points;
    Eigen::VectorXd measured;
    if (failure_counted && failureMemoryEvidence(fields["vbc_evidence"],&evidence_stamp,&strict_points) &&
        has_joint_state_ && extractMeasuredQ(latest_joint_state_,measured)) {
      const auto inherited=vbc_failure_direction_memory_.reject(raw,audited,evidence_stamp,
          current_query_stamp_.toNSec(),vbcFailureContextLocked(),strict_points,measured,ros::Time::now().toSec());
      for (const auto& p:inherited.repeated)
        if (std::none_of(repeated_points.begin(),repeated_points.end(),[&](const Eigen::Vector3d& old) {
              return (p-old).lpNorm<Eigen::Infinity>()<=1e-5;
            })) repeated_points.push_back(p);
      if (!inherited.repeated.empty())
        ROS_INFO_STREAM("[VBC_FAILURE_DIRECTION_INHERITED] previous_raw_candidate_stamp_ns="
            << inherited.previous_raw << " raw_candidate_stamp_ns=" << raw
            << " audited_trajectory_stamp_ns=" << audited
            << " query_stamp_ns=" << current_query_stamp_.toNSec()
            << " observation_token=" << latest_observation_token_
            << " repeated_points=" << inherited.repeated.size());
    }
    const bool replacement_trigger = failure_counted && candidate_replacement_enabled_ &&
        safe_frontier_recovery_enabled_ && repair_observation_phase_ &&
        latest_vbc_obligation_id_ >= 0 && !repeated_points.empty() &&
        !candidate_replacement_pending_ && !candidate_replacement_trigger_pending_ &&
        !safe_frontier_recovery_.blocked();
    if (replacement_trigger) {
      candidate_replacement_trigger_pending_ = true;
      candidate_replacement_trigger_reason_ = "final_vbc_repeat";
      candidate_replacement_trigger_observation_token_ = latest_observation_token_;
      candidate_replacement_trigger_direction_advanced_ = recovery_direction_advanced;
      candidate_replacement_trigger_raw_ = raw;
      candidate_replacement_trigger_obligation_id_ = latest_vbc_obligation_id_;
      candidate_replacement_epoch_ = mode_epoch_;
      candidate_replacement_trigger_deadline_ = ros::Time::now()+ros::Duration(.5);
      candidate_replacement_trigger_wall_deadline_ =
          ros::WallTime::now()+ros::WallDuration(.75);
      ++candidate_replacement_trigger_count_;
      std_msgs::String trigger;
      std::ostringstream s;
      s << "version=1 reason=final_vbc_repeat mode_epoch=" << mode_epoch_
        << " query_stamp_ns=" << current_query_stamp_.toNSec()
        << " query_ros_s=" << std::setprecision(17) << last_query_ros_time_.toSec()
        << " observation_token=" << latest_observation_token_
        << " obligation_id=" << latest_vbc_obligation_id_
        << " raw_candidate_stamp_ns=" << raw
        << " audited_trajectory_stamp_ns=" << audited
        << " repeated_points=";
      for (std::size_t i=0; i<repeated_points.size(); ++i) {
        if (i) s << ';';
        s << repeated_points[i].x() << ',' << repeated_points[i].y()
          << ',' << repeated_points[i].z();
      }
      trigger.data = s.str();
      candidate_replacement_trigger_pub_.publish(trigger);
      event = "repair_final_vbc_candidate_replacement_wait";
    } else if (candidate_replacement_enabled_ && safe_frontier_recovery_.blocked()) {
      event = "repair_candidate_replacement_exhausted_hold";
    } else {
      event = outcome == RepairNoProgress::Failure::Exhausted
          ? "repair_final_vbc_no_progress_hold" :
          (failure_counted ? "repair_final_vbc_retry" : "repair_final_vbc_feedback_replan");
    }
    if (outcome == RepairNoProgress::Failure::Exhausted ||
        event == "repair_candidate_replacement_exhausted_hold") {
      // Invalidate an already-started solve as well as future Bool replan
      // pulses. Never cancel/commandeer a trajectory owned by the tracker.
      ++plan_sequence_;
      plan_running_ = false;
      waiting_for_cdf_ = false;
      pending_batch_.reset();
    }
    requestPlanLocked(event);
    if (!plan_requested_ && !candidate_replacement_trigger_pending_)
      event = plan_request_reason_;
    ROS_WARN_STREAM("[FINAL_VBC_DISPOSITION] raw_candidate_stamp_ns=" << raw
        << " audited_trajectory_stamp_ns=" << audited
        << " accounting=" << (failure_counted ? "current" : "stale")
        << " candidate_target_revision=" << failure_ticket.revision
        << " live_target_revision=" << live_ticket.revision
        << " candidate_progress_epoch=" << failure_ticket.progress
        << " live_progress_epoch=" << live_ticket.progress
        << " recovery_attempts_before=" << recovery_attempts_before
        << " recovery_attempts_after=" << safe_frontier_recovery_.attempts
        << " recovery_rejected_before=" << recovery_rejected_before
        << " recovery_rejected_after=" << safe_frontier_recovery_.rejected
        << " recovery_generation=" << safe_frontier_recovery_.generation
        << " direction_advanced=" << recovery_direction_advanced
        << " requested=" << plan_requested_ << " reason=" << event);
    ROS_WARN_STREAM("[LocalSparseSCPPlanner] " << event << " raw_candidate_stamp_ns=" << raw
        << " audited_trajectory_stamp_ns=" << audited
        << " failures=" << final_vbc_no_progress_.progress.failures());
  }
  publishSummary(event);
}

void LocalSparseSCPPlanner::gcdfRejectionFeedbackCallback(
    const care_collision_cdf::CollisionCDFRejectionFeedbackConstPtr& msg) {
  if (!msg || msg->header.stamp.isZero() || msg->raw_candidate_stamp.isZero() ||
      msg->point_flat.size() % 3 || msg->point_flat.size() > 96 ||
      (!msg->overflow && msg->point_flat.empty()) ||
      !std::all_of(msg->point_flat.begin(), msg->point_flat.end(),
                   [](double v) { return std::isfinite(v); })) return;
  std::string event="repair_final_gcdf_witness_requery";
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = gcdf_feedback_pending_.find(msg->raw_candidate_stamp.toNSec());
    if (it == gcdf_feedback_pending_.end() || !repair_mode_ || probe_mode_ ||
        msg->header.frame_id != current_frame_id_ || it->second.identity.mode_epoch != mode_epoch_ ||
        it->second.observation_token!=latest_observation_token_ ||
        msg->header.stamp<it->second.raw_stamp || msg->header.stamp>ros::Time::now() ||
        (ros::Time::now()-it->second.raw_stamp).toSec()>cdf_wait_timeout_s_ ||
        ros::WallTime::now()>it->second.deadline ||
        !repair_no_progress_.current(it->second.identity.ticket)) return;
    gcdf_feedback_pending_.erase(it);  // one feedback per exact raw candidate
    const auto raw=msg->raw_candidate_stamp.toNSec();
    const auto accounting=final_vbc_no_progress_.reject(raw,mode_epoch_);
    if(accounting==RepairNoProgress::Failure::Stale)return;
    const bool direction_advanced=safe_frontier_recovery_enabled_ && safe_frontier_recovery_.reject(raw);
    std::vector<Eigen::Vector3d> repeated_points,feedback_points;
    repair_witness_overflow_ = repair_witness_overflow_ || msg->overflow;
    int active_obligation_feedback_ignored = 0;
    auto is_active_obligation_point = [&](const Eigen::Vector3d& point) {
      return std::any_of(
          latest_vbc_obligation_points_.begin(),
          latest_vbc_obligation_points_.end(),
          [&](const Eigen::Vector3d& obligation_point) {
            return (point - obligation_point).lpNorm<Eigen::Infinity>() <= 1e-5;
          });
    };
    // Carry only point identity. A handoff/braking knot has no general
    // one-to-one QP index: explicitly re-query every EXISTING hard-prefix
    // knot at its new q. Never reuse the executable's distance or gradient.
    for (std::size_t i = 0; i < msg->point_flat.size(); i += 3) {
      const Eigen::Vector3d p(msg->point_flat[i], msg->point_flat[i+1], msg->point_flat[i+2]);
      feedback_points.push_back(p);
      if(std::any_of(final_gcdf_previous_points_.begin(),final_gcdf_previous_points_.end(),
          [&](const Eigen::Vector3d& old){return (old-p).lpNorm<Eigen::Infinity>()<=1e-5;}))
        repeated_points.push_back(p);
      // The active visibility target already has a full executable-horizon
      // obligation guard. Promoting the same identity to the persistent
      // final-GCDF witness list would make its identity survive after the
      // target changes. Independent final-GCDF points remain persistent
      // witnesses.
      if (is_active_obligation_point(p)) {
        ++active_obligation_feedback_ignored;
        continue;
      }
      const bool present = std::any_of(
          repair_unknown_witnesses_.begin(), repair_unknown_witnesses_.end(),
          [&](const RepairWitness& w) {
            return (w.point-p).lpNorm<Eigen::Infinity>() <= 1e-5;
          });
      if (present) continue;
      if (repair_unknown_witnesses_.size() >= 32) {
        repair_witness_overflow_ = true;
        continue;
      }
      repair_unknown_witnesses_.push_back(
          RepairWitness{p, 1, cdf_safety_margin_});
    }
    final_gcdf_previous_points_=feedback_points;
    repair_witness_requires_reobserve_ = true;
    // Invalidate any solve started without the newly discovered identities.
    ++plan_sequence_;
    plan_running_ = false;
    waiting_for_cdf_ = false;
    pending_batch_.reset();
    if(!msg->overflow && !repeated_points.empty() && candidate_replacement_enabled_ &&
        safe_frontier_recovery_enabled_ && repair_observation_phase_ && latest_vbc_obligation_id_>=0 &&
        !candidate_replacement_pending_ && !candidate_replacement_trigger_pending_ && !safe_frontier_recovery_.blocked()) {
      candidate_replacement_trigger_pending_=true;
      candidate_replacement_trigger_reason_="final_gcdf_repeat";
      candidate_replacement_trigger_observation_token_=latest_observation_token_;
      candidate_replacement_trigger_direction_advanced_=direction_advanced;
      candidate_replacement_trigger_raw_=raw;
      candidate_replacement_trigger_obligation_id_=latest_vbc_obligation_id_;
      candidate_replacement_epoch_=mode_epoch_;
      candidate_replacement_trigger_deadline_=ros::Time::now()+ros::Duration(.5);
      candidate_replacement_trigger_wall_deadline_=ros::WallTime::now()+ros::WallDuration(.75);
      ++candidate_replacement_trigger_count_;
      std_msgs::String trigger;std::ostringstream s;
      s<<std::setprecision(17)<<"version=1 reason=final_gcdf_repeat mode_epoch="<<mode_epoch_
        <<" query_stamp_ns="<<current_query_stamp_.toNSec()<<" query_ros_s="<<last_query_ros_time_.toSec()
        <<" observation_token="<<latest_observation_token_<<" obligation_id="<<latest_vbc_obligation_id_
        <<" raw_candidate_stamp_ns="<<raw<<" audited_trajectory_stamp_ns="<<msg->header.stamp.toNSec()
        <<" repeated_points=";
      for(std::size_t i=0;i<repeated_points.size();++i) {
        if(i)s<<';';const auto& p=repeated_points[i];s<<p.x()<<','<<p.y()<<','<<p.z();
      }
      trigger.data=s.str();candidate_replacement_trigger_pub_.publish(trigger);
      event="repair_final_gcdf_candidate_replacement_wait";
    } else if(accounting==RepairNoProgress::Failure::Exhausted)event="repair_final_gcdf_no_progress_hold";
    requestPlanLocked(event);
    ROS_WARN_STREAM("[FINAL_GCDF_DISPOSITION] raw_candidate_stamp_ns="<<raw
        <<" audited_trajectory_stamp_ns="<<msg->header.stamp.toNSec()
        <<" observation_token="<<latest_observation_token_
        <<" failures="<<final_vbc_no_progress_.progress.failures()
        <<" attempts="<<safe_frontier_recovery_.attempts<<" direction_advanced="<<direction_advanced
        <<" requested="<<plan_requested_<<" reason="<<event);
    ROS_WARN_STREAM("[LocalSparseSCPPlanner] repair_final_gcdf_witness_requery raw_stamp_ns="
        << msg->raw_candidate_stamp.toNSec() << " audited_stamp_ns=" << msg->header.stamp.toNSec()
        << " witnesses=" << repair_unknown_witnesses_.size()
        << " active_obligation_feedback_ignored="
        << active_obligation_feedback_ignored
        << " overflow=" << repair_witness_overflow_);
  }
  publishSummary(event);
}

bool LocalSparseSCPPlanner::buildGeometricRestoreProposalLocked(
    const SparseSolveResult& failure, const std::string& observation_token,
    double prefix_s, Eigen::MatrixXd* q, Eigen::MatrixXd* u) {
  if(!q || !u || !has_joint_state_ || observation_token!=latest_observation_token_ ||
      !latest_single_waypoint_active_ || !has_single_waypoint_q_)return false;
  const auto now=ros::Time::now();
  const double age=(now-latest_joint_state_.header.stamp).toSec();
  Eigen::VectorXd measured;
  if(latest_joint_state_.header.stamp.isZero() || age<0. || age>.05 ||
      !extractMeasuredQ(latest_joint_state_,measured))return false;
  std::vector<PrimitiveObservationPoint> points;
  for(const auto& p:failure.geometric_restore_guard_points) {
    PrimitiveObservationPoint point;point.point=p;points.push_back(point);
  }
  auto merge_guard=[&](const PrimitiveObservationPoint& p) {
    auto old=std::find_if(points.begin(),points.end(),[&](const PrimitiveObservationPoint& point) {
      return (point.point-p.point).lpNorm<Eigen::Infinity>()<=1e-5;
    });
    if(old==points.end())points.push_back(p);
    else {
      old->ordinary_path_guard=old->ordinary_path_guard || p.ordinary_path_guard;
      old->proposal_reserve_m=std::max(old->proposal_reserve_m,p.proposal_reserve_m);
    }
  };
  for(const auto& p:failure.geometric_restore_path_guards)merge_guard(p);
  if(restore_path_guard_scope_==std::to_string(mode_epoch_)+":"+observation_token)
    for(const auto& p:restore_path_guard_points_)merge_guard(p);
  if(points.empty() || points.size()>32)return false;
  std::vector<PrimitiveObservationGuard::Row> rows;std::string error;
  if(!primitive_observation_guard_.linearize(measured,points,&rows,&error,true))return false;
  GeometricRestoreObjective objective;
  if(latest_single_waypoint_joint_mask_.size()==dof_)
    objective.observation_mask=latest_single_waypoint_joint_mask_;
  Eigen::VectorXd desired=latest_single_waypoint_q_;
  const bool guided=rolling_visibility_guidance_enabled_ && latest_visibility_guidance_.drive &&
      latest_visibility_guidance_.compatible(observation_token,latest_observation_region_,
          latest_observation_sensor_,latest_single_waypoint_joint_mask_,mode_epoch_,measured,now.toSec());
  if(guided) {
    desired=latest_visibility_guidance_.target;
    objective.visibility_gradient=latest_visibility_guidance_.gradient;
    objective.observation_mask=latest_visibility_guidance_.mask;
  }
  // A missing/expired guide contributes no claimed visibility gain. Geometry
  // may still propose a full-arm escape, with a small goal-direction bias.
  std::vector<Eigen::VectorXd> gradients;std::vector<double> distances;
  for(const auto& row:rows) {
    const auto point=std::find_if(points.begin(),points.end(),[&](const PrimitiveObservationPoint& p) {
      return (p.point-row.point).lpNorm<Eigen::Infinity>()<=1e-5;
    });
    if(point==points.end())return false;
    gradients.push_back(row.gradient);distances.push_back(row.distance_m);
    const bool fixed_start = std::any_of(failure.fixed_start_recovery_points.begin(),
        failure.fixed_start_recovery_points.end(), [&](const Eigen::Vector3d& p) {
          return (p-row.point).lpNorm<Eigen::Infinity>()<=1e-5;
        });
    // Only a deficient, explicitly declared point enters the escape proposal.
    // The fixed-start QP remains rejected; full native proof is still required.
    if(fixed_start && (!fixed_start_recovery_enabled_ ||
        row.distance_m<=initial_shell_execution_margin_m_+1e-6))return false;
    objective.point_floor_m.push_back((point->ordinary_path_guard && !fixed_start) ||
        row.distance_m>primitive_observation_margin_m_+1e-12 ?
        primitive_observation_margin_m_ : initial_shell_execution_margin_m_);
    objective.point_reserve_m.push_back(fixed_start ?
        primitive_observation_margin_m_-objective.point_floor_m.back()+.002 :
        point->ordinary_path_guard ? point->proposal_reserve_m : 0.);
    if(fixed_start && row.distance_m<primitive_observation_margin_m_ && row.gradient.norm()>1e-9)
      objective.visibility_gradient+=row.gradient.normalized()*
          (primitive_observation_margin_m_-row.distance_m);
  }
  if(primitive_observation_guard_.knownGroundEnabled()) {
    std::vector<PrimitiveObservationGuard::Row> ground_rows;
    if(!primitive_observation_guard_.linearizeKnownGround(measured,&ground_rows,&error))return false;
    for(const auto& row:ground_rows) {
      gradients.push_back(row.gradient);distances.push_back(row.distance_m);
      objective.point_floor_m.push_back(row.required_margin_m);
      objective.point_reserve_m.push_back(0.);
    }
  }
  if(!failure.fixed_start_recovery_points.empty()) {
    // Escape steering uses the deficient-point normals, not a new sensor goal.
    objective.visibility_gradient.setZero();
    for(const auto& row:rows)for(const auto& p:failure.fixed_start_recovery_points)
      if((p-row.point).lpNorm<Eigen::Infinity>()<=1e-5 && row.gradient.norm()>1e-9)
        objective.visibility_gradient+=row.gradient.normalized()*
            std::max(0.,primitive_observation_margin_m_-row.distance_m);
    objective.observation_mask.setOnes();desired=measured+objective.visibility_gradient;
  }
  const bool proposed=geometricRestoreProposal(measured,desired,gradients,distances,
      safe_frontier_recovery_.failed_directions,q_min_,q_max_,velocity_limits_,joint_position_margin_,
      num_intervals_,dt_,q,u,prefix_s,&objective);
  if(proposed)safe_frontier_recovery_.monitor_detour_observation=guided;
  ROS_INFO_STREAM("[GEOMETRIC_RESTORE_PROPOSAL] query_stamp_ns="<<current_query_stamp_.toNSec()
      <<" mode_epoch="<<mode_epoch_<<" observation_token="<<observation_token
      <<" guidance="<<(guided ? "compatible" : "geometry_only")<<" points="<<points.size()
      <<" prefix_s="<<prefix_s<<" attempts="<<safe_frontier_recovery_.attempts<<" proposed="<<proposed);
  const char* root=std::getenv("ROOT_OUT");
  if(root && *root) {
    std::ostringstream record;
    record<<"{\"schema\":1,\"executable\":false,\"proposed\":"<<(proposed ? "true" : "false")
        <<",\"query_stamp_ns\":\""<<current_query_stamp_.toNSec()<<"\",\"plan_seq\":"<<plan_sequence_
        <<",\"mode_epoch\":"<<mode_epoch_<<",\"observation_token\":";jsonString(record,observation_token);
    record<<",\"measured_stamp_ns\":\""<<latest_joint_state_.header.stamp.toNSec()
        <<"\",\"decision_ros_ns\":\""<<now.toNSec()<<"\",\"guidance_used\":"<<(guided ? "true" : "false")
        <<",\"guidance_sequence\":"<<latest_visibility_guidance_.sequence<<",\"prefix_s\":";jsonNumber(record,prefix_s);
    record<<",\"attempts\":"<<safe_frontier_recovery_.attempts<<",\"segments\":"<<safe_frontier_recovery_.segments
        <<",\"q_measured\":";jsonVector(record,measured);
    record<<",\"measured_velocity\":";jsonVector(record,latest_joint_state_.velocity);
    record<<",\"actuator_command\":";jsonVector(record,latest_executed_command_);
    record<<",\"published_q_vis\":";jsonVector(record,latest_single_waypoint_q_);
    record<<",\"desired\":";jsonVector(record,desired);
    record<<",\"visibility_gradient\":";jsonVector(record,objective.visibility_gradient);
    record<<",\"observation_mask\":";jsonVector(record,objective.observation_mask);
    record<<",\"rows\":[";
    for(std::size_t i=0;i<rows.size();++i) {
      if(i)record<<',';
      record<<"{\"point\":";jsonVector(record,rows[i].point);
      record<<",\"gradient\":";jsonVector(record,rows[i].gradient);
      record<<",\"distance_m\":";jsonNumber(record,distances[i]);
      record<<",\"floor_m\":";jsonNumber(record,objective.point_floor_m[i]);
      record<<",\"reserve_m\":";jsonNumber(record,objective.point_reserve_m[i]);record<<'}';
    }
    record<<"],\"failed_directions\":[";
    for(std::size_t i=0;i<safe_frontier_recovery_.failed_directions.size();++i) {
      if(i)record<<',';
      jsonVector(record,safe_frontier_recovery_.failed_directions[i]);
    }
    record<<']';
    if(proposed){record<<",\"q\":";jsonTrajectory(record,*q);record<<",\"u\":";jsonTrajectory(record,*u);}
    record<<"}\n";
    const std::string latest=std::string(root)+"/recovery_proposal_latest.json",temporary=latest+".tmp";
    std::ofstream file(temporary);file<<record.str();file.close();
    if(file && std::rename(temporary.c_str(),latest.c_str())==0) {
      const auto scope=std::to_string(mode_epoch_)+":"+observation_token+":"+
          std::to_string(safe_frontier_recovery_.attempts);
      if(!geometric_restore_snapshot_scopes_.count(scope) && geometric_restore_snapshot_scopes_.size()<64) {
        std::ofstream first(std::string(root)+"/recovery_proposal_"+
            std::to_string(geometric_restore_snapshot_scopes_.size())+".json");
        first<<record.str();first.close();if(first)geometric_restore_snapshot_scopes_.insert(scope);
      }
    } else ROS_WARN_STREAM("[LocalSparseSCPPlanner] incomplete recovery proposal snapshot="<<temporary);
  }
  return proposed;
}

std::string LocalSparseSCPPlanner::handleHandoffResidualLocked(unsigned long long raw,
    unsigned long long audit,const HandoffResidualFailure& failure) {
  const auto pending=safe_frontier_recovery_.pending.find(raw);
  if(!safe_frontier_recovery_enabled_ || pending==safe_frontier_recovery_.pending.end() ||
      pending->second.generation!=safe_frontier_recovery_.generation)
    return "handoff_residual_context_unavailable_hold";
  if(executable_task_research_ && failure.initial_state_changed) {
    // The audited curve was never committed. A changed start invalidates that
    // curve's certificate, not its sensor direction. Rebuild from live q/v;
    // fresh GCDF, VBC and the unchanged live residual check still authorize it.
    safe_frontier_recovery_.pending.erase(pending);
    safe_frontier_recovery_.last_advance_reason="handoff_state_rebuild";
    ROS_INFO_STREAM("[HANDOFF_REBUILD] raw_candidate_stamp_ns="<<raw
        <<" audited_trajectory_stamp_ns="<<audit<<" required_m="<<failure.required_residual_m
        <<" certified_m="<<failure.certified_residual_m
        <<" attempts="<<safe_frontier_recovery_.attempts<<" direction_advanced=0");
    requestPlanLocked("handoff_state_rebuild");
    return plan_request_reason_;
  }
  const auto scope=std::to_string(mode_epoch_)+":"+latest_observation_token_;
  if(primitive_observation_guard_enabled_ && failure.point.ordinary_path_guard) {
    if(restore_path_guard_scope_!=scope){restore_path_guard_scope_=scope;restore_path_guard_points_.clear();}
    auto point=failure.point;
    // A soft, point-scoped proposal target. Hard safety floors and full audits remain authoritative.
    if(!failure.requires_reconstruction)point.proposal_reserve_m=std::min(failure.required_residual_m,
        primitive_observation_margin_m_-initial_shell_execution_margin_m_);
    auto old=std::find_if(restore_path_guard_points_.begin(),restore_path_guard_points_.end(),
        [&](const PrimitiveObservationPoint& p){return (p.point-point.point).lpNorm<Eigen::Infinity>()<=1e-5;});
    if(old==restore_path_guard_points_.end()) {
      if(restore_path_guard_points_.size()<32)restore_path_guard_points_.push_back(point);
    } else old->proposal_reserve_m=std::max(old->proposal_reserve_m,point.proposal_reserve_m);
  }
  const bool was_active=safe_frontier_recovery_.active;
  const bool advanced=safe_frontier_recovery_.reject(raw,true);
  ROS_INFO_STREAM("[HANDOFF_RESIDUAL] raw_candidate_stamp_ns="<<raw<<" audited_trajectory_stamp_ns="<<audit
      <<" required_m="<<failure.required_residual_m<<" reserve_m="<<failure.certified_residual_m
      <<" reconstruct="<<failure.requires_reconstruction<<" direction_advanced="<<advanced
      <<" initial_state_changed="<<failure.initial_state_changed
      <<" scheduling_only=1"
      <<" attempts="<<safe_frontier_recovery_.attempts<<" segments="<<safe_frontier_recovery_.segments);
  if(advanced && candidate_replacement_enabled_ && requestQpReplacementLocked(
      "geometric_restore_handoff_residual_insufficient",raw,audit)) {
    candidate_replacement_trigger_direction_advanced_=was_active;
    return "handoff_residual_candidate_replacement_waiting";
  }
  if(safe_frontier_recovery_.blocked() || safe_frontier_recovery_.segments>=SafeFrontierRecovery::max_segments)
    return "handoff_residual_recovery_budget_exhausted_hold";
  requestPlanLocked("handoff_residual_proposal_retry");return plan_request_reason_;
}

bool LocalSparseSCPPlanner::requestQpReplacementLocked(const std::string& reason,
    unsigned long long rejected_raw, unsigned long long rejected_audit) {
  const bool path_counterexample=reason=="geometric_restore_path_counterexample" ||
      reason=="geometric_restore_proof_resource_exhausted" ||
      reason=="geometric_restore_handoff_residual_insufficient";
  if (path_counterexample ? (!rejected_raw || !rejected_audit) :
      (rejected_raw || rejected_audit)) return false;
  if (!safe_frontier_recovery_enabled_ || latest_vbc_obligation_id_ < 0 ||
      latest_observation_token_.find("care_obs_v1_") != 0 ||
      latest_vbc_obligation_points_.empty() || safe_frontier_recovery_.blocked()) return false;
  candidate_replacement_trigger_pending_ = true;
  candidate_replacement_trigger_reason_ = reason;
  candidate_replacement_trigger_observation_token_ = latest_observation_token_;
  candidate_replacement_trigger_direction_advanced_ = false;
  candidate_replacement_trigger_raw_ = rejected_raw;
  candidate_replacement_trigger_obligation_id_ = latest_vbc_obligation_id_;
  candidate_replacement_epoch_ = mode_epoch_;
  // QP failure can coincide with a Python active-set projection callback.
  // Keep the exact token/query handshake bounded, but allow the replacement
  // runtime enough time to restore and lock the still-live original owner.
  candidate_replacement_trigger_deadline_ = ros::Time::now()+ros::Duration(1.0);
  candidate_replacement_trigger_wall_deadline_ = ros::WallTime::now()+ros::WallDuration(1.25);
  ++candidate_replacement_trigger_count_;
  std_msgs::String trigger;
  std::ostringstream s;
  s << std::setprecision(17) << "version=1 reason=" << reason
    << " mode_epoch=" << mode_epoch_ << " query_stamp_ns=" << current_query_stamp_.toNSec()
    << " query_ros_s=" << last_query_ros_time_.toSec()
    << " observation_token=" << latest_observation_token_
    << " obligation_id=" << latest_vbc_obligation_id_
    << " raw_candidate_stamp_ns=" << rejected_raw
    << " audited_trajectory_stamp_ns=" << rejected_audit << " repeated_points=";
  for (std::size_t i=0; i<latest_vbc_obligation_points_.size(); ++i) {
    const auto& p=latest_vbc_obligation_points_[i];
    if (i) s << ';';
    s << p.x() << ',' << p.y() << ',' << p.z();
  }
  trigger.data=s.str(); candidate_replacement_trigger_pub_.publish(trigger);
  plan_requested_=false;
  return true;
}

bool LocalSparseSCPPlanner::qpReplacementRetryCurrentLocked() const {
  return candidate_replacement_enabled_ && safe_frontier_recovery_enabled_ &&
      repair_mode_ && !probe_mode_ && repair_observation_phase_ &&
      !repair_no_progress_.blocked() && !final_vbc_no_progress_.progress.blocked() &&
      safe_frontier_recovery_.canReserveReplacement() &&
      candidate_replacement_retry_context_.blocks(repairTargetKeyLocked(),
          latest_observation_token_, mode_epoch_, repair_no_progress_.active.progress) &&
      repair_candidate_failure_.blocks(repairTargetKeyLocked(),
          latest_observation_token_, mode_epoch_, repair_no_progress_.active.progress);
}

void LocalSparseSCPPlanner::requestPlanLocked(
    const std::string& reason) {
  if(primitive_observation_guard_.knownGroundEnabled() && ground_resource_blocked_ &&
      ground_retry_goal_==terminal_goal_revision_) {
    plan_requested_=false;plan_request_reason_=geometry_resource_blocked_reason_;return;
  }
  if(primitive_observation_guard_.knownGroundEnabled() && known_ground_hold_.blocked()) {
    if(known_ground_hold_.goal==terminal_goal_revision_ && known_ground_hold_.context==groundContextLocked()) {
      plan_requested_=false;plan_request_reason_=known_ground_hold_.reason;return;
    }
    known_ground_hold_.cancel("ground_context_changed");
  }
  if(fixed_start_clearance_hold_.armed) {
    plan_requested_=false;plan_request_reason_="fixed_start_unknown_clearance_hold";return;
  }
  const bool goal_observation_wakeup=repair_goal_stopping_enabled_ &&
      (repair_mode_ || repair_goal_handoff_execution_==terminal_retention_.armed) &&
      (reason.find("visibility")!=std::string::npos ||
       reason=="vbc_obligation_points_changed" || reason=="repair_prefix_selection_changed" ||
       reason.find("candidate_replacement")!=std::string::npos);
  const bool optional=reason=="smooth_handoff_normal" || reason=="normal_trajectory_complete" ||
      goal_observation_wakeup ||
      (repair_goal_stopping_enabled_ && repair_mode_ &&
       (reason=="smooth_handoff_repair" || reason=="repair_trajectory_complete"));
  if(optional && retainTerminalLocked()) {
    ++terminal_retention_.suppressed_count;
    ROS_INFO_STREAM("[NORMAL_TERMINAL] suppressed reason="<<reason<<" execution_stamp_ns="<<terminal_retention_.armed
        <<" count="<<terminal_retention_.suppressed_count);
    return;
  }
  terminal_retention_.replan(reason);
  // A different plan request supersedes any failed-QP retry ticket, including
  // reference/mode/progress changes. Its delayed delivery cannot preempt work.
  qp_retry_token_.clear();
  if (execution_response_monitor_.blocked) {
    plan_requested_=false; plan_request_reason_="execution_command_no_response_hold"; return;
  }
  selectRepairTargetLocked();
  if(!state_evidence_retry_pending_.empty()) {
    const auto key=std::to_string(mode_epoch_)+":"+safe_frontier_recovery_.key;
    const auto pending=safe_frontier_recovery_.pending.find(state_residual_retry_raw_);
    const bool proposal_current=!state_residual_retry_raw_ ||
        (pending!=safe_frontier_recovery_.pending.end() &&
         pending->second.generation==safe_frontier_recovery_.generation);
    if(repair_mode_ && !probe_mode_ && state_evidence_retry_pending_==key &&
        state_residual_retry_epoch_==mode_epoch_ && state_residual_retry_token_==latest_observation_token_ &&
        proposal_current) {
      // Completion/progress callbacks may request work during this one wait.
      // Keep one deferred request; only the matching state window or its
      // original deadline selects the next action.
      plan_requested_=true;plan_request_reason_="state_evidence_waiting_fresh_state";return;
    }
    if(state_residual_retry_raw_)safe_frontier_recovery_.pending.erase(state_residual_retry_raw_);
    state_residual_retry_raw_=0;state_evidence_retry_pending_.clear();state_evidence_retry_wait_.reset();
  }
  if (candidate_replacement_retry_pending_ && !qpReplacementRetryCurrentLocked())
    candidate_replacement_retry_pending_ = false;
  if (candidate_replacement_pending_ || candidate_replacement_trigger_pending_) {
    plan_requested_=false;
    plan_request_reason_=candidate_replacement_trigger_pending_
        ? "repair_candidate_replacement_trigger_wait" : "repair_candidate_replacement_wait";
    return;
  }
  if (safe_frontier_recovery_enabled_ && repair_mode_ && !probe_mode_ &&
      safe_frontier_recovery_.blocked()) {
    plan_requested_ = false;
    plan_request_reason_ = candidate_replacement_enabled_ && latest_vbc_obligation_id_ >= 0
        ? "repair_candidate_replacement_exhausted_hold"
        : "repair_safe_stall_reselect_hold";
    return;
  }
  if (candidate_replacement_retry_pending_) {
    plan_requested_ = true;
    plan_request_reason_ = "repair_qp_replacement_refresh";
    return;
  }
  if (candidate_replacement_enabled_ && repair_mode_ && !probe_mode_ &&
      repair_candidate_failure_.blocks(
          repairTargetKeyLocked(), latest_observation_token_, mode_epoch_,
          repair_no_progress_.active.progress) &&
      geometric_restore_scope_!=std::to_string(mode_epoch_)+":"+safe_frontier_recovery_.key) {
    plan_requested_ = false;
    plan_request_reason_ = "repair_failed_candidate_hold";
    return;
  }
  if (repair_mode_ && final_vbc_no_progress_.progress.blocked()) {
    plan_requested_ = false;
    plan_request_reason_ = "repair_final_vbc_no_progress_hold";
    return;
  }
  if (repair_mode_ && repair_no_progress_.blocked()) {
    plan_requested_ = false;
    plan_request_reason_ = "repair_qp_no_progress_hold";
    return;
  }
  if (!repair_mode_ && task_no_progress_.blocked()) {
    plan_requested_ = false;
    plan_request_reason_ = "qp_no_progress_hold";
    return;
  }
  plan_requested_ = true;
  plan_request_reason_ = reason;
}

bool LocalSparseSCPPlanner::startPlan() {
  sensor_msgs::JointState joint_state;
  trajectory_msgs::JointTrajectory reference;
  std::vector<DeadlineWaypoint> schedule;
  bool single_waypoint_active = false;
  bool has_single_waypoint_q = false;
  Eigen::VectorXd single_waypoint_q;
  Eigen::VectorXd single_waypoint_joint_mask;
  Eigen::VectorXd previous_command;
  FrontierObjective frontier;
  bool repair = false;
  bool observation_phase = true;
  bool probe = false;
  std::string reason;
  unsigned long long start_mode_epoch = 0;
  RepairPrefixSelection start_prefix_selection;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (execution_response_monitor_.blocked || !state_evidence_retry_pending_.empty() ||
        candidate_replacement_pending_ || candidate_replacement_trigger_pending_ ||
        plan_running_ || !plan_requested_) return false;
    if (repair_prefix_binding_enabled_ && repair_mode_ && !probe_mode_) {
      if (!prefixSelectionCompatible(latest_prefix_selection_))return false;
      start_prefix_selection=latest_prefix_selection_;
    }
    plan_candidate_replacement_refresh_only_ = candidate_replacement_retry_pending_;
    candidate_replacement_retry_pending_ = false;
    if (plan_candidate_replacement_refresh_only_) {
      const double age = (ros::Time::now()-latest_joint_state_.header.stamp).toSec();
      if (!qpReplacementRetryCurrentLocked() || !has_joint_state_ ||
          latest_joint_state_.header.stamp.isZero() || age < 0. || age > .2) {
        plan_candidate_replacement_refresh_only_ = false;
        plan_requested_ = false;
        ROS_WARN("[LocalSparseSCPPlanner] replacement refresh stopped: changed context or stale measured state");
        return false;
      }
    }
    if (repair_mode_ && (repair_no_progress_.blocked() || final_vbc_no_progress_.progress.blocked() ||
        (safe_frontier_recovery_enabled_ && !probe_mode_ && safe_frontier_recovery_.blocked()))) {
      plan_requested_ = false;
      return false;
    }
    if (!has_joint_state_ || !has_reference_) {
      ROS_WARN_THROTTLE(
          1.0,
          "[LocalSparseSCPPlanner] waiting for joint state/reference");
      return false;
    }

    joint_state = latest_joint_state_;
    reference = latest_reference_;
    plan_terminal_context_=terminalContextLocked();
    schedule = latest_schedule_;
    single_waypoint_active = latest_single_waypoint_active_;
    has_single_waypoint_q = has_single_waypoint_q_;
    single_waypoint_q = latest_single_waypoint_q_;
    single_waypoint_joint_mask = latest_single_waypoint_joint_mask_;
    plan_observation_token_ = repair_mode_ && schedule.empty() &&
        single_waypoint_active && has_single_waypoint_q
        ? latest_observation_token_ : "none";
    previous_command = latest_executed_command_;
    if (previous_command.size() != dof_)
      previous_command = Eigen::VectorXd::Zero(dof_);
    frontier = latest_frontier_;
    Eigen::VectorXd guidance_measured;
    if (rolling_visibility_guidance_enabled_ && extractMeasuredQ(joint_state, guidance_measured) &&
        latest_visibility_guidance_.compatible(latest_observation_token_, latest_observation_region_,
            latest_observation_sensor_, latest_single_waypoint_joint_mask_, mode_epoch_,
            guidance_measured, ros::Time::now().toSec()) && latest_visibility_guidance_.drive) {
      frontier.q=latest_visibility_guidance_.target;
      frontier.joint_mask=latest_visibility_guidance_.mask;
      frontier.visibility_gradient=latest_visibility_guidance_.gradient;
    }
    safe_frontier_recovery_.monitor_detour_observation=frontier.visibility_gradient.size()==7;
    if (safe_frontier_recovery_enabled_ && repair_mode_ && !probe_mode_ &&
        repair_observation_phase_ && frontier.active) {
      frontier.recovery_attempt = safe_frontier_recovery_.stage();
      frontier.recovery_target = safe_frontier_recovery_.target;
      frontier.failed_directions = safe_frontier_recovery_.failed_directions;
    }
    repair = repair_mode_;
    probe = probe_mode_;
    start_mode_epoch = mode_epoch_;
    plan_repair_ticket_ = repair_no_progress_.active;
    if (repair && (repair_witness_mode_epoch_ != start_mode_epoch ||
                   repair_witness_ticket_.target != plan_repair_ticket_.target ||
                   repair_witness_ticket_.revision != plan_repair_ticket_.revision)) {
      if (repair_witness_mode_epoch_ == start_mode_epoch) {
        // q_vis reselection is not new safety evidence. Retain confirmed
        // point-scoped VBC guards (including their raised margins) across a
        // same-mode target refresh; normal freshness queries still apply.
        repair_unknown_witnesses_.erase(std::remove_if(
            repair_unknown_witnesses_.begin(), repair_unknown_witnesses_.end(),
            [](const RepairWitness& w) { return !w.vbc_feedback; }),
            repair_unknown_witnesses_.end());
      } else {
        repair_unknown_witnesses_.clear();
      }
      repair_witness_overflow_ = false;
      repair_witness_requires_reobserve_ = false;
      repair_observation_phase_ = true;
      repair_witness_ticket_ = plan_repair_ticket_;
      repair_witness_mode_epoch_ = start_mode_epoch;
    }
    plan_repair_mode_ = repair;
    plan_repair_observation_phase_ = repair_observation_phase_;
    observation_phase = plan_repair_observation_phase_;
    if (repair && !observation_phase) plan_observation_token_ = "none";
    plan_probe_mode_ = probe;
    plan_mode_epoch_ = start_mode_epoch;
    reason = plan_request_reason_;

    plan_requested_ = false;
    plan_running_ = true;
    waiting_for_cdf_ = false;
    pending_batch_.reset();
    ++plan_sequence_;
  }

  // C4.7/C4.9 blocker-aware acquisition intentionally publishes only the
  // currently selected persistent q_vis through the legacy single-waypoint
  // topics; its multi-deadline schedule topic stays empty.  The C5.4 local
  // planner accepts both interfaces. In REPAIR, fall back to a synthetic
  // terminal-horizon obligation for that active q_vis.
  if (!repair || !observation_phase) {
    schedule.clear();
    frontier.active = false;
    frontier.frontier_weight_scale = 0.0;
    frontier.qvis_weight_scale = 1.0;
  } else if (schedule.empty() &&
             single_waypoint_active &&
             has_single_waypoint_q &&
             single_waypoint_q.size() == dof_ &&
             single_waypoint_joint_mask.size() == dof_) {
    DeadlineWaypoint wp;
    wp.id = -1;
    wp.terminal_objective = true;
    wp.q = single_waypoint_q;
    wp.joint_mask = single_waypoint_joint_mask;
    schedule.push_back(std::move(wp));
  }

  // A REPAIR plan without an active visibility obligation has no steering
  // objective. Solving it would produce a no-op hold candidate, which then
  // stays "fresh" on the predicted-trajectory topic and prevents the VBC
  // selector from falling back to the task trajectory to discover the next
  // blocker. Treat this as a normal waiting state instead: request an immediate
  // task-trajectory bootstrap and publish no candidate until q_vis arrives.
  if (repair && observation_phase && schedule.empty()) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      plan_running_ = false;
      waiting_for_cdf_ = false;
      plan_repair_mode_ = true;
      plan_initialization_mode_ = "waiting_visibility_obligation";
      plan_schedule_.clear();
      last_plan_finish_time_ = ros::Time::now();
    }
    std_msgs::Bool bootstrap_msg;
    bootstrap_msg.data = true;
    force_vbc_bootstrap_pub_.publish(bootstrap_msg);
    publishSummary("waiting_visibility_obligation");
    ROS_INFO_THROTTLE(
        0.5,
        "[LocalSparseSCPPlanner] REPAIR waiting for visibility obligation; "
        "forcing task-trajectory VBC bootstrap");
    return false;
  }

  // A real visibility target is available, so normal predicted-trajectory VBC
  // selection may resume.
  std_msgs::Bool bootstrap_msg;
  bootstrap_msg.data = false;
  force_vbc_bootstrap_pub_.publish(bootstrap_msg);

  Eigen::VectorXd q_current;
  if (!extractMeasuredQ(joint_state, q_current)) {
    abortPlan("joint_state_decode_failed");
    return false;
  }

  const bool rolling_task = std::getenv("CARE_ROLLING_TASK_OBSERVATION") &&
      std::string(std::getenv("CARE_ROLLING_TASK_OBSERVATION")) == "true";
  if (repair_executable_view_enabled_ && (!probe || rolling_task)) {
    // NORMAL optimizes the same acceleration ramp that will be certified.
    // Its boundary is the measured velocity, not an instantaneous setpoint.
    previous_command.resize(dof_);
    for (int j = 0; j < dof_; ++j) {
      const auto it = std::find(joint_state.name.begin(), joint_state.name.end(), joint_names_[j]);
      const auto index = std::distance(joint_state.name.begin(), it);
      if (it == joint_state.name.end() || index >= static_cast<long>(joint_state.velocity.size()) ||
          !std::isfinite(joint_state.velocity[index])) {
        abortPlan("normal_measured_velocity_missing");
        return false;
      }
      previous_command[j] = joint_state.velocity[index];
    }
  }

  if (rolling_task && !repair && !reference.points.empty()) {
    std::vector<int> goal_mapping;
    if (!buildTrajectoryMapping(reference, goal_mapping)) {
      abortPlan("rolling_task_goal_mapping_failed"); return false;
    }
    Eigen::VectorXd goal(dof_);
    for (int j=0;j<dof_;++j) goal[j]=reference.points.back().positions[goal_mapping[j]];
    double duration=2.;
    for (int j=0;j<dof_;++j) {
      const double delta=std::abs(goal[j]-q_current[j]);
      duration=std::max(duration,1.9*delta/velocity_limits_[j]);
      duration=std::max(duration,std::sqrt(6.*delta/repair_executable_settings_.acceleration[j]));
    }
    const auto refreshed=arm_trajectory::JointTrajectory::makeQuinticBoundaryVelocityAcceleration(
        q_current,previous_command,Eigen::VectorXd::Zero(dof_),goal,
        Eigen::VectorXd::Zero(dof_),Eigen::VectorXd::Zero(dof_),duration,.02);
    reference.header.stamp=ros::Time::now(); reference.joint_names=joint_names_;reference.points.clear();
    for (std::size_t k=0;k<refreshed.size();++k) {
      trajectory_msgs::JointTrajectoryPoint p;
      const auto a=refreshed.positions()[k],b=refreshed.velocities()[k],c=refreshed.accelerations()[k];
      p.positions.assign(a.data(),a.data()+dof_);p.velocities.assign(b.data(),b.data()+dof_);
      p.accelerations.assign(c.data(),c.data()+dof_);p.time_from_start=ros::Duration(refreshed.times()[k]);
      reference.points.push_back(p);
    }
    ROS_INFO_STREAM("[ROLLING_TASK_REFERENCE] goal_revision="<<terminal_goal_revision_
        <<" probe="<<probe<<" duration_s="<<duration<<" measured_stamp_ns="<<joint_state.header.stamp.toNSec());
  }
  Eigen::MatrixXd q_ref, u_ref, q_init, u_init;
  if (!buildReferenceHorizon(
          q_current, reference, ros::Time::now(), probe,
          q_ref, u_ref, q_init, u_init)) {
    abortPlan("reference_horizon_failed");
    return false;
  }
  if (repair_executable_view_enabled_ && !repair && !probe &&
      !buildNormalMotionSeed(q_current, previous_command, q_ref, q_init, u_init)) {
    abortPlan("normal_motion_seed_failed");
    return false;
  }

  std::string initialization_mode =
      probe ? "probe_short_horizon_hold_tail" : "task_reference";
  if (repair && repair_hold_initialization_enabled_) {
    for (int k = 0; k <= num_intervals_; ++k) {
      q_init.col(k) = q_current;
    }
    u_init.setZero();
    initialization_mode = "repair_hold";
    if (!observation_phase) initialization_mode = "repair_safety_retreat_hold";
  }
  if(repair_executable_view_enabled_ && repair && !probe) {
    const Eigen::MatrixXd seed_target=q_init;
    if(!buildNormalMotionSeed(q_current,previous_command,seed_target,q_init,u_init)) {
      abortPlan("repair_motion_seed_failed");return false;
    }
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (start_mode_epoch != mode_epoch_ ||
        (repair_prefix_binding_enabled_ && repair && !probe &&
         !prefixSelectionCurrentLocked(start_prefix_selection)) ||
        (repair && !repair_no_progress_.targetCurrent(plan_repair_ticket_))) {
      plan_running_ = false;
      return false;
    }
    int warm_iterations = 0;
    if (repair && !probe && observation_phase && repair_executable_view_enabled_ &&
        takeRepairWarmStartLocked(q_current, previous_command, q_init, u_init, warm_iterations)) {
      initialization_mode = "repair_progress_shape_rebuilt";
      ROS_INFO_STREAM("[REPAIR_WARM_START] plan_seq=" << plan_sequence_
          << " charged_iterations=" << warm_iterations << " max_iterations=" << max_scp_iterations_
          << " fresh_query_required=1");
    }
    plan_q_current_ = q_current;
    plan_previous_command_ = previous_command;
    plan_q_ref_ = q_ref;
    plan_u_ref_ = u_ref;
    plan_q_bar_ = q_init;
    plan_previous_q_bar_ = q_init;
    plan_previous_u_bar_ = u_init;
    plan_has_hard_solution_ = false;
    plan_qp_backtrack_policy_.reset();
    plan_normal_reseed_used_ = false;
    plan_repair_conflict_reseed_ = RepairConflictReseed{};
    plan_normal_retained_q_.resize(0,0);plan_normal_retained_u_.resize(0,0);
    plan_executable_iterates_ = RepairExecutableIterates{};
    plan_u_bar_ = u_init;
    plan_schedule_ = schedule;
    plan_frontier_ = frontier;
    plan_repair_mode_ = repair;
    plan_probe_mode_ = probe;
    plan_prefix_selection_=start_prefix_selection;
    plan_initialization_mode_ = initialization_mode;
    plan_mode_epoch_ = mode_epoch_;
    plan_probe_feasibility_restore_attempts_ = 0;
    plan_probe_restore_pending_hard_recheck_ = false;
    scp_iteration_ = warm_iterations;
    trust_radius_ = trust_region_initial_;
    plan_cdf_slack_linear_weight_ = cdf_slack_linear_weight_;
    previous_query_min_distance_ =
        std::numeric_limits<double>::quiet_NaN();
    current_frame_id_ =
        reference.header.frame_id.empty()
            ? "base_link"
            : reference.header.frame_id;
    current_plan_start_wall_ = ros::WallTime::now();
    last_cdf_roundtrip_ms_ = 0.0;
    plan_cdf_roundtrip_sum_ms_ = 0.0;
    plan_cdf_roundtrip_max_ms_ = 0.0;
    plan_cdf_roundtrip_count_ = 0;
  }

  publishQueryTrajectory(q_init, u_init, current_frame_id_);

  ROS_INFO_STREAM(
      "[LocalSparseSCPPlanner] plan " << plan_sequence_
      << " started reason=" << reason
      << " repair=" << static_cast<int>(repair)
      << " probe=" << static_cast<int>(probe)
      << " init=" << initialization_mode
      << " vis_obligations=" << schedule.size()
      << " frontier_active=" << static_cast<int>(frontier.active)
      << " frontier_weight_scale=" << frontier.frontier_weight_scale
      << " qvis_weight_scale=" << frontier.qvis_weight_scale);
  publishSummary("plan_started");
  return true;
}

void LocalSparseSCPPlanner::abortPlan(const std::string& reason) {
  SparseSolveResult trace;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    trace.trace_plan_sequence = plan_sequence_;
    trace.trace_observation_token = plan_observation_token_;
    trace.trace_repair = plan_repair_mode_;
    trace.trace_repair_observation_phase = plan_repair_observation_phase_;
    trace.trace_repair_witnesses = static_cast<int>(repair_unknown_witnesses_.size());
    trace.trace_repair_witnesses_fresh = !repair_witness_requires_reobserve_;
    trace.trace_probe = plan_probe_mode_;
    trace.trace_repair_ticket = plan_repair_ticket_;
    trace.status = reason;
    if (witness_response_required_ && reason == "cdf_wait_timeout") {
      for (std::size_t i = 0; i < repair_unknown_witnesses_.size(); ++i)
        publishWitnessDiagnosticLocked(i, repair_unknown_witnesses_[i].timestep,
            repair_unknown_witnesses_[i].point, "witness_response_timeout");
    }
    plan_running_ = false;
    waiting_for_cdf_ = false;
    pending_batch_.reset();
    last_plan_finish_time_ = ros::Time::now();
    const bool refresh_only = plan_candidate_replacement_refresh_only_;
    plan_candidate_replacement_refresh_only_ = false;
    if (!refresh_only) ++solve_failure_count_;
    if (!refresh_only && repair_mode_ && plan_repair_mode_ && plan_mode_epoch_ == mode_epoch_ &&
        repair_no_progress_.targetCurrent(plan_repair_ticket_)) {
      const auto outcome = repair_no_progress_.fail(plan_repair_ticket_);
      trace.repair_failures = repair_no_progress_.failures();
      trace.repair_feedback = outcome == RepairNoProgress::Failure::Exhausted
          ? "repair_plan_exhausted" : "repair_plan_retry_scheduled";
      requestPlanLocked(trace.repair_feedback);
    }
  }
  publishSummary("aborted_" + reason, &trace);
  if (reason == "cdf_wait_timeout" && !trace.trace_repair && !trace.trace_probe) {
    // Transport failure is not evidence of collision or geometric infeasibility.
    // Remain fail-closed; no unbounded automatic retry of the same reference.
    std_msgs::Bool uncertified; uncertified.data = true;
    task_uncertified_pub_.publish(uncertified);
    std_msgs::String status;
    status.data = "status=uncertified reason=cdf_wait_timeout action=hold_wait_external_replan";
    task_stall_pub_.publish(status);
  }
  if (!trace.repair_feedback.empty()) publishSummary(trace.repair_feedback, &trace);
  ROS_WARN_STREAM(
      "[LocalSparseSCPPlanner] plan aborted: " << reason);
}

void LocalSparseSCPPlanner::startWorker() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (worker_.joinable()) return;
  worker_stop_ = false;
  worker_ = std::thread(
      &LocalSparseSCPPlanner::workerLoop, this);
}

void LocalSparseSCPPlanner::stopWorker() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    worker_stop_ = true;
    pending_batch_.reset();
  }
  worker_cv_.notify_all();
  if (worker_.joinable()) worker_.join();
}

void LocalSparseSCPPlanner::workerLoop() {
  while (true) {
    care_collision_cdf::CollisionCDFConstraintBatchConstPtr batch;
    Eigen::MatrixXd q_bar, u_bar, q_ref, u_ref;
    Eigen::VectorXd previous_command;
    std::vector<DeadlineWaypoint> schedule;
    FrontierObjective frontier;
    std::vector<Eigen::Vector3d> obligation_points;
    bool repair = false;
    bool probe = false;
    double trust = 0.0;
    double slack_linear_weight = 0.0;
    int iteration = 0;
    unsigned long long solve_mode_epoch = 0;
    unsigned long long solve_plan_sequence = 0;
    bool observation_phase = true;
    bool witnesses_fresh = true;
    bool witnesses_have_prior_margin = false;
    int witness_count = 0;
    std::string solve_observation_token;
    RepairNoProgress::Ticket solve_repair_ticket;
    ros::Time expected_stamp;
    std::string executable_context;
    double executable_step_scale = 1.;
    bool executable_recheck_only = false;
    RepairPrefixSelection solve_prefix_selection;

    {
      std::unique_lock<std::mutex> lock(mutex_);
      worker_cv_.wait(
          lock,
          [&]() {
            return worker_stop_ ||
                   static_cast<bool>(pending_batch_);
          });
      if (worker_stop_) return;

      batch = pending_batch_;
      pending_batch_.reset();
      if (!batch || !plan_running_) continue;
      if (plan_candidate_replacement_refresh_only_) {
        plan_candidate_replacement_refresh_only_ = false;
        plan_running_ = false;
        waiting_for_cdf_ = false;
        Eigen::VectorXd measured;
        const double measured_age = (ros::Time::now()-latest_joint_state_.header.stamp).toSec();
        const double query_age = (ros::Time::now()-last_query_ros_time_).toSec();
        const bool fresh = qpReplacementRetryCurrentLocked() &&
            plan_mode_epoch_ == mode_epoch_ && repair_no_progress_.current(plan_repair_ticket_) &&
            plan_observation_token_ == latest_observation_token_ &&
            batch->header.stamp == current_query_stamp_ && query_age >= 0. && query_age <= 1. &&
            (ros::WallTime::now()-current_query_wall_).toSec() <= cdf_wait_timeout_s_ &&
            !repair_witness_requires_reobserve_ && has_joint_state_ &&
            !latest_joint_state_.header.stamp.isZero() && measured_age >= 0. && measured_age <= .2 &&
            extractMeasuredQ(latest_joint_state_, measured) &&
            (measured-plan_q_current_).lpNorm<Eigen::Infinity>() <= .005;
        const bool queued = fresh && requestQpReplacementLocked("repair_qp_failure");
        lock.unlock();
        publishSummary(queued ? "repair_qp_replacement_refresh_ready" :
                                "repair_qp_replacement_refresh_invalid_hold");
        continue; // This query is never submitted to PIQP or published as motion.
      }
      solve_mode_epoch = plan_mode_epoch_;
      solve_plan_sequence = plan_sequence_;
      solve_observation_token = plan_observation_token_;
      solve_repair_ticket = plan_repair_ticket_;

      repair = plan_repair_mode_;
      observation_phase = plan_repair_observation_phase_;
      witnesses_fresh = !repair_witness_requires_reobserve_;
      witnesses_have_prior_margin = std::any_of(
          repair_unknown_witnesses_.begin(), repair_unknown_witnesses_.end(),
          [](const RepairWitness& w) { return w.previous_margin >= 0. &&
              w.safety_margin > w.previous_margin; });
      witness_count = static_cast<int>(repair_unknown_witnesses_.size());
      probe = plan_probe_mode_;
      const bool executable_prefix_mode = repair || probe;

      double current_min_d =
          std::numeric_limits<double>::infinity();
      for (std::size_t i = 0; i < batch->distance.size(); ++i) {
        if (i >= batch->original_timestep.size()) break;
        const int k = batch->original_timestep[i];
        if (k < 1 || k > num_intervals_) continue;
        if (executable_prefix_mode && k > cdf_constraint_horizon_steps_)
          continue;
        const double d = batch->distance[i];
        if (std::isfinite(d))
          current_min_d = std::min(current_min_d, d);
      }

      if (scp_iteration_ > 0 &&
          std::isfinite(previous_query_min_distance_) &&
          std::isfinite(current_min_d)) {
        const double improvement =
            current_min_d - previous_query_min_distance_;
        if (improvement > trust_region_improvement_tol_) {
          trust_radius_ = std::min(
              trust_region_max_,
              trust_radius_ * trust_region_grow_);
        } else if (improvement <
                   -trust_region_improvement_tol_) {
          trust_radius_ = std::max(
              trust_region_min_,
              trust_radius_ * trust_region_shrink_);
        }
      }

      q_bar = plan_q_bar_;
      u_bar = plan_u_bar_;
      q_ref = plan_q_ref_;
      u_ref = plan_u_ref_;
      previous_command = plan_previous_command_;
      schedule = plan_schedule_;
      frontier = plan_frontier_;
      // A current VBC obligation is a persistent visibility-target identity,
      // including while the state machine is in PROBE_NORMAL. Keep it in the
      // hard-QP row classifier; the query path supplies only its near-term
      // guard rows. Confirmed GCDF witnesses are added separately and retain
      // their full persistent horizon.
      for (const auto& point : latest_vbc_obligation_points_) {
        if (std::none_of(
                obligation_points.begin(), obligation_points.end(),
                [&](const Eigen::Vector3d& old) {
                  return (old - point).lpNorm<Eigen::Infinity>() <= 1e-5;
                })) {
          obligation_points.push_back(point);
        }
      }
      if (repair) {
        for (const auto& witness : repair_unknown_witnesses_) {
          if (std::none_of(
                  obligation_points.begin(), obligation_points.end(),
                  [&](const Eigen::Vector3d& p) {
                    return (p - witness.point).lpNorm<Eigen::Infinity>() <= 1e-5;
                  })) {
            obligation_points.push_back(witness.point);
          }
        }
      }
      if (repair && !probe && repair_executable_view_enabled_ &&
          !cdf_slack_enabled_ && witnesses_fresh) {
        executable_context = repairExecutableContextLocked();
        plan_executable_iterates_.bind(executable_context);
        executable_step_scale = plan_executable_iterates_.step_scale;
        executable_recheck_only = plan_executable_iterates_.recheck_only;
      }
      solve_prefix_selection=plan_prefix_selection_;
      trust = trust_radius_*executable_step_scale;
      slack_linear_weight = plan_cdf_slack_linear_weight_;
      iteration = scp_iteration_;
      expected_stamp = current_query_stamp_;
      if(view_preview_aligned_enabled_ && batch->header.stamp==expected_stamp &&
          witnesses_fresh && (!repair || observation_phase) && !probe &&
          frontier.recovery_attempt==0 && !executable_recheck_only) {
        auto context=std::make_shared<ViewLocalContext>();
        context->batch=*batch;context->q_bar=q_bar;context->u_bar=u_bar;
        context->q_ref=q_ref;context->u_ref=u_ref;context->previous=previous_command;
        context->obligation_points=obligation_points;
        context->prefix=solve_prefix_selection;
        if(!prefixSelectionCompatible(context->prefix))context->prefix=latest_prefix_selection_;
        context->trust=trust;context->slack=slack_linear_weight;
        context->step_scale=executable_step_scale;context->iteration=iteration;
        context->repair=repair;context->epoch=solve_mode_epoch;
        context->plan_sequence=solve_plan_sequence;context->goal_revision=terminal_goal_revision_;
        context->owner=latest_observation_token_;context->region=latest_observation_region_;
        context->wall=ros::WallTime::now();context->measured_stamp=latest_joint_state_.header.stamp;
        extractMeasuredQ(latest_joint_state_,context->measured);
        context->snapshot.witnesses=repair_unknown_witnesses_;
        context->snapshot.points=latest_vbc_obligation_points_;
        if(restore_path_guard_scope_==std::to_string(mode_epoch_)+":"+latest_observation_token_)
          context->snapshot.path_guards=restore_path_guard_points_;
        context->snapshot.shell=initial_shell_recovery_;
        context->snapshot.goal=terminal_goal_;context->snapshot.goal_valid=terminal_goal_valid_;
        view_local_context_=context;
      }
    }

    if (batch->header.stamp != expected_stamp) {
      continue;
    }

    publishSummary("sparse_qp_started");

    ObservationDecisionPolicy decision_policy;
    decision_policy.repair=repair;decision_policy.probe=probe;
    decision_policy.observation_phase=observation_phase;
    decision_policy.iteration=iteration;decision_policy.max_iterations=max_scp_iterations_;
    decision_policy.recheck_only=executable_recheck_only;
    decision_policy.free_joint_task_enabled=repair_free_joint_task_tracking_scale_>0.;
    decision_policy.frontier_active=frontier.active;
    decision_policy.recovery_attempt=frontier.recovery_attempt;
    decision_policy.schedule_size=schedule.size();
    decision_policy.mask_has_free_joint=schedule.size()==1 &&
        schedule[0].joint_mask.size()==dof_ && (schedule[0].joint_mask.array()==0.).any();
    if(repair && observation_phase)native_snapshot_observation_seen_=true;
    auto decision=observationLocalDecision(decision_policy, [&](bool free_joint_task) {
      SparseSolveResult value;
      if (repair && !witnesses_fresh) value.status="repair_witness_missing_or_invalid";
      else if (repair && !observation_phase && cdf_slack_enabled_)
        value.status="repair_retreat_requires_hard_gcdf";
      else {
        const char* requested=std::getenv("CARE_NATIVE_TASK_SNAPSHOT");
        auto& selected_seq=probe ? native_snapshot_probe_seq_ : native_snapshot_normal_seq_;
        const bool capture_task=requested && std::string(requested)=="true" && !repair &&
          !free_joint_task && native_snapshot_observation_seen_ && native_snapshot_count_<16 &&
          (selected_seq==0 || selected_seq==solve_plan_sequence);
        if(capture_task && selected_seq==0)selected_seq=solve_plan_sequence;
        ViewPreviewSnapshot frozen;
        value=solveSparseSubproblem(*batch,q_bar,u_bar,q_ref,u_ref,previous_command,
          schedule,frontier,obligation_points,repair,probe,trust,slack_linear_weight,
          false,false,executable_step_scale,executable_recheck_only,
          repair_prefix_binding_enabled_ && repair && !probe ? &solve_prefix_selection : nullptr,
          free_joint_task,nullptr,capture_task?&frozen:nullptr);
        if(capture_task) {
          ++native_snapshot_count_;ViewPreviewSnapshot echo_input;
          const auto tic=ros::WallTime::now();
          const auto echo=solveSparseSubproblem(*batch,q_bar,u_bar,q_ref,u_ref,previous_command,
            schedule,frontier,obligation_points,repair,probe,trust,slack_linear_weight,
            false,false,executable_step_scale,executable_recheck_only,nullptr,
            free_joint_task,&frozen,&echo_input);
          const char* root=std::getenv("ROOT_OUT");
          if(root && *root) {
            std::ofstream file(std::string(root)+"/native_task_"+std::to_string(native_snapshot_count_)+".json");
            file<<"{\"schema\":1,\"plan_sequence\":"<<solve_plan_sequence<<",\"mode_epoch\":"<<solve_mode_epoch
              <<",\"iteration\":"<<iteration<<",\"probe\":"<<(probe?"true":"false")<<",\"repair\":false,\"owner\":";
            jsonString(file,solve_observation_token);file<<",\"query_ns\":\""<<batch->header.stamp.toNSec()<<"\",\"dt\":";jsonNumber(file,dt_);
            file<<",\"probe_velocity_cap\":";jsonNumber(file,probe_velocity_cap_);
            file<<",\"trust\":";jsonNumber(file,trust);file<<",\"q_bar\":";jsonTrajectory(file,q_bar);
            file<<",\"u_bar\":";jsonTrajectory(file,u_bar);file<<",\"q_ref\":";jsonTrajectory(file,q_ref);
            file<<",\"u_ref\":";jsonTrajectory(file,u_ref);file<<",\"previous_command\":";jsonVector(file,previous_command);
            file<<",\"batch\":";auditBatch(file,*batch);
            file<<",\"matrix_identical\":"<<(frozen.native_matrix_json==echo_input.native_matrix_json?"true":"false");
            file<<",\"matrix\":"<<(frozen.native_matrix_json.empty()?"null":frozen.native_matrix_json);
            file<<",\"normal_live_fresh\":"<<(frozen.normal_live_fresh?"true":"false")<<",\"normal_live_q\":";jsonVector(file,frozen.normal_live_q);
            file<<",\"normal_live_v\":";jsonVector(file,frozen.normal_live_v);
            file<<",\"primitive_live_fresh\":"<<(frozen.primitive_live_fresh?"true":"false")<<",\"primitive_live_q\":";jsonVector(file,frozen.primitive_live_q);
            file<<",\"primitive_live_v\":";jsonVector(file,frozen.primitive_live_v);
            file<<",\"active_observation_points\":[";int pi=0;
            for(const auto& point:frozen.points){if(pi++)file<<',';jsonVector(file,point);}file<<']';
            file<<",\"obligation_points\":[";pi=0;
            for(const auto& point:obligation_points){if(pi++)file<<',';jsonVector(file,point);}file<<']';
            file<<",\"q_min\":";jsonVector(file,q_min_);file<<",\"q_max\":";jsonVector(file,q_max_);
            file<<",\"velocity_limits\":";jsonVector(file,velocity_limits_);file<<",\"acceleration_limits\":";jsonVector(file,acceleration_limits_);
            file<<",\"goal_valid\":"<<(frozen.goal_valid?"true":"false")<<",\"goal_position\":";jsonVector(file,frozen.goal.translation());
            file<<",\"schedule_size\":"<<schedule.size()<<",\"frontier_active\":"<<(frontier.active?"true":"false");
            file<<",\"frontier_q\":";jsonVector(file,frontier.q);
            file<<",\"witness_count\":"<<frozen.witnesses.size()<<",\"path_guard_count\":"<<frozen.path_guards.size();

            auto write=[&](const SparseSolveResult& result) {
              file<<"{\"status\":";jsonString(file,result.status);file<<",\"solved\":"<<(result.solved?"true":"false");
              file<<",\"normal_curve_checked\":"<<(result.normal_curve_checked?"true":"false")<<",\"normal_curve_valid\":"<<(result.normal_curve_valid?"true":"false");
              file<<",\"normal_live_checked\":"<<(result.normal_live_checked?"true":"false")<<",\"normal_live_valid\":"<<(result.normal_live_valid?"true":"false");
              file<<",\"primitive_valid\":"<<(result.primitive_executable_valid?"true":"false");
              file<<",\"q\":";jsonTrajectory(file,result.q);file<<",\"u\":";jsonTrajectory(file,result.u);file<<'}';
            };
            file<<",\"original\":";write(value);file<<",\"echo\":";write(echo);
            file<<",\"diagnostic_extra_ms\":";jsonNumber(file,(ros::WallTime::now()-tic).toSec()*1000.);
            file<<",\"motion_authorized\":false,\"scope\":\"Original native subproblem including nonlinear/live checks; not an end-to-end task certificate\"}\n";
          }
        }
      }
      return value;
    }, [&](const SparseSolveResult& primary,const SparseSolveResult& trial) {
      return freeJointTaskImproves(primary,trial,schedule,frontier,q_ref);
    });
    SparseSolveResult result=std::move(decision.result);
    const bool free_joint_trial_ran=decision.free_joint_trial_ran;
    const bool free_joint_trial_used=decision.free_joint_trial_used;
    bool margin_cause_proven = false;
    if (!result.solved && repair && !probe && witnesses_fresh &&
        (result.status == "primal infeasible" || result.status == "max iterations reached") &&
        witnesses_have_prior_margin) {
      const auto prior = solveSparseSubproblem(*batch, q_bar, u_bar, q_ref, u_ref,
          previous_command, schedule, frontier, obligation_points, repair, probe,
          trust, slack_linear_weight, false, true,
          executable_step_scale, executable_recheck_only,
          repair_prefix_binding_enabled_ && repair && !probe ? &solve_prefix_selection : nullptr);
      margin_cause_proven = prior.solved;
    }
    result.trace_repair_observation_phase = observation_phase;
    result.trace_repair_witnesses = witness_count;
    result.trace_repair_witnesses_fresh = witnesses_fresh;

    // C5.25: compute the existing soft diagnostic before deciding whether a
    // failed PROBE plan is terminal. The soft solve is NON-EXECUTABLE; when
    // enabled it may only provide the next SCP linearization point.
    SparseSolveResult diagnostic;
    bool diagnostic_ran = false;
    if (!result.solved && result.status!="known_ground_start_clearance_hold" && (!repair || result.first_step_conflicting_cdf_rows > 0) &&
        task_failure_slack_diagnostic_enabled_) {
      diagnostic =
          solveSparseSubproblem(
              *batch,
              q_bar,
              u_bar,
              q_ref,
              u_ref,
              previous_command,
              schedule,
              frontier,
              obligation_points,
              repair,
              probe,
              trust,
              task_failure_slack_diagnostic_weight_,
              true,
              false,
              executable_step_scale,
              executable_recheck_only,
              repair_prefix_binding_enabled_ && repair && !probe
                  ? &solve_prefix_selection : nullptr);
      diagnostic_ran = true;
    }

    result.trace_plan_sequence = solve_plan_sequence;
    result.trace_observation_token = solve_observation_token;
    result.trace_repair = repair;
    result.trace_probe = probe;
    result.trace_repair_ticket = solve_repair_ticket;
    diagnostic.trace_plan_sequence = solve_plan_sequence;
    diagnostic.trace_observation_token = solve_observation_token;
    diagnostic.trace_repair = repair;
    diagnostic.trace_probe = probe;
    bool publish_candidate = false;
    bool publish_next_query = false;
    bool restoration_applied = false;
    bool normal_reseed_applied = false;
    bool repair_reseed_applied = false;
    bool geometric_restore_applied = false;
    bool normal_qp_backtrack_applied = false;
    int normal_qp_backtrack_attempt = 0;
    int normal_qp_backtrack_max_attempts = 0;
    Eigen::MatrixXd q_next, u_next;
    std::string frame;
    double total_ms = 0.0;

    {
      std::lock_guard<std::mutex> lock(mutex_);
      // Failure evidence/restoration is as request-sensitive as candidates.
      // An old worker result must not clear a newer plan's running state.
      if (solve_plan_sequence != plan_sequence_) continue;
      if (solve_mode_epoch != mode_epoch_ || repair != repair_mode_ ||
          (repair_prefix_binding_enabled_ && repair && !probe &&
           !prefixSelectionCurrentLocked(solve_prefix_selection)) ||
          probe != probe_mode_ ||
          (repair && !repair_no_progress_.current(solve_repair_ticket))) {
        plan_running_ = false;
        waiting_for_cdf_ = false;
        if (repair && repair_mode_ && solve_mode_epoch == mode_epoch_ &&
            repair_no_progress_.targetCurrent(solve_repair_ticket)) {
          const int charged = iteration+1+(free_joint_trial_ran ? 1 : 0);
          if (!probe && !probe_mode_ && observation_phase &&
              !repair_no_progress_.current(solve_repair_ticket) &&
              solve_observation_token == latest_observation_token_ &&
              (!repair_prefix_binding_enabled_ || prefixSelectionCurrentLocked(solve_prefix_selection)) &&
              result.solved && result.max_slack <= 1e-9 && result.primitive_executable_valid &&
              !result.primitive_live_rebase_required && result.q.rows()==dof_ &&
              result.q.cols()==num_intervals_+1 && finiteMatrix(result.q) &&
              charged < max_scp_iterations_) {
            repair_warm_q_ = result.q;
            repair_warm_key_ = repairWarmStartKeyLocked();
            repair_warm_stamp_ = ros::Time::now();
            repair_warm_iterations_ = charged;
          }
          requestPlanLocked("repair_stale_measured_progress");
        }
        continue;
      }
      if (!executable_context.empty() &&
          (executable_context != repairExecutableContextLocked() ||
           repair_witness_requires_reobserve_ || repair_witness_overflow_)) {
        plan_executable_iterates_ = RepairExecutableIterates{};
        plan_running_ = false;
        waiting_for_cdf_ = false;
        requestPlanLocked("repair_executable_context_changed_hold");
        continue;
      }
      ++solve_count_;
      if (free_joint_trial_ran) {
        // The additional hard QP consumes one of the original six iterations,
        // including rejected trials. The ordinary result consumes the next.
        ++scp_iteration_;
        ++solve_count_;
        ROS_INFO_STREAM("[REPAIR_TASK_REJOIN] plan_seq=" << solve_plan_sequence
            << " iteration=" << scp_iteration_+1 << " max_iterations=" << max_scp_iterations_
            << " used=" << free_joint_trial_used << " original_observation_retained=1");
      }
      // A current UNKNOWN halfspace blocking the desired local frontier is
      // an observation dependency hypothesis, not a collision certificate or
      // permission to execute. Preserve the parent observation identity.
      const double dependency_age = (ros::Time::now()-last_query_ros_time_).toSec();
      if (safe_frontier_recovery_enabled_ && repair && !probe && observation_phase &&
          dependency_age >= 0.0 && dependency_age <= .5 &&
          !result.observation_dependency_points.empty()) {
        std::ostringstream s;
        s << std::setprecision(17) << "{\"version\":1,\"plan_seq\":" << solve_plan_sequence
          << ",\"mode_epoch\":" << solve_mode_epoch
          << ",\"query_stamp_ns\":" << batch->header.stamp.toNSec()
          << ",\"query_ros_s\":" << last_query_ros_time_.toSec()
          << ",\"observation_token\":\"" << solve_observation_token << "\",\"points\":[";
        for (std::size_t i=0; i<result.observation_dependency_points.size(); ++i) {
          const auto& p = result.observation_dependency_points[i];
          if (i) s << ',';
          s << '[' << p[0] << ',' << p[1] << ',' << p[2] << ']';
        }
        s << "]}";
        std_msgs::String event; event.data = s.str();
        observation_dependency_pub_.publish(event);
      }
      if (safe_frontier_recovery_enabled_ && repair && !probe &&
          frontier.recovery_attempt > 0 && result.recovery_frontier.size() == dof_) {
        safe_frontier_recovery_.target = result.recovery_frontier;
        if (frontier.visibility_gradient.size()==7 && !result.recovery_rejoining &&
            plan_q_current_.size()==dof_)
          safe_frontier_recovery_.detour_direction=result.recovery_frontier-plan_q_current_;
        if (result.solved)
          safe_frontier_recovery_.following_frontier = result.recovery_rejoining;
        plan_frontier_.recovery_target = result.recovery_frontier;
      }
      if (result.solved && frontier.visibility_gradient.size()==7 &&
          plan_observation_token_==latest_observation_token_) {
        safe_frontier_recovery_.noteClearanceFeedback(plan_observation_token_,
            result.recovery_clearance_signature,last_query_ros_time_.toSec(),
            ros::Time::now().toSec(),result.recovery_start_clearance);
        ROS_INFO_STREAM_THROTTLE(.5,"[LocalSparseSCPPlanner] visibility_guidance_solve token="
            <<plan_observation_token_<<" rejoining="<<result.recovery_rejoining
            <<" clearance_q_proxy="<<result.recovery_start_clearance
            <<" failed_directions="<<frontier.failed_directions.size());
      }
      if (!result.solved) {
        if(result.status!="known_ground_start_clearance_hold")++solve_failure_count_;
        // A changed diagnostic center is queried and hard-solved within the
        // original SCP budget. A repeated measured hold cannot restore it.
        bool internal_restore=false;
        if(repair && !probe && witnesses_fresh && !margin_cause_proven && diagnostic_ran && diagnostic.solved &&
            plan_repair_conflict_reseed_.consume(true,result.first_step_conflicting_cdf_rows,
                q_bar,diagnostic.q,scp_iteration_,max_scp_iterations_,scp_step_tolerance_inf_)) {
          ++scp_iteration_;plan_q_bar_=diagnostic.q;plan_u_bar_=diagnostic.u;
          previous_query_min_distance_=std::numeric_limits<double>::quiet_NaN();
          plan_initialization_mode_="repair_feasibility_restore_requery";
          result.repair_feedback=plan_initialization_mode_;repair_reseed_applied=true;internal_restore=true;
          publish_next_query=true;frame=current_frame_id_;q_next=plan_q_bar_;u_next=plan_u_bar_;
        }
        if(!internal_restore && repair && !probe && witnesses_fresh &&
            result.primitive_conflicting_rows>0 &&
            measuredBrakingSeed(q_bar.col(0),previous_command,acceleration_limits_,
                num_intervals_,dt_,q_next,u_next) &&
            plan_repair_conflict_reseed_.consume(true,result.primitive_conflicting_rows,
                q_bar,q_next,scp_iteration_,max_scp_iterations_,scp_step_tolerance_inf_)) {
          ++scp_iteration_;plan_q_bar_=q_next;plan_u_bar_=u_next;
          previous_query_min_distance_=std::numeric_limits<double>::quiet_NaN();
          plan_initialization_mode_="repair_primitive_measured_reseed";
          result.repair_feedback="repair_primitive_measured_reseed_requery";
          repair_reseed_applied=true;internal_restore=true;
          publish_next_query=true;frame=current_frame_id_;
        }
        if(!internal_restore && repair && !probe && witnesses_fresh && !repair_witness_overflow_ &&
            repair_prefix_binding_enabled_ && initial_shell_recovery_enabled_ && primitive_observation_guard_enabled_ &&
            (result.status=="primal infeasible" || result.status=="max iterations reached" ||
             (fixed_start_recovery_enabled_ && result.fixed_start_unknown_blocked &&
              !result.fixed_start_recovery_points.empty())) &&
            result.first_step_conflicting_occupied_rows==0 &&
            !result.geometric_restore_guard_points.empty() && result.geometric_restore_guard_points.size()<=32 &&
            latest_observation_token_==solve_observation_token && has_single_waypoint_q_ &&
            !safe_frontier_recovery_.blocked() && !candidate_replacement_pending_) {
          if(buildGeometricRestoreProposalLocked(result,solve_observation_token,solve_prefix_selection.prefix_s,
              &result.q,&result.u)) {
              if(!safe_frontier_recovery_.active)safe_frontier_recovery_.advance();
              geometric_restore_scope_=std::to_string(mode_epoch_)+":"+safe_frontier_recovery_.key;
              // A geometry proposal supersedes an unreserved replacement
              // trigger. A later response cannot exhaust this in-flight path.
              candidate_replacement_trigger_pending_=false;
              candidate_replacement_retry_pending_=false;
              safe_frontier_recovery_.detour_direction=result.q.col(result.q.cols()-1)-result.q.col(0);
              safe_frontier_recovery_.following_frontier=false;
              result.repair_feedback=result.fixed_start_recovery_points.empty() ? "geometric_restore_proposal" : "fixed_start_recovery_proposal";
              geometric_restore_applied=true;publish_candidate=true;plan_running_=false;waiting_for_cdf_=false;
              last_plan_finish_time_=ros::Time::now();
          }
        }
        if(internal_restore || geometric_restore_applied) {
          // The failed original QP stays failed. Only the explicitly tagged
          // geometric proposal may enter the alternate certification route.
        } else if (margin_cause_proven) {
          for (auto& w : repair_unknown_witnesses_) {
            if (w.previous_margin < 0. || w.safety_margin <= w.previous_margin) continue;
            ROS_WARN_STREAM("[LocalSparseSCPPlanner] point_margin_rollback point=["
                << w.point.transpose() << "] from=" << w.safety_margin
                << " to=" << w.previous_margin << " cause=old_margin_hard_qp_solved");
            w.safety_margin = w.previous_margin;
            w.margin_ceiling = std::min(w.margin_ceiling, w.previous_margin);
            w.previous_margin = -1.;
          }
          repair_witness_requires_reobserve_ = true;
        }
        // An internal requery or a geometric proposal already owns this
        // failed solve. Do not also replace its observation context before
        // the requery or independent certification can finish.
        if (!internal_restore && !geometric_restore_applied) {
        if(result.status=="known_ground_start_clearance_hold") {
          plan_running_=false;waiting_for_cdf_=false;pending_batch_.reset();plan_requested_=false;
          known_ground_hold_.enter(terminal_goal_valid_ ? terminal_goal_revision_ : 0,
              groundContextLocked(),ros::Time::now().toSec(),ros::WallTime::now().toSec(),latest_execution_stamp_ns_);
          result.repair_feedback=known_ground_hold_.reason;
          ROS_INFO_STREAM("[GROUND_HOLD] event=enter goal="<<known_ground_hold_.goal
              <<" wakeups="<<known_ground_hold_.wakeups<<" ros_deadline="<<known_ground_hold_.ros_deadline
              <<" wall_deadline="<<known_ground_hold_.wall_deadline<<" reason="<<known_ground_hold_.reason);
        } else if (result.fixed_start_unknown_blocked) {
          plan_running_=false;waiting_for_cdf_=false;pending_batch_.reset();
          candidate_replacement_trigger_pending_=false;
          candidate_replacement_retry_pending_=false;
          fixed_start_clearance_hold_.record(q_bar.col(0),result.fixed_start_unknown_point,
              result.fixed_start_unknown_distance_m,result.fixed_start_unknown_floor_m,ros::Time::now().toSec());
          result.repair_feedback="fixed_start_unknown_clearance_hold";
          requestPlanLocked(result.repair_feedback);
        } else if (repair && result.status == "primitive_observation_start_blocked") {
          plan_running_ = false; waiting_for_cdf_ = false; pending_batch_.reset();
          repair_candidate_failure_.record(repairTargetKeyLocked(),
              solve_observation_token, mode_epoch_, solve_repair_ticket.progress);
          result.repair_feedback = "primitive_observation_start_blocked_hold";
          requestPlanLocked(result.repair_feedback);
        } else if (candidate_replacement_enabled_ && repair && !probe && witnesses_fresh &&
            !margin_cause_proven &&
            result.reachable_conflicting_repair_witness_rows > 0 &&
            (result.status == "primal infeasible" || result.status == "max iterations reached") &&
            measuredBrakingSeed(q_bar.col(0), previous_command, acceleration_limits_,
                                num_intervals_, dt_, q_next, u_next) &&
            plan_repair_conflict_reseed_.consume(
                true, result.reachable_conflicting_repair_witness_rows,
                q_bar, q_next, scp_iteration_, max_scp_iterations_, scp_step_tolerance_inf_)) {
          // A later hard iterate can make this local trust box infeasible.
          // Spend one remaining SCP iteration on a fresh measured-centered
          // query. The seed never reaches candidate publication, and this
          // branch changes no row, margin, target, sensor or outer budget.
          ++scp_iteration_;
          plan_q_bar_ = q_next;
          plan_u_bar_ = u_next;
          previous_query_min_distance_ = std::numeric_limits<double>::quiet_NaN();
          plan_initialization_mode_ = "repair_conflict_measured_reseed";
          result.repair_feedback = "repair_conflict_measured_reseed_requery";
          repair_reseed_applied = true;
          publish_next_query = true;
          frame = current_frame_id_;
        } else if (candidate_replacement_enabled_ && repair && !probe && witnesses_fresh &&
            !margin_cause_proven &&
            result.reachable_conflicting_repair_witness_rows > 0 &&
            (result.status == "primal infeasible" || result.status == "max iterations reached")) {
          // This convex subproblem contains an unreachable hard witness.
          // If a changed center was available it has already had its one
          // bounded requery. Keep all rows and wait for new evidence.
          plan_running_ = false; waiting_for_cdf_ = false; pending_batch_.reset();
          repair_candidate_failure_.record(
              repairTargetKeyLocked(), solve_observation_token, mode_epoch_,
              solve_repair_ticket.progress);
          repair_observation_phase_ = true;
          result.repair_feedback = result.first_step_conflicting_repair_witness_rows > 0
              ? "repair_witness_start_infeasible_hold"
              : "repair_witness_linearization_infeasible_hold";
          requestPlanLocked(result.repair_feedback);
        } else if (candidate_replacement_enabled_ && repair && !probe && witnesses_fresh &&
            (result.status == "primal infeasible" || result.status == "max iterations reached")) {
          plan_running_ = false; waiting_for_cdf_ = false; pending_batch_.reset();
          repair_candidate_failure_.record(
              repairTargetKeyLocked(), solve_observation_token, mode_epoch_,
              solve_repair_ticket.progress);
          repair_observation_phase_ = true;
          const bool observation_current =
              solve_observation_token == latest_observation_token_;
          const bool queued = observation_current &&
              requestQpReplacementLocked("repair_qp_failure");
          result.repair_feedback = queued ? "repair_qp_candidate_replacement_wait" :
              observation_current ? "repair_qp_candidate_exhausted_hold" :
                  "repair_qp_stale_observation_retry";
          if (!queued) requestPlanLocked(result.repair_feedback);
        } else {

        const bool restoration_available =
            probe &&
            probe_feasibility_restoration_enabled_ &&
            diagnostic_ran &&
            diagnostic.solved &&
            plan_probe_feasibility_restore_attempts_ <
                probe_feasibility_restoration_max_attempts_;

        // One bounded NORMAL relinearization only after an independently
        // proven row/box conflict. Preserve the task objective, measured q_0,
        // trust radius, PIQP settings and total SCP budget. This seed is never
        // published as an executable candidate; fresh GCDF + hard QP required.
        if (!repair && !probe && result.box_conflicting_cdf_rows > 0 &&
            !plan_normal_reseed_used_ && scp_iteration_ + 1 < max_scp_iterations_ &&
            measuredBrakingSeed(q_bar.col(0), previous_command, acceleration_limits_,
                                num_intervals_, dt_, q_next, u_next)) {
          plan_normal_reseed_used_ = true;
          ++scp_iteration_;  // failed solve consumes an iteration too
          plan_q_bar_ = q_next;
          plan_u_bar_ = u_next;
          previous_query_min_distance_ = std::numeric_limits<double>::quiet_NaN();
          plan_initialization_mode_ = "normal_conflict_measured_reseed";
          normal_reseed_applied = true;
          publish_next_query = true;
          frame = current_frame_id_;
        } else if (
            plan_previous_q_bar_.rows() == q_bar.rows() &&
            plan_previous_q_bar_.cols() == q_bar.cols() &&
            plan_previous_u_bar_.rows() == u_bar.rows() &&
            plan_previous_u_bar_.cols() == u_bar.cols() &&
            plan_previous_q_bar_.allFinite() &&
            plan_previous_u_bar_.allFinite() &&
            plan_qp_backtrack_policy_.consume(
                !repair && !probe,
                plan_has_hard_solution_,
                result.selected_unknown_cdf_rows,
                result.status)) {
          // The previous hard iterate is only a new linearization center. It
          // must be re-queried and solved again before the normal candidate
          // reaches the final GCDF/VBC commit gates.
          plan_q_bar_ = plan_previous_q_bar_;
          plan_u_bar_ = plan_previous_u_bar_;
          previous_query_min_distance_ =
              std::numeric_limits<double>::quiet_NaN();
          plan_initialization_mode_ = "normal_unknown_qp_backtrack";
          ++normal_qp_backtrack_count_;
          normal_qp_backtrack_applied = true;
          normal_qp_backtrack_attempt = plan_qp_backtrack_policy_.attempts;
          normal_qp_backtrack_max_attempts =
              plan_qp_backtrack_policy_.max_attempts;
          publish_next_query = true;
          q_next = plan_q_bar_;
          u_next = plan_u_bar_;
          frame = current_frame_id_;
        } else if (restoration_available) {
          // The soft trajectory stays internal.  It may require substantial
          // slack on UNKNOWN rows, so it must not become the next hard-QP
          // linearization center: doing that can move the center into a
          // region that is only safe in the relaxed diagnostic problem.  Use
          // the currently measured state as a conservative hold seed instead;
          // the task q_ref objective remains active in the hard QP, and a
          // fresh GCDF query plus hard solve is still required before any
          // candidate publication.
          const bool measured_hold_seed_available =
              q_bar.cols() > 0 && q_bar.col(0).allFinite();
          if (measured_hold_seed_available) {
            plan_q_bar_ = q_bar;
            for (int k = 0; k < plan_q_bar_.cols(); ++k)
              plan_q_bar_.col(k) = q_bar.col(0);
            plan_u_bar_ = u_bar;
            plan_u_bar_.setZero();
            plan_initialization_mode_ = "probe_measured_hold_recovery";
          } else {
            // q_bar is expected to contain the measured q_0. Keep a guarded
            // fallback for malformed input so restoration remains bounded.
            plan_q_bar_ = diagnostic.q;
            plan_u_bar_ = diagnostic.u;
            plan_initialization_mode_ = "probe_soft_diagnostic_recovery";
          }
          previous_query_min_distance_ =
              std::numeric_limits<double>::quiet_NaN();
          ++plan_probe_feasibility_restore_attempts_;
          ++probe_feasibility_restore_count_;
          plan_probe_restore_pending_hard_recheck_ = true;
          restoration_applied = true;
          publish_next_query = true;
          q_next = plan_q_bar_;
          u_next = plan_u_bar_;
          frame = current_frame_id_;
        } else {
          plan_running_ = false;
          waiting_for_cdf_ = false;
          plan_probe_restore_pending_hard_recheck_ = false;
          last_plan_finish_time_ = ros::Time::now();
          if (repair) {
            if (observation_phase && (result.selected_unknown_cdf_rows > 0 || !witnesses_fresh)) {
              repair_witness_requires_reobserve_ = true;
              repair_observation_phase_ = false;
            }
            const auto outcome = repair_no_progress_.fail(solve_repair_ticket);
            result.repair_failures = repair_no_progress_.failures();
            result.repair_feedback = outcome == RepairNoProgress::Failure::Exhausted
                ? "repair_qp_exhausted" : outcome == RepairNoProgress::Failure::Retry
                ? "repair_qp_retry_scheduled" : "repair_qp_stale_progress_retry";
            // A fresh GCDF query and hard solve are required for every retry.
            // Measured motion during this solve belongs to a new episode.
            requestPlanLocked(result.repair_feedback);
          }
        }
        } // legacy failure handling only when replacement is unavailable/inapplicable
        } // no competing failure disposition for a restored solve
      } else {
        for (auto& w : repair_unknown_witnesses_) w.previous_margin = -1.;
        if (probe && plan_probe_restore_pending_hard_recheck_) {
          ++probe_feasibility_restore_success_count_;
          plan_probe_restore_pending_hard_recheck_ = false;
        }

        // Keep the center that produced the last accepted hard iterate. A
        // later CDF re-query can introduce new UNKNOWN rows; if that hard QP
        // fails, the bounded recovery branch above relinearizes from this
        // prior center instead of terminating immediately.
        if (plan_has_hard_solution_) {
          plan_previous_q_bar_ = plan_q_bar_;
          plan_previous_u_bar_ = plan_u_bar_;
        } else {
          // The first hard solve may follow the existing measured-q reseed;
          // use that actual center instead of the pre-reseed task reference.
          plan_previous_q_bar_ = q_bar;
          plan_previous_u_bar_ = u_bar;
        }
        plan_q_bar_ = result.q;
        plan_u_bar_ = result.u;
        plan_has_hard_solution_ = true;
        previous_query_min_distance_ = result.min_distance;
        ++scp_iteration_;

        const bool slack_satisfied =
            !cdf_adaptive_slack_penalty_ ||
            result.max_slack <= cdf_slack_tolerance_;
        const bool converged =
            result.step_inf <= scp_step_tolerance_inf_ &&
            slack_satisfied;
        const bool exhausted =
            scp_iteration_ >= max_scp_iterations_;

        if (!repair && !probe && result.normal_curve_checked && result.normal_curve_valid &&
            (!result.normal_live_checked || result.normal_live_valid)) {
          plan_normal_retained_q_=result.q;plan_normal_retained_u_=result.u;
        }
        const bool executable_policy = !executable_context.empty() &&
            result.primitive_executable_samples > 0;
        if (executable_policy) {
          if (result.primitive_executable_valid && !result.primitive_live_rebase_required) {
            plan_executable_iterates_.retain(result.q,result.u);
          } else {
            // A solved convex subproblem can still miss the exact brake map.
            // Keep the previous locally feasible candidate and reduce both q
            // and brake-velocity trial steps before the next fresh query.
            plan_executable_iterates_.rejectTrial();
            if (!exhausted && plan_executable_iterates_.restore(
                    q_bar.col(0),&plan_q_bar_,&plan_u_bar_)) {
              result.executable_trial_restored = true;
              previous_query_min_distance_ = std::numeric_limits<double>::quiet_NaN();
            }
          }
          result.executable_retained_available = plan_executable_iterates_.available();
        }

        if (cdf_adaptive_slack_penalty_ &&
            !slack_satisfied && !exhausted) {
          plan_cdf_slack_linear_weight_ =
              std::min(
                  cdf_slack_penalty_max_,
                  std::max(
                      plan_cdf_slack_linear_weight_,
                      cdf_slack_linear_weight_) *
                      cdf_slack_penalty_multiplier_);
        }

        if (!repair && !probe && result.normal_curve_checked &&
            (!result.normal_curve_valid || (result.normal_live_checked && !result.normal_live_valid))) {
          if (!exhausted) {
            if (result.normal_curve_valid && result.normal_live_checked && !result.normal_live_valid) {
              // Rebuild from one measured q/v snapshot. A different origin
              // invalidates every old linearization; the next query is fresh.
              Eigen::MatrixXd seed_q,seed_u;
              if(buildNormalMotionSeed(result.normal_live_q,result.normal_live_v,result.q,seed_q,seed_u)) {
                plan_q_current_=result.normal_live_q;plan_previous_command_=result.normal_live_v;
                plan_q_bar_=seed_q;plan_u_bar_=seed_u;plan_q_ref_.col(0)=result.normal_live_q;
                plan_normal_retained_q_.resize(0,0);plan_normal_retained_u_.resize(0,0);
              }
            } else if (scp_iteration_+1==max_scp_iterations_ && plan_normal_retained_q_.size()) {
              // Reserve the final original iteration for an earlier feasible
              // shape. It is a new query/solve, never reuse of a certificate.
              plan_q_bar_=plan_normal_retained_q_;plan_u_bar_=plan_normal_retained_u_;
            }
            plan_has_hard_solution_=false;
            previous_query_min_distance_=std::numeric_limits<double>::quiet_NaN();
            plan_initialization_mode_="normal_curve_relinearization";
            result.repair_feedback="normal_curve_relinearization";
            publish_next_query=true;q_next=plan_q_bar_;u_next=plan_u_bar_;frame=current_frame_id_;
          } else {
            plan_running_=false;waiting_for_cdf_=false;last_plan_finish_time_=ros::Time::now();
            result.repair_feedback="normal_curve_iteration_budget";
            requestPlanLocked(result.repair_feedback);
          }
        } else if (result.primitive_live_rebase_required && !exhausted) {
          // The receiver rebases the newly proposed path to measured q before
          // certification. If that already destroys a known geometric row,
          // spend the next ORIGINAL SCP iteration at that measured origin.
          // This is a query seed only: old CDF rows cannot certify the shift.
          const Eigen::VectorXd shift = result.primitive_live_rebase_start-result.q.col(0);
          plan_q_bar_ = result.q.colwise()+shift;
          plan_u_bar_ = result.u;
          plan_q_current_ = result.primitive_live_rebase_start;
          plan_previous_command_ = result.primitive_live_rebase_velocity;
          plan_executable_iterates_.invalidateCurve();
          plan_has_hard_solution_ = false;
          plan_previous_q_bar_ = plan_q_bar_;
          plan_previous_u_bar_ = plan_u_bar_;
          previous_query_min_distance_ = std::numeric_limits<double>::quiet_NaN();
          plan_initialization_mode_ = "repair_live_rebase_requery";
          result.repair_feedback = "repair_live_rebase_requery";
          publish_next_query = true;
          q_next = plan_q_bar_; u_next = plan_u_bar_; frame = current_frame_id_;
        } else if ((!result.primitive_executable_valid || result.primitive_live_rebase_required) && exhausted) {
          plan_running_ = false;
          waiting_for_cdf_ = false;
          repair_candidate_failure_.record(repairTargetKeyLocked(),
              solve_observation_token,mode_epoch_,solve_repair_ticket.progress);
          // A local executable linearization failure is not global
          // infeasibility. Use the same authenticated, finite sensor/direction
          // replacement path as other QP failures; it spends the existing ledger.
          const bool queued=candidate_replacement_enabled_ && witnesses_fresh &&
              solve_observation_token==latest_observation_token_ &&
              requestQpReplacementLocked("repair_qp_failure");
          result.repair_feedback=queued ? "repair_executable_candidate_replacement_wait" :
              result.primitive_live_rebase_required ? "repair_live_rebase_exhausted_hold" :
                                         "repair_executable_linearization_exhausted_hold";
          if(!queued)requestPlanLocked(result.repair_feedback);
        } else if ((converged || exhausted) && result.primitive_executable_valid) {
          if (executable_recheck_only)
            result.repair_feedback = "repair_executable_retained_recheck_passed";
          if (repair && !observation_phase) {
            // The active plan remains retreat through publication. Only the
            // NEXT plan regains visibility objectives, after this complete
            // hard solve with fresh witness coverage. UNKNOWN rows may remain.
            repair_observation_phase_ = true;
          }
          publish_candidate = true;
          plan_running_ = false;
          waiting_for_cdf_ = false;
          last_plan_finish_time_ = ros::Time::now();
          total_ms =
              (ros::WallTime::now() - current_plan_start_wall_)
                  .toSec() * 1000.0;
        } else {
          if (!result.primitive_executable_valid)
            result.repair_feedback = result.executable_trial_restored
                ? "repair_executable_trial_rejected_requery"
                : "repair_executable_linearization_requery";
          if (executable_policy && plan_executable_iterates_.reserveRecheck(
                  scp_iteration_,max_scp_iterations_,q_bar.col(0),
                  &plan_q_bar_,&plan_u_bar_)) {
            previous_query_min_distance_ = std::numeric_limits<double>::quiet_NaN();
            result.repair_feedback = "repair_executable_retained_recheck_reserved";
          }
          publish_next_query = true;
          q_next = plan_q_bar_;
          u_next = plan_u_bar_;
          frame = current_frame_id_;
        }
      }
    }

    // Keep the first six hard failures plus one atomically replaced latest
    // failure. The old first-conflict-only cap omitted terminal coupled QPs.
    // Identity checks above still apply; capture never submits another solve.
    if (!result.repair_hard_qp_snapshot.empty()) {
      const char* root = std::getenv("ROOT_OUT");
      if (root != nullptr && *root != '\0') {
        const int index = repair_hard_qp_snapshots_written_.load();
        const std::string latest = std::string(root) + "/qp_repair_latest_failure.json";
        const std::string path = latest + ".tmp";
        std::ofstream file(path);
        if (file) {
          file << "{\"schema\":1,\"executable\":false,\"plan_seq\":" << solve_plan_sequence
               << ",\"mode_epoch\":" << solve_mode_epoch
               << ",\"observation_token\":"; jsonString(file,solve_observation_token);
          file << ",\"repair_target_revision\":" << solve_repair_ticket.revision
               << ",\"repair_progress_epoch\":" << solve_repair_ticket.progress
               << ",\"scp_iteration\":" << iteration
               << ",\"max_scp_iterations\":" << max_scp_iterations_
               << ",\"reseed_applied\":" << (repair_reseed_applied ? "true" : "false")
               << ",\"margin_rollback_proven\":" << (margin_cause_proven ? "true" : "false")
               << ",\"decision\":"; jsonString(file,result.repair_feedback);
          file << ",\"subproblem\":" << result.repair_hard_qp_snapshot << "}\n";
          file.close();
          if (!file || std::rename(path.c_str(),latest.c_str()) != 0) {
            ROS_WARN_STREAM("[LocalSparseSCPPlanner] incomplete REPAIR snapshot=" << path);
          } else if (index < 6) {
            const std::string first=std::string(root)+"/qp_repair_conflict_"+
                std::to_string(index)+"_"+std::to_string(batch->header.stamp.toNSec())+".json";
            std::ifstream saved(latest); std::ofstream archive(first);
            archive << saved.rdbuf(); archive.close();
            if(archive)repair_hard_qp_snapshots_written_.fetch_add(1);
          }
        } else ROS_WARN_STREAM("[LocalSparseSCPPlanner] cannot write REPAIR snapshot=" << path);
      }
    }

    publishSummary(
        result.solved ? "scp_solved" : "scp_failed",
        &result,
        total_ms);
    if (result.solved && !result.repair_feedback.empty())
      publishSummary(result.repair_feedback,&result,total_ms);

    if(result.status=="known_ground_start_clearance_hold") {
      publishSummary("ground_wait_disposition",&result,total_ms);
      continue; // no QP infeasibility or generic retry counter from a geometry wait
    }
    if (!result.solved && !geometric_restore_applied) {
      ROS_WARN_STREAM(
          "[LocalSparseSCPPlanner] sparse PIQP failed: "
          << result.status);

      if (diagnostic_ran) {
        publishSummary(
            "task_failure_slack_diagnostic",
            &diagnostic,
            0.0);
        ROS_WARN_STREAM(
            "[LocalSparseSCPPlanner] task failure slack diagnostic: "
            << "mode=" << (probe ? "PROBE_NORMAL" : "NORMAL")
            << " hard_status='" << result.status << "'"
            << " soft_solved=" << (diagnostic.solved ? 1 : 0)
            << " soft_status='" << diagnostic.status << "'"
            << " required_max_slack=" << diagnostic.max_slack
            << " required_mean_slack=" << diagnostic.mean_slack
            << " soft_primal=" << diagnostic.primal_residual);
      }

      if (repair_reseed_applied) {
        publishSummary(result.repair_feedback, &result, 0.0);
        publishQueryTrajectory(q_next, u_next, frame, solve_plan_sequence,
                               solve_mode_epoch, &solve_repair_ticket);
        continue;  // No candidate publication from this failed hard solve.
      }

      if (normal_reseed_applied) {
        ROS_WARN_STREAM("[LocalSparseSCPPlanner] NORMAL_BOX_RESEED conflicts="
                        << result.box_conflicting_cdf_rows << " used_iteration=" << iteration+1
                        << " fresh_gcdf_hard_qp_required=1");
        publishQueryTrajectory(q_next, u_next, frame);
        continue;
      }

      if (normal_qp_backtrack_applied) {
        ROS_WARN_STREAM(
            "[LocalSparseSCPPlanner] NORMAL_UNKNOWN_QP_BACKTRACK "
            << "unknown_rows=" << result.selected_unknown_cdf_rows
            << " hard_status='" << result.status
            << "' attempt=" << normal_qp_backtrack_attempt
            << "/" << normal_qp_backtrack_max_attempts
            << " fresh_gcdf_hard_qp_required=1");
        publishSummary("normal_unknown_qp_backtrack_requery", &result, 0.0);
        publishQueryTrajectory(q_next, u_next, frame);
        continue;
      }

      if (restoration_applied) {
        publishSummary(
            "probe_feasibility_restore_applied",
            diagnostic_ran ? &diagnostic : nullptr,
            0.0);
        ROS_WARN_STREAM(
            "[LocalSparseSCPPlanner] C5.25 PROBE feasibility restoration "
            "applied; soft iterate is internal only, fresh GCDF + hard solve "
            "required before candidate publication");

        publishQueryTrajectory(q_next, u_next, frame);
        continue;
      }

      if (repair) {
        publishSummary(result.repair_feedback, &result, total_ms);
      }
      if (!repair) {
        bool stalled = false;
        int failures = 0;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          if (solve_mode_epoch != mode_epoch_ || solve_plan_sequence != plan_sequence_ || repair != repair_mode_ ||
              probe != probe_mode_) continue;
          Eigen::VectorXd measured;
          if (!extractMeasuredQ(latest_joint_state_, measured)) measured = q_bar.col(0);
          stalled = task_no_progress_.fail(measured);
          failures = task_no_progress_.failures;
          if (stalled) plan_requested_ = false;
        }
        if (stalled) {
          std::lock_guard<std::mutex> lock(mutex_);
          if (solve_mode_epoch != mode_epoch_ || solve_plan_sequence != plan_sequence_ ||
              !task_no_progress_.blocked()) continue;
          std_msgs::String msg;
          std::ostringstream s;
          s << "status=stalled classification=QP_NO_PROGRESS_UNCERTIFIED"
            << " failures=" << failures << " query_stamp_ns=" << batch->header.stamp.toNSec()
            << " snapshot=disabled";
          msg.data = s.str(); task_stall_pub_.publish(msg);
          ROS_ERROR_STREAM("[LocalSparseSCPPlanner] " << msg.data);
          // No infeasibility claim or soft candidate. The regime manager
          // requests same-query exact VBC; row presence is not blocker evidence.
          continue;
        }
        if (result.status.find("primal infeasible") != std::string::npos) {
          // A certificate for this linearized QP does not identify which
          // UNKNOWN/OCCUPIED rows caused it. Never turn row presence into a
          // visibility obligation (the final-GCDF rejection path is unchanged).
          std::lock_guard<std::mutex> lock(mutex_);
          if (solve_mode_epoch != mode_epoch_ || solve_plan_sequence != plan_sequence_) continue;
          std_msgs::String msg;
          qp_retry_plan_sequence_ = solve_plan_sequence;
          qp_retry_mode_epoch_ = solve_mode_epoch;
          qp_retry_token_ = std::to_string(ros::WallTime::now().toNSec()) + "_" +
              std::to_string(solve_plan_sequence) + "_" +
              std::to_string(batch->header.stamp.toNSec());
          msg.data = "status=retry classification=QP_PRIMAL_INFEASIBLE_UNATTRIBUTED token=" +
              qp_retry_token_ + " mode=" + (probe ? "PROBE_NORMAL" : "NORMAL") +
              " failures=" + std::to_string(failures) +
              " query_stamp_ns=" + std::to_string(batch->header.stamp.toNSec());
          task_stall_pub_.publish(msg);
          ROS_WARN_STREAM("[LocalSparseSCPPlanner] " << msg.data
                          << " unknown_rows=" << result.selected_unknown_cdf_rows
                          << " occupied_rows=" << result.selected_occupied_cdf_rows
                          << " -> bounded retry; no recovery evidence published");
        } else if (
            result.status.find("max iterations") != std::string::npos ||
            result.status.find("maximum iterations") != std::string::npos) {
          std_msgs::Bool uncertified_msg;
          uncertified_msg.data = true;
          task_uncertified_pub_.publish(uncertified_msg);
          ROS_WARN_STREAM(
              "[LocalSparseSCPPlanner] task QP uncertified in "
              << (probe ? "PROBE_NORMAL" : "NORMAL")
              << " status='" << result.status
              << "' primal_res=" << result.primal_residual
              << " -> publish task uncertified signal");
        }
      }
      continue;
    }

    if (publish_candidate) {
      bool mode_stale = false;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        mode_stale =
            solve_mode_epoch != mode_epoch_ ||
            solve_plan_sequence != plan_sequence_ ||
            repair != repair_mode_ ||
            probe != probe_mode_ ||
            (repair_prefix_binding_enabled_ && repair && !probe &&
             !prefixSelectionCurrentLocked(solve_prefix_selection)) ||
            (repair && !repair_no_progress_.current(solve_repair_ticket)) ||
            (!executable_context.empty() &&
             (executable_context != repairExecutableContextLocked() ||
              repair_witness_requires_reobserve_ || repair_witness_overflow_));
        if (mode_stale) {
          ++stale_mode_candidate_discard_count_;
        } else {
          // Publish under the same identity lock as the check: a callback
          // cannot replace the target in between validation and publication.
          publishCandidateTrajectory(
              result.q, result.u, current_frame_id_, solve_observation_token, *batch, q_bar, u_bar,
              result.normal_blocking_unknown_points,
              geometric_restore_applied ? result.geometric_restore_guard_points : result.primitive_margin_recovery_points,
              !geometric_restore_applied && !result.primitive_margin_recovery_points.empty(),
              geometric_restore_applied ? result.fixed_start_recovery_points : std::vector<Eigen::Vector3d>{});
          if (repair && !observation_phase)
            requestPlanLocked("repair_observation_after_safe_retreat");
        }
      }
      if (mode_stale) {
        publishSummary("stale_mode_candidate_discarded", &result, total_ms);
        ROS_INFO_STREAM(
            "[LocalSparseSCPPlanner] stale-mode candidate discarded before "
            "commit pipeline");
        continue;
      }

      publishSummary(geometric_restore_applied ? (result.fixed_start_recovery_points.empty() ?
          "geometric_restore_candidate_published" : "fixed_start_recovery_candidate_published") :
          "candidate_published", &result, total_ms);
      continue;
    }

    if (publish_next_query) {
      if (repair) publishQueryTrajectory(q_next,u_next,frame,solve_plan_sequence,
                                         solve_mode_epoch,&solve_repair_ticket);
      else publishQueryTrajectory(q_next,u_next,frame);
    }

    (void)iteration;
  }
}

bool LocalSparseSCPPlanner::freeJointTaskImproves(
    const SparseSolveResult& primary, const SparseSolveResult& trial,
    const std::vector<DeadlineWaypoint>& schedule,
    const FrontierObjective& frontier, const Eigen::MatrixXd& q_ref) const {
  if (!primary.solved || !trial.solved || primary.max_slack > 1e-9 || trial.max_slack > 1e-9 ||
      !primary.primitive_executable_valid || !trial.primitive_executable_valid ||
      primary.primitive_live_rebase_required || trial.primitive_live_rebase_required ||
      schedule.size()!=1 || !frontier.active || frontier.recovery_attempt!=0 ||
      schedule[0].q.size()!=dof_ || schedule[0].joint_mask.size()!=dof_ ||
      frontier.q.size()!=dof_ || frontier.joint_mask.size()!=dof_ ||
      primary.q.rows()!=dof_ || primary.q.cols()!=num_intervals_+1 ||
      trial.q.rows()!=dof_ || trial.q.cols()!=num_intervals_+1 ||
      q_ref.rows()!=dof_ || q_ref.cols()!=num_intervals_+1 ||
      !finiteMatrix(primary.q) || !finiteMatrix(trial.q) || !finiteMatrix(q_ref) ||
      !finiteVector(schedule[0].q) || !finiteVector(frontier.q)) return false;
  const auto& mask=schedule[0].joint_mask;
  if (!finiteVector(mask) || !finiteVector(frontier.joint_mask) ||
      (mask-frontier.joint_mask).cwiseAbs().maxCoeff()>1e-12) return false;
  int free_count=0;
  double primary_task=0.,trial_task=0.;
  for(int j=0;j<dof_;++j) {
    if(mask[j]!=0. && mask[j]!=1.) return false;
    if(mask[j]==1.) {
      // Compare the actual primary result, rather than a normalized gradient.
      // This covers every observation-joint knot, including its tail target.
      if((trial.q.row(j)-primary.q.row(j)).cwiseAbs().maxCoeff()>1e-5) return false;
    } else {
      ++free_count;
      primary_task+=(primary.q.row(j)-q_ref.row(j)).squaredNorm();
      trial_task+=(trial.q.row(j)-q_ref.row(j)).squaredNorm();
    }
  }
  const int k=std::max(1,std::min(visibility_frontier_horizon_step_,num_intervals_));
  const auto distance=[&](const Eigen::MatrixXd& q,const Eigen::VectorXd& target,int step) {
    return ((q.col(step)-target).cwiseProduct(mask)).squaredNorm();
  };
  return free_count>0 && trial_task+1e-8<primary_task &&
      distance(trial.q,frontier.q,k)<=distance(primary.q,frontier.q,k)+1e-9 &&
      distance(trial.q,schedule[0].q,num_intervals_)<=distance(primary.q,schedule[0].q,num_intervals_)+1e-9;
}

int LocalSparseSCPPlanner::qIndex(int k, int j) const {
  // q_0 is measured/fixed; decision q starts at q_1.
  return (k - 1) * dof_ + j;
}

int LocalSparseSCPPlanner::uIndex(int k, int j) const {
  return num_intervals_ * dof_ + k * dof_ + j;
}

LocalSparseSCPPlanner::SparseSolveResult
LocalSparseSCPPlanner::solveSparseSubproblem(
    const care_collision_cdf::CollisionCDFConstraintBatch& batch,
    const Eigen::MatrixXd& q_bar,
    const Eigen::MatrixXd& u_bar,
    const Eigen::MatrixXd& q_ref,
    const Eigen::MatrixXd& u_ref,
    const Eigen::VectorXd& previous_command,
    const std::vector<DeadlineWaypoint>& schedule,
    const FrontierObjective& frontier,
    const std::vector<Eigen::Vector3d>& obligation_points,
    bool repair_mode,
    bool probe_mode,
    double trust_radius,
    double slack_linear_weight,
    bool force_diagnostic_slack,
    bool previous_vbc_margins,
    double executable_step_scale,
    bool executable_recheck_only,
    const RepairPrefixSelection* prefix_selection,
    bool free_joint_task, const ViewPreviewSnapshot* preview, ViewPreviewSnapshot* capture) const {
  const auto& geometry_guard=preview ? *preview->guard : primitive_observation_guard_;
  SparseSolveResult out;
  const char* recovery_policy = std::getenv("CARE_REPAIR_UNKNOWN_SOFT");
  const bool repair_unknown_soft = repair_mode && !probe_mode && !force_diagnostic_slack &&
      recovery_policy && std::string(recovery_policy) == "true";
  const bool task_unknown_soft = executable_task_research_ && !force_diagnostic_slack;
  const bool unknown_cdf_soft = repair_unknown_soft || task_unknown_soft;
  out.repair_unknown_soft = unknown_cdf_soft;
  const bool normal_motion = repair_executable_view_enabled_ && !repair_mode && !probe_mode;
  const bool repair_motion = repair_executable_view_enabled_ && repair_mode && !probe_mode;
  const bool nodal_motion = normal_motion || repair_motion;
  const bool acceleration_constraints = enforce_acceleration_constraints_ || nodal_motion;
  if (nodal_motion && (previous_command.size() != dof_ || !finiteVector(previous_command))) {
    out.status = normal_motion ? "normal_measured_velocity_missing" : "repair_measured_velocity_missing";
    return out;
  }
  const char* prefix_objective_policy=std::getenv("CARE_PREFIX_TASK_OBJECTIVE");
  const bool normal_prefix_task_objective=normal_motion && !force_diagnostic_slack &&
      prefix_objective_policy && std::string(prefix_objective_policy)=="true";
  Eigen::Isometry3d prefix_task_goal=Eigen::Isometry3d::Identity();
  bool prefix_task_goal_valid=false;
  if(preview) {prefix_task_goal=preview->goal;prefix_task_goal_valid=preview->goal_valid;}
  else {std::lock_guard<std::mutex> lock(mutex_);
    prefix_task_goal=terminal_goal_;prefix_task_goal_valid=terminal_goal_valid_;}
  if(capture) {
    capture->native_task_replay=true;capture->goal=prefix_task_goal;
    capture->goal_valid=prefix_task_goal_valid;capture->guard=geometry_guard.cloneForPreview();
  }
  Eigen::Matrix<double,6,1> current_task_error;
  // Five task tolerances define a soft objective activation zone. No motion
  // is rejected by this condition, and far observation motions keep their cost.
  const bool repair_prefix_task_objective=repair_goal_stopping_enabled_ &&
      repair_motion && !force_diagnostic_slack && prefix_task_goal_valid &&
      taskPoseResidual(geometry_guard,q_bar.col(0),terminal_ee_frame_,prefix_task_goal,&current_task_error) &&
      current_task_error.head<3>().norm()<=1. && current_task_error.tail<3>().norm()<=1.;
  const bool prefix_task_objective=normal_prefix_task_objective || repair_prefix_task_objective;
  auto executable_settings=repair_executable_settings_;
  if(repair_motion)executable_settings.initial_velocity=previous_command;
  if (repair_prefix_binding_enabled_ && repair_mode && !probe_mode) {
    if (!prefix_selection || !prefixSelectionCompatible(*prefix_selection)) {
      out.status="missing_or_invalid_selected_prefix";return out;
    }
    executable_settings.prefixes={prefix_selection->prefix_s};
  }
  out.executable_step_scale = executable_step_scale;
  out.executable_recheck_only = executable_recheck_only;
  if (!std::isfinite(executable_step_scale) || executable_step_scale <= 0. ||
      executable_step_scale > 1. ||
      (executable_recheck_only && (!repair_mode || probe_mode ||
          force_diagnostic_slack || cdf_slack_enabled_))) {
    out.status = "executable_iterate_contract_error";
    return out;
  }
  // Callbacks may retire or replace a witness while this worker is solving.
  // The entire subproblem uses one immutable witness snapshot.
  care_confidence_map::InitialShellRecovery preview_shell;
  if(preview)preview_shell=preview->shell;
  std::vector<RepairWitness> witnesses;
  std::vector<PrimitiveObservationPoint> path_guards;
  std::vector<Eigen::Vector3d> active_observation_points;
  if(preview) {
    witnesses=preview->witnesses;path_guards=preview->path_guards;
    active_observation_points=preview->points;
  } else {
    std::lock_guard<std::mutex> lock(mutex_);
    witnesses = repair_unknown_witnesses_;
    active_observation_points = latest_vbc_obligation_points_;
    if(restore_path_guard_scope_==std::to_string(mode_epoch_)+":"+latest_observation_token_)
      path_guards=restore_path_guard_points_;
  }
  if(capture) {
    capture->witnesses=witnesses;capture->path_guards=path_guards;
    capture->points=active_observation_points;
    std::lock_guard<std::mutex> lock(mutex_);capture->shell=initial_shell_recovery_;
  }
  const bool slack_enabled =
      cdf_slack_enabled_ || force_diagnostic_slack || unknown_cdf_soft;
  const bool per_constraint_slack =
      cdf_per_constraint_slack_ || force_diagnostic_slack || unknown_cdf_soft;
  out.slack_linear_weight_used = slack_linear_weight;
  out.batch_pairs = batch.num_pairs;
  out.min_distance = std::numeric_limits<double>::infinity();

  if (q_bar.rows() != dof_ ||
      q_bar.cols() != num_intervals_ + 1 ||
      u_bar.rows() != dof_ ||
      u_bar.cols() != num_intervals_ ||
      q_ref.rows() != dof_ ||
      q_ref.cols() != num_intervals_ + 1 ||
      u_ref.rows() != dof_ ||
      u_ref.cols() != num_intervals_) {
    out.status = "trajectory_dimension_error";
    return out;
  }
  std::vector<RepairExecutableSample> cdf_executable_samples;
  if(repair_motion) {
    std::string error;
    if(!repairCdfExecutableSamples(q_bar,u_bar,dt_,executable_settings,&cdf_executable_samples,&error)) {
      out.status="repair_cdf_executable_contract_error";return out;
    }
  }

  const int n_pairs = batch.num_pairs;
  if (unknown_cdf_soft && (batch.source_type.size() != static_cast<std::size_t>(n_pairs) ||
      std::any_of(batch.source_type.begin(), batch.source_type.end(), [](uint8_t source) {
        return source != care_collision_cdf::CollisionCDFConstraintBatch::SOURCE_UNKNOWN &&
               source != care_collision_cdf::CollisionCDFConstraintBatch::SOURCE_OCCUPIED;
      }))) {
    out.status = "recovery_cdf_source_invalid"; return out;
  }
  if (n_pairs < 0 ||
      batch.dof != dof_ ||
      batch.original_timestep.size() !=
          static_cast<std::size_t>(n_pairs) ||
      (!batch.source_type.empty() &&
       batch.source_type.size() != static_cast<std::size_t>(n_pairs)) ||
      batch.distance.size() !=
          static_cast<std::size_t>(n_pairs) ||
      batch.q_linearization_flat.size() !=
          static_cast<std::size_t>(n_pairs * dof_) ||
      batch.gradient_flat.size() !=
          static_cast<std::size_t>(n_pairs * dof_)) {
    out.status = "cdf_batch_dimension_error";
    return out;
  }

  struct SelectedRow {
    int pair = -1;
    int k = -1;
    bool repair_witness = false;
    bool obligation = false;
    int executable_sample = -1;
    double d = 0.0;
    double safety_margin = 0.0;
    Eigen::VectorXd g;
    Eigen::VectorXd qlin;
  };
  std::vector<SelectedRow> selected;
  selected.reserve(static_cast<std::size_t>(n_pairs));

  // The row classifier receives both near-term visibility-target guards and
  // confirmed GCDF repair witnesses. A VBC target remains identifiable while
  // the planner is in PROBE_NORMAL, so do not gate this identity check on
  // repair_mode. All rows keep the existing finite-value and safe-row
  // screening behavior.
  auto is_obligation_row = [&](int pair) {
    if (obligation_points.empty() || pair < 0 ||
        batch.point_flat.size() < static_cast<std::size_t>(3 * (pair + 1))) {
      return false;
    }
    const Eigen::Vector3d point(
        batch.point_flat[static_cast<std::size_t>(3 * pair)],
        batch.point_flat[static_cast<std::size_t>(3 * pair + 1)],
        batch.point_flat[static_cast<std::size_t>(3 * pair + 2)]);
    if (!point.allFinite()) return false;
    return std::any_of(
        obligation_points.begin(), obligation_points.end(),
        [&](const Eigen::Vector3d& obligation_point) {
          return (point - obligation_point).lpNorm<Eigen::Infinity>() <= 1e-5;
        });
  };

  // A final-GCDF UNKNOWN/occupied witness is a confirmed hard-safety point,
  // just like the active visibility obligation.  The request builder carries
  // each witness through the full task horizon so a late brake/hold knot can
  // be checked too.  Do not let the ordinary near-term CDF horizon silently
  // discard those rows after they have been re-queried.
  auto is_repair_witness_row = [&](int pair) {
    if (witnesses.empty() || pair < 0 ||
        batch.point_flat.size() < static_cast<std::size_t>(3 * (pair + 1))) {
      return false;
    }
    const Eigen::Vector3d point(
        batch.point_flat[static_cast<std::size_t>(3 * pair)],
        batch.point_flat[static_cast<std::size_t>(3 * pair + 1)],
        batch.point_flat[static_cast<std::size_t>(3 * pair + 2)]);
    if (!point.allFinite()) return false;
    return std::any_of(
        witnesses.begin(), witnesses.end(),
        [&](const RepairWitness& witness) {
          return (point - witness.point).lpNorm<Eigen::Infinity>() <= 1e-5;
        });
  };

  auto repair_witness_margin_for_row = [&](int pair) {
    if (witnesses.empty() || pair < 0 ||
        batch.point_flat.size() < static_cast<std::size_t>(3 * (pair + 1))) {
      return cdf_safety_margin_;
    }
    const Eigen::Vector3d point(
        batch.point_flat[static_cast<std::size_t>(3 * pair)],
        batch.point_flat[static_cast<std::size_t>(3 * pair + 1)],
        batch.point_flat[static_cast<std::size_t>(3 * pair + 2)]);
    double margin = cdf_safety_margin_;
    for (const auto& witness : witnesses) {
      if ((point - witness.point).lpNorm<Eigen::Infinity>() > 1e-5 ||
          !std::isfinite(witness.safety_margin)) {
        continue;
      }
      margin = std::max(margin, previous_vbc_margins && witness.previous_margin >= 0.
          ? witness.previous_margin : witness.safety_margin);
    }
    return margin;
  };

  std::vector<Eigen::Vector3d> fresh_unknown_points;
  for (int i = 0; i < n_pairs; ++i) {
    const int k =
        batch.original_timestep[static_cast<std::size_t>(i)];
    const bool obligation_row = is_obligation_row(i);
    if (k == 0) {
      ++out.skipped_step0_rows;
      continue;
    }
    const bool repair_witness_row = is_repair_witness_row(i);
    const int sample_index=k-num_intervals_-1;
    const bool executable_row=repair_motion && sample_index>=0 &&
        sample_index<static_cast<int>(cdf_executable_samples.size()) && (obligation_row || repair_witness_row);
    if (k < 1 || (k > num_intervals_ && !executable_row)) {
      ++out.skipped_horizon_rows;
      continue;
    }
    const bool executable_prefix_mode = repair_mode || probe_mode;
    // The active visibility obligation is an UNKNOWN target. Keep its hard
    // guard over the complete requested obligation horizon so the body cannot
    // sweep through the point before the sensor has observed it. A confirmed
    // final-GCDF witness remains hard over the complete horizon as well. The
    // active target is still kept out of the persistent witness list; this is
    // a full-horizon query identity, not a permanent GCDF wall.
    if (executable_prefix_mode && k > cdf_constraint_horizon_steps_ &&
        !repair_witness_row && !obligation_row) {
      ++out.skipped_safety_horizon_rows;
      continue;
    }

    const double d =
        batch.distance[static_cast<std::size_t>(i)];
    Eigen::VectorXd g(dof_);
    Eigen::VectorXd qlin(dof_);
    bool finite = std::isfinite(d);
    for (int j = 0; j < dof_; ++j) {
      g[j] = batch.gradient_flat[
          static_cast<std::size_t>(i * dof_ + j)];
      qlin[j] = batch.q_linearization_flat[
          static_cast<std::size_t>(i * dof_ + j)];
      finite = finite &&
               std::isfinite(g[j]) &&
               std::isfinite(qlin[j]);
    }
    if (!finite) {
      ++out.skipped_horizon_rows;
      continue;
    }

    out.min_distance = std::min(out.min_distance, d);
    out.qlin_error_inf = std::max(
        out.qlin_error_inf,
        ((executable_row ? cdf_executable_samples[sample_index].q : q_bar.col(k).eval()) - qlin).lpNorm<Eigen::Infinity>());

    if(!batch.source_type.empty() && batch.source_type[i]==care_collision_cdf::CollisionCDFConstraintBatch::SOURCE_UNKNOWN &&
        batch.point_flat.size()==static_cast<std::size_t>(3*n_pairs) && fresh_unknown_points.size()<32) {
      const Eigen::Vector3d p(batch.point_flat[3*i],batch.point_flat[3*i+1],batch.point_flat[3*i+2]);
      if(p.allFinite() && std::none_of(fresh_unknown_points.begin(),fresh_unknown_points.end(),
          [&](const Eigen::Vector3d& old){return (old-p).lpNorm<Eigen::Infinity>()<=1e-5;}))fresh_unknown_points.push_back(p);
    }

    // Active visibility obligations and confirmed GCDF witnesses are
    // persistent hard guards. Keep their rows in the QP even when the current
    // linearization is temporarily clear: otherwise the optimizer can move
    // through a nonlinear/continuous-sweep boundary between re-queries. The
    // ordinary environment rows retain the inexpensive safe-row screening.
    if (cdf_safe_row_screening_ && !obligation_row && !repair_witness_row) {
      const double worst_linearized =
          d - trust_radius * g.lpNorm<1>();
      if (worst_linearized >= cdf_safety_margin_) {
        ++out.screened_safe_rows;
        continue;
      }
    }

    const uint8_t source_type =
        batch.source_type.empty()
            ? care_collision_cdf::CollisionCDFConstraintBatch::SOURCE_UNKNOWN
            : batch.source_type[static_cast<std::size_t>(i)];
    if (obligation_row) ++out.selected_obligation_cdf_rows;
    if (source_type ==
        care_collision_cdf::CollisionCDFConstraintBatch::SOURCE_OCCUPIED) {
      ++out.selected_occupied_cdf_rows;
    } else {
      ++out.selected_unknown_cdf_rows;
      out.selected_unknown_pair_indices.push_back(i);
    }

    SelectedRow row;
    row.pair = i;
    row.k = k;
    row.repair_witness = repair_witness_row;
    row.obligation = obligation_row;
    row.executable_sample = executable_row ? sample_index : -1;
    row.d = d;
    row.safety_margin = obligation_row
        ? visibility_obligation_cdf_margin_effective_.load()
        : cdf_safety_margin_;
    // Repair witnesses also appear in obligation_points to retain their
    // full-horizon guards. That shared classification must not replace the
    // witness's point-specific margin with the observation margin.
    // Keep the stricter requirement when both identities apply.
    if (repair_witness_row) {
      row.safety_margin = std::max(
          row.safety_margin, repair_witness_margin_for_row(i));
    }
    row.g = g;
    row.qlin = qlin;
    // Declare every deficient owned UNKNOWN point, not just the first-step
    // box witness. Other owned rows can remain below the final learned gate
    // after that first step; native proof must explicitly cover them too.
    if(source_type==care_collision_cdf::CollisionCDFConstraintBatch::SOURCE_UNKNOWN &&
        (row.obligation || row.repair_witness) && d<row.safety_margin) {
      const Eigen::Vector3d point(batch.point_flat[3*i],batch.point_flat[3*i+1],batch.point_flat[3*i+2]);
      if(std::none_of(out.geometric_restore_guard_points.begin(),out.geometric_restore_guard_points.end(),
          [&](const Eigen::Vector3d& old){return (old-point).lpNorm<Eigen::Infinity>()<=1e-5;}))
        out.geometric_restore_guard_points.push_back(point);
    }
    selected.push_back(std::move(row));
  }

  out.selected_cdf_rows = static_cast<int>(selected.size());
  if (unknown_cdf_soft) {
    // A proposal cost must not grow with the number of sampled knots/points.
    // Occupied slacks stay zero. Use an average
    // UNKNOWN penalty so repeated samples cannot overwhelm the view objective.
    slack_linear_weight /= std::max(1, out.selected_unknown_cdf_rows);
    out.slack_linear_weight_used = slack_linear_weight;
  }

  if (!std::isfinite(out.min_distance))
    out.min_distance = std::numeric_limits<double>::quiet_NaN();

  if (n_pairs == 0) {
    ROS_INFO(
        "[LocalSparseSCPPlanner] received explicit empty CDF batch: no active forbidden pairs");
  }

  if (out.qlin_error_inf > cdf_linearization_tolerance_inf_) {
    out.status = "cdf_linearization_mismatch";
    return out;
  }

  // Fresh UNKNOWN rows already present in this exact query also describe the
  // neighboring approach corridor. Give those points analytic20mm guards on
  // the executable prefix instead of discovering them only at final VBC.
  // Active owned points keep their original restoration semantics. These
  // identities are transient: a later resolved-free response does not retain
  // an obsolete point wall. Every original native-CDF row remains unchanged.
  std::vector<Eigen::Vector3d> transient_path_points;
  if(unknown_corridor_guard_enabled_ && primitive_observation_guard_enabled_ && repair_executable_view_enabled_ &&
      repair_prefix_binding_enabled_ && initial_shell_recovery_enabled_ && repair_mode && !probe_mode &&
      batch.point_flat.size()==static_cast<std::size_t>(3*n_pairs) && !batch.source_type.empty()) {
    for(const auto& p:fresh_unknown_points) {
      if(!p.allFinite() || std::any_of(active_observation_points.begin(),active_observation_points.end(),
          [&](const Eigen::Vector3d& old){return (old-p).lpNorm<Eigen::Infinity>()<=1e-5;}))continue;
      if(std::any_of(path_guards.begin(),path_guards.end(),[&](const PrimitiveObservationPoint& old){
          return (old.point-p).lpNorm<Eigen::Infinity>()<=1e-5;}) ||
          std::any_of(witnesses.begin(),witnesses.end(),[&](const RepairWitness& old){
            return old.has_primitive_guard && (old.point-p).lpNorm<Eigen::Infinity>()<=1e-5;}))continue;
      if(path_guards.size()+witnesses.size()>=32)break;
      PrimitiveObservationPoint point;point.point=p;point.ordinary_path_guard=true;
      // Reuse the native auditor's optional2mm certificate-quality target as
      // a future-only soft cost. The ramp below starts at zero extra margin;
      // neither the fixed start nor the original20mm hard floor is raised.
      point.proposal_reserve_m=std::min(.002,primitive_observation_margin_m_-initial_shell_execution_margin_m_);
      path_guards.push_back(point);transient_path_points.push_back(p);
    }
  }

  // CARE historically shared one user slack across all CDF rows at a
  // timestep. G0 can switch to the GCDF convention: one slack per inequality.
  std::vector<int> step_to_slack(
      static_cast<std::size_t>(num_intervals_ + 1+cdf_executable_samples.size()), -1);
  int n_s = 0;
  if (slack_enabled) {
    if (per_constraint_slack) {
      n_s = static_cast<int>(selected.size());
    } else {
      for (const auto& row : selected) {
        if (step_to_slack[static_cast<std::size_t>(row.k)] < 0) {
          step_to_slack[static_cast<std::size_t>(row.k)] = n_s++;
        }
      }
    }
  }

  const int n_q = num_intervals_ * dof_;
  const int n_u = num_intervals_ * dof_;

  const int n_eq = num_intervals_ * dof_;
  // Acceleration rows include the executed-command -> u0 boundary, all
  // inter-stage velocity changes, and u_{K-1} -> 0 terminal braking.
  const int n_acc =
      acceleration_constraints
          ? (num_intervals_ + 1) * dof_
          : 0;
  const int n_cdf_rows = static_cast<int>(selected.size());
  std::vector<PrimitiveObservationGuard::Row> primitive_rows;
  std::vector<RepairExecutableSample> executable_samples;
  std::vector<PrimitiveObservationPoint> primitive_points;
  std::map<care_confidence_map::InitialShellRecovery::Key,double> primitive_floors;
  std::map<care_confidence_map::InitialShellRecovery::Key,double> primitive_proposal_targets;
  std::set<care_confidence_map::InitialShellRecovery::Key> shell_points;
  if (primitive_observation_guard_enabled_ && repair_mode && !probe_mode) {
    for (const auto& w : witnesses) if (w.has_primitive_guard) primitive_points.push_back(w.primitive_guard);
    for(const auto& guard:path_guards) {
      const auto old=std::find_if(primitive_points.begin(),primitive_points.end(),[&](const PrimitiveObservationPoint& p){
        return (p.point-guard.point).lpNorm<Eigen::Infinity>()<=1e-5;});
      if(old==primitive_points.end())primitive_points.push_back(guard);
      else {old->ordinary_path_guard=true;old->proposal_reserve_m=std::max(old->proposal_reserve_m,guard.proposal_reserve_m);}
    }
    out.geometric_restore_guard_points.erase(std::remove_if(out.geometric_restore_guard_points.begin(),
        out.geometric_restore_guard_points.end(),[&](const Eigen::Vector3d& p){
          return std::any_of(path_guards.begin(),path_guards.end(),[&](const PrimitiveObservationPoint& guard){
            return (p-guard.point).lpNorm<Eigen::Infinity>()<=1e-5;});
        }),out.geometric_restore_guard_points.end());
    out.primitive_observation_points = static_cast<int>(primitive_points.size());
    if (!primitive_points.empty()) {
      std::string error;
      Eigen::MatrixXd geometry_q = q_bar;
      if (repair_executable_view_enabled_) {
        if (!repairExecutableSamples(q_bar,u_bar,dt_,executable_settings,&executable_samples,&error)) {
          out.status = "primitive_executable_contract_error"; return out;
        }
        out.primitive_executable_samples = executable_samples.size();
        out.primitive_executable_numerical_tightening_m = 2.*piqp_eps_abs_;
        geometry_q.conservativeResize(Eigen::NoChange,q_bar.cols()+executable_samples.size());
        for (std::size_t i=0;i<executable_samples.size();++i) geometry_q.col(q_bar.cols()+i)=executable_samples[i].q;
      }
      if (!geometry_guard.linearize(geometry_q, primitive_points, &primitive_rows, &error)) {
        ROS_ERROR_STREAM("Primitive observation guard failed closed: " << error);
        out.status = "primitive_observation_geometry_error";
        return out;
      }
      double start_distance = std::numeric_limits<double>::infinity();
      for (const auto& row : primitive_rows) if (row.timestep == 0)
        start_distance = std::min(start_distance, row.distance_m);
      out.primitive_observation_start_distance_m = start_distance;
      if (!initial_shell_recovery_enabled_ && start_distance <= primitive_observation_margin_m_ + 1e-12) {
        out.status = "primitive_observation_start_blocked";
        return out;
      }
      for (const auto& point : primitive_points) {
        double start = std::numeric_limits<double>::infinity();
        for (const auto& row : primitive_rows) if (row.timestep==0 &&
            (row.point-point.point).lpNorm<Eigen::Infinity>()<=1e-5) start=std::min(start,row.distance_m);
        double required=primitive_observation_margin_m_; bool recovering=false;
        // Fresh UNKNOWN provenance binds this point to the current query.
        // Reuse the existing10mm execution floor and continuous certificate;
        // the20mm proposal reserve remains an objective, not a fixed-q veto.
        const bool ordinary_execution_floor=point.ordinary_path_guard && repair_prefix_binding_enabled_ &&
            std::any_of(fresh_unknown_points.begin(),fresh_unknown_points.end(),
                [&](const Eigen::Vector3d& p){return (p-point.point).lpNorm<Eigen::Infinity>()<=1e-5;});
        if (initial_shell_recovery_enabled_ && (!point.ordinary_path_guard || ordinary_execution_floor)) {
          std::lock_guard<std::mutex> lock(mutex_);
          auto& shell=preview ? preview_shell : initial_shell_recovery_;
          if (!shell.floor(point.point,start,primitive_observation_margin_m_,
                  initial_shell_execution_margin_m_,&required,&recovering,repair_prefix_binding_enabled_)) {
            out.status="primitive_observation_start_blocked"; return out;
          }
          if (repair_prefix_binding_enabled_) out.primitive_margin_recovery_points.push_back(point.point);
        }
        const auto key=care_confidence_map::InitialShellRecovery::key(point.point);
        primitive_floors[key]=required;
        if(fixedStartUnknownBlocked(start,required,point.ordinary_path_guard)) {
          out.status="fixed_start_unknown_clearance";
          out.fixed_start_unknown_blocked=true;out.fixed_start_unknown_point=point.point;
          out.fixed_start_unknown_distance_m=start;out.fixed_start_unknown_floor_m=required;
          // Retain the failed fixed-start result. A separate, bounded geometry
          // proposal may restore only fresh UNKNOWN points already in (10,20)mm.
          // All ordinary guards survive this handoff with their original role.
          if(fixed_start_recovery_enabled_ && repair_prefix_binding_enabled_ &&
              initial_shell_recovery_enabled_ && !force_diagnostic_slack && !cdf_slack_enabled_) {
            bool eligible=true;
            for(const auto& guard:primitive_points) {
              if(!guard.ordinary_path_guard)continue;
              double distance=std::numeric_limits<double>::infinity();
              for(const auto& row:primitive_rows)if(row.timestep==0 &&
                  (row.point-guard.point).lpNorm<Eigen::Infinity>()<=1e-5)
                distance=std::min(distance,row.distance_m);
              if(!fixedStartUnknownBlocked(distance,primitive_observation_margin_m_,true))continue;
              const bool fresh=std::any_of(fresh_unknown_points.begin(),fresh_unknown_points.end(),
                  [&](const Eigen::Vector3d& p){return (p-guard.point).lpNorm<Eigen::Infinity>()<=1e-5;});
              if(!fresh || distance<=initial_shell_execution_margin_m_+1e-6) {eligible=false;break;}
              out.fixed_start_recovery_points.push_back(guard.point);
            }
            if(!eligible || out.fixed_start_recovery_points.size()>32)out.fixed_start_recovery_points.clear();
            if(!out.fixed_start_recovery_points.empty()) {
              out.geometric_restore_path_guards=primitive_points;
              for(const auto& p:out.fixed_start_recovery_points)
                if(std::none_of(out.geometric_restore_guard_points.begin(),out.geometric_restore_guard_points.end(),
                    [&](const Eigen::Vector3d& old){return (old-p).lpNorm<Eigen::Infinity>()<=1e-5;}))
                  out.geometric_restore_guard_points.push_back(p);
            }
          }
          return out; // fixed q_0 cannot satisfy this hard row; do not run PIQP.
        }
        if(point.ordinary_path_guard && (point.proposal_reserve_m>0. || ordinary_execution_floor))
          primitive_proposal_targets[key]=std::max(required,primitive_observation_margin_m_)+point.proposal_reserve_m;
        if (recovering) shell_points.insert(key);
      }
      out.primitive_shell_recovery_points=shell_points.size();
      primitive_rows.erase(std::remove_if(primitive_rows.begin(), primitive_rows.end(),
          [&](const PrimitiveObservationGuard::Row& row) {
            if(row.timestep==0)return true;
            const double prefix=executable_settings.prefixes.empty()?0.:
                *std::max_element(executable_settings.prefixes.begin(),executable_settings.prefixes.end());
            if(row.timestep>=q_bar.cols() || row.timestep*dt_<=prefix+1e-9)return false;
            return std::any_of(transient_path_points.begin(),transient_path_points.end(),[&](const Eigen::Vector3d& p){
              return (row.point-p).lpNorm<Eigen::Infinity>()<=1e-5;});
          }), primitive_rows.end());
      for (const auto& row : primitive_rows) if (row.timestep >= q_bar.cols()) ++out.primitive_executable_rows;
    }
  }
  std::vector<Eigen::Vector3d> occupied_volume_points;
  const char* volume_feature=std::getenv("CARE_OCCUPIED_VOLUME_QP_ENABLED");
  const bool occupied_volume_enabled=volume_feature && std::string(volume_feature)=="true";
  if(occupied_volume_enabled) {
    if(batch.source_type.size()!=static_cast<std::size_t>(n_pairs) || batch.point_flat.size()!=static_cast<std::size_t>(3*n_pairs)) {
      out.status="occupied_volume_source_identity_missing";return out;
    }
    for(int i=0;i<n_pairs;++i)if(batch.source_type[i]==care_collision_cdf::CollisionCDFConstraintBatch::SOURCE_OCCUPIED) {
      Eigen::Vector3d point(batch.point_flat[3*i],batch.point_flat[3*i+1],batch.point_flat[3*i+2]);
      if(!point.allFinite()){out.status="occupied_volume_nonfinite_point";return out;}
      if(std::none_of(occupied_volume_points.begin(),occupied_volume_points.end(),[&](const Eigen::Vector3d& old){return (old-point).lpNorm<Eigen::Infinity>()<=1e-5;}))
        occupied_volume_points.push_back(point);
    }
    if(occupied_volume_points.size()>32){out.status="occupied_volume_proposal_capacity";return out;}
    if(!occupied_volume_points.empty()) {
      Eigen::MatrixXd metric_q=q_bar;
      metric_q.conservativeResize(Eigen::NoChange,q_bar.cols()+executable_samples.size());
      for(std::size_t i=0;i<executable_samples.size();++i)metric_q.col(q_bar.cols()+i)=executable_samples[i].q;
      std::vector<PrimitiveObservationGuard::Row> rows;std::string error;
      if(!geometry_guard.linearizeOccupiedVoxels(metric_q,occupied_volume_points,.05,.02,.02,&rows,&error)) {
        out.status=error;return out;
      }
      for(const auto& row:rows) {
        if(row.timestep==0 && row.distance_m<row.required_margin_m-1e-9) {
          out.status="occupied_volume_start_clearance_hold";return out;
        }
        if(row.timestep>0)primitive_rows.push_back(row);
      }
    }
  }
  if(geometry_guard.knownGroundEnabled()) {
    Eigen::MatrixXd ground_q=q_bar;
    ground_q.conservativeResize(Eigen::NoChange,q_bar.cols()+executable_samples.size());
    for(std::size_t i=0;i<executable_samples.size();++i)ground_q.col(q_bar.cols()+i)=executable_samples[i].q;
    std::vector<PrimitiveObservationGuard::Row> ground_rows;std::string error;
    if(!geometry_guard.linearizeKnownGround(ground_q,&ground_rows,&error)) {
      out.status="known_ground_geometry_error";return out;
    }
    for(const auto& row:ground_rows) {
      if(row.timestep==0) {
        if(row.distance_m<row.required_margin_m-1e-9) {
          out.status="known_ground_start_clearance_hold";return out;
        }
      } else primitive_rows.push_back(row);
    }
  }
  out.primitive_observation_rows = static_cast<int>(primitive_rows.size());
  std::vector<int> shell_penalty_rows;
  for (std::size_t r=0;r<primitive_rows.size();++r) {
    const auto& row=primitive_rows[r];if(row.known_ground || row.occupied_volume)continue;
    const auto key=care_confidence_map::InitialShellRecovery::key(row.point);
    const bool nominal=row.timestep>0 && row.timestep<q_bar.cols();
    const bool future=nominal || (row.timestep>=q_bar.cols() &&
        executable_samples.at(row.timestep-q_bar.cols()).time_s>0.);
    if((nominal && shell_points.count(key)) || (future && primitive_proposal_targets.count(key)))
      shell_penalty_rows.push_back(r);
  }
  out.primitive_shell_penalty_rows=shell_penalty_rows.size();
  std::vector<NormalExecutableLimitRow> normal_limit_samples;
  if(normal_motion && std::getenv("CARE_ROLLING_TASK_OBSERVATION") &&
      std::string(std::getenv("CARE_ROLLING_TASK_OBSERVATION"))=="true") {
    std::string error;
    if(!normalExecutableLimitRows(q_bar,u_bar,dt_,repair_executable_settings_,
        previous_command,&normal_limit_samples,&error)) {
      out.status="normal_executable_limit_model_error";return out;
    }
  }
  const int n=n_q+n_u+n_s+shell_penalty_rows.size();
  std::set<int> executable_velocity_knots;
  for (const auto& sample : executable_samples) for (const auto& term : sample.u_weights)
    if (term.second.cwiseAbs().maxCoeff() > 1e-12) executable_velocity_knots.insert(term.first);
  for (const auto& sample : cdf_executable_samples) for (const auto& term : sample.u_weights)
    if (term.second.cwiseAbs().maxCoeff() > 1e-12) executable_velocity_knots.insert(term.first);
  // NORMAL joint-limit envelopes need no brake-derivative velocity trust box.
  const int normal_limit_row_offset=n_acc+n_cdf_rows+out.primitive_observation_rows+shell_penalty_rows.size();
  const int n_ineq = normal_limit_row_offset+normal_limit_samples.size()*dof_;

  using Triplet = Eigen::Triplet<double>;
  std::vector<Triplet> p_triplets;
  std::vector<Triplet> a_triplets;
  std::vector<Triplet> g_triplets;

  p_triplets.reserve(
      static_cast<std::size_t>(n * 4));
  a_triplets.reserve(
      static_cast<std::size_t>(n_eq * 3));
  g_triplets.reserve(
      static_cast<std::size_t>(
          n_acc * 2 + (n_cdf_rows + out.primitive_observation_rows) * (dof_ + 1)));

  Eigen::VectorXd c = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(n_eq);
  Eigen::VectorXd h_l =
      Eigen::VectorXd::Constant(n_ineq, -PIQP_INF);
  Eigen::VectorXd h_u =
      Eigen::VectorXd::Constant(n_ineq, PIQP_INF);
  Eigen::VectorXd x_l =
      Eigen::VectorXd::Constant(n, -PIQP_INF);
  Eigen::VectorXd x_u =
      Eigen::VectorXd::Constant(n, PIQP_INF);

  const double task_scale =
      repair_mode ? repair_task_tracking_scale_ : 1.0;
  Eigen::VectorXd task_free_mask = Eigen::VectorXd::Zero(dof_);
  if (free_joint_task && repair_mode && !probe_mode && frontier.active &&
      frontier.recovery_attempt == 0 && schedule.size() == 1 &&
      schedule[0].joint_mask.size() == dof_ && frontier.joint_mask.size() == dof_ &&
      (schedule[0].joint_mask-frontier.joint_mask).cwiseAbs().maxCoeff() < 1e-12) {
    for (int j=0; j<dof_; ++j)
      if (schedule[0].joint_mask[j] == 0.) task_free_mask[j] = 1.;
  }

  auto addQuadraticTarget =
      [&](int idx, double weight, double target) {
        if (weight <= 0.0) return;
        p_triplets.emplace_back(idx, idx, 2.0 * weight);
        c[idx] += -2.0 * weight * target;
      };

  for (int k = 1; k <= num_intervals_; ++k) {
    const double w =
        task_scale *
        (q_tracking_weight_ +
         (k == num_intervals_
              // The NORMAL endpoint is now an actual stopped position. Give
              // it the horizon's tracking mass so smoothing the deceleration
              // cannot cheaply trade arrival accuracy for a short stop.
              // This remains soft, including when a rolling target is distant.
              ? terminal_q_tracking_weight_ * (normal_motion ? num_intervals_ : 1)
              : 0.0));

    for (int j = 0; j < dof_; ++j) {
      const int qi = qIndex(k, j);
      addQuadraticTarget(qi, w, q_ref(j, k));
      if (task_free_mask[j] > 0.) {
        const double target = q_bar(j,0)+clampValue(q_ref(j,k)-q_bar(j,0),-.05,.05);
        const double weight = repair_free_joint_task_tracking_scale_ *
            (q_tracking_weight_+(k==num_intervals_ ? terminal_q_tracking_weight_ : 0.));
        addQuadraticTarget(qi,weight,target);
      }

      const double lo = std::max(
          q_min_[j] + joint_position_margin_,
          q_bar(j, k) - trust_radius);
      const double hi = std::min(
          q_max_[j] - joint_position_margin_,
          q_bar(j, k) + trust_radius);
      if (lo > hi) {
        out.status = "trust_region_joint_limit_empty";
        return out;
      }
      x_l[qi] = lo;
      x_u[qi] = hi;
    }
  }

  // The receiver may execute only a short prefix. Penalize the Cartesian
  // goal error AFTER its entire braking tail, rather than assuming the QP's
  // distant terminal improvement will actually be executed. Keep all original
  // route/reference costs and hard constraints; a detour can still move away.
  int prefix_task_targets=0;
  if(prefix_task_objective && prefix_task_goal_valid) {
    auto settings=repair_executable_settings_;
    settings.initial_velocity=previous_command;settings.prefixes={.1,.15,.25,.5};
    if(repair_prefix_task_objective) {
      settings.prefixes=executable_settings.prefixes;
    }
    std::vector<ExecutableTaskTarget> targets;std::string error;
    if(executableTaskTargets(primitive_observation_guard_,q_bar,u_bar,dt_,settings,
        terminal_ee_frame_,prefix_task_goal,&targets,&error)) {
      const double mass=(q_tracking_weight_+terminal_q_tracking_weight_)*num_intervals_;
      // NORMAL uses 10cm/1rad objective units. Near-goal REPAIR uses
      // the actual 2cm/.2rad task units (5^2), so observation/reference
      // attraction does not dominate the feasible terminal stop.
      const double weight=mass/targets.size()*(repair_prefix_task_objective ? 25. : 1.);
      for(const auto& target:targets) {
        std::map<int,Eigen::Matrix<double,6,1>> columns;
        auto add=[&](int index,const Eigen::Matrix<double,6,1>& column) {
          auto it=columns.find(index);
          if(it==columns.end())columns.emplace(index,column);else it->second+=column;
        };
        for(const auto& term:target.stopped.q_weights)if(term.first>0)
          for(int j=0;j<dof_;++j)add(qIndex(term.first,j),term.second*target.jacobian.col(j));
        for(const auto& term:target.stopped.u_weights)
          for(int j=0;j<dof_;++j)add(uIndex(term.first,j),term.second[j]*target.jacobian.col(j));
        Eigen::Matrix<double,6,1> offset=target.error;
        for(const auto& term:columns) {
          const int index=term.first;
          const double nominal=index<n_q ? q_bar(index%dof_,index/dof_+1) :
              u_bar((index-n_q)%dof_,(index-n_q)/dof_);
          offset-=term.second*nominal;
        }
        for(const auto& a:columns) {
          c[a.first]+=2.*weight*a.second.dot(offset);
          for(const auto& b:columns)
            p_triplets.emplace_back(a.first,b.first,2.*weight*a.second.dot(b.second));
        }
      }
      prefix_task_targets=targets.size();
    } else {
      // An unavailable optional objective is diagnosed; it is not a new
      // feasibility or execution gate on the existing planner.
      ROS_WARN_STREAM("[PREFIX_TASK_OBJECTIVE] unavailable reason="<<error);
    }
  }

  if(probe_task_objective_enabled_ && probe_mode && !repair_mode &&
      !force_diagnostic_slack && prefix_task_goal_valid) {
    Eigen::Matrix<double,6,1> residual;Eigen::MatrixXd jacobian;
    if(probeTaskResidualJacobian(geometry_guard,q_bar,u_bar,dt_,previous_command,
        probe_task_settings_,terminal_ee_frame_,prefix_task_goal,&residual,&jacobian)) {
      const double weight=(q_tracking_weight_+terminal_q_tracking_weight_)*num_intervals_;
      std::vector<int> active;Eigen::Matrix<double,6,1> offset=residual;
      for(int col=0;col<jacobian.cols();++col)if(jacobian.col(col).cwiseAbs().maxCoeff()>1e-10) {
        active.push_back(col);
        const double nominal=col<n_q ? q_bar(col%dof_,col/dof_+1) : u_bar((col-n_q)%dof_,(col-n_q)/dof_);
        offset-=jacobian.col(col)*nominal;
      }
      for(int a:active) {
        c[a]+=2.*weight*jacobian.col(a).dot(offset);
        for(int b:active)p_triplets.emplace_back(a,b,2.*weight*jacobian.col(a).dot(jacobian.col(b)));
      }
      ROS_INFO_STREAM("[PROBE_EXECUTABLE_TASK] active_columns="<<active.size()
          <<" weight="<<weight<<" stopped_task_cost="<<residual.squaredNorm());
    } else ROS_WARN("[PROBE_EXECUTABLE_TASK] optional objective unavailable");
  }

  // Visibility obligations are long-range trajectory objectives. A learned
  // frontier target may simultaneously lower their weight and add a short
  // local visibility-improvement objective; neither changes hard safety.
  const double qvis_weight =
      visibility_waypoint_weight_ *
      ((repair_mode && frontier.active)
           ? frontier.qvis_weight_scale
           : 1.0);
  const double now_s = ros::Time::now().toSec();
  for (const auto& wp : schedule) {
    if (wp.q.size() != dof_ || wp.joint_mask.size() != dof_) continue;
    const int k = visibilityObjectiveStep(wp.terminal_objective,
        wp.deadline_abs_s, now_s, dt_, num_intervals_);
    if (k < 1) continue;
    out.visibility_objective_step = k;
    for (int j = 0; j < dof_; ++j) {
      addQuadraticTarget(
          qIndex(k, j),
          qvis_weight * wp.joint_mask[j],
          wp.q[j]);
    }
  }

  int stopped_observation_targets = 0;
  if (repair_mode &&
      frontier.active &&
      frontier.q.size() == dof_ &&
      finiteVector(frontier.q) &&
      frontier.frontier_weight_scale > 0.0) {
    const int k_frontier = std::max(
        1, std::min(visibility_frontier_horizon_step_, num_intervals_));
    const double frontier_weight =
        visibility_waypoint_weight_ * frontier.frontier_weight_scale;
    Eigen::VectorXd objective_target = frontier.q;
    Eigen::VectorXd objective_mask =
        frontier.joint_mask.size() == dof_
            ? frontier.joint_mask
            : Eigen::VectorXd::Ones(dof_);
    // Sensing dependencies are available during ordinary observation steering,
    // before a measured stall activates the bounded detour ledger. The same
    // selected hard rows remain in the QP; this only reports a local hypothesis.
    std::vector<Eigen::VectorXd> blocking_normals;
    std::vector<double> blocking_slack;
    double clearance=std::numeric_limits<double>::infinity();
    std::ostringstream clearance_identity;
    clearance_identity<<std::setprecision(17);
    for (const auto& row : selected) {
        if (row.k != k_frontier) continue;
        // A soft UNKNOWN row is a cost, not a blocking halfspace. Using it
        // here would recreate mandatory observation dependencies immediately
        // after allowing the QP to cross its learned margin.
        if (unknown_cdf_soft && batch.source_type[row.pair] ==
            care_collision_cdf::CollisionCDFConstraintBatch::SOURCE_UNKNOWN) continue;
        const double at_start = row.d + row.g.dot(q_bar.col(0)-row.qlin);
        if (row.g.norm()>1e-9 && row.pair>=0 &&
            batch.point_flat.size()>=static_cast<std::size_t>(3*(row.pair+1))) {
          clearance=std::min(clearance,(at_start-row.safety_margin)/row.g.norm());
          clearance_identity<<batch.point_flat[3*row.pair]<<':'<<batch.point_flat[3*row.pair+1]
            <<':'<<batch.point_flat[3*row.pair+2]<<':'<<row.safety_margin<<';';
        }
        if (at_start-row.safety_margin < .005 && row.g.norm() > 1e-9) {
          const double along = row.g.dot(frontier.q-q_bar.col(0));
          // Proximity alone does not oppose forward motion. Preserve steps
          // that increase clearance; every selected row remains in the hard
          // QP, including deficient starts and rows not used for steering.
          if (frontier.recovery_attempt > 0) {
            ++out.recovery_tight_normals;
            if (recoveryForwardBlocked(at_start-row.safety_margin, along)) {
              blocking_normals.push_back(row.g);
              blocking_slack.push_back(at_start-row.safety_margin);
            }
          }
          // Only explicit UNKNOWN provenance may request sensing. OCCUPIED
          // remains a geometric obstacle; absent provenance is not enough.
          if (along < -1e-9 && at_start+along < row.safety_margin &&
              row.pair >= 0 &&
              batch.source_type.size() > static_cast<std::size_t>(row.pair) &&
              batch.source_type[row.pair] == care_collision_cdf::CollisionCDFConstraintBatch::SOURCE_UNKNOWN &&
              batch.point_flat.size() >= static_cast<std::size_t>(3*(row.pair+1))) {
            const Eigen::Vector3d p(batch.point_flat[3*row.pair],
                batch.point_flat[3*row.pair+1], batch.point_flat[3*row.pair+2]);
            auto& points = out.observation_dependency_points;
            if (p.allFinite() && points.size()<32 && std::none_of(points.begin(), points.end(),
                [&](const Eigen::Vector3d& old) { return (p-old).lpNorm<Eigen::Infinity>()<=1e-5; }))
              points.push_back(p);
          }
        }
    }
    if (unknown_cdf_soft) {
      // Under this research policy, actual analytic UNKNOWN guards, not the
      // soft learned distance, own dependency and detour scheduling. Occupied
      // and ground rows remain hard constraints and are never sensing targets.
      for (const auto& row : primitive_rows) {
        if (row.timestep != k_frontier || row.known_ground || row.occupied_volume ||
            row.gradient.norm() <= 1e-9) continue;
        const auto key=care_confidence_map::InitialShellRecovery::key(row.point);
        const double margin=primitive_floors.at(key);
        const double at_start=row.distance_m+row.gradient.dot(q_bar.col(0)-q_bar.col(k_frontier));
        const double gap=at_start-margin;
        clearance=std::min(clearance,gap/row.gradient.norm());
        clearance_identity<<"metric:"<<row.point[0]<<':'<<row.point[1]<<':'<<row.point[2]<<':'<<margin<<';';
        const double along=row.gradient.dot(frontier.q-q_bar.col(0));
        if (gap < .005 && frontier.recovery_attempt > 0) {
          ++out.recovery_tight_normals;
          if (recoveryForwardBlocked(gap,along)) {
            blocking_normals.push_back(row.gradient);blocking_slack.push_back(gap);
          }
        }
        if (gap < .005 && along < -1e-9 && gap+along < 0. &&
            std::none_of(obligation_points.begin(),obligation_points.end(),[&](const Eigen::Vector3d& p) {
              return (p-row.point).lpNorm<Eigen::Infinity>()<=1e-5;
            }) && out.observation_dependency_points.size()<32 &&
            std::none_of(out.observation_dependency_points.begin(),out.observation_dependency_points.end(),
                [&](const Eigen::Vector3d& p){return (p-row.point).lpNorm<Eigen::Infinity>()<=1e-5;}))
          out.observation_dependency_points.push_back(row.point);
      }
    }
    if (frontier.visibility_gradient.size()==7 && std::isfinite(clearance)) {
      out.recovery_start_clearance=clearance;
      out.recovery_clearance_signature=clearance_identity.str();
    }
    if (frontier.recovery_attempt > 0) {
      // A certified safety recovery target may deliberately use a visibility-
      // free joint. The sensor mask applies to observation steering only.
      objective_mask.setOnes();
      out.recovery_normals = static_cast<int>(blocking_normals.size());
      out.recovery_rejoining = blocking_normals.empty();
      if (out.recovery_rejoining) {
        // No nearby row opposes proposing the forward frontier. This also
        // replaces any cached detour target from the preceding solve.
        // This is NOT observation completion or whole-path certification.
        objective_target = frontier.q;
      } else if (frontier.recovery_target.size() == dof_ && finiteVector(frontier.recovery_target)) {
        objective_target = frontier.recovery_target;
      } else {
        objective_target = frontier.visibility_gradient.size()==7 ?
            boundedObservationTarget(q_bar.col(0), frontier.q, blocking_normals,
                blocking_slack, frontier.visibility_gradient, frontier.joint_mask,
                frontier.failed_directions, std::min(.03, trust_radius)) :
            boundedTangentTarget(q_bar.col(0), frontier.q,
                blocking_normals, frontier.recovery_attempt, std::min(.03, trust_radius));
        // Degenerate tangent space is a bounded failed attempt, not permission
        // to cross a witness. Certified micro-holds exhaust the attempt budget.
        if (objective_target.size() != dof_) objective_target = q_bar.col(0);
      }
      out.recovery_frontier = objective_target;
    }
    // Ordinary sensing must improve the pose the receiver actually reaches
    // after its selected prefix and brake. A target at k_frontier alone can
    // improve briefly, then reverse before the prefix stops. Use the same
    // differentiated executable view already used by the geometric guards.
    // This changes a soft objective only; detour objectives remain unchanged.
    if (repair_motion && frontier.recovery_attempt == 0) {
      std::vector<RepairExecutableSample> samples;
      std::string error;
      if (repairExecutableSamples(q_bar,u_bar,dt_,executable_settings,&samples,&error)) {
        const double weight = frontier_weight / executable_settings.prefixes.size();
        for (double prefix : executable_settings.prefixes) {
          const auto stop = std::find_if(samples.rbegin(),samples.rend(),
              [&](const RepairExecutableSample& sample) {
                return std::abs(sample.prefix_s-std::min(prefix,num_intervals_*dt_))<1e-9;
              });
          if (stop == samples.rend()) continue;
          for (int j=0;j<dof_;++j) {
            if (objective_mask[j] <= 0.) continue;
            std::map<int,double> columns;
            for (const auto& term : stop->q_weights)
              if (term.first > 0) columns[qIndex(term.first,j)] += term.second;
            for (const auto& term : stop->u_weights)
              columns[uIndex(term.first,j)] += term.second[j];
            double offset = stop->q[j] - objective_target[j];
            for (const auto& term : columns) {
              const double nominal = term.first < n_q
                  ? q_bar(j,term.first/dof_+1) : u_bar(j,(term.first-n_q)/dof_);
              offset -= term.second * nominal;
            }
            const double w = weight * objective_mask[j];
            for (const auto& a : columns) {
              c[a.first] += 2.*w*a.second*offset;
              for (const auto& b : columns)
                p_triplets.emplace_back(a.first,b.first,2.*w*a.second*b.second);
            }
          }
          ++stopped_observation_targets;
        }
      }
    }
    for (int j = 0; j < dof_; ++j) {
      if (!stopped_observation_targets)
      addQuadraticTarget(
          qIndex(k_frontier, j),
          frontier_weight * objective_mask[j],
          objective_target[j]);
      // A short detour target also owns the tail objective. This suppresses
      // accelerating through the tiny target then being pulled back. It is
      // a soft tracking term, not a claimed hard displacement bound.
      if (frontier.recovery_attempt > 0)
        for (int k = k_frontier+1; k <= num_intervals_; ++k)
          addQuadraticTarget(
              qIndex(k, j), frontier_weight * objective_mask[j],
              objective_target[j]);
    }
  }

  for (int k = 0; k < num_intervals_; ++k) {
    for (int j = 0; j < dof_; ++j) {
      const int ui = uIndex(k, j);
      addQuadraticTarget(
          ui, u_tracking_weight_,
          u_reference_tracking_enabled_ ? u_ref(j, k) : 0.0);
      const double speed_limit = std::min(velocity_limits_[j],
          probe_mode ? probe_velocity_cap_ : velocity_limits_[j]);
      x_l[ui] = -speed_limit;
      x_u[ui] = speed_limit;
      // u_k is the END velocity of the kth constant-acceleration interval
      // in NORMAL. Stop within the existing horizon; q_K remains a soft target.
      if (nodal_motion && k == num_intervals_ - 1) x_l[ui] = x_u[ui] = 0.;
      if (executable_velocity_knots.count(k)) {
        // Discrete trapezoidal brake displacement is piecewise affine.
        // Its local derivative cannot support a jump across many clipping
        // intervals. Limit these updates to one existing deceleration quantum,
        // alongside the original global velocity bounds and q trust box.
        const double radius=executable_step_scale*repair_executable_settings_.acceleration[j]*repair_executable_settings_.brake_dt;
        x_l[ui]=std::max(x_l[ui],u_bar(j,k)-radius);
        x_u[ui]=std::min(x_u[ui],u_bar(j,k)+radius);
      }
    }
  }

  // Certified-handoff boundary continuity. Keep this boundary-only term
  // active even when general intra-horizon smoothness is disabled.
  const double boundary_velocity_weight =
      u_smooth_weight_ + handoff_velocity_weight_;
  for (int j = 0; j < dof_; ++j) {
    const int u0 = uIndex(0, j);
    p_triplets.emplace_back(
        u0, u0, 2.0 * boundary_velocity_weight);
    const double prev =
        previous_command.size() == dof_
            ? previous_command[j]
            : 0.0;
    c[u0] += -2.0 * boundary_velocity_weight * prev;
  }
  for (int k = 1; k < num_intervals_; ++k) {
    for (int j = 0; j < dof_; ++j) {
      const int ua = uIndex(k - 1, j);
      const int ub = uIndex(k, j);
      p_triplets.emplace_back(
          ua, ua, 2.0 * u_smooth_weight_);
      p_triplets.emplace_back(
          ub, ub, 2.0 * u_smooth_weight_);
      p_triplets.emplace_back(
          ua, ub, -2.0 * u_smooth_weight_);
      p_triplets.emplace_back(
          ub, ua, -2.0 * u_smooth_weight_);
    }
  }

  // User CDF slacks guarantee every convex subproblem remains feasible even
  // when the current iterate is deeply inside forbidden space. G0 uses one
  // slack per CDF inequality, matching GCDF.
  const int slack0 = n_q + n_u;
  for (int s = 0; s < n_s; ++s) {
    const int si = slack0 + s;
    c[si] += slack_linear_weight;
    p_triplets.emplace_back(
        si, si, 2.0 * cdf_slack_quadratic_weight_);
    x_l[si] = 0.0;
    // UNKNOWN is a proposal cost under this policy. A finite cap silently
    // made it hard again (CASE025 needs >0.30 in model units). Known OCCUPIED
    // rows are pinned to zero slack below; final exact audits are unchanged.
    if (cdf_slack_use_upper_bound_ && !force_diagnostic_slack && !unknown_cdf_soft)
      x_u[si] = std::min(x_u[si], cdf_slack_upper_bound_);
  }

  // Small diagonal regularization makes P strictly positive definite enough
  // for stable sparse factorization without changing the optimizer materially.
  for (int i = 0; i < n; ++i)
    p_triplets.emplace_back(i, i, 1e-8);

  // Multiple-shooting dynamics:
  // NORMAL uses trapezoidal acceleration integration from measured v0.
  // REPAIR/PROBE retain their existing prefix-construction variable semantics.
  for (int k = 0; k < num_intervals_; ++k) {
    for (int j = 0; j < dof_; ++j) {
      const int row = k * dof_ + j;
      a_triplets.emplace_back(
          row, qIndex(k + 1, j), 1.0);
      a_triplets.emplace_back(
          row, uIndex(k, j), nodal_motion ? -.5 * dt_ : -dt_);
      if (k == 0) {
        b[row] = q_bar(j, 0) + (nodal_motion ? .5 * dt_ * previous_command[j] : 0.);
      } else {
        a_triplets.emplace_back(
            row, qIndex(k, j), -1.0);
        if (nodal_motion) a_triplets.emplace_back(row, uIndex(k - 1, j), -.5 * dt_);
      }
    }
  }

  if (acceleration_constraints) {
    // CARE acceleration envelope.
    for (int k = 0; k < num_intervals_; ++k) {
      for (int j = 0; j < dof_; ++j) {
        const int row = k * dof_ + j;
        g_triplets.emplace_back(
            row, uIndex(k, j), 1.0);

        const double du =
            acceleration_limits_[j] * dt_;
        if (k == 0) {
          const double prev =
              previous_command.size() == dof_
                  ? previous_command[j]
                  : 0.0;
          h_l[row] = prev - du;
          h_u[row] = prev + du;
        } else {
          g_triplets.emplace_back(
              row, uIndex(k - 1, j), -1.0);
          h_l[row] = -du;
          h_u[row] = du;
        }
      }
    }

    // CARE terminal braking row.
    for (int j = 0; j < dof_; ++j) {
      const int row = num_intervals_ * dof_ + j;
      const double du = acceleration_limits_[j] * dt_;
      g_triplets.emplace_back(
          row, uIndex(num_intervals_ - 1, j), 1.0);
      h_l[row] = -du;
      h_u[row] = du;
    }
  }

  // Linearized CDF:
  // d + g'(q_k-qbar) + s >= d_safe
  // -> g' q_k + s >= d_safe - d + g' qbar.
  for (int r = 0; r < n_cdf_rows; ++r) {
    const auto& row_data =
        selected[static_cast<std::size_t>(r)];
    const int row = n_acc + r;
    double center_dot=row_data.g.dot(row_data.qlin);
    if(row_data.executable_sample<0) {
      for(int j=0;j<dof_;++j)g_triplets.emplace_back(row,qIndex(row_data.k,j),row_data.g[j]);
    } else {
      center_dot=0.;
      const auto& sample=cdf_executable_samples[row_data.executable_sample];
      for(const auto& term:sample.q_weights)if(term.first>0) {
        for(int j=0;j<dof_;++j)g_triplets.emplace_back(row,qIndex(term.first,j),term.second*row_data.g[j]);
        center_dot+=term.second*row_data.g.dot(q_bar.col(term.first));
      }
      for(const auto& term:sample.u_weights) {
        for(int j=0;j<dof_;++j)g_triplets.emplace_back(row,uIndex(term.first,j),term.second[j]*row_data.g[j]);
        center_dot+=(term.second.array()*row_data.g.array()*u_bar.col(term.first).array()).sum();
      }
    }
    if (slack_enabled) {
      const int slack_slot =
          per_constraint_slack
              ? r
              : step_to_slack[static_cast<std::size_t>(row_data.k)];
      if (slack_slot < 0 || slack_slot >= n_s) {
        out.status = "cdf_slack_mapping_error";
        return out;
      }
      g_triplets.emplace_back(
          row, slack0 + slack_slot, 1.0);
      const char* soft_unknown=std::getenv("CARE_RESEARCH_SOFT_UNKNOWN");
      if(!force_diagnostic_slack && (unknown_cdf_soft ||
          (soft_unknown && std::string(soft_unknown)=="true")) &&
          !batch.source_type.empty() && batch.source_type[row_data.pair]==care_collision_cdf::CollisionCDFConstraintBatch::SOURCE_OCCUPIED)
        x_u[slack0+slack_slot]=0.; // Known occupied CDF remains hard.
    }
    h_l[row] =
        row_data.safety_margin - row_data.d +
        center_dot;
    h_u[row] = PIQP_INF;
  }

  // Additional proposal constraints in metres, without slack. Every original
  // CDF row remains. Prefix/brake rows differentiate the constructed position
  // with respect to both q and u. Final GCDF/VBC still certify the exact sweep.
  for (std::size_t r = 0; r < primitive_rows.size(); ++r) {
    const auto& geometry = primitive_rows[r];
    const int row = n_acc + n_cdf_rows + static_cast<int>(r);
    h_l[row] = ((geometry.known_ground || geometry.occupied_volume) ? geometry.required_margin_m :
        primitive_floors.at(care_confidence_map::InitialShellRecovery::key(geometry.point))) - geometry.distance_m;
    if (geometry.timestep < q_bar.cols()) {
      for (int j = 0; j < dof_; ++j)
        g_triplets.emplace_back(row,qIndex(geometry.timestep,j),geometry.gradient[j]);
      h_l[row] += geometry.gradient.dot(geometry.q_linearization);
    } else {
      const auto& sample = executable_samples.at(geometry.timestep-q_bar.cols());
      for (const auto& term : sample.q_weights) if (term.first > 0) {
        for (int j=0;j<dof_;++j) g_triplets.emplace_back(row,qIndex(term.first,j),term.second*geometry.gradient[j]);
        h_l[row] += term.second*geometry.gradient.dot(q_bar.col(term.first));
      }
      for (const auto& term : sample.u_weights) {
        for (int j=0;j<dof_;++j) g_triplets.emplace_back(row,uIndex(term.first,j),term.second[j]*geometry.gradient[j]);
        h_l[row] += (term.second.array()*geometry.gradient.array()*u_bar.col(term.first).array()).sum();
      }
      // These added rows have metre-valued residuals. Tighten the proposal
      // using the solver tolerance so roundoff near the boundary does not
      // repeatedly exhaust SCP. Nonlinear checking and final VBC enforce the
      // ordinary margin or the explicit point recovery floor. This padding
      // is not a tracking-error allowance.
      // The measured start is fixed and checked exactly above. Do not turn
      // its already satisfied constant row into an artificial infeasibility.
      if (sample.time_s > 0.) h_l[row] += out.primitive_executable_numerical_tightening_m;
    }
  }

  // Only the missing candidate margin is soft. The point-scoped initial
  // clearance floor above and every original CDF row remain hard. Normalize
  // by the candidate margin and reuse the existing frontier objective scale.
  for (std::size_t i=0;i<shell_penalty_rows.size();++i) {
    const auto& geometry=primitive_rows[shell_penalty_rows[i]];
    const int si=n_q+n_u+n_s+i;
    const int row=n_acc+n_cdf_rows+primitive_rows.size()+i;
    x_l[si]=0.;
    p_triplets.emplace_back(si,si,2.*visibility_waypoint_weight_ /
        (primitive_observation_margin_m_*primitive_observation_margin_m_));
    g_triplets.emplace_back(row,si,1.);
    const auto target=primitive_proposal_targets.find(care_confidence_map::InitialShellRecovery::key(geometry.point));
    double desired=primitive_observation_margin_m_;
    if(target!=primitive_proposal_targets.end()) {
      const double floor=primitive_floors.at(care_confidence_map::InitialShellRecovery::key(geometry.point));
      const double time=geometry.timestep<q_bar.cols()?geometry.timestep*dt_:
          executable_samples.at(geometry.timestep-q_bar.cols()).time_s;
      const double prefix=executable_settings.prefixes.empty()?dt_:
          *std::max_element(executable_settings.prefixes.begin(),executable_settings.prefixes.end());
      desired=floor+(target->second-floor)*std::min(1.,time/std::max(dt_,prefix));
    }
    h_l[row]=desired-geometry.distance_m;
    if(geometry.timestep<q_bar.cols()) {
      for (int j=0;j<dof_;++j)g_triplets.emplace_back(row,qIndex(geometry.timestep,j),geometry.gradient[j]);
      h_l[row]+=geometry.gradient.dot(geometry.q_linearization);
    } else {
      const auto& sample=executable_samples.at(geometry.timestep-q_bar.cols());
      for(const auto& term:sample.q_weights)if(term.first>0) {
        for(int j=0;j<dof_;++j)g_triplets.emplace_back(row,qIndex(term.first,j),term.second*geometry.gradient[j]);
        h_l[row]+=term.second*geometry.gradient.dot(q_bar.col(term.first));
      }
      for(const auto& term:sample.u_weights) {
        for(int j=0;j<dof_;++j)g_triplets.emplace_back(row,uIndex(term.first,j),term.second[j]*geometry.gradient[j]);
        h_l[row]+=(term.second.array()*geometry.gradient.array()*u_bar.col(term.first).array()).sum();
      }
    }
  }

  // Joint position bounds on the actual .25 s task prefix and its complete
  // discrete stop, rather than only the unexecuted one-second nodal reference.
  int normal_padding_reduced_rows=0;
  const Eigen::VectorXd normal_lower=q_min_.array()+joint_position_margin_;
  const Eigen::VectorXd normal_upper=q_max_.array()-joint_position_margin_;
  for(std::size_t i=0;i<normal_limit_samples.size();++i) {
    const auto& limit=normal_limit_samples[i];
    const auto& sample=limit.sample;
    for(int j=0;j<dof_;++j) {
      const int row=normal_limit_row_offset+i*dof_+j;
      double offset=sample.q[j];
      for(const auto& term:sample.q_weights)if(term.first>0 && term.second!=0.) {
        g_triplets.emplace_back(row,qIndex(term.first,j),term.second);
        offset-=term.second*q_bar(j,term.first);
      }
      for(const auto& term:sample.u_weights)if(term.second[j]!=0.) {
        g_triplets.emplace_back(row,uIndex(term.first,j),term.second[j]);
        offset-=term.second[j]*u_bar(j,term.first);
      }
      const auto padding=normalExecutableLimitPadding(limit,j,q_bar.col(0),
          previous_command,velocity_limits_,acceleration_limits_,normal_lower,
          normal_upper,dt_,2.*piqp_eps_abs_);
      if((limit.lower && padding.lower<2.*piqp_eps_abs_) ||
          (limit.upper && padding.upper<2.*piqp_eps_abs_))++normal_padding_reduced_rows;
      if(limit.lower)h_l[row]=normal_lower[j]+padding.lower-offset;
      if(limit.upper)h_u[row]=normal_upper[j]-padding.upper-offset;
    }
  }
  Eigen::SparseMatrix<double> P(n, n);
  Eigen::SparseMatrix<double> A(n_eq, n);
  Eigen::SparseMatrix<double> G(n_ineq, n);
  P.setFromTriplets(
      p_triplets.begin(), p_triplets.end());
  A.setFromTriplets(
      a_triplets.begin(), a_triplets.end());
  G.setFromTriplets(
      g_triplets.begin(), g_triplets.end());
  P.makeCompressed();
  A.makeCompressed();
  G.makeCompressed();

  constexpr double fixed_recheck_tolerance=1e-7;
  if (executable_recheck_only) {
    // Intersect, never widen, every original bound. All acceleration, CDF,
    // primitive and numerical-padding rows remain in this complete hard QP.
    // The saved candidate is an internal seed until this fresh solve passes.
    // A solved equality (notably terminal u=0) carries floating-point residual.
    // Project that seed into the ORIGINAL box within the same tolerance used
    // below to check the fixed solution. The hard bounds are never expanded;
    // acceleration, CDF and primitive constraints must still solve together.
    int projected=0;double projection_inf=0.;
    const auto pin=[&](int index,double value) {
      if(!std::isfinite(value) || x_l[index]>x_u[index])return false;
      const double feasible=std::max(x_l[index],std::min(x_u[index],value));
      const double delta=std::abs(feasible-value);
      if(delta>fixed_recheck_tolerance)return false;
      if(delta>0.){++projected;projection_inf=std::max(projection_inf,delta);}
      x_l[index]=x_u[index]=feasible;
      return true;
    };
    for (int k=1;k<=num_intervals_;++k) for (int j=0;j<dof_;++j) {
      const int index=qIndex(k,j); const double value=q_bar(j,k);
      if (!pin(index,value)) {
        out.status="executable_recheck_bound_error"; return out;
      }
    }
    for (int k=0;k<num_intervals_;++k) for (int j=0;j<dof_;++j) {
      const int index=uIndex(k,j); const double value=u_bar(j,k);
      if (!pin(index,value)) {
        out.status="executable_recheck_bound_error"; return out;
      }
    }
    if(projected)ROS_INFO_STREAM("[EXECUTABLE_RECHECK_PROJECT] variables="<<projected
        <<" delta_inf="<<std::setprecision(17)<<projection_inf);
  }

  ROS_WARN_STREAM(
      "[LocalSparseSCPPlanner] sparse QP dimensions n=" << n
      << " q_vars=" << n_q
      << " u_vars=" << n_u
      << " cdf_slacks=" << n_s
      << " slack_enabled=" << (slack_enabled ? 1 : 0)
      << " unknown_cdf_soft=" << (unknown_cdf_soft ? 1 : 0)
      << " per_constraint_slack=" << (per_constraint_slack ? 1 : 0)
      << " diagnostic_slack=" << (force_diagnostic_slack ? 1 : 0)
      << " accel_constraints="
      << (acceleration_constraints ? 1 : 0)
      << " prefix_task_targets=" << prefix_task_targets
      << " stopped_observation_targets=" << stopped_observation_targets
      << " normal_acceleration_motion=" << (normal_motion ? 1 : 0)
      << " repair_acceleration_motion=" << (repair_motion ? 1 : 0)
      << " normal_executable_joint_limit_rows=" << normal_limit_samples.size()*dof_
      << " normal_padding_reduced_rows=" << normal_padding_reduced_rows
      << " executable_cdf_samples=" << cdf_executable_samples.size()
      << " eq=" << n_eq
      << " ineq=" << n_ineq
      << " selected_cdf_rows=" << n_cdf_rows
      << " primitive_observation_rows=" << out.primitive_observation_rows
      << " unknown_rows=" << out.selected_unknown_cdf_rows
      << " occupied_rows=" << out.selected_occupied_cdf_rows
      << " nnz(P)=" << P.nonZeros()
      << " nnz(A)=" << A.nonZeros()
      << " nnz(G)=" << G.nonZeros());

  piqp::SparseSolver<double> solver;
  auto& settings = solver.settings();
  settings.max_iter =
      force_diagnostic_slack
          ? std::max(piqp_max_iterations_, 1000)
          : piqp_max_iterations_;
  settings.eps_abs = piqp_eps_abs_;
  settings.eps_rel = piqp_eps_rel_;
  settings.verbose = piqp_verbose_;
  settings.compute_timings = true;
  settings.kkt_solver = piqp::KKTSolver::sparse_ldlt;

  const ros::WallTime tic = ros::WallTime::now();
  const piqp::Status status = runSparseQpEntry(
      solver,P,c,A,b,G,h_l,h_u,x_l,x_u,batch.header.stamp.toNSec());
  out.setup_and_solve_ms =
      (ros::WallTime::now() - tic).toSec() * 1000.0;
  ROS_WARN_STREAM(
      "[LocalSparseSCPPlanner] sparse PIQP solve returned status="
      << piqp::status_to_string(status)
      << " total_ms=" << out.setup_and_solve_ms);

  const auto& result = solver.result();
  out.iterations = static_cast<int>(result.info.iter);
  out.primal_residual = result.info.primal_res;
  out.dual_residual = result.info.dual_res;
  out.status = piqp::status_to_string(status);
  if(capture) {
    std::ostringstream m;m<<"{\"P\":";jsonSparse(m,P);
    m<<",\"c\":";jsonVector(m,c);m<<",\"A\":";jsonSparse(m,A);
    m<<",\"b\":";jsonVector(m,b);m<<",\"G\":";jsonSparse(m,G);
    m<<",\"h_l\":";jsonVector(m,h_l);m<<",\"h_u\":";jsonVector(m,h_u);
    m<<",\"x_l\":";jsonVector(m,x_l);m<<",\"x_u\":";jsonVector(m,x_u);m<<'}';
    capture->native_matrix_json=m.str();
  }


  if (!force_diagnostic_slack && status != piqp::PIQP_SOLVED) {
    out.box_conflicting_cdf_rows = lowerBoxConflicts(G, h_l, x_l, x_u, n_acc);
    if (!slack_enabled) {
      for (const auto& row : selected) {
        if(row.executable_sample>=0)continue; // Not a raw reachability index.
        const double upper = reachableCdfUpperBound(
            row.d, row.g, row.qlin, q_bar.col(row.k), q_bar.col(0),
            q_min_, q_max_, velocity_limits_, joint_position_margin_,
            trust_radius, dt_, row.k,nodal_motion ? &previous_command : nullptr,
            nodal_motion ? &acceleration_limits_ : nullptr);
        if (!std::isfinite(upper) || row.safety_margin - upper <= 1e-8) continue;
        ++out.reachable_conflicting_cdf_rows;
        if(row.k==1) {
          ++out.first_step_conflicting_cdf_rows;
          const bool occupied=!batch.source_type.empty() && batch.source_type[row.pair]==
              care_collision_cdf::CollisionCDFConstraintBatch::SOURCE_OCCUPIED;
          if(occupied)++out.first_step_conflicting_occupied_rows;
          else {
            const Eigen::Vector3d point(batch.point_flat[3*row.pair],batch.point_flat[3*row.pair+1],batch.point_flat[3*row.pair+2]);
            if(std::none_of(out.first_step_unknown_conflict_points.begin(),out.first_step_unknown_conflict_points.end(),
                [&](const Eigen::Vector3d& old){return (old-point).lpNorm<Eigen::Infinity>()<=1e-5;}))
              out.first_step_unknown_conflict_points.push_back(point);
            if(std::none_of(out.geometric_restore_guard_points.begin(),out.geometric_restore_guard_points.end(),
                [&](const Eigen::Vector3d& old){return (old-point).lpNorm<Eigen::Infinity>()<=1e-5;}))
              out.geometric_restore_guard_points.push_back(point);
          }
        } else ++out.later_conflicting_cdf_rows;
        if (repair_mode && row.repair_witness) {
          ++out.reachable_conflicting_repair_witness_rows;
          if (row.k == 1) ++out.first_step_conflicting_repair_witness_rows;
          else ++out.later_conflicting_repair_witness_rows;
          out.worst_repair_witness_reachable_deficit = std::max(
              out.worst_repair_witness_reachable_deficit,
              row.safety_margin - upper);
        }
      }
    }
    const auto primitive_upper=primitiveReachableUpperBounds(A,b,G,h_l,h_u,x_l,x_u,
        n_acc,n_acc+n_cdf_rows,primitive_rows.size());
    for(std::size_t i=0;i<primitive_rows.size();++i) {
      if(!std::isfinite(primitive_upper[i]) ||
          h_l[n_acc+n_cdf_rows+i]-primitive_upper[i]<=1e-8)continue;
      ++out.primitive_conflicting_rows;
    }
    if(out.primitive_conflicting_rows>0) {
      // The proposal must keep ALL guards from this exact query, including
      // transient ordinary corridor points. A point identity alone cannot
      // convey its floor or reserve. Native complete-path proof still owns
      // acceptance, and the failed hard QP remains failed.
      out.geometric_restore_path_guards=primitive_points;
      for(const auto& point:primitive_points)
        if(std::none_of(out.geometric_restore_guard_points.begin(),out.geometric_restore_guard_points.end(),
            [&](const Eigen::Vector3d& old){return (old-point.point).lpNorm<Eigen::Infinity>()<=1e-5;}))
          out.geometric_restore_guard_points.push_back(point.point);
    }
    const char* snapshot_root = std::getenv("ROOT_OUT");
    if (repair_mode && !probe_mode && !slack_enabled && !previous_vbc_margins &&
        snapshot_root != nullptr && *snapshot_root != '\0') {
      std::ostringstream s;
      s << "{\"query_stamp_ns\":" << batch.header.stamp.toNSec()
        << ",\"status\":"; jsonString(s,out.status);
      s << ",\"cdf_units\":\"configuration_space_model\",\"dt\":"; jsonNumber(s,dt_);
      s << ",\"trust_radius\":"; jsonNumber(s,trust_radius);
      s << ",\"ordinary_margin\":"; jsonNumber(s,cdf_safety_margin_);
      s << ",\"obligation_margin\":"; jsonNumber(s,visibility_obligation_cdf_margin_effective_.load());
      s << ",\"position_margin\":"; jsonNumber(s,joint_position_margin_);
      s << ",\"qmin\":"; jsonVector(s,q_min_);
      s << ",\"qmax\":"; jsonVector(s,q_max_);
      s << ",\"velocity_limits\":"; jsonVector(s,velocity_limits_);
      s << ",\"q_bar\":"; jsonTrajectory(s,q_bar);
      s << ",\"u_bar\":"; jsonTrajectory(s,u_bar);
      s << ",\"q_ref\":"; jsonTrajectory(s,q_ref);
      s << ",\"previous_command\":"; jsonVector(s,previous_command);
      s << ",\"max_iter\":" << piqp_max_iterations_ << ",\"eps_abs\":"; jsonNumber(s,piqp_eps_abs_);
      s << ",\"eps_rel\":"; jsonNumber(s,piqp_eps_rel_);
      s << ",\"P\":"; jsonSparse(s,P);
      s << ",\"c\":"; jsonVector(s,c);
      s << ",\"A\":"; jsonSparse(s,A);
      s << ",\"b\":"; jsonVector(s,b);
      s << ",\"G\":"; jsonSparse(s,G);
      s << ",\"h_l\":"; jsonVector(s,h_l);
      s << ",\"h_u\":"; jsonVector(s,h_u);
      s << ",\"x_l\":"; jsonVector(s,x_l);
      s << ",\"x_u\":"; jsonVector(s,x_u);
      s << ",\"acceleration_rows\":" << n_acc
        << ",\"primitive_rows\":" << out.primitive_observation_rows
        << ",\"shell_penalty_rows\":" << shell_penalty_rows.size()
        << ",\"capture_reason\":\"primary_hard_failure\""
        << ",\"batch_pairs\":" << batch.num_pairs << ",\"selected_cdf_rows\":" << selected.size()
        << ",\"first_step_witness_conflicts\":" << out.first_step_conflicting_repair_witness_rows
        << ",\"later_witness_conflicts\":" << out.later_conflicting_repair_witness_rows
        << ",\"cdf_rows\":[";
      for (std::size_t i=0; i<selected.size(); ++i) {
        if (i) s << ',';
        const auto& row = selected[i];
        s << "{\"matrix_row\":" << n_acc+i << ",\"pair\":" << row.pair
          << ",\"timestep\":" << row.k << ",\"repair_witness\":" << (row.repair_witness ? "true" : "false")
          << ",\"obligation\":" << (row.obligation ? "true" : "false")
          << ",\"source\":" << (batch.source_type.empty() ? 0 : static_cast<int>(batch.source_type[row.pair]))
          << ",\"point\":[";
        for (int j=0; j<3; ++j) {
          if (j) s << ',';
          const std::size_t p = static_cast<std::size_t>(3*row.pair+j);
          jsonNumber(s,p < batch.point_flat.size() ? batch.point_flat[p] : std::numeric_limits<double>::quiet_NaN());
        }
        s << "],\"distance\":"; jsonNumber(s,row.d);
        s << ",\"margin\":"; jsonNumber(s,row.safety_margin);
        s << ",\"gradient\":"; jsonVector(s,row.g);
        s << ",\"q_linearization\":"; jsonVector(s,row.qlin);
        const double upper = row.executable_sample>=0 ? std::numeric_limits<double>::quiet_NaN() :
            reachableCdfUpperBound(row.d,row.g,row.qlin,q_bar.col(row.k),q_bar.col(0),
            q_min_,q_max_,velocity_limits_,joint_position_margin_,trust_radius,dt_,row.k,
            nodal_motion ? &previous_command : nullptr,nodal_motion ? &acceleration_limits_ : nullptr);
        s << ",\"reachable_upper\":"; jsonNumber(s,upper);
        s << '}';
      }
      s << "],\"primitive_conflicting_rows\":"<<out.primitive_conflicting_rows<<",\"primitive_row_details\":[";
      for(std::size_t i=0;i<primitive_rows.size();++i) {
        if(i)s<<',';
        const auto& row=primitive_rows[i];
        const auto point=std::find_if(primitive_points.begin(),primitive_points.end(),[&](const PrimitiveObservationPoint& p){
          return (p.point-row.point).lpNorm<Eigen::Infinity>()<=1e-5;
        });
        const bool executable=row.timestep>=q_bar.cols();
        s<<"{\"matrix_row\":"<<n_acc+n_cdf_rows+i<<",\"timestep\":"<<row.timestep<<",\"point\":";jsonVector(s,row.point);
        s<<",\"known_ground\":"<<(row.known_ground?"true":"false");
        s<<",\"ordinary\":"<<(!row.known_ground && !row.occupied_volume && point->ordinary_path_guard?"true":"false")<<",\"floor_m\":";
        jsonNumber(s,(row.known_ground || row.occupied_volume) ? row.required_margin_m :
            primitive_floors.at(care_confidence_map::InitialShellRecovery::key(row.point)));
        s<<",\"reserve_m\":";jsonNumber(s,(row.known_ground || row.occupied_volume)?0.:point->proposal_reserve_m);
        s<<",\"link\":";jsonString(s,row.link);s<<",\"collision_name\":";jsonString(s,row.collision_name);
        s<<",\"executable\":"<<(executable?"true":"false")<<",\"time_s\":";
        jsonNumber(s,executable?executable_samples.at(row.timestep-q_bar.cols()).time_s:row.timestep*dt_);
        s<<",\"distance_m\":";jsonNumber(s,row.distance_m);
        s<<",\"gradient\":";jsonVector(s,row.gradient);
        s<<",\"q_linearization\":";jsonVector(s,row.q_linearization);
        s<<",\"reachable_upper\":";jsonNumber(s,primitive_upper[i]);s<<'}';
      }
      s << "]}";
      out.repair_hard_qp_snapshot = s.str();
    }
    // Preserve one exact NORMAL hard-QP matrix for each obligation state.
    // The diagnostic soft solve is deliberately excluded. This file is only
    // for offline replay; it does not change a candidate or any safety gate.
    if (!repair_mode && !probe_mode) {
      const bool owned = out.selected_obligation_cdf_rows > 0;
      auto& written = owned ? normal_qp_snapshot_owned_written_
                            : normal_qp_snapshot_unowned_written_;
      bool expected = false;
      if (written.compare_exchange_strong(expected, true)) {
        const char* root = std::getenv("ROOT_OUT");
        if (root != nullptr && *root != '\0') {
          const std::string path = std::string(root) +
              (owned ? "/qp_normal_owned_first.json" : "/qp_normal_unowned_first.json");
          std::ofstream file(path);
          if (file) {
            file << "{\"schema\":1,\"query_stamp_ns\":"
                 << batch.header.stamp.toNSec()
                 << ",\"status\":\"" << out.status
                 << "\",\"box_conflicting_cdf_rows\":"
                 << out.box_conflicting_cdf_rows
                 << ",\"max_iter\":" << piqp_max_iterations_
                 << ",\"eps_abs\":"; jsonNumber(file,piqp_eps_abs_);
            file << ",\"eps_rel\":"; jsonNumber(file,piqp_eps_rel_);
            file << ",\"P\":"; jsonSparse(file,P);
            file << ",\"c\":"; jsonVector(file,c);
            file << ",\"A\":"; jsonSparse(file,A);
            file << ",\"b\":"; jsonVector(file,b);
            file << ",\"G\":"; jsonSparse(file,G);
            file << ",\"h_l\":"; jsonVector(file,h_l);
            file << ",\"h_u\":"; jsonVector(file,h_u);
            file << ",\"x_l\":"; jsonVector(file,x_l);
            file << ",\"x_u\":"; jsonVector(file,x_u);
            file << ",\"cdf_rows\":[";
            for (int r=0; r<n_cdf_rows; ++r) {
              if (r) file << ',';
              const auto& row = selected[static_cast<std::size_t>(r)];
              file << "{\"matrix_row\":" << n_acc+r
                   << ",\"pair\":" << row.pair
                   << ",\"timestep\":" << row.k
                   << ",\"distance\":"; jsonNumber(file,row.d);
              file << ",\"margin\":"; jsonNumber(file,row.safety_margin);
              file << ",\"gradient\":"; jsonVector(file,row.g);
              file << ",\"q_linearization\":"; jsonVector(file,row.qlin);
              file << ",\"source\":"
                   << (batch.source_type.empty() ? 0 :
                       static_cast<int>(batch.source_type[static_cast<std::size_t>(row.pair)]))
                   << ",\"point\":[";
              const std::size_t p = static_cast<std::size_t>(3*row.pair);
              for (int j=0;j<3;++j) {
                if (j) file << ',';
                jsonNumber(file,p+j<batch.point_flat.size() ? batch.point_flat[p+j]
                        : std::numeric_limits<double>::quiet_NaN());
              }
              file << "]}";
            }
            file << "]}\n";
            ROS_WARN_STREAM("[LocalSparseSCPPlanner] NORMAL hard-QP snapshot=" << path);
          } else {
            ROS_WARN_STREAM("[LocalSparseSCPPlanner] cannot write NORMAL hard-QP snapshot=" << path);
          }
        }
      }
    }
  }

  if (status != piqp::PIQP_SOLVED ||
      result.x.size() != n ||
      !finiteVector(result.x)) {
    return out;
  }

  out.q = Eigen::MatrixXd::Zero(
      dof_, num_intervals_ + 1);
  out.u = Eigen::MatrixXd::Zero(
      dof_, num_intervals_);
  out.q.col(0) = q_bar.col(0);

  for (int k = 1; k <= num_intervals_; ++k)
    for (int j = 0; j < dof_; ++j)
      out.q(j, k) = result.x[qIndex(k, j)];

  for (int k = 0; k < num_intervals_; ++k)
    for (int j = 0; j < dof_; ++j)
      out.u(j, k) = result.x[uIndex(k, j)];

  if (executable_recheck_only &&
      ((out.q-q_bar).cwiseAbs().maxCoeff()>fixed_recheck_tolerance ||
       (out.u-u_bar).cwiseAbs().maxCoeff()>fixed_recheck_tolerance)) {
    out.status="executable_recheck_not_fixed"; return out;
  }

  out.step_inf =
      (out.q - q_bar).cwiseAbs().maxCoeff();

  double slack_sum = 0.0;
  out.max_slack = 0.0;
  for (int s = 0; s < n_s; ++s) {
    const double value = result.x[slack0 + s];
    slack_sum += value;
    out.max_slack = std::max(out.max_slack, value);
  }
  out.mean_slack =
      n_s > 0 ? slack_sum / static_cast<double>(n_s) : 0.0;

  // Model units, not metres. This is scheduling evidence, never a safety verdict.
  if (!repair_mode && !probe_mode && !force_diagnostic_slack && n_s == 0) {
    const Eigen::VectorXd direction = q_ref.col(num_intervals_)-q_bar.col(0);
    for (const auto& row : selected) {
      if (!is_obligation_row(row.pair) || batch.source_type.empty() ||
          batch.source_type[row.pair] != care_collision_cdf::CollisionCDFConstraintBatch::SOURCE_UNKNOWN)
        continue;
      const double clearance = row.d+row.g.dot(out.q.col(row.k)-row.qlin)-row.safety_margin;
      if (!std::isfinite(clearance) || std::abs(clearance) > .001 ||
          row.g.dot(direction) >= -1e-6) continue;
      const Eigen::Vector3d p(batch.point_flat[3*row.pair],
          batch.point_flat[3*row.pair+1], batch.point_flat[3*row.pair+2]);
      if (out.normal_blocking_unknown_points.size() < 32 &&
          std::none_of(out.normal_blocking_unknown_points.begin(), out.normal_blocking_unknown_points.end(),
              [&](const Eigen::Vector3d& old) { return (p-old).lpNorm<Eigen::Infinity>() <= 1e-5; }))
        out.normal_blocking_unknown_points.push_back(p);
    }
  }

  out.solved = true;
  if(occupied_volume_enabled && !occupied_volume_points.empty()) {
    std::vector<PrimitiveObservationGuard::Row> rows;std::string error;
    if(!geometry_guard.linearizeOccupiedVoxels(out.q,occupied_volume_points,.05,.02,.02,&rows,&error) ||
        std::any_of(rows.begin(),rows.end(),[&](const PrimitiveObservationGuard::Row& row){return row.distance_m<row.required_margin_m-2.*piqp_eps_abs_;})) {
      out.solved=false;out.status="occupied_volume_nonlinear_recheck_failed";return out;
    }
  }
  if(geometry_guard.knownGroundEnabled()) {
    std::vector<PrimitiveObservationGuard::Row> ground_rows;std::string error;
    if(!geometry_guard.linearizeKnownGround(out.q,&ground_rows,&error) ||
        std::any_of(ground_rows.begin(),ground_rows.end(),[&](const PrimitiveObservationGuard::Row& row) {
          return row.distance_m<row.required_margin_m-2.*piqp_eps_abs_;
        })) {
      out.solved=false;out.status="known_ground_nonlinear_recheck_failed";return out;
    }
  }
  if (normal_motion && !normal_limit_samples.empty() && !force_diagnostic_slack) {
    out.normal_curve_checked=true;
    const Eigen::VectorXd lo=q_min_.array()+joint_position_margin_,hi=q_max_.array()-joint_position_margin_;
    std::vector<NormalCurveKnot> curve;std::string error;
    out.normal_curve_valid=normalExecutableCurve(out.q,out.u,dt_,repair_executable_settings_,previous_command,
        out.q.col(0),previous_command,&curve,&error);
    if(out.normal_curve_valid){const auto range=normalCurveRange(curve,lo,hi,velocity_limits_);
      out.normal_curve_valid=range.valid;out.normal_curve_violation=range.violation;}
    if(!preview || preview->native_task_replay) {
      Eigen::VectorXd measured,velocity(dof_);bool fresh=false;
      if(preview){fresh=preview->normal_live_fresh;measured=preview->normal_live_q;velocity=preview->normal_live_v;}
      else
      {std::lock_guard<std::mutex> lock(mutex_);
        const double age=(ros::Time::now()-latest_joint_state_.header.stamp).toSec();
        fresh=has_joint_state_ && !latest_joint_state_.header.stamp.isZero() && age>=0. && age<=.05 && extractMeasuredQ(latest_joint_state_,measured);
        for(int j=0;fresh && j<dof_;++j){const auto it=std::find(latest_joint_state_.name.begin(),latest_joint_state_.name.end(),joint_names_[j]);
          const auto k=std::distance(latest_joint_state_.name.begin(),it);
          fresh=it!=latest_joint_state_.name.end() && k<static_cast<long>(latest_joint_state_.velocity.size()) && std::isfinite(latest_joint_state_.velocity[k]);
          if(fresh)velocity[j]=latest_joint_state_.velocity[k];}}
      if(capture){capture->normal_live_fresh=fresh;capture->normal_live_q=measured;capture->normal_live_v=velocity;}
      if(fresh){out.normal_live_checked=true;out.normal_live_q=measured;out.normal_live_v=velocity;
        out.normal_live_valid=normalExecutableCurve(out.q,out.u,dt_,repair_executable_settings_,previous_command,measured,velocity,&curve,&error);
        if(out.normal_live_valid){const auto range=normalCurveRange(curve,lo,hi,velocity_limits_);
          out.normal_live_valid=range.valid;out.normal_live_violation=range.violation;}}
    }
  }
  std::set<care_confidence_map::InitialShellRecovery::Key> consumed_margin_points;
  if (repair_executable_view_enabled_ && !primitive_points.empty()) {
    std::vector<RepairExecutableSample> samples;
    std::vector<PrimitiveObservationGuard::Row> rows;
    std::string error;
    if (!repairExecutableSamples(out.q,out.u,dt_,executable_settings,&samples,&error)) {
      out.primitive_executable_valid = false;
    } else {
      Eigen::MatrixXd q(dof_,samples.size());
      for (std::size_t i=0;i<samples.size();++i) q.col(i)=samples[i].q;
      if (!geometry_guard.linearize(q,primitive_points,&rows,&error)) {
        out.primitive_executable_valid = false;
      } else {
        double min_distance = std::numeric_limits<double>::infinity();
        for (const auto& row : rows) {
          min_distance=std::min(min_distance,row.distance_m);
          if(row.distance_m<primitive_observation_margin_m_)
            consumed_margin_points.insert(care_confidence_map::InitialShellRecovery::key(row.point));
        }
        out.primitive_executable_min_distance_m=min_distance;
        out.primitive_executable_valid=std::all_of(rows.begin(),rows.end(),[&](const PrimitiveObservationGuard::Row& row) {
          return row.distance_m>=primitive_floors.at(care_confidence_map::InitialShellRecovery::key(row.point));
        });
        if (out.primitive_executable_valid && !force_diagnostic_slack && !previous_vbc_margins && (!preview || preview->native_task_replay)) {
          Eigen::VectorXd measured,velocity(dof_);
          bool fresh = false;
          if(preview){fresh=preview->primitive_live_fresh;measured=preview->primitive_live_q;velocity=preview->primitive_live_v;}
          else {
            std::lock_guard<std::mutex> lock(mutex_);
            const double age=(ros::Time::now()-latest_joint_state_.header.stamp).toSec();
            fresh=has_joint_state_ && !latest_joint_state_.header.stamp.isZero() &&
                age>=0. && age<=.05 && extractMeasuredQ(latest_joint_state_,measured);
            for(int j=0;fresh && j<dof_;++j) {
              const auto it=std::find(latest_joint_state_.name.begin(),latest_joint_state_.name.end(),joint_names_[j]);
              const auto index=std::distance(latest_joint_state_.name.begin(),it);
              fresh=it!=latest_joint_state_.name.end() && index<static_cast<long>(latest_joint_state_.velocity.size()) &&
                  std::isfinite(latest_joint_state_.velocity[index]);
              if(fresh)velocity[j]=latest_joint_state_.velocity[index];
            }
          }
          if(capture){capture->primitive_live_fresh=fresh;capture->primitive_live_q=measured;capture->primitive_live_v=velocity;}
          if (fresh) {
            const Eigen::VectorXd shift=measured-out.q.col(0);
            out.primitive_live_rebase_shift_inf=shift.lpNorm<Eigen::Infinity>();
            std::vector<RepairExecutableSample> live;
            if(!repairLiveExecutableSamples(out.q,out.u,dt_,executable_settings,measured,velocity,&live,&error)) {
              out.primitive_executable_valid=false;
              return out;
            }
            Eigen::MatrixXd rebased(dof_,live.size());
            for(std::size_t i=0;i<live.size();++i)rebased.col(i)=live[i].q;
            rows.clear();
            if (geometry_guard.linearize(rebased,primitive_points,&rows,&error)) {
              out.primitive_live_rebase_checked=true;
              out.primitive_live_rebase_min_m=std::numeric_limits<double>::infinity();
              for (const auto& row:rows) {
                out.primitive_live_rebase_min_m=std::min(out.primitive_live_rebase_min_m,row.distance_m);
                if(row.distance_m<primitive_observation_margin_m_)
                  consumed_margin_points.insert(care_confidence_map::InitialShellRecovery::key(row.point));
                if (row.distance_m<primitive_floors.at(care_confidence_map::InitialShellRecovery::key(row.point)))
                  out.primitive_live_rebase_required=true;
              }
              if(out.primitive_live_rebase_required) {
                out.primitive_live_rebase_start=measured;
                out.primitive_live_rebase_velocity=velocity;
              }
            } else {
              out.primitive_executable_valid=false;
            }
          }
        }
      }
    }
  }
  out.primitive_margin_recovery_points.erase(std::remove_if(out.primitive_margin_recovery_points.begin(),
      out.primitive_margin_recovery_points.end(),[&](const Eigen::Vector3d& p){
        return !consumed_margin_points.count(care_confidence_map::InitialShellRecovery::key(p));
      }),out.primitive_margin_recovery_points.end());
  return out;
}

std::string LocalSparseSCPPlanner::repairWarmStartKeyLocked() const {
  std::ostringstream s;
  s << std::setprecision(17) << mode_epoch_ << ':' << repair_no_progress_.active.revision
    << ':' << std::quoted(repair_no_progress_.active.target)
    << ':' << std::quoted(latest_observation_token_) << ':' << latest_vbc_obligation_points_seq_
    << ':' << safe_frontier_recovery_.stage() << ':' << std::quoted(safe_frontier_recovery_.key)
    << ':' << latest_reference_.header.stamp.toNSec();
  if (repair_prefix_binding_enabled_) s << ':' << latest_prefix_selection_.session << ':'
      << latest_prefix_selection_.revision << ':' << latest_prefix_selection_.prefix_s;
  return s.str();
}

bool LocalSparseSCPPlanner::takeRepairWarmStartLocked(
    const Eigen::VectorXd& q_current, const Eigen::VectorXd& v_current,
    Eigen::MatrixXd& q, Eigen::MatrixXd& u, int& charged_iterations) {
  const auto shape = repair_warm_q_;
  const auto key = repair_warm_key_;
  const int iterations = repair_warm_iterations_;
  const double age = (ros::Time::now()-repair_warm_stamp_).toSec();
  repair_warm_q_.resize(0,0); repair_warm_key_.clear();
  charged_iterations = 0;
  if (!repair_mode_ || probe_mode_ || key.empty() || key != repairWarmStartKeyLocked() ||
      age < 0. || age > .2 || iterations < 1 || iterations >= max_scp_iterations_ ||
      shape.rows()!=dof_ || shape.cols()!=num_intervals_+1 || !finiteMatrix(shape) ||
      q_current.size()!=dof_ || !finiteVector(q_current) ||
      (shape.col(0)-q_current).lpNorm<Eigen::Infinity>() > trust_region_initial_) return false;
  const Eigen::MatrixXd target = shape.colwise() + (q_current-shape.col(0));
  Eigen::MatrixXd rebuilt_q, rebuilt_u;
  if (!buildNormalMotionSeed(q_current,v_current,target,rebuilt_q,rebuilt_u)) return false;
  for(int j=0;j<dof_;++j)
    if (rebuilt_q.row(j).minCoeff()<q_min_[j]+joint_position_margin_ ||
        rebuilt_q.row(j).maxCoeff()>q_max_[j]-joint_position_margin_) return false;
  q=rebuilt_q; u=rebuilt_u; charged_iterations=iterations;
  return true;
}

bool LocalSparseSCPPlanner::buildNormalMotionSeed(
    const Eigen::VectorXd& q_current, const Eigen::VectorXd& v_current,
    const Eigen::MatrixXd& q_ref, Eigen::MatrixXd& q, Eigen::MatrixXd& u) const {
  if (q_current.size()!=dof_ || v_current.size()!=dof_ ||
      q_ref.rows()!=dof_ || q_ref.cols()!=num_intervals_+1 ||
      !finiteVector(q_current) || !finiteVector(v_current) || !finiteMatrix(q_ref)) return false;
  q.resize(dof_,num_intervals_+1);u.resize(dof_,num_intervals_);
  q.col(0)=q_current;
  Eigen::VectorXd velocity=v_current;
  for (int k=0;k<num_intervals_;++k) {
    Eigen::VectorXd next(dof_);
    for (int j=0;j<dof_;++j) {
      const double brake_cap=acceleration_limits_[j]*dt_*(num_intervals_-1-k);
      const double cap=std::min(velocity_limits_[j],brake_cap);
      const double desired=clampValue((q_ref(j,k+1)-q(j,k))/dt_,-cap,cap);
      next[j]=clampValue(desired,velocity[j]-acceleration_limits_[j]*dt_,
                                velocity[j]+acceleration_limits_[j]*dt_);
    }
    u.col(k)=next;q.col(k+1)=q.col(k)+.5*dt_*(velocity+next);velocity=next;
  }
  return finiteMatrix(q) && finiteMatrix(u) && velocity.lpNorm<Eigen::Infinity>()<=1e-6;
}

trajectory_msgs::JointTrajectory
LocalSparseSCPPlanner::makeTrajectoryMessage(
    const Eigen::MatrixXd& q,
    const Eigen::MatrixXd& u,
    const std::string& frame_id,
    const ros::Time& stamp,
    const Eigen::VectorXd* normal_initial_velocity) const {
  trajectory_msgs::JointTrajectory msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;
  msg.joint_names = joint_names_;
  msg.points.resize(
      static_cast<std::size_t>(num_intervals_ + 1));

  for (int k = 0; k <= num_intervals_; ++k) {
    auto& p = msg.points[static_cast<std::size_t>(k)];
    p.time_from_start = ros::Duration(k * dt_);
    p.positions.resize(static_cast<std::size_t>(dof_));
    p.velocities.resize(static_cast<std::size_t>(dof_));
    p.accelerations.resize(static_cast<std::size_t>(dof_));

    for (int j = 0; j < dof_; ++j) {
      p.positions[static_cast<std::size_t>(j)] = q(j, k);
      p.velocities[static_cast<std::size_t>(j)] =
          normal_initial_velocity
              ? (k == 0 ? (*normal_initial_velocity)[j] : u(j, k - 1))
              : (k < num_intervals_ ? u(j, k) : 0.0);

      double a = 0.0;
      if (k == 0) {
        a = 0.0;
      } else if (normal_initial_velocity) {
        a = (u(j, k - 1) - (k == 1 ? (*normal_initial_velocity)[j] : u(j, k - 2))) / dt_;
      } else if (k < num_intervals_) {
        a = (u(j, k) - u(j, k - 1)) / dt_;
      } else {
        a = -u(j, k - 1) / dt_;
      }
      p.accelerations[static_cast<std::size_t>(j)] = a;
    }
  }
  return msg;
}

ros::Time LocalSparseSCPPlanner::publishQueryTrajectory(
    const Eigen::MatrixXd& q,
    const Eigen::MatrixXd& u,
    const std::string& frame_id,
    unsigned long long expected_plan_sequence,
    unsigned long long expected_mode_epoch,
    const RepairNoProgress::Ticket* expected_repair_ticket) {
  ros::Time stamp;
  bool armed = false;
  bool invalid_clock = false;
  bool query_cancelled = false;
  std::string query_rejection;
  care_collision_cdf::CollisionCDFWitnessRequest request;
  Eigen::VectorXd normal_initial_velocity;
  std::vector<RepairExecutableSample> executable_queries;
  bool executable_contract_error=false;
  {
    // C5.31: arm the expected stamp BEFORE publishing. In the event-driven
    // GCDF path a batch can return within a few milliseconds; publishing first
    // creates a race where the callback sees waiting_for_cdf_=false.
    std::lock_guard<std::mutex> lock(mutex_);
    // A callback may supersede a failed solve after the worker decision but
    // before this query is armed. Never bind the old seed to the new plan.
    if (expected_plan_sequence != 0 &&
        (expected_plan_sequence != plan_sequence_ ||
         expected_mode_epoch != mode_epoch_ || expected_mode_epoch != plan_mode_epoch_ ||
         !plan_running_ || !repair_mode_ || !plan_repair_mode_ || probe_mode_ ||
         expected_repair_ticket == nullptr ||
         !repair_no_progress_.current(*expected_repair_ticket))) {
      // Reject the stale seed, but finish the cancellation handoff if this
      // worker still owns the current plan and no newer query is armed.
      // Otherwise running=true/waiting=false has neither a timeout nor a
      // worker wakeup, and the timer cannot start the requested fresh plan.
      // A delayed worker must never clear another plan or its active query.
      const bool same_plan = expected_plan_sequence == plan_sequence_;
      if (same_plan && plan_running_ && !waiting_for_cdf_ && !pending_batch_) {
        plan_running_ = false;
        waiting_for_cdf_ = false;
        pending_batch_.reset();
        current_witness_request_ = care_collision_cdf::CollisionCDFWitnessRequest{};
        witness_response_required_ = false;
        plan_executable_iterates_ = RepairExecutableIterates{};
        plan_has_hard_solution_ = false;
        plan_q_bar_.resize(0, 0);
        plan_u_bar_.resize(0, 0);
        plan_previous_q_bar_.resize(0, 0);
        plan_previous_u_bar_.resize(0, 0);
        plan_probe_restore_pending_hard_recheck_ = false;
        plan_candidate_replacement_refresh_only_ = false;
        last_plan_finish_time_ = ros::Time::now();
        // Identity changes are not failed geometry evidence. Preserve the
        // existing failure/direction ledgers and use all original replan gates.
        requestPlanLocked("repair_query_identity_changed");
        query_cancelled = true;
      }
      std::ostringstream disposition;
      disposition << "disposition=" << (query_cancelled ? "cancelled" : "ignored")
          << " expected_plan_seq=" << expected_plan_sequence
          << " current_plan_seq=" << plan_sequence_
          << " expected_mode_epoch=" << expected_mode_epoch
          << " current_mode_epoch=" << mode_epoch_
          << " expected_target_revision=" << (expected_repair_ticket ? expected_repair_ticket->revision : 0)
          << " expected_progress_epoch=" << (expected_repair_ticket ? expected_repair_ticket->progress : 0)
          << " current_target_revision=" << repair_no_progress_.active.revision
          << " current_progress_epoch=" << repair_no_progress_.active.progress
          << " running=" << plan_running_ << " waiting_for_cdf=" << waiting_for_cdf_
          << " replan_requested=" << plan_requested_
          << " replan_reason=" << plan_request_reason_;
      query_rejection = disposition.str();
    } else if (plan_running_) {
      const ros::Time now = ros::Time::now();
      // Keep the existing stamp-correlated wire API while avoiding collisions
      // when several SCP iterations run during one /clock update. Do not reset
      // identity between plans. A backward clock must not mint future evidence.
      invalid_clock = now.isZero() || now < last_query_ros_time_;
      if (!invalid_clock) {
        stamp = now;
        if (stamp <= current_query_stamp_) {
          stamp = current_query_stamp_ + ros::Duration(0, 1);
          ++query_stamp_adjustments_;
        }
        last_query_ros_time_ = now;
        current_query_stamp_ = stamp;
        current_query_wall_ = ros::WallTime::now();
        waiting_for_cdf_ = true;
        request.header.stamp = stamp;
        request.header.frame_id = frame_id;
        request.dof = dof_;
        request.plan_sequence = plan_sequence_;
        request.mode_epoch = plan_mode_epoch_;
        request.target_revision = plan_repair_ticket_.revision;
        request.progress_epoch = plan_repair_ticket_.progress;
        if (plan_repair_mode_ || !latest_vbc_obligation_points_.empty()) {
          // Split the active visibility target from confirmed collision
          // witnesses. The target is queried over the configured obligation
          // horizon (20 knots in C5.5) so the body cannot enter an UNKNOWN
          // target voxel before the sensor has observed it. This also runs in
          // PROBE_NORMAL, where an obligation can arrive while a probe is in
          // flight. Ordinary UNKNOWN rows retain their existing mode-specific
          // horizon and the target is not copied into the persistent witness
          // list.
          auto has_witness_identity = [&](int timestep,
                                          const Eigen::Vector3d& point) {
            for (std::size_t i = 0; i < request.original_timestep.size(); ++i) {
              if (request.original_timestep[i] != timestep ||
                  request.point_flat.size() < 3 * i + 3) {
                continue;
              }
              bool match = true;
              for (int j = 0; j < 3; ++j) {
                match = match &&
                        std::fabs(request.point_flat[3 * i + j] - point[j]) <=
                            1e-5;
              }
              if (match) return true;
            }
            return false;
          };

          // The blocker-aware visibility node publishes only the stack-top
          // target. Keep it separate from GCDF witnesses and request fresh
          // CDF rows only over the short target-guard horizon. The planner is
          // receding-horizon, so the guard is refreshed on every replan.
          std::vector<Eigen::Vector3d> vbc_obligation_points =
              latest_vbc_obligation_points_;
          const int visibility_target_horizon = std::max(
              1, std::min(num_intervals_,
                          visibility_obligation_cdf_horizon_steps_));
          for (const auto& point : vbc_obligation_points) {
            for (int timestep = 1; timestep <= visibility_target_horizon;
                 ++timestep) {
              if (has_witness_identity(timestep, point)) continue;
              for (int j = 0; j < 3; ++j) request.point_flat.push_back(point[j]);
              request.original_timestep.push_back(timestep);
              for (int j = 0; j < dof_; ++j) request.q_flat.push_back(q(j, timestep));
            }
          }

          // A persistent witness is one point identity. Re-query it over the
          // complete current task horizon rather than storing a separate slot
          // for every old trajectory timestep where it was first observed.
          for (const auto& witness : repair_unknown_witnesses_) {
            for (int timestep = 1; timestep <= num_intervals_; ++timestep) {
              if (has_witness_identity(timestep, witness.point)) continue;
              for (int j = 0; j < 3; ++j) request.point_flat.push_back(witness.point[j]);
              request.original_timestep.push_back(timestep);
              for (int j = 0; j < dof_; ++j) request.q_flat.push_back(q(j, timestep));
            }
          }
        }
        if(repair_executable_view_enabled_ && plan_repair_mode_ && !plan_probe_mode_) {
          auto settings=repair_executable_settings_;settings.initial_velocity=plan_previous_command_;
          if(repair_prefix_binding_enabled_)settings.prefixes={plan_prefix_selection_.prefix_s};
          std::string error;
          if(!repairCdfExecutableSamples(q,u,dt_,settings,&executable_queries,&error))executable_contract_error=true;
          else {
            std::vector<Eigen::Vector3d> points=latest_vbc_obligation_points_;
            for(const auto& w:repair_unknown_witnesses_)if(std::none_of(points.begin(),points.end(),
                [&](const Eigen::Vector3d& p){return (p-w.point).lpNorm<Eigen::Infinity>()<=1e-5;}))points.push_back(w.point);
            for(std::size_t i=0;i<executable_queries.size();++i)for(const auto& p:points) {
              request.original_timestep.push_back(num_intervals_+1+i);
              for(int j=0;j<3;++j)request.point_flat.push_back(p[j]);
              for(int j=0;j<dof_;++j)request.q_flat.push_back(executable_queries[i].q[j]);
            }
          }
        }
        current_witness_request_ = request;
        witness_response_required_ = !request.original_timestep.empty();
        if (witness_response_required_) repair_witness_requires_reobserve_ = true;
        if (repair_executable_view_enabled_ && !plan_probe_mode_)
          normal_initial_velocity = plan_previous_command_;
        armed = !invalid_clock && !executable_contract_error;
      }
    }
  }
  if (!query_rejection.empty()) {
    ROS_WARN_STREAM("[CDF_QUERY_DISPOSITION] " << query_rejection);
    publishSummary(query_cancelled ? "cdf_query_identity_cancelled" :
                                    "cdf_query_identity_rejected");
    return ros::Time();
  }
  if (armed) {
    publishSummary("cdf_query_armed");
    witness_request_pub_.publish(request);
    auto query=makeTrajectoryMessage(q,u,frame_id,stamp,
        normal_initial_velocity.size()==dof_ ? &normal_initial_velocity : nullptr);
    // Extra points are a pose-query array, not a proposed execution schedule.
    // Their exact index/q is also bound in the witness request above.
    for(const auto& sample:executable_queries) {
      trajectory_msgs::JointTrajectoryPoint point;
      point.positions.assign(sample.q.data(),sample.q.data()+dof_);
      point.velocities.assign(dof_,0.);point.accelerations.assign(dof_,0.);
      point.time_from_start=ros::Duration(query.points.size()*dt_);query.points.push_back(point);
    }
    query_trajectory_pub_.publish(query);
  }
  if (invalid_clock) abortPlan("cdf_query_clock_invalid");
  if (executable_contract_error) abortPlan("cdf_query_executable_contract_error");
  return stamp;
}

void LocalSparseSCPPlanner::publishCandidateTrajectory(
    const Eigen::MatrixXd& q,
    const Eigen::MatrixXd& u,
    const std::string& frame_id,
    const std::string& observation_token,
    const care_collision_cdf::CollisionCDFConstraintBatch& batch,
    const Eigen::MatrixXd& query_q, const Eigen::MatrixXd& query_u,
    const std::vector<Eigen::Vector3d>& normal_blocking_unknown_points,
    const std::vector<Eigen::Vector3d>& geometric_restore_points,
    bool margin_recovery,
    const std::vector<Eigen::Vector3d>& fixed_start_recovery_points) {
  if (repair_prefix_binding_enabled_ && plan_repair_mode_ && !plan_probe_mode_ &&
      !prefixSelectionCurrentLocked(plan_prefix_selection_))return;
  const auto* normal_initial_velocity =
      repair_executable_view_enabled_ && !plan_probe_mode_ && (geometric_restore_points.empty() || margin_recovery) ? &plan_previous_command_ : nullptr;
  const auto msg = makeTrajectoryMessage(q, u, frame_id, ros::Time::now(), normal_initial_velocity);
  if((!plan_repair_mode_ || repair_goal_stopping_enabled_) && !plan_probe_mode_ &&
      plan_terminal_context_==terminalContextLocked())
    terminal_retention_.candidate(msg.header.stamp.toNSec(),plan_terminal_context_);
  if (!plan_repair_mode_ && !plan_probe_mode_) {
    const double age = (ros::Time::now()-last_query_ros_time_).toSec();
    const bool matched = std::any_of(normal_blocking_unknown_points.begin(), normal_blocking_unknown_points.end(),
        [&](const Eigen::Vector3d& p) {
          return std::any_of(latest_vbc_obligation_points_.begin(), latest_vbc_obligation_points_.end(),
              [&](const Eigen::Vector3d& live) { return (p-live).lpNorm<Eigen::Infinity>() <= 1e-5; });
        });
    const double span = (q.colwise()-q.col(0)).cwiseAbs().maxCoeff();
    const auto context = normalUnknownContextLocked();
    normal_unknown_no_progress_.remember(msg.header.stamp.toNSec(), batch.header.stamp.toNSec(),
        context, span, matched && age >= 0. && age <= .5);
    const auto task_context=normalTaskProgressContextLocked();
    if (normal_unknown_task_progress_.remember(msg.header.stamp.toNSec(),batch.header.stamp.toNSec(),
        task_context,matched && age>=0. && age<=.5))
      ROS_INFO_STREAM("[NORMAL_TASK_CANDIDATE] raw_candidate_stamp_ns=" << msg.header.stamp.toNSec()
          << " query_stamp_ns=" << batch.header.stamp.toNSec() << " context=" << task_context
          << " raw_span_inf=" << span);
    if (normal_unknown_no_progress_.pending.count(msg.header.stamp.toNSec()))
      ROS_INFO_STREAM("[NORMAL_UNKNOWN_CANDIDATE] raw_candidate_stamp_ns=" << msg.header.stamp.toNSec()
          << " query_stamp_ns=" << batch.header.stamp.toNSec() << " context=" << context
          << " raw_span_inf=" << span);
  }
  if (plan_repair_mode_ && !plan_probe_mode_ &&
      !final_vbc_no_progress_.remember(msg.header.stamp.toNSec(), mode_epoch_, plan_sequence_)) {
    ROS_ERROR("[LocalSparseSCPPlanner] duplicate/zero candidate identity; not published");
    return;
  }
  if (plan_repair_mode_ && !plan_probe_mode_) {
    const auto context=vbcFailureContextLocked();
    if (context.token==observation_token)
      vbc_failure_direction_memory_.remember(msg.header.stamp.toNSec(),batch.header.stamp.toNSec(),context);
    if (safe_frontier_recovery_enabled_ && plan_repair_observation_phase_ &&
        (plan_frontier_.active || !geometric_restore_points.empty())) {
      // Owned-margin hard-QP proposals can precede recovery activation. Keep
      // their nominal direction so a later path counterexample can exclude it.
      if(!safe_frontier_recovery_.active && !geometric_restore_points.empty())
        safe_frontier_recovery_.detour_direction=q.col(q.cols()-1)-q.col(0);
      safe_frontier_recovery_.remember(msg.header.stamp.toNSec(),
          (q.colwise()-q.col(0)).cwiseAbs().maxCoeff());
    }
    while (gcdf_feedback_pending_.size() >= FinalVbcNoProgress::max_pending)
      gcdf_feedback_pending_.erase(gcdf_feedback_pending_.begin());
    gcdf_feedback_pending_.emplace(msg.header.stamp.toNSec(),GcdfCandidate{
        FinalVbcNoProgress::Candidate{plan_repair_ticket_,mode_epoch_,plan_sequence_},observation_token,
        msg.header.stamp,ros::WallTime::now()+ros::WallDuration(cdf_wait_timeout_s_)});
  }
  if (rejection_snapshots_enabled_ || (std::getenv("CARE_ROLLING_TASK_OBSERVATION") &&
      std::string(std::getenv("CARE_ROLLING_TASK_OBSERVATION"))=="true")) {
    std::ostringstream out;
    out << "{\"schema\":1,\"raw_candidate_stamp_ns\":\"" << msg.header.stamp.toNSec()
        << "\",\"plan_seq\":" << plan_sequence_ << ",\"mode_epoch\":" << mode_epoch_
        << ",\"target_revision\":" << final_vbc_no_progress_.progress.active.revision
        << ",\"progress_epoch\":" << final_vbc_no_progress_.progress.active.progress
        << ",\"observation_token\":";
    auditString(out,observation_token);
    if (std::getenv("CARE_ROLLING_TASK_OBSERVATION") &&
        std::string(std::getenv("CARE_ROLLING_TASK_OBSERVATION"))=="true") {
      out << ",\"rolling_task\":true,\"goal_revision\":"<<terminal_goal_revision_
          <<",\"repair\":"<<(plan_repair_mode_?"true":"false")
          <<",\"probe\":"<<(plan_probe_mode_?"true":"false")
          <<",\"candidate_trajectory\":"; auditTrajectory(out,msg);
    }
    out << ",\"query_trajectory\":";
    auto query_message=makeTrajectoryMessage(query_q,query_u,frame_id,batch.header.stamp,
        repair_executable_view_enabled_ && !plan_probe_mode_ ? &plan_previous_command_ : nullptr);
    if(repair_executable_view_enabled_ && plan_repair_mode_ && !plan_probe_mode_) {
      auto settings=repair_executable_settings_;settings.initial_velocity=plan_previous_command_;
      if(repair_prefix_binding_enabled_)settings.prefixes={plan_prefix_selection_.prefix_s};
      std::vector<RepairExecutableSample> samples;
      if(repairCdfExecutableSamples(query_q,query_u,dt_,settings,&samples))for(const auto& s:samples) {
        trajectory_msgs::JointTrajectoryPoint point;point.positions.assign(s.q.data(),s.q.data()+dof_);
        point.velocities.assign(dof_,0.);point.accelerations.assign(dof_,0.);
        point.time_from_start=ros::Duration(query_message.points.size()*dt_);query_message.points.push_back(point);
      }
    }
    auditTrajectory(out,query_message);
    out << ",\"local_gcdf_batch\":"; auditBatch(out,batch); out << '}';
    std_msgs::String context; context.data = out.str();
    if (context.data.size() <= 2*1024*1024) candidate_audit_context_pub_.publish(context);
    else ROS_WARN("[LocalSparseSCPPlanner] audit context exceeds 2 MiB cap; diagnostic omitted");
  }
  std_msgs::String identity;
  identity.data = "observation_token=" + observation_token +
      " raw_candidate_stamp_ns=" + std::to_string(msg.header.stamp.toNSec());
  observation_identity_pub_.publish(identity);
  ROS_INFO_STREAM("[OBS_CANDIDATE] observation_token=" << observation_token
                  << " raw_candidate_stamp_ns=" << msg.header.stamp.toNSec());
  candidate_trajectory_pub_.publish(msg);
  if (repair_prefix_binding_enabled_) {
    CandidateTransfer planned;
    planned.action="planned";planned.trajectory=msg;
    if (plan_repair_mode_ && !plan_probe_mode_) {
      const auto& c=plan_prefix_selection_;
      std::ostringstream out;out<<std::setprecision(17)<<"{\"schema\":1,\"session\":\""<<c.session
          <<"\",\"revision\":"<<c.revision<<",\"prefix_s\":"<<c.prefix_s
          <<",\"brake_dt_s\":"<<c.brake_dt_s<<",\"brake_steps\":"<<c.brake_steps<<",\"acceleration\":[";
      for (std::size_t j=0;j<c.acceleration.size();++j) {if(j)out<<',';out<<c.acceleration[j];}
      out<<']';
      if(!geometric_restore_points.empty()) {
        out<<",\"geometric_restore\":{\"schema\":2,\"raw_candidate_stamp_ns\":\""<<msg.header.stamp.toNSec()
           <<"\",\"query_stamp_ns\":\""<<batch.header.stamp.toNSec()<<"\",\"mode_epoch\":\""<<mode_epoch_
           <<"\",\"query_ros_ns\":\""<<last_query_ros_time_.toNSec()
           <<"\",\"observation_token\":\""<<observation_token<<"\",\"points\":[";
        for(std::size_t i=0;i<geometric_restore_points.size();++i){if(i)out<<',';const auto& p=geometric_restore_points[i];out<<'['<<p.x()<<','<<p.y()<<','<<p.z()<<']';}
        out<<']';
        if(margin_recovery)out<<",\"margin_recovery\":true";
        if(!fixed_start_recovery_points.empty()) {
          out<<",\"fixed_start_recovery_points\":[";
          for(std::size_t i=0;i<fixed_start_recovery_points.size();++i) {
            if(i)out<<',';const auto& p=fixed_start_recovery_points[i];out<<'['<<p.x()<<','<<p.y()<<','<<p.z()<<']';
          }
          out<<']';
        }
        out<<'}';
      }
      out<<'}';planned.planned_prefix_context=out.str();
    }
    if(plan_repair_mode_ && !plan_probe_mode_ &&
        (!geometric_restore_points.empty() || primitive_observation_guard_.knownGroundEnabled())) {
      while(admission_pending_.size()>=FinalVbcNoProgress::max_pending)admission_pending_.erase(admission_pending_.begin());
      admission_pending_.emplace(msg.header.stamp.toNSec(),AdmissionCandidate{
          planned,plan_prefix_selection_,mode_epoch_,observation_token,safe_frontier_recovery_.key});
    }
    if(primitive_observation_guard_.knownGroundEnabled() && terminal_goal_valid_) {
      while(ground_admission_.size()>=64)ground_admission_.erase(ground_admission_.begin());
      ground_admission_.emplace(msg.header.stamp.toNSec(),GroundAdmission{
          planned,terminal_goal_revision_,mode_epoch_,plan_repair_mode_,plan_probe_mode_,
          groundContextLocked(),ros::WallTime::now()+ros::WallDuration(cdf_wait_timeout_s_)});
    }
    planned_candidate_pub_.publish(planned);
  }
}


bool LocalSparseSCPPlanner::publishLocalGcdfRecoveryEvidence(
    const care_collision_cdf::CollisionCDFConstraintBatch& batch,
    const SparseSolveResult& result,
    const Eigen::MatrixXd& q_bar,
    const Eigen::MatrixXd& u_bar,
    const std::string& frame_id) {
  if (result.selected_unknown_pair_indices.empty()) {
    ++local_gcdf_recovery_drop_count_;
    return false;
  }
  if (batch.header.stamp.isZero()) {
    ++local_gcdf_recovery_drop_count_;
    ROS_WARN(
        "[LocalSparseSCPPlanner] local-GCDF recovery dropped: zero batch stamp");
    return false;
  }

  int earliest = std::numeric_limits<int>::max();
  for (const int pair : result.selected_unknown_pair_indices) {
    if (pair < 0 ||
        pair >= static_cast<int>(batch.original_timestep.size())) {
      continue;
    }
    earliest = std::min(
        earliest,
        batch.original_timestep[static_cast<std::size_t>(pair)]);
  }
  if (earliest < 1 || earliest > num_intervals_) {
    ++local_gcdf_recovery_drop_count_;
    ROS_WARN_STREAM(
        "[LocalSparseSCPPlanner] local-GCDF recovery dropped: invalid earliest="
        << earliest);
    return false;
  }

  std::set<std::tuple<double, double, double>> seen;
  std::vector<std::tuple<double, double, double>> points;
  for (const int pair : result.selected_unknown_pair_indices) {
    if (pair < 0 ||
        pair >= static_cast<int>(batch.original_timestep.size()) ||
        batch.original_timestep[static_cast<std::size_t>(pair)] != earliest) {
      continue;
    }
    const std::size_t j = static_cast<std::size_t>(3 * pair);
    if (j + 2 >= batch.point_flat.size()) {
      continue;
    }
    const double x = batch.point_flat[j];
    const double y = batch.point_flat[j + 1];
    const double z = batch.point_flat[j + 2];
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      continue;
    }
    const auto key = std::make_tuple(x, y, z);
    if (seen.insert(key).second) {
      points.push_back(key);
    }
  }

  if (points.empty()) {
    ++local_gcdf_recovery_drop_count_;
    ROS_WARN(
        "[LocalSparseSCPPlanner] local-GCDF recovery dropped: no earliest UNKNOWN points");
    return false;
  }

  const unsigned long long seq = ++local_gcdf_recovery_event_count_;
  trajectory_msgs::JointTrajectory trajectory =
      makeTrajectoryMessage(
          q_bar, u_bar, frame_id, batch.header.stamp);
  trajectory.header.seq = static_cast<uint32_t>(
      seq & 0xffffffffULL);

  std_msgs::Float64MultiArray event;
  const double sweep_s = static_cast<double>(earliest) * dt_;
  event.data.reserve(6 + 3 * points.size());
  event.data.push_back(static_cast<double>(seq));
  event.data.push_back(static_cast<double>(batch.header.stamp.sec));
  event.data.push_back(static_cast<double>(batch.header.stamp.nsec));
  event.data.push_back(sweep_s);
  event.data.push_back(static_cast<double>(earliest));
  event.data.push_back(static_cast<double>(points.size()));
  for (const auto& p : points) {
    event.data.push_back(std::get<0>(p));
    event.data.push_back(std::get<1>(p));
    event.data.push_back(std::get<2>(p));
  }

  // Publish trajectory first so the blocker-aware acquisition cache can match
  // the subsequent exact-voxel event by immutable header stamp.
  gcdf_recovery_trajectory_pub_.publish(trajectory);
  gcdf_recovery_event_pub_.publish(event);

  std::ostringstream point_oss;
  bool first = true;
  for (const auto& p : points) {
    if (!first) point_oss << ";";
    first = false;
    point_oss << "[" << std::get<0>(p) << ","
              << std::get<1>(p) << ","
              << std::get<2>(p) << "]";
  }
  ROS_WARN_STREAM(
      "[LOCAL_GCDF_RECOVERY] seq=" << seq
      << " stamp=" << batch.header.stamp
      << " earliest_timestep=" << earliest
      << " sweep_s=" << sweep_s
      << " point_count=" << points.size()
      << " points=" << point_oss.str());
  return true;
}

void LocalSparseSCPPlanner::publishSummary(
    const std::string& event,
    const SparseSolveResult* result,
    double total_plan_ms) {
  unsigned long long plan_seq = 0;
  unsigned long long batches = 0;
  unsigned long long misses = 0;
  unsigned long long solves = 0;
  unsigned long long failures = 0;
  int scp_iter = 0;
  double trust = 0.0;
  bool running = false;
  bool repair = false;
  bool probe = false;
  unsigned long long goal_stop_execution=0,goal_stop_sample=0;
  int probe_restore_attempts = 0;
  bool observation_phase = true;
  int witness_count = 0;
  int vbc_obligation_point_count = 0;
  long long vbc_obligation_id = -1;
  unsigned long long vbc_obligation_points_seq = 0;
  bool witnesses_fresh = true;
  int final_vbc_failures = 0;
  bool final_vbc_hold = false;
  bool replacement_pending = false, replacement_trigger_pending = false;
  bool fixed_start_unknown_hold = false;
  unsigned long long replacement_trigger_count = 0;
  unsigned long long replacement_trigger_raw = 0;
  bool probe_restore_pending_recheck = false;
  unsigned long long probe_restore_total = 0;
  unsigned long long probe_restore_hard_success_total = 0;
  double active_slack_weight = 0.0;
  unsigned long long latest_execution_stamp_ns = 0;
  unsigned long long last_repair_completion_stamp_ns = 0;
  unsigned long long repair_completion_replan_count = 0;
  unsigned long long repair_completion_duplicate_count = 0;
  unsigned long long last_normal_completion_stamp_ns = 0;
  unsigned long long normal_completion_replan_count = 0;
  unsigned long long normal_completion_duplicate_count = 0;
  unsigned long long normal_reference_refresh_count = 0;
  unsigned long long normal_refresh_blocked_replan_count = 0;
  bool normal_reference_refresh_pending = false;
  unsigned long long smooth_handoff_replan_count = 0;
  unsigned long long smooth_handoff_replan_suppressed_probe_count = 0;
  unsigned long long smooth_handoff_replan_suppressed_busy_count = 0;
  unsigned long long stale_mode_candidate_discard_count = 0;
  int normal_qp_backtrack_attempts = 0;
  unsigned long long normal_qp_backtrack_total = 0;
  double last_cdf_roundtrip_ms = 0.0;
  unsigned long long query_stamp_ns = 0, query_ros_time_ns = 0, query_stamp_adjustments = 0;
  double plan_cdf_roundtrip_sum_ms = 0.0;
  double plan_cdf_roundtrip_max_ms = 0.0;
  int plan_cdf_roundtrip_count = 0;
  std::string init_mode = "unknown";
  std::string observation_token;
  bool frontier_active = false;
  double frontier_weight_scale = 0.0;
  double frontier_qvis_weight_scale = 1.0;
  double frontier_target_shift_inf =
      std::numeric_limits<double>::quiet_NaN();
  double obligation_margin_effective = 0.0;
  int vbc_feedback_rejections = 0;
  int vbc_feedback_matched_points = 0;
  int safe_stall_attempts = 0, safe_stall_tiny_commits = 0;
  int safe_stall_trigger_commits = 0;
  double safe_stall_trigger_duration = 0., safe_stall_trigger_motion = 0.;
  double safe_stall_raw_span = 0.;
  bool safe_stall_active = false, safe_stall_reselect = false;
  std::string observation_stall_reason;
  double observation_best_gap = -1., observation_window_age = -1.;
  int observation_window_commits = 0;
  int view_window_commits=0;
  double view_window_age=-1.,view_fov_best=0.;
  std::string recovery_advance_reason, action_progress_decision;
  double action_progress_elapsed=0.,action_progress_budget=0.,action_fov_gain=0.;
  int action_progress_commits=0,executed_view_trials=0;
  bool execution_response_blocked = false;
  unsigned long long execution_response_faults = 0;
  double execution_requested_travel = 0., execution_measured_motion = 0.;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    plan_seq = plan_sequence_;
    execution_response_blocked = execution_response_monitor_.blocked;
    execution_response_faults = execution_response_monitor_.faults;
    execution_requested_travel = execution_response_monitor_.expected_travel;
    execution_measured_motion = execution_response_monitor_.measured_motion;
    final_vbc_failures = final_vbc_no_progress_.progress.failures();
    final_vbc_hold = final_vbc_no_progress_.progress.blocked();
    replacement_pending = candidate_replacement_pending_;
    fixed_start_unknown_hold = fixed_start_clearance_hold_.armed;
    replacement_trigger_pending = candidate_replacement_trigger_pending_;
    replacement_trigger_count = candidate_replacement_trigger_count_;
    replacement_trigger_raw = candidate_replacement_trigger_raw_;
    safe_stall_attempts = safe_frontier_recovery_.attempts;
    safe_stall_tiny_commits = safe_frontier_recovery_.tiny_commits;
    safe_stall_trigger_commits = safe_frontier_recovery_.last_stall_commits;
    safe_stall_trigger_duration = safe_frontier_recovery_.last_stall_duration;
    safe_stall_trigger_motion = safe_frontier_recovery_.last_stall_motion;
    safe_stall_raw_span = safe_frontier_recovery_.last_raw_span;
    observation_stall_reason = safe_frontier_recovery_.last_stall_reason;
    observation_best_gap = safe_frontier_recovery_.observation_best_gap;
    observation_window_commits = safe_frontier_recovery_.observation_commits;
    view_window_commits=safe_frontier_recovery_.view_window_commits;
    view_fov_best=safe_frontier_recovery_.geometry_best;
    recovery_advance_reason=safe_frontier_recovery_.last_advance_reason;
    const auto& action=safe_frontier_recovery_.action_progress;
    action_progress_decision=action.decision;action_progress_elapsed=action.elapsed;
    action_progress_budget=action.budget;action_fov_gain=action.fov_gain;action_progress_commits=action.commits;
    executed_view_trials=safe_frontier_recovery_.executed_view_trials;
    if(safe_frontier_recovery_.view_window_start>=0.)
      view_window_age=ros::Time::now().toSec()-safe_frontier_recovery_.view_window_start;
    if (safe_frontier_recovery_.observation_window_start>=0.)
      observation_window_age = ros::Time::now().toSec()-safe_frontier_recovery_.observation_window_start;
    safe_stall_active = safe_frontier_recovery_enabled_ && safe_frontier_recovery_.active;
    safe_stall_reselect = safe_stall_active && (safe_frontier_recovery_.blocked() ||
        final_vbc_no_progress_.progress.blocked() || repair_no_progress_.blocked());
    if(repair_goal_stopping_enabled_ && terminal_retention_.armed) {
      if(retainTerminalLocked()) {
        goal_stop_execution=terminal_retention_.armed;
        goal_stop_sample=latest_joint_state_.header.stamp.toNSec();
      } else requestPlanLocked("terminal_retention_released");
    }
    observation_token = plan_observation_token_;
    batches = cdf_batch_received_;
    query_stamp_ns = current_query_stamp_.toNSec();
    query_ros_time_ns = last_query_ros_time_.toNSec();
    query_stamp_adjustments = query_stamp_adjustments_;
    misses = cdf_stamp_miss_;
    solves = solve_count_;
    failures = solve_failure_count_;
    scp_iter = scp_iteration_;
    trust = trust_radius_;
    running = plan_running_;
    repair = plan_repair_mode_;
    probe = plan_probe_mode_;
    observation_phase = plan_repair_observation_phase_;
    witness_count = static_cast<int>(repair_unknown_witnesses_.size());
    vbc_obligation_point_count =
        static_cast<int>(latest_vbc_obligation_points_.size());
    vbc_obligation_id = latest_vbc_obligation_id_;
    vbc_obligation_points_seq = latest_vbc_obligation_points_seq_;
    witnesses_fresh = !repair_witness_requires_reobserve_;
    probe_restore_attempts = plan_probe_feasibility_restore_attempts_;
    probe_restore_pending_recheck =
        plan_probe_restore_pending_hard_recheck_;
    probe_restore_total = probe_feasibility_restore_count_;
    probe_restore_hard_success_total =
        probe_feasibility_restore_success_count_;
    active_slack_weight = plan_cdf_slack_linear_weight_;
    latest_execution_stamp_ns = latest_execution_stamp_ns_;
    last_repair_completion_stamp_ns =
        last_repair_completed_execution_stamp_ns_;
    repair_completion_replan_count = repair_completion_replan_count_;
    repair_completion_duplicate_count =
        repair_completion_duplicate_count_;
    last_normal_completion_stamp_ns =
        last_normal_completed_execution_stamp_ns_;
    normal_completion_replan_count =
        normal_completion_replan_count_;
    normal_completion_duplicate_count =
        normal_completion_duplicate_count_;
    normal_reference_refresh_count =
        normal_reference_refresh_count_;
    normal_refresh_blocked_replan_count =
        normal_refresh_blocked_replan_count_;
    normal_reference_refresh_pending =
        normal_reference_refresh_pending_;
    smooth_handoff_replan_count = smooth_handoff_replan_count_;
    smooth_handoff_replan_suppressed_probe_count =
        smooth_handoff_replan_suppressed_probe_count_;
    smooth_handoff_replan_suppressed_busy_count =
        smooth_handoff_replan_suppressed_busy_count_;
    stale_mode_candidate_discard_count =
        stale_mode_candidate_discard_count_;
    normal_qp_backtrack_attempts = plan_qp_backtrack_policy_.attempts;
    normal_qp_backtrack_total = normal_qp_backtrack_count_;
    last_cdf_roundtrip_ms = last_cdf_roundtrip_ms_;
    plan_cdf_roundtrip_sum_ms = plan_cdf_roundtrip_sum_ms_;
    plan_cdf_roundtrip_max_ms = plan_cdf_roundtrip_max_ms_;
    plan_cdf_roundtrip_count = plan_cdf_roundtrip_count_;
    init_mode = plan_initialization_mode_;
    frontier_active = plan_frontier_.active;
    frontier_weight_scale = plan_frontier_.frontier_weight_scale;
    frontier_qvis_weight_scale = plan_frontier_.qvis_weight_scale;
    obligation_margin_effective =
        visibility_obligation_cdf_margin_effective_.load();
    vbc_feedback_rejections = vbc_feedback_rejection_count_;
    vbc_feedback_matched_points = vbc_feedback_matched_point_count_;
    if (plan_frontier_.q.size() == dof_ &&
        plan_q_current_.size() == dof_ &&
        finiteVector(plan_frontier_.q) &&
        finiteVector(plan_q_current_)) {
      frontier_target_shift_inf =
          (plan_frontier_.q - plan_q_current_)
              .lpNorm<Eigen::Infinity>();
    }
  }

  std::ostringstream oss;
  if (result && result->trace_plan_sequence) {
    plan_seq = result->trace_plan_sequence;
    observation_token = result->trace_observation_token;
    repair = result->trace_repair;
    probe = result->trace_probe;
    observation_phase = result->trace_repair_observation_phase;
    witness_count = result->trace_repair_witnesses;
    witnesses_fresh = result->trace_repair_witnesses_fresh;
  }
  oss << "C5_4_LOCAL_SCP"
      << " event=" << event
      << " execution_response_blocked=" << static_cast<int>(execution_response_blocked)
      << " execution_response_faults=" << execution_response_faults
      << " execution_requested_travel_rad=" << execution_requested_travel
      << " execution_measured_motion_rad=" << execution_measured_motion
      << " query_stamp_ns=" << query_stamp_ns
      << " query_ros_time_ns=" << query_ros_time_ns
      << " query_stamp_adjustments=" << query_stamp_adjustments
      << " final_vbc_failures=" << final_vbc_failures
      << " final_vbc_hold=" << static_cast<int>(final_vbc_hold)
      << " candidate_replacement_pending=" << static_cast<int>(replacement_pending)
      << " candidate_replacement_trigger_pending="
      << static_cast<int>(replacement_trigger_pending)
      << " candidate_replacement_trigger_count=" << replacement_trigger_count
      << " candidate_replacement_trigger_raw_candidate_stamp_ns="
      << replacement_trigger_raw
      << " safe_stall_active=" << static_cast<int>(safe_stall_active)
      << " safe_stall_attempts=" << safe_stall_attempts
      << " safe_stall_tiny_commits=" << safe_stall_tiny_commits
      << " safe_stall_trigger_commits=" << safe_stall_trigger_commits
      << " safe_stall_trigger_duration_s=" << safe_stall_trigger_duration
      << " safe_stall_trigger_motion_inf=" << safe_stall_trigger_motion
      << " safe_stall_raw_span=" << safe_stall_raw_span
      << " observation_stall_reason=" << observation_stall_reason
      << " observation_best_gap_rad=" << observation_best_gap
      << " observation_window_age_s=" << observation_window_age
      << " observation_window_commits=" << observation_window_commits
      << " view_window_commits=" << view_window_commits
      << " view_window_age_s=" << view_window_age
      << " view_fov_best_m=" << view_fov_best
      << " action_progress_decision=" << action_progress_decision
      << " action_progress_elapsed_s=" << action_progress_elapsed
      << " action_progress_budget_s=" << action_progress_budget
      << " action_progress_commits=" << action_progress_commits
      << " executed_view_trials=" << executed_view_trials
      << " action_executed_fov_gain_m=" << action_fov_gain
      << " recovery_advance_reason=" << recovery_advance_reason
      << " safe_stall_reselect=0"
      << " observation_dependency_policy=" << static_cast<int>(safe_frontier_recovery_enabled_)
      << " safe_stall_dependency_wait=" << static_cast<int>(safe_stall_reselect)
      << " plan_seq=" << plan_seq
      << " observation_token=" << observation_token
      << " running=" << static_cast<int>(running)
      << " repair=" << static_cast<int>(repair)
      << " repair_phase=" << (observation_phase ? "observation" : "retreat")
      << " repair_witness_count=" << witness_count
      << " vbc_obligation_id=" << vbc_obligation_id
      << " vbc_obligation_point_count=" << vbc_obligation_point_count
      << " vbc_obligation_points_seq=" << vbc_obligation_points_seq
      << " repair_witness_fresh=" << static_cast<int>(witnesses_fresh)
      << " probe=" << static_cast<int>(probe)
      << " task_ref_horizon_steps="
      << (probe ? probe_task_horizon_steps_ : num_intervals_)
      << " probe_hold_tail="
      << static_cast<int>(probe && probe_task_horizon_steps_ < num_intervals_)
      << " probe_restore_enabled="
      << static_cast<int>(probe_feasibility_restoration_enabled_)
      << " probe_restore_attempts=" << probe_restore_attempts
      << " probe_restore_pending_recheck="
      << static_cast<int>(probe_restore_pending_recheck)
      << " probe_restore_total=" << probe_restore_total
      << " probe_restore_hard_success_total="
      << probe_restore_hard_success_total
      << " cdf_horizon_steps="
      << ((repair || probe) ? cdf_constraint_horizon_steps_ : num_intervals_)
      << " visibility_target_cdf_horizon_steps="
      << visibility_obligation_cdf_horizon_steps_
      << " visibility_obligation_cdf_margin="
      << obligation_margin_effective
      << " visibility_obligation_cdf_margin_configured="
      << visibility_obligation_cdf_margin_base_
      << " cdf_margin_units=configuration_space_model"
      << " primitive_observation_guard_enabled=" << primitive_observation_guard_enabled_
      << " primitive_observation_margin_m=" << primitive_observation_margin_m_
      << " vbc_feedback_rejections="
      << vbc_feedback_rejections
      << " vbc_feedback_matched_points="
      << vbc_feedback_matched_points
      << " vbc_feedback_repair_witnesses="
      << vbc_feedback_repair_witness_count_
      << " init=" << init_mode
      << " scp_iter=" << scp_iter
      << " trust_q_inf=" << trust
      << " slack_mu=" << active_slack_weight
      << " latest_execution_stamp_ns=" << latest_execution_stamp_ns
      << " last_repair_completion_stamp_ns="
      << last_repair_completion_stamp_ns
      << " repair_completion_replan_count="
      << repair_completion_replan_count
      << " repair_completion_duplicate_count="
      << repair_completion_duplicate_count
      << " last_normal_completion_stamp_ns="
      << last_normal_completion_stamp_ns
      << " normal_completion_replan_count="
      << normal_completion_replan_count
      << " normal_completion_duplicate_count="
      << normal_completion_duplicate_count
      << " normal_reference_refresh_pending="
      << static_cast<int>(normal_reference_refresh_pending)
      << " normal_reference_refresh_count="
      << normal_reference_refresh_count
      << " normal_refresh_blocked_replan_count="
      << normal_refresh_blocked_replan_count
      << " smooth_handoff_replan_count="
      << smooth_handoff_replan_count
      << " smooth_handoff_replan_suppressed_probe_count="
      << smooth_handoff_replan_suppressed_probe_count
      << " smooth_handoff_replan_suppressed_busy_count="
      << smooth_handoff_replan_suppressed_busy_count
      << " stale_mode_candidate_discard_count="
      << stale_mode_candidate_discard_count
      << " normal_qp_backtrack_attempts="
      << normal_qp_backtrack_attempts
      << " normal_qp_backtrack_total="
      << normal_qp_backtrack_total
      << " frontier_active=" << static_cast<int>(frontier_active)
      << " frontier_horizon_step=" << visibility_frontier_horizon_step_
      << " frontier_weight_scale=" << frontier_weight_scale
      << " frontier_qvis_weight_scale=" << frontier_qvis_weight_scale
      << " frontier_target_shift_inf=" << frontier_target_shift_inf
      << " handoff_velocity_weight=" << handoff_velocity_weight_
      << " batches=" << batches
      << " cdf_roundtrip_ms=" << last_cdf_roundtrip_ms
      << " cdf_roundtrip_count=" << plan_cdf_roundtrip_count
      << " cdf_roundtrip_mean_ms="
      << (plan_cdf_roundtrip_count > 0
              ? plan_cdf_roundtrip_sum_ms /
                    static_cast<double>(plan_cdf_roundtrip_count)
              : 0.0)
      << " cdf_roundtrip_max_ms=" << plan_cdf_roundtrip_max_ms
      << " stamp_miss=" << misses
      << " solves=" << solves
      << " solve_failures=" << failures;
  oss << " fixed_start_recovery_enabled=" << static_cast<int>(fixed_start_recovery_enabled_);
  oss << " fixed_start_unknown_hold=" << static_cast<int>(fixed_start_unknown_hold);

  if (result) {
    oss << " solved=" << static_cast<int>(result->solved)
        << " repair_failures=" << result->repair_failures
        << " repair_max_failures=" << RepairNoProgress::max_failures
        << " repair_target_revision=" << result->trace_repair_ticket.revision
        << " repair_progress_epoch=" << result->trace_repair_ticket.progress
        << " snapshot=disabled"
        << " visibility_objective_step=" << result->visibility_objective_step
        << " status=" << result->status
        << " piqp_iter=" << result->iterations
        << " solve_ms=" << result->setup_and_solve_ms
        << " batch_pairs=" << result->batch_pairs
        << " cdf_rows=" << result->selected_cdf_rows
        << " reachable_conflicting_cdf_rows="
        << result->reachable_conflicting_cdf_rows
      << " first_step_conflicting_cdf_rows=" << result->first_step_conflicting_cdf_rows
      << " later_conflicting_cdf_rows=" << result->later_conflicting_cdf_rows
      << " first_step_conflicting_occupied_rows=" << result->first_step_conflicting_occupied_rows
        << " reachable_conflicting_repair_witness_rows="
        << result->reachable_conflicting_repair_witness_rows
        << " first_step_conflicting_repair_witness_rows="
        << result->first_step_conflicting_repair_witness_rows
        << " later_conflicting_repair_witness_rows="
        << result->later_conflicting_repair_witness_rows
        << " worst_repair_witness_reachable_deficit="
        << result->worst_repair_witness_reachable_deficit
        << " obligation_cdf_rows=" << result->selected_obligation_cdf_rows
        << " primitive_observation_rows=" << result->primitive_observation_rows
      << " primitive_conflicting_rows=" << result->primitive_conflicting_rows
        << " primitive_observation_points=" << result->primitive_observation_points
        << " primitive_executable_rows=" << result->primitive_executable_rows
        << " primitive_executable_samples=" << result->primitive_executable_samples
        << " primitive_executable_valid=" << (result->primitive_executable_valid ? 1 : 0)
        << " primitive_executable_min_distance_m=" << result->primitive_executable_min_distance_m
        << " primitive_executable_numerical_tightening_m=" << result->primitive_executable_numerical_tightening_m
        << " executable_step_scale=" << result->executable_step_scale
        << " executable_recheck_only=" << (result->executable_recheck_only ? 1 : 0)
        << " executable_trial_restored=" << (result->executable_trial_restored ? 1 : 0)
        << " executable_retained_available=" << (result->executable_retained_available ? 1 : 0)
        << " normal_curve_checked=" << result->normal_curve_checked
        << " normal_curve_valid=" << result->normal_curve_valid
        << " normal_curve_violation=" << result->normal_curve_violation
        << " normal_live_checked=" << result->normal_live_checked
        << " normal_live_valid=" << result->normal_live_valid
        << " normal_live_violation=" << result->normal_live_violation
        << " primitive_live_rebase_checked=" << result->primitive_live_rebase_checked
        << " primitive_live_rebase_required=" << result->primitive_live_rebase_required
        << " primitive_live_rebase_shift_inf=" << result->primitive_live_rebase_shift_inf
        << " primitive_live_rebase_min_m=" << result->primitive_live_rebase_min_m
        << " primitive_observation_start_distance_m=" << result->primitive_observation_start_distance_m
        << " fixed_start_recovery_points=" << result->fixed_start_recovery_points.size()
        << " fixed_start_unknown_blocked=" << static_cast<int>(result->fixed_start_unknown_blocked)
        << " fixed_start_unknown_distance_m=" << result->fixed_start_unknown_distance_m
        << " fixed_start_unknown_floor_m=" << result->fixed_start_unknown_floor_m
        << " fixed_start_unknown_point=[" << result->fixed_start_unknown_point.transpose() << ']'
        << " primitive_shell_recovery_points=" << result->primitive_shell_recovery_points
        << " primitive_shell_penalty_rows=" << result->primitive_shell_penalty_rows
        << " unknown_cdf_rows=" << result->selected_unknown_cdf_rows
        << " occupied_cdf_rows=" << result->selected_occupied_cdf_rows
        << " screened_safe=" << result->screened_safe_rows
        << " skipped_step0=" << result->skipped_step0_rows
        << " skipped_horizon=" << result->skipped_horizon_rows
        << " skipped_safety_horizon="
        << result->skipped_safety_horizon_rows
        << " qlin_error_inf=" << result->qlin_error_inf
        << " min_d=" << result->min_distance
        << " max_slack=" << result->max_slack
        << " repair_unknown_soft=" << (result->repair_unknown_soft ? 1 : 0)
        << " observation_dependency_basis=" << (result->repair_unknown_soft ? "analytic_unknown" : "learned_unknown")
        << " mean_slack=" << result->mean_slack
        << " slack_mu_used=" << result->slack_linear_weight_used
        << " step_inf=" << result->step_inf
        << " recovery_frontier_active=" << static_cast<int>(result->recovery_frontier.size() == dof_)
        << " recovery_frontier_normals=" << result->recovery_normals
        << " recovery_frontier_tight_normals=" << result->recovery_tight_normals
        << " recovery_frontier_rejoining=" << static_cast<int>(result->recovery_rejoining)
        << " primal=" << result->primal_residual
        << " dual=" << result->dual_residual;
  }
  if (total_plan_ms > 0.0)
    oss << " total_plan_ms=" << total_plan_ms;

  if (result && result->recovery_frontier.size() == dof_) {
    oss << " recovery_frontier_target=[" << std::setprecision(17);
    for (int j=0;j<dof_;++j) oss << (j?",":"") << result->recovery_frontier[j];
    oss << ']';
  }
  std_msgs::String msg;
  msg.data = oss.str();
  msg.data+=" goal_stop_execution_stamp_ns="+std::to_string(goal_stop_execution)+
      " goal_stop_sample_ros_ns="+std::to_string(goal_stop_sample);
  summary_pub_.publish(msg);
  ROS_INFO_STREAM_THROTTLE(
      0.25, "[LocalSparseSCPPlanner] " << msg.data);
}

#include "native_view_preview.inc"
#include "aligned_view_preview.inc"
}  // namespace egocentric_arm_planner
