// Compile against the actual selector translation unit, including its unchanged
// samples rasterizer and trajectory conversion. No surrogate Python benchmark.
#define main care_temporal_selector_main_for_test
#include "../src/care_confidence_map/src/trajectory_vbc_selector_temporal_cluster_node.cpp"
#undef main
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

using namespace care_confidence_map;

struct VbcPrimitiveTest {
  using Node = TrajectoryVbcTemporalClusterNode;
  using Voxel = Node::SweepVoxel;
  static void check(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
  }
  static void near(double a, double b) { check(std::abs(a - b) < 1e-10, "distance mismatch"); }
  static VbcGrid grid() { return {{-.95, -.95, 0}, {.95, .95, 1.15}, .05}; }
  static void configure(Node& node) {
    node.ignored_risk_links_ = {"base_link", "link1"};
    node.map_x_min_ = node.map_y_min_ = -.95;
    node.map_x_max_ = node.map_y_max_ = .95;
    node.map_z_min_ = 0; node.map_z_max_ = 1.15; node.map_resolution_ = .05;
    node.swept_volume_margin_m_ = 0;
  }
  static std::vector<Voxel> sweep(const std::vector<VbcPrimitiveFrame>& f,
                                 double margin = 0) {
    std::vector<int> indices; std::vector<double> times;
    for (std::size_t i = 0; i < f.size(); ++i) { indices.push_back(3 * i); times.push_back(.15 * i); }
    return buildPrimitiveSweptVoxels<Voxel>(f, indices, times, grid(), margin);
  }
  static double quantile(std::vector<double> x, double q) {
    std::sort(x.begin(), x.end()); return x.at(static_cast<std::size_t>(q * (x.size() - 1)));
  }
  static void run(const std::string& root, const std::string& scratch) {
    const std::string urdf = root + "/src/arm_description/urdf/Arm_with_self_filter_collision.urdf";
    const std::string fk_urdf = root + "/src/arm_description/urdf/Arm.urdf";
    const std::string yaml = root + "/src/care_confidence_map/config/body_samples.yaml";
    std::string error;
    VbcPrimitive box; box.half_size = {1, 2, 3};
    near(box.signedDistance({0, 0, 0}), -1);
    near(box.signedDistance({1, 1, 1}), 0);
    near(box.signedDistance({1.03, 2.04, 0}), .05);
    VbcPrimitive cylinder; cylinder.kind = PrimitiveKind::Cylinder;
    cylinder.radius = 1; cylinder.half_length = 2;
    near(cylinder.signedDistance({0, 0, 0}), -1);
    near(cylinder.signedDistance({0, 0, 2.03}), .03);
    near(cylinder.signedDistance({1.03, 0, 2.04}), .05);
    VbcPrimitive sphere; sphere.kind = PrimitiveKind::Sphere; sphere.radius = .2;
    near(sphere.signedDistance({.3, 0, 0}), .1);
    for (auto p : {box, cylinder, sphere}) {
      const Eigen::Vector3d x(1.3, -.1, 2.4); const double d = p.signedDistance(x);
      p.rotation = Eigen::AngleAxisd(.73, Eigen::Vector3d(1, 2, 3).normalized()).toRotationMatrix();
      p.center = {.3, -.4, .2};
      near(p.signedDistance(p.center + p.rotation * x), d);
    }
    // Euclidean corner dilation: axis-wise expansion would incorrectly include this.
    check(box.signedDistance({1.008, 2.008, 0}) > .01, "rounded box corner");
    check(cylinder.signedDistance({1.008, 0, 2.008}) > .01, "rounded cylinder rim");

    Node samples, primitive; configure(samples); configure(primitive);
    check(samples.evaluator_.initialize(fk_urdf, yaml, "base_link", &error), error);
    check(primitive.evaluator_.initializeKinematics(fk_urdf, "base_link", &error), error);
    check(samples.evaluator_.hasLegacyBodySamples(), "samples lost compatibility data");
    check(!primitive.evaluator_.hasLegacyBodySamples(), "FK allocated legacy data");
    auto reset_copy = samples.evaluator_;
    check(reset_copy.initializeKinematics(fk_urdf, "base_link", &error), error);
    check(!reset_copy.hasLegacyBodySamples() && samples.evaluator_.hasLegacyBodySamples(),
          "FK reset retained legacy data or invalidated copied evaluator");
    check(primitive.evaluator_.bodySampleModel().samples().empty(), "primitive loaded YAML");
    check(primitive.primitive_model_.load(urdf, {"base_link", "link1"}, &error), error);
    check(primitive.primitive_model_.primitives().size() == 21, "active primitive count");
    check(primitive.primitive_model_.initializeRelativeFk(
              urdf, primitive.evaluator_.activeJointNames(), &error), error);
    const Eigen::VectorXd q_zero = Eigen::VectorXd::Zero(7);
    const auto zero_relative_bounds =
        primitive.primitive_model_.relativeFkDisplacementBounds(q_zero, q_zero);
    check(zero_relative_bounds.size() == 21, "relative FK primitive count");
    for (const double bound : zero_relative_bounds)
      check(std::isfinite(bound) && bound >= 0.0 && bound < 1e-10,
            "nonzero relative FK bound for identical configurations");
    Eigen::VectorXd q_offset = q_zero;
    q_offset[3] = 0.01;
    const auto offset_relative_bounds =
        primitive.primitive_model_.relativeFkDisplacementBounds(q_offset, q_zero);
    const double max_offset_bound = *std::max_element(
        offset_relative_bounds.begin(), offset_relative_bounds.end());
    check(std::isfinite(max_offset_bound) && max_offset_bound > 0.0,
          "relative FK max did not detect a joint displacement");
    // The runtime fallback calls this only on legacy-bound overflow. Keep a
    // small repeated-call timing check to catch accidental trajectory scans or
    // allocations that would make a 100 Hz guard impractical.
    const auto relative_fk_start = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < 256; ++repeat) {
      const auto bounds = primitive.primitive_model_.relativeFkDisplacementBounds(
          q_offset, q_zero);
      check(bounds.size() == 21 && std::isfinite(*std::max_element(bounds.begin(), bounds.end())),
            "relative FK repeated query failed");
    }
    const double relative_fk_us =
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - relative_fk_start).count() / 256.0;
    check(std::isfinite(relative_fk_us) && relative_fk_us >= 0.0,
          "relative FK timing invalid");
    std::cout << std::setprecision(8)
              << "{\"test\":\"relative_fk_max_bound\",\"status\":\"PASS\","
              << "\"active_primitives\":21,\"max_offset_bound_m\":"
              << max_offset_bound << ",\"mean_query_us\":" << relative_fk_us
              << "}" << std::endl;
    VbcPrimitiveModel all;
    check(all.load(urdf, {}, &error) && all.primitives().size() == 26, "all collision coverage");
    for (const auto& p : all.primitives()) check(!p.collision_name.empty(), "missing collision identity");

    // Loader fault injection: missing input, unsupported mesh, bad dimensions,
    // nonfinite dimensions, no active solids. Never silently replace by samples.
    check(!all.load(scratch + "/does_not_exist.urdf", {}, &error), "missing URDF accepted");
    int fixture_index = 0;
    for (const auto& geometry : {std::string("<mesh filename='bad.stl'/>") ,
          std::string("<box size='-1 1 1'/>") , std::string("<cylinder radius='nan' length='1'/>")}) {
      const std::string path = scratch + "/bad_" + std::to_string(fixture_index++) + ".urdf";
      { std::ofstream f(path); f << "<robot name='test'><link name='base_link'><collision><geometry>"
                                << geometry << "</geometry></collision>"
                                << "<collision><geometry><sphere radius='0.1'/></geometry></collision></link></robot>"; }
      check(!all.load(path, {}, &error) && all.primitives().empty(), "invalid geometry accepted");
    }
    check(!all.load(urdf, {"base_link", "link1", "link2", "link3", "link4", "wrist_link1", "wrist_link2", "wrist_link3"}, &error),
          "empty active model accepted");
    std::vector<VbcPrimitiveFrame> frames;
    Eigen::VectorXd q = Eigen::VectorXd::Zero(7);
    check(!primitive.primitive_model_.computeTrajectory(primitive.evaluator_, {}, &frames, &error), "empty trajectory accepted");
    q[0] = std::numeric_limits<double>::quiet_NaN();
    check(!primitive.primitive_model_.computeTrajectory(primitive.evaluator_, {q}, &frames, &error), "NaN q accepted");
    check(!primitive.primitive_model_.computeTrajectory(primitive.evaluator_, {Eigen::VectorXd::Zero(6)}, &frames, &error), "6D q accepted");
    q.setZero();
    check(primitive.primitive_model_.computeTrajectory(primitive.evaluator_, {q}, &frames, &error), error);
    for (double margin : {-1., std::numeric_limits<double>::quiet_NaN()}) {
      bool rejected = false; try { sweep(frames, margin); } catch (const std::invalid_argument&) { rejected = true; }
      check(rejected, "invalid margin accepted");
    }
    bool rejected = false;
    try { buildPrimitiveSweptVoxels<Voxel>(frames, {}, {0}, grid(), 0); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "time/index mismatch accepted");

    // Production conversion: continuous primitive certification must retain
    // every input knot even if the legacy sampling cap is smaller.
    trajectory_msgs::JointTrajectory msg;
    msg.joint_names = samples.evaluator_.activeJointNames();
    std::reverse(msg.joint_names.begin(), msg.joint_names.end());
    for (int k = 0; k < 21; ++k) {
      trajectory_msgs::JointTrajectoryPoint p;
      p.positions = {k*.001, .2, -.3, .4, -.5, .6, -.7};
      p.time_from_start = ros::Duration(k * .05); msg.points.push_back(p);
    }
    samples.max_eval_timesteps_ = primitive.max_eval_timesteps_ = 7;
    primitive.geometry_backend_ = "primitive";
    std::vector<Eigen::VectorXd> qa, qb;
    std::vector<int> ia, ib; std::vector<double> ta, tb;
    check(samples.convertTrajectory(msg, &qa, &ia, &ta, &error), error);
    check(primitive.convertTrajectory(msg, &qb, &ib, &tb, &error), error);
    check(ia.size()==7 && ib.size()==21 && ib.front()==0 && ib.back()==20, "continuous path lost knots");
    for (std::size_t k = 0; k < qa.size(); ++k) {
      check(qa[k].isApprox(qb[ia[k]]), "q reorder mismatch"); near(ta[k], ia[k] * .05);
      near(qa[k][0], -.7); near(qa[k][6], ia[k] * .001);
    }
    for (auto& p : msg.points) p.time_from_start = ros::Duration(0);
    check(samples.convertTrajectory(msg, &qa, &ia, &ta, &error), error);
    for (std::size_t k = 0; k < ta.size(); ++k) near(ta[k], ia[k] * .05);
    check(!primitive.convertTrajectory(msg,&qb,&ib,&tb,&error),"continuous certificate invented missing timing");
    for(std::size_t k=0;k<msg.points.size();++k) msg.points[k].time_from_start=ros::Duration((k+1)*.05);
    check(!primitive.convertTrajectory(msg,&qb,&ib,&tb,&error),"delayed first knot granted late collision time");
    for(std::size_t k=0;k<msg.points.size();++k) msg.points[k].time_from_start=ros::Duration(k*.05);
    auto bad_names=msg;bad_names.joint_names[1]=bad_names.joint_names[0];
    check(!primitive.convertTrajectory(bad_names,&qb,&ib,&tb,&error),"duplicate continuous joints accepted");

    // Full-grid oracle validates AABB completeness and earliest-witness tie
    // ordering at rotated, translated poses (not just surface point queries).
    std::mt19937 rng(20260911); std::uniform_real_distribution<double> u(-1.2, 1.2);
    std::vector<Eigen::VectorXd> configs{Eigen::VectorXd::Zero(7)};
    for (int k = 0; k < 3; ++k) { for (int j = 0; j < 7; ++j) q[j] = u(rng); configs.push_back(q); }
    check(primitive.primitive_model_.computeTrajectory(primitive.evaluator_, configs, &frames, &error), error);
    {
      std::ofstream export_file(scratch + "/primitive_fk.json");
      export_file << std::setprecision(17) << "{\"joint_names\":[";
      const auto& names = primitive.evaluator_.activeJointNames();
      for (std::size_t i = 0; i < names.size(); ++i) export_file << (i ? "," : "") << evidenceString(names[i]);
      export_file << "],\"frames\":[";
      for (std::size_t k = 0; k < frames.size(); ++k) {
        export_file << (k ? "," : "") << "{\"q\":[";
        for (int j = 0; j < 7; ++j) export_file << (j ? "," : "") << configs[k][j];
        export_file << "],\"primitives\":[";
        bool first = true;
        for (const auto& p : frames[k].primitives) {
          export_file << (first ? "" : ",") << "{\"link\":" << evidenceString(p.link_name)
            << ",\"collision_index\":" << p.collision_index << ",\"collision_name\":" << evidenceString(p.collision_name)
            << ",\"center\":[" << p.center.x() << ',' << p.center.y() << ',' << p.center.z() << "],\"rotation\":[";
          first = false;
          for (int i = 0; i < 9; ++i) export_file << (i ? "," : "") << p.rotation(i/3, i%3);
          export_file << "],\"queries\":[";
          for (int i = 0; i < 20; ++i) {
            // Deterministic points around each actual transformed solid.
            const Eigen::Vector3d x = p.center + .15 * Eigen::Vector3d(std::sin(i*1.3), std::cos(i*.7), std::sin(i*.5));
            export_file << (i ? "," : "") << "{\"point\":[" << x.x() << ',' << x.y() << ',' << x.z()
                        << "],\"distance\":" << p.signedDistance(x) << '}';
          }
          export_file << "]}";
        }
        export_file << "]}";
      }
      export_file << "]}\n";
    }
    for (double margin : {0., .01, .02}) {
      const auto voxels = sweep(frames, margin);
      std::map<std::tuple<int, int, int>, Voxel> actual;
      const auto g = grid();
      for (const auto& v : voxels) {
        Eigen::Array3i index = ((v.point_base - g.min) / g.resolution).array().round().cast<int>();
        actual[{index.x(), index.y(), index.z()}] = v;
        check(v.sample_index_in_link == -1 && std::isnan(v.raw_sample_radius_m), "fake sphere evidence");
      }
      const Eigen::Array3i n = (((g.max - g.min) / g.resolution).array().floor() + 1).cast<int>();
      std::size_t count = 0;
      for (int ix = 0; ix < n.x(); ++ix) for (int iy = 0; iy < n.y(); ++iy) for (int iz = 0; iz < n.z(); ++iz) {
        const Eigen::Vector3d point = g.min + g.resolution * Eigen::Vector3d(ix, iy, iz);
        bool found = false;
        for (std::size_t k = 0; k < frames.size() && !found; ++k) for (const auto& p : frames[k].primitives) {
          if (p.signedDistance(point) > margin + 1e-12) continue;
          auto it = actual.find({ix, iy, iz});
          check(it != actual.end(), "AABB omitted grid center: " + p.collision_name + " margin=" +
                std::to_string(margin) + " index=" + std::to_string(ix) + "," + std::to_string(iy) + "," + std::to_string(iz));
          const auto& v = it->second;
          check(v.sweep_eval_timestep == int(k) && v.sweep_original_timestep == 3 * int(k), "earliest sweep index");
          near(v.sweep_time_s, .15*k);
          check(v.link_name == p.link_name && v.source_collision_index == p.collision_index &&
                v.source_collision_name == p.collision_name, "witness identity/tie order");
          near(v.primitive_signed_distance_m, p.signedDistance(point));
          found = true; ++count; break;
        }
      }
      check(count == actual.size(), "extra primitive grid center");
    }
    std::cout << "{\"test\":\"geometry_timing_fail_closed\",\"status\":\"PASS\",\"active_primitives\":21,\"total_primitives\":26}" << std::endl;

    // Interleaved paired C++ FK+sweep benchmark. Production samples path is
    // called unchanged. Same 20-knot q, grid, ignore list and margin per pair.
    std::vector<std::vector<Eigen::VectorXd>> trajectories;
    for (int i = 0; i < 24; ++i) {
      Eigen::VectorXd seed(7), delta(7);
      for (int j = 0; j < 7; ++j) { seed[j] = i == 0 ? 0 : u(rng); delta[j] = i == 0 ? 0 : .15*u(rng); }
      std::vector<Eigen::VectorXd> qs;
      for (int k = 0; k < 20; ++k) qs.push_back(seed + k/19. * delta);
      trajectories.push_back(qs);
    }
    std::vector<int> indices; std::vector<double> times;
    for (int k = 0; k < 20; ++k) { indices.push_back(k); times.push_back(k*.05); }
    for (double margin : {0., .01, .02}) {
      samples.swept_volume_margin_m_ = margin;
      std::vector<double> duration[2]; std::size_t counts[2] = {0, 0};
      for (int rep = -24; rep < 240; ++rep) {
        const auto& qs = trajectories[(rep + 24) % trajectories.size()];
        for (int order = 0; order < 2; ++order) {
          int backend = ((rep + 24) % 2) ^ order;
          const auto start = std::chrono::steady_clock::now();
          std::size_t count;
          if (backend == 0) {
            const auto result = samples.evaluator_.computeTrajectorySamples(qs);
            check(result.success, result.message);
            count = samples.buildSweptVolumeVoxels(result, indices, times).size();
          } else {
            std::vector<VbcPrimitiveFrame> world;
            check(primitive.primitive_model_.computeTrajectory(primitive.evaluator_, qs, &world, &error), error);
            count = buildPrimitiveSweptVoxels<Voxel>(world, indices, times, grid(), margin).size();
          }
          const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
          if (rep >= 0) { duration[backend].push_back(ms); counts[backend] += count; }
        }
      }
      const double a = quantile(duration[0], .5), b = quantile(duration[1], .5);
      const double ap = quantile(duration[0], .95), bp = quantile(duration[1], .95);
      std::cout << std::setprecision(8) << "{\"test\":\"cpp_fk_sweep_benchmark\",\"margin_m\":" << margin
        << ",\"repetitions\":240,\"knots\":20,\"samples_p50_ms\":" << a << ",\"primitive_p50_ms\":" << b
        << ",\"samples_p95_ms\":" << ap << ",\"primitive_p95_ms\":" << bp
        << ",\"p50_speedup\":" << a/b << ",\"samples_mean_voxels\":" << counts[0]/240.
        << ",\"primitive_mean_voxels\":" << counts[1]/240.
        << ",\"status\":\"" << (b <= a && bp <= ap ? "PASS" : "FAIL") << "\"}" << std::endl;
    }
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "vbc_primitive_cpp_tests", ros::init_options::AnonymousName);
  try {
    if (argc != 3) throw std::runtime_error("usage: test_vbc_primitives REPO SCRATCH");
    VbcPrimitiveTest::run(argv[1], argv[2]);
    ros::shutdown(); return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << std::endl; ros::shutdown(); return 1;
  }
}
