// Real recorded trajectories, three interleaved production-path variants.
#define main care_temporal_selector_main_for_replay
#include "../src/care_confidence_map/src/trajectory_vbc_selector_temporal_cluster_node.cpp"
#undef main
#include "fixtures/vbc_primitive_sweep_reference.hpp"
#include <boost/property_tree/json_parser.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

struct VbcPrimitiveReplay {
  using Node = TrajectoryVbcTemporalClusterNode;
  using Voxel = Node::SweepVoxel;
  using Clock = std::chrono::steady_clock;
  static void check(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
  }
  static void equal(const std::vector<Voxel>& a, const std::vector<Voxel>& b) {
    check(a.size() == b.size(), "voxel count mismatch");
    for (std::size_t i = 0; i < a.size(); ++i) {
      const auto& x = a[i]; const auto& y = b[i];
      check(x.point_base == y.point_base && x.sample_center_base == y.sample_center_base &&
        x.link_name == y.link_name && x.source_type == y.source_type &&
        x.sample_index_in_link == y.sample_index_in_link &&
        x.source_collision_index == y.source_collision_index &&
        x.source_collision_name == y.source_collision_name &&
        x.primitive_signed_distance_m == y.primitive_signed_distance_m &&
        x.point_center_distance_m == y.point_center_distance_m &&
        std::isnan(x.raw_sample_radius_m) && std::isnan(y.raw_sample_radius_m) &&
        std::isnan(x.swept_radius_m) && std::isnan(y.swept_radius_m) &&
        x.sweep_eval_timestep == y.sweep_eval_timestep &&
        x.sweep_original_timestep == y.sweep_original_timestep && x.sweep_time_s == y.sweep_time_s,
        "ordered voxel/evidence mismatch at " + std::to_string(i));
    }
  }
  static void run(const std::string& root, const std::string& input, const std::string& output, int reps) {
    using namespace care_confidence_map;
    boost::property_tree::ptree data;
    boost::property_tree::read_json(input, data);
    Node samples, primitive;
    std::string error;
    const auto fk = root + "/src/arm_description/urdf/Arm.urdf";
    check(samples.evaluator_.initialize(fk, root + "/src/care_confidence_map/config/body_samples.yaml", "base_link", &error), error);
    check(primitive.evaluator_.initializeKinematics(fk, "base_link", &error), error);
    check(primitive.primitive_model_.load(root + "/src/arm_description/urdf/Arm_with_self_filter_collision.urdf",
                                         {"base_link", "link1"}, &error), error);
    samples.ignored_risk_links_ = {"base_link", "link1"};
    VbcGrid grid;
    int j = 0; for (const auto& v : data.get_child("grid.min")) grid.min[j++] = v.second.get_value<double>();
    j = 0; for (const auto& v : data.get_child("grid.max")) grid.max[j++] = v.second.get_value<double>();
    grid.resolution = data.get<double>("grid.resolution");
    samples.map_x_min_ = grid.min.x(); samples.map_y_min_ = grid.min.y(); samples.map_z_min_ = grid.min.z();
    samples.map_x_max_ = grid.max.x(); samples.map_y_max_ = grid.max.y(); samples.map_z_max_ = grid.max.z();
    samples.map_resolution_ = grid.resolution;
    samples.max_eval_timesteps_ = primitive.max_eval_timesteps_ = data.get<int>("max_eval_timesteps");
    samples.fallback_dt_ = primitive.fallback_dt_ = data.get<double>("fallback_dt");
    std::ofstream out(output); check(bool(out), "cannot open result"); out << std::setprecision(17);
    std::size_t comparisons = 0;
    for (const auto& item : data.get_child("trajectories")) {
      const auto& t = item.second;
      trajectory_msgs::JointTrajectory msg;
      for (const auto& name : t.get_child("joint_names")) msg.joint_names.push_back(name.second.get_value<std::string>());
      for (const auto& point : t.get_child("points")) {
        trajectory_msgs::JointTrajectoryPoint p;
        for (const auto& v : point.second.get_child("q")) p.positions.push_back(v.second.get_value<double>());
        p.time_from_start.fromNSec(point.second.get<unsigned long long>("time_ns"));
        msg.points.push_back(p);
      }
      std::vector<Eigen::VectorXd> qs, qs2;
      std::vector<int> indices, indices2; std::vector<double> times, times2;
      check(samples.convertTrajectory(msg, &qs, &indices, &times, &error), error);
      check(primitive.convertTrajectory(msg, &qs2, &indices2, &times2, &error), error);
      check(indices == indices2 && times == times2 && qs.size() == qs2.size(), "converter mismatch");
      for (std::size_t k = 0; k < qs.size(); ++k) check(qs[k] == qs2[k], "converted q mismatch");
      for (const auto& m : data.get_child("margins")) {
        const double margin = m.second.get_value<double>();
        samples.swept_volume_margin_m_ = margin;
        std::vector<VbcPrimitiveFrame> world;
        check(primitive.primitive_model_.computeTrajectory(primitive.evaluator_, qs, &world, &error), error);
        auto reference = buildPrimitiveSweptVoxelsReference<Voxel>(world, indices, times, grid, margin);
        equal(reference, buildPrimitiveSweptVoxels<Voxel>(world, indices, times, grid, margin)); ++comparisons;
        for (int rep = -1; rep < reps; ++rep) for (int order = 0; order < 3; ++order) {
          // Six permutations; each backend appears in each position equally.
          const int cycle = (rep + 1 + t.get<int>("id")) % 6;
          const int backend = (cycle % 3 + (cycle < 3 ? order : 2-order)) % 3;
          std::vector<Voxel> voxels;
          const auto begin = Clock::now();
          Clock::time_point middle, end;
          if (backend == 0) {
            const auto f = samples.evaluator_.computeTrajectorySamples(qs); check(f.success, f.message);
            middle = Clock::now(); voxels = samples.buildSweptVolumeVoxels(f, indices, times);
            end = Clock::now();
          } else {
            std::vector<VbcPrimitiveFrame> f;
            check(primitive.primitive_model_.computeTrajectory(primitive.evaluator_, qs, &f, &error), error);
            middle = Clock::now();
            voxels = backend == 1 ? buildPrimitiveSweptVoxelsReference<Voxel>(f, indices, times, grid, margin)
                                 : buildPrimitiveSweptVoxels<Voxel>(f, indices, times, grid, margin);
            end = Clock::now();
          }
          if (backend != 0) equal(reference, voxels);
          if (rep < 0) continue;
          out << "{\"id\":" << t.get<int>("id") << ",\"rep\":" << rep << ",\"order\":" << order
              << ",\"backend\":" << backend << ",\"margin\":" << margin << ",\"knots\":" << qs.size()
              << ",\"voxels\":" << voxels.size() << ",\"fk_ms\":"
              << std::chrono::duration<double, std::milli>(middle-begin).count() << ",\"sweep_ms\":"
              << std::chrono::duration<double, std::milli>(end-middle).count() << ",\"total_ms\":"
              << std::chrono::duration<double, std::milli>(end-begin).count() << "}\n";
        }
      }
    }
    // Exercise non-monotone/equal-within-tolerance times and the large-grid
    // sparse route. Comparing ordered full evidence also checks replacement.
    VbcPrimitive p; p.half_size = {.1,.1,.1}; p.link_name="test"; p.collision_name="box";
    std::vector<VbcPrimitiveFrame> frames;
    for (int k = 0; k < 3; ++k) {
      VbcPrimitiveFrame f;
      for (int j = 0; j < 2; ++j) {
        p.collision_index = j; p.collision_name = "box_"+std::to_string(k)+"_"+std::to_string(j);
        f.primitives.push_back(p);
      }
      frames.push_back(f);
    }
    for (auto g : {grid, VbcGrid{{-100,-100,-100},{100,100,100},.05}})
      for (const auto& times : {std::vector<double>{.2,.1,.1}, std::vector<double>{.1,.1+5e-13,.1-5e-13}})
        equal(buildPrimitiveSweptVoxelsReference<Voxel>(frames,{9,3,6},times,g,0),
              buildPrimitiveSweptVoxels<Voxel>(frames,{9,3,6},times,g,0));
    std::cout << "PASS exact ordered evidence " << comparisons << " real trajectory/margin combinations; all timed repeats; tie/sparse cases\n";
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "vbc_real_replay", ros::init_options::AnonymousName);
  try {
    if (argc != 5) throw std::runtime_error("usage: benchmark REPO INPUT.json OUTPUT.jsonl REPETITIONS");
    const int reps = std::stoi(argv[4]);
    if (reps < 6 || reps % 6) throw std::runtime_error("repetitions must be a positive multiple of six");
    VbcPrimitiveReplay::run(argv[1],argv[2],argv[3],reps);
    ros::shutdown(); return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << std::endl; ros::shutdown(); return 1; }
}
