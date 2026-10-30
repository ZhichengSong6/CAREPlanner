#include <care_confidence_map/vbc_evidence.hpp>
#include <Eigen/Core>
#include <cassert>
#include <iostream>
#include <limits>
struct Candidate {
  Eigen::Vector3d point_base{.015, -.025, .035};
  std::string link_name = "link 4\"\\", source_type = "sphere";
  int sample_index_in_link = 2, source_collision_index = 3;
  std::string source_collision_name;
  double primitive_signed_distance_m = std::numeric_limits<double>::quiet_NaN();
  std::string refinement_status = "sampled_intersection";
  int refinement_fk_before = 256, refinement_fk_after = 256;
  double candidate_margin_m = .02;
  double initial_support_distance_m = .03901809669363678;
  bool initial_support_unknown = true;
  double confidence = 0, current_visibility = 0;
  double raw_sample_radius_m = .04, swept_radius_m = .05, point_center_distance_m = .045;
  int sweep_eval_timestep = 3, sweep_original_timestep = 3;
  double sweep_time_s = .15;
  bool nominally_visible = false;
  int see_eval_timestep = -1, see_original_timestep = -1;
  double see_time_s = std::numeric_limits<double>::infinity();
  double margin_s = -std::numeric_limits<double>::infinity();
};
struct Layer { std::vector<int> member_indices; };
int main() {
  std::vector<Candidate> candidates(2);
  candidates[1].sweep_time_s = .4;
  candidates[1].see_time_s = .2;
  candidates[1].margin_s = .2;
  candidates[1].nominally_visible = true;
  std::vector<Layer> layers{{{0}}, {{1}}};
  const auto json = care_confidence_map::vbcEvidence(
      9007199254740993ULL, 7, .3, .01, candidates, layers);
  assert(json.find("9007199254740993") != std::string::npos);
  assert(json.find("\"see_s\":null") != std::string::npos);
  assert(json.find(' ') == std::string::npos);
  const auto bounded = care_confidence_map::vbcEvidence(
      9007199254740993ULL, 7, .3, .05, candidates, layers, "primitive", true);
  assert(bounded.find("voxel_centers_piecewise_linear_enclosure") != std::string::npos);
  assert(bounded.find("\"initial_support_unknown\":true") != std::string::npos);
  std::cout << json << '\n';
}
