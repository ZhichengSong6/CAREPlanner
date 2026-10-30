#include "care_collision_cdf/shared_distance_field.hpp"
#include <iostream>
#include <limits>

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::invalid_argument("Usage: test_shared_distance_field CDF.so RDF.so");
    using Field = care_collision_cdf::SharedDistanceField;
    const std::vector<std::array<double, 10>> inputs = {
      {{.15, 0., .3, 0., .2, -.3, 0., .1, .2, -.1}},
      {{.4, -.1, .7, .2, .3, .1, -.4, .2, -.1, .3}}};
    Field cdf(argv[1], "configuration_space_model");
    const auto before = cdf.evaluate(inputs);
    Field rdf(argv[2], "m");
    const auto sdf = rdf.evaluate(inputs);
    const auto after = cdf.evaluate(inputs);
    if (cdf.backend() != "cdf" || rdf.backend() != "rdf" || sdf.size()!=inputs.size())
      throw std::runtime_error("Backend dispatch failed");
    for (std::size_t i=0;i<inputs.size();++i) {
      if (before[i].distance != after[i].distance || before[i].gradient != after[i].gradient)
        throw std::runtime_error("Loading RDF changed an existing CDF handle");
    }
    bool rejected_units=false, rejected_nan=false, rejected_missing=false;
    try { Field wrong(argv[2], "configuration_space_model"); } catch (const std::exception&) { rejected_units=true; }
    try { auto bad=inputs; bad[0][0]=std::numeric_limits<double>::quiet_NaN(); rdf.evaluate(bad); }
    catch (const std::exception&) { rejected_nan=true; }
    try { Field missing("/nonexistent/care-distance-field.so", "m"); } catch (const std::exception&) { rejected_missing=true; }
    if (!rejected_units || !rejected_nan || !rejected_missing || !rdf.evaluate({}).empty())
      throw std::runtime_error("Invalid-input/backend checks failed");
    std::cout << "SHARED_DISTANCE_FIELD_CPP_PASS switching units nonfinite missing empty\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what()<<'\n'; return 1; }
}
