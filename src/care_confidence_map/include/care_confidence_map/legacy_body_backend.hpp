#pragma once

#include <care_confidence_map/body_sample_model.hpp>

namespace care_confidence_map {
// Cold-path compatibility only. Loads the legacy library beside the evaluator;
// never searches another workspace and never substitutes geometry on failure.
bool loadLegacyBodySamples(BodySampleModel* model, const std::string& file,
                           std::string* error = nullptr);

inline std::size_t legacyRiskSampleCount(const BodySampleModel& model) {
  std::size_t count = 0;
  for (const auto& sample : model.samples()) count += sample.include_for_risk;
  return count;
}
}  // namespace care_confidence_map
