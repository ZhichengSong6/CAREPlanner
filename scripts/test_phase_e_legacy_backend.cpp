#include <care_confidence_map/legacy_body_backend.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

bool mapped() {
  std::ifstream maps("/proc/self/maps");
  return std::string(std::istreambuf_iterator<char>(maps), {}).find("libbody_sample_model.so") != std::string::npos;
}
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  care_confidence_map::BodySampleModel model;
  std::string error;
#ifdef LEGACY_DIRECT
  if (!model.loadFromYaml(argv[1], &error) || !model.riskSampleCount()) return 3;
#else
  if (mapped()) return 4;
  if (care_confidence_map::loadLegacyBodySamples(nullptr, argv[1], &error)) return 5;
  const bool ok = care_confidence_map::loadLegacyBodySamples(&model, argv[1], &error);
  if (std::string(argv[2]) == "missing") {
    if (ok || mapped() || model.size() || error.find("Legacy backend unavailable") == std::string::npos) return 6;
    std::cout << "PASS missing library fails closed: " << error << '\n';
    return 0;
  }
  if (!ok || !mapped() || !care_confidence_map::legacyRiskSampleCount(model)) {
    std::cerr << error << '\n'; return 7;
  }
  auto copied = model;
  if (copied.size() != model.size()) return 8;
  if (care_confidence_map::loadLegacyBodySamples(&model, "/nonexistent/legacy-test.yaml", &error)) return 9;
  if (model.size() || !copied.size()) return 10;
#endif
  std::cout << "PASS legacy compatibility\n";
  return 0;
}
