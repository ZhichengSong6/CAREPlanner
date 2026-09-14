#include <care_confidence_map/legacy_body_backend.hpp>

#include <dlfcn.h>
#include <cstdlib>
#include <limits.h>
#include <exception>

namespace care_confidence_map {
namespace {
using Loader = bool (*)(BodySampleModel*, const std::string&, std::string*);
struct LegacyBackend {
  void* handle = nullptr;
  Loader load = nullptr;
  std::string error;
  LegacyBackend() {
    Dl_info info{};
    char resolved[PATH_MAX];
    if (!dladdr(reinterpret_cast<void*>(&loadLegacyBodySamples), &info) ||
        !info.dli_fname || !realpath(info.dli_fname, resolved)) {
      error = "Cannot resolve evaluator location for legacy backend";
      return;
    }
    const std::string owner(resolved);
    const auto path = owner.substr(0, owner.find_last_of('/') + 1) + "libbody_sample_model.so";
    handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) { error = "Legacy backend unavailable: " + path + ": " + dlerror(); return; }
    load = reinterpret_cast<Loader>(dlsym(handle, "care_load_body_samples_v1"));
    if (!load) {
      error = "Legacy backend missing care_load_body_samples_v1: " + path;
      dlclose(handle); handle = nullptr;
    }
  }
  // Keep successful code loaded for process lifetime (including copied models).
};
}

bool loadLegacyBodySamples(BodySampleModel* model, const std::string& file, std::string* error) {
  if (!model) { if (error) *error = "Null legacy model"; return false; }
  // C++11 initialization is thread safe. Primitive/FK paths never get here.
  static const LegacyBackend backend;
  if (!backend.load) { if (error) *error = backend.error; return false; }
  try { return backend.load(model, file, error); }
  catch (const std::exception& exc) { if (error) *error = exc.what(); return false; }
  catch (...) { if (error) *error = "Legacy backend exception"; return false; }
}
}  // namespace care_confidence_map
